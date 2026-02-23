# coding=utf-8
# Copyright (c) 2025, Qwerky AI, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mamba mixer implementation for MambaInLlama hybrid models.

Contains the MambaInLlamaMambaMixer class (both vLLM and fallback versions)
and the custom op registration for CUDA graph compatibility.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from transformers.utils import logging

from ..configuration import MambaInLlamaMambaConfig

logger = logging.get_logger(__name__)

# =============================================================================
# vLLM IMPORTS
# =============================================================================

_vllm_available = False

try:
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.platforms import current_platform
    from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
    from vllm.forward_context import ForwardContext, get_forward_context
    from vllm.utils.torch_utils import direct_register_custom_op

    _vllm_available = True
except ImportError as e:
    logger.warning(f"vLLM not available: {e}")
    get_current_vllm_config = None
    get_forward_context = None

# MambaBase import for proper vLLM integration
_MambaBase = None
try:
    from vllm.model_executor.layers.mamba.abstract import MambaBase as _MambaBase
except ImportError:
    pass

# CustomOp import for proper callability with MambaBase
_CustomOp = None
try:
    from vllm.model_executor.custom_op import CustomOp as _CustomOp
except ImportError:
    pass

# Mamba ops imports
_mamba_ops_available = False
try:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        selective_scan_fn,
        selective_state_update,
    )
    _mamba_ops_available = True
except ImportError:
    pass

# State calculators
_vllm_MambaStateShapeCalculator = None
_vllm_MambaStateDtypeCalculator = None
try:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator as _vllm_MambaStateShapeCalculator,
        MambaStateDtypeCalculator as _vllm_MambaStateDtypeCalculator,
    )
except ImportError:
    pass


# =============================================================================
# MAMBAINLLAMA MAMBA MIXER (Custom Op Pattern for CUDA Graphs)
# =============================================================================

if _MambaBase is not None and _CustomOp is not None:
    @_CustomOp.register("mambainllama_mixer")
    class MambaInLlamaMambaMixer(_MambaBase, _CustomOp):
        """MambaInLlama Mamba mixer with vLLM V1 integration.

        Uses vLLM's custom op pattern for CUDA graph compatibility:
        - forward() dispatches via torch.ops.vllm.mambainllama_mixer
        - forward_cuda() contains the actual computation
        - Registered in static_forward_context for V1 state binding

        Key architectural differences from standard Mamba:
        - Fused in_proj: outputs [z, x, B, C, dt] instead of separate projections
        - x is d_xb (needs repeat_kv expansion), C is d_inner (already full size)
        - Grouped heads with repeat_kv expansion for x and B
        """

        def __init__(
            self,
            config: MambaInLlamaMambaConfig,
            layer_idx: int,
            prefix: str = "",
            model_config: "ModelConfig | None" = None,
            cache_config: "CacheConfig | None" = None,
            is_lora_enabled: bool = False,
            quant_config=None,
        ):
            super().__init__()
            self.layer_idx = layer_idx
            self.prefix = prefix
            self.model_config = model_config
            self.cache_config = cache_config
            self.is_lora_enabled = is_lora_enabled

            # Core dimensions
            self.d_model = config.d_model
            self.d_inner = config.d_inner
            self.d_xb = config.d_xb
            self.d_state = config.ssm_cfg.get("d_state", 16)
            self.d_conv = config.ssm_cfg.get("d_conv", 4)
            self.dt_rank = math.ceil(self.d_model / 16)

            # Grouped head configuration
            self.num_xb_head = self.d_xb // self.d_state
            self.num_heads = self.d_inner // self.d_state
            self.repeat_group = self.d_inner // self.d_xb
            self.num_C_head = self.num_heads
            self.repeat_kv_before_conv = config.ssm_cfg.get("repeat_kv_before_conv", True)
            self.conv_dim = self.d_inner if self.repeat_kv_before_conv else self.d_xb

            # TP dimensions — per-partition sizes for forward split
            tp_size = get_tensor_model_parallel_world_size()
            self.d_inner_local = self.d_inner // tp_size
            self.d_xb_local = self.d_xb // tp_size
            self.dt_rank_local = self.dt_rank // tp_size
            self.conv_dim_local = self.conv_dim // tp_size
            self.num_heads_local = self.num_heads // tp_size
            self.num_xb_head_local = self.num_xb_head // tp_size

            # Fused input projection: [z, x, B, C, dt]
            # z: d_inner, x: d_xb, B: d_xb, C: d_inner, dt: dt_rank
            # MergedColumnParallelLinear shards each output by tp_size
            self.in_proj = MergedColumnParallelLinear(
                self.d_model,
                [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
                bias=False,
                prefix=f"{prefix}.in_proj",
                quant_config=quant_config,
            )

            # Conv1d as ColumnParallelLinear (matching vLLM MambaMixer pattern)
            # Weight shape: (conv_dim, 1, d_conv) stored as (conv_dim, d_conv)
            self.conv1d = ColumnParallelLinear(
                input_size=self.d_conv,
                output_size=self.conv_dim,
                bias=True,
                prefix=f"{prefix}.conv1d",
            )
            # Unsqueeze to fit conv1d weight shape into linear weight shape
            self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

            # Delta time projection (bias applied in forward, then again via
            # delta_bias for double-bias matching the trained model)
            self.dt_proj = ColumnParallelLinear(
                self.dt_rank,
                self.d_inner,
                bias=True,
                prefix=f"{prefix}.dt_proj",
                quant_config=quant_config,
            )

            # A matrix with TP-aware weight loading
            # Checkpoint stores A_log, weight_loader converts: A = -exp(A_log)
            def _weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
                tp_rank = get_tensor_model_parallel_rank()
                tp_sz = get_tensor_model_parallel_world_size()
                param.data.copy_(
                    loaded_weight.data.split(
                        loaded_weight.shape[0] // tp_sz, dim=0
                    )[tp_rank]
                )

            def _A_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
                _weight_loader(param, -torch.exp(loaded_weight.float()))

            self.A = nn.Parameter(
                torch.empty(
                    self.d_inner // tp_size,
                    self.d_state,
                    dtype=torch.float32,
                )
            )
            set_weight_attrs(self.A, {"weight_loader": _A_weight_loader})

            # D skip parameter with TP-aware weight loading
            self.D = nn.Parameter(torch.ones(self.d_inner // tp_size))
            set_weight_attrs(self.D, {"weight_loader": _weight_loader})

            # Output projection (all-reduce across TP ranks)
            self.out_proj = RowParallelLinear(
                self.d_inner,
                self.d_model,
                bias=False,
                input_is_parallel=True,
                prefix=f"{prefix}.out_proj",
                quant_config=quant_config,
            )

            self.activation = "silu"

            # Store cache config for prefix caching support
            self.mamba_block_size = (
                cache_config.mamba_block_size
                if cache_config is not None else 0
            )

            # Register in static_forward_context (unconditionally)
            # Required for V1 cache discovery and custom op dispatch
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

            # kv_cache placeholder — V1 engine binds real tensors via MambaBase
            self.kv_cache = (torch.tensor([]), torch.tensor([]))

            # Precomputed static tensors (lazily initialized on first forward)
            self._precomputed = False

        # =================================================================
        # Precomputed static tensors (avoid per-forward recomputation)
        # =================================================================

        def _precompute_static_tensors(self):
            """Precompute static tensor views/conversions once after weight loading."""
            self._conv_weight = self.conv1d.weight.reshape(self.conv_dim_local, self.d_conv)
            self._D_float = self.D.float()
            self._dt_bias_float = self.dt_proj.bias.float()
            nheads = self.num_heads_local
            head_dim = self.d_state
            self._A_mh = self.A.view(nheads, head_dim, self.d_state)
            self._D_mh = self._D_float.view(nheads, head_dim)
            self._dt_bias_mh = self._dt_bias_float.view(nheads, head_dim)
            self._precomputed = True

        # =================================================================
        # MambaBase interface (required for V1 cache allocation)
        # =================================================================

        def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
            """Return state shapes for vLLM cache allocation.

            Uses MambaStateShapeCalculator for standard case (conv_dim == d_inner).
            Falls back to manual calculation for non-standard conv_dim.
            Convention: conv_state (d_conv-1, conv_dim), ssm_state (d_inner, d_state).
            """
            tp_size = get_tensor_model_parallel_world_size()
            if (self.conv_dim == self.d_inner
                    and _vllm_MambaStateShapeCalculator is not None):
                return _vllm_MambaStateShapeCalculator.mamba1_state_shape(
                    tp_world_size=tp_size,
                    intermediate_size=self.d_inner,
                    state_size=self.d_state,
                    conv_kernel=self.d_conv,
                )
            conv_state_shape = (self.d_conv - 1, self.conv_dim // tp_size)
            ssm_state_shape = (self.d_inner // tp_size, self.d_state)
            return (conv_state_shape, ssm_state_shape)

        def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
            """Return state dtypes for vLLM cache allocation.

            SSM state defaults to float32 because Mamba's recurrent nature
            compounds bfloat16 rounding errors across tokens. Conv state
            can remain at model dtype (just a sliding window, no accumulation).
            Users can override via --mamba-ssm-cache-dtype.
            """
            if (self.model_config is not None and self.cache_config is not None
                    and _vllm_MambaStateDtypeCalculator is not None):
                # If user explicitly set mamba_ssm_cache_dtype, honor it.
                # Otherwise, default SSM state to float32 for accuracy.
                if self.cache_config.mamba_ssm_cache_dtype != "auto":
                    return _vllm_MambaStateDtypeCalculator.mamba1_state_dtype(
                        self.model_config.dtype,
                        self.cache_config.mamba_cache_dtype,
                        self.cache_config.mamba_ssm_cache_dtype,
                    )
                # Auto mode: conv state at model dtype, SSM state at float32
                from vllm.model_executor.layers.mamba.mamba_utils import (
                    get_kv_cache_torch_dtype,
                )
                conv_dtype = get_kv_cache_torch_dtype(
                    self.cache_config.mamba_cache_dtype,
                    self.model_config.dtype,
                )
                return (conv_dtype, torch.float32)
            dtype = self.out_proj.weight.dtype
            return (dtype, torch.float32)

        @property
        def mamba_type(self) -> str:
            """Return mamba type for vLLM backend selection."""
            return "mamba1"

        # =================================================================
        # CustomOp forward methods
        # =================================================================

        def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
            """Dispatch via custom op for CUDA graph compatibility.

            The torch.ops.vllm.mambainllama_mixer custom op looks up this
            layer by prefix in forward_context.no_compile_layers and calls
            forward_cuda(). The torch compiler excludes this op from
            CUDA graphs, so Mamba runs in eager mode while everything
            else gets compiled.
            """
            torch.ops.vllm.mambainllama_mixer(hidden_states, output, self.prefix)

        def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor):
            """Empty stub — forward_cuda handles all computation."""
            pass

        def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor):
            """CUDA forward with V1 state management.

            Called by torch.ops.vllm.mambainllama_mixer via forward_context lookup.
            Gets state from self.kv_cache (bound by vLLM V1 engine).
            Handles mixed prefill+decode batches (V1 sends decode first, then prefill).
            """
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

            if attn_metadata is None:
                # V1 profile run — write dummy output (don't precompute yet,
                # weights may not be loaded during early profiling)
                num_tokens = hidden_states.shape[0]
                output[:num_tokens] = self.out_proj(
                    hidden_states[..., :self.d_inner_local]
                )[0]
                return

            # Lazily precompute static tensors (after profile check —
            # weights are guaranteed loaded by the time real forward runs)
            if not self._precomputed:
                self._precompute_static_tensors()

            assert isinstance(attn_metadata, dict)
            layer_metadata = attn_metadata[self.prefix]

            # Get state from kv_cache (bound by vLLM V1)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            # Allocated as (pool, d_conv-1, conv_dim), transpose to
            # (pool, conv_dim, d_conv-1) so stride(1)=1 for causal_conv1d ops
            conv_state = self_kv_cache[0].transpose(-1, -2)
            # ssm_state: (pool, d_inner, d_state) — used directly by ops
            ssm_state = self_kv_cache[1]

            state_indices = layer_metadata.state_indices_tensor

            num_prefill_tokens = layer_metadata.num_prefill_tokens
            num_decode_tokens = layer_metadata.num_decode_tokens
            num_prefills = layer_metadata.num_prefills
            num_actual_tokens = num_prefill_tokens + num_decode_tokens
            has_prefill = num_prefill_tokens > 0
            has_decode = num_decode_tokens > 0

            # Prefix caching: extract block_idx fields for state read/write
            prefix_caching_enabled = (
                self.cache_config is not None
                and self.cache_config.enable_prefix_caching
            )
            if (prefix_caching_enabled
                    and layer_metadata.block_idx_last_computed_token is not None):
                block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                    torch.split(
                        layer_metadata.block_idx_last_computed_token,
                        [num_decode_tokens, num_prefills],
                        dim=0,
                    )
                )
                block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                    torch.split(
                        layer_metadata.block_idx_last_scheduled_token,
                        [num_decode_tokens, num_prefills],
                        dim=0,
                    )
                )
                block_idx_first_scheduled_token_p = (
                    layer_metadata.block_idx_first_scheduled_token_p
                )
                num_computed_tokens_p = layer_metadata.num_computed_tokens_p
            else:
                block_idx_last_computed_token_d = None
                block_idx_last_computed_token_p = None
                block_idx_last_scheduled_token_d = None
                block_idx_last_scheduled_token_p = None
                block_idx_first_scheduled_token_p = None
                num_computed_tokens_p = None

            # ===== PROJECTION =====
            # Process all tokens (including CUDA graph padding) through in_proj
            # MergedColumnParallelLinear returns (output, bias) tuple
            # LoRA kernel requires contiguous tensor; ROCm non-contiguous
            # causes incorrect GEMM results when batch > 1
            if self.is_lora_enabled or current_platform.is_rocm():
                hidden_states = hidden_states.contiguous()
            zxbcdt = self.in_proj(hidden_states)[0]

            # Early slice to actual tokens — skip padding before expensive ops
            zxbcdt = zxbcdt[:num_actual_tokens]

            z, x, B, C, dt = torch.split(
                zxbcdt,
                [self.d_inner_local, self.d_xb_local, self.d_xb_local,
                 self.d_inner_local, self.dt_rank_local],
                dim=-1,
            )

            # Delta time projection WITH bias (model trained with double bias)
            # ColumnParallelLinear returns (output, bias) tuple
            dt = self.dt_proj(dt)[0]

            # Expand x via expand (zero-copy stride-0 view + one reshape)
            if self.repeat_kv_before_conv:
                x = x.view(-1, self.num_xb_head_local, 1, self.d_state) \
                     .expand(-1, -1, self.repeat_group, -1) \
                     .reshape(-1, self.d_inner_local)

            # Expand B via expand (zero-copy stride-0 view + one reshape)
            B = B.view(-1, self.num_xb_head_local, 1, self.d_state) \
                 .expand(-1, -1, self.repeat_group, -1) \
                 .reshape(-1, self.num_heads_local, self.d_state)

            # C is already d_inner_local, just reshape (direct view)
            C = C.view(-1, self.num_heads_local, self.d_state)

            # ===== SPLIT AND PROCESS =====
            # In V1: decode tokens come first, then prefill tokens
            ssm_outputs = []

            if has_prefill:
                # Prefill tokens are AFTER decode tokens
                x_p = x[num_decode_tokens:num_actual_tokens]
                z_p = z[num_decode_tokens:num_actual_tokens]
                B_p = B[num_decode_tokens:num_actual_tokens]
                C_p = C[num_decode_tokens:num_actual_tokens]
                dt_p = dt[num_decode_tokens:num_actual_tokens]

                state_indices_p = state_indices[num_decode_tokens:num_decode_tokens + num_prefills]
                query_start_loc_p = layer_metadata.query_start_loc_p
                has_initial_states_p = layer_metadata.has_initial_states_p

                # Conv1d (full sequence)
                # Input: (conv_dim, num_prefill_tokens)
                x_conv_p = causal_conv1d_fn(
                    x_p.transpose(0, 1),
                    self._conv_weight,
                    self.conv1d.bias,
                    conv_state,
                    query_start_loc_p,
                    cache_indices=state_indices_p,
                    has_initial_state=has_initial_states_p,
                    activation="silu",
                    block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                    block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                    initial_state_idx=block_idx_last_computed_token_p,
                    num_computed_tokens=num_computed_tokens_p,
                    block_size_to_align=self.mamba_block_size,
                    metadata=layer_metadata,
                )

                # SSM scan
                # Double bias: dt already has bias from dt_proj, delta_bias adds it again
                y_p = selective_scan_fn(
                    x_conv_p,
                    ssm_state,
                    dt_p.transpose(0, 1),
                    self.A,
                    B_p.permute(1, 2, 0),
                    C_p.permute(1, 2, 0),
                    D=self._D_float,
                    z=z_p.transpose(0, 1),
                    delta_bias=self._dt_bias_float,
                    delta_softplus=True,
                    query_start_loc=query_start_loc_p,
                    cache_indices=state_indices_p,
                    has_initial_state=has_initial_states_p,
                    block_size=self.mamba_block_size,
                    block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                    block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                    initial_state_idx=block_idx_last_computed_token_p,
                )

                # Transpose to tokens-first immediately (tokens, d_inner)
                ssm_outputs.append(y_p.transpose(0, 1))

            if has_decode:
                # Decode tokens are first
                x_d = x[:num_decode_tokens]
                z_d = z[:num_decode_tokens]
                B_d = B[:num_decode_tokens]
                C_d = C[:num_decode_tokens]
                dt_d = dt[:num_decode_tokens]

                state_indices_d = state_indices[:num_decode_tokens]

                # Prefix caching: separate read/write state indices
                if block_idx_last_computed_token_d is not None:
                    state_indices_d_input = state_indices_d.gather(
                        1, block_idx_last_computed_token_d.unsqueeze(1)
                    ).squeeze(1)
                    state_indices_d_output = state_indices_d.gather(
                        1, block_idx_last_scheduled_token_d.unsqueeze(1)
                    ).squeeze(1)
                else:
                    state_indices_d_input = state_indices_d
                    state_indices_d_output = state_indices_d

                # Conv update — batch-first: (num_decode, d_inner)
                # causal_conv1d_update expects x=(batch, dim), NOT (dim, batch)
                x_conv_d = causal_conv1d_update(
                    x_d,
                    conv_state,
                    self._conv_weight,
                    bias=self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=state_indices_d,
                    block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                    initial_state_idx=block_idx_last_computed_token_d,
                )

                # SSM state update — multi-head format for selective_state_update
                # The kernel asserts nheads % ngroups == 0, so we reshape
                # state/x/dt/z to (*, nheads, head_dim, ...) format.
                # Double bias: dt already has bias from dt_proj, dt_bias adds it again
                nheads = self.num_heads_local
                head_dim = self.d_state

                x_mh = x_conv_d.view(-1, nheads, head_dim)
                dt_mh = dt_d.view(-1, nheads, head_dim)
                z_mh = z_d.view(-1, nheads, head_dim)
                ssm_state_mh = ssm_state.view(
                    ssm_state.shape[0], nheads, head_dim, self.d_state
                )

                scan_outputs_d = torch.empty_like(x_mh)
                selective_state_update(
                    ssm_state_mh,
                    x_mh,
                    dt_mh,
                    self._A_mh,
                    B_d,
                    C_d,
                    D=self._D_mh,
                    z=z_mh,
                    dt_bias=self._dt_bias_mh,
                    dt_softplus=True,
                    state_batch_indices=state_indices_d_input,
                    dst_state_batch_indices=state_indices_d_output,
                    out=scan_outputs_d,
                )

                # Keep tokens-first: (num_decode, d_inner_local)
                scan_outputs_d = scan_outputs_d.reshape(
                    -1, self.d_inner_local
                )

                ssm_outputs.insert(0, scan_outputs_d)  # decode comes first

            # Combine in tokens-first layout and project
            # All ssm_outputs are now (tokens, d_inner_local)
            # RowParallelLinear returns (output, bias) tuple
            y_combined = (
                ssm_outputs[0] if len(ssm_outputs) == 1
                else torch.cat(ssm_outputs, dim=0)
            )
            if self.is_lora_enabled:
                out = self.out_proj(y_combined.contiguous())[0]
            else:
                out = self.out_proj(y_combined)[0]
            output[:num_actual_tokens] = out

else:
    # Fallback when vLLM components not available
    class MambaInLlamaMambaMixer(nn.Module):
        """Fallback MambaInLlama mixer when vLLM not available."""

        def __init__(
            self,
            config: MambaInLlamaMambaConfig,
            layer_idx: int,
            prefix: str = "",
            model_config=None,
            cache_config=None,
        ):
            super().__init__()
            self.layer_idx = layer_idx
            self.prefix = prefix

            self.d_model = config.d_model
            self.d_inner = config.d_inner
            self.d_xb = config.d_xb
            self.d_state = config.ssm_cfg.get("d_state", 16)
            self.d_conv = config.ssm_cfg.get("d_conv", 4)
            self.dt_rank = math.ceil(self.d_model / 16)

            self.num_xb_head = self.d_xb // self.d_state
            self.num_heads = self.d_inner // self.d_state
            self.repeat_group = self.d_inner // self.d_xb
            self.num_C_head = self.num_heads
            self.repeat_kv_before_conv = config.ssm_cfg.get("repeat_kv_before_conv", True)
            self.conv_dim = self.d_inner if self.repeat_kv_before_conv else self.d_xb

            self.in_proj = nn.Linear(
                self.d_model,
                2 * self.d_inner + 2 * self.d_xb + self.dt_rank,
                bias=False,
            )

            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=self.d_conv,
                groups=self.conv_dim,
                padding=self.d_conv - 1,
                bias=True,
            )

            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            self.A = nn.Parameter(-A)
            self.A._no_weight_decay = True

            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.D._no_weight_decay = True

            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

            self.kv_cache: tuple[torch.Tensor, ...] = (torch.tensor([]), torch.tensor([]))

        def get_state_shape(self):
            conv_state_shape = (self.d_conv - 1, self.conv_dim)
            ssm_state_shape = (self.d_inner, self.d_state)
            return (conv_state_shape, ssm_state_shape)

        def get_state_dtype(self):
            dtype = self.out_proj.weight.dtype
            return (dtype, torch.float32)

        @property
        def mamba_type(self):
            return "mamba1"

        def forward(self, hidden_states, output=None, **kwargs):
            # Stub for non-vLLM environments
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            batch, seqlen, _ = hidden_states.shape

            zxbcdt = self.in_proj(hidden_states)
            z, x, B, C, dt = torch.split(
                zxbcdt,
                [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
                dim=-1,
            )

            dt = self.dt_proj(dt)

            if self.repeat_kv_before_conv:
                x = rearrange(x, "b l (g d) -> b l g d", g=self.num_xb_head)
                x = torch.repeat_interleave(x, self.repeat_group, dim=-2)
                x = rearrange(x, "b l g d -> b l (g d)")

            x = rearrange(x, "b l d -> b d l")
            z = rearrange(z, "b l d -> b d l")

            x = F.silu(self.conv1d(x)[..., :seqlen])

            # Simplified SSM (no state caching in fallback)
            y = x * F.silu(z)
            y = y + self.D.to(x.dtype).unsqueeze(0).unsqueeze(-1) * x

            y = rearrange(y, "b d l -> b l d")
            result = self.out_proj(y).squeeze(0)
            if output is not None:
                output[:result.shape[0]] = result
            return result


# Register the custom op so torch.ops.vllm.mambainllama_mixer exists.
# This is what makes the custom op pattern work — forward() calls the op,
# the op looks up the layer by prefix and calls forward_cuda().
if _vllm_available:
    def _mambainllama_mixer_op(
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        forward_context: ForwardContext = get_forward_context()
        self = forward_context.no_compile_layers[layer_name]
        self.forward_cuda(hidden_states=hidden_states, output=output)

    def _mambainllama_mixer_fake(
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        return

    direct_register_custom_op(
        op_name="mambainllama_mixer",
        op_func=_mambainllama_mixer_op,
        mutates_args=["output"],
        fake_impl=_mambainllama_mixer_fake,
    )
