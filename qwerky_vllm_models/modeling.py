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
"""QwerkyLlamaMambaHybrid model for vLLM using native Triton ops.

This module uses vLLM's native Mamba ops for maximum performance.
No mamba_ssm or causal_conv1d compilation required.
"""

from collections.abc import Iterable
import json
import math
import os
from typing import ClassVar, Literal, Optional

import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    get_kv_cache_torch_dtype,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

from .configuration import QwerkyLlamaMambaHybridConfig

logger = init_logger(__name__)


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def _load_mamba_config(model_path: str) -> dict:
    """Load mamba_config.json from model directory if it exists."""
    path = os.path.join(model_path, "mamba_config.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                mamba_config = json.load(f)
                logger.info("Loaded mamba_config.json from %s", path)
                return mamba_config
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
    return {}


def _augment_config_from_mamba_json(
    config: QwerkyLlamaMambaHybridConfig,
    model_config: ModelConfig,
) -> None:
    """Augment config from mamba_config.json for backward compat.

    Many older models store attn_layers, d_inner, d_xb in
    a separate mamba_config.json rather than the main config.json. This
    function fetches that file and patches the config in-place.
    """
    # Check alternative attribute names that PretrainedConfig may have set
    if not config.attn_layers:
        for attr in ("attention_layers",):
            val = getattr(config, attr, None)
            if val:
                config.attn_layers = val
                return
        if isinstance(config.ssm_cfg, dict):
            val = (config.ssm_cfg.get("attn_layers")
                   or config.ssm_cfg.get("attention_layers"))
            if val:
                config.attn_layers = val
                return

    # Try fetching mamba_config.json from HuggingFace Hub or local path
    mamba_cfg: dict = {}
    model_path = getattr(model_config, "model", None)
    if model_path:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_path, "mamba_config.json")
            with open(path) as f:
                mamba_cfg = json.load(f)
            logger.info("Loaded mamba_config.json from %s", path)
        except Exception:
            mamba_cfg = _load_mamba_config(model_path)

    if mamba_cfg.get("attn_layers"):
        config.attn_layers = mamba_cfg["attn_layers"]
    if mamba_cfg.get("d_model"):
        config.d_model = mamba_cfg["d_model"]
    if mamba_cfg.get("d_inner"):
        config.d_inner = mamba_cfg["d_inner"]
    if mamba_cfg.get("d_xb"):
        config.d_xb = mamba_cfg["d_xb"]
    if mamba_cfg.get("ssm_config"):
        config.ssm_cfg = mamba_cfg["ssm_config"]


# =============================================================================
# MAMBAINLLAMA MAMBA MIXER (Custom Op Pattern for CUDA Graphs)
# =============================================================================

@CustomOp.register("mambainllama_mixer")
class QwerkyLlamaMambaHybridMixer(MambaBase, CustomOp):
    """QwerkyLlamaMambaHybrid Mamba mixer with vLLM V1 integration.

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
        config: QwerkyLlamaMambaHybridConfig,
        layer_idx: int,
        prefix: str = "",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        is_lora_enabled: bool = False,
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
        )

        self.activation = "silu"

        # Store cache config for prefix caching support
        self.mamba_block_size = (
            cache_config.mamba_block_size
            if cache_config is not None else 0
        )

        # Register in static_forward_context (required for V1 cache
        # discovery and custom op dispatch)
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
        if self.conv_dim == self.d_inner:
            return MambaStateShapeCalculator.mamba1_state_shape(
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
        if self.model_config is not None and self.cache_config is not None:
            if self.cache_config.mamba_ssm_cache_dtype != "auto":
                return MambaStateDtypeCalculator.mamba1_state_dtype(
                    self.model_config.dtype,
                    self.cache_config.mamba_cache_dtype,
                    self.cache_config.mamba_ssm_cache_dtype,
                )
            # Auto mode: conv state at model dtype, SSM state at float32
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

            ssm_outputs.append(y_p)

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

            # Reshape back to (d_inner_local, num_decode) for cat with prefill
            scan_outputs_d = scan_outputs_d.reshape(
                -1, self.d_inner_local
            ).transpose(0, 1)

            ssm_outputs.insert(0, scan_outputs_d)  # decode comes first

        # Combine and project
        # RowParallelLinear returns (output, bias) tuple
        y_combined = (
            ssm_outputs[0] if len(ssm_outputs) == 1
            else torch.cat(ssm_outputs, dim=-1)
        )
        if self.is_lora_enabled:
            # LoRA kernel requires contiguous tensor
            out = self.out_proj(
                y_combined.transpose(0, 1).contiguous()
            )[0]
        else:
            out = self.out_proj(y_combined.transpose(0, 1))[0]
        output[:num_actual_tokens] = out


# Register the custom op so torch.ops.vllm.mambainllama_mixer exists.
# This is what makes the custom op pattern work — forward() calls the op,
# the op looks up the layer by prefix and calls forward_cuda().
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


# =============================================================================
# MLP LAYER
# =============================================================================

class MLP(nn.Module):
    """MLP layer with fused gate+up projection and SiluAndMul activation."""

    def __init__(self, d_model: int, intermediate_size: int, hidden_act: str = "silu",
                 prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            d_model,
            [intermediate_size, intermediate_size],
            bias=False,
            prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, d_model, bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.down_proj" if prefix else "down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


# =============================================================================
# ATTENTION LAYER
# =============================================================================

class MHADecoderLayer(nn.Module):
    """Multi-Head Attention decoder layer using vLLM's Attention for KV caching.

    Uses vLLM's Attention class which handles KV cache, PagedAttention,
    GQA, causal masking, and flash attention internally.
    """

    def __init__(self, config: QwerkyLlamaMambaHybridConfig, layer_idx: int,
                 cache_config: CacheConfig | None = None, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # TP-aware head counts
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)

        # Fused QKV — takes TOTAL heads, shards internally
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection — takes TOTAL input_size, shards internally
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

        # vLLM RoPE — build rope_parameters from config's rope_scaling + rope_theta
        rope_params = dict(getattr(config, 'rope_scaling', None) or {})
        rope_params["rope_theta"] = getattr(config, 'rope_theta', 10000.0)
        if "rope_type" not in rope_params:
            rope_params["rope_type"] = "default"
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_params,
        )

        # vLLM Attention — takes PER-RANK heads
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_act,
                       prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        positions: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [num_tokens, hidden_size] (2D, vLLM V1 format)
        # Fused RMSNorm residual: norm(x, residual) -> (normed, x+residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        hidden_states, _ = self.o_proj(attn_output)

        # Fused post-attention norm: adds residual + norms in one kernel
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# =============================================================================
# MAMBA DECODER LAYER
# =============================================================================

class MambaDecoderLayer(nn.Module):
    """Mamba SSM decoder layer."""

    def __init__(self, config: QwerkyLlamaMambaHybridConfig, layer_idx: int,
                 prefix: str = "", model_config: ModelConfig | None = None,
                 cache_config: CacheConfig | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.prefix = prefix

        # Pass prefix, model_config, and cache_config to mixer
        mamba_prefix = f"{prefix}.mamba" if prefix else f"model.layers.{layer_idx}.mamba"
        self.mamba = QwerkyLlamaMambaHybridMixer(
            config, layer_idx, prefix=mamba_prefix,
            model_config=model_config, cache_config=cache_config,
        )
        self.mlp = MLP(config.d_model, config.intermediate_size, config.hidden_act,
                       prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused RMSNorm residual: norm(x, residual) -> (normed, x+residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Output tensor pattern for custom op compatibility
        output = torch.empty_like(hidden_states)
        self.mamba(hidden_states, output)
        hidden_states = output

        # Fused post-attention norm: adds residual + norms in one kernel
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# =============================================================================
# MODEL BACKBONE
# =============================================================================

@support_torch_compile
class QwerkyLlamaMambaHybridModel(nn.Module):
    """QwerkyLlamaMambaHybrid Model backbone with PP support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: QwerkyLlamaMambaHybridConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix

        # Register splitting op so torch.compile doesn't try to compile our custom op
        compilation_config = vllm_config.compilation_config
        op_name = "vllm::mambainllama_mixer"
        if (compilation_config.splitting_ops is not None
                and op_name not in compilation_config.splitting_ops):
            compilation_config.splitting_ops.append(op_name)

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            if layer_idx in config.attn_layers:
                return MHADecoderLayer(
                    config, layer_idx, cache_config=cache_config, prefix=prefix,
                )
            return MambaDecoderLayer(
                config, layer_idx, prefix=prefix,
                model_config=model_config, cache_config=cache_config,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (required by VllmModel interface)."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with name transformations.

        Handles:
        1. mha.in_proj.weight -> split Q/K/V into qkv_proj (QKVParallelLinear)
        2. mha.out_proj -> rename to o_proj (RowParallelLinear)
        3. mamba.A_log -> mamba.A (custom weight_loader does -exp conversion)
        4. mamba.in_proj.weight -> 5-way split for MergedColumnParallelLinear
        5. mlp.gate_proj/up_proj -> gate_up_proj shards (MergedColumnParallelLinear)
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MHA dimensions for Q/K/V split
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads or num_heads
        head_dim = self.config.hidden_size // num_heads
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # Mamba in_proj shard sizes: [z, x, B, C, dt]
        d_inner = self.config.d_inner
        d_xb = self.config.d_xb
        dt_rank = math.ceil(self.config.d_model / 16)
        mamba_in_proj_sizes = [d_inner, d_xb, d_xb, d_inner, dt_rank]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # A_log -> A rename
            if "A_log" in name:
                name = name.replace("A_log", "A")

            # MHA: fused in_proj -> split Q/K/V
            if ".mha.in_proj.weight" in name:
                param_name = name.replace(".mha.in_proj.weight", ".qkv_proj.weight")
                if is_pp_missing_parameter(param_name, self):
                    continue
                if param_name not in params_dict:
                    continue
                param = params_dict[param_name]
                q_weight = loaded_weight[:q_dim, :]
                k_weight = loaded_weight[q_dim:q_dim + kv_dim, :]
                v_weight = loaded_weight[q_dim + kv_dim:, :]
                param.weight_loader(param, q_weight, "q")
                param.weight_loader(param, k_weight, "k")
                param.weight_loader(param, v_weight, "v")
                loaded_params.add(param_name)
                continue

            # MHA: out_proj -> o_proj rename
            if ".mha.out_proj." in name:
                name = name.replace(".mha.out_proj.", ".o_proj.")

            # Mamba: fused in_proj -> 5-way split
            if ".mamba.in_proj.weight" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                shards = torch.split(loaded_weight, mamba_in_proj_sizes, dim=0)
                for shard_id, shard_weight in enumerate(shards):
                    param.weight_loader(param, shard_weight, shard_id)
                loaded_params.add(name)
                continue

            # Stacked params: gate_proj/up_proj -> gate_up_proj
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Default loading
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


# =============================================================================
# CAUSAL LM MODEL
# =============================================================================

class QwerkyLlamaMambaHybridForCausalLMNative(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsPP,
    SupportsMambaPrefixCaching,
):
    """Native vLLM-compatible QwerkyLlamaMambaHybrid model.

    Supports the 'generate' runner via HasInnerState, IsHybrid,
    SupportsPP, and SupportsMambaPrefixCaching protocol inheritance.
    """

    # Protocol-required class variables for vLLM model inspection
    is_hybrid: ClassVar[Literal[True]] = True
    has_inner_state: ClassVar[Literal[True]] = True
    supports_mamba_prefix_caching: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: QwerkyLlamaMambaHybridConfig = vllm_config.model_config.hf_config

        # Augment config from mamba_config.json for backward compat
        # (older models store attn_layers/d_inner/d_xb there, not in config.json)
        if not config.attn_layers:
            _augment_config_from_mamba_json(config, vllm_config.model_config)
            if not config.attn_layers:
                logger.warning(
                    "No attn_layers found! Model will use ALL Mamba layers (no attention)."
                )

        self.config = config

        self.model = QwerkyLlamaMambaHybridModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (required by VllmModelForTextGeneration)."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits for vLLM sampling."""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate Mamba state shapes."""
        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config

        d_inner = getattr(hf_config, "d_inner", hf_config.hidden_size)
        ssm_cfg = getattr(hf_config, "ssm_cfg", {})
        d_state = ssm_cfg.get("d_state", 16)
        d_conv = ssm_cfg.get("d_conv", 4)

        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=d_inner,
            state_size=d_state,
            conv_kernel=d_conv,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Get Mamba state dtypes.

        Must match instance get_state_dtype(): SSM state defaults to float32
        when mamba_ssm_cache_dtype is "auto" to avoid bfloat16 rounding errors.
        """
        cache_config = vllm_config.cache_config
        if cache_config.mamba_ssm_cache_dtype != "auto":
            return MambaStateDtypeCalculator.mamba1_state_dtype(
                vllm_config.model_config.dtype,
                cache_config.mamba_cache_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )
        conv_dtype = get_kv_cache_torch_dtype(
            cache_config.mamba_cache_dtype,
            vllm_config.model_config.dtype,
        )
        return (conv_dtype, torch.float32)


# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================
# Short alias (without "Native" suffix)
QwerkyLlamaMambaHybridForCausalLM = QwerkyLlamaMambaHybridForCausalLMNative

# Legacy MambaInLlama aliases — HuggingFace model configs specify
# "MambaInLlamaMambaForCausalLM" as the architecture, so these ensure
# vLLM can still find and load models using the old names.
MambaInLlamaMambaForCausalLMNative = QwerkyLlamaMambaHybridForCausalLMNative
MambaInLlamaMambaForCausalLM = QwerkyLlamaMambaHybridForCausalLMNative
MambaInLlamaMambaMixer = QwerkyLlamaMambaHybridMixer
MambaInLlamaMambaModel = QwerkyLlamaMambaHybridModel
