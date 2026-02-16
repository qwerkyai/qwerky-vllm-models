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
"""MambaInLlama model for vLLM using native Triton ops.

This module uses vLLM's native Mamba ops for maximum performance.
No mamba_ssm or causal_conv1d compilation required.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Iterable, ClassVar, Literal
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

from transformers.utils import logging

from .configuration import MambaInLlamaMambaConfig


def _load_mamba_config(model_path: str) -> dict:
    """Load mamba_config.json from model directory if it exists.

    Many MambaInLlama models store Mamba-specific config (attn_layers, d_inner, d_xb)
    in a separate mamba_config.json file rather than the main config.json.
    """
    mamba_config = {}

    # Try to find mamba_config.json
    possible_paths = [
        os.path.join(model_path, "mamba_config.json"),
    ]

    # Handle HuggingFace cache paths
    if "huggingface" in model_path or "hub" in model_path:
        # The model_path might be the cache directory
        possible_paths.append(os.path.join(model_path, "mamba_config.json"))

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    mamba_config = json.load(f)
                    logging.get_logger(__name__).info(f"Loaded mamba_config.json from {path}")
                    break
            except Exception as e:
                logging.get_logger(__name__).warning(f"Failed to load {path}: {e}")

    return mamba_config

logger = logging.get_logger(__name__)

# =============================================================================
# vLLM NATIVE IMPORTS
# =============================================================================

_vllm_available = False

# Core vLLM imports
try:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
        ParallelLMHead,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.attention.layer import Attention
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.platforms import current_platform
    from vllm.config import VllmConfig, CacheConfig, ModelConfig, get_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.forward_context import ForwardContext, get_forward_context
    from vllm.utils.torch_utils import direct_register_custom_op

    _vllm_available = True
    logger.info("vLLM core components loaded successfully")
except ImportError as e:
    logger.warning(f"vLLM not available: {e}")
    RMSNorm = None
    get_current_vllm_config = None
    get_forward_context = None

# MambaBase import for proper vLLM integration
_MambaBase = None
try:
    from vllm.model_executor.layers.mamba.abstract import MambaBase as _MambaBase
    logger.info("vLLM MambaBase loaded successfully")
except ImportError as e:
    logger.warning(f"vLLM MambaBase not available: {e}")

# CustomOp import for proper callability with MambaBase
_CustomOp = None
try:
    from vllm.model_executor.custom_op import CustomOp as _CustomOp
    logger.info("vLLM CustomOp loaded successfully")
except ImportError as e:
    logger.warning(f"vLLM CustomOp not available: {e}")

# Mamba1AttentionMetadata for state indices
_Mamba1AttentionMetadata = None
try:
    from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata as _Mamba1AttentionMetadata
    logger.info("vLLM Mamba1AttentionMetadata loaded successfully")
except ImportError as e:
    logger.warning(f"vLLM Mamba1AttentionMetadata not available: {e}")

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
    logger.info("vLLM Mamba ops loaded successfully")
except ImportError as e:
    logger.warning(f"vLLM Mamba ops not available: {e}")

# Try to import Sampler (location varies by vLLM version)
_vllm_Sampler = None
try:
    from vllm.model_executor.layers.sampler import Sampler as _vllm_Sampler
except ImportError:
    try:
        from vllm.v1.sample.sampler import Sampler as _vllm_Sampler
    except ImportError:
        pass

# Try to import MambaModelConfig for hybrid model support
_vllm_MambaModelConfig = None
try:
    from vllm.model_executor.models.config import MambaModelConfig as _vllm_MambaModelConfig
except ImportError:
    pass

# Try to import protocol interfaces for model registration
_HasInnerState = None
_IsHybrid = None
try:
    from vllm.model_executor.models.interfaces import HasInnerState as _HasInnerState
    from vllm.model_executor.models.interfaces import IsHybrid as _IsHybrid
except ImportError:
    pass

# Try to import state calculators
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
# FALLBACK IMPLEMENTATIONS (for when vLLM ops not available)
# =============================================================================

class RMSNormFallback(nn.Module):
    """RMSNorm fallback."""
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = self.weight * (x * torch.rsqrt(variance + self.eps)).to(input_dtype)
        if residual is not None:
            return x, residual
        return x


if RMSNorm is None:
    RMSNorm = RMSNormFallback


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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

            # Register in static_forward_context (unconditionally)
            # Required for V1 cache discovery and custom op dispatch
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

            # kv_cache placeholder — V1 engine binds real tensors via MambaBase
            self.kv_cache = (torch.tensor([]), torch.tensor([]))

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
            """Return state dtypes for vLLM cache allocation."""
            if (self.model_config is not None and self.cache_config is not None
                    and _vllm_MambaStateDtypeCalculator is not None):
                return _vllm_MambaStateDtypeCalculator.mamba1_state_dtype(
                    self.model_config.dtype,
                    self.cache_config.mamba_cache_dtype,
                    self.cache_config.mamba_ssm_cache_dtype,
                )
            dtype = self.out_proj.weight.dtype
            return (dtype, dtype)

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
                # V1 profile run — write dummy output
                num_tokens = hidden_states.shape[0]
                output[:num_tokens] = self.out_proj(
                    hidden_states[..., :self.d_inner_local]
                )[0]
                return

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
            if prefix_caching_enabled:
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
            z, x, B, C, dt = torch.split(
                zxbcdt,
                [self.d_inner_local, self.d_xb_local, self.d_xb_local,
                 self.d_inner_local, self.dt_rank_local],
                dim=-1,
            )

            # Delta time projection WITH bias (model trained with double bias)
            # ColumnParallelLinear returns (output, bias) tuple
            dt = self.dt_proj(dt)[0]

            # Expand x via repeat_interleave if needed
            if self.repeat_kv_before_conv:
                x = rearrange(x, "t (g d) -> t g d", g=self.num_xb_head_local)
                x = torch.repeat_interleave(x, self.repeat_group, dim=-2)
                x = rearrange(x, "t g d -> t (g d)")

            # Expand B via repeat_interleave
            B = rearrange(B, "t (g d) -> t g d", d=self.d_state)
            B = torch.repeat_interleave(B, self.repeat_group, dim=-2)

            # C is already d_inner_local, just reshape
            C = rearrange(C, "t (g d) -> t g d", d=self.d_state)

            # Conv weight: (conv_dim_local, d_conv) from (conv_dim_local, 1, d_conv)
            conv_weight = rearrange(self.conv1d.weight, "d 1 w -> d w")

            D = self.D.float()

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
                    conv_weight,
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
                    rearrange(B_p, "t h d -> h d t"),
                    rearrange(C_p, "t h d -> h d t"),
                    D=D,
                    z=z_p.transpose(0, 1),
                    delta_bias=self.dt_proj.bias.float(),
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
                if prefix_caching_enabled:
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
                    conv_weight,
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
                A_mh = self.A.view(nheads, head_dim, self.d_state)
                D_mh = D.view(nheads, head_dim)
                dt_bias_mh = self.dt_proj.bias.float().view(nheads, head_dim)
                ssm_state_mh = ssm_state.view(
                    ssm_state.shape[0], nheads, head_dim, self.d_state
                )

                scan_outputs_d = torch.empty_like(x_mh)
                selective_state_update(
                    ssm_state_mh,
                    x_mh,
                    dt_mh,
                    A_mh,
                    B_d,
                    C_d,
                    D=D_mh,
                    z=z_mh,
                    dt_bias=dt_bias_mh,
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
            return (dtype, dtype)

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


# =============================================================================
# MLP LAYER
# =============================================================================

class MLP(nn.Module):
    """MLP layer with SiLU activation."""

    def __init__(self, d_model: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_fn = nn.SiLU() if hidden_act == "silu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# ATTENTION LAYER
# =============================================================================

class MHADecoderLayer(nn.Module):
    """Multi-Head Attention decoder layer using vLLM's Attention for KV caching.

    Uses vLLM's Attention class which handles KV cache, PagedAttention,
    GQA, causal masking, and flash attention internally.
    """

    def __init__(self, config: MambaInLlamaMambaConfig, layer_idx: int,
                 cache_config: "CacheConfig" = None, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

        # vLLM RoPE
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, 'rope_parameters', None),
        )

        # vLLM Attention — handles KV cache, paging, GQA, masking
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, hidden_size] (2D, vLLM V1 format)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = self.rotary_emb(positions, q, k)

        # vLLM Attention handles KV cache, paging, GQA, masking internally
        attn_output = self.attn(q, k, v)

        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# MAMBA DECODER LAYER
# =============================================================================

class MambaDecoderLayer(nn.Module):
    """Mamba SSM decoder layer."""

    def __init__(self, config: MambaInLlamaMambaConfig, layer_idx: int,
                 prefix: str = "", model_config=None, cache_config=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.prefix = prefix

        # Pass prefix, model_config, and cache_config to mixer
        mamba_prefix = f"{prefix}.mamba" if prefix else f"model.layers.{layer_idx}.mamba"
        self.mamba = MambaInLlamaMambaMixer(
            config, layer_idx, prefix=mamba_prefix,
            model_config=model_config, cache_config=cache_config,
        )
        self.mlp = MLP(config.d_model, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Output tensor pattern for custom op compatibility
        output = torch.empty_like(hidden_states)
        self.mamba(hidden_states, output)
        hidden_states = residual + output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# MODEL BACKBONE
# =============================================================================

class MambaInLlamaMambaModel(nn.Module):
    """MambaInLlama Model backbone."""

    def __init__(self, config: MambaInLlamaMambaConfig, prefix: str = "",
                 cache_config: "CacheConfig" = None, model_config: "ModelConfig" = None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{layer_idx}" if prefix else f"model.layers.{layer_idx}"
            if layer_idx in config.attn_layers:
                self.layers.append(MHADecoderLayer(
                    config, layer_idx,
                    cache_config=cache_config,
                    prefix=layer_prefix,
                ))
            else:
                self.layers.append(MambaDecoderLayer(
                    config, layer_idx, prefix=layer_prefix,
                    model_config=model_config, cache_config=cache_config,
                ))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (required by VllmModel interface)."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata=None,
    ) -> torch.Tensor:
        """Forward pass with vLLM state management.

        Args:
            input_ids: Input token IDs [num_tokens] or [batch, seq]
            positions: Position indices for RoPE [num_tokens]
            attn_metadata: vLLM attention metadata for Mamba state indices
        """
        hidden_states = self.embed_input_ids(input_ids)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, MambaDecoderLayer):
                hidden_states = layer(hidden_states)
            else:
                # MHA layer — vLLM Attention handles KV cache internally
                hidden_states = layer(hidden_states, positions)

        hidden_states = self.norm(hidden_states)
        return hidden_states


# =============================================================================
# NATIVE vLLM MODEL CLASS
# =============================================================================

# Dynamically create base classes with protocol inheritance
_NativeBaseClasses = [nn.Module]
if _HasInnerState is not None:
    _NativeBaseClasses.append(_HasInnerState)
if _IsHybrid is not None:
    _NativeBaseClasses.append(_IsHybrid)
_NativeBaseClasses = tuple(_NativeBaseClasses)


class MambaInLlamaMambaForCausalLMNative(*_NativeBaseClasses):
    """Native vLLM-compatible MambaInLlama model.

    This model supports the 'generate' runner by:
    1. Inheriting from HasInnerState and IsHybrid protocols
    2. Implementing compute_logits() and sample() methods
    3. Having architecture name ending in 'ForCausalLM'
    """

    # Protocol-required class variables for vLLM model inspection
    is_hybrid: ClassVar[Literal[True]] = True
    has_inner_state: ClassVar[Literal[True]] = True

    def __init__(
        self,
        vllm_config=None,
        config: MambaInLlamaMambaConfig = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()

        if vllm_config is not None and hasattr(vllm_config, "model_config"):
            model_config = vllm_config.model_config
            if hasattr(model_config, "hf_config"):
                hf_cfg = model_config.hf_config
                hidden_size = getattr(hf_cfg, "hidden_size", 4096)
                intermediate_size = getattr(hf_cfg, "intermediate_size", 11008)

                config_kwargs = dict(
                    vocab_size=getattr(hf_cfg, "vocab_size", 32000),
                    hidden_size=hidden_size,
                    num_hidden_layers=getattr(hf_cfg, "num_hidden_layers", 32),
                    num_attention_heads=getattr(hf_cfg, "num_attention_heads", 32),
                    num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", None),
                    intermediate_size=intermediate_size,
                    rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
                    rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
                )

                # Try to load mamba_config.json for Mamba-specific settings
                # Many MambaInLlama models store attn_layers, d_inner, d_xb there
                mamba_cfg = {}
                if hasattr(model_config, "model") and model_config.model:
                    model_path = model_config.model
                    logger.info(f"Looking for mamba_config.json for model: {model_path}")
                    # Handle HuggingFace hub models
                    try:
                        from huggingface_hub import hf_hub_download
                        # Try to download mamba_config.json (will use cache if available)
                        try:
                            mamba_config_path = hf_hub_download(
                                model_path, "mamba_config.json"
                            )
                            with open(mamba_config_path, "r") as f:
                                mamba_cfg = json.load(f)
                                logger.info(f"Loaded mamba_config.json from {mamba_config_path}")
                                logger.info(f"mamba_config contents: attn_layers={mamba_cfg.get('attn_layers')}, d_inner={mamba_cfg.get('d_inner')}, d_xb={mamba_cfg.get('d_xb')}")
                        except Exception as e:
                            logger.warning(f"Could not load mamba_config.json: {e}")
                            # Try local path as fallback
                            mamba_cfg = _load_mamba_config(model_path)
                    except ImportError:
                        # huggingface_hub not available, try local path
                        logger.warning("huggingface_hub not available, trying local path")
                        mamba_cfg = _load_mamba_config(model_path)

                # Try to get attn_layers from various possible locations
                # Priority: mamba_config.json > hf_config attributes
                attn_layers = None

                # First check mamba_config.json
                if mamba_cfg.get("attn_layers"):
                    attn_layers = mamba_cfg["attn_layers"]
                    logger.info(f"Found attn_layers from mamba_config.json: {attn_layers}")
                # Then check HF config
                elif hasattr(hf_cfg, "attn_layers") and hf_cfg.attn_layers is not None:
                    attn_layers = hf_cfg.attn_layers
                elif hasattr(hf_cfg, "attention_layers") and hf_cfg.attention_layers is not None:
                    attn_layers = hf_cfg.attention_layers
                elif hasattr(hf_cfg, "ssm_cfg") and isinstance(hf_cfg.ssm_cfg, dict):
                    attn_layers = hf_cfg.ssm_cfg.get("attn_layers") or hf_cfg.ssm_cfg.get("attention_layers")

                if attn_layers:
                    config_kwargs["attn_layers"] = attn_layers
                    logger.info(f"Using attn_layers: {attn_layers}")
                else:
                    logger.warning(f"No attn_layers found! Model will use ALL Mamba layers (no attention).")
                    logger.warning(f"HF config attrs: {[a for a in dir(hf_cfg) if not a.startswith('_')]}")

                # Get Mamba dimensions - priority: mamba_config.json > hf_config
                if mamba_cfg.get("d_model"):
                    config_kwargs["d_model"] = mamba_cfg["d_model"]
                elif hasattr(hf_cfg, "d_model") and hf_cfg.d_model is not None:
                    config_kwargs["d_model"] = hf_cfg.d_model

                if mamba_cfg.get("d_inner"):
                    config_kwargs["d_inner"] = mamba_cfg["d_inner"]
                elif hasattr(hf_cfg, "d_inner") and hf_cfg.d_inner is not None:
                    config_kwargs["d_inner"] = hf_cfg.d_inner

                if mamba_cfg.get("d_xb"):
                    config_kwargs["d_xb"] = mamba_cfg["d_xb"]
                elif hasattr(hf_cfg, "d_xb") and hf_cfg.d_xb is not None:
                    config_kwargs["d_xb"] = hf_cfg.d_xb

                if mamba_cfg.get("ssm_config"):
                    config_kwargs["ssm_cfg"] = mamba_cfg["ssm_config"]
                elif hasattr(hf_cfg, "ssm_cfg") and hf_cfg.ssm_cfg is not None:
                    config_kwargs["ssm_cfg"] = hf_cfg.ssm_cfg

                logger.info(f"Final config_kwargs: d_inner={config_kwargs.get('d_inner')}, d_xb={config_kwargs.get('d_xb')}, attn_layers={config_kwargs.get('attn_layers')}")
                config = MambaInLlamaMambaConfig(**config_kwargs)

        if config is None:
            raise ValueError("Config required for model initialization")

        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix

        # Extract cache_config and model_config from vllm_config
        cache_config = None
        vllm_model_config = None
        if vllm_config is not None:
            if hasattr(vllm_config, 'cache_config'):
                cache_config = vllm_config.cache_config
            if hasattr(vllm_config, 'model_config'):
                vllm_model_config = vllm_config.model_config

        # Pass prefix, cache_config, and model_config to model backbone
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = MambaInLlamaMambaModel(
            config, prefix=model_prefix,
            cache_config=cache_config, model_config=vllm_model_config,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)

        # vLLM components
        self._vllm_logits_processor = None
        self._vllm_sampler = None
        if _vllm_available:
            try:
                self._vllm_logits_processor = LogitsProcessor(config.vocab_size)
            except:
                pass
        if _vllm_Sampler is not None:
            try:
                self._vllm_sampler = _vllm_Sampler()
            except:
                pass

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (required by VllmModelForTextGeneration)."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        kv_caches: list = None,
        attn_metadata=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_tensors=None,
        **kwargs,
    ) -> torch.Tensor:
        """vLLM-style forward pass.

        With vLLM V1:
        - Mamba layers get state from self.kv_cache (bound by vLLM via MambaBase)
        - Attention layers get KV cache from vLLM Attention class (via forward context)
        - Positions and attn_metadata come from vLLM model runner
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits for vLLM sampling."""
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

        if self._vllm_logits_processor is not None:
            return self._vllm_logits_processor(self.lm_head, hidden_states)

        return self.lm_head(hidden_states)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Sample tokens from logits."""
        if self._vllm_sampler is not None:
            return self._vllm_sampler(logits, sampling_metadata)
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint using vLLM weight_loader pattern.

        Handles weight name transformations:
        1. mha.in_proj.weight -> split into q_proj, k_proj, v_proj
        2. mha.out_proj.weight -> rename to o_proj.weight
        3. mamba.A_log -> mamba.A (via set_weight_attrs weight_loader)
        4. mamba.in_proj.weight -> split into 5 shards for MergedColumnParallelLinear
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Mamba in_proj shard sizes: [z, x, B, C, dt]
        d_inner = self.config.d_inner
        d_xb = self.config.d_xb
        dt_rank = math.ceil(self.config.d_model / 16)
        mamba_in_proj_sizes = [d_inner, d_xb, d_xb, d_inner, dt_rank]

        # MHA attention dimensions for QKV split
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads or num_heads
        head_dim = self.config.hidden_size // num_heads
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        for name, loaded_weight in weights:
            # === MHA attention: fused in_proj -> split into q, k, v ===
            if ".mha.in_proj.weight" in name:
                base_name = name.replace(".mha.in_proj.weight", "")
                q_weight = loaded_weight[:q_dim, :]
                k_weight = loaded_weight[q_dim:q_dim + kv_dim, :]
                v_weight = loaded_weight[q_dim + kv_dim:, :]
                for suffix, weight in [(".q_proj.weight", q_weight),
                                       (".k_proj.weight", k_weight),
                                       (".v_proj.weight", v_weight)]:
                    param_name = base_name + suffix
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, weight)
                        loaded_params.add(param_name)
                continue

            # === MHA attention: out_proj rename ===
            if ".mha.out_proj." in name:
                new_name = name.replace(".mha.out_proj.", ".o_proj.")
                if new_name in params_dict:
                    param = params_dict[new_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(new_name)
                continue

            # === Mamba A_log -> A (weight_loader handles conversion + TP) ===
            if ".mamba.A_log" in name:
                new_name = name.replace(".mamba.A_log", ".mamba.A")
                if new_name in params_dict:
                    param = params_dict[new_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(new_name)
                continue

            # === Mamba in_proj: split fused weight into 5 shards ===
            if ".mamba.in_proj.weight" in name:
                param_name = name
                if param_name not in params_dict:
                    param_name = f"model.{name}" if not name.startswith("model.") else name[6:]
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    shards = torch.split(loaded_weight, mamba_in_proj_sizes, dim=0)
                    for shard_id, shard_weight in enumerate(shards):
                        weight_loader(param, shard_weight, shard_id)
                    loaded_params.add(param_name)
                continue

            # === Default: use weight_loader if available ===
            param_name = name
            if param_name not in params_dict:
                # Try with/without model prefix
                if name.startswith("model."):
                    param_name = name[6:]
                else:
                    param_name = f"model.{name}"

            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(param_name)

        logger.info(f"Loaded {len(loaded_params)}/{len(params_dict)} parameters")
        if len(loaded_params) < len(params_dict):
            missing = set(params_dict.keys()) - loaded_params
            logger.warning(f"{len(missing)} params not loaded: {list(missing)[:10]}...")

        return loaded_params

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple:
        """Calculate Mamba state shapes."""
        if _vllm_MambaStateShapeCalculator is None:
            return ((3, 4096), (4096, 16))

        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config

        d_inner = getattr(hf_config, "d_inner", hf_config.hidden_size)
        ssm_cfg = getattr(hf_config, "ssm_cfg", {})
        d_state = ssm_cfg.get("d_state", 16)
        d_conv = ssm_cfg.get("d_conv", 4)

        return _vllm_MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=d_inner,
            state_size=d_state,
            conv_kernel=d_conv,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config) -> tuple:
        """Get Mamba state dtypes."""
        if _vllm_MambaStateDtypeCalculator is None:
            return (torch.bfloat16, torch.bfloat16)

        return _vllm_MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def is_backend_compatible(cls) -> bool:
        return True


# =============================================================================
# ALIAS FOR HF CONFIG COMPATIBILITY
# =============================================================================
# HuggingFace model configs specify "MambaInLlamaMambaForCausalLM" as the
# architecture. This alias ensures vLLM can find and load the class.
MambaInLlamaMambaForCausalLM = MambaInLlamaMambaForCausalLMNative
