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
    from vllm.attention.layer import Attention
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.forward_context import ForwardContext, get_forward_context

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
# MAMBAINLLAMA MAMBA MIXER (V1 State Management Integration)
# =============================================================================

# Try to import vLLM envs for V1 detection
_vllm_envs = None
try:
    from vllm import envs as _vllm_envs
except ImportError:
    pass


def _create_mamba_mixer_class():
    """Factory function to create MambaInLlamaMambaMixer with V1 state management.

    This properly integrates with vLLM's V1 engine by:
    1. Inheriting from CustomOp for proper forward dispatch
    2. Registering in static_forward_context for V1 state binding
    3. Using kv_cache for state storage (populated by vLLM)
    4. Getting layer-specific metadata from attn_metadata dict
    """

    # Determine base class - use CustomOp if available for proper V1 integration
    if _CustomOp is not None:
        # Use CustomOp for V1 integration with proper forward dispatch
        @_CustomOp.register("mambainllama_mixer")
        class MambaInLlamaMambaMixerVLLM(_CustomOp):
            """MambaInLlama Mamba mixer with V1 state management.

            Inherits from CustomOp to integrate with vLLM's V1 engine:
            - Registers in static_forward_context for state binding
            - Uses kv_cache for conv_state and ssm_state storage
            - Gets layer-specific metadata from attn_metadata[prefix]

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
            ):
                super().__init__()
                self.layer_idx = layer_idx
                self.prefix = prefix

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

                # Fused input projection: [z, x, B, C, dt]
                # z: d_inner, x: d_xb, B: d_xb, C: d_inner, dt: dt_rank
                self.in_proj = nn.Linear(
                    self.d_model,
                    2 * self.d_inner + 2 * self.d_xb + self.dt_rank,
                    bias=False,
                )

                # Conv1d - depthwise convolution
                self.conv1d = nn.Conv1d(
                    in_channels=self.conv_dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.d_conv,
                    groups=self.conv_dim,
                    padding=self.d_conv - 1,
                    bias=True,
                )

                # Delta time projection
                self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

                # Initialize dt_proj bias with inverse softplus
                dt_min, dt_max = 0.001, 0.1
                dt = torch.exp(
                    torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    self.dt_proj.bias.copy_(inv_dt)

                # A matrix (stored as -exp for vLLM ops)
                A = repeat(
                    torch.arange(1, self.d_state + 1, dtype=torch.float32),
                    "n -> d n",
                    d=self.d_inner,
                ).contiguous()
                # Store as -exp(log(A)) = -A for direct use in SSM
                self.A = nn.Parameter(-A)
                self.A._no_weight_decay = True

                # D skip parameter
                self.D = nn.Parameter(torch.ones(self.d_inner))
                self.D._no_weight_decay = True

                # Output projection
                self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

                self.activation = "silu"

                # =========================================================
                # V1 STATE MANAGEMENT REGISTRATION
                # This is the critical part that was missing!
                # =========================================================
                self._is_v1 = _vllm_envs is not None and getattr(_vllm_envs, 'VLLM_USE_V1', False)

                if self._is_v1 and prefix:
                    try:
                        compilation_config = get_current_vllm_config().compilation_config
                        if prefix in compilation_config.static_forward_context:
                            raise ValueError(f"Duplicate layer name: {prefix}")
                        compilation_config.static_forward_context[prefix] = self
                        logger.info(f"[V1] Registered MambaInLlamaMambaMixer '{prefix}' in static_forward_context")
                    except Exception as e:
                        logger.warning(f"[V1] Could not register in static_forward_context: {e}")
                        self._is_v1 = False

                # kv_cache placeholder - V1 engine will bind real tensors here
                # Structure: list of (conv_state, ssm_state) tuples, indexed by virtual_engine
                # Using list for V0 PP compatibility, inner tuple for (conv, ssm) states
                self.kv_cache = [(torch.tensor([]), torch.tensor([]))]

                # Internal state caches (fallback for non-V1 mode)
                self._conv_state: Optional[torch.Tensor] = None
                self._ssm_state: Optional[torch.Tensor] = None
                self._max_batch_size = 0

            def allocate_inference_cache(
                self,
                batch_size: int,
                max_seqlen: int,
                dtype: torch.dtype,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Allocate state caches for inference.

                Allocates in transposed format for stride_dim == 1 after transpose.
                """
                device = self.out_proj.weight.device
                # Store transposed: (batch, d_conv-1, conv_dim) so after transpose -> stride_dim == 1
                conv_state = torch.zeros(
                    batch_size, self.d_conv - 1, self.conv_dim,
                    device=device, dtype=dtype
                )
                # Store transposed: (batch, d_state, d_inner) so after transpose -> stride_dim == 1
                ssm_state = torch.zeros(
                    batch_size, self.d_state, self.d_inner,
                    device=device, dtype=dtype
                )
                return conv_state, ssm_state

            def _ensure_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
                """Ensure internal caches are allocated and sized correctly.

                Allocates in transposed format for stride_dim == 1 after transpose.
                """
                if self._conv_state is None or self._max_batch_size < batch_size:
                    # Store transposed for correct strides after transpose
                    self._conv_state = torch.zeros(
                        batch_size, self.d_conv - 1, self.conv_dim,
                        device=device, dtype=dtype
                    )
                    self._ssm_state = torch.zeros(
                        batch_size, self.d_state, self.d_inner,
                        device=device, dtype=dtype
                    )
                    self._max_batch_size = batch_size

            # =================================================================
            # MambaBase interface methods (required for vLLM state allocation)
            # =================================================================

            def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
                """Return state shapes for vLLM cache allocation.

                IMPORTANT: vLLM's causal_conv1d_fn expects conv_state shape (batch, dim, state_len)
                with stride_dim == 1 (dim axis must be innermost in memory). To achieve this,
                we store as (batch, state_len, dim) and transpose when using.

                So we return (d_conv-1, conv_dim) here, vLLM allocates (batch, d_conv-1, conv_dim),
                then we transpose to (batch, conv_dim, d_conv-1) with correct strides.
                """
                # Store transposed so stride_dim == 1 after transpose
                conv_state_shape = (self.d_conv - 1, self.conv_dim)
                ssm_state_shape = (self.d_state, self.d_inner)
                return (conv_state_shape, ssm_state_shape)

            def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
                """Return state dtypes for vLLM cache allocation."""
                dtype = self.out_proj.weight.dtype
                return (dtype, dtype)

            @property
            def mamba_type(self) -> str:
                """Return mamba type for vLLM backend selection."""
                return "mamba1"

            # =================================================================
            # CustomOp forward methods (forward_native and forward_cuda)
            # CustomOp.forward() dispatches to these based on platform
            # =================================================================

            def forward_native(
                self,
                hidden_states: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                ssm_state: Optional[torch.Tensor] = None,
                **kwargs,
            ) -> torch.Tensor:
                """PyTorch-native forward (used on CPU or when CUDA ops unavailable)."""
                return self._forward_common(hidden_states, conv_state, ssm_state, use_cuda_ops=False, **kwargs)

            def forward_cuda(
                self,
                hidden_states: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                ssm_state: Optional[torch.Tensor] = None,
                **kwargs,
            ) -> torch.Tensor:
                """CUDA forward with V1 state management integration."""
                return self._forward_common(hidden_states, conv_state, ssm_state, use_cuda_ops=True, **kwargs)

            def _forward_common(
                self,
                hidden_states: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                ssm_state: Optional[torch.Tensor] = None,
                use_cuda_ops: bool = True,
                **kwargs,
            ) -> torch.Tensor:
                """Common forward implementation with V1 state management.

                This is the key integration point with vLLM V1:
                1. Gets forward_context from vLLM
                2. Retrieves attn_metadata[self.prefix] for this layer
                3. Gets state from self.kv_cache (bound by vLLM)
                4. Uses state_indices for proper batched state updates
                """
                state_indices = None
                query_start_loc = None
                device = hidden_states.device
                dtype = hidden_states.dtype

                # Determine batch size
                batch_size = 1 if hidden_states.dim() == 2 else hidden_states.shape[0]

                # =============================================================
                # V1 STATE RETRIEVAL (Critical for proper state management!)
                # =============================================================
                if self._is_v1 and get_forward_context is not None:
                    try:
                        forward_context = get_forward_context()
                        if forward_context is not None:
                            fc_attn_metadata = forward_context.attn_metadata

                            # V1 profile run - attn_metadata is None
                            if fc_attn_metadata is None:
                                if not hasattr(self, '_profile_logged'):
                                    logger.info(f"[V1] Layer {self.layer_idx}: Profile run (attn_metadata=None)")
                                    self._profile_logged = True
                                # Return dummy output for profile run
                                return self.out_proj(hidden_states[..., :self.d_inner])

                            # V1 attn_metadata is a dict keyed by layer prefix
                            if isinstance(fc_attn_metadata, dict):
                                if self.prefix in fc_attn_metadata:
                                    layer_metadata = fc_attn_metadata[self.prefix]
                                    state_indices = getattr(layer_metadata, 'state_indices_tensor', None)
                                    query_start_loc = getattr(layer_metadata, 'query_start_loc', None)

                                    # Get state from kv_cache (bound by vLLM)
                                    virtual_engine = getattr(forward_context, 'virtual_engine', 0)
                                    if self.kv_cache and len(self.kv_cache) > virtual_engine:
                                        kv = self.kv_cache[virtual_engine]
                                        if isinstance(kv, (list, tuple)) and len(kv) >= 2:
                                            if kv[0].numel() > 0:
                                                conv_state = kv[0]
                                                ssm_state = kv[1]

                                    if not hasattr(self, '_v1_state_logged'):
                                        logger.info(f"[V1] Layer {self.layer_idx}: Got state from kv_cache, "
                                                   f"state_indices={'present' if state_indices is not None else 'None'}")
                                        self._v1_state_logged = True
                                else:
                                    if not hasattr(self, '_v1_prefix_warn'):
                                        logger.warning(f"[V1] prefix '{self.prefix}' not in attn_metadata. "
                                                      f"Keys: {list(fc_attn_metadata.keys())[:5]}")
                                        self._v1_prefix_warn = True
                    except Exception as e:
                        if not hasattr(self, '_v1_error_logged'):
                            logger.warning(f"[V1] Error getting forward context: {e}")
                            self._v1_error_logged = True

                # =============================================================
                # FALLBACK STATE (when V1 doesn't provide state)
                # =============================================================
                if conv_state is None or ssm_state is None:
                    self._ensure_cache(batch_size, device, dtype)
                    conv_state = conv_state if conv_state is not None else self._conv_state
                    ssm_state = ssm_state if ssm_state is not None else self._ssm_state

                    # Reset state on prefill (seqlen > 1) to avoid warmup contamination
                    # Each new sequence should start fresh, not inherit from previous requests
                    seqlen = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
                    if seqlen > 1:
                        conv_state.zero_()
                        ssm_state.zero_()

                # =============================================================
                # CONV STATE DEBUG (D-1 investigation)
                # =============================================================
                if self.layer_idx == 0:
                    seqlen_dbg = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
                    if not hasattr(self, '_conv_debug_count'):
                        self._conv_debug_count = 0
                    if self._conv_debug_count < 10:
                        is_fallback = conv_state is self._conv_state
                        logger.info(
                            f"[CONV DEBUG L0] step={self._conv_debug_count} seqlen={seqlen_dbg} "
                            f"conv_state_id={id(conv_state)} is_fallback={is_fallback} "
                            f"conv_norm={conv_state.norm().item():.6f} "
                            f"conv_state[0,:3]={conv_state[0, :3].tolist() if conv_state.dim() >= 2 else 'N/A'}"
                        )
                        self._conv_debug_count += 1

                # =============================================================
                # PROJECTION AND EXPANSION
                # =============================================================
                zxbcdt = self.in_proj(hidden_states)
                z, x, B, C, dt = torch.split(
                    zxbcdt,
                    [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
                    dim=-1,
                )

                # Debug: log split component stats (layer 0 only, first call)
                if self.layer_idx == 0 and not hasattr(self, '_split_debug'):
                    logger.info(f"[SPLIT DEBUG] z: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
                    logger.info(f"[SPLIT DEBUG] x: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                    logger.info(f"[SPLIT DEBUG] B: mean={B.mean().item():.4f}, std={B.std().item():.4f}")
                    logger.info(f"[SPLIT DEBUG] C: mean={C.mean().item():.4f}, std={C.std().item():.4f}")
                    logger.info(f"[SPLIT DEBUG] dt: mean={dt.mean().item():.4f}, std={dt.std().item():.4f}")
                    self._split_debug = True

                # Delta time projection WITH bias (model trained with double bias)
                dt = self.dt_proj(dt)  # W @ dt + bias

                # Expand x via repeat_interleave if needed
                if self.repeat_kv_before_conv:
                    x = rearrange(x, "... (g d) -> ... g d", g=self.num_xb_head)
                    x = torch.repeat_interleave(x, self.repeat_group, dim=-2)
                    x = rearrange(x, "... g d -> ... (g d)")

                # Expand B via repeat_interleave
                B = rearrange(B, "... (g d) -> ... g d", d=self.d_state)
                B = torch.repeat_interleave(B, self.repeat_group, dim=-2)

                # C is already d_inner, just reshape
                C = rearrange(C, "... (g d) -> ... g d", d=self.d_state)

                # =============================================================
                # COMPUTATION PATH SELECTION
                # =============================================================
                # Use vLLM native ops only if: CUDA, ops available, state available, state_indices present
                use_vllm = (use_cuda_ops and _mamba_ops_available and
                           conv_state is not None and ssm_state is not None and
                           state_indices is not None)

                if not hasattr(self, '_path_logged'):
                    logger.info(f"[V1] Layer {self.layer_idx}: use_vllm={use_vllm}, "
                               f"state_indices={'present' if state_indices is not None else 'None'}")
                    self._path_logged = True

                if use_vllm:
                    return self._forward_with_vllm_ops(
                        x, z, B, C, dt, conv_state, ssm_state, state_indices, query_start_loc
                    )
                else:
                    return self._forward_pytorch(x, z, B, C, dt, conv_state, ssm_state)

            def _forward_with_vllm_ops(
                self,
                x: torch.Tensor,
                z: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor,
                dt: torch.Tensor,
                conv_state: torch.Tensor,
                ssm_state: torch.Tensor,
                state_indices: torch.Tensor,
                query_start_loc: Optional[torch.Tensor],
            ) -> torch.Tensor:
                """Forward using vLLM's native Triton ops with V1 state management."""
                # x: (tokens, d_inner), z: (tokens, d_inner)
                # B, C: (tokens, num_heads, d_state), dt: (tokens, d_inner)

                # Transpose states from storage format to computation format
                # Storage: (batch, d_conv-1, conv_dim) -> Compute: (batch, conv_dim, d_conv-1)
                # Storage: (batch, d_state, d_inner) -> Compute: (batch, d_inner, d_state)
                conv_state_t = conv_state.transpose(-1, -2).contiguous()
                ssm_state_t = ssm_state.transpose(-1, -2).contiguous()

                seqlen = x.shape[0] if x.dim() == 2 else x.shape[1]
                is_decode = seqlen == 1

                # Get conv weight in correct format (d_inner, d_conv)
                conv_weight = rearrange(self.conv1d.weight, "d 1 w -> d w")

                # Construct query_start_loc if not provided
                if query_start_loc is None:
                    if x.dim() == 2:
                        # (total_tokens, dim) - treat as single sequence
                        total_tokens = x.shape[0]
                        query_start_loc = torch.tensor([0, total_tokens], dtype=torch.int32, device=x.device)
                    else:
                        # (batch, seq, dim) - construct from batch
                        batch_size = x.shape[0]
                        seq_len = x.shape[1]
                        # All sequences have same length in this case
                        query_start_loc = torch.arange(0, (batch_size + 1) * seq_len, seq_len,
                                                       dtype=torch.int32, device=x.device)

                if is_decode:
                    # Decode path - single token update
                    x_t = x.squeeze(0) if x.dim() == 3 else x
                    if x_t.dim() == 1:
                        x_t = x_t.unsqueeze(0)

                    x_conv = causal_conv1d_update(
                        x_t.transpose(0, 1),
                        conv_state_t,
                        conv_weight,
                        bias=self.conv1d.bias,
                        activation="silu",
                        conv_state_indices=state_indices,
                    )
                    x_conv = x_conv.transpose(0, 1)

                    # SSM state update
                    dt_s = dt.squeeze(0) if dt.dim() == 3 else dt
                    if dt_s.dim() == 1:
                        dt_s = dt_s.unsqueeze(0)
                    B_s = B.squeeze(0) if B.dim() == 3 else B
                    if B_s.dim() == 2:
                        B_s = B_s.unsqueeze(0)
                    C_s = C.squeeze(0) if C.dim() == 3 else C
                    if C_s.dim() == 2:
                        C_s = C_s.unsqueeze(0)
                    z_s = z.squeeze(0) if z.dim() == 3 else z
                    if z_s.dim() == 1:
                        z_s = z_s.unsqueeze(0)

                    y = selective_state_update(
                        ssm_state_t,
                        x_conv,
                        dt_s,
                        self.A,
                        B_s,
                        C_s,
                        D=self.D,
                        z=z_s,
                        dt_bias=self.dt_proj.bias,
                        dt_softplus=True,
                        state_batch_indices=state_indices,
                    )
                else:
                    # Prefill path - full sequence
                    orig_shape = x.shape
                    if x.dim() == 2:
                        x_t = x.transpose(0, 1)
                    else:
                        x_t = rearrange(x, "b t d -> d (b t)")

                    x_conv = causal_conv1d_fn(
                        x_t,
                        conv_weight,
                        self.conv1d.bias,
                        conv_state_t,
                        query_start_loc,
                        cache_indices=state_indices,
                        activation="silu",
                    )

                    # Prepare for selective_scan
                    dt_t = dt.transpose(0, 1) if dt.dim() == 2 else rearrange(dt, "b t d -> d (b t)")
                    B_t = rearrange(B, "t h d -> h d t") if B.dim() == 3 else rearrange(B, "b t h d -> h d (b t)")
                    C_t = rearrange(C, "t h d -> h d t") if C.dim() == 3 else rearrange(C, "b t h d -> h d (b t)")
                    z_t = z.transpose(0, 1) if z.dim() == 2 else rearrange(z, "b t d -> d (b t)")

                    y = selective_scan_fn(
                        x_conv,
                        ssm_state_t,
                        dt_t,
                        self.A,
                        B_t,
                        C_t,
                        D=self.D,
                        z=z_t,
                        delta_bias=self.dt_proj.bias,
                        delta_softplus=True,
                        query_start_loc=query_start_loc,
                        cache_indices=state_indices,
                    )

                    # Reshape back
                    if len(orig_shape) == 2:
                        y = y.transpose(0, 1)
                    else:
                        y = rearrange(y, "d (b t) -> b t d", b=orig_shape[0])

                return self.out_proj(y)

            def _forward_pytorch(
                self,
                x: torch.Tensor,
                z: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor,
                dt: torch.Tensor,
                conv_state: Optional[torch.Tensor],
                ssm_state: Optional[torch.Tensor],
            ) -> torch.Tensor:
                """Fallback PyTorch implementation for non-CUDA or profile runs."""
                orig_shape = x.shape
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                    z = z.unsqueeze(0)
                    B = B.unsqueeze(0)
                    C = C.unsqueeze(0)
                    dt = dt.unsqueeze(0)

                batch, seqlen, _ = x.shape

                # Transpose for conv: (batch, seq, dim) -> (batch, dim, seq)
                x = rearrange(x, "b l d -> b d l")
                z = rearrange(z, "b l d -> b d l")

                # Handle conv_state for causal convolution
                # conv_state stored as (batch, d_conv-1, conv_dim), transpose to (batch, conv_dim, d_conv-1)
                if conv_state is not None and conv_state.numel() > 0:
                    conv_state_t = conv_state.transpose(1, 2)  # (batch, conv_dim, d_conv-1)
                    # Prepend conv_state to x for proper causal context
                    x_with_state = torch.cat([conv_state_t, x], dim=-1)  # (batch, conv_dim, d_conv-1+seqlen)
                    # Apply conv (no padding needed since we have the state)
                    x_conv = self.conv1d.weight.squeeze(1)  # (conv_dim, d_conv)
                    x_out = F.conv1d(x_with_state, x_conv.unsqueeze(1), self.conv1d.bias, groups=self.conv_dim)
                    x = F.silu(x_out)  # (batch, conv_dim, seqlen)
                    # Update conv_state with last d_conv-1 inputs
                    # Take from x_with_state (which has the full history)
                    new_conv_state = x_with_state[:, :, -(self.d_conv - 1):].transpose(1, 2)

                    # D-1 debug: log conv state update (layer 0, first 10 steps)
                    if self.layer_idx == 0 and hasattr(self, '_conv_debug_count') and self._conv_debug_count <= 10:
                        logger.info(
                            f"[CONV UPDATE L0] seqlen={seqlen} "
                            f"old_conv_norm={conv_state.norm().item():.6f} "
                            f"new_conv_norm={new_conv_state.norm().item():.6f} "
                            f"x_input_norm={x_with_state[:,:,-(self.d_conv-1):].norm().item():.6f} "
                            f"x_conv_out_norm={x.norm().item():.6f} "
                            f"window_vals=[{', '.join(f'{v:.4f}' for v in x_with_state[0, 0, :].tolist())}]"
                        )

                    conv_state.copy_(new_conv_state)
                else:
                    # No state available, use regular conv with padding
                    x = F.silu(self.conv1d(x)[..., :seqlen])

                # Apply softplus to dt with double bias (model was trained this way)
                # dt = W @ dt + bias (from dt_proj), now add bias again and softplus
                dt = rearrange(dt, "b l d -> b d l")
                dt = F.softplus(dt + self.dt_proj.bias.to(dt.dtype).unsqueeze(0).unsqueeze(-1))

                # SSM scan setup
                # Cast A to same dtype as input to avoid float32/bfloat16 mismatch
                A = self.A.to(x.dtype)  # Already -exp(log(A))
                # dA = exp(dt * A) with shape (batch, d_inner, seqlen, d_state)
                dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))

                # Debug: log SSM parameters (layer 0 only, first call)
                if self.layer_idx == 0 and not hasattr(self, '_ssm_debug'):
                    logger.info(f"[SSM DEBUG] A: min={A.min().item():.4f}, max={A.max().item():.4f}, mean={A.mean().item():.4f}")
                    logger.info(f"[SSM DEBUG] dt: min={dt.min().item():.4f}, max={dt.max().item():.4f}, mean={dt.mean().item():.4f}")
                    logger.info(f"[SSM DEBUG] dA: min={dA.min().item():.4f}, max={dA.max().item():.4f}, mean={dA.mean().item():.4f}")
                    self._ssm_debug = True

                # Reshape for grouped scan
                x_grouped = rearrange(x, "b (h d) l -> b h d l", h=self.num_heads)
                dA_grouped = rearrange(dA, "b (h d) l n -> b h d l n", h=self.num_heads)
                B_t = rearrange(B, "b l h n -> b h n l")
                C_t = rearrange(C, "b l h n -> b h n l")

                head_dim = self.d_inner // self.num_heads
                dt_grouped = rearrange(dt, "b (h d) l -> b h d l", h=self.num_heads)
                dB_u = dt_grouped.unsqueeze(3) * B_t.unsqueeze(2) * x_grouped.unsqueeze(3)

                # Initialize state from ssm_state if available
                # ssm_state stored as (batch, d_state, d_inner), need (batch, num_heads, head_dim, d_state)
                if ssm_state is not None and ssm_state.numel() > 0:
                    # Transpose: (batch, d_state, d_inner) -> (batch, d_inner, d_state)
                    ssm_state_t = ssm_state.transpose(1, 2)
                    # Reshape: (batch, d_inner, d_state) -> (batch, num_heads, head_dim, d_state)
                    state = rearrange(ssm_state_t, "b (h d) n -> b h d n", h=self.num_heads).to(x.dtype)
                else:
                    state = torch.zeros(batch, self.num_heads, head_dim, self.d_state,
                                       device=x.device, dtype=x.dtype)

                # Debug: log state before scan (layer 0 only, first few calls)
                if self.layer_idx == 0 and not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                if self.layer_idx == 0 and self._debug_count < 5:
                    logger.info(f"[DEBUG L0] seqlen={seqlen}, state_before_norm={state.norm().item():.4f}, "
                               f"x_norm={x.norm().item():.4f}")

                # Sequential SSM scan
                outputs = []
                for t in range(seqlen):
                    state = dA_grouped[:, :, :, t, :] * state + dB_u[:, :, :, :, t]
                    y_t = torch.einsum("bhdn,bhn->bhd", state, C_t[:, :, :, t])
                    outputs.append(y_t)

                # Debug: log state after scan (layer 0 only)
                if self.layer_idx == 0 and self._debug_count < 5:
                    logger.info(f"[DEBUG L0] state_after_norm={state.norm().item():.4f}, "
                               f"y_norm={torch.stack(outputs, dim=-1).norm().item():.4f}")
                    self._debug_count += 1

                # Update ssm_state with final state
                if ssm_state is not None and ssm_state.numel() > 0:
                    # Reshape back: (batch, num_heads, head_dim, d_state) -> (batch, d_inner, d_state)
                    final_state = rearrange(state, "b h d n -> b (h d) n")
                    # Transpose back: (batch, d_inner, d_state) -> (batch, d_state, d_inner)
                    ssm_state.copy_(final_state.transpose(1, 2))

                y = torch.stack(outputs, dim=-1)
                y = rearrange(y, "b h d l -> b (h d) l")

                # Skip connection and gate
                # Cast D to input dtype to avoid float32/bfloat16 mismatch
                y = y + self.D.to(x.dtype).unsqueeze(0).unsqueeze(-1) * x
                y = y * F.silu(z)

                y = rearrange(y, "b d l -> b l d")

                if orig_shape[0] != batch or (len(orig_shape) == 2):
                    y = y.squeeze(0)

                return self.out_proj(y)

        return MambaInLlamaMambaMixerVLLM

    else:
        # Fallback when vLLM components not available
        class MambaInLlamaMambaMixerFallback(nn.Module):
            """Fallback MambaInLlama mixer when vLLM not available."""

            def __init__(
                self,
                config: MambaInLlamaMambaConfig,
                layer_idx: int,
                prefix: str = "",
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
                conv_state_shape = (self.conv_dim, self.d_conv - 1)
                ssm_state_shape = (self.d_inner, self.d_state)
                return (conv_state_shape, ssm_state_shape)

            def get_state_dtype(self):
                dtype = self.out_proj.weight.dtype
                return (dtype, dtype)

            @property
            def mamba_type(self):
                return "mamba1"

            def forward(self, hidden_states, **kwargs):
                # Simple forward for non-vLLM use
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
                return self.out_proj(y).squeeze(0)

        return MambaInLlamaMambaMixerFallback


# Create the class using the factory
MambaInLlamaMambaMixer = _create_mamba_mixer_class()


# =============================================================================
# MLP LAYER (placeholder - will be replaced when we find the actual content)
# =============================================================================

# Remove orphaned code marker - this helps identify what to delete
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
    """Multi-Head Attention decoder layer using vLLM's Attention for KV caching."""

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

        # Q/K/V projections (separate — matches existing weight loading)
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
        logger.info(f"[MHA L{self.layer_idx}] enter: hidden={hidden_states.shape} pos={positions.shape}")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project to Q, K, V (stay 2D)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        logger.info(f"[MHA L{self.layer_idx}] proj: q={q.shape} k={k.shape} v={v.shape}")

        # Apply RoPE (vLLM handles position encoding)
        q, k = self.rotary_emb(positions, q, k)
        logger.info(f"[MHA L{self.layer_idx}] rope done: q={q.shape} k={k.shape}")

        # vLLM Attention — KV cache, paging, GQA, masking all handled internally
        logger.info(f"[MHA L{self.layer_idx}] calling self.attn...")
        attn_output = self.attn(q, k, v)
        logger.info(f"[MHA L{self.layer_idx}] attn done: out={attn_output.shape}")

        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        logger.info(f"[MHA L{self.layer_idx}] exit: out={hidden_states.shape}")
        return hidden_states


# =============================================================================
# MAMBA DECODER LAYER
# =============================================================================

class MambaDecoderLayer(nn.Module):
    """Mamba SSM decoder layer."""

    def __init__(self, config: MambaInLlamaMambaConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.prefix = prefix

        # Pass prefix to mixer for static_forward_context registration
        mamba_prefix = f"{prefix}.mamba" if prefix else f"model.layers.{layer_idx}.mamba"
        self.mamba = MambaInLlamaMambaMixer(config, layer_idx, prefix=mamba_prefix)
        self.mlp = MLP(config.d_model, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
        cache_position: int = 0,
        attn_metadata=None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            hidden_states,
            conv_state=conv_state,
            ssm_state=ssm_state,
            cache_position=cache_position,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

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
                 cache_config: "CacheConfig" = None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

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
                self.layers.append(MambaDecoderLayer(config, layer_idx, prefix=layer_prefix))

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
                hidden_states = layer(
                    hidden_states,
                    attn_metadata=attn_metadata,
                )
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
    is_attention_free: ClassVar[Literal[False]] = False

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

        # Extract cache_config from vllm_config for attention layers
        cache_config = None
        if vllm_config is not None and hasattr(vllm_config, 'cache_config'):
            cache_config = vllm_config.cache_config

        # Pass prefix and cache_config to model backbone
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = MambaInLlamaMambaModel(config, prefix=model_prefix, cache_config=cache_config)
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

    # =========================================================================
    # CUDA Graph Compatibility Methods (for vLLM V1 engine)
    # =========================================================================

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """Copy inputs before CUDA graph capture.

        This method is called by vLLM's V1 engine to prepare inputs for
        CUDA graph execution. For Mamba models, this involves copying
        state buffers to ensure they persist across graph executions.

        Args:
            input_buffers: Dict of input buffers from vLLM
            **kwargs: Additional arguments

        Returns:
            Dict of buffers to use during CUDA graph execution
        """
        # For now, pass through - state is managed via self.kv_cache in MambaMixer
        # which is populated by vLLM's infrastructure
        return input_buffers

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """Get inputs for sequence-length agnostic CUDA graph capture.

        Returns inputs that can be used across different sequence lengths
        for efficient CUDA graph reuse.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Dict of capture inputs for CUDA graphs
        """
        # Return empty dict - state tensors are managed by vLLM via MambaBase.kv_cache
        return {}

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from checkpoint.

        Handles weight name transformations:
        1. mha.in_proj.weight -> split into q_proj, k_proj, v_proj
        2. mha.out_proj.weight -> rename to o_proj.weight
        """
        params_dict = dict(self.named_parameters())
        loaded_count = 0
        skipped_weights = []

        # Log model parameter names for debugging (first 10)
        param_names = list(params_dict.keys())
        logger.info(f"Model has {len(param_names)} parameters")
        logger.info(f"First 20 model params: {param_names[:20]}")

        # Get dimensions for attention splitting
        # Q: num_heads * head_dim, K/V: num_kv_heads * head_dim
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads or num_heads
        head_dim = self.config.hidden_size // num_heads
        q_dim = num_heads * head_dim  # 4096 for 32 heads * 128 dim
        kv_dim = num_kv_heads * head_dim  # 1024 for 8 heads * 128 dim

        logger.info(f"Attention dims: q_dim={q_dim}, kv_dim={kv_dim}, head_dim={head_dim}")

        checkpoint_names = []
        for name, loaded_weight in weights:
            checkpoint_names.append(name)
            # Log first 20 checkpoint weight names
            if len(checkpoint_names) <= 20:
                logger.info(f"Checkpoint weight: {name} shape={loaded_weight.shape}")

            # Handle fused attention in_proj -> split into q, k, v
            if ".mha.in_proj.weight" in name:
                # Split fused QKV weight: [q_dim + kv_dim + kv_dim, hidden]
                base_name = name.replace(".mha.in_proj.weight", "")

                q_weight = loaded_weight[:q_dim, :]
                k_weight = loaded_weight[q_dim:q_dim + kv_dim, :]
                v_weight = loaded_weight[q_dim + kv_dim:, :]

                found_any = False
                for suffix, weight in [(".q_proj.weight", q_weight),
                                       (".k_proj.weight", k_weight),
                                       (".v_proj.weight", v_weight)]:
                    param_name = base_name + suffix
                    if param_name in params_dict:
                        params_dict[param_name].data.copy_(weight)
                        loaded_count += 1
                        found_any = True
                    else:
                        skipped_weights.append(f"{name} -> {param_name} (param not found - is attn_layers configured?)")
                if not found_any:
                    logger.warning(f"Attention layer weights {name} found but no q/k/v params exist. Check attn_layers config!")
                continue

            # Handle attention out_proj rename
            if ".mha.out_proj.weight" in name:
                new_name = name.replace(".mha.out_proj.", ".o_proj.")
                if new_name in params_dict:
                    params_dict[new_name].data.copy_(loaded_weight)
                    loaded_count += 1
                continue

            # Handle attention out_proj bias if present
            if ".mha.out_proj.bias" in name:
                new_name = name.replace(".mha.out_proj.", ".o_proj.")
                if new_name in params_dict:
                    params_dict[new_name].data.copy_(loaded_weight)
                    loaded_count += 1
                continue

            # Handle A_log -> A conversion for Mamba layers
            # Checkpoint stores A_log, we need A = -exp(A_log) as per Mamba paper
            if ".mamba.A_log" in name:
                new_name = name.replace(".mamba.A_log", ".mamba.A")
                if new_name in params_dict:
                    param = params_dict[new_name]
                    # A = -exp(A_log) as per Mamba paper
                    converted = -torch.exp(loaded_weight)
                    if param.shape == converted.shape:
                        param.data.copy_(converted)
                        loaded_count += 1
                        continue
                    else:
                        skipped_weights.append(f"{name} -> {new_name} (shape mismatch: {converted.shape} vs {param.shape})")
                        continue

            # Try direct match
            if name in params_dict:
                param = params_dict[name]
                if param.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                    loaded_count += 1
                    continue

            # Try with/without model prefix
            candidates = [name]
            if name.startswith("model."):
                candidates.append(name[6:])
            else:
                candidates.append(f"model.{name}")

            matched = False
            for candidate in candidates:
                if candidate in params_dict:
                    param = params_dict[candidate]
                    if param.shape == loaded_weight.shape:
                        param.data.copy_(loaded_weight)
                        loaded_count += 1
                        matched = True
                        break
                    else:
                        skipped_weights.append(f"{name} (shape mismatch: checkpoint {loaded_weight.shape} vs model {param.shape})")
                        matched = True  # Don't add to skipped again
                        break

            if not matched:
                skipped_weights.append(f"{name} (no matching param)")

        logger.info(f"Loaded {loaded_count}/{len(params_dict)} parameters from {len(checkpoint_names)} checkpoint weights")
        if skipped_weights:
            logger.info(f"Skipped {len(skipped_weights)} checkpoint weights. First 20:")
            for w in skipped_weights[:20]:
                logger.info(f"  - {w}")

        # Log model params that weren't loaded (helps diagnose attn_layers issues)
        if loaded_count < len(params_dict):
            # Track which params were loaded by checking if they're still at init values
            # This is approximate - better to track explicitly
            missing_params = len(params_dict) - loaded_count
            logger.warning(f"{missing_params} model parameters may not have been loaded from checkpoint!")
            logger.info(f"Config attn_layers: {self.config.attn_layers}")
            # Show some attention-related params to help debug
            attn_params = [p for p in param_names if 'q_proj' in p or 'k_proj' in p or 'v_proj' in p or 'o_proj' in p]
            if attn_params:
                logger.info(f"Model has {len(attn_params)} attention params: {attn_params[:8]}...")

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
