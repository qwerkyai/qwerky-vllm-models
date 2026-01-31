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
    from vllm.attention import Attention, AttentionMetadata
    from vllm.attention.backends.abstract import AttentionType
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
# MAMBAINLLAMA MAMBA MIXER (Native vLLM ops with MambaBase integration)
# =============================================================================

def _create_mamba_mixer_class():
    """Factory function to create MambaInLlamaMambaMixer.

    Note: We intentionally do NOT inherit from MambaBase because:
    1. MambaBase inherits from AttentionLayerBase which breaks nn.Module callability
    2. MambaBase.__init__ doesn't properly initialize nn.Module parameters

    Instead, we use a plain nn.Module with manual state management.
    State is allocated by the model and passed to forward(), or managed internally.
    """

    # Always use plain nn.Module for proper callability and parameter registration
    if True:  # Was: if _MambaBase is not None
        class MambaInLlamaMambaMixerVLLM(nn.Module):
            """MambaInLlama Mamba mixer using vLLM native ops.

            Uses plain nn.Module for proper parameter registration and callability.
            State is managed internally via self._conv_state and self._ssm_state.

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

                # Conv1d - depthwise convolution (store weight in transposed form for vLLM ops)
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

                # Register in static_forward_context for vLLM state binding
                if get_current_vllm_config is not None and prefix:
                    try:
                        compilation_config = get_current_vllm_config().compilation_config
                        if prefix in compilation_config.static_forward_context:
                            raise ValueError(f"Duplicate layer name: {prefix}")
                        compilation_config.static_forward_context[prefix] = self
                        logger.info(f"Registered MambaInLlamaMambaMixer {prefix} in static_forward_context")
                    except Exception as e:
                        logger.warning(f"Could not register in static_forward_context: {e}")

                # Internal state caches (allocated on first forward if not provided)
                self._conv_state: Optional[torch.Tensor] = None
                self._ssm_state: Optional[torch.Tensor] = None
                self._max_batch_size = 0

                # Placeholder for vLLM-managed state (may be populated by bind_kv_cache)
                self.kv_cache: tuple[torch.Tensor, ...] = (torch.tensor([]), torch.tensor([]))

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
            # Forward method (standard nn.Module interface)
            # =================================================================

            def forward(
                self,
                hidden_states: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                ssm_state: Optional[torch.Tensor] = None,
                cache_position: int = 0,
                attn_metadata=None,
                **kwargs,
            ) -> torch.Tensor:
                """Forward pass with state management.

                In vLLM V1 mode, state is retrieved via get_forward_context():
                - attn_metadata is a dict keyed by layer prefix
                - kv_cache is indexed by virtual_engine
                - state_indices comes from layer-specific Mamba1AttentionMetadata

                Fallback for non-V1 mode or direct calls uses internal caches.
                """
                state_indices = None
                query_start_loc = None
                device = hidden_states.device
                dtype = hidden_states.dtype

                # Determine batch size
                if hidden_states.dim() == 2:
                    batch_size = 1
                else:
                    batch_size = hidden_states.shape[0]

                # =============================================================
                # V1 MODE: Use get_forward_context() to retrieve state
                # This is how vLLM's native MambaMixer works in V1
                # =============================================================
                if get_forward_context is not None:
                    try:
                        forward_context = get_forward_context()
                        if forward_context is not None:
                            fc_attn_metadata = forward_context.attn_metadata

                            # Debug log (once per layer)
                            if not hasattr(self, '_v1_debug_logged'):
                                logger.warning(f"[DEBUG V1] prefix={self.prefix}, "
                                               f"forward_context available, "
                                               f"attn_metadata type={type(fc_attn_metadata).__name__ if fc_attn_metadata else None}, "
                                               f"is_dict={isinstance(fc_attn_metadata, dict)}")
                                self._v1_debug_logged = True

                            # In V1, attn_metadata is a dict keyed by layer prefix
                            if fc_attn_metadata is not None and isinstance(fc_attn_metadata, dict):
                                if self.prefix in fc_attn_metadata:
                                    layer_metadata = fc_attn_metadata[self.prefix]

                                    # Get state indices and query_start_loc from layer metadata
                                    state_indices = getattr(layer_metadata, 'state_indices_tensor', None)
                                    query_start_loc = getattr(layer_metadata, 'query_start_loc', None)

                                    if not hasattr(self, '_v1_meta_debug_logged'):
                                        logger.warning(f"[DEBUG V1] layer_metadata type={type(layer_metadata).__name__}, "
                                                       f"state_indices={state_indices is not None}, "
                                                       f"query_start_loc={query_start_loc is not None}")
                                        self._v1_meta_debug_logged = True

                                    # Get kv_cache from self.kv_cache indexed by virtual_engine
                                    virtual_engine = getattr(forward_context, 'virtual_engine', 0)
                                    if hasattr(self, 'kv_cache') and self.kv_cache is not None:
                                        # kv_cache may be tuple (conv_state, ssm_state) or indexed by virtual_engine
                                        if isinstance(self.kv_cache, (list, tuple)):
                                            if len(self.kv_cache) > virtual_engine and isinstance(self.kv_cache[virtual_engine], (list, tuple)):
                                                # kv_cache[virtual_engine] = (conv_state, ssm_state)
                                                kv = self.kv_cache[virtual_engine]
                                                if len(kv) >= 2 and kv[0].numel() > 0:
                                                    conv_state = kv[0]
                                                    ssm_state = kv[1]
                                            elif len(self.kv_cache) >= 2:
                                                # Direct tuple (conv_state, ssm_state)
                                                if self.kv_cache[0].numel() > 0:
                                                    conv_state = self.kv_cache[0]
                                                    ssm_state = self.kv_cache[1]

                                        if not hasattr(self, '_v1_kv_debug_logged'):
                                            logger.warning(f"[DEBUG V1] kv_cache type={type(self.kv_cache)}, "
                                                           f"conv_state={conv_state.shape if conv_state is not None and hasattr(conv_state, 'shape') else None}, "
                                                           f"ssm_state={ssm_state.shape if ssm_state is not None and hasattr(ssm_state, 'shape') else None}")
                                            self._v1_kv_debug_logged = True
                                else:
                                    # Prefix not in dict - log available keys
                                    if not hasattr(self, '_v1_prefix_debug_logged'):
                                        logger.warning(f"[DEBUG V1] prefix '{self.prefix}' not in attn_metadata dict, "
                                                       f"available keys (first 5): {list(fc_attn_metadata.keys())[:5]}")
                                        self._v1_prefix_debug_logged = True

                    except Exception as e:
                        if not hasattr(self, '_v1_error_logged'):
                            logger.warning(f"[DEBUG V1] get_forward_context() error: {e}")
                            self._v1_error_logged = True

                # =============================================================
                # FALLBACK: Try legacy approaches if V1 didn't provide state
                # =============================================================
                if conv_state is None or ssm_state is None:
                    # Try to get from kv_cache directly (may be bound by vLLM)
                    if hasattr(self, 'kv_cache') and self.kv_cache is not None:
                        if isinstance(self.kv_cache, (list, tuple)) and len(self.kv_cache) >= 2:
                            kv0, kv1 = self.kv_cache[0], self.kv_cache[1]
                            if hasattr(kv0, 'numel') and kv0.numel() > 0 and conv_state is None:
                                conv_state = kv0
                            if hasattr(kv1, 'numel') and kv1.numel() > 0 and ssm_state is None:
                                ssm_state = kv1

                # Fall back to internal caches
                if conv_state is None or ssm_state is None:
                    self._ensure_cache(batch_size, device, dtype)
                    if conv_state is None:
                        conv_state = self._conv_state
                    if ssm_state is None:
                        ssm_state = self._ssm_state

                # Try to get state_indices from attn_metadata argument (legacy V0 path)
                if state_indices is None and attn_metadata is not None:
                    mamba_meta = getattr(attn_metadata, 'mamba_metadata', None)
                    if mamba_meta is not None:
                        state_indices = getattr(mamba_meta, 'state_indices_tensor', None)
                    if state_indices is None:
                        state_indices = getattr(attn_metadata, 'state_indices_tensor', None)

                # Run the actual computation
                return self._forward_impl(hidden_states, conv_state, ssm_state, state_indices, query_start_loc, attn_metadata)

            def _forward_impl(
                self,
                hidden_states: torch.Tensor,
                conv_state: Optional[torch.Tensor],
                ssm_state: Optional[torch.Tensor],
                state_indices: Optional[torch.Tensor],
                query_start_loc: Optional[torch.Tensor],
                attn_metadata,
            ) -> torch.Tensor:
                """Actual forward implementation using vLLM native ops."""
                # DEBUG: Print at very start to confirm we're being called
                if not hasattr(self, '_entry_debug'):
                    print(f"[ENTRY DEBUG] MambaInLlamaMambaMixerVLLM.forward called! layer={self.layer_idx}, attn_metadata={attn_metadata is not None}", flush=True)
                    self._entry_debug = True

                # Early return for V1 profile/warmup runs (like vLLM's native MambaMixer)
                # When attn_metadata is None, skip SSM computation entirely
                if attn_metadata is None:
                    # Just do a simple projection for shape/memory profiling
                    # This matches vLLM's native approach: out_proj(in_proj(x)[...])
                    projected = self.in_proj(hidden_states)
                    # Take the z portion (first d_inner) and pass through out_proj
                    z_dummy = projected[..., :self.d_inner]
                    return self.out_proj(z_dummy)

                batch_or_tokens = hidden_states.shape[0]

                # Fused projection: [z, x, B, C, dt]
                zxbcdt = self.in_proj(hidden_states)
                z, x, B, C, dt = torch.split(
                    zxbcdt,
                    [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
                    dim=-1,
                )

                # Debug: log split statistics on first forward (helps verify split order)
                if not hasattr(self, '_split_debug_logged'):
                    logger.warning(f"[SPLIT DEBUG] Layer {self.layer_idx}: "
                                   f"z.shape={z.shape}, mean={z.float().mean():.4f}, std={z.float().std():.4f} | "
                                   f"x.shape={x.shape}, mean={x.float().mean():.4f}, std={x.float().std():.4f} | "
                                   f"B.shape={B.shape}, mean={B.float().mean():.4f}, std={B.float().std():.4f} | "
                                   f"C.shape={C.shape}, mean={C.float().mean():.4f}, std={C.float().std():.4f} | "
                                   f"dt.shape={dt.shape}, mean={dt.float().mean():.4f}, std={dt.float().std():.4f}")
                    self._split_debug_logged = True

                # Delta time projection WITH bias (model trained with double bias)
                # First bias application here, second in SSM kernel softplus
                dt = self.dt_proj(dt)  # Full Linear with bias: (tokens, d_inner)

                # Expand x via repeat_interleave if needed
                if self.repeat_kv_before_conv:
                    x = rearrange(x, "... (g d) -> ... g d", g=self.num_xb_head)
                    x = torch.repeat_interleave(x, self.repeat_group, dim=-2)
                    x = rearrange(x, "... g d -> ... (g d)")

                # Expand B via repeat_interleave
                B = rearrange(B, "... (g d) -> ... g d", d=self.d_state)
                B = torch.repeat_interleave(B, self.repeat_group, dim=-2)
                # B now: (..., num_heads, d_state)

                # C is already d_inner, just reshape
                C = rearrange(C, "... (g d) -> ... g d", d=self.d_state)
                # C now: (..., num_heads, d_state)

                # Use vLLM native ops if available AND we have valid state indices
                # During warmup/profiling, state_indices is None and vLLM ops don't support that
                use_vllm = _mamba_ops_available and conv_state is not None and ssm_state is not None and state_indices is not None

                # DEBUG: Log which path we're taking (remove after debugging)
                if not hasattr(self, '_debug_logged'):
                    logger.warning(f"[DEBUG] _mamba_ops_available={_mamba_ops_available}, "
                                   f"conv_state={conv_state is not None}, ssm_state={ssm_state is not None}, "
                                   f"state_indices={state_indices is not None}, use_vllm={use_vllm}")
                    if conv_state is not None:
                        logger.warning(f"[DEBUG] conv_state.shape={conv_state.shape}")
                    if ssm_state is not None:
                        logger.warning(f"[DEBUG] ssm_state.shape={ssm_state.shape}")
                    if state_indices is not None:
                        logger.warning(f"[DEBUG] state_indices.shape={state_indices.shape}, values={state_indices[:10] if len(state_indices) > 0 else 'empty'}")
                    self._debug_logged = True

                if use_vllm:
                    if not hasattr(self, '_path_debug_logged'):
                        logger.warning(f"[PATH DEBUG] Layer {self.layer_idx}: Using vLLM ops path")
                        self._path_debug_logged = True
                    return self._forward_with_vllm_ops(
                        x, z, B, C, dt, conv_state, ssm_state, state_indices, query_start_loc, attn_metadata
                    )
                else:
                    if not hasattr(self, '_path_debug_logged'):
                        logger.warning(f"[PATH DEBUG] Layer {self.layer_idx}: Using PyTorch fallback path")
                        self._path_debug_logged = True
                    # Fallback to pure PyTorch (used during warmup when state_indices is None)
                    return self._forward_pytorch(x, z, B, C, dt, conv_state, ssm_state, state_indices)

            def _forward_with_vllm_ops(
                self,
                x: torch.Tensor,
                z: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor,
                dt: torch.Tensor,
                conv_state: torch.Tensor,
                ssm_state: torch.Tensor,
                state_indices: Optional[torch.Tensor],
                query_start_loc: Optional[torch.Tensor],
                attn_metadata=None,
            ) -> torch.Tensor:
                """Forward using vLLM's native Triton ops."""
                # x: (tokens, d_inner), z: (tokens, d_inner)
                # B, C: (tokens, num_heads, d_state), dt: (tokens, d_inner)

                # Transpose states from storage format to computation format
                # Storage: (batch, d_conv-1, conv_dim) -> Compute: (batch, conv_dim, d_conv-1)
                # Storage: (batch, d_state, d_inner) -> Compute: (batch, d_inner, d_state)
                # This gives stride_dim == 1 as required by causal_conv1d_fn
                conv_state = conv_state.transpose(-1, -2)
                ssm_state = ssm_state.transpose(-1, -2)

                seqlen = x.shape[0] if x.dim() == 2 else x.shape[1]
                is_decode = seqlen == 1

                # Get conv weight in correct format (d_inner, d_conv)
                conv_weight = rearrange(self.conv1d.weight, "d 1 w -> d w")

                # Use provided query_start_loc, or construct from input shape
                if query_start_loc is None:
                    # Try to get from attn_metadata
                    if attn_metadata is not None:
                        if hasattr(attn_metadata, 'query_start_loc'):
                            query_start_loc = attn_metadata.query_start_loc
                        elif hasattr(attn_metadata, 'seq_start_loc'):
                            query_start_loc = attn_metadata.seq_start_loc

                # If still not available, construct from input shape
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

                if is_decode and state_indices is not None:
                    # Decode path - single token update
                    # causal_conv1d_update expects: x (dim, batch), conv_state (slots, dim, width)
                    x_t = x.squeeze(0) if x.dim() == 3 else x  # (batch, d_inner) or (d_inner,)
                    if x_t.dim() == 1:
                        x_t = x_t.unsqueeze(0)  # (1, d_inner)

                    x_conv = causal_conv1d_update(
                        x_t.transpose(0, 1),  # (d_inner, batch)
                        conv_state,
                        conv_weight,
                        bias=self.conv1d.bias,
                        activation="silu",
                        conv_state_indices=state_indices,
                    )
                    x_conv = x_conv.transpose(0, 1)  # (batch, d_inner)

                    # SSM state update
                    # selective_state_update expects specific shapes
                    dt_squeezed = dt.squeeze(0) if dt.dim() == 3 else dt
                    if dt_squeezed.dim() == 1:
                        dt_squeezed = dt_squeezed.unsqueeze(0)

                    B_squeezed = B.squeeze(0) if B.dim() == 3 else B
                    if B_squeezed.dim() == 2:
                        B_squeezed = B_squeezed.unsqueeze(0)

                    C_squeezed = C.squeeze(0) if C.dim() == 3 else C
                    if C_squeezed.dim() == 2:
                        C_squeezed = C_squeezed.unsqueeze(0)

                    z_squeezed = z.squeeze(0) if z.dim() == 3 else z
                    if z_squeezed.dim() == 1:
                        z_squeezed = z_squeezed.unsqueeze(0)

                    y = selective_state_update(
                        ssm_state,
                        x_conv,
                        dt_squeezed,
                        self.A,
                        B_squeezed,
                        C_squeezed,
                        D=self.D,
                        z=z_squeezed,
                        dt_bias=self.dt_proj.bias,
                        dt_softplus=True,
                        state_batch_indices=state_indices,
                        dst_state_batch_indices=state_indices,
                    )
                else:
                    # Prefill path - full sequence
                    # For prefill, use causal_conv1d_fn and selective_scan_fn
                    # vLLM 0.14+ expects x as (dim, total_tokens) for varlen batching
                    total_tokens = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
                    orig_shape = x.shape

                    if x.dim() == 2:
                        x_t = x.transpose(0, 1)  # (tokens, dim) -> (dim, tokens)
                    else:
                        x_t = rearrange(x, "b t d -> d (b t)")  # Flatten batch

                    # Note: We only reach here when state_indices is not None (checked in _forward_impl)
                    x_conv = causal_conv1d_fn(
                        x_t,
                        conv_weight,
                        self.conv1d.bias,
                        conv_state,
                        query_start_loc,
                        cache_indices=state_indices,
                        activation="silu",
                    )
                    # x_conv is (dim, total_tokens)

                    # Prepare dt, B, C, z for selective_scan - need (dim, total_tokens) format
                    if dt.dim() == 2:
                        dt_t = dt.transpose(0, 1)  # (tokens, dim) -> (dim, tokens)
                    else:
                        dt_t = rearrange(dt, "b t d -> d (b t)")

                    # B, C are (tokens, num_heads, d_state) - need to flatten
                    if B.dim() == 3:
                        B_t = rearrange(B, "t h d -> h d t")  # (num_heads, d_state, tokens)
                    else:
                        B_t = rearrange(B, "b t h d -> h d (b t)")
                    if C.dim() == 3:
                        C_t = rearrange(C, "t h d -> h d t")
                    else:
                        C_t = rearrange(C, "b t h d -> h d (b t)")

                    if z.dim() == 2:
                        z_t = z.transpose(0, 1)  # (tokens, dim) -> (dim, tokens)
                    else:
                        z_t = rearrange(z, "b t d -> d (b t)")

                    y = selective_scan_fn(
                        x_conv,
                        ssm_state,
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
                    # y is (dim, total_tokens)

                    # Reshape back to original format
                    if len(orig_shape) == 2:
                        y = y.transpose(0, 1)  # (dim, tokens) -> (tokens, dim)
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
                state_indices: Optional[torch.Tensor],
            ) -> torch.Tensor:
                """Fallback PyTorch implementation."""
                # Simple implementation for when vLLM ops not available
                orig_shape = x.shape
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                    z = z.unsqueeze(0)
                    B = B.unsqueeze(0)
                    C = C.unsqueeze(0)
                    dt = dt.unsqueeze(0)

                batch, seqlen, _ = x.shape

                # Transpose for conv
                x = rearrange(x, "b l d -> b d l")
                z = rearrange(z, "b l d -> b d l")

                # Apply conv
                x = F.silu(self.conv1d(x)[..., :seqlen])

                # Apply softplus to dt with bias (double bias as in original)
                dt = rearrange(dt, "b l d -> b d l")
                dt = F.softplus(dt + self.dt_proj.bias.unsqueeze(0).unsqueeze(-1))

                # Simple SSM scan
                A = self.A  # Already -exp(log(A))
                dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1))

                # Reshape for scan
                x_grouped = rearrange(x, "b (h d) l -> b h d l", h=self.num_heads)
                dt_grouped = rearrange(dt, "b (h d) l -> b h d l", h=self.num_heads)
                dA_grouped = rearrange(dA, "b (h d) l n -> b h d l n", h=self.num_heads)
                B_t = rearrange(B, "b l h n -> b h n l")
                C_t = rearrange(C, "b l h n -> b h n l")

                head_dim = self.d_inner // self.num_heads
                dB_u = dt_grouped.unsqueeze(3) * B_t.unsqueeze(2) * x_grouped.unsqueeze(3)

                # Sequential scan
                state = torch.zeros(batch, self.num_heads, head_dim, self.d_state,
                                   device=x.device, dtype=x.dtype)
                outputs = []
                for t in range(seqlen):
                    state = dA_grouped[:, :, :, t, :] * state + dB_u[:, :, :, :, t]
                    y_t = torch.einsum("bhdn,bhn->bhd", state, C_t[:, :, :, t])
                    outputs.append(y_t)

                y = torch.stack(outputs, dim=-1)
                y = rearrange(y, "b h d l -> b (h d) l")

                # Skip connection and gate
                y = y + self.D.unsqueeze(0).unsqueeze(-1) * x
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
                y = y + self.D.unsqueeze(0).unsqueeze(-1) * x

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
    """Multi-Head Attention decoder layer."""

    def __init__(self, config: MambaInLlamaMambaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = None  # Will be initialized on first forward
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)

        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _init_rope(self, device):
        """Initialize rotary embeddings."""
        if self.rotary_emb is None:
            inv_freq = 1.0 / (self.rope_theta ** (
                torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim
            ))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rotary_pos_emb(self, q, k, positions):
        """Apply rotary position embeddings."""
        self._init_rope(q.device)

        # positions: (batch, seq_len)
        seq_len = positions.shape[-1] if positions.dim() > 1 else positions.shape[0]
        positions = positions.view(-1, seq_len)

        # Compute freqs in float32 for precision, then cast to input dtype
        freqs = torch.outer(positions[0].float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(q.dtype)  # (1, 1, seq, head_dim)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(q.dtype)

        # Apply rotation
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        # Handle both 2D [total_tokens, hidden] and 3D [batch, seq, hidden] input
        input_2d = hidden_states.dim() == 2
        logger.info(f"MHA forward: input shape={hidden_states.shape}, input_2d={input_2d}")
        if input_2d:
            hidden_states = hidden_states.unsqueeze(0)  # [1, total_tokens, hidden]
            logger.info(f"MHA forward: after unsqueeze shape={hidden_states.shape}")

        batch_size, seq_len, _ = hidden_states.shape
        logger.info(f"MHA forward: batch_size={batch_size}, seq_len={seq_len}")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        logger.info(f"MHA forward: after proj q={q.shape}, k={k.shape}, v={v.shape}")

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        logger.info(f"MHA forward: after view/transpose q={q.shape}, k={k.shape}, v={v.shape}")

        q, k = self._apply_rotary_pos_emb(q, k, positions)
        logger.info(f"MHA forward: after rotary q={q.shape}, k={k.shape}")

        # KV cache handling
        # k, v shape after transpose: [batch, num_kv_heads, seq_len, head_dim]
        # k_cache, v_cache shape: [batch, num_kv_heads, max_seq_len, head_dim]
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            cache_seq_len = k_cache.shape[2]

            # During warmup, seq_len may exceed cache size - skip caching in that case
            if cache_position + seq_len <= cache_seq_len:
                k_cache[:, :, cache_position:cache_position+seq_len, :] = k
                v_cache[:, :, cache_position:cache_position+seq_len, :] = v
                k = k_cache[:, :, :cache_position+seq_len, :]
                v = v_cache[:, :, :cache_position+seq_len, :]
            else:
                # Warmup/dummy run with more tokens than cache can hold - skip caching
                logger.warning(f"seq_len ({seq_len}) + cache_position ({cache_position}) exceeds cache size ({cache_seq_len}), skipping KV cache")

        # GQA expansion
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[-2], device=q.device, dtype=torch.bool),
                diagonal=k.shape[-2] - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Remove batch dim if we added it
        if input_2d:
            hidden_states = hidden_states.squeeze(0)

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

    def __init__(self, config: MambaInLlamaMambaConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{layer_idx}" if prefix else f"model.layers.{layer_idx}"
            if layer_idx in config.attn_layers:
                self.layers.append(MHADecoderLayer(config, layer_idx))
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
        mamba_cache: Optional[dict] = None,
        attn_cache: Optional[dict] = None,
        cache_position: int = 0,
        attn_metadata=None,
    ) -> torch.Tensor:
        """Forward pass with vLLM state management support.

        Args:
            input_ids: Input token IDs
            positions: Position indices for RoPE
            mamba_cache: Optional dict mapping layer_idx to (conv_state, ssm_state)
            attn_cache: Optional dict mapping layer_idx to (k_cache, v_cache)
            cache_position: Current position in sequence for decode
            attn_metadata: vLLM attention metadata for state indices
        """
        hidden_states = self.embed_input_ids(input_ids)

        # Use empty dicts if caches not provided
        if mamba_cache is None:
            mamba_cache = {}
        if attn_cache is None:
            attn_cache = {}

        for i, layer in enumerate(self.layers):
            if isinstance(layer, MambaDecoderLayer):
                conv_state, ssm_state = mamba_cache.get(i, (None, None))
                hidden_states = layer(
                    hidden_states,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    cache_position=cache_position,
                    attn_metadata=attn_metadata,
                )
            else:
                kv_cache = attn_cache.get(i)
                hidden_states = layer(hidden_states, positions, kv_cache, cache_position)

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

        # Pass prefix to model backbone for proper layer prefix registration
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = MambaInLlamaMambaModel(config, prefix=model_prefix)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

        # Cache position tracking (for non-vLLM use)
        self._cache_position = 0

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

        With vLLM V1, state is managed by vLLM:
        - Mamba layers get state from self.kv_cache (bound by vLLM)
        - Attention layers get KV cache from kv_caches parameter
        """
        # Handle 1D input
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.dim() == 1:
            positions = positions.unsqueeze(0)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # Determine if this is prefill or decode
        is_prefill = seq_len > 1 or self._cache_position == 0

        # Reset cache position for new prefill
        if is_prefill and self._cache_position > 0:
            self._cache_position = 0

        # Create positions if not provided
        if positions is None:
            positions = torch.arange(
                self._cache_position, self._cache_position + seq_len,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Build attention cache dict from vLLM's kv_caches list if provided
        # vLLM passes kv_caches as a list where each element corresponds to a layer
        attn_cache = {}
        if kv_caches is not None:
            attn_layer_indices = self.config.attn_layers or []
            for idx, layer_idx in enumerate(attn_layer_indices):
                if idx < len(kv_caches) and kv_caches[idx] is not None:
                    # kv_caches[idx] is typically (k_cache, v_cache) tuple
                    attn_cache[layer_idx] = kv_caches[idx]

        # Forward through model
        # Mamba layers get state from their own kv_cache attribute (bound by vLLM)
        # Attention layers get cache from attn_cache dict
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            mamba_cache=None,  # Mamba layers get state from self.kv_cache
            attn_cache=attn_cache,
            cache_position=self._cache_position,
            attn_metadata=attn_metadata,
        )

        # Update cache position
        self._cache_position += seq_len

        # Flatten for vLLM
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

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
