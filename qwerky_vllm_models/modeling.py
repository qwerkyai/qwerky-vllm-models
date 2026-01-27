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
"""PyTorch MambaInLlama (Mamba) model for vLLM inference.

This module uses vLLM's native Mamba ops - no mamba_ssm or causal_conv1d required.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

from transformers.utils import logging

# Import config from our package
from .configuration import MambaInLlamaMambaConfig

logger = logging.get_logger(__name__)

# =============================================================================
# vLLM NATIVE IMPORTS (replaces mamba_ssm)
# =============================================================================

# Try to import vLLM's native Mamba ops
_vllm_available = False
_selective_scan_fn = None
_selective_state_update = None
_causal_conv1d_fn = None
_causal_conv1d_update = None
_RMSNorm = None
_vllm_LogitsProcessor = None
_vllm_Sampler = None
_vllm_MambaModelConfig = None
_vllm_MambaStateShapeCalculator = None
_vllm_MambaStateDtypeCalculator = None
_vllm_Attention = None
_vllm_RotaryEmbedding = None

try:
    # vLLM's native Mamba ops (Triton-accelerated)
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        selective_scan_fn as _selective_scan_fn,
        selective_state_update as _selective_state_update,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn as _causal_conv1d_fn,
        causal_conv1d_update as _causal_conv1d_update,
    )

    # vLLM's RMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as _RMSNorm

    # vLLM model utilities
    from vllm.model_executor.layers.logits_processor import (
        LogitsProcessor as _vllm_LogitsProcessor,
    )

    try:
        from vllm.model_executor.layers.sampler import Sampler as _vllm_Sampler
    except ImportError:
        try:
            from vllm.v1.sample.sampler import Sampler as _vllm_Sampler
        except ImportError:
            pass

    # MambaModelConfig for proper prefix caching handling
    try:
        from vllm.model_executor.models.config import MambaModelConfig as _vllm_MambaModelConfig
    except ImportError:
        pass

    # Mamba state shape calculators
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator as _vllm_MambaStateShapeCalculator,
            MambaStateDtypeCalculator as _vllm_MambaStateDtypeCalculator,
        )
    except ImportError:
        pass

    # vLLM's attention for hybrid model
    try:
        from vllm.model_executor.layers.attention import Attention as _vllm_Attention
        from vllm.model_executor.layers.rotary_embedding import get_rope as _vllm_get_rope
    except ImportError:
        pass

    _vllm_available = True
    logger.info("vLLM native Mamba ops loaded successfully")

except ImportError as e:
    logger.warning(f"vLLM native Mamba ops not available: {e}")
    logger.warning("Falling back to pure PyTorch implementation")


# =============================================================================
# PURE PYTORCH FALLBACKS (for non-vLLM environments)
# =============================================================================

class RMSNormFallback(nn.Module):
    """RMSNorm fallback when vLLM is not available."""

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
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)

        if residual is not None:
            return x, residual
        return x


# Use vLLM's RMSNorm if available, otherwise fallback
RMSNorm = _RMSNorm if _RMSNorm is not None else RMSNormFallback


def selective_scan_fn_fallback(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
):
    """Pure PyTorch fallback for selective scan (slow but works)."""
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    if delta_softplus:
        delta = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1) if delta_bias is not None else delta)
    elif delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)

    # Discretize A
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1))  # (B, D, L, N)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, D, L, N)

    # Scan
    x = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (B, D, L)

    if D is not None:
        y = y + D.unsqueeze(0).unsqueeze(-1) * u

    if z is not None:
        y = y * F.silu(z)

    if return_last_state:
        return y, x
    return y


def causal_conv1d_fn_fallback(x, weight, bias=None, activation=None):
    """Pure PyTorch fallback for causal conv1d."""
    # x: (batch, dim, seqlen)
    # weight: (dim, kernel_size)
    batch, dim, seqlen = x.shape
    kernel_size = weight.shape[1]

    # Pad for causal convolution
    x_padded = F.pad(x, (kernel_size - 1, 0))

    # Depthwise conv
    weight_reshaped = weight.unsqueeze(1)  # (dim, 1, kernel_size)
    out = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=dim)

    if activation in ["silu", "swish"]:
        out = F.silu(out)

    return out


def causal_conv1d_update_fallback(x, conv_state, weight, bias=None, activation=None):
    """Pure PyTorch fallback for causal conv1d state update."""
    # x: (batch, dim)
    # conv_state: (batch, dim, kernel_size)

    # Shift state left and add new input
    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
    conv_state[:, :, -1] = x

    # Compute output
    out = torch.sum(conv_state * weight, dim=-1)
    if bias is not None:
        out = out + bias

    if activation in ["silu", "swish"]:
        out = F.silu(out)

    return out


def selective_state_update_fallback(
    ssm_state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """Pure PyTorch fallback for selective state update (decode step)."""
    # ssm_state: (batch, nheads, dim, dstate)
    # x: (batch, nheads, dim)

    if dt_softplus:
        dt = F.softplus(dt + dt_bias if dt_bias is not None else dt)
    elif dt_bias is not None:
        dt = dt + dt_bias

    # A: (nheads, dim, dstate)
    dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, nheads, dim, dstate)
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (batch, nheads, dim, dstate)

    # Update state
    ssm_state.copy_(ssm_state * dA + x.unsqueeze(-1) * dB)

    # Compute output
    y = torch.einsum("bhdn,bhdn->bhd", ssm_state, C.unsqueeze(2))

    if D is not None:
        y = y + D * x

    if z is not None:
        y = y * F.silu(z)

    return y


# Use vLLM ops if available, otherwise fallbacks
selective_scan_fn = _selective_scan_fn if _selective_scan_fn is not None else selective_scan_fn_fallback
selective_state_update = _selective_state_update if _selective_state_update is not None else selective_state_update_fallback
causal_conv1d_fn = _causal_conv1d_fn if _causal_conv1d_fn is not None else causal_conv1d_fn_fallback
causal_conv1d_update = _causal_conv1d_update if _causal_conv1d_update is not None else causal_conv1d_update_fallback


# =============================================================================
# vLLM INFERENCE STATE MANAGEMENT
# =============================================================================

@dataclass
class VLLMInferenceParams:
    """Inference parameters adapter for vLLM mode."""
    max_seqlen: int = 8192
    max_batch_size: int = 256
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: torch.Tensor = None


_vllm_inference_params: VLLMInferenceParams = None


def _get_vllm_inference_params(
    seqlen: int, batch_size: int = 1, max_seqlen: int = 8192
) -> VLLMInferenceParams:
    """Get or create inference params for vLLM mode."""
    global _vllm_inference_params
    if _vllm_inference_params is None:
        _vllm_inference_params = VLLMInferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )
    return _vllm_inference_params


def _update_vllm_inference_offset(seqlen: int):
    """Update seqlen_offset after a forward pass in vLLM mode."""
    global _vllm_inference_params
    if _vllm_inference_params is not None:
        if seqlen == 1:
            _vllm_inference_params.seqlen_offset += 1
        else:
            _vllm_inference_params.seqlen_offset = seqlen


def _reset_vllm_inference_params():
    """Reset inference params for a new sequence in vLLM mode."""
    global _vllm_inference_params
    if _vllm_inference_params is not None:
        if _vllm_inference_params.key_value_memory_dict:
            _vllm_inference_params.key_value_memory_dict.clear()
    _vllm_inference_params = None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match number of attention heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# =============================================================================
# MAMBA LAYER (uses vLLM native ops)
# =============================================================================

class Mamba(nn.Module):
    """Mamba SSM layer implementation using vLLM native ops."""

    def __init__(
        self,
        d_model,
        d_inner,
        d_xb,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        repeat_kv_before_conv=True,
        conv_bias=True,
        proj_x_bias=False,
        proj_z_bias=False,
        out_proj_bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_xb = d_xb
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = (
            d_inner if d_inner is not None else int(self.expand * self.d_model)
        )
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.repeat_kv_before_conv = repeat_kv_before_conv

        if self.repeat_kv_before_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_xb,
                out_channels=self.d_xb,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_xb,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.num_xb_head = self.d_xb // self.d_state
        self.num_C_head = self.d_inner // self.d_state
        self.repeat_group = self.num_C_head // self.num_xb_head

        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_xb + 2 * self.d_inner + self.dt_rank,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=out_proj_bias, **factory_kwargs
        )

    def forward(self, hidden_states, inference_params=None):
        """Forward pass for Mamba layer using vLLM native ops."""
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        A = -torch.exp(self.A_log.float())

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        zxbcdt = self.in_proj(hidden_states)
        z, x, B, C, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
            dim=-1,
        )

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")

        B = rearrange(
            B, "b l (n_group dstate) -> b n_group l dstate", dstate=self.d_state
        )
        B = repeat_kv(B, self.repeat_group)
        B = rearrange(B, "b n_group l dstate -> b n_group dstate l").contiguous()
        C = rearrange(
            C, "b l (n_group dstate) -> b n_group dstate l", dstate=self.d_state
        ).contiguous()

        dt = self.dt_proj(dt)
        dt = rearrange(dt, "b l d -> b d l")

        if self.repeat_kv_before_conv:
            x = rearrange(
                x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state
            )
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        need_state_update = conv_state is not None
        if need_state_update:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        # Use vLLM's causal_conv1d_fn if available
        if _causal_conv1d_fn is not None:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            x = self.act(self.conv1d(x)[..., :seqlen])

        if not self.repeat_kv_before_conv:
            x = rearrange(
                x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state
            )
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        return_last_state = ssm_state is not None

        # Use vLLM's selective_scan_fn
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=return_last_state,
        )
        if return_last_state:
            y, last_state = y
            ssm_state.copy_(
                rearrange(last_state, "b (h d) n -> b h d n", h=self.num_C_head)
            )
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """Single step for decoding using vLLM native ops."""
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, (
            "Only support decoding with 1 token at a time for now"
        )

        hidden_states_input = hidden_states.squeeze(1)
        A = -torch.exp(self.A_log.float())

        zxbcdt = self.in_proj(hidden_states_input)
        z, x, B, C, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
            dim=-1,
        )

        B = rearrange(B, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)

        C = rearrange(
            C, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state
        ).contiguous()

        dt = self.dt_proj(dt)

        if self.repeat_kv_before_conv:
            x = rearrange(
                x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state
            )
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        # Use vLLM's causal_conv1d_update if available
        if _causal_conv1d_update is not None:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)

        if not self.repeat_kv_before_conv:
            x = rearrange(
                x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state
            )
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        x = rearrange(x, "b (h d) -> b h d", h=self.num_C_head)
        dt = rearrange(dt, "b (h d) -> b h d", h=self.num_C_head)
        A = rearrange(A, "(h d) n -> h d n", h=self.num_C_head)
        D = rearrange(self.D, "(h d) -> h d", h=self.num_C_head)
        z = rearrange(z, "b (h d) -> b h d", h=self.num_C_head)
        dt_bias = rearrange(self.dt_proj.bias, "(h d) -> h d", h=self.num_C_head)

        # Use vLLM's selective_state_update
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True
        )

        y = rearrange(y, "b h d -> b (h d)")
        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for this layer."""
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        if self.repeat_kv_before_conv:
            conv_state = torch.zeros(
                batch_size, self.d_inner, self.d_conv, device=device, dtype=conv_dtype
            )
        else:
            conv_state = torch.zeros(
                batch_size, self.d_xb, self.d_conv, device=device, dtype=conv_dtype
            )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.num_C_head,
            self.d_inner // self.num_C_head,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        """Get or create states from inference cache."""
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            if self.repeat_kv_before_conv:
                conv_state = torch.zeros(
                    batch_size,
                    self.d_inner,
                    self.d_conv,
                    device=self.conv1d.weight.device,
                    dtype=self.conv1d.weight.dtype,
                )
            else:
                conv_state = torch.zeros(
                    batch_size,
                    self.d_xb,
                    self.d_conv,
                    device=self.conv1d.weight.device,
                    dtype=self.conv1d.weight.dtype,
                )
            ssm_state = torch.zeros(
                batch_size,
                self.num_C_head,
                self.d_inner // self.num_C_head,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


# =============================================================================
# MLP LAYER
# =============================================================================

class MLP(nn.Module):
    """MLP layer."""

    def __init__(self, d_model, intermediate_size, hidden_act, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, **factory_kwargs
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, **factory_kwargs
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, **factory_kwargs
        )
        self.act_fn = nn.SiLU() if hidden_act == "silu" else nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# ATTENTION LAYER (uses vLLM native attention when available)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary position embedding fallback."""

    def __init__(self, dim, max_position_embeddings=8192, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len, x.device)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MHADecoderLayer(nn.Module):
    """Multi-Head Attention decoder layer."""

    def __init__(
        self,
        config: MambaInLlamaMambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MHADecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # QKV projection
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, **factory_kwargs)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, **factory_kwargs)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
            base=config.rope_theta,
            device=device,
        )

        self.mlp = MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            **factory_kwargs,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs
        )

        # KV cache for inference
        self.k_cache = None
        self.v_cache = None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.q_proj.weight.device
        cache_dtype = dtype or self.q_proj.weight.dtype
        self.k_cache = torch.zeros(
            batch_size, max_seqlen, self.num_kv_heads, self.head_dim,
            device=device, dtype=cache_dtype
        )
        self.v_cache = torch.zeros(
            batch_size, max_seqlen, self.num_kv_heads, self.head_dim,
            device=device, dtype=cache_dtype
        )
        return self.k_cache, self.v_cache

    def forward(
        self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embedding
        cos, sin = self.rotary_emb(q, seq_len)

        # Handle position_ids for inference
        if inference_params is not None and inference_params.seqlen_offset > 0:
            position_ids = torch.arange(
                inference_params.seqlen_offset,
                inference_params.seqlen_offset + seq_len,
                device=q.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # KV cache handling for inference
        if inference_params is not None:
            cache_key = f"attn_{self.layer_idx}"
            if cache_key not in inference_params.key_value_memory_dict:
                # Initialize cache
                max_cache_len = inference_params.max_seqlen
                inference_params.key_value_memory_dict[cache_key] = {
                    'k': torch.zeros(batch_size, self.num_kv_heads, max_cache_len, self.head_dim, device=k.device, dtype=k.dtype),
                    'v': torch.zeros(batch_size, self.num_kv_heads, max_cache_len, self.head_dim, device=v.device, dtype=v.dtype),
                }

            cache = inference_params.key_value_memory_dict[cache_key]
            start_pos = inference_params.seqlen_offset
            cache['k'][:, :, start_pos:start_pos+seq_len, :] = k.transpose(1, 2).transpose(2, 3).transpose(1, 2)
            cache['v'][:, :, start_pos:start_pos+seq_len, :] = v.transpose(1, 2).transpose(2, 3).transpose(1, 2)

            # Use cached K, V
            k = cache['k'][:, :, :start_pos+seq_len, :]
            v = cache['v'][:, :, :start_pos+seq_len, :]

        # Repeat K, V for grouped query attention
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[-2], device=q.device, dtype=torch.bool),
                diagonal=k.shape[-2] - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP
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

    def __init__(
        self, config: MambaInLlamaMambaConfig, layer_idx: int, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MambaDecoderLayer, self).__init__()
        self.layer_idx = layer_idx

        self.mamba = Mamba(
            d_model=config.d_model,
            d_inner=config.d_inner,
            d_xb=config.d_xb,
            layer_idx=layer_idx,
            **config.ssm_cfg,
            **factory_kwargs,
        )
        self.mlp = MLP(
            config.d_model,
            config.intermediate_size,
            config.hidden_act,
            **factory_kwargs,
        )
        self.input_layernorm = RMSNorm(
            config.d_model, eps=config.rms_norm_eps, **factory_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.d_model, eps=config.rms_norm_eps, **factory_kwargs
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states, inference_params=inference_params)
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
    """The bare MambaInLlama Model transformer."""

    def __init__(self, config: MambaInLlamaMambaConfig, **kwargs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                MHADecoderLayer(config, layer_idx, device=None, dtype=None)
                if layer_idx in config.attn_layers
                else MambaDecoderLayer(config, layer_idx, device=None, dtype=None)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params=None,
        num_last_tokens: int = 0,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params, **kwargs
            )
            if not hidden_states.is_contiguous():
                hidden_states = hidden_states.contiguous()

        hidden_states = self.norm(hidden_states)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for all layers."""
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }


# =============================================================================
# NATIVE vLLM MODEL CLASS
# =============================================================================

# Dynamically create base classes to include MambaModelConfig if available
_NativeBaseClasses = (nn.Module,)
if _vllm_MambaModelConfig is not None:
    _NativeBaseClasses = (_vllm_MambaModelConfig, nn.Module)


class MambaInLlamaMambaForCausalLMNative(*_NativeBaseClasses):
    """Native vLLM-compatible model class (no mamba_ssm dependency).

    Uses vLLM's native Mamba ops for maximum compatibility.
    """

    is_hybrid: bool = True  # We have both Mamba and Attention layers
    has_inner_state: bool = True  # Mamba layers have internal state
    is_attention_free: bool = False  # We have attention layers

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple:
        """Calculate shapes for Mamba's convolutional and state caches."""
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
        """Get dtypes for Mamba state caches."""
        if _vllm_MambaStateDtypeCalculator is None:
            return (torch.bfloat16, torch.bfloat16)

        return _vllm_MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def is_backend_compatible(cls) -> bool:
        """Required by vLLM for custom model classes."""
        return True

    @classmethod
    def register_for_auto_class(cls, auto_class=None):
        """No-op to satisfy Transformers' auto-registration."""
        pass

    @classmethod
    def _from_config(cls, config, **kwargs):
        """Create model from config."""
        return cls(config=config, **kwargs)

    def __init__(
        self,
        vllm_config=None,
        config: MambaInLlamaMambaConfig = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()

        if vllm_config is not None:
            if hasattr(vllm_config, "model_config"):
                model_config = vllm_config.model_config
                if hasattr(model_config, "hf_config"):
                    hf_cfg = model_config.hf_config
                    config = MambaInLlamaMambaConfig(
                        vocab_size=getattr(hf_cfg, "vocab_size", 32000),
                        hidden_size=getattr(hf_cfg, "hidden_size", 4096),
                        num_hidden_layers=getattr(hf_cfg, "num_hidden_layers", 32),
                        num_attention_heads=getattr(hf_cfg, "num_attention_heads", 32),
                        num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", None),
                        intermediate_size=getattr(hf_cfg, "intermediate_size", 11008),
                        rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
                        rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
                        attn_layers=getattr(hf_cfg, "attn_layers", []),
                        d_model=getattr(hf_cfg, "d_model", None),
                        d_inner=getattr(hf_cfg, "d_inner", None),
                        d_xb=getattr(hf_cfg, "d_xb", None),
                        ssm_cfg=getattr(hf_cfg, "ssm_cfg", {}),
                    )

        if config is None:
            raise ValueError("Config required for model initialization")

        self.config = config
        self.vocab_size = config.vocab_size

        self.model = MambaInLlamaMambaModel(config)

        self._use_vllm_lm_head = False
        try:
            from vllm.model_executor.layers.linear import ColumnParallelLinear
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                gather_output=False,
            )
            self._use_vllm_lm_head = True
        except ImportError:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if not self._use_vllm_lm_head and getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self._vllm_logits_processor = None
        self._vllm_sampler = None
        if _vllm_available and _vllm_LogitsProcessor is not None:
            self._vllm_logits_processor = _vllm_LogitsProcessor(config.vocab_size)
        if _vllm_available and _vllm_Sampler is not None:
            self._vllm_sampler = _vllm_Sampler()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids."""
        return self.model.embed_tokens(input_ids)

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
        """vLLM-style forward pass."""
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if input_ids is not None:
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1] if inputs_embeds.dim() > 2 else inputs_embeds.shape[0]
        else:
            seq_len = 1

        inference_params = _get_vllm_inference_params(seq_len)

        if seq_len > 1 and inference_params.seqlen_offset > 0:
            _reset_vllm_inference_params()
            inference_params = _get_vllm_inference_params(seq_len)

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=positions.unsqueeze(0) if positions is not None and positions.dim() == 1 else positions,
            inference_params=inference_params,
        )

        _update_vllm_inference_offset(seq_len)

        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute logits for vLLM sampling."""
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

        if self._use_vllm_lm_head and self._vllm_logits_processor is not None:
            return self._vllm_logits_processor(self.lm_head, hidden_states)

        lm_head_output = self.lm_head(hidden_states)
        if isinstance(lm_head_output, tuple):
            logits = lm_head_output[0]
        else:
            logits = lm_head_output
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """Sample tokens from logits."""
        if self._vllm_sampler is not None:
            return self._vllm_sampler(logits, sampling_metadata)
        return None

    def load_weights(self, weights):
        """Load weights from checkpoint."""
        weights_list = list(weights) if not isinstance(weights, list) else weights
        params_dict = dict(self.named_parameters())
        loaded_count = 0

        for name, loaded_weight in weights_list:
            if name in params_dict:
                param = params_dict[name]
                if param.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                    loaded_count += 1
                    continue

            candidates = [name]
            if name.startswith("model."):
                candidates.append(name[6:])
            else:
                candidates.append(f"model.{name}")

            for candidate in candidates:
                if candidate in params_dict:
                    param = params_dict[candidate]
                    if param.shape == loaded_weight.shape:
                        param.data.copy_(loaded_weight)
                        loaded_count += 1
                        break

        logger.info(f"[Native] Loaded {loaded_count}/{len(params_dict)} parameters")
