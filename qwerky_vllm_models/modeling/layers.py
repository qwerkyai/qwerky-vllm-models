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
"""Layer implementations for MambaInLlama hybrid models.

Contains RMSNorm fallback, MLP, and MambaDecoderLayer.
"""

import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from huggingface_hub import snapshot_download
from transformers.utils import logging

from ..configuration import MambaInLlamaMambaConfig
from .mixer import MambaInLlamaMambaMixer, Mamba2InLlamaMambaMixer

logger = logging.get_logger(__name__)


def _resolve_model_dir(model_config) -> Optional[str]:
    """Return local directory path for the model checkpoint, or None."""
    model = getattr(model_config, "model", None)
    if not model:
        return None
    if os.path.isdir(model):
        return model
    # HuggingFace Hub model, resolve from local cache
    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:
        pass
    return None


def _checkpoint_has_key(model_config, key: str) -> bool:
    """Probe safetensors checkpoint metadata to check if a weight key exists.

    Reads only the index JSON (or safetensors header) — no tensors are loaded.
    Returns True if the key is present, False if absent, and True (safe default)
    if the checkpoint format cannot be determined.
    """
    try:
        model_path = _resolve_model_dir(model_config)
        if not model_path:
            return True  # safe default

        # Multi-file: check the shard index
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            return key in index.get("weight_map", {})

        # Single-file safetensors: read header only
        st_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(st_path):
            from safetensors import safe_open

            with safe_open(st_path, framework="pt", device="cpu") as f:
                return key in f.keys()
    except Exception:
        pass
    return True  # safe default


# =============================================================================
# vLLM IMPORTS
# =============================================================================

_vllm_available = False
RMSNorm = None

try:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.activation import SiluAndMul

    _vllm_available = True
except ImportError:
    pass


# =============================================================================
# FALLBACK IMPLEMENTATIONS
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


# =============================================================================
# MLP LAYER
# =============================================================================


class MLP(nn.Module):
    """MLP layer with fused gate+up projection and SiluAndMul activation."""

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        prefix: str = "",
        quant_config=None,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        if _vllm_available:
            self.gate_up_proj = MergedColumnParallelLinear(
                d_model,
                [intermediate_size, intermediate_size],
                bias=False,
                prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
                quant_config=quant_config,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                d_model,
                bias=False,
                input_is_parallel=True,
                prefix=f"{prefix}.down_proj" if prefix else "down_proj",
                quant_config=quant_config,
            )
            self.act_fn = SiluAndMul()
            self._use_fused = True
        else:
            self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
            self.act_fn = nn.SiLU() if hidden_act == "silu" else nn.GELU()
            self._use_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fused:
            x, _ = self.gate_up_proj(x)
            x = self.act_fn(x)
            x, _ = self.down_proj(x)
            return x
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# MAMBA DECODER LAYER
# =============================================================================


class MambaDecoderLayer(nn.Module):
    """Mamba SSM decoder layer."""

    def __init__(
        self,
        config: MambaInLlamaMambaConfig,
        layer_idx: int,
        prefix: str = "",
        model_config=None,
        cache_config=None,
        quant_config=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.prefix = prefix

        # Pass prefix, model_config, and cache_config to mixer
        mamba_prefix = (
            f"{prefix}.mamba" if prefix else f"model.layers.{layer_idx}.mamba"
        )
        self.mamba = MambaInLlamaMambaMixer(
            config,
            layer_idx,
            prefix=mamba_prefix,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.mlp = MLP(
            config.d_model,
            config.intermediate_size,
            config.hidden_act,
            prefix=f"{prefix}.mlp",
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
# MAMBA 2 DECODER LAYER
# =============================================================================


class Mamba2DecoderLayer(nn.Module):
    """Mamba 2 SSM decoder layer using Mamba2InLlamaMambaMixer."""

    def __init__(
        self,
        config: MambaInLlamaMambaConfig,
        layer_idx: int,
        prefix: str = "",
        model_config=None,
        cache_config=None,
        quant_config=None,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.prefix = prefix

        mamba_prefix = (
            f"{prefix}.mamba" if prefix else f"model.layers.{layer_idx}.mamba"
        )
        ssm_state_size = config.ssm_state_size or 64
        intermediate_size = config.d_inner or config.d_model
        # n_groups_ssm = number of SSM heads (= mamba_num_heads in config)
        n_groups_ssm = config.mamba_num_heads or 128
        # d_xb: the small B/C projection size (per-group before repeat)
        d_xb = config.d_xb or intermediate_size
        # norm_key = f"model.layers.{layer_idx}.mamba.norm.weight"
        # use_rms_norm = _checkpoint_has_key(model_config, norm_key) # check for RMSNorm presence in checkpoint
        self.mamba = Mamba2InLlamaMambaMixer(
            hidden_size=config.d_model,
            ssm_state_size=ssm_state_size,
            conv_kernel_size=config.d_conv,
            intermediate_size=intermediate_size,
            use_conv_bias=True,
            use_bias=False,
            d_xb=d_xb,
            n_groups_ssm=n_groups_ssm,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu",
            use_rms_norm=False,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=mamba_prefix,
        )
        self.mlp = MLP(
            config.d_model,
            config.intermediate_size,
            config.hidden_act,
            prefix=f"{prefix}.mlp",
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # MambaMixer2 returns output directly (handles custom op internally)
        hidden_states = self.mamba(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
