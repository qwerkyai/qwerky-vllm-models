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
"""PyTorch MambaInLlama (Mamba) model for vLLM inference."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutput

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mha import MHA
from mamba_ssm.utils.generation import decode as mamba_decode
from transformers.activations import ACT2FN

# Import Mamba dependencies
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

# Import config from our package
from .configuration import MambaInLlamaMambaConfig

logger = logging.get_logger(__name__)

# =============================================================================
# vLLM DETECTION AND IMPORTS
# =============================================================================

VLLM_MODE = False
_vllm_LogitsProcessor = None
_vllm_Sampler = None
_vllm_ModelRegistry = None
_vllm_MambaModelConfig = None

try:
    from vllm.model_executor.layers.logits_processor import (
        LogitsProcessor as _vllm_LogitsProcessor,
    )
    from vllm.model_executor.models import ModelRegistry as _vllm_ModelRegistry

    try:
        from vllm.model_executor.layers.sampler import Sampler as _vllm_Sampler
    except ImportError:
        try:
            from vllm.v1.sample.sampler import Sampler as _vllm_Sampler
        except ImportError:
            _vllm_Sampler = None

    # Import MambaModelConfig for proper prefix caching handling
    try:
        from vllm.model_executor.models.config import MambaModelConfig as _vllm_MambaModelConfig
    except ImportError:
        _vllm_MambaModelConfig = None

    # Import Mamba state shape calculators
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator as _vllm_MambaStateShapeCalculator,
            MambaStateDtypeCalculator as _vllm_MambaStateDtypeCalculator,
        )
    except ImportError:
        _vllm_MambaStateShapeCalculator = None
        _vllm_MambaStateDtypeCalculator = None

    VLLM_MODE = True
    logger.info("vLLM detected - dual-mode inference enabled")
except ImportError:
    pass


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


class Mamba(nn.Module):
    """Mamba SSM layer implementation."""

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
        """Forward pass for Mamba layer."""
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
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        if not self.repeat_kv_before_conv:
            x = rearrange(
                x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state
            )
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        assert self.activation in ["silu", "swish"]
        return_last_state = ssm_state is not None
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
        """Single step for decoding."""
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

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

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

        assert selective_state_update is not None
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
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
        self.mha = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_key_value_heads,
            layer_idx=layer_idx,
            mlp_dim=0,
            qkv_proj_bias=False,
            out_proj_bias=False,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb_base=config.rope_theta,
            causal=True,
            device=device,
            dtype=dtype,
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
        self.residual_in_fp32 = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mha.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mha(hidden_states, inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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


class MambaInLlamaMambaPreTrainedModel(PreTrainedModel):
    """Base class for MambaInLlama models."""

    config_class = MambaInLlamaMambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MambaDecoderLayer", "MHADecoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class MambaInLlamaMambaModel(MambaInLlamaMambaPreTrainedModel):
    """The bare MambaInLlama Model transformer."""

    def __init__(self, config: MambaInLlamaMambaConfig, **kwargs):
        super().__init__(config, **kwargs)
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
        self.post_init()

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


class MambaInLlamaMambaForCausalLM(MambaInLlamaMambaPreTrainedModel):
    """MambaInLlama Model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def is_backend_compatible(cls) -> bool:
        """Check if model is compatible with the current vLLM backend."""
        return True

    def __init__(
        self,
        config_or_vllm_config=None,
        config: MambaInLlamaMambaConfig = None,
        **kwargs,
    ):
        self._vllm_mode = False
        actual_config = None

        if config is not None:
            actual_config = config
        elif config_or_vllm_config is not None:
            if hasattr(config_or_vllm_config, "model_config"):
                self._vllm_mode = True
                vllm_config = config_or_vllm_config
                model_path = None
                if hasattr(vllm_config.model_config, "model"):
                    model_path = vllm_config.model_config.model
                if model_path:
                    actual_config = MambaInLlamaMambaConfig.from_pretrained(model_path)
                elif hasattr(vllm_config.model_config, "hf_config"):
                    hf_cfg = vllm_config.model_config.hf_config
                    actual_config = MambaInLlamaMambaConfig(
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
            elif isinstance(config_or_vllm_config, MambaInLlamaMambaConfig):
                actual_config = config_or_vllm_config
            else:
                actual_config = config_or_vllm_config

        if actual_config is None:
            raise ValueError(
                "Could not determine config. Provide either a MambaInLlamaMambaConfig "
                "or vLLM config object."
            )

        super().__init__(actual_config, **kwargs)

        self.model = MambaInLlamaMambaModel(actual_config, **kwargs)
        self.vocab_size = actual_config.vocab_size
        self.lm_head = nn.Linear(
            actual_config.hidden_size, actual_config.vocab_size, bias=False
        )

        if actual_config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._cached_device = None

        self._vllm_logits_processor = None
        self._vllm_sampler = None
        if VLLM_MODE and _vllm_LogitsProcessor is not None:
            self._vllm_logits_processor = _vllm_LogitsProcessor(actual_config.vocab_size)
        if VLLM_MODE and _vllm_Sampler is not None:
            self._vllm_sampler = _vllm_Sampler()

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids."""
        return self.model.embed_tokens(input_ids)

    def _is_vllm_context(
        self, kv_caches=None, attn_metadata=None, kwargs=None
    ) -> bool:
        """Detect if we're being called by vLLM or HuggingFace."""
        if kv_caches is not None or attn_metadata is not None:
            return True
        if kwargs:
            return "kv_caches" in kwargs or "attn_metadata" in kwargs
        return False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.LongTensor] = None,
        kv_caches: Optional[list] = None,
        attn_metadata=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inference_params=None,
        num_last_tokens: int = 0,
        intermediate_tensors=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput, torch.Tensor]:
        """Unified forward pass supporting both HuggingFace and vLLM modes."""
        if self._is_vllm_context(kv_caches, attn_metadata, kwargs):
            return self._forward_vllm(
                input_ids=input_ids,
                positions=positions or position_ids,
                inputs_embeds=inputs_embeds,
                kv_caches=kv_caches or kwargs.get("kv_caches"),
                attn_metadata=attn_metadata or kwargs.get("attn_metadata"),
            )
        else:
            return self._forward_hf(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids if position_ids is not None else positions,
                labels=labels,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens,
                **kwargs,
            )

    def _forward_hf(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inference_params=None,
        num_last_tokens: int = 0,
        **kwargs,
    ) -> CausalLMOutput:
        """HuggingFace-style forward pass."""
        is_prefill = (
            labels is None
            and (
                inference_params is None
                or getattr(inference_params, "seqlen_offset", 0) == 0
            )
            and num_last_tokens == 0
        )

        if is_prefill:
            num_last_tokens = 1

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            **kwargs,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutput(loss=loss, logits=logits)

    def _forward_vllm(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_caches: list = None,
        attn_metadata=None,
    ) -> torch.Tensor:
        """vLLM-style forward pass."""
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if input_ids is not None:
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
        elif inputs_embeds is not None:
            seq_len = (
                inputs_embeds.shape[1]
                if inputs_embeds.dim() > 2
                else inputs_embeds.shape[0]
            )
        else:
            seq_len = 1

        inference_params = _get_vllm_inference_params(seq_len)

        if seq_len > 1 and inference_params.seqlen_offset > 0:
            _reset_vllm_inference_params()
            inference_params = _get_vllm_inference_params(seq_len)

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=positions.unsqueeze(0)
            if positions is not None and positions.dim() == 1
            else positions,
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

        if self._vllm_logits_processor is not None:
            return self._vllm_logits_processor(self.lm_head, hidden_states)

        return self.lm_head(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """Sample tokens from logits for vLLM."""
        if self._vllm_sampler is not None:
            return self._vllm_sampler(logits, sampling_metadata)
        return None

    def load_weights(self, weights):
        """Load weights from checkpoint for vLLM."""
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

            if ".mha." in name:
                candidates.append(name.replace(".mha.", ".self_attn."))

            for candidate in candidates:
                if candidate in params_dict:
                    param = params_dict[candidate]
                    if param.shape == loaded_weight.shape:
                        param.data.copy_(loaded_weight)
                        loaded_count += 1
                        break

        logger.info(f"Loaded {loaded_count}/{len(params_dict)} parameters")

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for all layers."""
        return self.model.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def generate(
        self,
        input_ids,
        max_length=1024,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        temperature=1.0,
        repetition_penalty=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        """Generate sequences using the model."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if self._cached_device is None:
            self._cached_device = next(self.parameters()).device
        device = self._cached_device

        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        batch_size = input_ids.shape[0]

        if kwargs is not None:
            max_new_tokens = kwargs.pop("max_new_tokens", None)
            if max_new_tokens is not None:
                max_length = max_new_tokens + input_ids.shape[1]

            do_sample = kwargs.pop("do_sample", True)
            if not do_sample:
                top_k, top_p, min_p = 1, 0.0, 0.0

            cg = kwargs.pop("cg", True)

            eos_token_id = kwargs.pop("eos_token_id", self.config.eos_token_id)
            if eos_token_id is not None:
                if isinstance(eos_token_id, (list, tuple)):
                    eos_token_id = torch.tensor(
                        eos_token_id, dtype=torch.long, device=device
                    )
                else:
                    eos_token_id = torch.tensor(
                        [eos_token_id], dtype=torch.long, device=device
                    )

            kwargs.pop("attention_mask", None)
            kwargs.pop("pad_token_id", None)
            repetition_penalty = kwargs.pop("repetition_penalty", repetition_penalty)

            kwargs.pop("use_cache", None)
            kwargs.pop("no_repeat_ngram_size", None)
            kwargs.pop("length_penalty", None)
            kwargs.pop("num_return_sequences", None)
            kwargs.pop("num_beams", None)
            kwargs.pop("low_memory", None)
            kwargs.pop("stopping_criteria", None)

        output = mamba_decode(
            input_ids,
            self,
            max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            output_scores=output_scores,
            cg=cg,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


# =============================================================================
# NATIVE vLLM MODEL CLASS
# =============================================================================

# Dynamically create base classes to include MambaModelConfig if available
_NativeBaseClasses = (nn.Module,)
if _vllm_MambaModelConfig is not None:
    _NativeBaseClasses = (_vllm_MambaModelConfig, nn.Module)


class MambaInLlamaMambaForCausalLMNative(*_NativeBaseClasses):
    """Native vLLM-compatible model class (no PreTrainedModel inheritance).

    Inherits from MambaModelConfig (when available) to properly handle:
    - CUDA graph mode optimization for Mamba layers
    - Automatic prefix caching disabling (not yet supported for hybrid models)
    """

    is_hybrid: bool = True  # We have both Mamba and Attention layers
    has_inner_state: bool = True  # Mamba layers have internal state
    is_attention_free: bool = False  # We have attention layers

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple:
        """Calculate shapes for Mamba's convolutional and state caches.

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        if _vllm_MambaStateShapeCalculator is None:
            # Fallback if calculator not available
            return ((3, 4096), (4096, 16))

        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config

        # Get Mamba parameters from config
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
            # Fallback to model dtype
            return (torch.bfloat16, torch.bfloat16)

        return _vllm_MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def is_backend_compatible(cls) -> bool:
        """Required by vLLM 0.9.x for custom model classes."""
        return True

    @classmethod
    def register_for_auto_class(cls, auto_class=None):
        """No-op to satisfy Transformers' auto-registration."""
        pass

    @classmethod
    def _from_config(cls, config, **kwargs):
        """Create model from config (required for AutoModel.from_config)."""
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
        if VLLM_MODE and _vllm_LogitsProcessor is not None:
            self._vllm_logits_processor = _vllm_LogitsProcessor(config.vocab_size)
        if VLLM_MODE and _vllm_Sampler is not None:
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
