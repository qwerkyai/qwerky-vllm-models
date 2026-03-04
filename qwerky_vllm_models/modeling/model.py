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
"""Model backbone and CausalLM classes for MambaInLlama hybrid models.

Contains MambaInLlamaMambaModel (backbone) and MambaInLlamaMambaForCausalLMNative
(the main vLLM-compatible model class), plus architecture aliases.
"""

import json
import math
import os
from typing import ClassVar, Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn

from transformers.utils import logging

from ..configuration import MambaInLlamaMambaConfig
from .layers import MambaDecoderLayer, Mamba2DecoderLayer, MLP, RMSNorm

logger = logging.get_logger(__name__)


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
                    logger.info(f"Loaded mamba_config.json from {path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    return mamba_config


# =============================================================================
# vLLM IMPORTS
# =============================================================================

_vllm_available = False

try:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
        ParallelLMHead,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.config import VllmConfig

    _vllm_available = True
except ImportError as e:
    logger.warning(f"vLLM not available: {e}")

# LlamaDecoderLayer import for MHA layers
_LlamaDecoderLayer = None
try:
    from vllm.model_executor.models.llama import LlamaDecoderLayer as _LlamaDecoderLayer
except ImportError:
    pass

# Try to import Sampler (location varies by vLLM version)
_vllm_Sampler = None
try:
    from vllm.model_executor.layers.sampler import Sampler as _vllm_Sampler
except ImportError:
    try:
        from vllm.v1.sample.sampler import Sampler as _vllm_Sampler
    except ImportError:
        pass

# Try to import protocol interfaces for model registration
_HasInnerState = None
_IsHybrid = None
_SupportsMambaPrefixCaching = None
try:
    from vllm.model_executor.models.interfaces import HasInnerState as _HasInnerState
    from vllm.model_executor.models.interfaces import IsHybrid as _IsHybrid
    from vllm.model_executor.models.interfaces import (
        SupportsMambaPrefixCaching as _SupportsMambaPrefixCaching,
    )
except ImportError:
    pass

# State calculators (for class methods)
_vllm_MambaStateShapeCalculator = None
_vllm_MambaStateDtypeCalculator = None
try:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator as _vllm_MambaStateShapeCalculator,
        MambaStateDtypeCalculator as _vllm_MambaStateDtypeCalculator,
    )
except ImportError:
    pass

# support_torch_compile for torch.compile integration
_support_torch_compile = None
try:
    from vllm.compilation.decorators import support_torch_compile as _support_torch_compile
except ImportError:
    pass


# =============================================================================
# MODEL BACKBONE
# =============================================================================

class MambaInLlamaMambaModel(nn.Module):
    """MambaInLlama Model backbone."""

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "",
                 config: MambaInLlamaMambaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.vllm_config = vllm_config

        # Extract cache_config, model_config, and quant_config from vllm_config
        cache_config = vllm_config.cache_config if vllm_config else None
        model_config = vllm_config.model_config if vllm_config else None
        quant_config = getattr(vllm_config, 'quant_config', None) if vllm_config else None

        # Register splitting ops so torch.compile doesn't try to compile our custom ops
        is_mamba2 = getattr(config, 'mamba_version', 'Mamba1') == 'Mamba2'
        if vllm_config is not None:
            compilation_config = vllm_config.compilation_config
            if is_mamba2:
                op_name = "vllm::mamba2inllama_mixer"
            else:
                op_name = "vllm::mambainllama_mixer"
            if (compilation_config.splitting_ops is not None
                    and op_name not in compilation_config.splitting_ops):
                compilation_config.splitting_ops.append(op_name)

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList()
        if config.attn_layers and _LlamaDecoderLayer is None:
            raise RuntimeError(
                "LlamaDecoderLayer could not be imported from vllm.model_executor.models.llama. "
                "Attention layers require vLLM >= 0.14.0."
            )
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{layer_idx}" if prefix else f"model.layers.{layer_idx}"
            if layer_idx in config.attn_layers:
                self.layers.append(_LlamaDecoderLayer(
                    vllm_config=vllm_config,
                    prefix=layer_prefix,
                ))
            elif is_mamba2:
                self.layers.append(Mamba2DecoderLayer(
                    config, layer_idx, prefix=layer_prefix,
                    model_config=model_config, cache_config=cache_config,
                    quant_config=quant_config,
                ))
            else:
                self.layers.append(MambaDecoderLayer(
                    config, layer_idx, prefix=layer_prefix,
                    model_config=model_config, cache_config=cache_config,
                    quant_config=quant_config,
                ))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (required by VllmModel interface)."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with vLLM state management and fused RMSNorm residual.

        Args:
            input_ids: Input token IDs [num_tokens] or [batch, seq]
            positions: Position indices for RoPE [num_tokens]
        """
        hidden_states = self.embed_input_ids(input_ids)

        # Thread residual through all layers for fused RMSNorm
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Final norm fuses last residual addition
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# Apply @support_torch_compile decorator to backbone
if _vllm_available and _support_torch_compile is not None:
    MambaInLlamaMambaModel = _support_torch_compile(MambaInLlamaMambaModel)


# =============================================================================
# NATIVE vLLM MODEL CLASS
# =============================================================================

# Dynamically create base classes with protocol inheritance
_NativeBaseClasses = [nn.Module]
if _HasInnerState is not None:
    _NativeBaseClasses.append(_HasInnerState)
if _IsHybrid is not None:
    _NativeBaseClasses.append(_IsHybrid)
if _SupportsMambaPrefixCaching is not None:
    _NativeBaseClasses.append(_SupportsMambaPrefixCaching)
_NativeBaseClasses = tuple(_NativeBaseClasses)


class MambaInLlamaMambaForCausalLMNative(*_NativeBaseClasses):
    """Native vLLM-compatible MambaInLlama model.

    This model supports the 'generate' runner by:
    1. Inheriting from HasInnerState, IsHybrid, and SupportsMambaPrefixCaching protocols
    2. Implementing compute_logits() and sample() methods
    3. Having architecture name ending in 'ForCausalLM'
    """

    # Protocol-required class variables for vLLM model inspection
    is_hybrid: ClassVar[Literal[True]] = True
    has_inner_state: ClassVar[Literal[True]] = True
    supports_mamba_prefix_caching: ClassVar[Literal[True]] = True

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
                    max_position_embeddings=getattr(hf_cfg, "max_position_embeddings", 2048),
                    rope_scaling=getattr(hf_cfg, "rope_scaling", None),
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

                # Mamba 2 parameters — priority: mamba_config.json > hf_config
                mamba_version = (
                    mamba_cfg.get("mamba_version")
                    or getattr(hf_cfg, "mamba_version", None)
                )
                if mamba_version:
                    config_kwargs["mamba_version"] = mamba_version
                if mamba_cfg.get("n_groups"):
                    config_kwargs["mamba_n_groups"] = mamba_cfg["n_groups"]
                if mamba_cfg.get("num_heads"):
                    config_kwargs["mamba_num_heads"] = mamba_cfg["num_heads"]
                if mamba_cfg.get("head_dim"):
                    config_kwargs["mamba_head_dim"] = mamba_cfg["head_dim"]
                if mamba_cfg.get("ssm_state_size"):
                    config_kwargs["ssm_state_size"] = mamba_cfg["ssm_state_size"]
                if mamba_cfg.get("d_conv"):
                    config_kwargs["d_conv"] = mamba_cfg["d_conv"]
                if mamba_cfg.get("mamba_intermediate_size"):
                    config_kwargs["mamba_intermediate_size"] = mamba_cfg["mamba_intermediate_size"]

                logger.info(f"Final config_kwargs: d_inner={config_kwargs.get('d_inner')}, d_xb={config_kwargs.get('d_xb')}, attn_layers={config_kwargs.get('attn_layers')}, mamba_version={config_kwargs.get('mamba_version', 'Mamba1')}")
                config = MambaInLlamaMambaConfig(**config_kwargs)

        if config is None:
            raise ValueError("Config required for model initialization")

        self.config = config
        self.vocab_size = config.vocab_size
        self.prefix = prefix

        # Pass vllm_config to model backbone (extracts cache_config/model_config internally)
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = MambaInLlamaMambaModel(
            vllm_config=vllm_config, config=config, prefix=model_prefix,
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
        intermediate_tensors=None,
        **kwargs,
    ) -> torch.Tensor:
        """vLLM-style forward pass.

        With vLLM V1:
        - Mamba layers get state from self.kv_cache (bound by vLLM via MambaBase)
        - Attention layers get KV cache from vLLM Attention class (via forward context)
        - Positions come from vLLM model runner
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
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
        1. mha.in_proj.weight -> split into q/k/v, load via self_attn.qkv_proj (LlamaDecoderLayer)
        2. mha.out_proj.weight -> rename to self_attn.o_proj.weight (LlamaDecoderLayer)
        3. mamba.A_log -> mamba.A (via set_weight_attrs weight_loader)
        4. mamba.in_proj.weight -> split into 5 shards for MergedColumnParallelLinear
        5. mlp.gate_proj/up_proj -> gate_up_proj shards for MergedColumnParallelLinear
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Mamba in_proj shard sizes depend on Mamba version
        is_mamba2 = getattr(self.config, 'mamba_version', 'Mamba1') == 'Mamba2'
        if is_mamba2:
            m2_intermediate = getattr(self.config, 'mamba_intermediate_size', None) or self.config.d_model
            m2_n_groups = getattr(self.config, 'mamba_n_groups', 1)
            m2_ssm_state = getattr(self.config, 'ssm_state_size', 64)
            m2_num_heads = getattr(self.config, 'mamba_num_heads', 128)
            groups_ssm_state_size = m2_n_groups * m2_ssm_state
            # Mamba 2 in_proj: [gate, x, B, C, dt]
            mamba_in_proj_sizes = [m2_intermediate, m2_intermediate, groups_ssm_state_size, groups_ssm_state_size, m2_num_heads]
        else:
            # Mamba 1 in_proj: [z, x, B, C, dt]
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
            # === Quantization scale/zero-point tensors: route to default handler ===
            # compressed-tensors format adds weight_scale, input_scale, etc.
            # These must NOT be caught by the substring-based weight handlers below.
            if name.endswith(("_scale", "_zero_point")):
                # Apply same name remappings as for weights
                param_name = name
                shard_id = None
                if ".mha.in_proj." in param_name:
                    param_name = param_name.replace(".mha.in_proj.", ".self_attn.qkv_proj.")
                elif ".mha.out_proj." in param_name:
                    param_name = param_name.replace(".mha.out_proj.", ".self_attn.o_proj.")
                elif ".mlp.gate_proj." in param_name:
                    param_name = param_name.replace(".mlp.gate_proj.", ".mlp.gate_up_proj.")
                    shard_id = 0
                elif ".mlp.up_proj." in param_name:
                    param_name = param_name.replace(".mlp.up_proj.", ".mlp.gate_up_proj.")
                    shard_id = 1
                if param_name not in params_dict:
                    if param_name.startswith("model."):
                        param_name = param_name[6:]
                    else:
                        param_name = f"model.{param_name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    if shard_id is not None:
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                continue

            # === MHA attention: fused in_proj -> split and load via QKVParallelLinear ===
            # LlamaDecoderLayer stores attention as self_attn.qkv_proj
            if ".mha.in_proj.weight" in name:
                base_name = name.replace(".mha.in_proj.weight", "")
                q_weight = loaded_weight[:q_dim, :]
                k_weight = loaded_weight[q_dim:q_dim + kv_dim, :]
                v_weight = loaded_weight[q_dim + kv_dim:, :]
                param_name = base_name + ".self_attn.qkv_proj.weight"
                if param_name not in params_dict:
                    if name.startswith("model."):
                        param_name = name[6:].replace(".mha.in_proj.weight", ".self_attn.qkv_proj.weight")
                    else:
                        param_name = f"model.{name}".replace(".mha.in_proj.weight", ".self_attn.qkv_proj.weight")
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, q_weight, "q")
                    weight_loader(param, k_weight, "k")
                    weight_loader(param, v_weight, "v")
                    loaded_params.add(param_name)
                continue

            # === MHA attention: out_proj rename (LlamaDecoderLayer uses self_attn.o_proj) ===
            if ".mha.out_proj." in name:
                new_name = name.replace(".mha.out_proj.", ".self_attn.o_proj.")
                if new_name not in params_dict:
                    if name.startswith("model."):
                        new_name = name[6:].replace(".mha.out_proj.", ".self_attn.o_proj.")
                    else:
                        new_name = f"model.{name}".replace(".mha.out_proj.", ".self_attn.o_proj.")
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

            # === Mamba in_proj: load directly for Mamba2 (qwerky layout) ===
            if ".mamba.in_proj.weight" in name:
                param_name = name
                if param_name not in params_dict:
                    param_name = f"model.{name}" if not name.startswith("model.") else name[6:]
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    if is_mamba2:
                        # QwerkyMamba2Mixer stores weights in qwerky's layout
                        # [z, x, B, C, dt]; load the full matrix directly.
                        weight_loader(param, loaded_weight)
                    else:
                        # Mamba1: split into 5 shards and load via shard_id
                        shards = torch.split(loaded_weight, mamba_in_proj_sizes, dim=0)
                        for shard_id, shard_weight in enumerate(shards):
                            weight_loader(param, shard_weight, shard_id)
                    loaded_params.add(param_name)
                continue

            # === MLP: gate_proj -> gate_up_proj shard 0 ===
            if ".mlp.gate_proj.weight" in name:
                param_name = name.replace(".mlp.gate_proj.weight",
                                          ".mlp.gate_up_proj.weight")
                if param_name not in params_dict:
                    if name.startswith("model."):
                        param_name = name[6:].replace(".mlp.gate_proj.weight",
                                                      ".mlp.gate_up_proj.weight")
                    else:
                        param_name = f"model.{name}".replace(".mlp.gate_proj.weight",
                                                             ".mlp.gate_up_proj.weight")
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight, 0)
                    loaded_params.add(param_name)
                continue

            # === MLP: up_proj -> gate_up_proj shard 1 ===
            if ".mlp.up_proj.weight" in name:
                param_name = name.replace(".mlp.up_proj.weight",
                                          ".mlp.gate_up_proj.weight")
                if param_name not in params_dict:
                    if name.startswith("model."):
                        param_name = name[6:].replace(".mlp.up_proj.weight",
                                                      ".mlp.gate_up_proj.weight")
                    else:
                        param_name = f"model.{name}".replace(".mlp.up_proj.weight",
                                                             ".mlp.gate_up_proj.weight")
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight, 1)
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
        mamba_version = getattr(hf_config, 'mamba_version', 'Mamba1')

        if mamba_version == 'Mamba2':
            # config.json typically uses model_type="llama" so hf_config won't have
            # Mamba2-specific fields. Load them from mamba_config.json directly.
            mamba_cfg = {}
            model_path = getattr(vllm_config.model_config, 'model', '')
            if model_path:
                try:
                    from huggingface_hub import hf_hub_download
                    mamba_config_path = hf_hub_download(model_path, "mamba_config.json")
                    with open(mamba_config_path, "r") as f:
                        mamba_cfg = json.load(f)
                except Exception:
                    mamba_cfg = _load_mamba_config(model_path)

            num_heads = (mamba_cfg.get("num_heads")
                         or getattr(hf_config, 'mamba_num_heads', 128))
            head_dim = (mamba_cfg.get("head_dim")
                        or getattr(hf_config, 'mamba_head_dim', 64))
            ssm_state_size = (mamba_cfg.get("ssm_state_size")
                              or getattr(hf_config, 'ssm_state_size', 64))
            d_conv = (mamba_cfg.get("d_conv")
                      or getattr(hf_config, 'd_conv', 4))
            d_xb = (mamba_cfg.get("d_xb")
                    or getattr(hf_config, 'd_xb', None))
            intermediate_size = (
                mamba_cfg.get("mamba_intermediate_size")
                or mamba_cfg.get("d_inner")
                or getattr(hf_config, 'mamba_intermediate_size', None)
                or hf_config.hidden_size
            )
            # n_groups for conv_dim: use d_xb // ssm_state_size (qwerky-distill layout)
            if d_xb:
                n_groups_conv = d_xb // ssm_state_size
            else:
                n_groups_conv = (mamba_cfg.get("n_groups")
                                 or getattr(hf_config, 'mamba_n_groups', 1))

            return _vllm_MambaStateShapeCalculator.mamba2_state_shape(
                tp_world_size=parallel_config.tensor_parallel_size,
                intermediate_size=intermediate_size,
                n_groups=n_groups_conv,
                num_heads=num_heads,
                head_dim=head_dim,
                state_size=ssm_state_size,
                conv_kernel=d_conv,
            )

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
        """Get Mamba state dtypes.

        Must match instance get_state_dtype(): SSM state defaults to float32
        when mamba_ssm_cache_dtype is "auto" to avoid bfloat16 rounding errors.
        """
        if _vllm_MambaStateDtypeCalculator is None:
            return (torch.bfloat16, torch.float32)

        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        mamba_version = getattr(hf_config, 'mamba_version', 'Mamba1')

        if mamba_version == 'Mamba2':
            return _vllm_MambaStateDtypeCalculator.mamba2_state_dtype(
                vllm_config.model_config.dtype,
                cache_config.mamba_cache_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )

        if cache_config.mamba_ssm_cache_dtype != "auto":
            return _vllm_MambaStateDtypeCalculator.mamba1_state_dtype(
                vllm_config.model_config.dtype,
                cache_config.mamba_cache_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )
        from vllm.model_executor.layers.mamba.mamba_utils import (
            get_kv_cache_torch_dtype,
        )
        conv_dtype = get_kv_cache_torch_dtype(
            cache_config.mamba_cache_dtype,
            vllm_config.model_config.dtype,
        )
        return (conv_dtype, torch.float32)

    @classmethod
    def is_backend_compatible(cls) -> bool:
        return True


# =============================================================================
# ALIASES FOR HF CONFIG COMPATIBILITY
# =============================================================================
# HuggingFace model configs specify "MambaInLlamaMambaForCausalLM" as the
# architecture. This alias ensures vLLM can find and load the class.
MambaInLlamaMambaForCausalLM = MambaInLlamaMambaForCausalLMNative

# New architecture alias for Qwerky models
QwerkyLlamaMambaHybridForCausalLM = MambaInLlamaMambaForCausalLMNative

# Mamba 2 hybrid architecture alias
QwerkyMamba2HybridForCausalLM = MambaInLlamaMambaForCausalLMNative
