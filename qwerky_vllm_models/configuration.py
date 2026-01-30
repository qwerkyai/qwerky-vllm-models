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
"""MambaInLlama (Mamba) model configuration for vLLM plugin."""

from typing import List, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MambaInLlamaMambaConfig(PretrainedConfig):
    r"""
    Configuration class for MambaInLlama hybrid Mamba-Attention models.

    This model combines Mamba SSM layers with traditional attention layers,
    distilled from larger Llama models.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads for grouped query attention.
        hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/values attentions.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary containing the scaling configuration for RoPE embeddings.
        d_model (`int`, *optional*):
            Model dimension for Mamba layers. Defaults to `hidden_size`.
        d_inner (`int`, *optional*):
            Inner dimension for Mamba layers. Defaults to `intermediate_size`.
        d_xb (`int`, *optional*, defaults to 1024):
            Dimension for Mamba xB projection.
        ssm_cfg (`dict`, *optional*, defaults to `{}`):
            State space model configuration dictionary.
        attn_layers (`List[int]`, *optional*, defaults to `[]`):
            List of layer indices that use attention instead of Mamba.
    """

    model_type = "mambainllama_mamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_dropout: float = 0.0,
        # Mamba-specific parameters
        d_model: Optional[int] = None,
        d_inner: Optional[int] = None,
        d_xb: int = 1024,
        ssm_cfg: Optional[dict] = None,
        attn_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        # Mamba-specific parameters with sensible defaults
        # IMPORTANT: MambaInLlama uses d_inner = hidden_size for Mamba layers
        # (NOT intermediate_size, which is only for MLP layers)
        self.d_model = d_model if d_model is not None else hidden_size
        self.d_inner = d_inner if d_inner is not None else hidden_size
        # d_xb default: num_key_value_heads * head_dim (GQA-style)
        # For 8B: 8 * 128 = 1024, For 3B: 8 * 96 = 768
        head_dim = hidden_size // num_attention_heads
        self.d_xb = d_xb if d_xb is not None else (self.num_key_value_heads * head_dim)
        self.ssm_cfg = ssm_cfg if ssm_cfg is not None else {}
        self.attn_layers = attn_layers if attn_layers is not None else []

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
