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
"""
Qwerky vLLM Models Plugin

This plugin registers Qwerky AI's MambaInLlama hybrid models with vLLM,
enabling serving without the --trust-remote-code flag.

Usage:
    pip install qwerky-vllm-models
    vllm serve QwerkyAI/Qwerky-Llama3.1-Mamba-8B-Llama3.3-70B-base-distill
"""

__version__ = "0.2.66"

# Track if we've already registered with Transformers
_transformers_registered = False


def _register_with_transformers():
    """Register our config class with HuggingFace Transformers AutoConfig."""
    global _transformers_registered
    if _transformers_registered:
        return

    try:
        from transformers import AutoConfig
        from .configuration import MambaInLlamaMambaConfig, QwerkyLlamaMambaHybridConfig

        # Register the config class for our model_type
        AutoConfig.register("mambainllama_mamba", MambaInLlamaMambaConfig)
        # Register the Qwick model_type (subclass with matching model_type attribute)
        AutoConfig.register("qwerky_llama_mamba_hybrid", QwerkyLlamaMambaHybridConfig)
        _transformers_registered = True
    except Exception:
        # Transformers not available or already registered
        pass


def register():
    """Register Qwerky models with vLLM and Transformers.

    This function is called automatically by vLLM's plugin system when
    the package is installed. It registers:
    1. The model architectures with vLLM's ModelRegistry
    2. The config class with HuggingFace's AutoConfig

    Uses lazy registration (string-based) to avoid CUDA re-initialization
    issues when vLLM spawns worker processes.
    """
    # First, register with Transformers so AutoConfig works
    _register_with_transformers()

    # Log environment versions for debugging across machines
    import logging
    logger = logging.getLogger(__name__)
    try:
        import vllm
        import torch
        import transformers
        logger.info(
            "qwerky-vllm-models==%s, vllm==%s, torch==%s, transformers==%s",
            __version__, vllm.__version__, torch.__version__, transformers.__version__,
        )
    except ImportError:
        logger.info("qwerky-vllm-models==%s (some dependencies not yet installed)", __version__)

    # Then register with vLLM
    try:
        from vllm import ModelRegistry
    except ImportError:
        # vLLM not installed - skip registration
        return

    # Get currently registered architectures
    try:
        registered = ModelRegistry.get_supported_archs()
    except Exception:
        registered = set()

    # Register using lazy loading (string path) to avoid CUDA issues
    # This defers the actual import until the model is needed

    # Register models with vLLM's ModelRegistry using lazy loading (string path)
    # to avoid CUDA re-initialization issues in worker subprocesses.
    #
    # vLLM determines task support ('generate' vs 'pooling') through:
    # 1. Model architecture name suffix (e.g., "ForCausalLM" -> generate)
    # 2. Model class inspection (is_text_generation_model, has_inner_state, etc.)
    # 3. Protocol inheritance (HasInnerState, IsHybrid)

    if "MambaInLlamaMambaForCausalLM" not in registered:
        try:
            ModelRegistry.register_model(
                "MambaInLlamaMambaForCausalLM",
                "qwerky_vllm_models.modeling:MambaInLlamaMambaForCausalLM"
            )
        except Exception:
            pass

    if "MambaInLlamaMambaForCausalLMNative" not in registered:
        try:
            ModelRegistry.register_model(
                "MambaInLlamaMambaForCausalLMNative",
                "qwerky_vllm_models.modeling:MambaInLlamaMambaForCausalLMNative"
            )
        except Exception:
            pass

    if "QwerkyLlamaMambaHybridForCausalLM" not in registered:
        try:
            ModelRegistry.register_model(
                "QwerkyLlamaMambaHybridForCausalLM",
                "qwerky_vllm_models.modeling:QwerkyLlamaMambaHybridForCausalLM"
            )
        except Exception:
            pass


# Also export the model classes for direct import if needed
def get_model_classes():
    """Get the model classes (triggers actual import)."""
    from .modeling import (
        MambaInLlamaMambaForCausalLM,
        MambaInLlamaMambaForCausalLMNative,
        QwerkyLlamaMambaHybridForCausalLM,
    )
    return {
        "MambaInLlamaMambaForCausalLM": MambaInLlamaMambaForCausalLM,
        "MambaInLlamaMambaForCausalLMNative": MambaInLlamaMambaForCausalLMNative,
        "QwerkyLlamaMambaHybridForCausalLM": QwerkyLlamaMambaHybridForCausalLM,
    }
