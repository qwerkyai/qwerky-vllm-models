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
"""Qwerky vLLM Models - Modeling Package.

Re-exports all public classes for backward compatibility.
vLLM's ModelRegistry loads via importlib.import_module("qwerky_vllm_models.modeling")
then getattr(mod, "ClassName"), so all classes must be accessible here.
"""
from .mixer import MambaInLlamaMambaMixer  # noqa: F401
from .layers import MambaDecoderLayer, Mamba2DecoderLayer, MLP, RMSNorm  # noqa: F401
from .model import (  # noqa: F401
    MambaInLlamaMambaModel,
    MambaInLlamaMambaForCausalLMNative,
    MambaInLlamaMambaForCausalLM,
    QwerkyLlamaMambaHybridForCausalLM,
    QwerkyMamba2HybridForCausalLM,
)
