# Qwerky vLLM Models

vLLM plugin for serving Qwerky AI's MambaInLlama hybrid models without the `--trust-remote-code` flag.

**Zero extra dependencies!** Uses vLLM's native Mamba ops - no `mamba_ssm` or `causal_conv1d` compilation required.

## Installation

```bash
pip install vllm qwerky-vllm-models
```

That's it! No compilation, no CUDA version conflicts.

## Usage

After installing, serve Qwerky models with vLLM directly:

```bash
# No --trust-remote-code needed!
vllm serve QwerkyAI/Qwerky-Llama3.1-Mamba-8B-Llama3.3-70B-base-distill-sft --max-model-len 4096
```

The plugin automatically registers the model architectures with vLLM on import.

## Supported Models

- `QwerkyAI/Qwerky-Llama3.1-Mamba-8B-Llama3.3-70B-base-distill-sft` (8B, instruction-tuned)
- `QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill` (3B, base)

## How It Works

This package uses vLLM's plugin system (`vllm.general_plugins` entry point) to register the MambaInLlama model architecture when the package is installed. This means:

1. No fork of vLLM needed
2. No `--trust-remote-code` flag required
3. Works with standard vLLM installation
4. Uses vLLM's native Triton-accelerated Mamba kernels

## Requirements

- Python >= 3.10
- vLLM >= 0.14.0
- PyTorch >= 2.0.0

## License

Apache 2.0
