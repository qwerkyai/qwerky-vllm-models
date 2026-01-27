# Qwerky vLLM Models

vLLM plugin for serving Qwerky AI's MambaInLlama hybrid models without the `--trust-remote-code` flag.

## Installation

```bash
pip install qwerky-vllm-models
```

## Usage

After installing, you can serve Qwerky models with vLLM directly:

```bash
# No --trust-remote-code needed!
vllm serve QwerkyAI/Qwerky-Llama3.1-Mamba-8B-Llama3.3-70B-base-distill-sft
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

## Requirements

- Python >= 3.10
- vLLM >= 0.14.0
- PyTorch >= 2.0.0
- mamba-ssm >= 2.0.0
- causal-conv1d >= 1.2.0

## License

Apache 2.0
