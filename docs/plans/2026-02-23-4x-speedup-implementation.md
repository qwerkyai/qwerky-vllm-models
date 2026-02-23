# Qwerky 3B: 4x Speedup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Qwerky 3B hybrid Mamba model match or beat Llama 3.2 3B throughput and latency at all batch sizes.

**Architecture:** Phased approach — multi-file restructure for maintainability, then FP8 quantization for Mamba layers (the biggest actionable win), then memory copy elimination, then speculative decoding + MARCONI caching. Mamba-2 migration evaluated after Phase 3.

**Tech Stack:** vLLM 0.14.0, PyTorch 2.6, Triton, CUDA, FP8 (Hopper), INT8 (Ampere)

**Key discovery from investigation:** Triton causal_conv1d (PR #18218) and split prefill/decode (PR #17146) are **already implemented** in our code. The biggest actionable win is wiring FP8 quantization through Mamba layers — currently only MHA layers (LlamaDecoderLayer) get quant_config.

---

## Task 1: Multi-File Restructure (v0.2.68)

Split `modeling.py` (1462 lines) into a `modeling/` package. Zero functional change.

**Files:**
- Delete: `qwerky_vllm_models/modeling.py`
- Create: `qwerky_vllm_models/modeling/__init__.py`
- Create: `qwerky_vllm_models/modeling/mixer.py`
- Create: `qwerky_vllm_models/modeling/layers.py`
- Create: `qwerky_vllm_models/modeling/model.py`

### Step 1: Create the modeling/ package directory

```bash
cd /workspace/qwerky-vllm-models
mkdir -p qwerky_vllm_models/modeling
```

### Step 2: Create mixer.py (lines 1-201 imports + lines 232-873 mixer + custom op)

Extract from `modeling.py`:
- All imports (lines 1-201) — each file will need its own subset
- `_load_mamba_config()` function (lines 38-66)
- `MambaInLlamaMambaMixer` class (vLLM version, lines 238-733)
- `MambaInLlamaMambaMixer` class (fallback version, lines 736-844)
- Custom op registration (lines 847-872)
- Logger setup

File should be ~600 lines. Include all imports needed by the mixer classes.

### Step 3: Create layers.py (lines 204-230 + 875-962)

Extract from `modeling.py`:
- `RMSNormFallback` class (lines 208-226)
- RMSNorm assignment (lines 228-229)
- `MLP` class (lines 879-913)
- `MambaDecoderLayer` class (lines 920-962)

File should be ~150 lines. Import `MambaInLlamaMambaMixer` from `.mixer`.

### Step 4: Create model.py (lines 965-1462)

Extract from `modeling.py`:
- `MambaInLlamaMambaModel` class (lines 969-1044)
- `MambaInLlamaMambaForCausalLMNative` class (lines 1062-1450)
- Architecture aliases (lines 1458-1461)

File should be ~500 lines. Import from `.mixer` and `.layers`.

### Step 5: Create modeling/__init__.py with re-exports

```python
"""Qwerky vLLM Models - Modeling Package.

Re-exports all public classes for backward compatibility.
vLLM's ModelRegistry loads via importlib.import_module("qwerky_vllm_models.modeling")
then getattr(mod, "ClassName"), so all classes must be accessible here.
"""
from .mixer import MambaInLlamaMambaMixer  # noqa: F401
from .layers import MambaDecoderLayer, MLP, RMSNorm  # noqa: F401
from .model import (  # noqa: F401
    MambaInLlamaMambaModel,
    MambaInLlamaMambaForCausalLMNative,
    MambaInLlamaMambaForCausalLM,
    QwerkyLlamaMambaHybridForCausalLM,
)
```

### Step 6: Delete old modeling.py

```bash
git rm qwerky_vllm_models/modeling.py
```

### Step 7: Bump version to 0.2.68

Update both files:
- `pyproject.toml`: `version = "0.2.68"`
- `qwerky_vllm_models/__init__.py`: `__version__ = "0.2.68"`

### Step 8: Build and smoke test

```bash
cd /workspace/qwerky-vllm-models && rm -rf dist/ && python -m build
pip install --force-reinstall dist/qwerky_vllm_models-0.2.68-py3-none-any.whl
python -c "from qwerky_vllm_models.modeling import MambaInLlamaMambaForCausalLM; print('Import OK')"
```

### Step 9: Serve and verify

```bash
vllm serve QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill --max-model-len 4096 --port 8100
# In another terminal:
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill","prompt":"The capital of France is","max_tokens":20}'
```

Expected: 343/343 weights load, coherent output, no errors.

### Step 10: Commit

```bash
git add -A && git commit -m "v0.2.68: split modeling.py into modeling/ package"
```

---

## Task 2: Wire FP8/Quantization Through Mamba Layers (v0.2.69)

Currently `quant_config` is only passed to LlamaDecoderLayer (MHA layers). Mamba mixer and MLP layers in MambaDecoderLayer don't receive it. This means FP8 quantization only applies to 6/34 layers.

**Files:**
- Modify: `qwerky_vllm_models/modeling/mixer.py`
- Modify: `qwerky_vllm_models/modeling/layers.py`
- Modify: `qwerky_vllm_models/modeling/model.py`

### Step 1: Update MambaInLlamaMambaMixer.__init__() to accept quant_config

In `mixer.py`, add `quant_config` parameter and pass to all linear layers:

```python
# In __init__ signature, add:
#   quant_config=None,

# Then pass to each linear layer:
self.in_proj = MergedColumnParallelLinear(
    self.d_model,
    [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank],
    bias=False,
    prefix=f"{prefix}.in_proj",
    quant_config=quant_config,  # ADD THIS
)

self.dt_proj = ColumnParallelLinear(
    self.dt_rank,
    self.d_inner,
    bias=True,
    prefix=f"{prefix}.dt_proj",
    quant_config=quant_config,  # ADD THIS
)

self.out_proj = RowParallelLinear(
    self.d_inner,
    self.d_model,
    bias=False,
    input_is_parallel=True,
    prefix=f"{prefix}.out_proj",
    quant_config=quant_config,  # ADD THIS
)

# NOTE: conv1d should NOT get quant_config — it's a tiny depthwise conv,
# quantizing it could hurt accuracy with no speed gain
```

### Step 2: Update MLP.__init__() to accept quant_config

In `layers.py`:

```python
# In MLP.__init__ signature, add:
#   quant_config=None,

self.gate_up_proj = MergedColumnParallelLinear(
    d_model,
    [intermediate_size, intermediate_size],
    bias=False,
    prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
    quant_config=quant_config,  # ADD THIS
)
self.down_proj = RowParallelLinear(
    intermediate_size, d_model, bias=False,
    input_is_parallel=True,
    prefix=f"{prefix}.down_proj" if prefix else "down_proj",
    quant_config=quant_config,  # ADD THIS
)
```

### Step 3: Update MambaDecoderLayer.__init__() to pass quant_config through

In `layers.py`:

```python
# In MambaDecoderLayer.__init__ signature, add:
#   quant_config=None,

self.mamba = MambaInLlamaMambaMixer(
    ...,
    quant_config=quant_config,  # ADD THIS
)
self.mlp = MLP(
    ...,
    quant_config=quant_config,  # ADD THIS
)
```

### Step 4: Update MambaInLlamaMambaModel.__init__() to extract and pass quant_config

In `model.py`:

```python
# In MambaInLlamaMambaModel.__init__:
quant_config = getattr(vllm_config, 'quant_config', None)

# When creating MambaDecoderLayer:
self.layers.append(MambaDecoderLayer(
    ...,
    quant_config=quant_config,  # ADD THIS
))
```

### Step 5: Build, install, and test WITHOUT quantization (regression test)

```bash
cd /workspace/qwerky-vllm-models && rm -rf dist/ && python -m build
pip install --force-reinstall dist/qwerky_vllm_models-0.2.69-py3-none-any.whl
vllm serve QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill --max-model-len 4096 --port 8100
```

Verify: still loads 343/343 weights, coherent output. quant_config=None should be a no-op.

### Step 6: Test WITH FP8 quantization (on H100)

```bash
vllm serve QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill \
  --max-model-len 4096 --port 8100 \
  --quantization fp8
```

Verify:
- Model loads (may show FP8 weight conversion messages)
- Coherent output
- Run speed benchmark: `vllm bench serve --base-url http://localhost:8100 --model QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill --num-prompts 200 --input-len 1024 --output-len 128 --request-rate inf --max-concurrency 32`
- Compare throughput vs v0.2.67 baseline (3,225 tok/s at BS=32)

### Step 7: Run accuracy benchmark with FP8

```bash
lm_eval --model local-completions \
  --model_args model=QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill,base_url=http://localhost:8100/v1/completions,num_concurrent=32,tokenizer_backend=huggingface \
  --tasks gsm8k,mmlu,hellaswag \
  --batch_size auto
```

Acceptance criteria: GSM8K > 0.40, MMLU > 0.41 (max 2% drop from baseline).

### Step 8: Commit

```bash
git add -A && git commit -m "v0.2.69: wire quant_config through Mamba layers for FP8 support"
```

---

## Task 3: Eliminate Unnecessary Memory Copies (v0.2.70)

Audit the forward_cuda() method for redundant tensor operations. Based on vLLM PRs #14778 and #14857.

**Files:**
- Modify: `qwerky_vllm_models/modeling/mixer.py`

### Step 1: Profile current forward pass

Write a micro-benchmark script that measures time spent in each section of forward_cuda():
- in_proj matmul
- tensor splitting after in_proj
- conv1d (prefill + decode)
- selective_scan (prefill) / selective_state_update (decode)
- out_proj matmul
- result merging

### Step 2: Audit for unnecessary .contiguous() calls

Search for `.contiguous()` in mixer.py. Each one forces a memory copy. Check if:
- The downstream operation actually requires contiguous memory
- A view/reshape could work instead

### Step 3: Audit for unnecessary transposes

Check all `.transpose()` and `rearrange()` calls. Mamba ops have specific layout expectations:
- `causal_conv1d_fn` prefill: (dim, tokens) layout
- `causal_conv1d_update` decode: (batch, dim) layout
- `selective_scan_fn`: (batch, dim, seq_len) layout

Eliminate any transpose that's immediately followed by another transpose.

### Step 4: Check tensor creation in hot path

Look for `torch.zeros()`, `torch.empty()`, `torch.cat()`, `torch.stack()` in forward_cuda().
- Replace `torch.cat()` with pre-allocated buffers where possible
- Use `out=` parameter for in-place operations

### Step 5: Benchmark after changes

Re-run the same speed benchmarks from v0.2.67 baseline:
```bash
vllm bench serve --base-url http://localhost:8100 --model QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill \
  --num-prompts 500 --input-len 1024 --output-len 128 --request-rate inf
```

### Step 6: Commit

```bash
git add -A && git commit -m "v0.2.70: eliminate unnecessary memory copies in Mamba forward pass"
```

---

## Task 4: Quamba2 W8A8 Quantization Investigation (v0.2.71)

Investigate and prototype Quamba2 W8A8 quantization for the full model.

**Files:**
- Create: `qwerky_vllm_models/modeling/kernels/quamba2.py`
- Modify: `qwerky_vllm_models/modeling/mixer.py`

### Step 1: Read the Quamba2 paper

Fetch and analyze: https://arxiv.org/html/2503.22879

Key questions:
- What is "cluster-aware weight reordering" for SSM heads?
- How does it handle the float32 SSM state (A, D, delta_bias)?
- What's the exact accuracy/speed tradeoff at W8A8?

### Step 2: Check if Quamba2 has a public implementation

Search GitHub for Quamba2 code. If available:
- Fork/vendor the quantization scripts
- Apply to our model checkpoint
- Measure accuracy delta

### Step 3: If no public implementation, prototype selective W8A8

Apply INT8 quantization to linear projections only (not SSM state):
- Use `torch.ao.quantization` or vLLM's existing INT8 path
- Keep A, D, delta_bias, conv state, SSM state in float32
- This is safer than full Quamba2 but still gives significant GEMM speedup

### Step 4: Accuracy benchmark

Run full lm_eval suite. If accuracy is acceptable, commit and ship.

### Step 5: Commit

```bash
git add -A && git commit -m "v0.2.71: Quamba2-inspired W8A8 quantization for Mamba layers"
```

---

## Task 5: Speculative Decoding with Mamba Drafter (v0.2.72+)

Use a small Mamba model as speculative decoding drafter.

**Files:**
- Research-only initially — depends on vLLM speculative decoding infrastructure for hybrid models

### Step 1: Check vLLM speculative decoding support for SSM models

```bash
grep -r "speculative" /usr/local/lib/python3.12/dist-packages/vllm/spec_decode/ | head -20
```

Key question: Does vLLM's spec decode framework work with HasInnerState models?

### Step 2: Identify drafter model candidates

Options:
1. MambaInLlama repo small models (check if they exist)
2. Distill a small Mamba-only model from our 3B
3. Self-speculative: skip MHA layers during drafting (use Mamba layers only)

### Step 3: Prototype self-speculative decoding

The cheapest approach: during draft phase, run only Mamba layers (skip the 6 MHA layers at positions [3, 8, 13, 18, 23, 27]). This gives approximate logits at ~82% the cost (28/34 layers) with much faster KV-cache-free decoding.

### Step 4: Benchmark

Measure tokens/second with speculative decoding vs without. Target: 2-3x generation speedup.

---

## Task 6: MARCONI Prefix Caching (v0.2.73+)

Implement FLOP-aware cache eviction for hybrid Mamba-attention models.

**Files:**
- Research-heavy — may require changes to vLLM's cache management

### Step 1: Read MARCONI paper

Fetch: https://arxiv.org/abs/2411.19379

Key concepts:
- FLOP-aware eviction: evict entries that save the least compute to recompute
- Hybrid-specific: SSM state recomputation cost is O(n) per sequence (must rescan)
- Exact-match only: SSM in-place updates prevent partial cache reuse

### Step 2: Assess integration feasibility

Can we implement MARCONI's eviction policy within our plugin, or does it require vLLM core changes?

### Step 3: Prototype if feasible

Start with the eviction policy only (not the full MARCONI system).

---

## Task 7: Phase 4 Decision — Mamba-2 Migration Evaluation

After Tasks 1-6, assess:

1. Current throughput vs Llama baseline at BS=32 and BS=500
2. Accuracy vs v0.2.67 baseline
3. Gap to 4x target
4. If gap > 30%, evaluate Mamba-2 SSD migration:
   - Cost: full retraining (1-2 months)
   - Gain: fused SSD kernels (6x on SSM), 80-90% tensor core utilization, single all-reduce TP
   - Reference: Bamba-9B achieves 2.5x throughput vs transformers

---

## Benchmark Checkpoints

After each task, run and record:

```bash
# Speed (BS=1, BS=32, BS=500)
vllm bench serve --base-url http://localhost:8100 \
  --model QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill \
  --num-prompts 50 --input-len 1024 --output-len 128 --request-rate inf --max-concurrency 1

vllm bench serve --base-url http://localhost:8100 \
  --model QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill \
  --num-prompts 200 --input-len 1024 --output-len 128 --request-rate inf --max-concurrency 32

vllm bench serve --base-url http://localhost:8100 \
  --model QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill \
  --num-prompts 500 --input-len 1024 --output-len 128 --request-rate inf

# Accuracy (quick check)
lm_eval --model local-completions \
  --model_args model=QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill,base_url=http://localhost:8100/v1/completions,num_concurrent=32,tokenizer_backend=huggingface \
  --tasks gsm8k,mmlu,hellaswag --batch_size auto
```

## Baseline Numbers (v0.2.67, H100, no quantization)

| Metric | Value |
|--------|-------|
| BS=1 TPOT | 3.41 ms |
| BS=32 throughput | 3,225 tok/s |
| BS=500 throughput | 4,860 tok/s |
| GSM8K (flex) | 0.4647 |
| MMLU | 0.4369 |
| Hellaswag (norm) | 0.6275 |
