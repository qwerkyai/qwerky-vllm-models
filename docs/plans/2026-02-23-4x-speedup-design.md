# Qwerky 3B: Phased 4x Speedup Design

**Date**: 2026-02-23
**Goal**: Match or beat Llama 3.2 3B throughput AND latency across all concurrency levels
**Target GPUs**: H100 (primary), A100 (secondary)
**Approach**: Phased — quick serving-side wins first, evaluate Mamba-2 migration later

---

## Current Performance Baseline (v0.2.67, H100)

| Metric | Qwerky 3B | Llama 3.2 3B | Ratio |
|--------|-----------|-------------|-------|
| BS=1 decode (TPOT) | 3.40 ms | 3.28 ms | 1.04x slower |
| BS=1 TTFT (in=256) | 33 ms | 20 ms | 1.62x slower |
| BS=32 throughput | 3,225 tok/s | 4,290 tok/s | 0.75x |
| BS=500 throughput | 4,860 tok/s | 8,458 tok/s | 0.57x |
| P50 ITL @ BS=500 | 14 ms | 27 ms | 0.52x (Qwerky wins) |

**Root causes of throughput gap**:
1. Mamba SSM is sequential (state at t depends on t-1) — can't parallelize across positions
2. 82% of layers are Mamba (28/34) — only 6 attention layers batch-parallelize
3. causal_conv1d kernel processes one sequence at a time — no cross-sequence parallelism during prefill
4. Mamba-1 selective_scan achieves only 10-15% tensor core utilization

---

## Phase 0: Multi-File Restructure (Foundation, 2-3 days)

Convert `modeling.py` (1461 lines) into a `modeling/` package for maintainability.

**Confirmed**: vLLM uses `importlib.import_module()` — a package with `__init__.py` re-exports is fully compatible. Registration strings stay the same.

```
qwerky_vllm_models/
  __init__.py          (unchanged)
  configuration.py     (unchanged)
  modeling/
    __init__.py        (re-exports all public classes)
    mixer.py           (~500 lines: MambaInLlamaMambaMixer, custom op, SSM logic)
    layers.py          (~200 lines: MambaDecoderLayer, MLP, RMSNormFallback)
    model.py           (~400 lines: Backbone, ForCausalLM, load_weights)
    kernels/           (new optimized kernels)
      __init__.py
      triton_conv1d.py
      quantized.py
```

**Deliverable**: v0.2.68 — multi-file restructure, zero functional change, smoke test passes.

---

## Phase 1: Drop-in Kernel Improvements (Weeks 1-2, ~1.3x)

### 1a. Triton causal_conv1d (vLLM PR #18218)

Replace CUDA `causal_conv1d_fn` with Triton implementation.
- 5% token throughput + 11% TTFT improvement
- Fixes chunked prefill performance degradation
- Drop-in: same function signature, different backend
- Lives in `modeling/kernels/triton_conv1d.py`

### 1b. Split prefill/decode for conv1d (vLLM PR #17146)

CUDA conv1d degrades when mixing prefill+decode in same batch. Split by request type:
- Before conv1d: separate batch into prefill vs decode tensors
- Run conv1d on each separately
- Merge results back
- Significant throughput improvement on real serving workloads

### 1c. Eliminate unnecessary memory copies (vLLM PRs #14778, #14857)

Mamba2 PRs that fix unnecessary memory copies during prefill. Adapt for our Mamba-1 code:
- Review our prefill path for redundant `.contiguous()` calls
- Check for tensor copies that can be replaced with views
- Profile before/after to quantify

**Expected cumulative gain**: ~1.3x over current → ~0.73x vs Llama (up from 0.57x)

---

## Phase 2: Quantization (Weeks 2-4, ~2-2.5x cumulative)

### 2a. FP8 for linear projections (H100) / INT8 (A100)

Target all standard GEMMs while keeping SSM state in float32:

| Layer | Projection | Quantize? |
|-------|-----------|-----------|
| Mamba mixer | in_proj [3072→8384] | Yes — FP8/INT8 |
| Mamba mixer | out_proj [3072→3072] | Yes — FP8/INT8 |
| Mamba mixer | dt_proj [192→3072] | Yes — FP8/INT8 |
| MHA (Llama) | qkv_proj, o_proj | Yes — via vLLM quant_config (free from v0.2.67) |
| MLP | gate_up_proj, down_proj | Yes — FP8/INT8 |
| SSM state | A, D, delta_bias, conv/ssm state | NO — keep float32 |

- ~2x speedup on GEMMs for Hopper (FP8), ~1.5x for Ampere (INT8)
- vLLM already has `FP8LinearMethod` infrastructure
- LlamaDecoderLayer gets quant_config for free

### 2b. Quamba2 W8A8 full quantization

- Quantizes entire model including SSM parameters
- 1.3x prefill speedup, 3x generation speedup, 4x memory reduction
- ~1.6% accuracy drop (paper's numbers)
- Cluster-aware weight reordering for SSM heads
- Works with Mamba-1

### Accuracy checkpoint

After Phase 2, re-run lm_eval on all 8 tasks:
- If accuracy drop > 2% on any core task: back off to FP8-only (skip Quamba2)
- If GSM8K drops below 0.40 (from 0.46): investigate selective quantization

**Expected cumulative gain**: ~2-2.5x over current → ~1.1-1.4x vs Llama

---

## Phase 3: Speculative Decoding + Smart Caching (Weeks 4-8, ~3x cumulative)

### 3a. Mamba drafter for speculative decoding

- Small Mamba-only model as draft model (fast constant-memory generation)
- 3x faster drafting at 8K context vs transformer drafters
- MambaInLlama paper's hardware-aware spec decode: 300+ tok/s
- Primarily helps **generation throughput** (decode phase)
- Options for drafter model:
  - Distill a small (300M-500M) Mamba-only model from our 3B
  - Use existing small Mamba model from MambaInLlama repo
  - Self-speculative: use Mamba layers only (skip MHA) as cheap approximation

### 3b. MARCONI-style prefix caching

- Replace basic LRU eviction with FLOP-aware eviction policy
- Considers compute savings (not just recency) for cache decisions
- Up to 34.4x higher token hit rates for hybrid models
- Particularly valuable because SSM state recomputation is expensive
- Implementation: custom eviction policy in our plugin's state management

**Expected cumulative gain**: ~3x over current → ~1.7x vs Llama

---

## Phase 4: Evaluate Mamba-2 Migration (Month 2+, 4x+ potential)

**Decision point** after Phase 3: How close are we to 4x?

### What Mamba-2 SSD unlocks

| Feature | Mamba-1 (current) | Mamba-2 SSD |
|---------|-------------------|-------------|
| Tensor core utilization | 10-15% | 80-90% |
| All-reduces per TP layer | 2 | 1 |
| Fused kernel availability | Limited | IBM 6x, ThunderKittens 3x+ |
| FP8 training | Not available | Nemotron-H proved it works |
| Prefill parallelism | Sequential scan | Semi-parallel chunks |

### What it costs

- Full retraining or distillation (70B teacher → 3B Mamba-2 student)
- Architecture changes in config, modeling, weight loading
- 1-2 month timeline
- Need to re-validate all accuracy benchmarks

### Reference points

- Bamba-9B (IBM, Mamba-2): 2-2.5x throughput vs comparable transformers
- Nemotron-H-56B (NVIDIA, Mamba-2 + MoE): Up to 3x faster, FP8 pre-trained
- Jamba-1.5 (AI21, Mamba + MoE): Up to 2.5x faster inference

---

## Key vLLM PRs to Backport

| PR | Description | Phase |
|----|------------|-------|
| #18218 | Triton causal_conv1d | Phase 1 |
| #17146 | Split prefill/decode for conv1d | Phase 1 |
| #14778/#14857 | Fix unnecessary memory copies | Phase 1 |
| #16942 | Mamba2 SSD chunked prefill refactor | Phase 4 |
| #17140 | RFC: Native SSM support in vLLM V1 | Watch |

## Key Papers

| Paper | Relevance |
|-------|-----------|
| Quamba2 (arxiv 2503.22879) | W8A8 quantization for Mamba-1 |
| MARCONI (arxiv 2411.19379) | Prefix caching for hybrid LLMs |
| MambaInLlama (arxiv 2408.15237) | Speculative decoding with Mamba |
| Mamba Drafters (arxiv 2506.01206) | Mamba as spec decode drafter |
| Nemotron-H (arxiv 2504.03624) | FP8 hybrid model reference |

---

## Success Criteria

| Phase | Target | Metric |
|-------|--------|--------|
| Phase 1 | 1.3x current | BS=500 throughput > 6,300 tok/s |
| Phase 2 | 2-2.5x current | BS=500 throughput > 9,700 tok/s |
| Phase 3 | 3x current | BS=500 throughput > 14,500 tok/s |
| Phase 4 | 4x current | BS=500 throughput > 19,400 tok/s |
| All phases | Accuracy | GSM8K > 0.40, MMLU > 0.41 |
