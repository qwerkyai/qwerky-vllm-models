# Serving Framework Shootout: vLLM vs SGLang vs TensorRT-LLM

**Date**: 2026-02-24
**Goal**: Determine which production serving framework delivers the best throughput for hybrid SSM-Transformer models
**Motivation**: Evan's benchmarking paper showed hybrid models are 2.1x slower than Llama in vLLM at high concurrency. Chris's feedback and external evidence (SGLang 2-6x gains, Cartesia custom stack, NVIDIA TensorRT-LLM) suggest the bottleneck may be vLLM's architecture, not SSMs themselves.
**Hardware**: RunPod 1x H100 80GB
**Timeline**: ASAP (target: 4-5 hours benchmarking + 1 hour analysis)

---

## Background

### The Problem

Evan's paper "Mamba-Hybrid vs. Transformer Performance in vLLM" found:
- At c=100: Llama achieves 2.1x the output throughput of the best hybrid
- Hybrid throughput plateaus near 800-880 tok/s while Llama scales to 1,857 tok/s
- Prefill is 17-18x slower for hybrids at ISL=4096
- The advantage observed in HuggingFace Transformers (3x faster) disappears in vLLM

### The Question

Is this a **vLLM problem** or an **architecture problem**?

Evidence that it's a framework problem:
1. SGLang/PyTorch blog documents 2-6x throughput advantages for hybrids over transformers at high concurrency
2. NVIDIA's Nemotron Nano 2 reports up to 6x higher throughput vs Qwen3-8B on A10G
3. Bamba-9B team measured 2-2.5x throughput improvement over Llama-3.1-8B in vLLM itself
4. Cartesia (Mamba creators) achieves sub-90ms latency with custom inference stack
5. SGLang has purpose-built SSM infrastructure: dual memory pools, MambaRadixCache, elastic allocation, PD disaggregation

### Leadership Priorities

1. Are there quick wins in vLLM? (Probably exhausted)
2. Does TensorRT-LLM solve this problem?
3. Does SGLang solve this problem?
4. Focus on IMPACT, not fiddling with vLLM

---

## Test Matrix

| Model | vLLM | SGLang | TensorRT-LLM |
|-------|------|--------|--------------|
| Llama-3.1-8B-Instruct | Run on H100 | Run on H100 | Run on H100 |
| Nemotron-Nano-9B-v2 | Run on H100 | Run on H100 | Run on H100 |

**6 experiments total**: 2 models x 3 frameworks, all on the same H100.

Note: Evan's existing data is on RTX PRO 6000 Blackwell. We re-run vLLM on H100 for apples-to-apples comparison.

### Models

| Model | Architecture | Layers | SSM Variant | Why |
|-------|-------------|--------|------------|-----|
| Llama-3.1-8B-Instruct | Pure Transformer | 32 attn | N/A | Baseline — all frameworks optimize this well |
| Nemotron-Nano-9B-v2 | Mamba-2 Hybrid | 52 SSM + 4 attn | Mamba-2 | Supported on all 3 frameworks out-of-box |

### Benchmark Suite (Matching Evan's Paper)

**Experiment 1: Batch=1 Decode Latency**
- ISL = {512, 1024, 2048, 4096}
- OSL = 200
- max_num_seqs = 1
- 1 warmup + 5 timed generations
- Metric: tokens/s decode, prefill latency (ms)

**Experiment 2: Serving Throughput Under Load**
- Concurrency c = {1, 10, 25, 50, 100}
- ISL = 2048, OSL = 200
- Streaming completions
- Metrics: output tok/s, request throughput, ITL p50, TTFT p50

---

## Framework Setup

### vLLM (already installed)

```bash
# Version: 0.14.0 (already on RunPod)
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 --port 8000
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --max-model-len 4096 --port 8000
```

Benchmark: `vllm bench serve --base-url http://localhost:8000 ...`

### SGLang

```bash
pip install "sglang[all]>=0.4.6.post2"

# Serve
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 8000
python -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8000
```

Benchmark: `python -m sglang.bench_serving --backend sglang --port 8000 ...`

### TensorRT-LLM

```bash
apt-get -y install libopenmpi-dev
pip install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com

# Serve (OpenAI-compatible)
trtllm-serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --backend pytorch --port 8000
trtllm-serve meta-llama/Llama-3.1-8B-Instruct --backend pytorch --port 8000
```

Benchmark: `trtllm-bench throughput ...` or use same OpenAI-compatible client as others.

---

## Execution Order

Sequenced for speed on single GPU:

| Step | Framework | Model | Est. Time |
|------|-----------|-------|-----------|
| 0 | All | Install SGLang + TensorRT-LLM | 20 min |
| 1 | vLLM | Llama-3.1-8B | 20 min |
| 2 | vLLM | Nemotron-Nano-9B | 20 min |
| 3 | SGLang | Llama-3.1-8B | 20 min |
| 4 | SGLang | Nemotron-Nano-9B | 20 min |
| 5 | TRT-LLM | Llama-3.1-8B | 30 min |
| 6 | TRT-LLM | Nemotron-Nano-9B | 30 min |
| 7 | Analysis | Write comparison report | 30 min |

**Total: ~3-4 hours**

---

## Analysis Framework

### Primary Metric: Hybrid-to-Transformer Ratio

For each framework, compute: `Nemotron throughput / Llama throughput` at c=100.

| Framework | Ratio | Interpretation |
|-----------|-------|---------------|
| vLLM | 0.47x (from Evan's paper) | Hybrid is 2.1x slower |
| SGLang | ? | Target: >1.0x (hybrid faster) |
| TRT-LLM | ? | Target: >1.0x (hybrid faster) |

If SGLang or TRT-LLM shows ratio >0.8x, the gap is framework-level and fixable.
If all three show ratio ~0.5x, the gap is architectural and requires kernel work.

### Secondary Metrics

- Prefill scaling: Does the 17-18x penalty at ISL=4096 persist across frameworks?
- ITL under load: Which framework maintains lowest per-user latency at c=100?
- TTFT under load: SGLang claims dramatically better TTFT via PD disaggregation

---

## Decision Matrix

### If SGLang wins (Nemotron ratio >0.8x on SGLang vs <0.5x on vLLM)

**Action**: Build SGLang connector for Qwerky
- Port Qwerky model as SGLang plugin (similar to vLLM plugin architecture)
- Leverage SGLang's MambaRadixCache, dual memory pools, PD disaggregation
- Keep vLLM connector as fallback
- Estimate: 1-2 weeks for initial port

### If TensorRT-LLM wins

**Action**: Investigate NVIDIA NIM container path
- More effort for custom models, but NVIDIA relationship helps
- Potentially partner with NVIDIA on hybrid model optimization
- Consider NIM as production deployment path
- Estimate: 2-4 weeks (more complex model integration)

### If no framework significantly outperforms vLLM

**Action**: The bottleneck is truly architectural
- Kernel-level optimization: fused batched selective_scan
- Mamba-2 migration evaluation becomes urgent
- Consider Cartesia approach: custom inference stack
- Return to 4x speedup plan Phase 2+

### If all frameworks show similar improvement over vLLM

**Action**: Either SGLang or TRT-LLM could work
- Choose based on ecosystem fit: SGLang for open-source community, TRT-LLM for enterprise/NVIDIA
- Port to whichever has better SSM infrastructure

---

## Team Delegation

While Alex runs the shootout:

| Person | Task | Rationale |
|--------|------|-----------|
| Polina | Continue Mamba mixer verification + accuracy benchmarks | Her current focus, confirmed mixer correctness |
| Demeter | Profile decode anomalies he found | His current focus, may find optimization targets |
| Derek | Re-run Nemotron with reasoning off, clean tickets | His action items from standup |
| Arkadii | Check if MagPie runs in vLLM | Evan's ask from standup |

---

## Deliverables

1. `docs/reports/2026-02-24-framework-shootout.md` — Full comparison report
2. Summary table for Slack/email — framework recommendation with data
3. Updated CLAUDE.md — new project direction based on findings
4. Qwerky porting estimate — if winner != vLLM, effort to port
5. Raw benchmark data in `benchmark_results/framework-shootout/`
