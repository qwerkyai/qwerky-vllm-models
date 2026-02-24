# Serving Framework Shootout: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Benchmark Nemotron-Nano-9B-v2 and Llama-3.1-8B on vLLM, SGLang, and TensorRT-LLM on the same H100, producing an apples-to-apples comparison to determine which framework delivers the best hybrid SSM throughput.

**Architecture:** Install all three frameworks on RunPod H100, run identical benchmark suites (batch=1 decode + serving throughput at c=1/10/25/50/100), collect metrics, write comparison report. Each framework serves via OpenAI-compatible API. Benchmarks use each framework's native bench tool for fairest comparison.

**Tech Stack:** vLLM 0.14.0, SGLang >=0.4.6, TensorRT-LLM (latest), RunPod H100 80GB, CUDA 12.x

---

## Task 1: Environment Setup

**Files:**
- Create: `/workspace/benchmark_results/framework-shootout/` (results directory)
- Create: `/workspace/benchmark_results/framework-shootout/bench.sh` (master benchmark script)

### Step 1: Verify current vLLM installation and GPU

```bash
python -c "import vllm; print(vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | grep release
```

Expected: vLLM 0.14.x, H100 80GB, CUDA 12.x.

### Step 2: Install SGLang

```bash
pip install "sglang[all]>=0.4.6.post2" --no-deps 2>&1 | tail -5
# If dependency conflicts, try:
# pip install "sglang[all]" --force-reinstall
```

Verify:
```bash
python -c "import sglang; print('SGLang OK')"
```

Note: SGLang may need `mamba_ssm` package for Nemotron. If not installed:
```bash
pip install mamba-ssm causal-conv1d
```

### Step 3: Install TensorRT-LLM

```bash
apt-get -y install libopenmpi-dev
pip install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com 2>&1 | tail -10
```

Verify:
```bash
python -c "import tensorrt_llm; print(f'TRT-LLM {tensorrt_llm.__version__}')"
```

**CUDA version warning**: TRT-LLM wheels may require CUDA 13.x. If install fails:
- Try `pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com`
- If still fails, fall back to Docker: `docker pull nvcr.io/nvidia/tensorrt-llm/release:latest`
- If Docker not available, skip TRT-LLM and note it in report. SGLang is the higher-priority comparison.

### Step 4: Create results directory

```bash
mkdir -p /workspace/benchmark_results/framework-shootout/{vllm,sglang,trtllm}/{llama,nemotron}
```

### Step 5: Download models (warm cache)

```bash
# Llama-3.1-8B-Instruct — may need HF token
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --quiet 2>&1 | tail -1

# Nemotron-Nano-9B-v2
huggingface-cli download nvidia/NVIDIA-Nemotron-Nano-9B-v2 --quiet 2>&1 | tail -1
```

If Llama is gated and token not set, use `unsloth/Llama-3.1-8B-Instruct` (ungated mirror) instead. Adjust model name in all subsequent commands.

### Step 6: Commit setup

```bash
cd /workspace/qwerky-vllm-models
git add docs/plans/2026-02-24-serving-framework-shootout*.md
git commit -m "Add serving framework shootout design and implementation plan"
```

---

## Task 2: vLLM Benchmarks (H100 Baseline)

We need fresh H100 numbers since Evan's data is from RTX PRO 6000 Blackwell.

**Files:**
- Create: `/workspace/benchmark_results/framework-shootout/vllm/llama/results.json`
- Create: `/workspace/benchmark_results/framework-shootout/vllm/nemotron/results.json`

### Step 1: Kill any existing vLLM processes

```bash
pkill -f "vllm serve" 2>/dev/null; sleep 2
nvidia-smi | grep python && echo "WARNING: GPU still in use" || echo "GPU clear"
```

### Step 2: Serve Llama-3.1-8B on vLLM

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 4096 --port 8000 \
  > /tmp/vllm-llama.log 2>&1 &
```

Wait for "Started server" in log. Smoke test:
```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Hello","max_tokens":5}' | python -m json.tool
```

### Step 3: Run vLLM Llama benchmarks

**Batch=1 decode** (ISL=512,1024,2048,4096):
```bash
for ISL in 512 1024 2048 4096; do
  echo "=== vLLM Llama BS=1 ISL=$ISL ==="
  vllm bench serve --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 10 --input-len $ISL --output-len 200 \
    --request-rate inf --max-concurrency 1 \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/vllm/llama/bs1.log
done
```

**Serving throughput** (c=1,10,25,50,100):
```bash
for C in 1 10 25 50 100; do
  echo "=== vLLM Llama c=$C ==="
  vllm bench serve --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 200 --input-len 2048 --output-len 200 \
    --request-rate inf --max-concurrency $C \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/vllm/llama/throughput.log
done
```

### Step 4: Stop Llama, serve Nemotron

```bash
pkill -f "vllm serve"; sleep 5

CUDA_VISIBLE_DEVICES=0 vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --max-model-len 4096 --port 8000 \
  > /tmp/vllm-nemotron.log 2>&1 &
```

Wait for server ready. Smoke test:
```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-Nano-9B-v2","prompt":"Hello","max_tokens":5}' | python -m json.tool
```

**Important**: Nemotron-Nano-9B-v2 may need `--enforce-eager` if CUDA graph capture fails for Mamba layers. Try without first; add if it crashes.

### Step 5: Run vLLM Nemotron benchmarks

**Batch=1 decode**:
```bash
for ISL in 512 1024 2048 4096; do
  echo "=== vLLM Nemotron BS=1 ISL=$ISL ==="
  vllm bench serve --base-url http://localhost:8000 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --num-prompts 10 --input-len $ISL --output-len 200 \
    --request-rate inf --max-concurrency 1 \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/vllm/nemotron/bs1.log
done
```

**Serving throughput**:
```bash
for C in 1 10 25 50 100; do
  echo "=== vLLM Nemotron c=$C ==="
  vllm bench serve --base-url http://localhost:8000 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --num-prompts 200 --input-len 2048 --output-len 200 \
    --request-rate inf --max-concurrency $C \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/vllm/nemotron/throughput.log
done
```

### Step 6: Stop vLLM

```bash
pkill -f "vllm serve"; sleep 5
```

---

## Task 3: SGLang Benchmarks

**Files:**
- Create: `/workspace/benchmark_results/framework-shootout/sglang/llama/results.json`
- Create: `/workspace/benchmark_results/framework-shootout/sglang/nemotron/results.json`

### Step 1: Serve Llama-3.1-8B on SGLang

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --mem-fraction-static 0.85 \
  > /tmp/sglang-llama.log 2>&1 &
```

Wait for server ready (check log). Smoke test:
```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Hello","max_tokens":5}' | python -m json.tool
```

### Step 2: Run SGLang Llama benchmarks

**Batch=1 decode** (ISL=512,1024,2048,4096):
```bash
for ISL in 512 1024 2048 4096; do
  echo "=== SGLang Llama BS=1 ISL=$ISL ==="
  python -m sglang.bench_serving \
    --backend sglang \
    --port 8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --random-input-len $ISL --random-output-len 200 \
    --num-prompts 10 \
    --request-rate inf --max-concurrency 1 \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/sglang/llama/bs1.log
done
```

**Serving throughput** (c=1,10,25,50,100):
```bash
for C in 1 10 25 50 100; do
  echo "=== SGLang Llama c=$C ==="
  python -m sglang.bench_serving \
    --backend sglang \
    --port 8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 200 \
    --num-prompts 200 \
    --request-rate inf --max-concurrency $C \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/sglang/llama/throughput.log
done
```

### Step 3: Stop Llama, serve Nemotron on SGLang

```bash
pkill -f "sglang.launch_server"; sleep 5

python -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --port 8000 \
  --mem-fraction-static 0.85 \
  > /tmp/sglang-nemotron.log 2>&1 &
```

Wait for server ready. Smoke test.

**If Nemotron fails to load on SGLang**: Check error. Common issues:
- `ModuleNotFoundError: mamba_ssm` → `pip install mamba-ssm causal-conv1d`
- Architecture not recognized → Check if SGLang version supports NemotronH (needs >=0.5.4)
- OOM → Add `--max-total-tokens 4096`

### Step 4: Run SGLang Nemotron benchmarks

**Batch=1 decode**:
```bash
for ISL in 512 1024 2048 4096; do
  echo "=== SGLang Nemotron BS=1 ISL=$ISL ==="
  python -m sglang.bench_serving \
    --backend sglang \
    --port 8000 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --dataset-name random \
    --random-input-len $ISL --random-output-len 200 \
    --num-prompts 10 \
    --request-rate inf --max-concurrency 1 \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/sglang/nemotron/bs1.log
done
```

**Serving throughput**:
```bash
for C in 1 10 25 50 100; do
  echo "=== SGLang Nemotron c=$C ==="
  python -m sglang.bench_serving \
    --backend sglang \
    --port 8000 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 200 \
    --num-prompts 200 \
    --request-rate inf --max-concurrency $C \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/sglang/nemotron/throughput.log
done
```

### Step 5: Stop SGLang

```bash
pkill -f "sglang.launch_server"; sleep 5
```

---

## Task 4: TensorRT-LLM Benchmarks

**Files:**
- Create: `/workspace/benchmark_results/framework-shootout/trtllm/llama/results.json`
- Create: `/workspace/benchmark_results/framework-shootout/trtllm/nemotron/results.json`

**Note**: TRT-LLM install may fail due to CUDA version. If so, skip this task and note in report. SGLang comparison is higher priority.

### Step 1: Serve Llama-3.1-8B on TRT-LLM

```bash
trtllm-serve meta-llama/Llama-3.1-8B-Instruct \
  --backend pytorch \
  --host 0.0.0.0 --port 8000 \
  --max_seq_len 4096 \
  > /tmp/trtllm-llama.log 2>&1 &
```

Wait for server ready. Smoke test same as above.

**If trtllm-serve not available** (older version), try:
```bash
python -m tensorrt_llm.commands.serve meta-llama/Llama-3.1-8B-Instruct \
  --backend pytorch --host 0.0.0.0 --port 8000
```

### Step 2: Run TRT-LLM Llama benchmarks

Since TRT-LLM exposes OpenAI-compatible API, we can use SGLang's bench_serving as a universal client:

```bash
for ISL in 512 1024 2048 4096; do
  echo "=== TRT-LLM Llama BS=1 ISL=$ISL ==="
  python -m sglang.bench_serving \
    --backend gserver \
    --port 8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --random-input-len $ISL --random-output-len 200 \
    --num-prompts 10 \
    --request-rate inf --max-concurrency 1 \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/trtllm/llama/bs1.log
done
```

```bash
for C in 1 10 25 50 100; do
  echo "=== TRT-LLM Llama c=$C ==="
  python -m sglang.bench_serving \
    --backend gserver \
    --port 8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 200 \
    --num-prompts 200 \
    --request-rate inf --max-concurrency $C \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/trtllm/llama/throughput.log
done
```

### Step 3: Stop Llama, serve Nemotron on TRT-LLM

```bash
pkill -f "trtllm-serve"; sleep 5

trtllm-serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --backend pytorch \
  --host 0.0.0.0 --port 8000 \
  --max_seq_len 4096 \
  > /tmp/trtllm-nemotron.log 2>&1 &
```

### Step 4: Run TRT-LLM Nemotron benchmarks

Same pattern as Llama — substitute model name in commands and output paths.

```bash
for ISL in 512 1024 2048 4096; do
  echo "=== TRT-LLM Nemotron BS=1 ISL=$ISL ==="
  python -m sglang.bench_serving \
    --backend gserver \
    --port 8000 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --dataset-name random \
    --random-input-len $ISL --random-output-len 200 \
    --num-prompts 10 \
    --request-rate inf --max-concurrency 1 \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/trtllm/nemotron/bs1.log
done
```

```bash
for C in 1 10 25 50 100; do
  echo "=== TRT-LLM Nemotron c=$C ==="
  python -m sglang.bench_serving \
    --backend gserver \
    --port 8000 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 200 \
    --num-prompts 200 \
    --request-rate inf --max-concurrency $C \
    2>&1 | tee -a /workspace/benchmark_results/framework-shootout/trtllm/nemotron/throughput.log
done
```

### Step 5: Stop TRT-LLM

```bash
pkill -f "trtllm-serve"; sleep 5
```

---

## Task 5: Analysis and Report

**Files:**
- Create: `/workspace/benchmark_results/framework-shootout/REPORT.md`

### Step 1: Extract metrics from logs

For each framework x model combination, extract:
- **Batch=1 decode**: tokens/s at each ISL
- **Serving**: output tok/s, ITL p50 (ms), TTFT p50 (ms) at each concurrency level

### Step 2: Build comparison tables

**Table 1: Batch=1 Decode Throughput (tok/s)**

| ISL | vLLM Llama | vLLM Nem | SGLang Llama | SGLang Nem | TRT Llama | TRT Nem |
|-----|-----------|---------|-------------|-----------|----------|--------|
| 512 | | | | | | |
| 1024 | | | | | | |
| 2048 | | | | | | |
| 4096 | | | | | | |

**Table 2: Serving Throughput at c=100 (tok/s)**

| Framework | Llama | Nemotron | Ratio (Nem/Llama) |
|-----------|-------|---------|-------------------|
| vLLM | | | |
| SGLang | | | |
| TRT-LLM | | | |

**Table 3: ITL p50 at c=100 (ms)**

| Framework | Llama | Nemotron |
|-----------|-------|---------|
| vLLM | | |
| SGLang | | |
| TRT-LLM | | |

**Table 4: TTFT p50 at c=100 (ms)**

| Framework | Llama | Nemotron |
|-----------|-------|---------|
| vLLM | | |
| SGLang | | |
| TRT-LLM | | |

### Step 3: Compute the key metric

**Hybrid-to-Transformer Ratio at c=100** = Nemotron throughput / Llama throughput per framework.

- If SGLang ratio > 0.8x AND vLLM ratio < 0.5x → Framework is the bottleneck. Build on SGLang.
- If all ratios < 0.5x → Architecture is the bottleneck. Kernel optimization needed.
- If TRT-LLM ratio > 0.8x → NVIDIA has solved it. Consider NIM path.

### Step 4: Write recommendation

Based on findings, recommend one of:
1. **Build SGLang connector** — if SGLang closes the gap
2. **Explore TRT-LLM/NIM path** — if TRT-LLM closes the gap
3. **Kernel optimization** — if no framework closes the gap
4. **Hybrid approach** — if different frameworks win at different concurrency levels

### Step 5: Share report

Post key findings to Slack with comparison table and recommendation. Full report at `/workspace/benchmark_results/framework-shootout/REPORT.md`.

---

## Task 6: Next Steps Based on Results

### If SGLang wins → Port Qwerky to SGLang

**Rough scope**:
1. Study SGLang model registration (how NemotronH was added — PR #10909)
2. Create `QwerkyLlamaMambaForCausalLM` class matching SGLang's model interface
3. Register Mamba-1 selective_scan with SGLang's hybrid backend
4. Weight loading (same remapping as vLLM plugin)
5. Test: serve Qwerky 3B on SGLang, verify coherent output
6. Benchmark: compare Qwerky on SGLang vs vLLM

Estimate: 1-2 weeks for initial port.

### If TRT-LLM wins → Investigate NIM path

1. Get NIM container running with Nemotron
2. Study TRT-LLM model integration for custom architectures
3. Evaluate effort to add Qwerky Mamba-1 support
4. Consider partnering with NVIDIA

### If nobody wins → Kernel optimization

1. Return to 4x speedup plan
2. Prioritize fused batched selective_scan kernel
3. Evaluate Mamba-2 migration timeline
4. Consider custom inference stack (Cartesia approach)

---

## Baseline Numbers (Evan's Paper, RTX PRO 6000 Blackwell, for reference)

| Metric | Llama-8B | Nemotron-9B | Qwerky-8B |
|--------|---------|------------|----------|
| BS=1 decode (tok/s) ISL=2048 | 78.7 | 71.1 | 76.2 |
| Prefill (ms) ISL=2048 | 17.6 | 170.5 | 183.8 |
| c=100 throughput (tok/s) | 1,856.8 | 797.4 | 878.8 |
| c=100 ITL p50 (ms) | 40.3 | 107.5 | 103.8 |
| c=100 TTFT p50 (ms) | 107.3 | 2,078.8 | 1,714.7 |

Note: Our H100 numbers will differ from these Blackwell numbers. The ratios are what matter.
