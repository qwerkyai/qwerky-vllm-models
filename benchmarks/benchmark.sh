#!/usr/bin/env bash
# =============================================================================
# Full Logged AIPerf Benchmark
# Uses benchmarks/logged_aiperf.py with wandb logging
# =============================================================================

MODEL="QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill"
URL="http://localhost:8000/v1"
EP="chat"

run_logged_aiperf() {
    local CONCURRENCY=$1
    local ISL=$2
    local OSL=$3

    NAME="qwerky_c${CONCURRENCY}_i${ISL}_o${OSL}"

    python benchmarks/logged_aiperf.py \
        --wandb_args "project=vllm-eval,entity=qwerky-ai,group=Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill,name=${NAME}" \
        --aiperf_args "model=$MODEL,url=$URL,streaming,endpoint-type=$EP,concurrency=$CONCURRENCY,request-count=$REQUEST_COUNT,isl=$ISL,osl=$OSL"
}

# =============================================================================
# EXPERIMENT 1: Batch=1 Decode Latency
# =============================================================================
REQUEST_COUNT=5
for ISL in 512 1024; do
    OSL=200
    run_logged_aiperf 1 $ISL $OSL
done

# =============================================================================
# EXPERIMENT 2: Serving Throughput Under Concurrent Load
# =============================================================================
REQUEST_COUNT=50
OSL=200
for CONCURRENCY in 1 10 25 50; do
    ISL=1024
    run_logged_aiperf $CONCURRENCY $ISL $OSL
done

# =============================================================================
# EXPERIMENT 3: Full Context Sweep (Mamba-style, push to 128K)
# =============================================================================
REQUEST_COUNT=5
OSL=128
for ISL in 512 1024; do
    run_logged_aiperf 1 $ISL $OSL
done

# =============================================================================
# EXPERIMENT 4: Blackwell-style Throughput Scaling
# =============================================================================
REQUEST_COUNT=200
OSL=256
for CONCURRENCY in 1 8 32 64; do
    ISL=512
    run_logged_aiperf $CONCURRENCY $ISL $OSL
done

# =============================================================================
# EXPERIMENT 5: Combined — Concurrency × Context Length Matrix
# =============================================================================
REQUEST_COUNT=50
OSL=200
for CONCURRENCY in 1 10 25 50; do
    for ISL in 512 1024; do
        run_logged_aiperf $CONCURRENCY $ISL $OSL
    done
done

# =============================================================================
# EXPERIMENT 6: ISL=512, multiple OSL, REQUEST_COUNT=5, CONCURRENCY=1
# =============================================================================
REQUEST_COUNT=5
CONCURRENCY=1
ISL=512
for OSL in 128 200 256 512 1024; do
    run_logged_aiperf $CONCURRENCY $ISL $OSL
done

echo "All logged_aiperf experiments completed."