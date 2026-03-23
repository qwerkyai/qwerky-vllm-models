"""
Benchmark: original cumsum+chunk_state (2 kernels) vs fused (1 kernel).
Also benchmarks full SSD pipeline: original 5 kernels vs fused 1+2 + 3 + fused 4+5.

Usage:
    python bench_fused_cumsum_state.py | tee results_fused_cumsum_state.txt
"""

import torch
import time
from einops import rearrange

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)
from vllm.model_executor.layers.mamba.ops.ssd_fused_cumsum_state import (
    _fused_cumsum_chunk_state_fwd,
)
from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd
from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd
from vllm.model_executor.layers.mamba.ops.ssd_fused_scan import _fused_chunk_scan_fwd

dev = "cuda"
nheads, ngroups, hdim, dstate = 24, 24, 128, 128
N_ITERS = 200
WARMUP = 10


def bench_cumsum_state(seqlen, chunk_size):
    """Benchmark just kernels 1+2: separate vs fused."""
    nchunks = seqlen // chunk_size

    dt_raw = torch.randn(seqlen, nheads, device=dev, dtype=torch.float32)
    A = -torch.ones(nheads, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=dev, dtype=torch.float32) * 0.1
    x = torch.randn(seqlen, nheads, hdim, device=dev, dtype=torch.bfloat16)
    B = torch.randn(seqlen, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    cu_sl = torch.arange(0, seqlen + 1, chunk_size, device=dev, dtype=torch.int32)

    # warmup
    for _ in range(WARMUP):
        _chunk_cumsum_fwd(
            dt_raw,
            A,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
        )
        _chunk_state_fwd(
            B,
            x,
            torch.empty(nheads, nchunks, chunk_size, device=dev, dtype=torch.float32),
            torch.empty(nheads, nchunks, chunk_size, device=dev, dtype=torch.float32),
            cu_sl,
            states_in_fp32=True,
        )
        _fused_cumsum_chunk_state_fwd(
            dt_raw,
            A,
            B,
            x,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            states_in_fp32=True,
        )
    torch.cuda.synchronize()

    # original (2 kernels)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dA, dt = _chunk_cumsum_fwd(
            dt_raw,
            A,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
        )
        _chunk_state_fwd(B, x, dt, dA, cu_sl, states_in_fp32=True)
    torch.cuda.synchronize()
    t_orig = (time.perf_counter() - t0) / N_ITERS * 1000

    # fused (1 kernel)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _fused_cumsum_chunk_state_fwd(
            dt_raw,
            A,
            B,
            x,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            states_in_fp32=True,
        )
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / N_ITERS * 1000

    return t_orig, t_fused


def bench_full_ssd(seqlen, chunk_size):
    """Benchmark full SSD: original 5 vs (fused 1+2) + 3 + (fused 4+5)."""
    nchunks = seqlen // chunk_size

    dt_raw = torch.randn(seqlen, nheads, device=dev, dtype=torch.float32)
    A = -torch.ones(nheads, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=dev, dtype=torch.float32) * 0.1
    x = torch.randn(seqlen, nheads, hdim, device=dev, dtype=torch.bfloat16)
    B = torch.randn(seqlen, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    C = torch.randn(seqlen, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    D = torch.randn(nheads, device=dev, dtype=torch.float32)
    cu_sl = torch.arange(0, seqlen + 1, chunk_size, device=dev, dtype=torch.int32)
    seq_idx = torch.zeros(nchunks, device=dev, dtype=torch.int32)
    out = torch.zeros(seqlen, nheads, hdim, device=dev, dtype=torch.bfloat16)

    def run_original():
        dA, dt = _chunk_cumsum_fwd(
            dt_raw,
            A,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
        )
        states = _chunk_state_fwd(B, x, dt, dA, cu_sl, states_in_fp32=True)
        states = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"), dA, cu_sl, seq_idx=seq_idx
        )
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        CB = _bmm_chunk_fwd(
            C, B, chunk_size, cu_sl, causal=True, output_dtype=torch.float32
        )
        _chunk_scan_fwd(CB, x, dt, dA, C, states, cu_sl, out, seq_idx, D=D)

    def run_fused():
        dA, dt, states = _fused_cumsum_chunk_state_fwd(
            dt_raw,
            A,
            B,
            x,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            states_in_fp32=True,
        )
        states = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"), dA, cu_sl, seq_idx=seq_idx
        )
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        _fused_chunk_scan_fwd(x, B, C, dt, dA, states, cu_sl, out, seq_idx, D=D)

    # warmup
    for _ in range(WARMUP):
        run_original()
        run_fused()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        run_original()
    torch.cuda.synchronize()
    t_orig = (time.perf_counter() - t0) / N_ITERS * 1000

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        run_fused()
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / N_ITERS * 1000

    return t_orig, t_fused


if __name__ == "__main__":
    print("Kernels 1+2 only: cumsum+chunk_state separate vs fused")
    print(
        f"{'chunk':>6} {'seqlen':>7} {'orig(ms)':>9} {'fused(ms)':>10} {'speedup':>8}"
    )
    print("-" * 45)
    for chunk_size in [64, 128]:
        for seqlen in [128, 512, 1024, 2048, 4096, 8192, 16384]:
            to, tf = bench_cumsum_state(seqlen, chunk_size)
            print(
                f"{chunk_size:>6} {seqlen:>7} {to:>9.3f} {tf:>10.3f} {to / tf:>7.2f}x"
            )

    print()
    print("Full SSD: original 5 kernels vs (fused 1+2) + 3 + (fused 4+5)")
    print(
        f"{'chunk':>6} {'seqlen':>7} {'orig(ms)':>9} {'fused(ms)':>10} {'speedup':>8}"
    )
    print("-" * 45)
    for chunk_size in [64, 128]:
        for seqlen in [128, 512, 1024, 2048, 4096, 8192, 16384]:
            to, tf = bench_full_ssd(seqlen, chunk_size)
            print(
                f"{chunk_size:>6} {seqlen:>7} {to:>9.3f} {tf:>10.3f} {to / tf:>7.2f}x"
            )
