"""
Benchmark: original 5 SSD kernels vs fused (kernels 1-3 unchanged + fused 4+5).
Measures only the SSD portion (not full model prefill).

Usage:
    python bench_ssd_fusion.py | tee results_ssd_fusion.txt
"""

import torch
import time
from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd
from vllm.model_executor.layers.mamba.ops.ssd_fused_scan import _fused_chunk_scan_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)
from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd
from einops import rearrange

dev = "cuda"
nheads, ngroups, hdim, dstate = 24, 24, 128, 128
N_ITERS = 200
WARMUP = 10


def bench(seqlen, chunk_size):
    nchunks = seqlen // chunk_size

    x = torch.randn(seqlen, nheads, hdim, device=dev, dtype=torch.bfloat16)
    B = torch.randn(seqlen, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    C = torch.randn(seqlen, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    dt_raw = torch.randn(seqlen, nheads, device=dev, dtype=torch.float32)
    A = -torch.ones(nheads, device=dev, dtype=torch.float32)
    dt_bias = torch.zeros(nheads, device=dev, dtype=torch.float32)
    D = torch.randn(nheads, device=dev, dtype=torch.float32)
    cu_sl = torch.arange(0, seqlen + 1, chunk_size, device=dev, dtype=torch.int32)
    seq_idx = torch.zeros(nchunks, device=dev, dtype=torch.int32)
    out = torch.zeros(seqlen, nheads, hdim, device=dev, dtype=torch.bfloat16)

    def run_steps_123():
        dA_cs, dt2 = _chunk_cumsum_fwd(
            dt_raw.clone(),
            A,
            chunk_size,
            cu_sl,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
        )
        states = _chunk_state_fwd(B, x, dt2, dA_cs, cu_sl, states_in_fp32=True)
        states = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"), dA_cs, cu_sl, seq_idx=seq_idx
        )
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        return dA_cs, dt2, states

    # warmup
    for _ in range(WARMUP):
        dA_cs, dt2, st = run_steps_123()
        CB = _bmm_chunk_fwd(
            C, B, chunk_size, cu_sl, causal=True, output_dtype=torch.float32
        )
        _chunk_scan_fwd(CB, x, dt2, dA_cs, C, st, cu_sl, out, seq_idx, D=D)
        _fused_chunk_scan_fwd(x, B, C, dt2, dA_cs, st, cu_sl, out, seq_idx, D=D)
    torch.cuda.synchronize()

    # original (all 5 kernels)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dA_cs, dt2, st = run_steps_123()
        CB = _bmm_chunk_fwd(
            C, B, chunk_size, cu_sl, causal=True, output_dtype=torch.float32
        )
        _chunk_scan_fwd(CB, x, dt2, dA_cs, C, st, cu_sl, out, seq_idx, D=D)
    torch.cuda.synchronize()
    t_orig = (time.perf_counter() - t0) / N_ITERS * 1000

    # fused (kernels 1-3 + fused 4+5)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dA_cs, dt2, st = run_steps_123()
        _fused_chunk_scan_fwd(x, B, C, dt2, dA_cs, st, cu_sl, out, seq_idx, D=D)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / N_ITERS * 1000

    return t_orig, t_fused


if __name__ == "__main__":
    print(
        f"{'chunk':>6} {'seqlen':>7} {'orig(ms)':>9} {'fused(ms)':>10} {'speedup':>8}"
    )
    print("-" * 45)
    for chunk_size in [64, 128]:
        for seqlen in [128, 512, 1024, 2048, 4096, 8192, 16384]:
            t_orig, t_fused = bench(seqlen, chunk_size)
            speedup = t_orig / t_fused
            print(
                f"{chunk_size:>6} {seqlen:>7} {t_orig:>9.3f} {t_fused:>10.3f} {speedup:>7.2f}x"
            )
