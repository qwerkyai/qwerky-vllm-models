"""Benchmark that cleanly separates prefill and decode latency.

For each model × context length × mode, measures:
  1. TTFT proxy: wall-clock of 2-token generation (prefill + 2 decode + overhead)
  2. Pure prefill: TTFT minus 2 × decode step
  3. Pure decode ms/tok: subtraction of 2-tok vs 128-tok generation
  4. E2E ms/tok: total time / output tokens (for reference)

Usage:
    # Compare default Qwerky vs Llama
    python bench_latency.py

    # Custom model paths
    python bench_latency.py --qwerky /path/to/model --llama /path/to/model

    # HuggingFace model names (downloaded automatically)
    python bench_latency.py --qwerky QwerkyAI/Qwerky-Mamba2-3B --llama meta-llama/Llama-3.2-3B-Instruct

    # Single model only
    python bench_latency.py --qwerky-only
    python bench_latency.py --llama-only

    # Options
    python bench_latency.py --eager
    python bench_latency.py --contexts 512 4096 16384
    python bench_latency.py --batch-sizes 1 4 8
"""

import argparse
import logging
import os
import time

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import torch

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DEFAULT_QWERKY_PATH = os.environ.get("QWERKY_MODEL_PATH", "QwerkyAI/Qwerky-Llama3.2-Mamba2-3B-Llama3.1-8B-base-distill-false-rms")
DEFAULT_LLAMA_PATH = os.environ.get(
    "LLAMA_MODEL_PATH", "meta-llama/Llama-3.2-3B-Instruct"
)

SHORT_TOKENS = 2
LONG_TOKENS = 128
WARMUP_RUNS = 3
BENCH_RUNS = 5


def resolve_model_path(path_or_name: str) -> str:
    """Return path as-is if it exists locally, otherwise treat as HF repo name."""
    if os.path.isdir(path_or_name):
        return path_or_name
    # Assume HuggingFace model name — vLLM will download it automatically
    log.info("'%s' not found locally, will use as HuggingFace model name", path_or_name)
    return path_or_name


def make_prompt(n_tokens: int) -> str:
    """Build a prompt of approximately n_tokens tokens."""
    base = "The quick brown fox jumps over the lazy dog. "
    return base * max(1, n_tokens // 10)


def measure(llm, prompts, sp, runs: int) -> list[dict]:
    """Run generation `runs` times, return per-run timing."""
    results = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        per_req_out = len(outputs[0].outputs[0].token_ids)
        in_toks = len(outputs[0].prompt_token_ids)
        results.append(
            {
                "elapsed": elapsed,
                "total_out_toks": total_out,
                "per_req_out_toks": per_req_out,
                "in_toks": in_toks,
                "batch": len(prompts),
            }
        )
    return results


def bench_model(
    model_path: str,
    model_name: str,
    context_lengths: list[int],
    batch_sizes: list[int],
    enforce_eager: bool = False,
) -> dict:
    """Benchmark a single model across context lengths and batch sizes."""
    from vllm import LLM, SamplingParams

    mode = "eager" if enforce_eager else "compiled+CG"
    max_ctx = max(context_lengths) + LONG_TOKENS + 128
    max_batch = max(batch_sizes)

    log.info("=" * 70)
    log.info("  %s (%s)", model_name, mode)
    log.info("  Path: %s", model_path)
    log.info("  Contexts: %s, Batches: %s", context_lengths, batch_sizes)
    log.info("  Decode isolation: %d-tok minus %d-tok", LONG_TOKENS, SHORT_TOKENS)
    log.info("=" * 70)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_ctx,
        gpu_memory_utilization=0.90,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        max_num_seqs=max_batch,
    )

    # Warmup
    sp_warmup = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    for _ in range(WARMUP_RUNS):
        llm.generate(["Hello world"], sp_warmup)

    results = {}
    for ctx_len in context_lengths:
        prompt = make_prompt(ctx_len) if ctx_len > 100 else "The meaning of life is"

        for bs in batch_sizes:
            prompts = [prompt] * bs
            sp_short = SamplingParams(temperature=0.0, max_tokens=SHORT_TOKENS, ignore_eos=True)
            sp_long = SamplingParams(temperature=0.0, max_tokens=LONG_TOKENS, ignore_eos=True)

            # Warmup this config
            try:
                llm.generate(prompts, sp_short)
                llm.generate(prompts, sp_long)
            except Exception as e:
                log.warning(
                    "ctx=%6d, batch=%2d: FAILED (%s)", ctx_len, bs, e.__class__.__name__
                )
                results[(ctx_len, bs)] = None
                continue

            try:
                short_runs = measure(llm, prompts, sp_short, BENCH_RUNS)
                long_runs = measure(llm, prompts, sp_long, BENCH_RUNS)
            except Exception as e:
                log.warning(
                    "ctx=%6d, batch=%2d: FAILED during bench (%s)",
                    ctx_len,
                    bs,
                    e.__class__.__name__,
                )
                results[(ctx_len, bs)] = None
                continue

            input_toks = short_runs[0]["in_toks"]
            short_toks = short_runs[0]["per_req_out_toks"]
            avg_short = np.mean([r["elapsed"] for r in short_runs])
            avg_long = np.mean([r["elapsed"] for r in long_runs])
            avg_long_total_toks = np.mean([r["total_out_toks"] for r in long_runs])
            avg_long_per_req_toks = np.mean([r["per_req_out_toks"] for r in long_runs])

            # Pure decode = (long - short) / (long_total_toks - short_toks * batch)
            decode_time = avg_long - avg_short
            decode_toks = avg_long_total_toks - short_toks * bs
            decode_ms = (
                decode_time / decode_toks * 1000 if decode_toks > 0 else float("inf")
            )
            decode_tok_s = 1000 / decode_ms if decode_ms > 0 else 0

            # E2E
            e2e_ms = avg_long / avg_long_total_toks * 1000
            e2e_tok_s = 1000 / e2e_ms

            # TTFT proxy: wall-clock of 2-token generation
            ttft_proxy_ms = avg_short * 1000

            # Pure prefill: TTFT minus decode steps
            prefill_ms = ttft_proxy_ms - short_toks * bs * decode_ms

            results[(ctx_len, bs)] = {
                "input_toks": input_toks,
                "batch": bs,
                "e2e_ms": e2e_ms,
                "e2e_tok_s": e2e_tok_s,
                "decode_ms": decode_ms,
                "decode_tok_s": decode_tok_s,
                "ttft_proxy_ms": ttft_proxy_ms,
                "prefill_ms": prefill_ms,
                "long_total_ms": avg_long * 1000,
                "long_out_toks": avg_long_total_toks,
            }

            bs_label = f"batch={bs:>2}, " if len(batch_sizes) > 1 else ""
            log.info(
                "ctx=%6d (%5d in), %sdecode=%.2f ms/tok (%.1f tok/s)  "
                "ttft~=%.1f ms  prefill~=%.1f ms  e2e=%.2f ms/tok (%.1f tok/s)",
                ctx_len,
                input_toks,
                bs_label,
                decode_ms,
                decode_tok_s,
                ttft_proxy_ms,
                prefill_ms,
                e2e_ms,
                e2e_tok_s,
            )

    del llm
    torch.cuda.empty_cache()
    return results


def print_comparison(
    qwerky: dict,
    llama: dict,
    context_lengths: list[int],
    batch_sizes: list[int],
    qwerky_name: str,
    llama_name: str,
):
    """Print side-by-side comparison of two model benchmarks."""
    log.info("=" * 100)
    log.info("  COMPARISON: %s vs %s", qwerky_name, llama_name)
    log.info("=" * 100)

    for bs in batch_sizes:
        if len(batch_sizes) > 1:
            log.info("--- Batch = %d ---", bs)

        header = (
            f"  {'Context':>8} | {'Qwerky decode':>14} | {'Llama decode':>13} | {'Decode':>7}"
            f" | {'Q TTFT':>9} | {'L TTFT':>9} | {'TTFT':>6}"
            f" | {'Q prefill':>10} | {'L prefill':>10} | {'Prefill':>7}"
        )
        log.info(header)
        log.info("  %s", "-" * len(header.strip()))

        for ctx in context_lengths:
            q_res = qwerky.get((ctx, bs))
            l_res = llama.get((ctx, bs))
            if not q_res or not l_res:
                continue
            d_ratio = q_res["decode_tok_s"] / l_res["decode_tok_s"]
            t_ratio = (
                q_res["ttft_proxy_ms"] / l_res["ttft_proxy_ms"]
                if l_res["ttft_proxy_ms"] > 0
                else float("inf")
            )
            p_ratio = (
                q_res["prefill_ms"] / l_res["prefill_ms"]
                if l_res["prefill_ms"] > 0
                else float("inf")
            )
            d_marker = "*" if d_ratio > 1.0 else " "
            log.info(
                "  %8d | %7.2f ms/tok | %6.2f ms/tok | %5.2fx%s"
                " | %6.1f ms | %6.1f ms | %5.1fx"
                " | %7.1f ms | %7.1f ms | %5.1fx",
                ctx,
                q_res["decode_ms"],
                l_res["decode_ms"],
                d_ratio,
                d_marker,
                q_res["ttft_proxy_ms"],
                l_res["ttft_proxy_ms"],
                t_ratio,
                q_res["prefill_ms"],
                l_res["prefill_ms"],
                p_ratio,
            )

        log.info("  * = %s faster at decode", qwerky_name)

    log.info("  TTFT = time-to-first-token proxy (2-token generation wall-clock)")
    log.info("  Prefill = TTFT minus decode steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode vs Prefill benchmark — cleanly separates prefill and decode latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--qwerky",
        type=str,
        default=DEFAULT_QWERKY_PATH,
        help="Path or HuggingFace name for Qwerky model",
    )
    parser.add_argument(
        "--llama",
        type=str,
        default=DEFAULT_LLAMA_PATH,
        help="Path or HuggingFace name for Llama model",
    )
    parser.add_argument(
        "--qwerky-name",
        type=str,
        default="Qwerky Mamba2",
        help="Display name for Qwerky model",
    )
    parser.add_argument(
        "--llama-name",
        type=str,
        default="Llama 3.2 3B",
        help="Display name for Llama model",
    )
    parser.add_argument(
        "--qwerky-only", action="store_true", help="Only benchmark Qwerky model"
    )
    parser.add_argument(
        "--llama-only", action="store_true", help="Only benchmark Llama model"
    )
    parser.add_argument(
        "--eager", action="store_true", help="Use eager mode (default: compiled+CG)"
    )
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[6, 4096, 16384, 32768],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1], help="Batch sizes to test"
    )
    args = parser.parse_args()

    if args.qwerky_only and args.llama_only:
        parser.error("--qwerky-only and --llama-only are mutually exclusive")

    qwerky_path = resolve_model_path(args.qwerky)
    llama_path = resolve_model_path(args.llama)

    log.info("=" * 70)
    log.info("  DECODE vs PREFILL BENCHMARK")
    log.info("  Qwerky: %s", qwerky_path)
    log.info("  Llama:  %s", llama_path)
    log.info("  Contexts: %s", args.contexts)
    log.info("  Batch sizes: %s", args.batch_sizes)
    log.info("  Mode: %s", "eager" if args.eager else "compiled+CG")
    log.info(
        "  Decode isolation: %d-tok minus %d-tok generation", LONG_TOKENS, SHORT_TOKENS
    )
    log.info("  TTFT proxy: wall-clock of %d-tok generation", SHORT_TOKENS)
    log.info("  Pure prefill: TTFT minus %d × decode step", SHORT_TOKENS)
    log.info("  Runs: %d per config (after %d warmup)", BENCH_RUNS, WARMUP_RUNS)
    log.info("=" * 70)

    qwerky_results = None
    llama_results = None

    if not args.llama_only:
        qwerky_results = bench_model(
            qwerky_path,
            args.qwerky_name,
            args.contexts,
            args.batch_sizes,
            enforce_eager=args.eager,
        )

    if not args.qwerky_only:
        llama_results = bench_model(
            llama_path,
            args.llama_name,
            args.contexts,
            args.batch_sizes,
            enforce_eager=args.eager,
        )

    if qwerky_results and llama_results:
        print_comparison(
            qwerky_results,
            llama_results,
            args.contexts,
            args.batch_sizes,
            args.qwerky_name,
            args.llama_name,
        )
