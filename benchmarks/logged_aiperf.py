#!/usr/bin/env python3
"""
Thin wrapper: parses two raw strings, runs aiperf, logs to W&B.

Usage:
  python benchmarks/logged_aiperf.py \
    --wandb_args "project=vllm-eval,entity=qwerky-ai,group=hybrid_mamba,name=run1,job_type=eval" \
    --aiperf_args "model=QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill,url=http://localhost:8000/v1,streaming,endpoint-type=chat,concurrency=1,request-count=10,isl=16,osl=1024"

Format for both:
  key=value  ->  passed as key:value (wandb) or --key value (aiperf)
  bare_key   ->  passed as flag  (wandb: added to tags / aiperf: --bare_key)
"""

import argparse
import glob
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import wandb

log = logging.getLogger(__name__)


def parse_kv_string(raw: str) -> tuple[dict, list[str]]:
    """Split 'k=v,k=v,bare,...' -> (dict, list_of_bare_keys)."""
    kv, bare = {}, []
    if not raw:
        return kv, bare
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            kv[k.strip()] = v.strip()
        else:
            bare.append(item)
    return kv, bare


def kv_to_aiperf_cmd(kv: dict, bare: list[str]) -> list[str]:
    """Convert parsed key-values into aiperf CLI args."""
    cmd = []
    for k, v in kv.items():
        flag = f"--{k}" if not k.startswith("-") else k
        cmd.extend([flag, v])
    for b in bare:
        flag = f"--{b}" if not b.startswith("-") else b
        cmd.append(flag)
    return cmd


def kv_to_wandb_kwargs(kv: dict, bare: list[str]) -> dict:
    """Convert parsed key-values into wandb.init() kwargs."""
    kwargs = dict(kv)
    if bare:
        kwargs.setdefault("tags", [])
        if isinstance(kwargs["tags"], str):
            kwargs["tags"] = kwargs["tags"].split(";")
        kwargs["tags"].extend(bare)
    return kwargs


def find_json_artifact(d: str) -> str | None:
    for m in glob.glob(os.path.join(d, "**", "*aiperf*.json"), recursive=True):
        if not m.endswith(".jsonl"):
            return m
    return None


def find_records(d: str) -> list[dict] | None:
    for m in glob.glob(os.path.join(d, "**", "profile_export*.jsonl"), recursive=True):
        if "_raw" in m:
            continue
        recs = [json.loads(line) for line in open(m) if line.strip()]
        if recs:
            return recs
    return None


def flatten(obj, prefix="") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else k))
    elif isinstance(obj, (int, float)):
        out[prefix] = obj
    return out


# Summary table (like aiperf console output)
STAT_COLS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

# Exact metrics in exact order, matching aiperf console output
METRIC_ORDER = [
    ("time_to_first_token", "Time to First Token (ms)"),
    ("time_to_second_token", "Time to Second Token (ms)"),
    ("time_to_first_output_token", "Time to First Output Token (ms)"),
    ("request_latency", "Request Latency (ms)"),
    ("inter_token_latency", "Inter Token Latency (ms)"),
    (
        "output_token_throughput_per_user",
        "Output Token Throughput Per User (tokens/sec/user)",
    ),
    ("output_sequence_length", "Output Sequence Length (tokens)"),
    ("input_sequence_length", "Input Sequence Length (tokens)"),
    ("output_token_throughput", "Output Token Throughput (tokens/sec)"),
    ("request_throughput", "Request Throughput (requests/sec)"),
    ("request_count", "Request Count (requests)"),
]


def build_summary_rows(data: dict) -> list[dict]:
    """Extract the fixed set of metrics from aiperf JSON for wandb.Table."""
    m = data
    if "metrics" in data and isinstance(data["metrics"], dict):
        m = data["metrics"]

    rows = []
    for tag, display in METRIC_ORDER:
        obj = m.get(tag)
        if not isinstance(obj, dict):
            continue
        row = {"Metric": display}
        for s in STAT_COLS:
            row[s] = obj.get(s)
        rows.append(row)
    return rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--aiperf_args",
        required=True,
        help='"model=X,url=Y,streaming,concurrency=10,..."',
    )
    p.add_argument(
        "--wandb_args", required=True, help='"project=X,entity=Y,name=Z,..."'
    )
    p.add_argument(
        "--artifact_dir",
        default=None,
        help="Where aiperf writes output (kept after run; default: temp 'artifacts/' dir, deleted after upload)",
    )
    args = p.parse_args()

    keep_artifacts = args.artifact_dir is not None
    if args.artifact_dir is None:
        args.artifact_dir = "artifacts"

    # Parse raw strings
    ai_kv, ai_bare = parse_kv_string(args.aiperf_args)
    wb_kv, wb_bare = parse_kv_string(args.wandb_args)

    # Build aiperf command
    cmd = ["aiperf", "profile"] + kv_to_aiperf_cmd(ai_kv, ai_bare)
    log.info("Executing %s", " ".join(shlex.quote(c) for c in cmd))

    # Init W&B
    wb_active = False
    if wb_kv and wandb is not None:
        wandb.init(**kv_to_wandb_kwargs(wb_kv, wb_bare))
        wandb.config.update({"aiperf/" + k: v for k, v in ai_kv.items()})
        for b in ai_bare:
            wandb.config.update({f"aiperf/{b}": True})
        wb_active = True

    # Run
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        log.error("aiperf exited with code %d", rc)
        if wb_active:
            wandb.finish(exit_code=rc)
        sys.exit(rc)

    # Parse & log results
    jp = find_json_artifact(args.artifact_dir)
    if not jp:
        log.warning("No JSON results in %s/", args.artifact_dir)
        if wb_active:
            wandb.finish()
        return

    with open(jp) as f:
        data = json.load(f)

    # Flat metrics for wandb.log()
    metrics = flatten(data)

    # Summary table for W&B
    summary_rows = build_summary_rows(data)

    if wb_active:
        wandb.log(metrics)

        # Summary table (the pretty one)
        if summary_rows:
            hdr = ["Metric"] + STAT_COLS
            st = wandb.Table(columns=hdr)
            for row in summary_rows:
                st.add_data(*[row.get(h) for h in hdr])
            wandb.log({"summary_table": st})

        # Per-request records
        recs = find_records(args.artifact_dir)
        if recs:
            cols = sorted({k for r in recs for k in r})
            t = wandb.Table(columns=cols)
            for r in recs:
                t.add_data(*[r.get(c) for c in cols])
            wandb.log({"per_request_records": t})

        # Upload artifacts
        art = wandb.Artifact(f"aiperf-{wandb.run.id}", type="benchmark")
        if os.path.isdir(args.artifact_dir):
            art.add_dir(args.artifact_dir)
            wandb.log_artifact(art)

        log.info("W&B run: %s", wandb.run.url)
        wandb.finish()

    # Cleanup: remove artifact dir if it wasn't explicitly requested
    if not keep_artifacts and os.path.isdir(args.artifact_dir):
        shutil.rmtree(args.artifact_dir)
        log.info("Removed temp artifact dir: %s", args.artifact_dir)


if __name__ == "__main__":
    main()
