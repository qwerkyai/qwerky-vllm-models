"""
Benchmark runner using aiperf.

Usage:
    python benchmarks/benchmark_runner.py --config benchmarks/configs/example.yaml

Working principle:
    Experiments are defined in a YAML config file. The config specifies models
    (name, URL, endpoint type) and experiments (combinations of request_count,
    concurrency, isl, osl). For each combination the script calls aiperf, parses
    the resulting JSON artifact, and accumulates stats.

    Results are saved to:
      - a local JSON file (if output_json is set in the config)
      - a wandb run (if wandb_project and wandb_entity are set)

    When wandb logging is used and the benchmark covers 2+ models, comparison
    tables are generated automatically - one table per variable parameter,
    showing output token throughput for every model side by side.

    The resuming option is provided. Set resume: true in the config to load existing results from the
    JSON file or wandb run and skip parameter combinations that were already
    benchmarked, appending only new runs.
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
from itertools import product
from typing import Any
import yaml
import wandb
import shutil

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Statistical columns collected for each metric
STAT_COLS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

# Ordered list of (metric_key, display_name) pairs used when building summary tables
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


def sanitize_metric_name(name: str) -> str:
    # Convert display name to a lowercase snake_case key without parentheses
    name = name.lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[()]", "", name)
    return name


def sanitize_model_name(name: str) -> str:
    # Replace slashes with underscores to make the name safe for use in log keys
    return name.replace("/", "_").lower()


def load_config(path: str) -> dict[str, Any]:
    # Load a YAML or JSON config file and return its contents as a dict
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def load_previous_json_results(
    path: str | None,
    resume: bool,
) -> dict[str, Any]:
    # Return cached results from a local JSON file when resuming a run
    if not resume:
        return {}
    if not path or not os.path.exists(path):
        log.info("No previous JSON results found")
        return {}
    log.info("Loading previous JSON results: %s", path)
    with open(path) as f:
        return json.load(f)


def load_previous_wandb_results(
    project: str | None,
    entity: str | None,
    run_id: str | None,
    resume: bool,
) -> dict[str, Any]:
    # Fetch the benchmark_raw summary field from an existing wandb run when resuming
    if not resume or not project or not entity or not run_id:
        return {}

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        if run is None:
            log.warning("wandb run not found: %s", run_id)
            return {}

        if "benchmark_raw" in run.summary:
            log.info("Loaded previous wandb results from run %s", run_id)
            return run.summary["benchmark_raw"]

        log.warning("Run found but benchmark_raw not present")

    except Exception as e:
        log.warning("Failed to load wandb results: %s", e)

    return {}


def warn_if_different(
    json_results: dict[str, Any], wandb_results: dict[str, Any]
) -> None:
    # Warn when local JSON and wandb caches have diverged
    if not json_results or not wandb_results:
        return
    if json_results != wandb_results:
        log.warning("JSON and wandb results differ!")


def params_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # Check whether two param dicts represent the same benchmark configuration
    return (
        a["request_count"] == b["request_count"]
        and a["concurrency"] == b["concurrency"]
        and a["isl"] == b["isl"]
        and a["osl"] == b["osl"]
    )


def run_exists(
    results: dict[str, Any],
    model_name: str,
    exp_name: str,
    params: dict[str, Any],
) -> bool:
    # Return True if a run with these exact params already exists in the results dict
    if model_name not in results:
        return False
    if exp_name not in results[model_name]:
        return False
    for run in results[model_name][exp_name]:
        if params_match(run["params"], params):
            return True
    return False


def run_aiperf(
    model: dict[str, Any],
    exp_name: str,
    rc: int,
    isl: int,
    osl: int,
    conc: int,
    output_dir: str,
) -> dict[str, int]:
    # Build and execute the aiperf CLI command, raising on non-zero exit code
    cmd: list[str] = [
        "aiperf",
        "profile",
        "--output-artifact-dir",
        output_dir,
        "--model",
        model["name"],
        "--url",
        model["url"],
        "--endpoint-type",
        model["endpoint"],
        "--request-count",
        str(rc),
        "--isl",
        str(isl),
        "--osl",
        str(osl),
        "--concurrency",
        str(conc),
        "--streaming",
    ]
    log.info("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    rc_val: int = subprocess.run(cmd).returncode
    if rc_val != 0:
        raise RuntimeError(f"aiperf failed with code {rc_val}")
    return {"request_count": rc, "concurrency": conc, "osl": osl, "isl": isl}


def find_json_artifact(output_dir: str) -> str | None:
    # Walk the output directory and return the first aiperf JSON artifact found
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".json") and "aiperf" in f:
                return os.path.join(root, f)
    return None


def build_summary_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    # Extract per-metric stat rows from raw aiperf output in METRIC_ORDER sequence
    m: dict[str, Any] = data.get("metrics", data)
    rows: list[dict[str, Any]] = []
    for tag, display in METRIC_ORDER:
        obj = m.get(tag)
        if not isinstance(obj, dict):
            continue
        row: dict[str, Any] = {"Metric": display}
        for s in STAT_COLS:
            row[s] = obj.get(s)
        rows.append(row)
    return rows


def log_tables_for_variable_param(
    cfg: dict[str, Any],
    results: dict[str, Any],
) -> None:
    # For each experiment, log a wandb Table per variable parameter showing
    # output token throughput across all models and fixed-parameter combinations

    # Short prefix letters used to build table names
    prefix: dict[str, str] = {
        "request_count": "r",
        "concurrency": "c",
        "isl": "i",
        "osl": "o",
    }

    for exp_name, exp_cfg in cfg["experiments"].items():
        models_sorted: list[str] = sorted(cfg["models"].keys())
        model_names: list[str] = [cfg["models"][m]["name"] for m in models_sorted]

        # Params with multiple values become the row axis of the table
        variable_params: list[str] = [
            p for p, v in exp_cfg.items() if isinstance(v, list) and len(v) >= 2
        ]
        if not variable_params:
            continue

        for row_param in variable_params:
            row_values: list[Any] = exp_cfg[row_param]

            other_params: list[str] = [p for p in exp_cfg if p != row_param]

            # Wrap scalar params in a list so product() can iterate over them
            other_values: list[list[Any]] = [
                exp_cfg[p] if isinstance(exp_cfg[p], list) else [exp_cfg[p]]
                for p in other_params
            ]

            # Each combo is a specific set of values for the non-variable params
            for combo in product(*other_values):
                fixed_values: dict[str, Any] = dict(zip(other_params, combo))

                table = wandb.Table(columns=[row_param] + model_names)

                for val in row_values:
                    row: list[Any] = [val]

                    for m in models_sorted:
                        model_name: str = cfg["models"][m]["name"]
                        found: Any = None

                        if model_name in results and exp_name in results[model_name]:
                            for run in results[model_name][exp_name]:
                                params: dict[str, Any] = run["params"]

                                # Check that all fixed params match this run
                                match: bool = True
                                for p in fixed_values:
                                    if params[p] != fixed_values[p]:
                                        match = False
                                        break

                                if match and params[row_param] == val:
                                    for r in run["summary"]:
                                        if (
                                            r["Metric"]
                                            == "Output Token Throughput (tokens/sec)"
                                        ):
                                            found = r["avg"]
                                            break

                        row.append(found)

                    table.add_data(*row)

                # Build a descriptive name like "exp_rN_c10_i512_o128"
                name_parts: list[str] = []

                for p in ["request_count", "concurrency", "isl", "osl"]:
                    if p == row_param:
                        name_parts.append(prefix[p] + "N")
                    else:
                        val = fixed_values.get(p)
                        if val is not None:
                            name_parts.append(prefix[p] + str(val))

                table_name: str = f"{exp_name}_" + "_".join(name_parts)

                wandb.log({table_name: table})


def run_experiments(
    cfg: dict[str, Any],
    results: dict[str, Any],
    output_dir: str,
) -> None:
    # Iterate over all models and experiment param combinations, skipping already completed runs
    for model_key, model in cfg["models"].items():
        model_name: str = model["name"]

        if model_name not in results:
            results[model_name] = {}

        for exp_name, exp in cfg["experiments"].items():
            if exp_name not in results[model_name]:
                results[model_name][exp_name] = []

            exp_results: list[dict[str, Any]] = results[model_name][exp_name]

            for rc, isl, osl, conc in product(
                exp["request_count"], exp["isl"], exp["osl"], exp["concurrency"]
            ):
                params: dict[str, Any] = {
                    "request_count": rc,
                    "concurrency": conc,
                    "osl": osl,
                    "isl": isl,
                }

                if run_exists(results, model_name, exp_name, params):
                    log.info(
                        "Skipping existing run %s %s %s",
                        model_name,
                        exp_name,
                        params,
                    )
                    continue

                params = run_aiperf(model, exp_name, rc, isl, osl, conc, output_dir)

                json_path: str | None = find_json_artifact(output_dir)

                if not json_path:
                    log.warning("No JSON results found for %s", params)
                    continue

                with open(json_path) as f:
                    data: dict[str, Any] = json.load(f)

                summary: list[dict[str, Any]] = build_summary_rows(data)

                exp_results.append({"params": params, "summary": summary})


def log_to_wandb(
    results: dict[str, Any],
    project: str,
    entity: str,
    group: str | None,
    name: str | None,
    job_type: str | None,
    cfg: dict[str, Any],
) -> None:
    # Initialize a wandb run, upload raw results and per-stat scalar metrics, then log tables
    wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        job_type=job_type,
    )

    # Store the full nested results for later retrieval when resuming
    wandb.log({"benchmark_raw": results})

    stats_to_log: list[str] = ["avg", "min", "max", "p50", "p90", "p99", "std"]
    summary_log: dict[str, Any] = {}

    for model_name, model_results in results.items():
        model_name_sanitized: str = sanitize_model_name(model_name)

        for exp_name, exp_runs in model_results.items():
            for run in exp_runs:
                params: dict[str, Any] = run["params"]
                summary: list[dict[str, Any]] = run["summary"]

                rc: int = params["request_count"]
                conc: int = params["concurrency"]
                osl: int = params["osl"]
                isl_val: int = params["isl"]

                for row in summary:
                    metric_key: str = sanitize_metric_name(row["Metric"])

                    # Log each stat as a flat key: model/exp_r{rc}_c{conc}_.../{stat}
                    for stat in stats_to_log:
                        if stat in row:
                            log_key: str = f"{model_name_sanitized}/{exp_name}_r{rc}_c{conc}_i{isl_val}_o{osl}_{metric_key}/{stat}"
                            summary_log[log_key] = row[stat]

    wandb.log(summary_log)

    log_tables_for_variable_param(cfg, results)

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg: dict[str, Any] = load_config(args.config)

    resume: bool = cfg.get("resume", False)

    # Extract optional wandb and output settings from config
    output_json: str | None = cfg.get("output_json")
    wandb_project: str | None = cfg.get("wandb_project")
    wandb_entity: str | None = cfg.get("wandb_entity")
    wandb_group: str | None = cfg.get("wandb_group")
    wandb_name: str | None = cfg.get("wandb_name")
    wandb_job_type: str | None = cfg.get("wandb_job_type")
    wandb_id: str | None = cfg.get("wandb_id")

    output_dir: str = cfg.get("output_dir", "artifacts")

    if not output_json and not wandb_project:
        raise ValueError("Need output_json or wandb_project")

    # Load previously saved results from both sources and warn if they differ
    json_results: dict[str, Any] = load_previous_json_results(output_json, resume)
    wandb_results: dict[str, Any] = load_previous_wandb_results(
        wandb_project, wandb_entity, wandb_id, resume
    )

    warn_if_different(json_results, wandb_results)

    results: dict[str, Any] = json_results if json_results else {}
    run_experiments(cfg, results, output_dir)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    if wandb_project:
        log_to_wandb(
            results,
            wandb_project,
            wandb_entity,
            wandb_group,
            wandb_name,
            wandb_job_type,
            cfg,
        )

    log.info("All experiments completed successfully.")

    # Post-experiment cleanup, remove aiperf artifacts directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
