# Benchmarks

Tools for evaluation:

- **aiperf** - measures inference speed (throughput, latency)
- **lm-eval** - measures model accuracy on standard benchmarks

## aiperf - Speed Benchmarking

`logged_aiperf.py` is a wrapper around `aiperf profile` that logs results to Weights & Biases.

### Usage

```bash
python benchmarks/logged_aiperf.py \
    --wandb_args "<wandb params>" \
    --aiperf_args "<aiperf params>" \
    [--artifact_dir <path>]
```

| Argument | Required | Description |
|---|---|---|
| `--aiperf_args` | yes | Comma-separated aiperf params (`key=value` or bare flags) |
| `--wandb_args` | yes | Comma-separated W&B params (`project=`, `entity=`, `name=`, etc.) |
| `--artifact_dir` | no | Directory to keep aiperf output files. If omitted, a temp dir is used and deleted after upload |

### Example

```bash
python benchmarks/logged_aiperf.py \
    --aiperf_args "model=QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill,url=http://localhost:8000/v1,streaming,endpoint-type=chat,concurrency=1,request-count=10,isl=16,osl=1024" \
    --wandb_args "project=vllm-eval,entity=qwerky-ai,group=hybrid_mamba,name=run1,job_type=eval"
```

### Key aiperf parameters

| Parameter | Description |
|---|---|
| `model` | Model name or HuggingFace ID |
| `url` | vLLM server base URL (e.g. `http://localhost:8000/v1`) |
| `endpoint-type` | `chat` or `completions` |
| `concurrency` | Number of concurrent requests |
| `request-count` | Total number of requests to send |
| `isl` | Input sequence length (tokens) |
| `osl` | Output sequence length (tokens) |
| `streaming` | Enable streaming mode (bare flag) |

### Key wandb parameters

| Parameter | Description |
|---|---|
| `project` | W&B project name |
| `entity` | W&B team or username |
| `name` | Run display name |
| `group` | Group name for organizing related runs |
| `job_type` | Run type label (e.g. `eval`) |


## lm-eval - Accuracy Benchmarking

Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `local-completions` model type to evaluate against a running vLLM server.

### Usage

```bash
lm_eval \
    --model local-completions \
    --model_args model=<model-id>,base_url=<server-url>/completions \
    --tasks <task1,task2,...> \
    --wandb_args project=<project>,entity=<entity>,name=<run-name>,job_type=eval
```

### Example

```bash
lm_eval \
    --model local-completions \
    --model_args model=QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill,base_url=http://localhost:8000/v1/completions \
    --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race \
    --wandb_args project=vllm-eval,entity=qwerky-ai,group=hybrid_mamba,name=vllm_Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distil,job_type=eval
```

### Notes

- The vLLM server must be running and accessible at the specified `base_url` before running either tool
- `base_url` for `lm-eval` should point to the `/v1/completions` endpoint, while `url` for `aiperf` points to `/v1`
- For chat models, use `endpoint-type=chat` in aiperf; lm-eval's `local-completions` uses the completions endpoint regardless
