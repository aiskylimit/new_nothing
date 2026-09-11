# Offline SDXL Q3/DSPO on 2xB200

The production entrypoint is:

```bash
bash project_command.sh
```

It sources `env.sh`, activates the uv environment created by the platform from
the root-level `sdxl-q3-offline-b200-2gpu.txt`, validates all local assets and two scheduler-assigned
Blackwell GPUs, launches Accelerate DDP,
evaluates the final model immediately, and writes `results/summary.json`.

The environment specification lives outside this source directory at the
repository root. Its filename exactly matches the environment directory:
`source /mnt/local/uvenvs/sdxl-q3-offline-b200-2gpu/bin/activate`.
The platform must also process this project's `download.txt` manifest before
the job starts. It downloads every model and dataset into `offline_assets/`;
the runtime itself sets all Hugging Face libraries to offline mode and
disables external reporting/upload.

Default production semantics:

- GPUs: `0,1` (the two GPUs visible inside the allocation)
- micro-batch: 2 pairs per GPU
- gradient accumulation: 16
- effective batch: 64 pairs
- full dataset: 851,293 binary preference pairs
- optimizer steps: 13,302
- precision: BF16
- model: SDXL 1.0 at 1024x1024
- method: Q3 (TBPO + reference-MSE policy weighting + DSPO winner anchor)

Detailed logs are in `runtime/logs/project.log`, per-run training logs are in
`runtime/runs/<run>/console.log`, and per-stage evaluation logs are in
`runtime/eval/<eval>/logs/`. Rerunning the entrypoint reuses completed model,
prompt, image, score, and report artifacts.

For a non-platform pilot, point `VENV_DIR` at an existing compatible uv
environment. For example:

```bash
PIPELINE_MODE=pilot GPU_IDS=2,3 NUM_GPUS=2 TARGET_GPU_FAMILY=A100 \
  VENV_DIR=/path/to/existing/venv \
  OFFLINE_EVAL_LIMIT=2 bash project_command.sh
```
