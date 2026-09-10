# Offline SDXL Q3/DSPO on 2xB200

The production entrypoint is:

```bash
bash project_command.sh
```

It sources `env.sh`, creates an offline uv environment, validates all local
assets, validates two scheduler-assigned Blackwell GPUs, launches Accelerate DDP,
evaluates the final model immediately, and writes `results/summary.json`.

All assets must already exist under `offline_assets/` (or be redirected with
the variables in `env.sh`). Follow `download.txt` on an Internet-connected
Linux machine. The server flow sets all Hugging Face libraries to offline mode
and disables external reporting/upload.

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

The Hessian A100 pilot uses the same entrypoint with overrides and is not the
production configuration:

```bash
PIPELINE_MODE=pilot GPU_IDS=2,3 NUM_GPUS=2 TARGET_GPU_FAMILY=A100 \
  OFFLINE_EVAL_LIMIT=2 bash project_command.sh
```
