# GRIT: how to run it

Every job is submitted by editing the header (first 3 lines) of the root `commands.sh`,
the same way as for the other projects. GRIT needs 4 kinds of jobs, in this order.
Everything else in `commands.sh` (the `export` lines, the other projects' commented
blocks) stays unchanged.

## Job 1: install the environment (once)

```
#i grit.txt
#grit
#v1
```

This builds `/mnt/local/uvenvs/grit` from the root `grit.txt`.

## Job 2: download models and data (once, ~12 GB)

Replace the header with the full content of [`GRIT/download.txt`](download.txt). It starts
with `#d` and fetches:
- the policy, the training reward model (Qwen3Guard-Gen-4B) and the eval judge (allenai/wildguard),
- the pre-built GRIT artifacts (`fushinguyenex/GRIT-artifacts`),
- the eval benchmarks.

Everything goes under `/mnt/local/aiskylimit_new_nothingnew_2/grit/`. Two repos are gated on HF and
need their terms accepted (once) by the account that downloads: `allenai/wildguard` (short form) and
`sorry-bench/sorry-bench-202406`.

## Job 3: run (repeat until it prints `ALL DONE`)

```
#1 +120
#grit
#v1
```

In the body, keep this block uncommented and comment out any other active project:

```bash
cd ./GRIT
bash ./project_commands.sh
```

Exactly these two lines and nothing else: no variables, no arguments.

What it does: train (40 steps) → eval → summary. It uses GPUs 4,5,6,7. Estimated total
~3.5-6 h (train ~2.5-4.5 h at ~3.5-6.5 min/step, eval ~1-1.5 h), scaled from the 0.5B run
(A100, ~4.6 min/step); so a 120-minute job will not finish in one go. `train/metrics.jsonl`
shows the real seconds per step (`step_seconds`).

**When a job stops (timeout or error), submit the same job again with `#v1` flipped to
`#v2` (or back).** Every stage resumes:
- finished stages are skipped,
- training continues from the latest checkpoint (saved every 5 steps),
- eval continues per model and benchmark.

## Job 4: fetch results

```
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/grit/results_grit_3b_lam1/ +a
#grit
#v1
```

To only peek at progress, use `#2 -0-3` (last 3 run logs).

## What to look at

All small outputs are in `results_grit_3b_lam1/` next to `project_commands.sh` (on the server:
`/mnt/local/aiskylimit_new_nothingnew_2/grit/results_grit_3b_lam1/`, the path printed at the end of every job):

| File | Content |
|---|---|
| `summary.md` | **final tables**: training curve, eval per benchmark, "Safety vs over-refusal", deltas vs base with 95% CI |
| `logs/full_run_*.log` | complete console output of each job |
| `logs/{train,eval}.log` | per-stage logs |
| `mem_config_in_use.txt` | memory setting in use (changes only after an out-of-memory fallback) |
| `train/metrics.jsonl` | one line per training step (reward, unsafe rate, KL, response length, alerts, VRAM, time) |
| `eval/eval_results.json` | all eval numbers |

The console also prints the summary at the end of every job, including a partial
summary when a job fails or is stopped.

## Defaults

`commands.sh` only ever contains the two lines above, so `project_commands.sh` runs with
its built-in defaults:
- `GPU_IDS="4,5,6,7"`: train on 4,5,6, vLLM rollout on 7, eval on all four (each needs ≥150 GB free);
- run id `grit_3b_lam1`, λ_pres = 1, 40 steps;
- eval of base and steps 20 and 40.

To change a setting, edit its default near the top of `GRIT/project_commands.sh`:

| Variable | Effect |
|---|---|
| `GPU_IDS` | the B200s assigned to GRIT (default `4,5,6,7`; the last one is the rollout GPU) |
| `STAGES` | e.g. `eval,summary` to skip training |
| `EVAL_STEPS` | e.g. `"40"` to evaluate only the last checkpoint |
| `EVAL_GPUS` | GPUs for eval engines (default: `GPU_IDS`) |
| `LAMBDA_PRES`, `RUN_ID` | e.g. `0` and `grit_3b_lam0` for the λ=0 baseline |
| `MEM_CONFIG` | force one memory setting, e.g. `8:2:32:0.30:0` (the original, most conservative one) |

## If something goes wrong

| Message | Meaning / action |
|---|---|
| `ERROR: /mnt/local/uvenvs/grit does not exist` | Job 1 has not run (or failed). |
| `ERROR: missing .../config.json` or `SHA256SUMS.json` | Job 2 has not run (or is incomplete). |
| `ERROR: GPU <id> has only ... MiB free` | Someone else is using that GPU (`nvidia-smi` is printed). Retry later or change `GPU_IDS`. |
| `train FAILED (not an out-of-memory error)` | Real error: the traceback is printed right above it (also in `logs/train.log`). |
| `train ran OUT OF MEMORY with MEM_CONFIG=...; falling back` | Training resumes with a safer memory setting. Nothing to do. |
| `training ran out of memory with every MEM_CONFIG` | Even the smallest setting does not fit: see `logs/train.log`. |
| Job stopped with `FAILED/STOPPED (exit 143)` | Timeout. Resubmit Job 3 (flip `#v1`/`#v2`). |

The code only reads local files and writes local outputs on the server.
