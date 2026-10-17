# GRIT

GRPO safety training that keeps the base model's general capability with two
protections: a null-space projection of the update (NSPO-style, per MLP layer)
and a KL trust region anchored to the frozen base model. This folder runs it
on **Qwen2.5-3B-Instruct** with **Qwen3Guard-Gen-4B** as the reward model on
**four B200s (180 GB, GPUs 4,5,6,7)**, on the offline server. **To run it, follow [RUN.md](RUN.md).** See [PLAN.md](PLAN.md) for the
method, the fixed configuration and the design decisions.

## Run

1. Environment: `../grit.txt` (uv, Python 3.11) builds `/mnt/local/uvenvs/grit`.
2. Download (`#d` job): the contents of [download.txt](download.txt). Models, the
   pre-built GRIT artifacts (`fushinguyenex/GRIT-artifacts`, ~5 GB) and the eval
   benchmarks. Nothing is downloaded at run time. `allenai/wildguard` and
   `sorry-bench/sorry-bench-202406` are gated: the HF account that runs the download must
   accept their terms first.
3. Run (`#1` job): `cd ./GRIT && bash ./project_commands.sh`

| Stage | What it does |
|---|---|
| `train` | 40 GRIT steps, checkpoint and validation every 5 steps; starts with the most aggressive memory config in `MEM_CONFIGS`, resumes from the latest checkpoint and falls back to the next config on failure |
| `eval` | base + checkpoints (`EVAL_STEPS`, default `20 40`) evaluated in parallel (`EVAL_SLOTS` vLLM engines per GPU): safety (PKU-SafeRLHF 1,000, AdvBench 520, HarmBench test DirectRequest 240, JailbreakBench 100, SORRY-Bench 450, HarmfulQA 1,960, ALERT 14,763: WildGuard harmful-response rate), over-refusal (XSTest safe, OR-Bench hard-1k, AlpacaEval: WildGuard refusal rate on benign prompts), MMLU, SuperGPQA, GSM8K, MATH-500, OlympiadBench, LiveCodeBench |
| `summary` | training curve + eval table + paired deltas vs base, printed and saved |

`commands.sh` runs only `cd ./GRIT` + `bash ./project_commands.sh`. Settings are the
defaults at the top of `project_commands.sh`: `GPU_IDS` (the B200s assigned to GRIT, default
`4,5,6,7`), `RUN_ID`, `LAMBDA_PRES` (default 1; 0 is the baseline), `MAX_STEPS`, `STAGES`
(comma list of the stages above), `EVAL_STEPS`, `EVAL_MAX_SAMPLES`, `MEM_CONFIGS` /
`MEM_CONFIG`, `EVAL_GPUS`, `EVAL_SLOTS`. When running by hand, the same names work as
env vars.

## GPU memory

- **Train:** the last GPU in `GPU_IDS` (7) runs the FP32 vLLM rollout alone; the others
  (4,5,6) run the FP32 actor with `torchrun`, one rank per GPU, each with its own
  Qwen3Guard. The 321 prompts and 48 KL contexts are split across the ranks (107 + 16 each)
  and gradients are summed, so the update is the same as on one GPU. A memory config is
  `TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP`; the default list is
  `16:4:64:0.85:0 8:2:32:0.85:0 4:1:16:0.85:0`. Sleep mode (`--rollout-sleep`) is still
  available for a colocated single-GPU layout.
- **Eval:** one vLLM engine per GPU (`EVAL_SLOTS=1`) on all four GPUs, so base, step 20
  and step 40 each get a whole GPU. Each engine runs up to 512 concurrent sequences.


## Outputs

- `/mnt/local/aiskylimit_new_nothingnew_2/grit/outputs/<RUN_ID>/step_XXXXXX`:
  checkpoints (weights for every saved step; optimizer state only in the latest).
- `/mnt/local/aiskylimit_new_nothingnew_2/grit/eval/<RUN_ID>/`: raw generations and WildGuard labels.
- `GRIT/results_<RUN_ID>/`: small files to fetch with a `#2` job: `logs/`,
  `train/metrics.jsonl`, `train/diagnostics_summary.json`, `train/validation/`,
  `eval/eval_results.json`, `summary.md`, `summary.json`.

## Code

| Path | Content |
|---|---|
| `grit/` | GRIT math: GRPO objective, AdamW predictor, projection, trust region, curvature (HVP), persistent vLLM rollout |
| `scripts/train_grit.py` | trainer (FP32 actor on torchrun ranks, vLLM on its own GPU, local checkpoints only) |
| `scripts/eval_run.py`, `eval_worker.py`, `eval_lib.py` | evaluation (vLLM generation, WildGuard judging, scoring, paired bootstrap) |
| `scripts/summarize_results.py` | console + markdown summary |
| `third_party/` | official scorers of OlympiadBench and LiveCodeBench, vendored unchanged (MIT), see `third_party/README.md` |

The code only reads local files and writes local outputs on the server.
