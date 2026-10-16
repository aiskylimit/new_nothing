# GRIT on the offline B200 server — project plan

Scope: **Qwen2.5-3B-Instruct on 4 × B200 (180 GB, GPUs 4,5,6,7)**, packaged for the offline
submit-and-wait framework of this repo (`aiskylimit/new_nothingnew_2`, branch `GRIT`).
Everything for this project lives in `GRIT/`. Files outside `GRIT/` are only touched
when explicitly approved (see §6).

---

## 1. Method (what GRIT does)

**Goal:** make the policy safer with GRPO while preserving the base model's general
capability through two protections: a null-space projection (NSPO-style) and a KL
trust region anchored to the base model.

### One-time preparation (from the base model)

| Set | Size | Source | Use |
|---|---|---|---|
| Projector set | 1,000 prompts (334 general / 333 math / 333 code) | `tatsu-lab/alpaca_farm`, `openai/gsm8k`, `newfacade/LeetCodeDataset` | Per-MLP-layer null-space projector `P` from base activations (`relative_threshold 5e-4`, compact form `UUᵀ`) |
| KL set | 1,920 prompts (640 per domain, disjoint from projector set) = 48 contexts × 40 steps | same three sources | Base next-token distribution per token, stored as top-64 + 1 tail bucket |
| Task set | 11,000 prompts | `PKU-Alignment/PKU-SafeRLHF` train (seed 66) | GRPO prompts |
| Safety eval | 1,000 prompts | `PKU-Alignment/PKU-SafeRLHF` test, disjoint from all sets above | Unsafe rate, length, refusal rate |

### Each training step

1. **Safety signal:** 5 responses per prompt, scored by Qwen3Guard (safe = 0,
   unsafe/controversial = −1). GRPO gives gradient `g`.
2. **Predictor step:** AdamW step from `g`, projected through `Q = P + γ(I − P)`,
   `γ = 0.05` → `Δ_task`.
3. **Trust region:** at `θ + Δ_task`, measure KL to the base over 65 layers on 48
   contexts. Tokens beyond `ε_pres` are pulled back to the KL-ball boundary → gradient `v`.
4. **Correction:** map `v` back to the current point (1st order via the AdamW
   Jacobian, 2nd order via a central finite-difference HVP with radius `±ρ`) → `c`.
5. **Update:** `θ_new = θ + Δ_task − lr · λ_pres · c`.

---

## 2. Lessons carried over from the 0.5B runs

1. vLLM in bf16 silently drops ~99.9% of the update → **rollout must be FP32**.
2. λ=0, FP32: unsafe rate 0.22 → 0.001, but validation length 85 → 13 tokens and
   99.6% of advantage groups are zero at step 40 (reward collapse).
3. General capability held (MMLU −1.2 pt; GSM8K, MATH-500, OlympiadBench, SuperGPQA,
   LiveCodeBench within 95% CI), but **heavy over-refusal**: 62.6% of AlpacaEval
   prompts refused (base 9.8%), length 122 vs 377 tokens.
4. Cause: a safe/unsafe-only reward makes refusal optimal. **Fixing the reward is the
   single most important change** (§4.7).
5. The preservation correction is only ≈1e-5 of the task step. If λ=1 does not differ
   from λ=0, raise λ to 10–100 or route the correction through `1/D`.

---

## 3. Fixed 3B configuration

| Parameter | Value |
|---|---|
| Policy | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Reward / judge | `Qwen/Qwen3Guard-Gen-4B` @ `6ec42827da0c1ff11e7a49dc269d2e810d27e108`, bf16 |
| Layout | GPUs 4,5,6,7: actor FP32 on 4,5,6 (torchrun, world size 3, Qwen3Guard per rank), vLLM FP32 rollout alone on 7 |
| Actor | FP32, TF32 off, attention `sdpa`, gradient checkpointing |
| `rollout_dtype` | `float32` |
| prep dtype / `base_statistics_dtype` | `float32` / `float32` (must match; the trainer checks) |
| `rollout_memory_utilization` | 0.85 (vLLM has GPU 7 to itself) |
| `task_microbatch_size` / `preservation_microbatch_size` / `reward_batch_size` | 16 / 4 / 64 first; on CUDA OOM 8 / 2 / 32, then 4 / 1 / 16 (`MEM_CONFIGS`) |
| Batch | 321 prompts × 5 responses, 48 KL contexts per step, top-64 + tail |
| `max_prompt_length` / `max_response_length` / `max_preserve_length` | 512 / 256 / 2304 |
| `lr` / `weight_decay` / `adam_eps` | 1e-6 / 0 / 1e-8 |
| `projector_relaxation` | 0.05 |
| `epsilon_pres` | 0.05 (**pass explicitly**; CLI default is 1e-3) |
| `lambda_pres` | 1 (λ=0 only as a baseline) |
| `use_curvature` / `fd_radius` / `fd_check_interval` | True / 0.05 / 10 |
| `validation_steps` / `save_steps` / `max_steps` | 5 / 5 / 40 |
| `seed` | 66 |

**Memory budget (estimate):** actor in the curvature phase ~62 GB, activations/logits
20–40 GB, Qwen3Guard 9 GB, vLLM FP32 at 0.30 ~54 GB → ~145–165 / 180 GB.
Host RAM ≥ 128 GB (AdamW moments, saved weights, gradient offload ~83 GB).

**Using the B200 fully:**
- Prepare: batched vLLM generation, KL statistics in a single forward
  (`--generation-backend vllm --statistics-mode full`), `eigh` on GPU.
- Train: raise microbatches until peak VRAM is ~170 GB.
- Eval: vLLM bf16 with large batches.

---

## 4. Adapting GRIT to this framework

How the original plan (written for a generic shared repo and a cloud runner) maps onto this
server.

| Original assumption | In this framework |
|---|---|
| Repo `aiskylimit/new_nothing`, branch `grit-3b`, folder `grit/` | Repo `aiskylimit/new_nothingnew_2`, branch **`GRIT`**, folder **`GRIT/`**. Push only this branch; the maintainer merges. |
| `run_all.sh` creates the venv and installs packages | The server builds the env from a root file **`grit.txt`** (uv, Python 3.11) into `/mnt/local/uvenvs/grit/`. The run script only `source`s it and never runs `pip install` (no internet). |
| Models/datasets downloaded at run time from HF | Server is **offline** (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`). Everything is pre-fetched through **`GRIT/download.txt`** (`#d` job), and code loads from local paths only. |
| One file for the full flow | **`GRIT/project_commands.sh`**, launched from root `commands.sh` (`#1 +N` job): `cd ./GRIT && bash ./project_commands.sh`. |
| Cloud-runner volumes, notebooks | Removed. Plain Python scripts plus one shell entry point. |
| Sending checkpoints off the server | Removed. Checkpoints stay on local disk under `/mnt/local/@PROJECT@/grit/outputs/`. |
| AlpacaEval judged by `gpt-4o-mini` | **Off.** No outbound API on the server, and sending data out is not allowed. AlpacaEval prompts are still used locally for refusal rate and length (judged by Qwen3Guard's Refusal field). |
| `vllm==0.15.1` (not verified on sm_100) | Use the stack already proven on this B200 / CUDA 12.8: **`torch==2.9.0`, `vllm==0.18.0`, `numpy==1.26.4`**. |
| Results read interactively | Logs go to the console (`tee` to a file). Results are fetched with a `#2 -f-<path>` job, so the summary must be a small `summary.md` / `summary.json`. |

### 4.1 Server paths

`@PROJECT@` = `aiskylimit_new_nothingnew_2`. Base dir: `/mnt/local/@PROJECT@/grit/`.

```
/mnt/local/@PROJECT@/grit/
  models/Qwen2.5-3B-Instruct
  models/Qwen3Guard-Gen-4B
  data/raw/<dataset>/            # HF snapshots from download.txt
  prepared/                      # snapshot of fushinguyenex/GRIT-artifacts (see 4.2.1)
    Qwen--Qwen2.5-3B-Instruct/
      SHA256SUMS.json
      data/task/task_train.parquet
      data/eval/evaluation_prompts.parquet
      data/preservation/split_v1/...
      artifacts/qwen--qwen2.5-3b-instruct_split_v1_projectors.pt
  outputs/<RUN_ID>/step_XXXXXX   # checkpoints (~12 GB each in FP32): kept on /mnt/local only
  eval/<RUN_ID>/                 # raw eval generations
```

**Where outputs go (decided):** checkpoints, projectors and raw generations stay on
`/mnt/local/@PROJECT@/grit/`. Small results (logs, `metrics.jsonl`,
`diagnostics_summary.json`, eval scores, `summary.md`/`summary.json`) are written inside the
project folder under `GRIT/results_<RUN_ID>/`, as in `offline_olmo7b_b200`, so a `#2` job
can fetch them easily.

### 4.2 `download.txt` (implemented)

See [download.txt](download.txt). Only what the first run needs:
- **Models:** Qwen2.5-3B-Instruct and Qwen3Guard-Gen-4B. On 2026-10-13 HF `main` of both equals
  the pinned SHAs (`aa8e7253…`, `6ec42827…`).
- **Prepared artifacts:** `fushinguyenex/GRIT-artifacts`.
- **Eval snapshots:** mmlu, SuperGPQA, gsm8k, MATH-500, OlympiadBench.
- **Single files via `--url`:** LiveCodeBench `test.jsonl` (release_v1, 1.25 GB, instead of the
  4.5 GB repo) and AlpacaEval `alpaca_eval.json`.

A `--hf-dataset` job yields a raw snapshot of the repo files, so code reads parquet/json
files by path and never calls `load_dataset(<repo id>)`. The raw training sources
(PKU-SafeRLHF, alpaca_farm, gsm8k train, LeetCodeDataset) are not downloaded, because the
artifacts already contain the task/eval/preservation sets. alpaca_farm and dolly are added
when the reward fix (§5.7) is implemented.

The download tool has no known syntax for pinning a revision.

#### 4.2.1 Prepared artifacts (`fushinguyenex/GRIT-artifacts`)

All preparation for Qwen2.5-3B was already done and published as a public HF dataset.
It covers the task set, safety eval set, projector and KL prompts, base contexts with
top-64 float32 statistics, and projectors. The folder keeps the layout that prepare
writes, so the trainer's own manifest checks still apply. Checked on 2026-10-13
(read-only, via the public HF API):

| Item | Value |
|---|---|
| Size | ~5.0 GB (projectors 4.1 GB, monitor contexts 0.87 GB) |
| `base_revision` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` (matches §3) |
| `generation_dtype` / `base_statistics_dtype` | `float32` / `float32` (matches §3) |
| Generation | vLLM, greedy, seed 66, `max_new_tokens 256`, `max_prompt_length 2048` |
| Monitor (KL) set | 1,920 contexts (640 math / 640 code / 640 general), `statistics_mode full`, top-64 |
| `full_vs_incremental_max_abs_diff` | 3.2e-5 (self-check passed) |
| Projector set | 1,000 contexts (334 general / 333 math / 333 code) |
| `tokenizer_sha256` | `563a118b…eb82a` |

Consequences:
- **Prepare becomes verify** (a dev-only script, not in the repo). It checks:
  - every file against `SHA256SUMS.json`;
  - the manifests;
  - that the local tokenizer fingerprint equals `tokenizer_sha256` (verified locally on
    2026-10-13 with transformers 4.57.6: match);
  - projector coverage and disjointness;
  - a base-logit KL spot check on 8 monitor contexts.

  Nothing is regenerated on the server. The original prepare scripts
  (`prepare_preservation_data.py`, `build_projectors.py`, …) were not ported, because they
  load from the Hub and the published artifacts replace them.
- **The model snapshot must match.** `--hf Qwen/Qwen2.5-3B-Instruct` pulls the current `main`.
  If its tokenizer or weights differ from `aa8e7253…`, the base KL statistics no longer
  describe that model. The tokenizer-hash check catches tokenizer drift. Weight drift is
  checked by comparing the base model's logits on a few monitor contexts against the stored
  top-64.
- **The trainer's revision check:** a local snapshot directory has no Hub revision, so
  `--model-revision` is taken as declared (default = pinned SHA). It is still guarded by the
  parameter count, the per-row tokenizer fingerprint and the verify-stage logit check.
  Profiles match a local directory by its name (`.../Qwen2.5-3B-Instruct`).
- The pre-built artifacts are a public dataset fetched by the server tool (`#d` job), like the benchmarks.

### 4.3 Environment file `grit.txt` (repo root, created)

Pins: `torch==2.10.0`, `vllm==0.18.0`, `transformers==4.57.6`, `numpy==2.2.6`,
`datasets==4.1.1`, `pyarrow==21.0.0`, `accelerate==1.10.1`, `math-verify[antlr4_11_0]` and
`antlr4-python3-runtime==4.11.0`.

Corrected on 2026-10-13. The first version copied `tropic.txt` (torch 2.9.0, numpy 1.26.4), which
cannot resolve, for two reasons:
- vllm 0.18.0 pins `torch==2.10.0`;
- vllm 0.18.0 pulls `opencv-python-headless>=4.13`, which needs numpy ≥ 2.

The new pins match `multi-mode-distill-eval.txt` and `reasoning-velocity-distill-eval.txt` on this
server. `uv pip compile` for Linux x86_64 and Python 3.11 resolves 192 packages (CUDA 12.8
runtime). vLLM runs FP32 rollouts with its FLEX_ATTENTION backend, because FlashAttention is
fp16/bf16 only. Never use wheel URLs.

### 4.4 `project_commands.sh` flow (implemented)

See [project_commands.sh](project_commands.sh) and [README.md](README.md). Stages:
`train` (40 steps, checkpoint + validation every 5; resumes from the latest
checkpoint) → `eval` (base + `EVAL_STEPS`, default 20 and 40) → `summary`.

Overrides: `GPU_IDS`, `RUN_ID`, `LAMBDA_PRES`, `MAX_STEPS`, `STAGES`, `EVAL_STEPS`,
`EVAL_MAX_SAMPLES`. Every stage is resumable, so a long run can be split across several
`#1 +N` jobs.

**GPU memory (added 2026-10-13):**
- GRIT got 4 GPUs (4,5,6,7). The trainer's distributed path is used: rollout on 7,
  3 actor ranks on 4,5,6, gradients summed across ranks (covered by the two-rank test).
  vLLM sleep mode (level 2) is kept for a colocated single-GPU layout but is off by default.
- The train stage starts with the most aggressive `MEM_CONFIGS` entry and falls back to the
  next one, resuming from the latest checkpoint, if a step fails. The check, verify and dry-run
  stages were removed from the flow (2026-10-13); the verify script, the tests and
  `config/` are dev-only and git-ignored.
- Eval runs one vLLM engine per GPU on all four GPUs (`EVAL_SLOTS=1`), so base, step 20 and
  step 40 are generated concurrently, each on a whole GPU.

**Disk:** a 3B FP32 checkpoint is ~12 GB of weights plus ~25 GB of AdamW state
(`training.pt`). `--prune-old-training-state` keeps `training.pt` only in the latest
checkpoint, which brings 8 checkpoints from ~300 GB down to ~120 GB. Older checkpoints remain
evaluable but not resumable.

---

## 5. Code changes (status 2026-10-13)

1. **Cloud runner removed (done):** none of its app file, the notebooks or its env vars was ported.
2. **Outbound code removed (done):** everything that sent checkpoints or artifacts off the server,
   and all credential handling, was left out of the port.
3. **Colocated 3B allowed (done):** the 0.5B-only restriction is gone.
4. **Local paths only (done):**
   - Profiles accept local snapshot dirs (`grit/model_profiles.py`).
   - Revisions are passed to `from_pretrained` / vLLM only for Hub ids.
   - `torch.load` of projectors is explicit `weights_only=False` (torch ≥2.6 default broke it).
5. **Prepare → verify (done):** dev-only, git-ignored. The prepare scripts were not
   ported (see §4.2.1).
6. **Eval (done):**
   - `scripts/eval_run.py` (orchestrator), `eval_worker.py` (vLLM generate / Qwen3Guard guard)
     and `eval_lib.py` (ported from `grit_general_benchmark_v2.ipynb`, local snapshot loaders,
     local judging only).
   - PKU safety reports `unsafe_%` (Unsafe or Controversial = the reward rule), `guard_refusal_%`
     and length.
   - AlpacaEval reports over-refusal (`guard_refusal_%`, `refusal_regex_%`) and length.
   - Over-refusal suites (added 2026-10-13), each paired with harmful counterparts:
     - XSTest (`Paul/XSTest`): 250 safe + 200 unsafe prompts.
     - OR-Bench (`bench-llm/or-bench`): hard-1k (1,319) + toxic (655).
     - Metrics: `over_refusal_%` = Qwen3Guard Refusal=Yes on benign prompts; `unsafe_%` on
       harmful prompts.
     - The summary prints a "Safety vs over-refusal" table.
     - XSTest and OR-Bench must never be used as training data (§5.7).
   - **Official scorers (added 2026-10-13):** OlympiadBench and LiveCodeBench are scored with
     their authors' code, vendored in `third_party/`.
     - OlympiadBench uses the official prompt template, answer extraction and
       `AutoScoringJudge` with the `error` precision. It agrees 36/36 with the authors' examples
       and scores 673/674 gold answers correct; the one exception is a malformed gold answer in
       the dataset.
     - LiveCodeBench uses the CodeQwenInstruct prompt, official `extract_code` and
       `codegen_metrics` (6 s per test, pass@1).
     - Requires ANTLR 4.11 (`grit.txt`).
     - MMLU and SuperGPQA are scored exactly (zero-shot, one constrained answer letter) but do
       not follow the official protocols (MMLU 5-shot, SuperGPQA zero-shot CoT). Decided on
       2026-10-13 to keep this: it is cheaper and sufficient for base-vs-checkpoint deltas.
   - Paired bootstrap deltas compare every checkpoint with base.
   - Loaders and scorers were tested locally on the real snapshots: all row counts match.
7. **Reward fix: deferred to the second run (decided 2026-10-13).** The first 3B run keeps the
   original safe = 0 / unsafe = −1 reward so it is directly comparable with the 0.5B results.
   Implement the fix behind a flag (default off) for the next run:
   - Mix in ~25% harmless prompts (alpaca_farm / dolly), deduplicated against the projector, KL and eval sets.
   - Do not use XSTest or OR-Bench.
   - A refused harmless prompt scores −1. The refusal label comes from Qwen3Guard-Gen's Refusal field.
8. **Summary (done):** `scripts/summarize_results.py`.
9. **Clean content (done):** code only, `.gitignore` for generated data and `results_*/`.
10. **Checks (done):**
    - 48/48 CPU tests pass locally (torch 2.14 CPU).
    - vLLM 0.18.0 source still has `LLM.apply_model` and `logprobs_mode="raw_logprobs"`.
    - **Not verified:** the CUDA/vLLM path itself. The first train job on the server is the first real check.

### Known issue (only if preparation is ever rerun)

The published artifacts passed the self-check in FP32 (3.2e-5), so this only matters if
preparation is rerun. `prepare_preservation_data.py` aborted at its self-check in another run:
`Full-forward statistics differ by 2.77e+00 > 0.001`. Full-forward and incremental
(KV-cache) statistics should agree.

- **Check 1, dtype:** rerun the check in FP32 (prep dtype must be `float32`). A bf16 run can produce a diff of this size from rounding alone.
- **Check 2, real bug:** if the diff persists in FP32, the incremental path has a real bug. Likely suspects are left-padding `position_ids`, the attention mask or `cache_position` during decoding, an off-by-one when aligning logits, or a different attention backend between the two paths.

---

## 6. Layout (implemented)

```
GRIT/
  PLAN.md  README.md  RUN.md  download.txt  project_commands.sh  .gitignore
  grit/                        # library: projection, curvature, trust_region, grpo, rollout, ...
  scripts/
    train_grit.py              # trainer (local snapshots, state pruning)
    preservation_data.py       # stored-context batching + split checks used by the trainer
    eval_run.py  eval_worker.py  eval_lib.py
    summarize_results.py
  third_party/                 # official OlympiadBench / LiveCodeBench scorers
```

Outside `GRIT/`:
- `grit.txt` at the repo root: the env spec (approved, created).
- Root `commands.sh`: needs approval before it is touched (job header + `cd ./GRIT && bash ./project_commands.sh`).

---

## 7. Open questions

1. ~~GPU index~~ Decided: GPUs 4,5,6,7 (`GPU_IDS` in project_commands.sh).
2. Time budget per `#1 +N` job. All stages are resumable, so splitting across jobs works either way.
3. ~~Reward fix in the first run?~~ Decided: no. The first run uses the original reward; the fix comes in the second run (§5.7).
4. Should the λ=0 baseline be part of the default flow?
5. Meaning of the `#v1` / `#v2` header line in `commands.sh`.
