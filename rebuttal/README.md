# ICLR 2027 main experiments

`PLAN.md` is the supplied finalized plan. This entry point schedules **15 runs**:
five native methods × A (7B/original), B (7B/Gemini), C (1.5B/original), one
training seed 42, default physical GPUs **3,4,5**. No 8B, extra training seeds or
CoReD ablations are scheduled. `configs/*.json` are the 15 frozen configurations;
they are validated against `common.matrix()` to prevent unnoticed drift.

## Main three-B200 launcher

For short commands that train all **A+C** runs before evaluation, see the
[root README](../README.md#chạy-lần-lượt-từng-bước-chỉ-setting-a-và-c).
Use `train --settings A C`, then `eval --settings A C` to omit B.
Use `--run-id` to select individual methods/runs instead.

```bash
# Offline B200 server: environments and attested assets must already be provisioned.
GPUS=3,4,5 bash rebuttal/project_commands.sh endtoend --offline
# Online machine with three B200s: also install dependencies and download assets.
GPUS=3,4,5 bash rebuttal/project_commands.sh endtoend --online
```

Choose the command matching the machine; `endtoend` defaults to offline. The
online mode runs `setup` (seven requirements files, isolated Python 3.11 environments)
and `download` (pinned model/data provisioner), then both modes run `prepare`,
`preflight`, `all` and `report` in order. The first failed stage stops the launcher.
Install `uv` before online setup if missing (`python3 -m pip install uv`). Runtime
stages remain offline. Logs are appended to `rebuttal/launcher-logs/endtoend.log`
or `REBUTTAL_LAUNCHER_LOGS`; native per-run status/logs retain their existing paths.
Set `GPUS` to the three actual physical indices allocated to the experiment.

`setup` and `download` can also be invoked separately, as in the H100/3090 scripts.
Download uses `rebuttal.provision` with the same sources listed in
`rebuttal/download.txt`, pinned revisions and `snapshot.json` hashes. For assets
already fetched by a managed downloader, use `download --attest-managed` once to
verify the local Hub commit metadata offline. Copy attested assets to the B200
server and provision compatible library environments there before offline runs;
the txt files alone are not offline installation bundles.

Existing `all` still means native training plus evaluation over the 15 configs;
it does not reinstall dependencies or alter the training/eval protocol. The
end-to-end wrapper changes no Python trainers, batch/epoch settings or frozen
configs. Use `REBUTTAL_ENV_ROOT` to relocate all seven environments and the usual
`REBUTTAL_ASSETS`, `REBUTTAL_DATA`, `REBUTTAL_RUNS`, `REBUTTAL_REPORTS` overrides
for storage. Resume with the same command, code, configuration and storage roots.

## Formal Setting B on two H100s

```bash
# Online only; skip if the seven environments and attested assets already exist.
bash rebuttal/formal_B_2h100.sh setup
bash rebuttal/formal_B_2h100.sh download
# End to end: real-data preparation, H100 preflight, five full train/eval runs, reports.
GPUS=0,1 bash rebuttal/formal_B_2h100.sh all
```

This separate formal profile uses the real pinned s1K-1.1 Gemini fields, all 1,000
canonical input rows, Qwen2.5-7B-Instruct, seed 42 and three epochs. Native
method filters remain active and their retained/rejected rows are recorded.
The user-approved topology adaptation is **batch 32**, microbatch 1 per GPU and
gradient accumulation 16 for the four baselines; CoReD uses exact global batch
32 with its adaptive microbatches and retains three experts. Training cutoffs
remain 32k for the baselines and 16k for CoReD. IG50, P-ALIGN generation/filtering,
LoRA/LR settings and the council/TIES/distillation recipe retain their main values.
There are no synthetic fixtures, smoke flags, update limits or shortened evals.
Memory fit and runtime still require verification on the actual H100 machine.

All four math benchmarks are evaluated with the normal 64 total responses per
question across seeds 1–5, 4,096 new tokens and math-verify. Setting B has no OOD.
The final report requires all five B evaluations and includes the four paired
CoReD-versus-baseline comparisons, bootstrap intervals and H100 cost tables.
It does not require A/C or produce missing-teacher/backbone comparisons.

Outputs default to `rebuttal/runs-B-2h100/` and `rebuttal/reports-B-2h100/`, with
separate profile/config snapshots and `launcher_status.json`. Override using
`REBUTTAL_H100_RUNS` / `REBUTTAL_H100_REPORTS`, not the B200 output variables.
Rerun the same `all` command to resume; changed code/config/data requires a fresh
output directory. Other actions: `prepare`, `preflight`, `train`, `eval`, `report`.
`all --run-id B_cored_7b_seed42` runs/resumes only that method and omits the final
five-method report. Run `report` after all five methods finish.
`project_commands.sh`, the frozen 15 main configs and their default three-B200
profile remain separate; do not combine H100/batch-32 timing with B200/batch-33
as if they used the same hardware/training batch.

## Environments and immutable assets

Keep the five training environments from `palign.txt`, `sft.txt`, `sgl.txt`,
`SegmentSelectiveSFT.txt`, and `cored.txt`: their stacks are incompatible.
`rebuttal.txt` provisions the common coordinator; `rebuttal_eval.txt` provisions
the separate shared evaluation environment. The code
is unified by orchestration, provenance, output layout, cost definitions and
evaluation; native method trainers/losses remain in their original folders.

On the **internet-connected provisioning machine**, from repository root:

```bash
python -m rebuttal.provision --assets rebuttal/assets
```

Copy those assets and code to persistent storage on B200. Provisioning pins both
model revisions, Original/Gemini training sources, four math benchmarks, and the
two OOD test files. It writes `snapshot.json` attestations with content hashes.
`download.txt` supports the existing managed downloader, but that downloader's
unpinned `--hf` syntax alone does not establish immutable revisions. For those
downloads, use `python -m rebuttal.provision --assets rebuttal/assets
--attest-managed`: it checks the Hub local-dir commit metadata offline and
rejects mismatches. Use the online pinned provisioner if metadata is absent.
Only `provision.py` accesses the Hub; runtime/evaluation stay offline.

Environment paths default to `/mnt/local/uvenvs/<folder>/bin/python`.
Override `REBUTTAL_PYTHON_LONGCOT`, `REBUTTAL_PYTHON_SGL`,
`REBUTTAL_PYTHON_SSFT`, `REBUTTAL_PYTHON_PALIGN`, `REBUTTAL_PYTHON_CORED` as needed.
The common vLLM evaluator uses **`rebuttal_eval`**, independently of every native
training environment. Override `REBUTTAL_EVAL_PYTHON` for an already-provisioned
evaluator, or `REBUTTAL_EVAL_VENV` to change the directory created by setup.

### Shared evaluator

`rebuttal_eval.txt` pins official Linux x86_64/CPython 3.11 CUDA 12.9 wheels:
Torch 2.13.0, torchvision 0.28.0, torchaudio 2.11.0 and vLLM 0.27.1+cu129.
These builds target driver >=575.51.03; no CUDA 13 Torch wheel is selected by
the eval installer. GPU execution still needs validation on the actual node.
The model/data sources, prompts, decoding budgets and math-verify rule are unchanged.
The [official vLLM installation guide](https://docs.vllm.ai/en/v0.27.1/getting_started/installation/gpu/)
describes CUDA-specific binaries and their PyTorch compatibility requirements.

Full `setup` in all three launchers installs the shared evaluator automatically.
For a fresh development smoke run:

```bash
bash rebuttal/smoke_2x3090.sh setup
bash rebuttal/smoke_2x3090.sh download
GPUS=0,1 bash rebuttal/smoke_2x3090.sh all --settings C
```

Main/H100 preflight and the development launcher automatically run these small
LoRA probes on every reserved GPU before training (outside training cost logs).
Reports/logs are under `<runs>/eval-preflight/`. This tests kernels and loading;
it does not certify 32k-context/batch memory fit. A failure stops before training.
Full `setup` now installs the evaluator as a seventh environment. Offline nodes
must provision it and its wheel dependencies first; a txt file is not a wheelhouse.

Evaluation records package/build versions in `evaluator_environment.json`.
All three launchers also support `setup-eval` to install only this environment,
and `check-eval --sizes 1.5b` for a standalone GPU probe with local assets.

This migration does **not** replace Torch/vLLM in `cored`, `palign` or other train
envs. Their GPU builds must separately match the driver. P-ALIGN/CoReD generation
preprocessing still runs in its native environment and can expose an independent
CUDA mismatch even after the shared evaluator passes.

## Prepare and validate on B200

```bash
python -m rebuttal.data --assets rebuttal/assets --out rebuttal/prepared
python -m rebuttal.preflight --assets rebuttal/assets --data rebuttal/prepared \
  --runs rebuttal/runs --environments --gpus 3,4,5
```

Both teachers use **one pinned `simplescaling/s1K-1.1` snapshot**. Original uses
`deepseek_thinking_trajectory` and `deepseek_attempt`; Gemini uses
`gemini_thinking_trajectory` and `gemini_attempt`. Teacher grades come from
`deepseek_grade` / `gemini_grade`. The two views are built from the same 1,000
rows with identical stable question IDs; missing teacher fields or exact duplicate
questions raise errors. Interior whitespace is preserved: the pinned source has
two distinct rows whose questions differ only in whitespace, with separate
teacher trajectories. There is no separate `simplescaling/s1K` download or
fallback to its legacy fields. No generic label-SFT
path is used. Teacher lengths/correctness are saved per teacher/backbone under
`prepared/statistics/`. Correctness uses math-verify, whose limitations on proof
and free-text targets are recorded; it is not an independent teacher regrade by
an LLM. `--skip-statistics` is for fixtures only; production preflight rejects
missing/stale statistics.

After updating from the earlier separate-Gemini-source setup, rerun `prepare`.
Preflight rejects old prepared manifests rather than silently reusing Gemini
traces with the wrong provenance. Existing checkpoints/results are not deleted.

Adapters preserve each method's response format. SGL receives aliases pointing
to the selected teacher, never silently to DeepSeek in B. SSFT keeps reasoning
as its solution and its native attribution target policy (last boxed answer,
fallback gold, <=200 characters); rejected IDs/counts are saved. Attribution runs
as three independent workers, one per GPU, with unchanged IG/segment selection.
Each worker writes a separate shard; the coordinator validates IDs and restores
source order before training. Empty shards are allowed for small smokes; zero
retained samples overall is an error. P-ALIGN reruns
sentence-level self-judged binary prefix search, student continuation, Eq.9
answer filtering, and prefix+continuation supervision for every setting/backbone;
it does not reuse the old fixed 966-row file. All accept/reject decisions are
persisted. Its continuation cap is 32,768 new tokens subject to the backbone's
native available context; overlong judge prompts fail explicitly.
The correctness filter follows [P-ALIGN Eq.9](https://arxiv.org/html/2601.10064v1).

The presets preserve the current rebuttal method implementations: baseline
LoRA rank/alpha 16, dropout .05, LR 5e-5, 3 epochs, batch 33; CoReD keeps its
16k cutoff, three experts, SVGD/key-step/council/TIES recipe and student LR 1e-4.
Other baselines keep 32k and their method-specific masks/truncation. These
presets do not certify that every inherited detail matches an original paper.
DeepSeek 1.5B uses LLaMA-Factory's `deepseekr1` template, not the Qwen template.

## Smoke tests and 15 training runs

For a separate development test on **two RTX 3090s**, use:

```bash
# Linux with Internet; install uv first if unavailable. Skip setup/download if provisioned.
bash rebuttal/smoke_2x3090.sh setup
bash rebuttal/smoke_2x3090.sh download
GPUS=0,1 bash rebuttal/smoke_2x3090.sh all
# Faster first check: the five native methods on the real 1.5B backbone only.
GPUS=0,1 bash rebuttal/smoke_2x3090.sh all --settings C
# Or one method, e.g. C_cored_1.5b_seed42.
GPUS=0,1 bash rebuttal/smoke_2x3090.sh all --run-id C_cored_1.5b_seed42
```

This dedicated entry point tests a real two-rank NCCL all-reduce and bf16 backward
in each needed environment, native method preprocessing, four optimizer updates,
checkpoint save/load, two vLLM workers, and math/MC scoring. It defaults to all
15 configurations using their real backbones, but **synthetic short arithmetic**
instead of s1K/benchmark data. It uses batch 2 (1/GPU), train cutoff 512, vLLM
context 2048/eager mode, up to 256 generated tokens, IG4, and one eval response.
CoReD retains three experts and runs prefix/key/Phase 1/council/TIES/distillation;
four Phase 1 updates on four examples reach the SVGD-active epoch.
P-ALIGN correctness filtering remains active; zero accepted examples or SSFT
attribution OOM is an error, not a successful smoke. GPU memory fit/runtime has
not been established until this runs on the target machine.
Short context, low concurrency and eager execution follow the
[vLLM memory guidance](https://docs.vllm.ai/en/v0.9.0/configuration/conserving_memory.html).

Outputs live in `rebuttal/smoke-2x3090/dev2_<run_id>/`, with aggregate
`summary.json` and `<runs>.launcher.log`. Rerun the same command to reuse completed
training/eval stages; changed code/config/data requires a fresh
`REBUTTAL_2GPU_RUNS`. Existing main/prepared data and the frozen main configs are
not used as writable test outputs. These results are not paper metrics or a
guarantee of B200/long-context/full-batch compatibility. The main entry point
below still requires three B200s and the original experimental protocol.

```bash
# Two input examples and two optimizer updates; isolated from main run IDs.
python -m rebuttal.runner smoke --runs rebuttal/smoke-runs

# All 15 runs sequentially, reserving the same three GPUs.
python -m rebuttal.runner train

# One selected run (repeat this command to resume).
python -m rebuttal.runner train --run-id B_sgl_7b_seed42
```

Smoke uses the same method-specific preprocessing, then small optimization runs.
CoReD smoke exercises both expert training and student distillation. A filter
retaining zero samples is an explicit failure; it does not silently train an
unfiltered substitute. Native checkpoint resume is enabled for all methods.
Preprocessing stages persist artifacts separately from optimization. P-ALIGN
persists completed examples; SSFT repeats all three shards of an interrupted
attribution stage, but does not repeat a completed one. CoReD uses its stage/sample resume path.
SGL retains its native mask-resume behavior. Resume can repeat work after the
last checkpoint; failed attempts remain in cost logs.

The scheduler locks `runs/.gpu.lock`; do not use different runs roots to run jobs
on the same GPUs concurrently. Main training and math/OOD evaluation are separate.
Each run has `run.json`, `status.json`, logs, method artifacts,
`final_checkpoint.json`, checkpoint/tokenizer/base configuration and
`training_cost.json`. LoRA checkpoints need only their adapter plus the immutable
base snapshot referenced in `checkpoint_manifest.json`; no mask, teacher or
training stage is needed at inference. Keep the shared base assets on persistent
storage. Git commit, dirty state, data hashes and training-code fingerprints are
recorded. Statistics/regrading code can change without retraining checkpoints.

## Unified evaluation

```bash
python -m rebuttal.runner eval
# Or train then evaluate each configuration:
python -m rebuttal.runner all
```

Each checkpoint uses three independent vLLM replicas, one GPU each, TP=1, with
disjoint questions. All methods use the P-ALIGN math prompt/chat-template helper
and math-verify extraction. Math uses seeds **1,2,3,4,5** with **13,13,13,13,12**
responses per question, temperature .6, top-p .9, repetition penalty 1.05 and
4,096 new tokens. The total is **64**, not 64 per seed. Context is 32,768.

OOD is only A/C: ARC test 1,172 questions, cap **1,024**; LogiQA2 English MRC test
1,572 questions, cap **4,096**. Both use temperature 0, one response, and a common
reasoning-allowed prompt ending in `Answer: X`. ARC numeric/variable choice
labels are mapped explicitly to display letters. LogiQA's passage and four
options are retained, and its zero-based answer is mapped to a letter. IDs include
the fixed snapshot row index because some source LogiQA IDs repeat.

Raw `evaluation/<benchmark>.rank*.jsonl` records contain the complete prompt/text,
run/checkpoint/benchmark hashes, question/seed/sample IDs, token counts, finish
and stop reasons, gold and decoding configuration. Prediction/correctness are
initially null; separate CPU scoring writes full `*.scored.jsonl` copies with
those fields populated. Raw generation is never overwritten by regrading.
Completed response keys are skipped, an incomplete final line can be repaired,
and interior corruption/duplicate keys are rejected. Each question-seed request
can be regenerated to recover missing sample indices without appending duplicates.
Completion is checked against the full response grid, not just file existence.

Regrade on CPU without generating again:

```bash
python -m rebuttal.evaluate --score-only \
  --config rebuttal/runs/A_cored_7b_seed42/run.json \
  --benchmarks rebuttal/prepared/benchmarks \
  --output rebuttal/runs/A_cored_7b_seed42/evaluation
```

Math grading follows `parse('$gold$')`, `parse(prediction)`, `verify`; MC grading
takes the final explicit answer letter (or boxed letter), ignores text inside
`<think>`, and rejects out-of-range/absent letters. `finish_reason=length` is
truncation; vLLM's `stop` and `stop_reason` are retained to distinguish EOS/stops.
Token caps stay fixed even when truncation is high.

## Analysis and cost tables

```bash
python -m rebuttal.statistics --runs rebuttal/runs --output rebuttal/reports
python -m rebuttal.cost --runs rebuttal/runs --output rebuttal/reports
```

Round-robin seed ordering gives the first 32 responses seed counts 7/7/6/6/6.
Avg@32 and Avg@64 share those same responses. Pass@1 is mean sampled correctness
(equal to Avg@64); Pass@3 is `1-C(64-c,3)/C(64,3)` per question then averaged.
All JSON metrics are fractions; Markdown tables display percentages.

95% CIs use 5,000 deterministic bootstrap resamples of **questions**. Paired
differences share question IDs. Math macro averages weight the four benchmarks
equally and bootstrap within each benchmark. Reports include CoReD-minus-baseline,
Gemini-minus-Original and 1.5B-minus-7B differences, per-seed accuracies, invalid
answer/truncation rates, mean/P95 lengths, raw failures and discordant questions.
The five seed-level means have 12–13 responses/question each; their variability
is not five independent Avg@64 runs or evidence of training-seed stability.
Missing runs cause final analysis to fail; `--allow-partial` labels an inspection
report as incomplete. AIME intervals may be wide.

Cost definitions are identical across methods:

- **Allocated GPU-hours:** training-job wall seconds × 3 / 3,600, summed across
  invocations, including failed attempts. This accounts for the reserved GPU set.
- **Active stage GPU-hours:** sum of each worker's wall seconds × its GPU count,
  including necessary prefix/attribution/mask/council/merge stages. CoReD is not
  reduced to student-only cost. End-to-end job wall time is recorded separately.
- **FLOPs are estimated**, not hardware-counter measurements: dense model-token
  costs `2*N*tokens` plus dense attention `4*layers*hidden*queries*keys`, then
  gradient/recompute equivalents, with explicit SVD/merge estimates.
  Optimizer and some IG bookkeeping are omitted; causal sparsity, prefix caching,
  cached decoding and adapter overhead can differ from this approximation.
  Missing counters remain null/incomplete, never fabricated zero. Do not label
  these as measured FLOPs in the paper.
- Torch peak allocated/reserved memory is per worker; nvidia-smi device-used
  memory is sampled separately each second. They are distinct metrics.
- Token counts are actual unpadded model-forward/autograd exposures, including
  repeated expert/teacher calls. Optimization-stage tokens are also separated;
  they are not assumed equal to unique dataset tokens. Native sampler padding,
  filtering and truncation can change budgets and must be reported.

SSFT's direct IG decoder/head calls have explicit counters. Reports check all
expected workers (including finalized idle shards) and required method stages;
missing counters or native CoReD timings are marked incomplete. Cost events are
committed before completion markers, and resume verifies the matching event.
An old coordinator marker without its cost event raises an accounting error;
it does not silently skip the stage or invent its missing cost.

`cost_setting_A.md` is the primary table, with B/C tables and CoReD stage details
in JSON. Throughput, optimization steps, process/library versions and all attempt
logs are retained. Evaluation is excluded from training cost.

Local tests validate scheduling/statistics/MC normalization, resume records and
CPU distributed training. They do not establish CUDA/Unsloth/vLLM compatibility,
memory fit, throughput or the 15 trained results. Run server preflight, small
smokes and inference pilot before committing the full GPU workload.
