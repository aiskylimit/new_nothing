# ICLR 2027 main experiments

`PLAN.md` is the supplied finalized plan. This entry point schedules **15 runs**:
five native methods × A (7B/original), B (7B/Gemini), C (1.5B/original), one
training seed 42, default physical GPUs **3,4,5**. No 8B, extra training seeds or
CoReD ablations are scheduled. `configs/*.json` are the 15 frozen configurations;
they are validated against `common.matrix()` to prevent unnoticed drift.

## Environments and immutable assets

Keep the five training environments from `palign.txt`, `sft.txt`, `sgl.txt`,
`SegmentSelectiveSFT.txt`, and `cored.txt`: their stacks are incompatible.
`rebuttal.txt` provisions the common coordinator/evaluation environment. The code
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
The common vLLM evaluator uses the P-ALIGN environment for every method.

## Prepare and validate on B200

```bash
python -m rebuttal.data --assets rebuttal/assets --out rebuttal/prepared
python -m rebuttal.preflight --assets rebuttal/assets --data rebuttal/prepared \
  --runs rebuttal/runs --environments --gpus 3,4,5
```

Original is `simplescaling/s1K-1.1`: `deepseek_thinking_trajectory` plus
`deepseek_attempt`. Gemini is public `simplescaling/s1K`:
`thinking_trajectories[0]` plus `attempt`. The two sets must have exactly 1,000
unique, matched questions. They are reordered to the same stable question IDs;
missing traces and question-set mismatches raise errors. No generic label-SFT
path is used. Teacher lengths/correctness are saved per teacher/backbone under
`prepared/statistics/`. Correctness uses math-verify, whose limitations on proof
and free-text targets are recorded; it is not an independent teacher regrade by
an LLM. `--skip-statistics` is for fixtures only; production preflight rejects
missing/stale statistics.

Adapters preserve each method's response format. SGL receives aliases pointing
to the selected teacher, never silently to DeepSeek in B. SSFT keeps reasoning
as its solution and its native attribution target policy (last boxed answer,
fallback gold, <=200 characters); rejected IDs/counts are saved. P-ALIGN reruns
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
persists completed examples; SSFT repeats an interrupted attribution stage, but
does not repeat a completed one. CoReD uses its stage/sample resume path.
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

`cost_setting_A.md` is the primary table, with B/C tables and CoReD stage details
in JSON. Throughput, optimization steps, process/library versions and all attempt
logs are retained. Evaluation is excluded from training cost.

Local tests validate scheduling/statistics/MC normalization, resume records and
CPU distributed training. They do not establish CUDA/Unsloth/vLLM compatibility,
memory fit, throughput or the 15 trained results. Run server preflight, small
smokes and inference pilot before committing the full GPU workload.
