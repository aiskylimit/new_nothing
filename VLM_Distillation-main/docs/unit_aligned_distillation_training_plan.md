# Unit-Aligned Distillation Training and Evaluation Plan

## Goal

Validate the refined paper claim:

> Joint text-span alignment and vision-token matching improve cross-architecture VLM distillation most when tokenizer mismatch and vision-token mismatch are both high.

This is not just an EMKD or SRE component study. The final claim requires four method rows across multiple teacher/student regimes:

- SFT baseline
- EMKD-only
- SRE-only
- Joint SRE+EMKD

## Required Code State

Before the final grid starts, the training code must support:

- `--kd_loss_type emkd` for vision-token matching and VL affinity.
- `--kd_loss_type sre` for assistant-response span alignment.
- `--kd_loss_type joint` or `--kd_loss_type unit_aligned` for combined SRE+EMKD.
- SRE pooler preparation for both `sre` and joint modes.
- EMKD weights controlled by CLI args: `em_kd_alpha`, `em_kd_beta`, `em_kd_gamma`, `em_kd_temperature`.
- A safe default `kd_loss_type` that does not crash if a teacher is provided and the script forgets to override it.

## Main Training Matrix

Minimum publishable grid:

| Pair | Regime | Methods |
|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct -> Qwen/Qwen2-VL-2B-Instruct` | Low/medium UMI control | SFT, EMKD, SRE, Joint |
| `Qwen/Qwen3-VL-8B-Instruct -> Qwen/Qwen2.5-VL-3B-Instruct` | High tokenizer + vision mismatch | SFT, EMKD, SRE, Joint |
| `lmms-lab/llava-onevision-qwen2-7b-ov -> Qwen/Qwen2.5-VL-3B-Instruct` | Cross-family, high vision-token mismatch | SFT, EMKD, SRE, Joint |

This is 12 runs total. The SFT row is per student model, so the Qwen2.5-VL-3B SFT checkpoint can serve both the Qwen3 and LLaVA-OneVision teacher comparisons if the training data and compute budget are matched.

## Multi-GPU Launch Defaults

Use `torchrun` for all training runs. Every training script uses `--nproc_per_node gpu`, so the process count is inferred from the GPUs visible through `CUDA_VISIBLE_DEVICES`.

Recommended defaults for the first server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 bash <script>.sh
```

Keep per-device batch size at `1` for VLM distillation and scale throughput with:

- the number of GPUs listed in `CUDA_VISIBLE_DEVICES`
- `gradient_accumulation_steps`
- `dataloader_num_workers`

Use these EMKD caps for the first multi-GPU runs:

```bash
--em_kd_max_vision_tokens 512
--em_kd_max_text_tokens 1024
```

For LLaVA-OneVision teacher runs, do not disable the vision-token cap unless the server has already passed a long smoke test. LLaVA-OneVision can emit thousands of vision tokens, and uncapped Hungarian matching is the most likely bottleneck.

The joint launch script is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PERCENT_DATA=0.01 bash script_train/unit_aligned/train_joint.sh
```

Override models per pair without editing the script:

```bash
STUDENT_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" \
TEACHER_MODEL="Qwen/Qwen3-VL-8B-Instruct" \
RUN_NAME="qwen3_teacher_8b_qwen25_student_3b_joint" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PERCENT_DATA=0.01 \
bash script_train/unit_aligned/train_joint.sh
```

## Run Order

### Phase 0: UMI Probe

Compute UMI before training:

- Text component: shared-vocab Jaccard over response tokens emitted in a fixed instruction-tuning probe set.
- Vision component: average `1 - min(n_v_student, n_v_teacher) / max(n_v_student, n_v_teacher)` over about 1K images.

Report one row per pair:

| Pair | Shared response-vocab Jaccard | Mean student vision tokens | Mean teacher vision tokens | UMI |
|---|---:|---:|---:|---:|

### Phase 1: Tiny Smoke

Use `1K-5K` samples on the Qwen2.5-7B -> Qwen2-2B pair.

Train:

- SFT
- EMKD
- SRE
- Joint

Success criteria:

- All four jobs load models and processors.
- Collator builds labels and, for SRE/Joint, span pooler tensors.
- Loss is finite for at least 100 optimizer steps.
- Checkpoints save and resume.
- No unexpected OOM at chosen `max_len`.

### Phase 2: First Pilot

Use `50K` samples on:

`Qwen/Qwen2.5-VL-7B-Instruct -> Qwen/Qwen2-VL-2B-Instruct`

Train:

- SFT
- EMKD
- SRE
- Joint

Evaluate on:

- MMBench-EN-dev
- MMStar
- MMMU-val
- MathVista-MINI

Decision:

- If EMKD and SRE both fail to beat or match SFT, fix component losses before scaling.
- If Joint is unstable, sweep down `em_kd_gamma` and joint EMKD/SRE balance before training larger pairs.

### Phase 3: Main Grid

Use `100K-200K` samples first. Increase to `300K+` only after the pilot shows signal.

Train all rows in the main matrix.

Primary benchmarks:

- MMBench-EN-dev
- MMStar
- MMMU-val
- MathVista-MINI
- MM-Vet
- AI2D
- ChartQA
- POPE

Optional expansion:

- RealWorldQA
- BLINK
- OCRBench
- DocVQA-val

Recommended eval order:

1. `MMBench-EN-dev`, `MMStar`, `MMMU-val`, `MathVista-MINI` for fast signal after every smoke/pilot checkpoint.
2. Add `MM-Vet`, `AI2D`, `ChartQA`, `POPE` for the main result table.
3. Add `RealWorldQA`, `BLINK`, `OCRBench`, and `DocVQA-val` only after the 8-benchmark table shows the joint method is worth scaling.

Eval launcher contract:

- Use `scripts/eval/run_eval.sh`; it generates a VLMEvalKit `--config` file and calls `run.py --config`.
- Do not rely on a direct `run.py --model-path` invocation. Current VLMEvalKit evaluation should go through the config system or a verified model wrapper.
- LoRA adapter-only checkpoints are auto-merged into `outputs/eval/merged_checkpoints/<checkpoint_name>/` before VLMEvalKit runs. Set `MERGE_LORA=false` to require a pre-merged HF checkpoint.
- For Qwen local checkpoints, keep the default `VLMEVAL_MODEL_CLASS=Qwen2VLChat` unless the checkpoint family changes.

## Paper Decision Criteria

The refined claim is supported if:

- Low/medium UMI pair: Joint is within `0.5` average points of the best single-side method.
- High-UMI pairs: Joint beats the better of EMKD-only and SRE-only by about `1.0-1.5` average points.
- Joint improvement increases with UMI across pairs.
- Practical non-additivity is positive:

```text
Delta_joint - max(Delta_emkd, Delta_sre) > 0
```

The stronger interaction test is:

```text
Delta_joint - (Delta_emkd + Delta_sre) > 0
```

Use the stronger test only if the baseline-normalized deltas are stable; otherwise report the practical joint-over-best-single result and frame the interaction claim more conservatively.

## Reporting Tables

### Table 1: UMI

One row per pair with text mismatch, vision mismatch, and combined UMI.

### Table 2: Main Results

Rows are methods. Columns are benchmarks and average score. Separate subtables by teacher/student pair.

### Table 3: Joint Gain vs UMI

One row per pair:

- SFT average
- EMKD average
- SRE average
- Joint average
- Joint minus best single-side
- UMI

### Figure 1: UMI Predicts Joint Gain

Scatter plot:

- x-axis: UMI
- y-axis: Joint minus best single-side

## Compute Discipline

- Keep data, max length, LoRA rank, batch size, and total optimizer steps matched across methods inside a pair.
- Log wall-clock time and token/sample count for every run.
- Do not compare a 300K Joint checkpoint against 50K EMKD/SRE baselines.
- Keep the SFT baseline for each student on the same training mixture used by distillation runs.
