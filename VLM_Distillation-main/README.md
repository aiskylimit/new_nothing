# BeD-VLM

BeD-VLM distills a frozen vision-language teacher into a smaller student using supervised
cross-entropy, Semantic Cluster-based Visual Alignment (SCVA), and Semantic Intervention
Consistency (SIC). SCVA matches text-to-vision attention over teacher-derived semantic
regions. SIC reuses the encoded image, zeros matched visual-token embeddings, and aligns
the resulting vocabulary-wide log-probability shifts in embedding space.

## Experimental setup

| Launcher pair | Teacher | Student |
| --- | --- | --- |
| `fastvlm` | Qwen2-VL-7B-Instruct | FastVLM-0.5B |
| `qwen2` | Qwen2.5-VL-7B-Instruct | Qwen2-VL-2B-Instruct |

Train on the LLaVA-1.5 665K instruction-tuning mix. All student runs use the
repository's PEFT LoRA implementation. The paper compares BeD-VLM with SFT,
SRA, MCW-KD, EM-KD, DWA-KD, and DSKD-v2 on both pairs.

From the repository root on a Linux CUDA machine, create the training environment
and prepare the LLaVA annotations and image folders.

```bash
uv sync
source .venv/bin/activate
bash download_datatrain.sh
```

The launcher expects `train_data/llava_v1_5_mix665k.json` and resolves relative
image paths against `train_data/`. Set `DATA_PATH` and `IMAGE_DIR` for an
existing copy. Set `STUDENT_MODEL` and `TEACHER_MODEL` to local Hugging Face
snapshots for offline training.

## Main results

```bash
bash script_train/run_bed_vlm.sh fastvlm bed_vlm
bash script_train/run_bed_vlm.sh qwen2 bed_vlm

for pair in fastvlm qwen2; do
  for method in sft sra mcw_kd em_kd dwa_kd dskd_v2; do
    bash script_train/run_bed_vlm.sh "$pair" "$method"
  done
done
```

The launcher uses student-only training for SFT. The `sra` recipe uses the
repository's SRE criterion implementation. The paper's teacher row is evaluated
directly from the teacher snapshot.

Checkpoints go to `outputs/bed_vlm/<pair>/<method>/seed-<seed>/`. The launcher
defaults to one epoch, BF16, LoRA rank 128, learning rate `1e-5`, one example per
GPU per microbatch, eight gradient accumulation steps, and seed 1337.
`NPROC_PER_NODE` defaults to the number of visible GPUs. Set `SEED`,
`PER_DEVICE_BS`, `GRAD_ACCUM`, or `OUTPUT_DIR` to override those settings.
BeD-VLM uses four SCVA layer pairs, `min_samples=12`, SCVA and SIC weights of
0.3 each, and a budget of up to 16 matched region interventions per image.
These are the launcher defaults. Use `--dry-run` to inspect the command, and
pass extra training arguments after `--`.

```bash
bash script_train/run_bed_vlm.sh fastvlm bed_vlm --dry-run
NPROC_PER_NODE=1 bash script_train/run_bed_vlm.sh fastvlm bed_vlm -- --max_steps 1
```

## Ablation studies

All ablations use the FastVLM pair and a shared 30% subset of the input annotation
file, sampled with a fixed `SUBSET_SEED` (default 1337). The three evaluation
benchmarks are AI2D, MMMU, and MMStar.

| Study | Launcher methods | Configuration |
| --- | --- | --- |
| Supervision depth | `low`, `middle`, `high` | 0–40%, 40–70%, and 70–100% of decoder depth. SCVA uses the selected layers and SIC uses decoder output probabilities. |
| Loss components | `scva_only`, `sic_only`, `low` | Cross-entropy with SCVA only, SIC only, or both. The joint reference uses low-layer SCVA. |
| DBSCAN sensitivity | `dbscan_4`, `dbscan_8`, `dbscan_12`, `dbscan_16` | Vary `min_samples` with low-layer SCVA and SIC. |

```bash
for method in low middle high scva_only sic_only dbscan_4 dbscan_8 dbscan_16; do
  bash script_train/run_bed_vlm.sh fastvlm "$method"
done
```

`dbscan_12` and `low` use the same settings. Reuse the `low` checkpoint for the
component study and the `min_samples=12` row.

## VLMEvalKit evaluation

The main suite matches the column order in the paper. All six scores contribute
equally to `Avg`.

| Paper column | VLMEvalKit dataset |
| --- | --- |
| GQA | `GQA_TestDev_Balanced` |
| RWQA | `RealWorldQA` |
| ScienceQA | `ScienceQA_TEST` |
| AI2D | `AI2D_TEST_NO_MASK` |
| MMMU | `MMMU_DEV_VAL` |
| MMStar | `MMStar` |

Create the isolated evaluation environment and extract the pinned VLMEvalKit
source once. Run the asset commands with network access or prepopulate
`LMUData` with the corresponding TSVs and pass `--offline`.

```bash
env -u PYTHONPATH uv sync --project envs/eval
source envs/eval/.venv/bin/activate
unzip -q VLMEvalKit.zip 'VLMEvalKit/*'
export LMUData="$PWD/eval_data/LMUData"
SUITE_CONFIG="$PWD/configs/eval/bed_vlm.json" bash scripts/eval/prepare_eval_assets.sh
SUITE_CONFIG="$PWD/configs/eval/bed_vlm_ablation.json" bash scripts/eval/prepare_eval_assets.sh
```

Evaluate a trained checkpoint or a full teacher snapshot. For every student
checkpoint, set `STUDENT_BASE` to the matching local base snapshot so the
launcher can merge the LoRA weights. Evaluation uses exact matching by default.

```bash
STUDENT_BASE=/path/to/FastVLM-0.5B
TEACHER_BASE=/path/to/Qwen2-VL-7B-Instruct

bash scripts/eval/run_eval.sh --suite bed_vlm \
  --checkpoint outputs/bed_vlm/fastvlm/bed_vlm/seed-1337 --base-model "$STUDENT_BASE"
bash scripts/eval/run_eval.sh --suite bed_vlm --checkpoint "$TEACHER_BASE"
bash scripts/eval/run_eval.sh --suite bed_vlm_ablation \
  --checkpoint outputs/bed_vlm/fastvlm/low/seed-1337 --base-model "$STUDENT_BASE"
```

Collect the benchmark scores and main-table averages from completed evaluations.

```bash
for suite in bed_vlm bed_vlm_ablation; do
  python scripts/eval/collect_eval_summaries.py --root "outputs/eval/$suite" \
    --paper-table "outputs/eval/$suite/paper_table.csv" --fail-on-incomplete
done
```

## Qualitative analysis

Use the training environment. Prepare a JSONL file with one GQA example per
line, with `image`, `question`, and `answer` fields. Image paths are relative
to the JSONL file. The script generates predictions and low-layer attention
overlays for the teacher, base student, and distilled student. It also writes
the base student's mean adjacent-layer cosine similarity and across-example
standard deviation at each normalized depth using ground-truth answers.

```bash
python -m scripts.analyze_bed_vlm_attention --examples /path/to/gqa_examples.jsonl \
  --student /path/to/FastVLM-0.5B --teacher /path/to/Qwen2-VL-7B-Instruct \
  --checkpoint outputs/bed_vlm/fastvlm/bed_vlm/seed-1337
```

Outputs go to `outputs/bed_vlm_attention/`.

## Repository map

| Path | Purpose |
| --- | --- |
| `script_train/run_bed_vlm.sh` | Training launcher for the paper experiments. |
| `train.py`, `src/distiller.py` | Model setup, training, and checkpoint handling. |
| `src/criterions/scva.py`, `src/criterions/scva_sic.py` | Shared region construction, SCVA, and SIC. |
| `configs/eval/bed_vlm*.json` | Main and ablation benchmark suites. |
| `scripts/eval/` | Evaluation, adapter merging, and score reporting. |
| `script_train/` | Additional model-pair and legacy method recipes. |
| `tests/` | Unit tests for training and evaluation plumbing. |

```bash
python -m unittest tests.test_kd_plumbing tests.test_eval_pipeline
```
