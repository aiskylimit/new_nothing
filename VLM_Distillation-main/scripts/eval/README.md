# Offline evaluation

The final suite evaluates every selected checkpoint on all eleven benchmarks in
one invocation:

## Isolated evaluation environment

The training project remains defined by the root `pyproject.toml`. Evaluation
has its own Python >=3.11 project and lock file under `envs/eval/`. Create it
without modifying the training `.venv`:

```bash
env -u PYTHONPATH uv sync --project envs/eval
```

This creates `envs/eval/.venv`. Point the evaluation launchers at it explicitly:

```bash
export PYTHON_BIN="$PWD/envs/eval/.venv/bin/python"
```

On a server that provisions environments from text manifests, use the sibling
file `../vlm-distillation-eval.txt`. The train manifest remains
`../vlm-distillation.txt`.

`VLMEvalKit/` is vendored source rather than a nested Git repository. Its
upstream identity is recorded in `VLMEvalKit/.vendored-commit`; evaluation
manifests continue to verify that commit without requiring `VLMEvalKit/.git`.

| Requested benchmark | VLMEvalKit dataset |
| --- | --- |
| GQA | `GQA_TestDev_Balanced` |
| MME-Perception | `MME` filtered to the 10 official perception categories |
| RealWorldQA | `RealWorldQA` |
| SQa-Image | `ScienceQA_TEST` |
| AI2D | `AI2D_TEST_NO_MASK` |
| MMMU | `MMMU_DEV_VAL` |
| MMStar | `MMStar` |
| ChartQA | `ChartQA_TEST` |
| DocVQA | `DocVQA_VAL` |
| TextVQA | `TextVQA_VAL` |
| OCRBench | `OCRBench` |

## 1. Download assets before submitting the evaluation job

Submit both [`download.txt`](../../download.txt) and
[`download_evaldata.txt`](../../download_evaldata.txt) to the server's asset
downloader. The latter puts evaluation TSV files under:

```text
/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
```

Together, the two manifests download all three student bases and all three
teachers under:

```text
/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/
```

No model is repeated between the manifests: `download.txt` owns
Qwen2.5-VL-3B/Qwen3-VL-8B, while `download_evaldata.txt` owns FastVLM-0.5B,
Qwen2-VL-2B, Qwen2.5-VL-7B, and Qwen3-VL-4B. The model directories preserve
their Hugging Face organization/repository layout. Student snapshots are used
when merging LoRA; evaluation does not fetch a base model from the Hub.

## 2. Run all checkpoints on the complete suite

From the repository root on the training server:

```bash
bash scripts/eval/run_requested_benchmarks.sh
```

The launcher checks all 11 TSV files and all three base-model snapshots first,
sets Hugging Face/Transformers offline mode, validates checksums, writes the
asset manifest, discovers the latest valid checkpoint from every training run,
then runs all 11 benchmarks for each checkpoint. It uses deterministic decoding
with `max_new_tokens=128`; FastVLM automatically uses eager attention and only
one generated token for multiple-choice datasets.

To restrict which training runs are evaluated while still running all 11
benchmarks for each selected checkpoint:

```bash
bash scripts/eval/run_requested_benchmarks.sh --pattern '*emkd*'
```

To evaluate exactly one trained LoRA checkpoint, supply its path and its local
base-model path. The embedded base path in `adapter_config.json` is replaced
explicitly, so it may refer to a different machine:

```bash
bash scripts/eval/run_one_trained_checkpoint.sh \
  /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/METHOD/checkpoint-STEP \
  /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-3B-Instruct
```

The same launcher works for FastVLM and Qwen2-VL by changing only those two
paths. Alternatively, edit `DEFAULT_TRAINED_CHECKPOINT` and
`DEFAULT_BASE_MODEL` near the top of the script, then invoke it without
arguments. Extra `run_eval.sh` options can be appended, such as
`--merge-device-map auto`, `--output-dir PATH`, or `--no-reuse`.

To evaluate one untouched baseline, pass only its full local model directory;
no LoRA merge is performed:

```bash
bash scripts/eval/run_one_baseline_model.sh \
  /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-3B-Instruct
```

The path can instead be edited in `DEFAULT_BASE_MODEL` near the top of
`run_one_baseline_model.sh`. The same script accepts the Qwen2-VL and FastVLM
snapshot directories.

To inspect checkpoint discovery and commands without loading models:

```bash
bash scripts/eval/run_requested_benchmarks.sh --dry-run
```

The core scripts remain under `scripts/eval/` because the single launcher calls
them for asset validation, checkpoint discovery, LoRA merging, model adapters,
inference, scoring, and summaries. Old per-model/per-dataset smoke launchers and
their temporary suite configs have been removed.

## Local AI2D + GQA smoke test

This machine has a dedicated two-dataset end-to-end smoke launcher. It defaults
to the local EMKD checkpoint, the cached Qwen2.5-VL base, and 10 samples from
each dataset:

```bash
bash scripts/eval/run_local_ai2d_gqa_smoke.sh
```

Choose another LoRA checkpoint/base pair or sample count without editing it:

```bash
SMOKE_SAMPLES=5 bash scripts/eval/run_local_ai2d_gqa_smoke.sh \
  /path/to/trained/checkpoint-STEP \
  /path/to/local/base-model \
  --merge-device-map auto
```

The original TSV files are never truncated or rewritten. Project-owned dataset
adapters expose only their deterministic first `SMOKE_SAMPLES` rows to
VLMEvalKit. Smoke results live below `outputs/eval/ai2d_gqa_smoke/`.

## Results

Per-checkpoint results are written to:

```text
outputs/eval/requested_benchmarks/<run-identity>/<checkpoint-id>/
```

Each directory contains `summary.json`, `run_context.json`, the generated
VLMEvalKit config, and native prediction/score artifacts. The combined batch
report is:

```text
outputs/eval/requested_benchmarks/batch_summary.json
```

The suite uses local scoring (`exact_matching`) by default and never enables a
paid/API judge implicitly. FastVLM is normalized consistently during training,
checkpoint saving, merge, and evaluation so `<|im_end|>` is EOS while
`<|endoftext|>` remains PAD.
