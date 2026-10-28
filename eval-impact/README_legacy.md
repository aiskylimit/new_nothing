# IMPACT: Importance-Aware Prefix Alignment for Cross-Tokenizer Distillation

OPD extension, saved training/calibration data, BI stability analysis and MATH-500/GSM8K evaluation: see [README_OPD.md](README_OPD.md).

Local B200 environment/download/run templates, offline eval and final results: [README_B200.md](README_B200.md).

## 🔍 Overview

IMPACT is a framework for cross-tokenizer knowledge distillation via importance-aware prefix alignment. This repo provides code for reproducing our experiments with IMPACT and several KD baselines.

## 🔧 Installation

```bash
cd IMPACT_Final
bash install_impact.sh
conda activate impact
```

Optional extras:

```bash
python -m spacy download en_core_web_sm   
pip install pytest                       
```

## 📦 Data and Weights

**Training data** (Dolly by default): put `train.jsonl` and `dev.jsonl` under `data/dolly/`.  
**Eval data** already lives under `data/{dialogsum,self-inst,dolly,vicuna,sinst}/`.

**Base models** under `model_hub/<name>/` (Hugging Face layout). Default student / teacher pairs:

| Scripts folder | Student | Teacher |
|----------------|---------|---------|
| `scripts/gpt2_120m/` | GPT2-120M | Qwen1.5-1.8B |
| `scripts/gpt2_340m/` | GPT2-340M | Qwen1.5-1.8B |
| `scripts/gptxl/` | GPT2-1.5B | Qwen2.5-7B-Instruct |
| `scripts/tinyllama/` | TinyLlama | Mistral-7B |
| `scripts/opt/` | OPT-2.7B | Qwen2.5-7B-Instruct |

For `gptxl/`, `tinyllama/`, and `opt/`, also place the teacher LoRA adapter at `lora_path/<TEACHER_MODEL_NAME>/`.  
(`gpt2_120m` / `gpt2_340m` do not use `--teacher-peft-path`.)

## 🚀 Training

Edit `GPUS=(...)` at the top of a script, or override with:

```bash
export IMPACT_CUDA_DEVICES=0
```

Run from the repo root. Example (DSKDv2 + IMPACT, GPT2-120M ← Qwen1.5-1.8B):

```bash
bash scripts/gpt2_120m/DSKDv2_IMPACT.sh
```

### Base method + IMPACT

Available under `scripts/gpt2_120m/` (same names in `gpt2_340m/`, `gptxl/`, `tinyllama/`):

| Method | Script |
|--------|--------|
| SRA + IMPACT | `SRA_IMPACT.sh` |
| DSKD + IMPACT | `DSKD_IMPACT.sh` |
| ALM + IMPACT | `ALM_IMPACT.sh` |
| ResidualKD + IMPACT | `ResidualKD_IMPACT_full.sh` |
| DSKDv2 (ETA) | `DSKDv2.sh` |
| DSKDv2 + IMPACT | `DSKDv2_IMPACT.sh` |

### Extra DSKDv2 + IMPACT variants

| Variant | Script |
|---------|--------|
| Intermediate layers by PPL | `scripts/gpt2_340m/DSKDv2_IMPACT_PPL.sh` (also `*_First` / `*_Last` / `*_Random`) |
| ALP-KD 1-all | `scripts/opt/DSKDv2_IMPACT_ALPKD.sh` |
| RAIL-KD | `scripts/opt/DSKDv2_IMPACT_RAILKD.sh` |

Baselines without IMPACT use the matching name without `_IMPACT` (e.g. `SRA.sh`, `DSKD.sh`, `ALM.sh`).

Checkpoints are written under `outputs/<CKPT_NAME>/<TASK>/.../`.

## 📊 Evaluation

### Evaluate full fine-tuning checkpoints

```bash
bash scripts/eval/run_eval.sh ${CKPT_PATH} ${EVAL_BATCH_SIZE}
```

`CKPT_PATH` should be the **absolute path** of the model files. Default eval batch size is `8`.

### Evaluate LoRA fine-tuning checkpoints

```bash
bash scripts/eval/run_eval_lora.sh ${LORA_ADAPTER_PATH} ${EVAL_BATCH_SIZE}
```

`LORA_ADAPTER_PATH` should be the **absolute path** of the LoRA adapter. Default eval batch size is `4`.

GPU override: `export IMPACT_EVAL_GPU=0`.

Eval runs over Dolly / DialogSum / Self-Inst / Vicuna / Super-NI with 5 seeds.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
