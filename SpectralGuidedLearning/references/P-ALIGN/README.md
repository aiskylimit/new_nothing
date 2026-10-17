# P-ALIGN

Long-chain reasoning distillation via adaptive prefix alignment. Student: **Qwen3-8B**.

Paper: https://arxiv.org/pdf/2601.10064

## Run

```bash
bash project_commands.sh
```

DeepSeek-R1-Distill-Qwen-1.5B, full finetuning, thinking off at train and eval:

```bash
bash project_commands_r1_1.5b.sh
```

Config `configs/r1_distill_qwen_1.5b_palign_full_sft.yaml` (template `deepseekr1`, `enable_thinking: false`).
Eval prompt ends with `<｜Assistant｜><think>\n\n</think>\n\n` (`src/test.py --force_empty_think`), identical to training.
Results: `output/eval_results_r1_1.5b.txt`. If one GPU OOMs at 32k context: `NPROC_PER_NODE=N CUDA_VISIBLE_DEVICES=... DEEPSPEED=configs/ds_z2.json`.

Hyperparameters vs paper: `HYPERPARAMETERS.md`.  
Eval table: `output/eval_results.txt`.

Do not upload data or weights. Train set is local (`data/palign_sft_qwen2.5-7b.json.gz`, unpacked by `project_commands.sh`).
