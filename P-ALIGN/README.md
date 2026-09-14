# P-ALIGN

Long-chain reasoning distillation via adaptive prefix alignment. Student: **Qwen2.5-7B-Instruct**.

Paper: https://arxiv.org/pdf/2601.10064

## Run

```bash
bash project_commands.sh
```

Hyperparameters vs paper: `HYPERPARAMETERS.md`.  
Eval table: `output/eval_results.txt`.

Do not upload data or weights. Train set is local (`data/palign_sft_qwen2.5-7b.json`).
