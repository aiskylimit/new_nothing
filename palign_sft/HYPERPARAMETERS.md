# Baseline hyperparameters

All four runs use the same executable settings as the adjacent `P-ALIGN`
folder; only the backbone, chat template, training target, and output path
change.

| Item | Value |
|---|---|
| Method | response-only SFT with LoRA |
| Epochs | 3 |
| Learning rate | `5e-5` |
| Effective batch size | 8 |
| Per-device batch size | 1 |
| Scheduler / warmup | cosine / ratio 0.1 |
| Optimizer | AdamW Torch, betas `(0.9, 0.999)`, eps `1e-8`, wd `0` |
| Max sequence length | 32,768 |
| LoRA | rank 16, alpha 16, dropout 0.05 |
| LoRA targets | q/k/v/o and gate/up/down projections |
| Precision | bf16 |
| Gradient checkpointing | enabled |
| Qwen3 thinking template | disabled consistently for train and eval |
| Eval sampling | `n=3`, temperature 0.6, top-p 0.9, repetition penalty 1.05 |
| Eval generation | max 4,096 new tokens, max model length 32,768 |

The effective batch size above follows `P-ALIGN/project_commands.sh`, which is
the value actually executed. It deliberately does not copy the stale value 32
shown in the older `P-ALIGN/HYPERPARAMETERS.md` table.
