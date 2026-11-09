# P-ALIGN rebuttal settings

These are the executable rebuttal defaults, not a claim that every setting was
published in the paper. See [REBUTTAL.md](../REBUTTAL.md).

| Item | Value |
|---|---|
| Student | Qwen2.5-7B-Instruct |
| Training data | unchanged 966-row processed alignment JSON |
| Tuning | response-only LoRA |
| Epochs / LR | 3 / 5e-5 |
| Seed / data seed | 42 / 42 |
| Effective batch | 33 = 11 per GPU x 3 GPUs x accumulation 1 |
| Sequence length | 32,768 |
| LoRA | rank 16, alpha 16, dropout 0.05, seven projections |
| Optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| Schedule | cosine, warmup ratio 0.1 |
| Precision / checkpointing | bf16 / enabled |
| Eval | vLLM, n=3 (overridable), temperature 0.6, top-p 0.9, repetition penalty 1.05 |
| Eval budget | 4,096 new tokens; context 32,768 |
| Grader | P-ALIGN-style math-verify |

All default wrappers use `palign/src/run_eval.py`. Optional Qwen3 presets are
retained but are not the default launcher target.
