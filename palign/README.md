# P-ALIGN

Adaptive prefix-alignment distillation. Default student: **Qwen2.5-7B-Instruct**.

```bash
bash project_commands.sh
EVAL_N=5 bash project_commands.sh eval
```

See [REBUTTAL.md](../REBUTTAL.md) for shared defaults: 3 epochs, seed 42,
effective batch 33 on 3 GPUs (11 per GPU, accumulation 1). The 966-row training
file is unchanged. Optional Qwen3 configs remain available for explicit use.

`src/run_eval.py` is the shared vLLM/math-verify evaluator for all four folders.
It preserves the original P-ALIGN prompt and raw-generation format. The context
limit is now independent of the maximum new-token count. Results are written
under `output/result/`, including raw outputs, scores and configuration.

Paper: https://arxiv.org/abs/2601.10064
