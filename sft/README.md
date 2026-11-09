# SFT Long-CoT baseline

The default run is response-only LoRA SFT on Qwen2.5-7B-Instruct using all
1,000 `deepseek_thinking_trajectory` targets from s1K-1.1. Dataset preparation
and targets are unchanged.

Current defaults and evaluation protocol: [REBUTTAL.md](../REBUTTAL.md).
Training uses 3 GPUs, batch 11 per GPU, accumulation 1, 3 epochs, seed 42.

```bash
bash project_commands.sh
bash project_commands.sh train
EVAL_N=5 bash project_commands.sh eval
```

Provision `download.txt` and the existing training environment beforehand.
Use `EVAL_PYTHON` to select the common vLLM/math-verify environment. Evaluation
uses the shared `../palign/src/run_eval.py` and `../palign/data/raw/` by default;
`EVAL_DATA_DIR` can select another normalized JSONL suite.

Optional explicit selectors remain available: `qwen25_longcot`, `qwen3_longcot`,
and the historical Label targets. `all` explicitly selects the full legacy
matrix; it is not the default. Outputs are isolated under `output/<target>/`.
Training configurations still start fresh with `overwrite_output_dir: true`.
