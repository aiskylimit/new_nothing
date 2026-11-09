# Spectral-guided Learning

Implementation of *Uncovering the Gradient Geometry of Long
CoT: A Spectral-guided Approach to Reasoning Distillation*.

This repository implements the primary method only:

1. Format each training sample with the model's native chat template.
2. Capture token loss gradients with respect to the final hidden state immediately before the LM head.
3. Run an exact, reduced, per-sample SVD in FP32.
4. Select the smallest rank whose squared singular values contain 95% of the gradient energy.
5. Compute equation (7), the mean truncated leverage score for each reasoning step.
6. Select the smallest stable, score-sorted step prefix containing 80% of the
   total step strength.
7. LoRA SFT on the complete sequence while masking the local losses of
   unselected reasoning steps.

## Training data format

Training uses all 1,000 rows from `simplescaling/s1K-1.1`. Each raw row is mapped as follows:

- `question` becomes the user message.
- `deepseek_thinking_trajectory` becomes the reasoning trajectory and is split
  into steps at blank lines (`\n\n`).
- `deepseek_attempt` becomes the final answer. It is prefixed with `Answer: `
  when that label is absent.

The formatter adds the Qwen system prompt and wraps the two assistant regions
with `<|im_start|>think` and `<|im_start|>answer` before applying the model's
native chat template. These tags are not expected in the raw dataset. Only
reasoning steps participate in spectral selection; the answer delimiter,
complete final answer, and native assistant end-of-turn token are always
supervised.

## Rebuttal run

See [the shared rebuttal settings](../REBUTTAL.md) for the current configuration.
The default is Qwen2.5-7B-Instruct, 3 epochs, seed 42, and effective batch 33
(11 per GPU, 3 GPUs, accumulation 1).

Install the training package in the training environment:

```bash
python -m pip install -e ".[dev]"
EVAL_PYTHON=/path/to/vllm-env/bin/python bash project_commands.sh
```

The launcher builds masks on three GPUs, trains with torchrun on three GPUs,
then uses the shared P-ALIGN evaluator. Set `EVAL_N=5` to change the number of
samples per question. Set `EVAL_DATA_DIR` for another normalized JSONL suite.
Adapter checkpoints remain under `checkpoints/`; evaluation supports them directly.

For evaluation without training:

```bash
python ../palign/src/run_eval.py \
  --model checkpoints/qwen2.5-7b-sgl \
  --result-dir outputs/qwen2.5-7b-sgl --n 3
```

The `sgl-evaluate` / `python -m sgl.evaluation` CLI delegates to the same evaluator;
`--model-name-or-path`, `--output-dir` and `--num-generations` remain accepted aliases.
Raw outputs, math-verify scores, input hashes, and configuration are persisted.
