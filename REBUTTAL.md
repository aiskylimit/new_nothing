# Rebuttal defaults

The default backbone for all four methods is `Qwen/Qwen2.5-7B-Instruct`.
Training uses LoRA, 3 epochs, seed 42, 3 GPUs, microbatch 11 per GPU, and
gradient accumulation 1: effective batch **33**. Dataset selection and the
method-specific masks/prefixes are unchanged. Optional Qwen3 presets remain
available in `sft/configs/` and `palign/configs/`.

Train environments must already be provisioned with their respective dependencies.
SSFT uses its Unsloth environment; vLLM evaluation can use a separate interpreter
through `EVAL_PYTHON`. Do not install the two SSFT requirement sets into one environment.

Default launchers:

```bash
(cd palign && bash project_commands.sh)
(cd sft && bash project_commands.sh)
(cd sgl && bash project_commands.sh)
(cd SegmentSelectiveSFT && EVAL_PYTHON=/path/to/vllm-env/bin/python bash commands.sh)
```

Run each command from the repository root in its own shell, or return to the root
between commands. The SFT default target is `qwen25_longcot`; it does not launch
Label training by default. SSFT's `commands.sh` now uses LoRA on the same backbone
and launches training with torchrun. Its separate `train.sh` also defaults to
three processes. Override `NPROC_PER_NODE` together with the microbatch if changing
the number of GPUs. A smaller microbatch with accumulation can be used if the
longest sequences exceed GPU memory; GPU memory fit has not been measured here.

## Shared evaluation

All default launchers call `palign/src/run_eval.py`:

- vLLM, temperature 0.6, top-p 0.9, repetition penalty 1.05, seed 42.
- 3 generations per question, overridable with `EVAL_N` or `--n`.
- 4,096 new tokens; context 32,768 to retain space for the prompt.
- Exact P-ALIGN user prompt, including the original punctuation and no extra
  system message: `Please reason step by step, and put your final answer within \boxed{}.`
  followed immediately by the question; native chat template, thinking disabled.
- P-ALIGN-style `math-verify`: parse prediction, parse `$<gold>$`, verify equality.
- pass@1 is mean sample accuracy. pass@n is any-correct. With n >= 3 the report
  also includes the unbiased pass@3 estimator (identical to any-correct for n=3).
- Raw generations, scored rows, input hashes, decoding configuration, and summaries
  are saved. Existing outputs require `--overwrite`; `--score-only` regrades them.
- Merged checkpoints and LoRA adapter directories are both supported. vLLM loads
  an adapter against the base named in `adapter_config.json`.

The default input files for all methods are the four files in `palign/data/raw/`.
Use `EVAL_DATA_DIR` in wrappers or the common CLI for another math/OOD suite.
The evaluator expects question/answer fields, and does not download or select OOD data.

```bash
EVAL_N=5 bash palign/project_commands.sh eval

python palign/src/run_eval.py \
  --model /path/to/checkpoint-or-adapter \
  --input-files /path/to/ood.jsonl \
  --result-dir /path/to/new-ood-results --n 3
```

vLLM defaults to tensor parallel size 1, independently of the three-process
training launch. It can be changed with `--tensor-parallel-size` when supported by
the model. Use the same evaluation environment for all methods.

## SSFT fixes

`train_mask.run()` initializes `TOKEN_STATS` and `SKIPPED` before reading them;
previously Python treated them as uninitialized local variables and raised
`UnboundLocalError` before model loading.

One-pass outputs now use `<OUTPUT_DIR>/<RUN_ID>/checkpoint-final`, rather than
sharing `<OUTPUT_DIR>/checkpoint-final`. Reusing a nonempty run directory is
rejected. A different run ID therefore preserves earlier weights. Evaluation can
be rerun independently with the common CLI if training succeeded but eval failed.

The trainer explicitly preserves the precomputed selective labels in its padding
collator. Final adapters/tokenizers and `training_stats.json` are saved by rank 0.

Training-time definitions and the implementation stacks still differ; these
configuration changes alone do not establish comparable end-to-end runtime or
GPU memory usage. No B200 training or vLLM inference was executed locally.
