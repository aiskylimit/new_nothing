# Rebuttal defaults

The finalized **ICLR 2027 main experiment plan supersedes the legacy evaluation
defaults below**. See [rebuttal/README.md](rebuttal/README.md) and
[rebuttal/PLAN.md](rebuttal/PLAN.md). The unified entry point is
`bash rebuttal/project_commands.sh`: 15 runs, GPUs 3/4/5, 64 math responses total
over five evaluation seeds and deterministic OOD. The older method launchers
remain available, with their earlier n=3 evaluation; do not use them for the
finalized main evaluation matrix.

## Server provisioning files

The server reads model/data commands from each folder's `download.txt`, and
Python dependencies from the matching file at the repository root:

| Folder | Model/data manifest | Dependency manifest | Default environment |
|---|---|---|---|
| `palign` | `palign/download.txt` | `palign.txt` | `/mnt/local/uvenvs/palign` |
| `sft` | `sft/download.txt` | `sft.txt` | `/mnt/local/uvenvs/sft` |
| `sgl` | `sgl/download.txt` | `sgl.txt` | `/mnt/local/uvenvs/sgl` |
| `SegmentSelectiveSFT` | `SegmentSelectiveSFT/download.txt` | `SegmentSelectiveSFT.txt` | `/mnt/local/uvenvs/SegmentSelectiveSFT` |
| `cored` | `cored/download.txt` | `cored.txt` | `/mnt/local/uvenvs/cored` or active Python |

All manifests use Python 3.11 and `# manager: uv`. Each download manifest starts
with `#d`; the downloader replaces `@PROJECT@` with the deployed project name.
Models and training datasets go under `/mnt/local/@PROJECT@/<folder>/`.
Every manifest also provisions the same four ID benchmark sources and two
planned OOD test files under `palign/datasets/`, using one shared benchmark store.

The default scripts resolve assets from their own deployed folder. Existing
external asset locations can be selected with `PALIGN_ASSET_ROOT`,
`PALIGN_SFT_ASSET_ROOT`, `SGL_ASSET_ROOT`, `SSFT_ASSET_ROOT`, or `CORED_ASSET_ROOT`.
Direct P-ALIGN/SFT YAML configurations use relative `models/...` paths.
P-ALIGN copies its downloaded `datasets/P-ALIGN/P-ALIGN.json` into the expected
training path only when the existing processed training JSON is absent.
SSFT reads the same preformatted `baesad/s1K-1.1-deepseek-cot/train.jsonl`
source as before, now under `SegmentSelectiveSFT/datasets/s1k/`.
SGL reads its downloaded s1K Parquet files directly, without a Hub request.

`palign.txt` and `sft.txt` share the LLaMA-Factory/vLLM environment recipe.
SGL and SSFT retain separate training environments and use
`EVAL_PYTHON=/mnt/local/uvenvs/palign/bin/python` for the shared vLLM evaluator.
Provision **palign.txt as well** when running either SGL or SSFT.
Override `PALIGN_VENV`, `SFT_VENV`, `SGL_VENV`, or `VENV` if the server creates
environments elsewhere. SSFT's launcher uses the provisioned environment and
does not rebuild/reinstall it during a timed run.

Use the five matching manifest pairs above for new jobs. Redundant aliases and
unused older P-ALIGN/SFT/CORED requirement files have been removed. Dependency resolution
was checked for Linux x86-64 (manylinux 2.35), Python 3.11; this does not replace
a CUDA import/kernel check on the B200 server.

### Planned OOD test data

All five `download.txt` files download only the requested OOD test files with
version-pinned `--url` entries.

| Benchmark | Evaluation split | Verified rows | Downloaded path relative to `palign/` |
|---|---|---|---|
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | `ARC-Challenge/test` | 1,172 | `datasets/ARC-Challenge/test-00000-of-00001.parquet` |
| [LogiQA 2.0](https://github.com/csitfun/LogiQA2.0) | English MRC `test.txt` | 1,572 | `datasets/LogiQA2.0/test.txt` |

The ARC snapshot is `210d026faf9955653af8916fad021475a3f00453`; the LogiQA
file revision is `31381f5634d201c4ed6f981b87de3c7d9f496002`.
Both counts and answer/choice schemas were checked directly from those files.
SHA-256 values:

- ARC test: `62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9`
- LogiQA test: `71940b37ae0184b677c253a148d57ad4e75d6113447b1563c2ca82483e4e4f8d`

These entries provision raw data; they do not add the datasets to the default
math evaluation loop. MCQ normalization/scoring must preserve ARC's own
`choices.label` mapping (labels can be letters or numbers, with 3–5 options),
and LogiQA's passage, question, four options, and zero-based answer index.
LogiQA uses the English MRC file, not its Chinese or NLI variants.

## Experiment defaults (four baselines)

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

CORED's `project_commands.sh` launches a separate three-GPU data-parallel
pipeline, also with global batch 33, three epochs per phase and seed 42.
Microbatches are limited by token budget and can accumulate within each global
batch. Its original modules/scripts retain the single-process execution path.
See `cored/README.md` for its full pipeline, timings and validation limits.

## Shared evaluation (four baselines)

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
