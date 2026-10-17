# Spectral-Guided Learning (SGL) framework

Step-level **selection** and **allocation** for distilling long chain-of-thought (CoT) into a small
student, with three method families behind one pipeline and one config system:

| Command  | Method family | What it decides |
|----------|---------------|-----------------|
| `sgl`    | Spectral-guided learning, IWC / IWC-Stable allocation, answer-gain allocation, provenance-aware SFT, L_trans | which steps to learn (spectral gate) and **how much** each step weighs |
| `palign` | [P-ALIGN](https://arxiv.org/abs/2601.10064): adaptive prefix alignment | which **prefix** of the teacher trace to keep; the student writes the rest |
| `ssft`   | Segment-Selective SFT (Wang, Liu & Ren, ICLR 2026) | which **segments** to learn, from Integrated-Gradients attribution to the answer |

Every method is one path through the same stages:

```
data -> (trace transform) -> segment -> per-step signal -> select (loss_mask) -> allocate (loss_weights)
     -> objective -> train -> eval
```

## Install

```bash
uv sync                      # core stack: torch, transformers, vLLM, DeepSpeed, PEFT (pyproject.toml)
# or: pip install -e .
```

The `sgl`, `palign` and `ssft` commands come with the package. The upstream reference trainers
(LLaMA-Factory for P-ALIGN, Unsloth for Segment-Selective SFT) need their own environments; point a
stage at one with `stages.<stage>.python=/path/to/env/bin/python` (or `run.python=` for every stage).

## Quickstart

```bash
sgl list                                              # experiments under configs/sgl/
sgl show r1-qwen-1.5b-palign/iwc-gain-l05             # fully resolved config
sgl run  r1-qwen-1.5b-palign/iwc-gain-l05 --dry-run   # the exact commands, nothing runs
sgl run  r1-qwen-1.5b-palign/iwc-gain-l05 run.gpus=[0,1]

palign run r1-qwen-1.5b                               # P-ALIGN, R1-Distill-Qwen-1.5B, full FT
ssft   run r1-qwen-1.5b                               # Segment-Selective SFT, same student
sgl    run r1-qwen-1.5b-palign/sft-nll --stages compare   # refresh results/comparison-table.md
```

* `key.path=value` overrides any config value (`train_seed=43`, `stages.train.args.learning-rate=1e-5`).
* `--stages a,b` runs just those stages (including optional ones such as `train_reference`),
  `--from b` resumes the pipeline at `b`, `--force` reruns stages whose output already exists.
* Each stage logs to `logs/<run>-<stage>.log`; a stage is skipped when its `creates` path exists.
* `sgl check` validates every config of the family against the argument parsers of the modules it
  calls, so a typo fails before anything runs.

## Layout

```
src/sgl/
  cli.py, config.py   the stage runner and the YAML config system
  data/               prepare (tokenize + segment), s1k (trace rows for P-ALIGN / SSFT)
  segment/            sentence | paragraph | cue segmentation, char -> token spans
  signals/            spectral capture (strength + entropy), answer gain
  selection/          spectral gate -> loss_mask
  allocation/         IWC / IWC-Stable step weights -> loss_weights
  transforms/         provenance (token_source), step transitions (L_trans)
  palign/             truncate -> align -> build                        (palign family)
  ssft/               attribution -> select -> build                    (ssft family)
  training/           masked / weighted objectives, trainers, Unsloth backend
  eval/               benchmarks, vLLM generation, graders, result tables
  diagnostics/        pre-training checks of the weighting signals
configs/
  common/             base, models, datasets, training recipes, evaluation protocol
  sgl/  palign/  ssft/   experiments of each family (+ methods/ and tracks/ building blocks)
references/           the original P-ALIGN and SegmentSelectiveSFT code, unmodified
tests/                unit tests, equivalence tests against references/, regression + end-to-end
results/              eval summaries (summary.json per run), comparison table
```

## Docs

* [docs/methods.md](docs/methods.md): what each family computes and how faithfully it is ported
* [docs/reproducing.md](docs/reproducing.md): every published row and the command that produces it
* [docs/extending.md](docs/extending.md): adding a signal, segmenter, method or experiment
* [docs/FRAMEWORK_PLAN.md](docs/FRAMEWORK_PLAN.md): the refactor plan and its status
* [CLAUDE.md](CLAUDE.md): commands, layout and conventions for contributors (and Claude Code)

## Tests

```bash
pytest                      # ~2.5 min on CPU, no GPU or network needed
pytest -m "not regression"  # unit + equivalence tests only (~1 min)
```

`tests/regression/` reruns every CPU stage on fixed assets and compares the outputs with hashes
recorded from the pre-refactor code (`paper-v1` = commit `7990089`), and trains each family's
configs end to end through the CLI with a tiny model. The equivalence tests import the untouched
upstream code from `references/` and check the ports against it.
