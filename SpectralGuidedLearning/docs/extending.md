# Extending the framework

## A stage is a module with `build_parser()` and `main(argv)`

```python
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
```

`sgl check` imports the module and parses the config's arguments with that parser, so keep imports
of heavy optional dependencies (vLLM, Unsloth) inside `main`.

## Data contracts between stages

| File | Written by | Fields |
|---|---|---|
| `train-segmented.jsonl` | `sgl.data.prepare` | `id, prompt, response, input_ids, response_token_span, steps[{token_start, token_end}], n_tokens` |
| signal parquet | `sgl.signals.capture`, `sgl.signals.gain_parquet` | `id, step_strengths, step_entropies, ...` (one value per step) |
| training jsonl | `selection`, `allocation`, `transforms`, `ssft.build` | `id, input_ids, loss_mask[, loss_weights][, token_source][, step fields]` |
| trace rows | `sgl.data.s1k` | `question, solution, answer[, segments]` |

Any training jsonl trains with `sgl.training.train`: `loss_mask` selects tokens, `loss_weights`
scales them (normalised over the optimizer step, Eq. 9), `token_source` picks a per-token objective.

## Add a per-step signal

Write a module that reads `train-segmented.jsonl` and writes one value per step into a parquet with
`id`, `step_strengths` and `step_entropies` (put your signal in `step_entropies` to feed the IWC
weights, as `sgl.signals.gain_parquet` does for answer gain). Then a method config:

```yaml
# configs/sgl/methods/iwc-mysignal.yaml
defaults: [_iwc.yaml]
signal_parquet: ${data_dir}/signals/my-signal.parquet
train_variant: iwc-stable-mysignal
pipeline: [prepare, my_signal, weights, train, eval]
stages:
  my_signal:
    module: sgl.signals.my_signal
    creates: ${signal_parquet}
    args: {data-path: "${data_dir}/train-segmented.jsonl", output: "${signal_parquet}"}
```

and an arm: `configs/sgl/r1-qwen-1.5b-palign/iwc-mysignal.yaml` with
`defaults: [../tracks/r1-qwen-1.5b-palign.yaml, ../methods/iwc-mysignal.yaml]` and `name:`.

## Add a segmenter

Add a lossless function (`"".join(pieces) == text`) to `sgl.segment.rules` and register it in
`sgl.segment.SEGMENTERS`; `sgl.data.prepare --segmenter <name>` then uses it for the step spans.

## Add a model or a dataset

`configs/common/models/<key>.yaml` (`model.key`, `model.name`) and a track under
`configs/<family>/tracks/` that combines it with a data block, a training recipe and the eval
protocol.

## Config reference

* `defaults: [...]` merges files first (paths relative to the file); mappings merge deeply, other
  values are replaced. Building-block files and directories starting with `_`, `methods/`, `tracks/`
  are not listed as experiments.
* `${a.b}` references another key (keeps the type when it is the whole string), `${env:VAR,default}`
  reads the environment. Only `true` / `false` are booleans (`off` stays a string).
* `family:` must match the command (`sgl`, `palign`, `ssft`).
* Stage keys: `module` or `script`, `args`, `positional`, `launcher` (python | torchrun), `creates`,
  `gpus`, `effective_batch`, `env`, `cwd`, `raw_keys` (see `src/sgl/cli.py`).
