# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fork of the ICLR 2026 paper "Segment-Level Attribution for Selective Learning of Long Reasoning Traces"
(see `README.md`, `2602.00425v1.pdf`). Upstream provides three stages — `Attribution/`, `SelectiveSFT/`,
`Eval/` — each driven by a shell script with hard-coded config. This fork adds parameterized wrappers at
the repo root (`run_pipeline.sh`, `train.sh`, `eval.sh`); prefer editing/extending those over the
upstream scripts (`Eval/run_eval.sh`, `SelectiveSFT/run_train.sh`, `Attribution/cal_attribution.sh`),
which are kept close to as-published and still carry the paper's original LIMO/DeepSeek config.

**This fork's current configuration diverges from the paper**: LoRA (not full finetuning) on
`Qwen/Qwen2.5-7B-Instruct`, trained on `simplescaling/s1K-1.1` (not LIMO), with segments split on
`\n\n` (not the paper's backtracking-cue regex). See "Current training configuration" below.

Comments and commit messages in this repo are written in **Vietnamese without diacritics**. Match that
style when editing shell scripts, adding comments, or writing commits.

## Two incompatible Python environments

This is the single most important constraint. Never install both requirement sets into one env:

| Env | Requirements | Pins | Used by |
|---|---|---|---|
| `ssft_train` | `requirements-sft.txt` (superset of `SelectiveSFT/requirements.txt`) | torch 2.9.0, transformers 4.57.1, unsloth, peft, `torchao<0.18` | `train.sh`, `SelectiveSFT/merge_lora.py` |
| `ssft_eval` | `requirements.txt` (root) + `Eval/latex2sympy` editable | torch 2.7.1, transformers 4.56.0, vllm 0.10.0 | `eval.sh`, attribution stages, CoT generation |

`peft` lives only in the train env, so **LoRA merging must run there**, not in the eval env.

Environment setup lives in **`setup.sh`**, not in the pipeline — `run_pipeline.sh` only runs stages.
`bash setup.sh check --for train|eval|all` verifies an environment without installing anything (useful
on air-gapped machines); `bash setup.sh eval` / `bash setup.sh train` install the respective set, with
`--conda <name>` / `--venv <dir>` to build an isolated one. It refuses quietly-broken combinations by
warning when it sees the other side's marker package (`unsloth` vs `vllm`) already installed.

**All three pipeline wrappers default to `USE_CONDA=0` — they use whatever Python is already active and build
nothing.** That suits managed cloud environments (Lightning Studio and similar) that ship a complete env
and no `conda` on PATH. Pass `--use-conda` (or `USE_CONDA=1`, or `--env <name>`, which implies it) to get
the original behavior: `run_pipeline.sh` creating/activating `selective_sft`, `train.sh` and `eval.sh`
each building their own env and pip-installing. Under `--use-conda`, `run_pipeline.sh`'s `train` stage
lets `train.sh` manage `ssft_train` itself; without it, the stage passes `--skip-setup` so nothing is
rebuilt. When the two environments above are NOT separated, the torch/vLLM conflict is yours to avoid.

`requirements-sft-lock.txt` (the fully-pinned transitive lock for the train env) was **removed from the
repo**; `setup.sh train --lock` still points at it and will fail unless you regenerate one with
`pip freeze`. `torchao<0.18` is load-bearing: 0.18 breaks `import unsloth`.

`train.sh` / `eval.sh` build their env on first run and drop a stamp file (`.setup_done_<env>_v<N>`) to
skip reinstalling. Bump the `_vN` suffix in the script when the requirements change, or the stamp will
mask the new deps. `--reinstall` forces, `--skip-setup` skips.

## Commands

```bash
# Environment (separate from the pipeline)
bash setup.sh check --for train             # verify without installing
bash setup.sh eval                          # requirements.txt + latex2sympy
bash setup.sh train --conda ssft_train      # requirements-sft.txt in its own env

# Data prep, once
python prepare_s1k.py                       # -> data/s1k/train.jsonl
bash run_pipeline.sh --stages prep,split    # -> data/s1k/solution_segments.jsonl

# Default pipeline: attribution + selective SFT (assumes solution_segments.jsonl exists)
bash run_pipeline.sh                        # = --stages ig,segments,train
bash run_pipeline.sh --stages ig --resume    # ig stage overwrites by default; --resume continues a partial run
bash run_pipeline.sh --offline               # air-gapped: sets HF_HUB_OFFLINE + HF_DATASETS_OFFLINE
bash run_pipeline.sh --segment-mode cue      # paper's backtracking-cue split instead of "\n\n"
DRY_RUN=1 bash run_pipeline.sh

# Training (creates env, trains, logs to logs/train_lora.log)
bash train.sh                       # selective SFT, LoRA — see config table below
bash train.sh --full-sft            # baseline: supervise the whole CoT
bash train.sh --full-finetune       # no LoRA (very heavy at seq 32768 on 7B)
bash train.sh --epochs 5 --lr 1e-5 --gpu 1
bash train.sh --lora-r 16 --lora-alpha 16 --target-modules "q_proj,v_proj"   # checkpoint dir gets _lora_r16; pass the same --lora-r to eval.sh
bash train.sh --grad-accum 16 --ckpt-suffix _bs16   # batch isn't in the dir name; suffix keeps a new run from rotating out the old checkpoints (pass the same --ckpt-suffix to eval.sh)
bash train.sh --no-grad-checkpoint  # faster, more VRAM
bash train.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --full-finetune   # --deepseek template auto-on
bash train.sh --dry-run             # print commands only

# LoRA checkpoints are adapters — merge before eval (train env, CPU is fine)
cd SelectiveSFT && python merge_lora.py --adapter checkpoints/<run>/checkpoint-<step>

# Eval (separate env; resumable, prints an accuracy table + writes summary.json)
bash eval.sh --model /abs/path/checkpoint-<step>-merged --tag sel_ep3
bash eval.sh --base                 # un-finetuned base model
bash eval.sh --quick                # math500, 100 questions, 1 sample/question
bash eval.sh --tasks "aime24 math500" --n-sampling 1
bash eval.sh --gpu 0,1,2,3 --data-parallel   # one task per GPU (faster than TP)
bash eval.sh --overwrite            # re-score from scratch

# The only unit tests in the repo (vendored latex2sympy parser)
cd Eval/latex2sympy && pytest tests/            # single test: pytest tests/trig_test.py
```

All three wrappers take `--dry-run` and `-h`.

## Current training configuration

`train.sh` defaults, all overridable by flag or env var:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Data | `data/s1k/solutions_selected.jsonl` (from `simplescaling/s1K-1.1`) |
| Tuning | LoRA r=64, alpha=64, dropout=0.05, bias=none |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| LR / epochs | 5e-5 / 3 |
| Effective batch | 32 (`per_device 1` x `grad_accum 32` x 1 GPU) |
| Optimizer | `adamw_torch`, betas (0.9, 0.999), eps 1e-8, weight_decay 0.0 |
| Scheduler | cosine + warmup, `warmup_ratio=0.1` (HF builds this as a `LambdaLR`) |
| `max_seq_length` | 32768 (= Qwen2.5-7B's `max_position_embeddings`; no RoPE scaling) |
| Gradient checkpointing | on, unsloth's variant (required at 32k on 7B) |
| Segmentation | `paragraph` — split on every `\n\n` |

Effective batch is `per_device x grad_accum x WORLD_SIZE` — launching under DDP with more than one process multiplies it past 32.

## Pipeline data flow

```
simplescaling/s1K-1.1 (HuggingFace)
  │  prepare_s1k.py — question / deepseek_thinking_trajectory / last \boxed{}
  ▼
data/s1k/train.jsonl                 (question, solution, answer)
  │  Attribution/segment_split.py — split on "\n\n" (--segment_mode cue for the paper's split)
  ▼
data/s1k/solution_segments.jsonl     (+ segments[])
  │  Attribution/grad_analyze.py — Integrated Gradients from each segment's tokens
  │                                to the \boxed{answer} tokens (ig_steps=20; paper used 50)
  ▼
Attribution/processed_data/s1k/IG.jsonl  +  IG_compact.jsonl
  │  Attribution/get_important_segments.py — per-segment score = sum|IG| / sqrt(len);
  │    keep top segments up to --cumulative_ratio (0.7) of the mass, drop those with
  │    |sum IG| / sum|IG| > --coherence_max (0.8)
  ▼
data/s1k/solutions_selected.jsonl    (+ selected_spans_ids[])
  │  SelectiveSFT/train_mask.py — labels = -100 except the selected segments
  ▼
SelectiveSFT/checkpoints/<model>_epoch<E>_lr<LR>_len<L>[_fullsft][_lora_r<R>][<--ckpt-suffix>]/checkpoint-<step>
  │  SelectiveSFT/merge_lora.py (LoRA only) -> checkpoint-<step>-merged
  │  Eval/math_eval.py via eval.sh
  ▼
Eval/outputs_<tag>/<task>/<tag>/*_metrics.json  +  Eval/outputs_<tag>/summary.json
```

`segment_utils.py` (repo root) owns the splitting rules and is imported by both
`Attribution/segment_split.py` and `SelectiveSFT/train_mask.py`, which each insert the repo root on
`sys.path`. It guarantees `"".join(split_segments(t)) == t` and no empty segments — `grad_analyze.py`
derives token spans from cumulative segment lengths, so a splitter that dropped characters would
silently misalign every downstream span, and a zero-length segment divides by zero in
`get_important_segments.py`. Keep both invariants if you add a split mode.

## Sharp edges

- **Masking is anchored to character offsets, not cumulative token counts.** `train_mask.py` tokenizes
  with `return_offsets_mapping=True` and assigns each token to the segment holding its first character.
  The earlier approach — locate `response_template` in the full sequence, then add
  `len(tokenizer("".join(segments[:k])))` — mixes two different tokenizations (prompt+response vs.
  response alone) and drifts a token or two at the junction, leaking text from *unselected* segments
  into the supervised span. This requires a fast tokenizer; the script exits early if it doesn't get one.
- **Qwen3 thinking mode is a train/eval contract.** `train_mask.py --think_prefix off` and
  `math_eval.py --disable_think` (`eval.sh --no-think`) both call
  `apply_chat_template(enable_thinking=False)`, so the Qwen3 template appends an empty
  `<think>\n\n</think>\n\n` block to the prompt (label `-100`) and the plain s1K trace follows it;
  `train_mask.py` probes the template and exits if the prefix does not match (Qwen2.5 ignores the
  kwarg, so `off` fails loudly there). The default `none` inserts nothing in either script. A checkpoint
  trained one way must be evaluated the same way — `train.sh` appends `_nothink` to the checkpoint dir
  and `eval.sh --no-think` looks for that suffix (and adds `_nothink` to the default tag) so the two
  cannot be mixed by accident.
- **DeepSeek-R1 models need the DeepSeek chat template** (`<｜User｜>...<｜Assistant｜><think>\n`), not
  ChatML. `train.sh` passes `train_mask.py --deepseek` automatically when the model basename contains
  `DeepSeek-R1` (`--deepseek` / `--no-deepseek` / `DEEPSEEK=0|1` override); only `--think-prefix none|off` is
  allowed with it. The R1 template ignores `enable_thinking`, so "thinking off" for R1 means closing its
  `<think>\n` into an empty `<think>\n\n</think>\n\n`: `train_mask.py --deepseek --think_prefix off` appends
  `\n</think>\n\n` (label `-100`) and `math_eval.py --disable_think` rewrites a prompt ending in `<think>` the
  same way (verified identical strings/tokens with the real 1.5B tokenizer). Both sides tokenize the rendered
  template with `add_special_tokens=False` (math_eval passes `prompt_token_ids` to vLLM) so R1's BOS is never
  doubled; Qwen has no BOS, so this is a no-op there. The `"qwen" in path` stop-token check still matches
  `DeepSeek-R1-Distill-Qwen-*` (151643 = R1 EOS).
- **Prompt wording is a train/eval contract too.** `train_mask.py --prompt_style default` (`<question>\nPlease
  reason step by step, ...\boxed{}.`) pairs with eval `--prompt-type deepseek-longcot` (default);
  `--prompt_style palign` (`Please reason step by step, ...\boxed{}.<question>`, as in P-ALIGN) pairs with
  `eval.sh --prompt-type palign`. `train.sh --prompt-style palign` appends `_palign` to the checkpoint dir, after
  `_nothink`. Attribution (`grad_analyze.py`) always uses the default wording.
- **A sample whose labels are all `-100` yields `nan` loss and poisons the run.** This happens when
  `max_seq_length` truncates away the response. `train_mask.py` drops such samples in `.map()` and
  reports the count; it does not crash.
- **The `ig` stage overwrites by default.** `grad_analyze.py` deletes its old `output_data_file` and
  opens it `'w'`, so a re-run never silently duplicates or interleaves records. Pass `--resume`
  (`run_pipeline.sh --resume` / `IG_RESUME=1`) to continue a partial run instead: it re-reads the
  output, checks each record's `question` against the input, continues from the first unprocessed
  sample and opens the file `'a'`; a truncated last line (from a `kill -9`) is dropped, and a mismatch
  stops the run rather than mixing two different runs.
- **Per-token IG is never needed downstream.** `get_important_segments.py` only uses
  `Σ|IG|`, `ΣIG` and the token count per segment (Eq 3), so `grad_analyze.py` also writes
  `IG_compact.jsonl` holding exactly those three numbers — ~25x smaller than `IG.jsonl`, with verified
  identical selections. `run_pipeline.sh` feeds the compact file to the `segments` stage when present.
  `get_important_segments.py` accepts either format, distinguished by JSON type (object vs array) rather
  than by length — a 3-token segment would otherwise be indistinguishable from a compact triple.
- **`make_bundle.sh` splits data files for a download-size-limited machine**, preserving content exactly —
  `cat <name>.part* > <name>` restores the original byte for byte. `.jsonl` files are split on line
  boundaries so each part is itself valid JSONL and can be inspected alone; other files split by byte.
  Writes SHA256SUMS and reassembly instructions; files already under the limit are copied whole.
- **Eval scripts must run with cwd = `Eval/`**: `parser.py`/`grader.py` do
  `from latex2sympy.latex2sympy2 import ...` (resolved via the local package dir), and `--data_dir`
  defaults to `../data`. All wrappers `cd` there.
- **`acc` in `*_metrics.json` scores only the first sample of each question** (`evaluate.py`:
  `mean_score[0]`), not the mean over `n_sampling`. `eval.sh`'s summary step imports `pass_at_k.py` and
  writes unbiased pass@1 and pass@n (n = `n_sampling`) per task + macro-average into
  `summary.json["pass_at_k"]`; `Eval/pass_at_k.py outputs_<tag> --k 1 3` recomputes the same numbers for
  arbitrary k into `pass_at_k.json`. All accuracies are rounded to 2 decimals.
- **`math_eval.py` skips a task whose `*_metrics.json` already exists** — that is what makes `eval.sh`
  resumable after a crash. The output filename encodes
  `num_test_sample/seed/temperature/n_sampling/max_tokens`, so a `--quick` run and a full run coexist
  without clobbering. Use `--overwrite` to force re-generation.
- `--quick` deliberately does **not** shrink `--max-tokens`: truncating a reasoning model's generation
  drops the boxed answer and deflates accuracy.
- **`run_pipeline.sh`'s `setup` stage builds only the eval-side env.** Its `train` stage shells out to
  `train.sh` without `--skip-setup` so that script activates `ssft_train` itself — do not "optimize" that
  into a direct `train_mask.py` call under the pipeline's env.
- **`EarlyStopAtEpochCallback` used to hard-stop at `state.epoch >= 9`** regardless of `--epochs`. It is
  now driven by `--stop_at_epoch` and disabled by default.
- Every generated artifact is gitignored (`logs/`, `SelectiveSFT/checkpoints/`,
  `Attribution/processed_data/`, `Eval/outputs*`). `data/s1k/*` is produced by `prepare_s1k.py` and the
  attribution stages.
- **`data/` ships no datasets anymore.** The eval test sets (`data/<task>/test.jsonl` for aime24, amc23,
  gpqa, math500, minerva, olympiad) and the paper's LIMO files (`data/limo/*.jsonl`) were removed from
  the repo; `Eval/data_loader.py` has no HF fallback for those tasks, so `eval.sh` fails on a missing
  `test.jsonl`. `prepare_eval_data.py --data-root <dir>` builds `data/{aime24,aime25,math500,amc12}/test.jsonl`
  from the HF snapshots listed in `downloads.txt` (`amc12` = `AI-MO/aimo-validation-amc`, 83 problems;
  registered in `Eval/parser.py`'s answer-field list). For the other tasks restore from `upstream/main`
  or point `--data_dir` at a copy.
- **`downloads.txt` lists what an offline server must fetch beforehand** (one `--hf-dataset` /
  `--hf <repo> <dest>` line each, `@PROJECT@` substituted by the download tool): the s1K CoT dataset
  (`baesad/s1K-1.1-deepseek-cot`, snapshot dir `s1k`, ships `train.jsonl`, `solution_segments.jsonl` (934 rows, `paragraph` split) **and** `solutions_selected.jsonl` (same rows + `selected_spans_ids` from the earlier R1-Distill IG run; 17 rows select nothing and fall back to the 3 default segments), so no attribution stage runs on the server unless that last file is missing; `prepare_s1k.py --dataset <dir>` also reads a raw `simplescaling/s1K-1.1` snapshot directly), the four eval benchmarks, `Qwen/Qwen2.5-7B-Instruct` (train) and
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (attribution). Pair with `run_pipeline.sh --offline`.
  `commands.sh` is the command sheet for that server; it currently runs **selective SFT with full finetuning on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** following a fixed hyperparameter table (its LoRA rows do not apply): 3 epochs, batch 1 x 32, seq 32768, lr 5e-5, max_grad_norm 1.0, AdamW (0.9, 0.999, 1e-8) wd 0, cosine warmup 0.1. Prompt and thinking mode match `../P-ALIGN/src/test.py` (`--force_empty_think`): P-ALIGN wording (`--prompt-style palign` / `--prompt-type palign`) and thinking off on both sides (`--think-prefix off` / `--no-think`); dir `..._epoch3_lr5e-5_len32768_nothink_palign`. The latest checkpoint is evaluated directly (no merge) with n=3 t=0.6 top_p=0.9 rep 1.05, `max_tokens 4096`, `max_model_len 4096`, 1 GPU, `gpu_mem 0.8`, tag `r1_1p5b_sel_ft_ep3_nothink_palign_4k` -> `pass_at_k.py` (grader.py scores, for reference) -> `Eval/score_palign.py` (rescores the same generations with P-ALIGN's `math_verify` + `report.py` formulas: gold = raw `answer`, pass@1 = mean correct/3 per question, pass@3 = any correct, unweighted AVG; writes `outputs_<tag>/palign_score.json`, the headline number for comparing with P-ALIGN; needs `math-verify` in the eval env) -> printed table (also shows the base model if evaluated). `commands.sh` now defaults to `SKIP_TRAIN=1` (eval the existing latest checkpoint, never touching `ssft_train`); `SKIP_TRAIN=0` retrains first (train.sh never skips on its own), `MODEL_CKPT=<dir>` evaluates a specific old checkpoint, `OVERWRITE=1` regenerates instead of resuming existing outputs; `EVAL_MAX_TOKENS` changes eval length, `max_model_len` and tag. Base-model eval and the `--full-sft` baseline sit commented under `[Tham khao]`.

## Defaults worth knowing

- Attribution model ≠ training model, as in the paper: `run_pipeline.sh` attributes with
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (the model that generated the s1K CoTs; paper App. C.3 uses
  it in every setting, even when the trained model is Qwen2.5-7B-Instruct) and `train.sh` trains
  `Qwen/Qwen2.5-7B-Instruct`. Both are listed in `downloads.txt`; `commands.sh` points `--attr-model` at
  the R1-Distill directory. Attributing with Qwen2.5-7B-Instruct instead runs, but that model was never
  trained on R1-style long CoT, so its IG scores are off-distribution relative to the paper's.
- IG attribution is the expensive stage: `ig_steps=20` (paper: 50) forward+backward passes over the full
  sequence per sample, on a 7B model. Model weights are frozen in `grad_analyze.py` — IG only needs
  gradients w.r.t. `inputs_embeds`, and freezing stops every `Linear` from saving its input activation.
  Paragraph splitting also produces far more segments per trace (~100-500) than the paper's cue
  splitting (~10-30), which changes how many segments clear the 70% mass threshold.
- **`--resume` does not check `ig_steps`.** It only compares each record's `question`, so resuming a
  file produced with J=50 under J=20 silently mixes the two. Start a fresh `ig` run after changing J.
- Eval benchmarks and samples/question: `aime24:32 amc23:32 math500:6 minerva:6 gpqa:6 olympiad:6`,
  temperature 0.6, top_p 1, max 32768 tokens, prompt type `deepseek-longcot` (the prompt string matches
  what `train_mask.py` builds, so training and eval stay aligned).
- W&B is off everywhere (`REPORT_TO=none`, `WANDB_MODE=disabled`); set `REPORT_TO=wandb` +
  `WANDB_PROJECT` to re-enable.
