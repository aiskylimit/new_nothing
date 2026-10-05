# Experiment notes, 2-4 October 2026

Running log of the seed-42 experiments on Qwen2.5-7B-Instruct, Qwen3-8B and R1-Distill-Qwen-1.5B
with P-ALIGN's 966 traces. Complements `docs/sarw-screening.md` (SARW design and first screening).
All scores are pass@1 / pass@3 (%) at eval seed 42 and training seed 42, with
`configs/common/eval/palign-protocol.yaml` (n=3, T=0.6, top_p 0.9, 4096 tokens, math_verify) and
the sgl evaluator unless stated otherwise. Avg = unweighted mean over the four benchmarks. One
seed: differences under ~1.5-2 points are within noise.

## Setup reminders

- LoRA tracks (Qwen2.5-7B, Qwen3-8B): r16 / alpha 16 / dropout 0.05 on all projections, 3 epochs,
  lr 5e-5 cosine to 1e-5, warmup 0.1, seq 32k, DeepSpeed ZeRO-2 CPU offload. Effective batch 32
  (93 steps) unless a run is marked "b8" (`stages.train.effective_batch=8`, 363 steps).
- R1-1.5B: full fine-tuning, same schedule, effective batch 32.
- Zero-shot = the untrained base model under the same prompt (`results/zeroshot-<track>`).
- Train/eval prompt consistency was checked token by token (966/966 on all three tracks) and 100%
  of trained-model generations start with `<Begin_of_Prefix>`: the sgl runs do not have the
  template mismatch of the earlier B200 P-ALIGN run (train template qwen3, eval Qwen2.5 template).

## Qwen2.5-7B-Instruct

| Run | MATH500 | AIME24 | AIME25 | AMC12 | Avg p@1 | Avg p@3 |
|---|---|---|---|---|---|---|
| Zero-shot | 74.7 / 85.2 | 8.9 / 13.3 | 4.4 / 10.0 | 45.0 / 63.9 | 33.24 | 43.10 |
| P-ALIGN, external repo code, b8 (1) | 71.9 / 83.4 | 12.2 / 20.0 | 12.2 / 23.3 | 38.6 / 51.8 | 33.72 | 44.64 |
| NLL | 60.1 / 77.8 | 5.6 / 16.7 | 3.3 / 10.0 | 30.9 / 49.4 | 24.97 | 38.47 |
| Entropy IWC lambda 0.5 (old run) | 56.2 / 76.6 | 5.6 / 13.3 | 8.9 / 13.3 | 30.1 / 43.4 | 25.19 | 36.66 |
| Entropy IWC lambda 0.5 (r2) | 55.9 / 75.6 | 13.3 / 23.3 | 4.4 / 13.3 | 29.3 / 47.0 | 25.74 | 39.81 |
| Answer-gain IWC lambda 0.1 | 58.5 / 77.8 | 8.9 / 16.7 | 6.7 / 13.3 | 30.9 / 50.6 | 26.25 | 39.60 |
| Answer-gain IWC lambda 0.3 | 62.1 / 79.6 | 7.8 / 16.7 | 12.2 / 20.0 | 30.1 / 53.0 | 28.05 | 42.32 |
| Answer-gain IWC lambda 0.5 (old run) | 60.5 / 76.4 | 8.9 / 13.3 | 6.7 / 13.3 | 37.8 / 54.2 | 28.44 | 39.32 |
| Region-IWC (a) | 57.7 / 77.6 | 10.0 / 20.0 | 6.7 / 20.0 | 38.2 / 60.2 | 28.12 | 44.46 |
| Region-Gain gamma 1 (b) | 67.1 / 81.8 | 8.9 / 16.7 | 11.1 / 20.0 | 34.5 / 51.8 | 30.42 | 42.57 |
| SARW (a * b) | 63.3 / 77.8 | 8.9 / 16.7 | 7.8 / 16.7 | 35.3 / 53.0 | 28.82 | 41.04 |
| Region-Gain gamma 2 | 69.3 / 83.2 | 7.8 / 13.3 | 10.0 / 23.3 | 37.8 / 61.4 | 31.22 | 45.33 |
| Continuation-only (`region-cont`) (2) | 73.2 / 84.2 | 12.2 / 23.3 | 8.9 / 16.7 | 38.6 / 53.0 | 33.22 | 44.30 |
| NLL + tail x3 (`nll-tail128`) | 61.3 / 79.6 | 6.7 / 10.0 | 5.6 / 10.0 | 32.1 / 50.6 | 26.40 | 37.55 |
| Region-Gain gamma 1 + tail x3 | 67.4 / 83.0 | 8.9 / 13.3 | 7.8 / 16.7 | 33.7 / 49.4 | 29.45 | 40.60 |
| Answer-gain IWC lambda 0.5, b8 | 71.3 / 84.0 | 13.3 / 20.0 | 5.6 / 10.0 | 37.3 / 53.0 | 31.89 | 41.75 |
| Region-Gain gamma 2, b8 | 71.1 / 83.8 | 7.8 / 10.0 | 7.8 / 16.7 | 37.0 / 51.8 | 30.91 | 40.57 |
| NLL, b8 | trained (loss 0.474, no NaN), not evaluated (stopped) | | | | | |

(1) Scored with the external repo's `test.py` / `evaluation.py` / `report.py` (no sampling seed),
LLaMA-Factory 0.9.5, `\n` between instruction and problem, lr to 0, eager attention (see below).
(2) Matches zero-shot but 0% of generations use `<Begin_of_Prefix>`: the continuation is the
student's own text, so this is self-distillation, not learning from R1.

Generation behaviour (all generations pooled: emits `<End_of_Prefix>` / truncated at 4096 /
degenerate loop / "Wait" per 1k characters):

| Run | End | Trunc | Loop | Wait/kc |
|---|---|---|---|---|
| Zero-shot | - | 0.3% | 0.1% | 0.00 |
| NLL | 45% | 46% | 15.7% | 1.84 |
| Entropy IWC (r2) | 27% | 56% | 16.8% | 3.13 |
| Region-IWC | 39% | 51% | 17.0% | 2.85 |
| Region-Gain gamma 1 | 64% | 32% | 10.5% | 0.42 |
| SARW | 61% | 37% | 12.7% | 0.59 |
| Region-Gain gamma 2 | 81% | 18% | 5.8% | 0.13 |
| NLL + tail x3 | 44% | 35% | 9.0% | 1.41 |
| Region-Gain gamma 1 + tail x3 | 61% | 25% | 6.9% | 0.45 |
| Answer-gain IWC lambda 0.5, b8 | 90% | 14% | 2.1% | 0.54 |
| Region-Gain gamma 2, b8 | 99% | 10% | 1.8% | 0.25 |
| P-ALIGN external, b8 | 87% | ~12% (est.) | 1.5% | 1.95 Wait/generation |

## Qwen3-8B (batch 32)

| Run | MATH500 | AIME24 | AIME25 | AMC12 | Avg p@1 | Avg p@3 |
|---|---|---|---|---|---|---|
| Zero-shot (thinking off) | 84.1 / 92.4 | 24.4 / 36.7 | 20.0 / 26.7 | 61.8 / 74.7 | 47.61 | 57.61 |
| NLL | 84.5 / 91.8 | 30.0 / 43.3 | 25.6 / 36.7 | 58.2 / 72.3 | 49.58 | 61.02 |
| Entropy IWC lambda 0.5 | 83.5 / 91.6 | 31.1 / 43.3 | 24.4 / 36.7 | 59.4 / 69.9 | 49.63 | 60.37 |
| Answer-gain IWC lambda 0.1 | 84.7 / 90.8 | 30.0 / 43.3 | 18.9 / 26.7 | 56.6 / 69.9 | 47.55 | 57.67 |
| Answer-gain IWC lambda 0.3 | 84.0 / 91.2 | 26.7 / 36.7 | 21.1 / 33.3 | 62.2 / 77.1 | 48.51 | 59.58 |
| Answer-gain IWC lambda 0.5 | 85.3 / 91.8 | 31.1 / 36.7 | 21.1 / 26.7 | 59.0 / 71.1 | 49.15 | 56.55 |
| Answer-gain IWC lambda 0.7 | 84.5 / 91.0 | 25.6 / 33.3 | 20.0 / 26.7 | 60.6 / 72.3 | 47.67 | 55.82 |
| Region-IWC (a) | 84.9 / 91.8 | 26.7 / 36.7 | 18.9 / 26.7 | 59.4 / 72.3 | 47.46 | 56.86 |
| Region-Gain (b) | 84.1 / 91.2 | 33.3 / 43.3 | 25.6 / 26.7 | 57.0 / 66.3 | 50.01 | 56.87 |
| SARW (a * b) | 83.5 / 90.2 | 26.7 / 40.0 | 20.0 / 26.7 | 59.4 / 74.7 | 47.41 | 57.89 |

Eval-seed means (seeds 42/43/44, one checkpoint each) for the gain sweep: lambda 0.1
48.05 +- 1.30, 0.3 48.62 +- 0.24, 0.5 48.48 +- 0.66 (pass@1).

## R1-Distill-Qwen-1.5B (full FT, batch 32)

| Run | MATH500 | AIME24 | AIME25 | AMC12 | Avg p@1 | Avg p@3 |
|---|---|---|---|---|---|---|
| Zero-shot (closed think block, as in training) | 71.5 / 85.4 | 10.0 / 26.7 | 17.8 / 30.0 | 48.6 / 66.3 | 36.96 | 52.08 |
| NLL (r2) | 76.5 / 87.4 | 17.8 / 20.0 | 20.0 / 26.7 | 51.4 / 66.3 | 41.41 | 50.08 |
| NLL (old run) | 76.7 / 87.6 | 20.0 / 30.0 | 16.7 / 23.3 | 54.6 / 66.3 | 41.99 | 51.80 |
| Entropy IWC lambda 0.5 (r2) | 78.1 / 87.2 | 16.7 / 30.0 | 15.6 / 23.3 | 55.0 / 71.1 | 41.34 | 52.90 |
| Entropy IWC lambda 0.5 (old run) | 77.6 / 87.4 | 20.0 / 33.3 | 14.4 / 23.3 | 52.6 / 65.1 | 41.16 | 52.28 |
| Answer-gain IWC lambda 0.5 (r2) | 78.3 / 87.4 | 20.0 / 26.7 | 18.9 / 26.7 | 53.8 / 66.3 | 42.74 | 51.75 |
| Answer-gain IWC lambda 0.5 (old run) | 78.9 / 88.8 | 21.1 / 33.3 | 16.7 / 23.3 | 52.6 / 62.7 | 42.31 | 52.03 |
| Region-IWC (a) | 77.2 / 88.6 | 18.9 / 26.7 | 11.1 / 30.0 | 51.4 / 68.7 | 39.65 | 53.49 |
| Region-Gain (b) | 78.1 / 87.8 | 18.9 / 30.0 | 14.4 / 20.0 | 54.6 / 71.1 | 41.52 | 52.22 |
| SARW (a * b) | 77.6 / 88.0 | 14.4 / 20.0 | 15.6 / 26.7 | 52.6 / 67.5 | 40.05 | 50.53 |

Older 1.5B training seeds 43/44: NLL 40.93 / 40.36, entropy IWC 42.64 / 43.00.

## Findings

1. **Region budget b_P / b_C from answer gain per token adapts to the student** without tuning:
   Qwen2.5 0.38, Qwen3 1.46, R1-1.5B 1.51 (last 10% of steps left out of the estimate).
2. **Region-Gain** has the best mean pass@1 at batch 32 (40.65 vs 38.65 for NLL over the three
   students): +5.4 on Qwen2.5 with far fewer loops, no loss on Qwen3 (+0.4) or R1-1.5B (+0.1).
   The factorization does not compose: SARW (a * b) is below Region-Gain on all three.
3. **Training helps Qwen3 (+2.4 over zero-shot) and R1-1.5B (+5.8) but not Qwen2.5 at 4096
   tokens**: at batch 32 every R1-learning arm stays below zero-shot (33.24) on pass@1.
4. **Batch size was the largest missing factor on Qwen2.5.** Effective batch 8 (4x the optimizer
   steps) lifts answer-gain IWC lambda 0.5 from 28.44 to 31.89, cuts loops from ~16% to ~2% and
   raises prefix exits to 90%. Region-Gain gamma 2 at b8 (30.91) is not above its b32 run (31.22).
5. **Tail boost** (last 128 supervised tokens x3) cuts loops by ~40% (15.7% -> 9.0% on NLL) but
   does not move accuracy beyond noise.
6. **P-ALIGN, run correctly, is a valid but slim baseline on Qwen2.5**: 33.72 / 44.64 with the
   learned format in use (100% `<Begin_of_Prefix>`), slightly above zero-shot. The older B200
   P-ALIGN number (34.28) came from a template mismatch and measured the base model.

## External P-ALIGN code: grad NaN

Code: https://github.com/aiskylimit/new_nothing branch `palign`, kept in
`/root/palign-aiskylimit-qwen25-7b` (provenance in its SOURCE.md; Qwen2.5 configs restored from
commit 10c9eda), fresh conda env `palign-repo` (torch 2.13 cu130, transformers 5.5.4,
LLaMA-Factory 0.9.5, vLLM 0.27.1).

- With its default `flash_attn: auto` (SDPA), the LoRA run gets grad_norm NaN from step 2 (step 1
  loss 0.708, grad 0.0175, lr 0; step 2 loss 0.666 finite, grad NaN), so every LoRA weight is NaN
  afterwards. Hugging Face's `logging_nan_inf_filter` hides it (loss logged as 0.0).
- Reproduced in a fresh env matching the B200 versions. Not caused by gradient checkpointing (the
  repo's patch), layernorm precision, the data, or the env clone.
- Plain transformers + PEFT with SDPA (no LLaMA-Factory) gives finite loss and gradients on all
  966 samples; the sgl trainer (same torch, SDPA) never produced NaN.
- `flash_attn=disabled` (eager attention) trains cleanly (363 steps, no NaN, train loss 0.491).
  Step-1 loss differs between SDPA (0.708) and eager (0.783) on the same batch, which points at
  LLaMA-Factory's SDPA path (mask handling) under transformers 5.5.x; not pinned to a line.
- The NaN run is archived in `output/run1-sdpa-nan/`; the valid run's results are in
  `output/eval_results.txt` and `output/result/`.

## Not finished / follow-ups

- NLL b8 on Qwen2.5 is trained (`checkpoints/sft-nll-b8-lora-qwen25-7b-palign`) but not evaluated:
  `sgl run qwen25-7b-palign/sft-nll-lora --stages eval name=sft-nll-b8-lora stages.train.effective_batch=8`.
- Seeds 43/44 for the b8 runs and for Region-Gain on all students.
- P-ALIGN in the sgl pipeline (NLL b8 is the closest match) to compare under one evaluator.
- Evaluation at 8k/16k tokens, where R1-style traces are not truncated.
