# SARW screening: region weighting on three students

Seed-42 screening of student-aligned region weighting (`sgl.allocation.region`) against NLL,
entropy IWC and answer-gain IWC on Qwen2.5-7B-Instruct, Qwen3-8B and R1-Distill-Qwen-1.5B
(October 2026). One training run per arm and one eval seed unless noted, so differences under
~1.5 points are within noise (eval-seed sd ~1 on Qwen3; training-seed sd ~0.8-1.0 on 1.5B).

## Setup

- Data: P-ALIGN's 966 traces (teacher R1 prefix closed by `<End_of_Prefix>`, then a continuation);
  the same traces for every track.
- Training: Qwen2.5-7B and Qwen3-8B with LoRA r16 (`configs/common/train/lora-r16.yaml`),
  R1-1.5B with full fine-tuning (`full-ds-z2.yaml`); 3 epochs, lr 5e-5 cosine to 1e-5, effective
  batch 32, training seed 42.
- Eval: `configs/common/eval/palign-protocol.yaml` (math500, aime24, aime25, amc12; n=3, T=0.6,
  top_p 0.9, 4096 tokens, math_verify). Scores are the mean over the four benchmarks.
- Arms: `region-iwc-l05` = a_k only (entropy IWC-Stable inside each region, lambda 0.5),
  `region-gain-g1` = b_r only (prefix/continuation budget from answer gain per token, gamma 1,
  last 10% of steps left out of the estimate), `sarw-l05-g1` = a_k * b_r.
  Configs: `configs/sgl/<track>/{region-iwc-l05,region-gain-g1,sarw-l05-g1}[-lora].yaml`.

## Offline signal analysis

- Global entropy IWC moves loss mass to the teacher prefix: prefix share 26.0% under NLL vs
  28.6% (Qwen3, +10%) and 29.9% (Qwen2.5, +15%) at lambda 0.5. Answer gain moves it the other
  way (-5% / -6%).
- Within-trace step-entropy ranks are nearly student-invariant (median Spearman 0.89 between
  Qwen3 and Qwen2.5); answer-gain ranks are not (0.21). Entropy and gain are uncorrelated
  (rho -0.05 / -0.12).
- Answer gain spikes on the last decile of steps (the answer, 44-52% of all continuation gain),
  hence `region_tail_fraction: 0.1`.
- Resulting budget ratio b_P / b_C (answer gain per token, prefix vs continuation):

| Student | G_P | G_C | b_P / b_C |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.00061 | 0.00159 | 0.38 |
| Qwen3-8B | 0.00222 | 0.00151 | 1.46 |
| R1-Distill-Qwen-1.5B | 0.00154 | 0.00101 | 1.51 |

## Results (seed 42)

pass@1:

| Arm | Qwen2.5-7B | Qwen3-8B | R1-1.5B | Mean |
|---|---|---|---|---|
| NLL | 24.97 | 49.58 | 41.41 | 38.65 |
| Entropy IWC lambda 0.5 | 25.74 | 49.63 | 41.34 | 38.90 |
| Answer-gain IWC lambda 0.5 | 28.44 | 49.15 | **42.74** | 40.11 |
| Region-IWC (a) | 28.12 | 47.46 | 39.65 | 38.41 |
| **Region-Gain (b)** | **30.42** | **50.01** | 41.52 | **40.65** |
| SARW (a * b) | 28.82 | 47.41 | 40.05 | 38.76 |

pass@3:

| Arm | Qwen2.5-7B | Qwen3-8B | R1-1.5B | Mean |
|---|---|---|---|---|
| NLL | 38.47 | **61.02** | 50.08 | 49.86 |
| Entropy IWC lambda 0.5 | 39.81 | 60.37 | 52.90 | 51.03 |
| Answer-gain IWC lambda 0.5 | 39.32 | 56.55 | 51.75 | 49.21 |
| Region-IWC (a) | **44.46** | 56.86 | **53.49** | **51.60** |
| Region-Gain (b) | 42.57 | 56.87 | 52.22 | 50.55 |
| SARW (a * b) | 41.04 | 57.89 | 50.53 | 49.82 |

Generation behaviour (share of generations that emit `<End_of_Prefix>` / that end in a degenerate
loop / "Wait" per 1k characters; all generations pooled):

| Arm | Qwen2.5-7B | Qwen3-8B | R1-1.5B |
|---|---|---|---|
| NLL | 45% / 15.7% / 1.84 | 22% / 0.0% / 0.68 | 92% / 0.5% / 0.33 |
| Entropy IWC | 27% / 16.8% / 3.13 | 14% / 0.1% / 0.74 | 90% / 0.3% / 0.38 |
| Region-IWC | 39% / 17.0% / 2.85 | 18% / 0.1% / 0.73 | 93% / 0.4% / 0.34 |
| Region-Gain | 64% / 10.5% / 0.42 | 16% / 0.0% / 0.67 | 89% / 0.5% / 0.43 |
| SARW | 61% / 12.7% / 0.59 | 13% / 0.1% / 0.73 | 89% / 0.4% / 0.39 |

Run names: Qwen tracks `<arm>-lora-<track>`; the Qwen2.5 entropy row is
`iwc-nogate-l05-r2-lora`, its gain row the older `iwc-gain-l05-lora`. 1.5B NLL / entropy / gain
are seed-42 reruns (`sft-nll-r2`, `iwc-nogate-l05-r2`, `iwc-gain-l05-r2`) so raw generations
exist; they match the older runs (41.99, 41.16, 42.31) within 0.6 points. Older 1.5B training
seeds 43/44: NLL 40.93 / 40.36, entropy IWC 42.64 / 43.00.

## Answer-gain lambda sweep (eval seeds 42/43/44, mean +- sd)

| lambda | Qwen3-8B pass@1 | Qwen3-8B pass@3 | Qwen2.5-7B pass@1 | Qwen2.5-7B pass@3 |
|---|---|---|---|---|
| 0.1 | 48.05 +- 1.30 | 57.69 +- 0.17 | 26.25 +- 0.84 | 39.27 +- 2.38 |
| 0.3 | 48.62 +- 0.24 | 58.54 +- 1.28 | 28.37 +- 0.29 | 42.00 +- 0.43 |
| 0.5 | 48.48 +- 0.66 | 57.61 +- 1.21 | 27.89 +- 1.98 | 40.36 +- 2.03 |
| 0.7 | 47.67 (seed 42) | 55.82 | - | - |

The sd is over eval seeds of one checkpoint, not over training runs.

## Findings

1. Region-Gain has the best mean pass@1 (+2.0 over NLL): +5.4 on Qwen2.5, where it also fixes the
   failure mode (prefix exit 45% -> 64%, loops 15.7% -> 10.5%, "Wait" 1.84 -> 0.42), and no loss
   on the students that are compatible with the R1 prefix (Qwen3 +0.4, R1-1.5B +0.1).
2. Global entropy IWC keeps both Qwen students in the prefix (fewer exits, more truncation); it
   only costs accuracy on Qwen2.5. The mechanism is shared, the damage is student dependent.
3. The factorization does not compose: SARW (a * b) is below Region-Gain on all three students.
   Within-region entropy (a) helps over global entropy on Qwen2.5 only, and gives the best mean
   pass@3, but lowers pass@1 elsewhere.
4. Still single seed, and Qwen2.5 stays below P-ALIGN's published ~34.3.

## Next

- Seeds 43/44 (training and eval) for NLL, Region-Gain and answer-gain lambda 0.5 on all three.
- a_k from answer gain inside each region instead of entropy, times b_r; a gamma sweep for b_r.
- P-ALIGN reproduced in this pipeline as the matched baseline.

## Reproducing

```bash
sgl run qwen25-7b-palign/region-gain-g1-lora          # likewise qwen3-8b-palign/..., r1-qwen-1.5b-palign/region-gain-g1
sgl run qwen25-7b-palign/sarw-l05-g1-lora region_gamma=2.0   # gamma / tail overrides
```

Behaviour metrics were computed from `results/<run>/raw/*.jsonl` (untracked); summaries are in
`results/<run>/summary.json` and the run logs in `logs/`.
