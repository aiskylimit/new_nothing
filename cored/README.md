# Council-Weighted Reasoning Distillation (proposal_v2, svgd_func + key step)

Base: `Qwen/Qwen2.5-7B-Instruct`, data: s1K. Phase 1 trains M=3 LoRA experts (grad dropout + dual-ortho + svgd_func on key steps); Phase 2 merges them with TIES and distills a single LoRA student.

## Pipeline
| Step | Command |
|---|---|
| 0. Adaptive CoT truncation | `python -m proposal_v2.prefix_truncate` → `data/train_s1k_prefixed.jsonl` |
| 1. Key step (top 10% entropy under base) | `python -m proposal_v2.keystep_select` → `outputs/proposal_v2/keystep_s1k_prefixed_10pct` |
| 2. Phase 1 experts | `python -m proposal_v2.phase1 --config configs/proposal_v2/phase1_svgd_func_keystep_k1.yaml` |
| 3. Council weights w_i (U, V, rho) | `python -m train.compute_weights ...` |
| 4–6. TIES merge → distill → eval | `bash scripts/run_merge_distill_keystep.sh` |

Phase 1 ablations: `configs/proposal_v2/phase1_svgd_func_keystep_k1_{allOff,noDropout,noOrtho,noSVGD,only*}.yaml`, run with `scripts/run_ablation_row.sh`.

## Results (Pass@1 / Pass@3, %, k=3, temp 0.6, top_p 0.9, max_tokens 4096)
| Model | AIME25 | AIME24 | AMC | MATH500 | Avg |
|---|--:|--:|--:|--:|--:|
| Zero-shot base | 7.8/16.7 | 10.0/16.7 | 43.4/56.6 | 76.3/85.4 | 34.37/43.84 |
| Expert 0 | 8.9/16.7 | 12.2/20.0 | 42.6/55.4 | 75.4/86.4 | 34.77/44.62 |
| Expert 1 | 11.1/20.0 | 11.1/20.0 | 44.2/62.7 | 76.2/86.4 | 35.65/47.26 |
| Expert 2 | 7.8/16.7 | 12.2/20.0 | 45.8/61.4 | 75.8/85.4 | 35.40/45.88 |
| Merge TIES (phi_merge) | 11.1/20.0 | 12.2/20.0 | 46.2/61.4 | 76.5/86.6 | 36.50/47.01 |
| Student, NLL only (lambda_KL=0) | 10.0/16.7 | 15.6/30.0 | 41.8/59.0 | 74.2/85.2 | 35.38/47.73 |
| **Student, full (lambda_KL=1)** | 11.1/16.7 | 14.4/23.3 | 45.8/59.0 | 77.2/86.4 | 37.13/46.36 |

AMC is read from `p_align_repo/data/result/amc12-output.jsonl` (83 problems); MATH500, AIME24 and AIME25 are loaded from the HF Hub. Raw per-problem scores are in `results/*.json`.
