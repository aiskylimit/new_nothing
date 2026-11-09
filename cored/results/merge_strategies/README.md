# Section 5.3.2: From council to student

All rows use the same 3 Phase 1 experts (phase1_svgd_func_keystep_k1), Qwen2.5-7B-Instruct, k=3, temp 0.6, top_p 0.9, max_tokens 4096.
Merges: `python -m proposal.merge --phase1_dir <experts> --M 3 --strategy <s> --out_rank 16` (CAT keeps rank 48); sweep in `scripts/run_keystep_merge_sweep.sh`.
The student row is TIES merge + weighted distillation (`scripts/run_merge_distill_keystep.sh`).

| Method | Pass@1 | Pass@3 | eval file | adapter (duyentl04/cwrd-qwen2.5-7b) |
|---|--:|--:|---|---|
| Expert 0 | 34.77 | 44.62 | proposal_v2_svgd_func_keystep_k1_expert0.json | experts/expert_0 |
| Expert 1 | 35.65 | 47.26 | proposal_v2_svgd_func_keystep_k1_expert1.json | experts/expert_1 |
| Expert 2 | 35.40 | 45.88 | proposal_v2_svgd_func_keystep_k1_expert2.json | experts/expert_2 |
| TIES | 36.50 | 47.01 | proposal_v2_svgd_func_keystep_k1_ties.json | merged_ties/ |
| Task Arithmetic | 36.08 | 49.54 | proposal_v2_svgd_func_keystep_k1_ta.json | merges/ta/ |
| CAT | 35.68 | 46.26 | proposal_v2_svgd_func_keystep_k1_cat.json | merges/cat/ |
| Linear averaging | 35.29 | 45.26 | proposal_v2_svgd_func_keystep_k1_linear.json | merges/linear/ |
| CtM | 35.13 | 46.26 | proposal_v2_svgd_func_keystep_k1_ctm.json | merges/ctm/ |
| DARE-TIES | 34.98 | 43.89 | proposal_v2_svgd_func_keystep_k1_dare_ties.json | merges/dare_ties/ |
| Iso-C | 34.91 | 43.86 | proposal_v2_svgd_func_keystep_k1_iso_c.json | merges/iso_c/ |
| TSV | 35.94 | 48.20 | proposal_v2_svgd_func_keystep_k1_tsv.json | merges/tsv/ |
| TIES + distill (student) | 37.13 | 46.36 | proposal_v2_svgd_func_keystep_k1_ties_distill.json | student/ |

The experts were also re-evaluated in a separate run for the Phase 1 ablation table (`abl_7b_full_expert{0,1,2}_exact.json`: 35.60 / 36.69 / 35.14); sampling is stochastic, so the two runs differ.
