# Table 4: Sensitivity to the fraction of steps selected for functional repulsion

Full pipeline per row (Phase 1 -> U,V weights -> TIES merge -> distill -> eval), Qwen2.5-7B-Instruct, k=3, temp 0.6, top_p 0.9, max_tokens 4096.
Run with `ROWS="K0 K20 K100" bash scripts/queue_ablation_keystep_trunc.sh <GPU>`.

| Key-step ratio | Pass@1 | eval file | adapters (duyentl04/cwrd-qwen2.5-7b) |
|---|--:|---|---|
| 0% | 36.16 | proposal_v2_abl_K0_distill.json | Phase 1: ablation_phase1/noSVGD/ (reused); merge/distill: ablation_keystep/K0/ |
| 20% | 34.96 | proposal_v2_abl_K20_distill.json | Phase 1: ablation_keystep/K20/phase1/; merge/distill: ablation_keystep/K20/ |
| 100% | 35.11 | proposal_v2_abl_K100_distill.json | Phase 1: ablation_keystep/K100/phase1/; merge/distill: ablation_keystep/K100/ |
