# Table: Uncertainty / Disagreement terms in w_i

w_i = 1 + λ_U·tanh(Û_i) + λ_D·tanh(ρ̂_i). Same experts (phase1_svgd_func_keystep_k1), same TIES teacher, λ_KL = 1; only w_i changes.
Weight caches are rebuilt from the cached U, V with `scripts/rewrite_weights.py` (dataset repo: `weights_svgd_func_keystep_k1_{B1,B2}`); A1 uses `--uniform_weights`.
Run with `bash scripts/run_ablation_w_distill.sh <GPU>`.

| Uncertainty | Disagreement | Run | Pass@1 | Pass@3 | eval file | adapter (duyentl04/cwrd-qwen2.5-7b) |
|---|---|---|--:|--:|---|---|
| off | off | A1 | 35.91 | 43.91 | proposal_v2_svgd_func_keystep_k1_ablw_A1_distill.json | ablation_weights/A1/ |
| on | off | B1 | 37.21 | 48.60 | proposal_v2_svgd_func_keystep_k1_ablw_B1_distill.json | ablation_weights/B1/ |
| off | on | B2 | 35.84 | 46.71 | proposal_v2_svgd_func_keystep_k1_ablw_B2_distill.json | ablation_weights/B2/ |
