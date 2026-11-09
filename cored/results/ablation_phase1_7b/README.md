# Table 2: Ablation of the Phase 1 objective (Qwen2.5-7B-Instruct, M=3)

Pass@1 = mean over the 3 experts of the 4-benchmark average (AIME24, AIME25, AMC, MATH500), k=3, temp 0.6, top_p 0.9, max_tokens 4096.
JS = token-level Jensen-Shannon divergence between experts, on key-step and non-key-step tokens.

| Configuration | config | JS-Key | JS-NonKey | Pass@1 | eval files | JS file | adapters (duyentl04/cwrd-qwen2.5-7b) |
|---|---|--:|--:|--:|---|---|---|
| Full objective | configs/proposal_v2/phase1_svgd_func_keystep_k1.yaml | 0.04145 | 0.01707 | 35.81 | abl_7b_full_expert{0,1,2}_exact.json | js/abl_7b_full_js.json | experts/ |
| w/o Lrep | configs/proposal_v2/phase1_svgd_func_keystep_k1_noSVGD.yaml | 0.00033 | 0.00018 | 35.18 | abl_7b_noSVGD_expert{0,1,2}_exact.json | js/abl_7b_noSVGD_js.json | ablation_phase1/noSVGD/ |
| w/o Lortho | configs/proposal_v2/phase1_svgd_func_keystep_k1_noOrtho.yaml | 0.04172 | 0.01709 | 34.86 | abl_7b_noOrtho_expert{0,1,2}_exact.json | js/abl_7b_noOrtho_js.json | ablation_phase1/noOrtho/ |
| w/o dropout | configs/proposal_v2/phase1_svgd_func_keystep_k1_noDropout.yaml | 0.04145 | 0.01704 | 35.41 | abl_7b_noDropout_expert{0,1,2}_exact.json | js/abl_7b_noDropout_js.json | ablation_phase1/noDropout/ |
| Lrep only | configs/proposal_v2/phase1_svgd_func_keystep_k1_onlySVGD.yaml | 0.04159 | 0.01704 | 34.44 | abl_7b_onlySVGD_expert{0,1,2}_exact.json | js/abl_7b_onlySVGD_js.json | ablation_phase1/onlySVGD/ |
| Lortho only | configs/proposal_v2/phase1_svgd_func_keystep_k1_onlyOrtho.yaml | 0.00030 | 0.00017 | 35.23 | abl_7b_onlyOrtho_expert{0,1,2}_exact.json | js/abl_7b_onlyOrtho_js.json | ablation_phase1/onlyOrtho/ |
| Dropout only | configs/proposal_v2/phase1_svgd_func_keystep_k1_onlyDropout.yaml | 0.00032 | 0.00017 | 35.14 | abl_7b_onlyDropout_expert{0,1,2}_exact.json | js/abl_7b_onlyDropout_js.json | ablation_phase1/onlyDropout/ |
| All three removed | configs/proposal_v2/phase1_svgd_func_keystep_k1_allOff.yaml | 0.00029 | 0.00016 | 34.83 | abl_7b_allOff_expert{0,1,2}_exact.json | js/abl_7b_allOff_js.json | ablation_phase1/allOff/ |
