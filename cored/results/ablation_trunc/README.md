# Table: CoT truncation strategy

Full pipeline per row (Phase 1 -> U,V weights -> TIES merge -> distill -> eval), Qwen2.5-7B-Instruct, k=3, temp 0.6, top_p 0.9, max_tokens 4096.
Retained tokens = Qwen2.5 tokenizer count of the kept `thinking` text / count on the untruncated CoT (adaptive: over its 999 samples).
Data variants are built by `python -m proposal_v2.prefix_variants --out_dir data/ablation_trunc` (dataset repo: `ablation_trunc/`).

| Strategy | Train data | Retained tokens (%) | Pass@1 | Pass@3 | eval file | adapters (duyentl04/cwrd-qwen2.5-7b) |
|---|---|--:|--:|--:|---|---|
| Fixed prefix, k = ceil(0.5 · n_steps) | ablation_trunc/f050.jsonl | 57.3 | 35.45 | 46.84 | proposal_v2_abl_T050_distill.json | ablation_trunc/T050/ |
| No truncation | ablation_trunc/full.jsonl | 100 | 36.38 | 46.58 | proposal_v2_abl_T100_distill.json | ablation_trunc/T100/ |
