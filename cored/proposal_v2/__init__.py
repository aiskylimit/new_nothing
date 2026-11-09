"""proposal_v2 — cùng ý tưởng (council đa dạng -> merge -> council-weighted distill),
cách làm cải thiện:

  #0  prefix_truncate.py   — cắt CoT teacher xuống prefix tối thiểu student giải được (P-ALIGN).
  Phase 1 (proposal/phase1.py + cờ mới): warmup λ_div + cap L_DPP <= 0.3*L_NLL + sweep {0.02,0.03,0.05}.
  Phase 2 (phase2_ekd.py) — Council-Weighted Ensemble Distillation:
        teacher = ensemble 3 expert (không phải merged), weight mức TOKEN theo bất đồng V_t,
        L_t = w_t * [ a*CE(student, y*) + (1-a)*KL(student || ensemble) ].

Merge vẫn giữ (Phase 2 baseline / khởi tạo student). Bảng so P-ALIGN Pass@1/@3 không đổi.
"""
