"""Bản dựng lại theo proposal "Council-Weighted Reasoning Distillation via Model Merging".

Khác nhánh `train/` ở Phase 1 §4.3: dùng DPP + Kneedle Dynamic Top-k (Eq 12-16)
thay cho controlled-JS. Phase 2 tái dùng train.compute_weights + train.phase2_distill
(lambda_kl=0 = Eq 27).
"""
