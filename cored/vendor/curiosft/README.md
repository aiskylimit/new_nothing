# vendor/curiosft — mã lấy lại từ CurioSFT

| | |
|---|---|
| Upstream | https://github.com/HaoooWang/CurioSFT |
| Commit tham chiếu | `a83b129` ("Update README.md") — clone `--depth 1`, 2026-09-04 |
| Bài báo | *Learning While Staying Curious: Entropy-Preserving SFT via Adaptive Self-Distillation for Large Reasoning Models*, ACL 2026 Main, [arXiv:2602.02244](https://arxiv.org/abs/2602.02244) |
| License upstream | Apache-2.0 (repo dựa trên fork của `verl`, Bytedance Ltd., Apache-2.0) |
| Data + model | `Hao0oWang/CurioSFT_Data`, `Hao0oWang/Qwen2.5-Math-7B-16k-think` (HF Hub) |

## Đã lấy gì

| File | Nguồn upstream | Nội dung |
|---|---|---|
| `tau_search.py` | `custom_verl/recipe/curio_sft/dp_actor.py::compute_tau_from_logits` (dòng 545–595) | Entropy-Guided Temperature Selection: binary search vectorized tìm `tau_i` cho teacher, xấp xỉ entropy bằng top-k logits. Đã gỡ phụ thuộc `verl`/`flash_attn`, tách hằng số ra keyword (giữ nguyên giá trị gốc). |

Công thức loss CurioSFT (`custom_verl/recipe/curio_sft/ppo/core_algos.py::custom_loss`,
nhánh `algo_type == "curio_sft"`) **không vendor thành file** vì chỉ 3 dòng — đã
cài trực tiếp ở `core/sed.py::k2_step_loss` + `train/sft_engine.py` (nhánh SED):

```python
# mse_loss = -log p_student(y_t)                      # = NLL expert token (giống core/losses.py)
# kd_loss  = alpha * 0.5 * (log p_student(y_t) - log p_teacher_tau(y_t))**2   # K2-loss, eq. (5), alpha=1
# loss     = mean_over_response_tokens(mse_loss + kd_loss)
```

`log p_teacher_tau` = log-prob của expert token dưới **base đóng băng đã scale
nhiệt độ per-token** bằng `tau_search.teacher_logprobs_with_egts`.

## KHÔNG lấy (không dùng được cho repo này)

Toàn bộ phần còn lại của CurioSFT chạy trên **`verl` + Ray + FSDP + vLLM async
rollout**, điều khiển qua `recipe.curio_sft.main_ppo` với `policy_loss.loss_mode=custom_loss`.
Repo của ta là vòng lặp AdamW đơn tiến trình + PEFT (`train/sft_engine.py`), nên:

- `custom_verl/` (fork verl ~ vài nghìn file) — bỏ.
- `recipe/curio_sft/{main_ppo,fsdp_workers,dp_actor,ppo/ray_trainer}.py` — hạ tầng PPO/FSDP, bỏ. Chỉ đọc `dp_actor.py` để lấy đúng chỗ EGTS được áp (chia teacher logits cho `tau`, token entropy ≤ 0.03 → `tau_min`).
- `recipe/curio_sft/agent_loop/`, `reward_function/` (reward server HTTP, oat-math grader) — bỏ; ta đã có `eval/grade.py` (math-verify).
- EMA teacher: upstream dùng cờ verl `sync_ref_model=True, ref_model_sync_steps=5, ref_model_mixup_alpha=0.99`. Ta **bỏ EMA**, dùng thẳng base đóng băng làm teacher (ablation "w/o Separate Teacher" của bài chỉ mất ~1 điểm) → tiết kiệm 1 bản 7B trên GPU share.
- `scripts/*.sh` — lệnh verl 8×H800, không tái dùng; đã đọc để trích siêu tham số (LR 1e-5, cosine, warmup 60 step, batch 256, 3 epoch, max_response 8192, eval `val_kwargs.n=8` temp 0.6).

## Hằng số EGTS (từ `dp_actor.py` + Fig. 6 của bài)

| | repo (`dp_actor.py`) | Fig. 6 trong bài | dùng ở đây |
|---|---|---|---|
| `tau_min, tau_max` | 1.1, 1.5 | 1.1, 1.5 | 1.1, 1.5 |
| `topk` | 512 | 512 | 512 |
| `iters` | 20 | 15 | **20** (theo repo) |
| `h_pivot` (c) | 1.2 | 1.2 | 1.2 |
| `gamma` (scale) | 2.0 | 2.0 | 2.0 |
| `delta_max` | 0.5 | 0.5 | 0.5 |
| low-entropy cutoff | `H_now <= 0.03 → tau_min` | (không nêu) | 0.03 |

## Cập nhật lại

```bash
git clone --depth 1 https://github.com/HaoooWang/CurioSFT.git /tmp/CurioSFT
# so lại /tmp/CurioSFT/custom_verl/recipe/curio_sft/dp_actor.py :: compute_tau_from_logits
```
