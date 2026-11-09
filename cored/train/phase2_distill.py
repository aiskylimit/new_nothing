"""Phase 2 — Bước C: Distillation student đầy đủ (NLL + KL).

Đề xuất v2, Mục 5.4:

    L_NLL = sum_i w_i * L_i^{student}        / sum_i w_i        (step-mean)
    D_i   = (1/T_i) sum_{t in step i} KL( p_teacher,t || p_student,t )
    L_KL  = sum_i w_i * D_i                  / sum_i w_i        (step-mean)
    L     = L_NLL + lambda_KL * L_KL

Trong đó:
  * teacher = theta_base + phi_merge (adapter đã gộp, ĐÓNG BĂNG);
  * student = adapter LoRA trainable, **khởi tạo bằng chính phi_merge**
    (proposal: init từ phi_merge, không phải từ expert đơn lẻ);
  * w_i lấy từ cache ``outputs/weights_*/{id}.npz`` (khoá ``w``), giống
    ``train/phase2_gate.py``; segment cuối (câu trả lời) luôn w = 1.

Khác với đề xuất ở một điểm có lợi: teacher được forward **on-the-fly** nên KL
tính trên TOÀN BỘ từ vựng, không phải xấp xỉ top-50. Với s1K (1000 mẫu, ~93
optimizer step) chi phí thêm một forward no_grad là chấp nhận được.

    python -m train.phase2_distill --config configs/phase2_s1k_distill.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.data_utils import JsonlLogger, StepDataset, set_seed, track_cli_provided
from core.losses import (
    step_nll_from_logprobs,
    step_token_counts,
    token_logprobs,
    weighted_step_loss,
)
from core.model_utils import (
    lora_parameters,
    load_base_model,
    load_tokenizer,
    set_lora_trainable,
)
from train.common import (
    Accumulator,
    add_common_args,
    build_optimizer,
    clip_and_step,
    cosine_schedule_with_warmup,
    resolve_device,
    save_run_config,
    total_optimizer_steps,
)
from train.sft_engine import build_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Bước C: student distillation (NLL + KL)")
    add_common_args(parser)
    parser.add_argument("--weights_dir", default="outputs/weights_s1k_rho",
                        help="Thư mục cache npz chứa khoá 'w' (từ train/compute_weights.py)")
    parser.add_argument("--teacher_adapter", default=None,
                        help="Thư mục adapter đã gộp phi_merge — vừa làm teacher vừa làm init student")
    parser.add_argument("--lambda_kl", type=float, default=1.0,
                        help="Trọng số số hạng KL (proposal quét {0.5, 1, 2}); 0 = chỉ w_i-NLL")
    parser.add_argument("--kl_chunk", type=int, default=256,
                        help="Số vị trí token / chunk khi tính KL toàn từ vựng")
    parser.add_argument("--teacher_chunk", type=int, default=512,
                        help="Số token / chunk khi stream forward teacher (KV cache)")
    parser.add_argument("--uniform_weights", action="store_true",
                        help="Ablation: bỏ w đã cache, dùng w ≡ 1")
    parser.add_argument("--ckpt_every", type=int, default=5,
                        help="Ghi checkpoint resume (student + optimizer + scheduler) mỗi K optimizer step; 0 = tắt")
    parser.add_argument("--no_auto_resume", action="store_true",
                        help="Không tự động tiếp tục từ checkpoint trong <out_dir>/resume_ckpt")
    parser.add_argument("--skip_missing_ok", type=int, default=1,
                        help="1: bỏ qua mẫu thiếu cache w; 0: báo lỗi")
    parser.set_defaults(out_dir="outputs/ours_s1k_distill", loss_reduction="step_mean",
                        lr=1e-4, cutoff_len=16384)
    return track_cli_provided(parser)


# --------------------------------------------------------------------------- #
# Trọng số segment (giống train/phase2_gate.py::_segment_weights)
# --------------------------------------------------------------------------- #
def load_cached_w(weights_dir: str, example_id: str) -> Optional[np.ndarray]:
    path = os.path.join(weights_dir, f"{example_id}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "w" not in data.files:
            return None
        return np.asarray(data["w"], dtype=np.float32)


def segment_weights(w: Optional[np.ndarray], n_segments: int, uniform: bool) -> Optional[torch.Tensor]:
    if uniform or w is None:
        return torch.ones(n_segments)
    w = torch.from_numpy(w).reshape(-1)
    if w.numel() == n_segments:
        return w
    if w.numel() == n_segments - 1:  # pad segment câu trả lời với w = 1
        return torch.cat([w, torch.ones(1)], dim=0)
    return None  # cache thuộc mẫu khác (id trùng) -> caller bỏ qua


# --------------------------------------------------------------------------- #
# Teacher forward (streaming) + KL toàn từ vựng theo step
# --------------------------------------------------------------------------- #
@torch.no_grad()
def teacher_logits_once(model, input_ids: torch.Tensor) -> torch.Tensor:
    """(T, V) bf16 — logits của theta_base + phi_merge (adapter 'teacher' đang active).

    Forward một lần dưới ``eval`` (tắt dropout của adapter) + ``no_grad``. Không cần
    KV-cache streaming: một tensor (T, V) bf16 ~5GB ở T=16k, giải phóng ngay sau KL.
    """
    was_training = model.training
    model.eval()
    try:
        out = model(input_ids=input_ids, use_cache=False)
        return out.logits[0].to(torch.bfloat16)
    finally:
        if was_training:
            model.train()


def step_kl_full_vocab(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    step_spans,
    chunk: int,
) -> torch.Tensor:
    """D_i = mean_{t in step i} KL( p_teacher,t || p_student,t ) trên toàn từ vựng.

    Quy ước alignment giống core/losses: logits[t] dự đoán token[t+1]. Với step
    span (s, e) trong toạ độ input_ids, các vị trí dự đoán là [s-1, e-1).
    Giữ nguyên đồ thị autograd của ``student_logits`` (gradient chảy qua KL).
    """
    seq_len = student_logits.shape[0]
    out = student_logits.new_zeros(len(step_spans))
    for i, (start, end) in enumerate(step_spans):
        start = max(int(start), 1)
        end = min(int(end), seq_len)
        lo, hi = start - 1, end - 1
        if hi <= lo:
            continue
        kl_sum = student_logits.new_zeros(())
        n_tok = 0
        for c_start in range(lo, hi, chunk):
            c_end = min(c_start + chunk, hi)
            t_log_p = F.log_softmax(teacher_logits[c_start:c_end].float(), dim=-1)
            t_p = t_log_p.exp()
            s_log_p = F.log_softmax(student_logits[c_start:c_end].float(), dim=-1)
            kl_tok = (t_p * (t_log_p - s_log_p)).sum(dim=-1)  # (chunk,)
            kl_sum = kl_sum + kl_tok.sum()
            n_tok += kl_tok.shape[0]
        out[i] = kl_sum / max(n_tok, 1)
    return out


# --------------------------------------------------------------------------- #
# Checkpoint / resume (ghi nguyên tử, giữ 2 bản gần nhất, tự tiếp tục nếu cấu hình khớp)
# --------------------------------------------------------------------------- #
_CKPT_KEYS = ("epochs", "lr", "weight_decay", "warmup_ratio", "global_batch", "cutoff_len", "max_grad_norm",
              "lora_r", "lora_alpha", "seed", "lambda_kl", "uniform_weights", "train_file", "weights_dir",
              "teacher_adapter", "model_name", "segmentation_mode")


def _ckpt_signature(args) -> dict:
    return {k: getattr(args, k, None) for k in _CKPT_KEYS}


def _param_checksum(params) -> float:
    return float(sum(p.detach().double().abs().sum().item() for p in params))


def save_resume_ckpt(args, params, optimizer, scheduler, opt_step, micro, n_missing, elapsed_sec) -> None:
    d = os.path.join(args.out_dir, "resume_ckpt")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"step-{opt_step:05d}.pt")
    tmp = path + ".tmp"
    state = {
        "opt_step": opt_step, "micro": micro, "n_missing": n_missing, "elapsed_sec": elapsed_sec,
        "signature": _ckpt_signature(args), "checksum": _param_checksum(params),
        "params": [p.detach().cpu() for p in params],
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
    }
    torch.save(state, tmp)
    os.replace(tmp, path)                       # nguyên tử: file .pt chỉ xuất hiện khi đã ghi xong
    old = sorted(f for f in os.listdir(d) if f.startswith("step-") and f.endswith(".pt"))
    for f in old[:-2]:
        os.remove(os.path.join(d, f))


def load_resume_ckpt(args, params, optimizer, scheduler):
    """Trả về dict trạng thái nếu resume được, ngược lại None (chạy mới)."""
    d = os.path.join(args.out_dir, "resume_ckpt")
    if args.no_auto_resume or not os.path.isdir(d):
        return None
    files = sorted((f for f in os.listdir(d) if f.startswith("step-") and f.endswith(".pt")), reverse=True)
    for f in files:
        try:
            st = torch.load(os.path.join(d, f), map_location="cpu", weights_only=False)
        except Exception as exc:                # file hỏng (bị kill giữa chừng) -> thử bản cũ hơn
            print(f"[distill] checkpoint {f} không đọc được ({exc!r}) -> thử bản trước", flush=True)
            continue
        if st["signature"] != _ckpt_signature(args):
            diff = {k: (st["signature"].get(k), v) for k, v in _ckpt_signature(args).items() if st["signature"].get(k) != v}
            print(f"[distill] checkpoint {f} KHÁC cấu hình hiện tại {diff} -> KHÔNG resume, chạy mới", flush=True)
            return None
        if len(st["params"]) != len(params):
            print("[distill] checkpoint lệch số tham số -> KHÔNG resume, chạy mới", flush=True)
            return None
        with torch.no_grad():
            for p, saved in zip(params, st["params"]):
                p.copy_(saved.to(p.device, dtype=p.dtype))
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        chk = _param_checksum(params)
        print(f"[distill] RESUME từ {f}: opt_step={st['opt_step']} micro={st['micro']} "
              f"checksum lưu={st['checksum']:.6f} sau nạp={chk:.6f}", flush=True)
        return st
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    from core.proctitle import set_proc_title
    set_proc_title("distill")
    device = resolve_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.teacher_adapter or not os.path.isdir(args.teacher_adapter):
        raise FileNotFoundError(
            f"--teacher_adapter không hợp lệ: {args.teacher_adapter!r} "
            "(trỏ tới thư mục adapter phi_merge, ví dụ outputs/ours_s1k_rho_bal/merged_adapter/merged)"
        )

    tokenizer = load_tokenizer(args.model_name)
    base = load_base_model(
        args.model_name, gradient_checkpointing=args.gradient_checkpointing
    ).to(device)

    from peft import PeftModel

    # teacher (đóng băng) + student (trainable), CẢ HAI khởi tạo từ phi_merge.
    model = PeftModel.from_pretrained(
        base, args.teacher_adapter, adapter_name="teacher", is_trainable=False
    )
    model.load_adapter(args.teacher_adapter, adapter_name="student", is_trainable=True)
    set_lora_trainable(model, ["teacher"], False)
    n_train = set_lora_trainable(model, ["student"], True)
    model.set_adapter("student")
    vocab_size = base.config.vocab_size

    # Bắt buộc gradient checkpointing hoạt động trên model đã bọc PEFT — trace s1K
    # dài (mean ~9.9k, tối đa 16k token) nên không có checkpointing sẽ OOM.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        base.config.use_cache = False
    print(f"[distill] student init từ {args.teacher_adapter} | {n_train:,} tham số trainable")

    dataset = StepDataset(
        args.train_file, tokenizer,
        cutoff_len=args.cutoff_len,
        segmentation_mode=args.segmentation_mode,
        limit=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{args.train_file} rỗng")

    params = [p for _, p in lora_parameters(model, ["student"])]
    optimizer = build_optimizer(params, args.lr, args.weight_decay)
    total_steps = total_optimizer_steps(len(dataset), args.epochs, args.global_batch)
    scheduler = cosine_schedule_with_warmup(optimizer, total_steps, args.warmup_ratio)
    logger = JsonlLogger(args.out_dir)
    save_run_config(args.out_dir, args, {
        "tag": "phase2_distill", "n_examples": len(dataset),
        "total_optimizer_steps": total_steps, "n_trainable": n_train,
    })

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    accumulator = Accumulator()
    micro, opt_step, n_missing = 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    run_t0 = time.time()
    resumed = load_resume_ckpt(args, params, optimizer, scheduler)
    skip_micro = 0
    if resumed is not None:
        opt_step, skip_micro = int(resumed["opt_step"]), int(resumed["micro"])
        run_t0 = time.time() - float(resumed["elapsed_sec"])
        print(f"[distill] bỏ qua {skip_micro} lượt đã huấn luyện, tiếp tục từ bước {opt_step + 1}/{total_steps}", flush=True)
    step_t0 = time.time()

    for epoch in range(args.epochs):
        for index in dataset.iter_epoch(args.seed, epoch):
            example = dataset[index]
            segments = build_segments(example)
            if not segments:
                continue

            w_np = load_cached_w(args.weights_dir, example.example_id)
            if w_np is None and not args.uniform_weights and not args.skip_missing_ok:
                raise FileNotFoundError(f"Thiếu cache w cho {example.example_id}")
            weights = segment_weights(w_np, len(segments), args.uniform_weights)
            if weights is None:
                n_missing += 1
                continue
            if micro < skip_micro:                      # resume: mẫu này đã nằm trong các bước đã xong
                micro += 1
                continue
            weights = weights.to(device)

            input_ids = example.input_ids.unsqueeze(0).to(device)
            lengths = step_token_counts(segments)

            # 1) Teacher forward (no_grad, streaming) -> logits toàn từ vựng (chỉ cần khi lambda_kl > 0).
            if args.lambda_kl > 0:
                model.set_adapter("teacher")
                t_logits = teacher_logits_once(model, input_ids)
            else:
                t_logits = None

            # 2) Student forward (có grad).
            model.set_adapter("student")
            set_lora_trainable(model, ["student"], True)
            model.train()
            out = model(input_ids=input_ids, use_cache=False)
            s_logits = out.logits[0]  # (T, V)

            # 3) NLL có trọng số + KL có trọng số, cùng step-mean.
            s_lp = token_logprobs(s_logits, input_ids)
            nll_i = step_nll_from_logprobs(s_lp, segments)
            l_nll = weighted_step_loss(nll_i, lengths, weights, "step_mean")

            if args.lambda_kl > 0:
                kl_i = step_kl_full_vocab(t_logits, s_logits, segments, args.kl_chunk)
                l_kl = weighted_step_loss(kl_i, lengths, weights, "step_mean")
                loss = l_nll + args.lambda_kl * l_kl
            else:
                l_kl = s_logits.new_zeros(())
                loss = l_nll

            (loss / args.global_batch).backward()

            accumulator.add(
                loss=float(loss.detach()),
                l_nll=float(l_nll.detach()),
                l_kl=float(l_kl.detach()),
                mean_weight=float(weights.mean()),
                n_steps=len(segments),
                seq_tokens=int(input_ids.shape[1]),
            )
            del out, s_logits, s_lp, t_logits

            micro += 1
            if micro % args.global_batch:
                continue

            grad_norm = clip_and_step(params, optimizer, scheduler, args.max_grad_norm)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == 1:
                now = time.time()
                means = accumulator.mean()
                record = {
                    "tag": "phase2_distill", "epoch": epoch, "step": opt_step,
                    "total_steps": total_steps,
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "grad_norm": round(grad_norm, 4),
                    "sec_per_opt_step": round(now - step_t0, 2),
                    "elapsed_min": round((now - run_t0) / 60, 2),
                    "eta_min": round((now - run_t0) / 60 / opt_step * (total_steps - opt_step), 1),
                    "cuda_max_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if is_cuda else None,
                    **{k: round(means[k], 5) for k in
                       ("loss", "l_nll", "l_kl", "mean_weight", "n_steps", "seq_tokens") if k in means},
                }
                logger.log(record)
                print(f"[distill] step {opt_step}/{total_steps} "
                      f"loss={record.get('loss', float('nan')):.4f} "
                      f"nll={record.get('l_nll', float('nan')):.4f} "
                      f"kl={record.get('l_kl', float('nan')):.4f} "
                      f"w̄={record.get('mean_weight', 1.0):.3f} "
                      f"| {record['sec_per_opt_step']}s/step mem={record.get('cuda_max_gb')}GB "
                      f"eta={record['eta_min']}m", flush=True)
                step_t0 = now
            accumulator.reset()

            if args.ckpt_every and opt_step % args.ckpt_every == 0 and opt_step < total_steps:
                save_resume_ckpt(args, params, optimizer, scheduler, opt_step, micro, n_missing,
                                 time.time() - run_t0)
            if args.save_every and opt_step % args.save_every == 0:
                model.set_adapter("student")
                model.save_pretrained(os.path.join(args.out_dir, f"checkpoint-{opt_step}"),
                                      selected_adapters=["student"])
            if args.dry_run and opt_step >= args.dry_run:
                print(f"[distill] dry_run đạt {opt_step} step — dừng")
                break
        else:
            continue
        break

    model.set_adapter("student")
    model.save_pretrained(args.out_dir, selected_adapters=["student"])
    tokenizer.save_pretrained(args.out_dir)
    logger.close()
    if not args.dry_run:                            # chạy trọn -> student đã lưu, bỏ resume_ckpt
        import shutil
        shutil.rmtree(os.path.join(args.out_dir, "resume_ckpt"), ignore_errors=True)
    if n_missing:
        print(f"[distill] CẢNH BÁO: bỏ qua {n_missing} lượt vì thiếu / lệch cache w")
    print(f"[distill] xong -> {args.out_dir}/student  (dùng làm --adapter khi eval)")


if __name__ == "__main__":
    main()
