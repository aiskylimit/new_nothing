"""Phase 1 — Diversity Training (bản proposal, standard autograd).

Eq (17):  L = Σ_i Σ_m L_i^(m)  +  λ_rep · L_ortho(Φ)  +  λ_div · L_DPP_div

Khác train/phase1_diversity.py: dùng DPP + Kneedle (proposal/dpp.py, §4.3) thay
JS; không dùng analytical-dZ, chỉ autograd chuẩn (chậm hơn, đơn giản & đúng).

    python -m proposal.phase1 --config configs/proposal/phase1.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from core.data_utils import JsonlLogger, StepDataset, set_seed, track_cli_provided, write_json
from core.losses import step_nll_from_logprobs, step_token_counts, token_logprobs


def _load_keystep_mask(keystep_dir: str, example_id: str, n_spans: int, cache: dict):
    """Đọc key_mask đã cache (proposal_v2.keystep_select), cache trong bộ nhớ.

    Trả None nếu không có cache hoặc độ dài không khớp (an toàn: tắt filter thay vì lỗi).
    """
    if example_id in cache:
        return cache[example_id]
    path = os.path.join(keystep_dir, f"{example_id}.npz")
    mask = None
    if os.path.exists(path):
        with np.load(path) as d:
            m = d["key_mask"]
            if len(m) == n_spans:
                mask = m
    cache[example_id] = mask
    return mask


def _step_nll_frugal(logits: torch.Tensor, input_ids: torch.Tensor, spans, chunk: int = 1024):
    """NLL/step không upcast toàn bộ (T,V) sang fp32 (tiết kiệm ~5GB @ T=8k).

    cross_entropy CUDA kernel nhận bf16 (accumulate fp32 nội bộ). Chỉ tính ở các
    vị trí dự đoán thực sự nằm trong step spans -> ít token hơn cả T.
    """
    T = logits.size(0)
    out = logits.new_zeros(len(spans))
    for i, (s, e) in enumerate(spans):
        s = max(int(s), 1)
        e = min(int(e), T)
        if e <= s:
            continue
        tot = 0.0
        n = 0
        for c in range(s - 1, e - 1, chunk):
            ce = min(c + chunk, e - 1)
            lp = -F.cross_entropy(logits[c:ce], input_ids[c + 1:ce + 1], reduction="sum")
            tot = tot + lp
            n += ce - c
        out[i] = -(tot / max(n, 1))
    return out
from core.model_utils import (
    attach_experts,
    expert_names,
    load_base_model,
    load_tokenizer,
    lora_parameters,
    save_adapter,
    set_lora_trainable,
)
from core.rep_loss import lora_diversity_loss, report_pairwise_similarity
from proposal.dpp import dpp_diversity_loss
from proposal_v2.svgd import apply_svgd_param, svgd_func_repulsion
from train.common import (
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
    p = argparse.ArgumentParser(description="Proposal Phase 1 — DPP diversity training")
    add_common_args(p)
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--p_drop", type=float, default=0.1, help="Eq 9: Bernoulli(1-p_drop) theo step/expert")
    p.add_argument("--lambda_rep", type=float, default=0.01, help="Eq 17: hệ số L_ortho")
    p.add_argument("--lambda_div", type=float, default=0.1, help="Eq 17: hệ số L_DPP_div")
    p.add_argument("--dpp_alpha", type=float, default=1e-5, help="Eq 15: α·I regularizer")
    p.add_argument("--dpp_max_cand", type=int, default=64, help="chặn |C| tối đa mỗi token")
    p.add_argument("--dpp_max_tokens", type=int, default=256,
                   help="Số vị trí token lấy mẫu / mẫu để tính DPP (0 = tất cả). Giữ nhỏ vì slogdet vòng lặp.")
    p.add_argument("--rep_every", type=int, default=1)
    p.add_argument("--warmup_div_epochs", type=int, default=1,
                   help="Số epoch đầu tắt hẳn diversity (expert học task trước) — chống sập expert.")
    p.add_argument("--dpp_cap_frac", type=float, default=0.3,
                   help="Chặn dpp_term <= frac * NLL_term (detach). 0 = không chặn. (chỉ diversity=dpp)")
    p.add_argument("--diversity", choices=["dpp", "svgd_param", "svgd_func", "none"], default="dpp",
                   help="dpp: L_DPP_div (loss). svgd_param: repulsion RBF trên tham số (hook optimizer, 0 forward). "
                        "svgd_func: repulsion RBF trên dự đoán (loss nhẹ, bỏ slogdet).")
    p.add_argument("--tr_off_k", type=float, default=1.0, help="hệ số lực đẩy SVGD")
    p.add_argument("--svgd_sigma", type=float, default=0.0, help="0 = median heuristic")
    p.add_argument("--svgd_grad_mix", type=int, default=1, help="svgd_param: trộn grad theo kernel (chống collapse)")
    p.add_argument("--keystep_dir", default=None,
                   help="Thư mục cache key-step (proposal_v2.keystep_select). Mặc định None = tắt, "
                        "giữ nguyên hành vi cũ (đẩy đa dạng trên MỌI step). Khi set, chỉ token thuộc "
                        "key step (top-frac entropy base model) được đưa vào pred_pos cho dpp/svgd_func "
                        "— NLL vẫn tính trên toàn bộ step như cũ, không đổi.")
    p.set_defaults(out_dir="outputs/proposal_v2/phase1", lr=5e-5, epochs=3, global_batch=32,
                   cutoff_len=16384, warmup_ratio=0.1, segmentation_mode="paragraph")
    return track_cli_provided(p)


def main() -> None:
    args = parse_args()
    from core.proctitle import set_proc_title
    set_proc_title("phase1")
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    from core.model_utils import make_lora_config

    base = load_base_model(args.model_name, gradient_checkpointing=args.gradient_checkpointing).to(device)
    lora_cfg = make_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model, adapter_names = attach_experts(base, args.M, lora_cfg)
    model.to(device)
    if args.gradient_checkpointing:  # PEFT có thể tắt GC khi wrap -> bật lại
        # use_reentrant=False bắt buộc với PEFT: reentrant GC không checkpoint đúng khi
        # có LoRA param -> lưu full activation -> OOM ở T lớn (giống train/phase2_distill.py).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        base.config.use_cache = False
        if hasattr(model, "config"):
            model.config.use_cache = False

    dataset = StepDataset(args.train_file, tokenizer, cutoff_len=args.cutoff_len,
                          segmentation_mode=args.segmentation_mode, limit=args.max_samples)

    # BẮT BUỘC trước build_optimizer: attach_experts() chỉ set requires_grad=True cho
    # adapter mặc định (expert_0, tạo qua get_peft_model); các adapter thêm sau qua
    # add_adapter() mặc định requires_grad=False. build_optimizer() lọc params theo
    # requires_grad NGAY TẠI THỜI ĐIỂM GỌI -> nếu gọi set_lora_trainable sau (như cũ,
    # bên trong vòng lặp train), optimizer vĩnh viễn không biết tới expert_1..M-1: gradient
    # vẫn chảy vào .grad nhưng optimizer.step() không bao giờ cập nhật chúng (B giữ nguyên
    # 0 suốt quá trình train). Đây là lý do mọi expert trừ expert_0 không hề được train.
    set_lora_trainable(model, adapter_names, True)

    params = [p for _, p in lora_parameters(model, adapter_names)]
    optimizer = build_optimizer(params, args.lr, args.weight_decay)
    total_steps = total_optimizer_steps(len(dataset), args.epochs, args.global_batch)
    scheduler = cosine_schedule_with_warmup(optimizer, total_steps, args.warmup_ratio)
    logger = JsonlLogger(args.out_dir)
    save_run_config(args.out_dir, args, {"tag": "proposal_phase1", "n_examples": len(dataset),
                                         "total_optimizer_steps": total_steps})

    model.train()  # BẮT BUỘC: GradientCheckpointingLayer chỉ checkpoint khi self.training=True.
    # Thiếu dòng này -> GC im lặng tắt -> full activation -> OOM ở T lớn (bug bản phase1 gốc,
    # là lý do bản gốc phải hạ cutoff 16384 -> 8192). Có train(): T=16384 chỉ ~54GB/expert.

    drop_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    micro = opt_step = 0
    optimizer.zero_grad(set_to_none=True)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    run_t0 = step_t0 = time.time()
    accum = {"nll": 0.0, "ortho": 0.0, "dpp": 0.0, "frac_valid": 0.0, "n": 0}
    keystep_cache: dict = {}

    for epoch in range(args.epochs):
        for index in dataset.iter_epoch(args.seed, epoch) if hasattr(dataset, "iter_epoch") else range(len(dataset)):
            example = dataset[index]
            spans = build_segments(example)
            if not spans:
                continue
            input_ids = example.input_ids.unsqueeze(0).to(device)
            T_seq = input_ids.shape[1]
            lengths = step_token_counts(spans)

            # Eq 9: d_{i,m} ~ Bernoulli(1 - p_drop)
            keep = torch.bernoulli(torch.full((args.M, len(spans)), 1.0 - args.p_drop),
                                   generator=drop_gen)

            set_lora_trainable(model, adapter_names, True)

            # Vị trí token dự đoán (Eq: logits[t] -> input_ids[t+1]); lấy mẫu cho DPP.
            # keystep_dir (tùy chọn, mặc định None): chỉ đưa step thuộc top-frac entropy base
            # model (Bước B, proposal_v2.keystep_select) vào pred_pos -> lực đẩy chỉ tập trung ở
            # "điểm rẽ nhánh" thay vì trải đều mọi step. NLL bên dưới KHÔNG bị ảnh hưởng (vẫn
            # tính trên toàn bộ spans như cũ) — chỉ số hạng diversity bị thu hẹp phạm vi.
            key_mask = None
            if args.keystep_dir:
                key_mask = _load_keystep_mask(args.keystep_dir, example.example_id,
                                              len(spans), keystep_cache)
            pred_pos: List[int] = []
            for i, (s, e) in enumerate(spans):
                if key_mask is not None and not key_mask[i]:
                    continue
                s = max(s, 1); e = min(e, T_seq)
                if e > s:
                    pred_pos.extend(range(s - 1, e - 1))
            div_on = args.diversity != "none" and epoch >= args.warmup_div_epochs
            need_logp = (div_on and args.diversity in ("dpp", "svgd_func")
                         and len(pred_pos) >= 2)
            do_dpp = need_logp and args.diversity == "dpp"
            do_sfunc = need_logp and args.diversity == "svgd_func"
            if need_logp:
                pos = torch.tensor(pred_pos, device=device)
                if args.dpp_max_tokens and pos.numel() > args.dpp_max_tokens:
                    g = torch.Generator().manual_seed(args.seed * 1000003 + index)
                    sel, _ = torch.sort(torch.randperm(pos.numel(), generator=g)[: args.dpp_max_tokens])
                    pos = pos[sel]
                gold_dpp = input_ids[0, pos + 1]

            gb = args.global_batch

            # --- Pass 1 (no_grad): thu logp DETACH cho từng expert (dpp / svgd_func) ---
            logp_dpp_detached: List[torch.Tensor] = []
            if need_logp:
                model.eval()  # tắt dropout cho teacher logp (no_grad -> không tốn activation)
                with torch.no_grad():
                    for name in adapter_names:
                        model.set_adapter(name)
                        lg = model(input_ids=input_ids, use_cache=False).logits[0]
                        logp_dpp_detached.append(
                            F.log_softmax(lg.index_select(0, pos).float(), dim=-1).detach())
                        del lg
                model.train()  # BẮT lại train mode cho Pass 2 (GC cần self.training=True)

            # --- Eq 10-11: dual-ortho (chỉ đọc A/B, backward riêng, rất nhẹ) ---
            ortho_loss = lora_diversity_loss(model, adapter_names, metric="dual_ortho")
            if isinstance(ortho_loss, tuple):
                ortho_loss = ortho_loss[0]
            if torch.is_tensor(ortho_loss) and ortho_loss.requires_grad:
                (args.lambda_rep * ortho_loss / gb).backward()
            ortho_val = float(ortho_loss.detach()) if torch.is_tensor(ortho_loss) else float(ortho_loss)

            # --- Pass 2: MỖI expert 1 graph tại 1 thời điểm (NLL + phần DPP của nó) ---
            nll_sum = 0.0
            dpp_val = 0.0
            frac_valid = 0.0
            for m, name in enumerate(adapter_names):
                model.set_adapter(name)
                lg = model(input_ids=input_ids, use_cache=False).logits[0]   # (T, V) bf16
                nll_i = _step_nll_frugal(lg, input_ids[0], spans)
                w_keep = keep[m].to(device)
                nll_term = (w_keep * nll_i).sum() / w_keep.sum().clamp_min(1.0)
                loss_m = nll_term

                if do_dpp:
                    # danh sách: chỉ expert m có grad, còn lại detach -> partial ∂/∂ṽ_m đúng
                    lst = [x for x in logp_dpp_detached]
                    lst[m] = F.log_softmax(lg.index_select(0, pos).float(), dim=-1)
                    dpp_m, dstats = dpp_diversity_loss(
                        lst, gold_dpp, [int(pos.numel())],
                        alpha=args.dpp_alpha, max_cand=args.dpp_max_cand)
                    dpp_term = args.lambda_div * dpp_m
                    if args.dpp_cap_frac > 0:  # chặn dpp_term <= cap*NLL -> không nuốt gradient
                        cap = args.dpp_cap_frac * float(nll_term.detach())
                        dt = float(dpp_term.detach())
                        if dt > cap > 0:
                            dpp_term = dpp_term * (cap / dt)
                    loss_m = loss_m + dpp_term
                    dpp_val = float(dpp_m.detach())  # log logdet thô (so sánh được với FINDINGS)
                    frac_valid = dstats.frac_valid_tokens

                elif do_sfunc:
                    lst = [x for x in logp_dpp_detached]
                    lst[m] = F.log_softmax(lg.index_select(0, pos).float(), dim=-1)
                    rep_loss, sst = svgd_func_repulsion(
                        lst, m, args.tr_off_k, args.svgd_sigma or None)
                    loss_m = loss_m + rep_loss
                    dpp_val = sst["svgd_func_K"]  # dùng chung ô log: K trung bình (0..1)

                (loss_m / gb).backward()
                nll_sum += float(nll_term.detach())
                del lg, nll_i, nll_term, loss_m

            accum["nll"] += nll_sum
            accum["ortho"] += ortho_val
            accum["dpp"] += dpp_val
            accum["frac_valid"] += frac_valid
            accum["n"] += 1

            del logp_dpp_detached
            if do_dpp:
                del gold_dpp
            micro += 1
            if is_cuda and micro % 4 == 0:
                torch.cuda.empty_cache()
            if micro % args.global_batch:
                continue

            # --- SVGD param-space: repulsion RBF thẳng vào .grad tích luỹ ---
            svgd_stats = {}
            if args.diversity == "svgd_param" and div_on:
                svgd_stats = apply_svgd_param(
                    model, adapter_names, args.tr_off_k,
                    args.svgd_sigma or None, bool(args.svgd_grad_mix))

            gnorm = clip_and_step(params, optimizer, scheduler, args.max_grad_norm)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == 1:
                n = max(accum["n"], 1)
                now = time.time()
                rec = {"tag": "proposal_phase1", "epoch": epoch, "step": opt_step,
                       "total_steps": total_steps, "grad_norm": round(gnorm, 4),
                       "lr": round(scheduler.get_last_lr()[0], 8),
                       "nll": round(accum["nll"] / n, 5),
                       "ortho": round(accum["ortho"] / n, 6),
                       "dpp_logdet": round(accum["dpp"] / n, 5),
                       "dpp_frac_valid": round(accum["frac_valid"] / n, 3),
                       "cuda_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if is_cuda else None,
                       "sec_per_step": round(now - step_t0, 1),
                       "diversity": args.diversity,
                       "eta_min": round((now - run_t0) / 60 / opt_step * (total_steps - opt_step), 1)}
                if svgd_stats:
                    rec.update({k: round(v, 5) for k, v in svgd_stats.items()})
                logger.log(rec)
                _extra = (f" K={svgd_stats.get('svgd_K_offdiag'):.4f} d={svgd_stats.get('svgd_dist'):.3f}"
                          if svgd_stats else "")
                print(f"[p1] step {opt_step}/{total_steps} nll={rec['nll']} ortho={rec['ortho']} "
                      f"div({args.diversity})={rec['dpp_logdet']} valid={rec['dpp_frac_valid']}{_extra} "
                      f"mem={rec['cuda_gb']}GB | {rec['sec_per_step']}s eta={rec['eta_min']}m", flush=True)
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats()
                step_t0 = now
            accum = {"nll": 0.0, "ortho": 0.0, "dpp": 0.0, "frac_valid": 0.0, "n": 0}

        model.save_pretrained(args.out_dir, selected_adapters=list(adapter_names))

    sim = report_pairwise_similarity(model, adapter_names)
    write_json(os.path.join(args.out_dir, "expert_similarity.json"), sim)
    model.save_pretrained(args.out_dir, selected_adapters=list(adapter_names))
    tokenizer.save_pretrained(args.out_dir)
    logger.close()
    print(f"[p1] xong -> {args.out_dir} | similarity={sim}", flush=True)


if __name__ == "__main__":
    main()
