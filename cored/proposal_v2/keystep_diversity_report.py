"""Bước C (post Phase 1) — so sánh tín hiệu hội đồng (U, V) trên key step vs step
thường, dùng CHÍNH các expert đã train (khác keystep_select.py: keystep_select
dùng entropy BASE MODEL để CHỌN key step trước khi train; script này dùng U/V
của 3 expert SAU KHI train để KIỂM TRA xem lực đẩy SVGD có thực sự tạo ra bất
đồng ở đúng những vị trí đã chọn hay không).

U_i = entropy trung bình qua M expert tại step i (core/council.py).
V_i = disagreement (Jensen-Shannon) giữa M expert tại step i.

Nếu SVGD (svgd_func, chỉ áp trên key step) hoạt động đúng: V trên key step
phải cao hơn rõ rệt so với V trên step thường (nơi không có lực đẩy).

Output: {out_dir}/keystep_diversity_summary.json + in ra bảng so sánh
key-step vs non-key-step (U, V, frac V≈0) — cùng định dạng cổng chặn với
train/compute_weights.py (§13 bước 6) để dễ đối chiếu.

    python -m proposal_v2.keystep_diversity_report \
        --phase1_dir outputs/proposal_v2/phase1_svgd_func_keystep_k1 \
        --keystep_dir outputs/proposal_v2/keystep_s1k_prefixed_10pct \
        --train_file data/train_s1k_prefixed.jsonl --device cuda
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from core.council import compute_council_signals
from core.data_utils import StepDataset, set_seed, write_json
from core.model_utils import expert_names, load_base_model, load_experts, load_tokenizer
from train.common import resolve_device
from train.sft_engine import build_segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="So sánh U/V hội đồng trên key step vs step thường")
    p.add_argument("--model_name", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--phase1_dir", required=True, help="Thư mục chứa expert_0..M-1 đã train")
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--keystep_dir", required=True, help="Cache key_mask (proposal_v2.keystep_select)")
    p.add_argument("--train_file", default="data/train_s1k_prefixed.jsonl")
    p.add_argument("--cutoff_len", type=int, default=16384)
    p.add_argument("--segmentation_mode", default="paragraph", choices=["paragraph", "sentence"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--v_zero_threshold", type=float, default=1e-3)
    p.add_argument("--out_dir", default=None, help="Mặc định = --phase1_dir")
    p.add_argument("--preview", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = args.out_dir or args.phase1_dir
    os.makedirs(out_dir, exist_ok=True)

    names = expert_names(args.M)
    expert_dirs = [os.path.join(args.phase1_dir, n) for n in names]
    missing = [d for d in expert_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"Thiếu expert dir: {missing} — chạy phase1 trước")

    tokenizer = load_tokenizer(args.model_name)
    base = load_base_model(args.model_name).to(device)
    model, adapter_names = load_experts(base, expert_dirs, names)
    model.eval()

    dataset = StepDataset(
        args.train_file, tokenizer, cutoff_len=args.cutoff_len,
        segmentation_mode=args.segmentation_mode, limit=args.max_samples,
    )

    key_U: List[np.ndarray] = []
    key_V: List[np.ndarray] = []
    other_U: List[np.ndarray] = []
    other_V: List[np.ndarray] = []
    n_used = n_skipped_no_mask = n_skipped_len_mismatch = 0

    for index in range(len(dataset)):
        example = dataset[index]
        spans = build_segments(example)  # PHẢI khớp keystep_select.py (bao gồm answer span)
        if not spans:
            continue

        mask_path = os.path.join(args.keystep_dir, f"{example.example_id}.npz")
        if not os.path.exists(mask_path):
            n_skipped_no_mask += 1
            continue
        with np.load(mask_path) as d:
            key_mask = d["key_mask"]
        if len(key_mask) != len(spans):
            n_skipped_len_mismatch += 1
            continue

        input_ids = example.input_ids.unsqueeze(0).to(device)
        signals = compute_council_signals(
            model, adapter_names, input_ids, spans, chunk_size=args.chunk_size
        )
        U = signals.U.float().cpu().numpy()
        V = signals.V.float().cpu().numpy()

        key_U.append(U[key_mask]); key_V.append(V[key_mask])
        other_U.append(U[~key_mask]); other_V.append(V[~key_mask])
        n_used += 1

        if n_used <= args.preview:
            _preview(example, U, V, key_mask)
        if n_used % 200 == 0:
            print(f"[keystep_div] {n_used} mẫu đã xử lý", flush=True)

    def describe(arrays: List[np.ndarray]) -> dict:
        flat = np.concatenate(arrays) if arrays else np.zeros(0)
        if flat.size == 0:
            return {"count": 0}
        return {
            "count": int(flat.size),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "median": float(np.median(flat)),
            "p10": float(np.percentile(flat, 10)),
            "p90": float(np.percentile(flat, 90)),
            "frac_near_zero": float((flat < args.v_zero_threshold).mean()),
        }

    summary = {
        "phase1_dir": args.phase1_dir,
        "keystep_dir": args.keystep_dir,
        "n_examples_used": n_used,
        "n_skipped_no_mask": n_skipped_no_mask,
        "n_skipped_len_mismatch": n_skipped_len_mismatch,
        "key_step": {"U": describe(key_U), "V": describe(key_V)},
        "non_key_step": {"U": describe(other_U), "V": describe(other_V)},
    }
    write_json(os.path.join(out_dir, "keystep_diversity_summary.json"), summary)
    _print_report(summary, args.v_zero_threshold)


def _preview(example, U, V, key_mask) -> None:
    order_key = np.where(key_mask)[0]
    print("\n" + "=" * 70)
    print(f"[keystep_div] mẫu {example.example_id}: N={len(U)} key={key_mask.sum()}")
    for i in order_key[:3]:
        print(f"  KEY   step {i}: U={U[i]:.3f} V={V[i]:.4f}")
    for i in np.where(~key_mask)[0][:2]:
        print(f"  other step {i}: U={U[i]:.3f} V={V[i]:.4f}")


def _print_report(summary: dict, thresh: float) -> None:
    k, o = summary["key_step"], summary["non_key_step"]
    print("\n" + "=" * 70)
    print(f"[keystep_div] {summary['n_examples_used']} mẫu "
          f"(bỏ qua {summary['n_skipped_no_mask']} thiếu mask, "
          f"{summary['n_skipped_len_mismatch']} lệch độ dài)")
    print(f"[keystep_div] KEY STEP     : U mean={k['U'].get('mean'):.4f} | "
          f"V mean={k['V'].get('mean'):.4f} p90={k['V'].get('p90'):.4f} "
          f"frac(V<{thresh})={k['V'].get('frac_near_zero'):.1%}")
    print(f"[keystep_div] NON-KEY STEP : U mean={o['U'].get('mean'):.4f} | "
          f"V mean={o['V'].get('mean'):.4f} p90={o['V'].get('p90'):.4f} "
          f"frac(V<{thresh})={o['V'].get('frac_near_zero'):.1%}")
    if k["V"].get("count") and o["V"].get("mean"):
        ratio = k["V"]["mean"] / max(o["V"]["mean"], 1e-12)
        verdict = "PASS (key step đa dạng hơn rõ rệt)" if ratio > 1.5 else \
                  "YẾU (key step không tách biệt nhiều so với step thường)"
        print(f"[keystep_div] V_key / V_other = {ratio:.2f}x -> {verdict}")


if __name__ == "__main__":
    main()
