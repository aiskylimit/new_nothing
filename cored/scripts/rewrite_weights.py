"""Tính lại ``w`` trong cache trọng số từ (U, V) — KHÔNG cần GPU, KHÔNG train lại.

``train/compute_weights.py`` đã cache U/V/E/L/spans cho từng mẫu. Chỉ mảng ``w``
phụ thuộc hyperparameter weighting (lambda_U, lambda_D, disagreement_mode). Script
này đọc từng ``.npz``, tính lại ``w``, ghi sang thư mục mới, giữ nguyên mọi mảng
khác. Dùng để thử biến thể ``rho = V/U`` mà không phải chạy lại hội đồng.

    python scripts/rewrite_weights.py --weights_dir outputs/weights \\
        --out_dir outputs/weights_rho --disagreement_mode normalized
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weighting import corpus_stats, step_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute w in a weights cache")
    p.add_argument("--weights_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lambda_U", type=float, default=0.5)
    p.add_argument("--lambda_D", type=float, default=0.5)
    p.add_argument("--disagreement_mode", default="normalized",
                   choices=["raw", "normalized"])
    p.add_argument("--zscore_scope", default="example", choices=["example", "corpus"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.weights_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"Không thấy .npz nào trong {args.weights_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    payloads = []
    for path in paths:
        with np.load(path) as z:
            payloads.append((os.path.basename(path), {k: z[k] for k in z.files}))

    stats = None
    if args.zscore_scope == "corpus":
        # rho được z-score bằng "V_mean/V_std" trong step_weights; với corpus scope
        # ta phải tính stats CỦA đại lượng disagreement thực sự dùng (V hoặc rho).
        def disagreement(d):
            U = torch.from_numpy(np.asarray(d["U"], dtype=np.float32))
            V = torch.from_numpy(np.asarray(d["V"], dtype=np.float32))
            return V if args.disagreement_mode == "raw" else V / torch.clamp(U, min=1e-8)

        u_stat = corpus_stats([torch.from_numpy(np.asarray(d["U"], np.float32)) for _, d in payloads])
        d_stat = corpus_stats([disagreement(d) for _, d in payloads])
        stats = {"U_mean": u_stat["mean"], "U_std": u_stat["std"],
                 "V_mean": d_stat["mean"], "V_std": d_stat["std"]}

    n = 0
    for name, d in payloads:
        U = torch.from_numpy(np.asarray(d["U"], dtype=np.float32))
        V = torch.from_numpy(np.asarray(d["V"], dtype=np.float32))
        w = step_weights(U, V, args.lambda_U, args.lambda_D, stats=stats,
                         disagreement_mode=args.disagreement_mode)
        d["w"] = w.float().cpu().numpy()
        np.savez_compressed(os.path.join(args.out_dir, name), **d)
        n += 1

    print(f"[rewrite] {n} mẫu | mode={args.disagreement_mode} "
          f"lU={args.lambda_U} lD={args.lambda_D} scope={args.zscore_scope} -> {args.out_dir}")


if __name__ == "__main__":
    main()
