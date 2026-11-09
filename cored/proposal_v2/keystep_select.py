"""Bước B (tiền xử lý, chạy 1 lần, độc lập với Phase 1) — chọn "key step".

Ý tưởng: thay vì đẩy đa dạng (SVGD/DPP) đều trên MỌI reasoning step, chỉ tập trung
lực đẩy vào một tỉ lệ nhỏ step (mặc định 5-10%) có entropy cao nhất DƯỚI BASE MODEL
(chưa gắn adapter nào) — coi đây là các "điểm rẽ nhánh" ứng viên trong CoT, nơi hội
đồng expert có cơ hội thực sự bất đồng. Các step còn lại (phần lớn — tính toán chắc
chắn, số học, ...) giữ nguyên NLL thuần, không bị đẩy, tránh lãng phí ngân sách lực
đẩy và tránh làm hỏng các bước mà mọi expert lẽ ra nên đồng thuận.

Dùng entropy của BASE MODEL (một lần, trước khi train, không phụ thuộc các expert
đang được huấn luyện) để tránh vòng lặp tự tham chiếu: nếu dùng entropy của chính
expert đang train để chọn key step, ở đầu quá trình 3 expert còn gần giống hệt nhau
nên entropy của chúng chưa phản ánh đúng "điểm rẽ nhánh" thật.

Output: ``{out_dir}/{example_id}.npz`` chứa:
    H         (N,) float32 — entropy từng step dưới base model
    key_mask  (N,) bool    — True nếu step thuộc top ``frac`` entropy cao nhất
    frac      scalar       — tỉ lệ đã dùng để chọn (ghi lại để đối chiếu)

Cách dùng trong Phase 1: ``proposal_v2/phase1.py --keystep_dir <out_dir>`` (mặc định
None = tắt, giữ nguyên hành vi hiện tại — không set thì mọi config cũ chạy y hệt).

    python -m proposal_v2.keystep_select --train_file data/train_s1k_prefixed.jsonl \
        --out_dir outputs/proposal_v2/keystep_s1k_prefixed --frac 0.075
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from core.council import compute_base_step_entropy
from core.data_utils import StepDataset, set_seed, track_cli_provided
from core.model_utils import load_base_model, load_tokenizer
from train.common import add_common_args, resolve_device
from train.sft_engine import build_segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chọn key-step theo entropy base model (Bước B)")
    add_common_args(p)
    p.add_argument("--frac", type=float, default=0.075,
                   help="Tỉ lệ step chọn làm key step (mặc định 7.5%%, giữa 5-10%%)")
    p.add_argument("--min_key_steps", type=int, default=1,
                   help="Số key step tối thiểu mỗi mẫu (mẫu ngắn vẫn có ít nhất 1)")
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(out_dir="outputs/proposal_v2/keystep")
    return track_cli_provided(p)


def main() -> None:
    args = parse_args()
    from core.proctitle import set_proc_title
    set_proc_title("keystep_select")
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    base = load_base_model(args.model_name).to(device)
    base.eval()

    dataset = StepDataset(
        args.train_file, tokenizer, cutoff_len=args.cutoff_len,
        segmentation_mode=args.segmentation_mode, limit=args.max_samples,
    )

    n_written = 0
    all_frac_real = []
    for index in range(len(dataset)):
        example = dataset[index]
        spans = build_segments(example)
        if not spans:
            continue
        path = os.path.join(args.out_dir, f"{example.example_id}.npz")
        if os.path.exists(path) and not args.overwrite:
            n_written += 1
            continue

        input_ids = example.input_ids.unsqueeze(0).to(device)
        H = compute_base_step_entropy(base, input_ids, spans, chunk_size=args.chunk_size)
        H = H.detach().cpu().numpy().astype(np.float32)

        n_steps = len(spans)
        k = max(args.min_key_steps, int(round(args.frac * n_steps)))
        k = min(k, n_steps)
        order = np.argsort(-H)  # entropy giảm dần
        key_mask = np.zeros(n_steps, dtype=bool)
        key_mask[order[:k]] = True

        np.savez(path, H=H, key_mask=key_mask, frac=np.float32(args.frac))
        all_frac_real.append(k / n_steps)
        n_written += 1
        if n_written % 200 == 0:
            print(f"[keystep] {n_written}/{len(dataset)}", flush=True)

    print(f"[keystep] xong -> {args.out_dir} | {n_written} mẫu | "
          f"frac thực tế trung bình={np.mean(all_frac_real):.4f}" if all_frac_real else
          f"[keystep] xong -> {args.out_dir} | {n_written} mẫu (đã cache từ trước)")


if __name__ == "__main__":
    main()
