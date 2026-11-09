"""§7.2 — Tính tín hiệu hội đồng (U, V), suy ra w_i, và cache ra đĩa.

Chạy sau Phase 1. Base + M expert đều đóng băng nên U/V/w là hằng số trong
suốt Phase 2 → tính một lần, cache lại.

Mỗi mẫu ghi ``outputs/weights/{example_id}.npz`` gồm:
    U (N,), V (N,), w (N,), E (M, N), L (M, N)*, spans (N, 2)
(*) ``L[m, i]`` = NLL của expert m trên step i, cũng là hằng số vì expert đóng
băng — cache luôn để Phase 2 không phải chạy lại M forward mỗi epoch.

Đây cũng là **cổng chặn của §13 bước 6**: nếu V_i ≈ 0 ở hầu hết step thì các
expert không phân hoá và Phase 2 vô nghĩa. Script in verdict PASS/FAIL và
``--fail_on_low_v`` sẽ exit code 1 để pipeline dừng lại.

    python -m train.compute_weights --phase1_dir outputs/phase1 --M 3
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

from core.council import compute_council_signals
from core.data_utils import StepDataset, set_seed, track_cli_provided, write_json
from core.losses import step_nll
from core.model_utils import expert_names, load_base_model, load_experts, load_tokenizer
from core.weighting import step_weights
from train.common import add_common_args, resolve_device
from train.sft_engine import build_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute council signals & step weights")
    add_common_args(parser)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--phase1_dir", default="outputs/phase1")
    parser.add_argument("--weights_dir", default="outputs/weights")
    parser.add_argument("--lambda_U", type=float, default=0.5)
    parser.add_argument("--lambda_D", type=float, default=0.5)
    parser.add_argument("--disagreement_mode", default="raw", choices=["raw", "normalized"],
                        help="raw = V (proposal eq. 16), normalized = rho = V / U")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--zscore_scope", default="example", choices=["example", "corpus"])
    parser.add_argument("--cache_expert_losses", action="store_true", default=True)
    parser.add_argument("--no_cache_expert_losses", dest="cache_expert_losses",
                        action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--v_zero_threshold", type=float, default=1e-3,
                        help="V_i dưới ngưỡng này coi như ~0")
    parser.add_argument("--v_zero_max_frac", type=float, default=0.9,
                        help="Tỉ lệ step V≈0 tối đa cho phép trước khi FAIL")
    parser.add_argument("--fail_on_low_v", action="store_true",
                        help="Exit code 1 nếu cổng chặn V FAIL (§13 bước 6)")
    parser.add_argument("--preview", type=int, default=3,
                        help="In vài mẫu đã weight để mắt thường kiểm tra")
    parser.set_defaults(out_dir="outputs/weights")
    return track_cli_provided(parser)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard_id/num_shards")
    if args.num_shards > 1 and args.zscore_scope != "example":
        raise ValueError("Sharded weights require per-example normalization; corpus stats need a global pass")
    from core.proctitle import set_proc_title
    set_proc_title("compute_weights")
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.weights_dir, exist_ok=True)

    names = expert_names(args.M)
    expert_dirs = [os.path.join(args.phase1_dir, name) for name in names]
    missing = [d for d in expert_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"Thiếu expert dir: {missing} — chạy phase1 trước")

    tokenizer = load_tokenizer(args.model_name)
    base = load_base_model(args.model_name).to(device)
    model, adapter_names = load_experts(base, expert_dirs, names)
    model.eval()

    dataset = StepDataset(
        args.train_file,
        tokenizer,
        cutoff_len=args.cutoff_len,
        segmentation_mode=args.segmentation_mode,
        limit=args.max_samples,
    )

    all_U: List[np.ndarray] = []
    all_V: List[np.ndarray] = []
    all_w: List[np.ndarray] = []
    written_paths: List[str] = []
    n_written = 0

    for index in range(args.shard_id, len(dataset), args.num_shards):
        example = dataset[index]
        spans = example.step_spans
        if not spans:
            continue
        path = os.path.join(args.weights_dir, f"{example.example_id}.npz")
        if os.path.exists(path) and not args.overwrite:
            with np.load(path) as cached:
                all_U.append(cached["U"])
                all_V.append(cached["V"])
                all_w.append(cached["w"])
            written_paths.append(path)
            continue

        input_ids = example.input_ids.unsqueeze(0).to(device)
        signals = compute_council_signals(
            model, adapter_names, input_ids, spans, chunk_size=args.chunk_size
        )
        weights = step_weights(
            signals.U,
            signals.V,
            args.lambda_U,
            args.lambda_D,
            disagreement_mode=args.disagreement_mode,
        )

        # L được cache trên TOÀN BỘ segment (step thinking + segment answer) để
        # Phase 2 dùng trực tiếp; U/V/w chỉ định nghĩa trên step thinking.
        segments = build_segments(example)
        payload: Dict[str, np.ndarray] = {
            "U": signals.U.float().cpu().numpy(),
            "V": signals.V.float().cpu().numpy(),
            "w": weights.float().cpu().numpy(),
            "E": signals.E.float().cpu().numpy(),
            "spans": np.asarray(segments, dtype=np.int64),
            "n_think_steps": np.asarray(len(spans), dtype=np.int64),
        }
        if args.cache_expert_losses:
            payload["L"] = _expert_step_losses(model, adapter_names, input_ids, segments)

        np.savez_compressed(path, **payload)
        all_U.append(payload["U"])
        all_V.append(payload["V"])
        all_w.append(payload["w"])
        written_paths.append(path)
        n_written += 1

        if n_written <= args.preview:
            _preview(example, payload)
        if (index + 1) % 200 == 0:
            print(f"[weights] {index + 1}/{len(dataset)}")

    corpus_stats = None
    if args.zscore_scope == "corpus":
        corpus_stats = _corpus_stats(all_U, all_V)
        all_w = _rewrite_weights_with_corpus_stats(args, written_paths, corpus_stats)

    summary = _summarize(args, all_U, all_V, all_w)
    summary["zscore_scope"] = args.zscore_scope
    summary["corpus_stats"] = corpus_stats
    summary_name = "summary.json" if args.num_shards == 1 else f"summary-shard-{args.shard_id:03d}.json"
    write_json(os.path.join(args.weights_dir, summary_name), summary)
    _print_gate(summary)

    if args.fail_on_low_v and not summary["gate"]["pass"]:
        print("[weights] CỔNG CHẶN FAIL — tăng lambda_rep / p_drop / M rồi train lại Phase 1.")
        sys.exit(1)


@torch.no_grad()
def _expert_step_losses(model, adapter_names, input_ids, spans) -> np.ndarray:
    rows = []
    for name in adapter_names:
        model.set_adapter(name)
        out = model(input_ids=input_ids, use_cache=False)
        rows.append(step_nll(out.logits, input_ids, spans).float().cpu())
        del out
    return torch.stack(rows).numpy()


def _corpus_stats(all_U: List[np.ndarray], all_V: List[np.ndarray]) -> dict:
    """Mean/std của U và V gộp qua toàn bộ tập train."""
    U = np.concatenate([np.asarray(a).reshape(-1) for a in all_U]) if all_U else np.zeros(1)
    V = np.concatenate([np.asarray(a).reshape(-1) for a in all_V]) if all_V else np.zeros(1)
    return {
        "U_mean": float(U.mean()), "U_std": float(U.std()),
        "V_mean": float(V.mean()), "V_std": float(V.std()),
    }


def _rewrite_weights_with_corpus_stats(
    args, paths: List[str], stats: dict
) -> List[np.ndarray]:
    """Pass 2: tính lại w bằng z-score toàn corpus rồi ghi đè vào từng npz.

    U/V/E/L không đổi (chỉ phụ thuộc model), nên chỉ mảng ``w`` được thay.
    """
    new_w: List[np.ndarray] = []
    for path in paths:
        with np.load(path) as cached:
            payload = {key: cached[key] for key in cached.files}
        weights = step_weights(
            torch.from_numpy(payload["U"]),
            torch.from_numpy(payload["V"]),
            args.lambda_U,
            args.lambda_D,
            stats=stats,
        )
        payload["w"] = weights.float().cpu().numpy()
        np.savez_compressed(path, **payload)
        new_w.append(payload["w"])
    print(f"[weights] z-score corpus: cập nhật w cho {len(paths)} mẫu "
          f"(U_mean={stats['U_mean']:.4f} U_std={stats['U_std']:.4f} "
          f"V_mean={stats['V_mean']:.4f} V_std={stats['V_std']:.4f})")
    return new_w


def _preview(example, payload) -> None:
    steps = example.raw.get("steps", [])
    order = np.argsort(-payload["w"])
    print("\n" + "=" * 70)
    print(f"[weights] mẫu {example.example_id}: N={len(payload['w'])}")
    for rank in list(order[:3]) + list(order[-2:]):
        text = steps[rank][:110].replace("\n", " ") if rank < len(steps) else ""
        print(f"  w={payload['w'][rank]:.3f} U={payload['U'][rank]:.3f} "
              f"V={payload['V'][rank]:.4f} | {text}")


def _summarize(args, all_U, all_V, all_w) -> dict:
    def describe(arrays) -> dict:
        if not arrays:
            return {"count": 0}
        flat = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
        return {
            "count": int(flat.size),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "p10": float(np.percentile(flat, 10)),
            "median": float(np.median(flat)),
            "p90": float(np.percentile(flat, 90)),
            "max": float(flat.max()),
        }

    V_flat = (
        np.concatenate([np.asarray(a).reshape(-1) for a in all_V]) if all_V else np.zeros(0)
    )
    frac_zero = float((V_flat < args.v_zero_threshold).mean()) if V_flat.size else 1.0
    passed = bool(V_flat.size and frac_zero <= args.v_zero_max_frac)

    return {
        "n_examples": len(all_w),
        "lambda_U": args.lambda_U,
        "lambda_D": args.lambda_D,
        "M": args.M,
        "U": describe(all_U),
        "V": describe(all_V),
        "w": describe(all_w),
        "gate": {
            "name": "V_i diversity gate (spec §13 buoc 6)",
            "v_zero_threshold": args.v_zero_threshold,
            "frac_steps_with_V_near_zero": frac_zero,
            "max_allowed_frac": args.v_zero_max_frac,
            "pass": passed,
        },
    }


def _print_gate(summary: dict) -> None:
    gate = summary["gate"]
    print("\n" + "=" * 70)
    print(f"[weights] U: mean={summary['U'].get('mean')} std={summary['U'].get('std')}")
    print(f"[weights] V: mean={summary['V'].get('mean')} p90={summary['V'].get('p90')}")
    print(f"[weights] w: min={summary['w'].get('min')} max={summary['w'].get('max')}")
    verdict = "PASS" if gate["pass"] else "FAIL"
    print(f"[weights] CỔNG CHẶN V_i: {verdict} "
          f"({gate['frac_steps_with_V_near_zero']:.1%} step có V≈0, "
          f"cho phép ≤ {gate['max_allowed_frac']:.0%})")


if __name__ == "__main__":
    main()
