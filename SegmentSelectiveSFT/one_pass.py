"""Pipeline mot luot: attribution -> LoRA SFT tren 3 GPU -> vLLM eval.

Offline: khong day gi len mang. Checkpoint ghi tren dia local,
{output_dir}/{run_id}/checkpoint-final, de vLLM doc. Eval loi thi weight van con.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["REPORT_TO"] = "none"

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SelectiveSFT"))
sys.path.insert(0, str(ROOT / "Attribution"))
sys.path.insert(0, str(ROOT / "Eval"))



def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cuda_peak_gb():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def empty_cuda():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def n_params_of(model):
    return sum(p.numel() for p in model.parameters())


def compact_ig(scores):
    import numpy as np
    out = []
    for seg in scores:
        n_tok = len(seg)
        if n_tok == 0:
            out.append((0, 0.0, 0.0))
        else:
            arr = np.asarray(seg, dtype=np.float64)
            out.append((n_tok, float(np.sum(np.abs(arr))), float(np.sum(arr))))
    return out


def stage_prep(args, records):
    """Stage 1. Vanilla SFT: prep = 0. Selective: IG roi chon segment, khong ghi file IG."""
    from segment_utils import split_segments

    for rec in records:
        if "segments" not in rec or not rec["segments"]:
            rec["segments"] = split_segments(rec["solution"], args.segment_mode)
        rec.setdefault("selected_spans_ids", [])

    if args.full_sft:
        return {
            "prep_wall_time_sec": 0.0,
            "prep_peak_vram_gb": 0.0,
            "prep_flops": 0.0,
            "prep_n_params": 0,
            "tokens_forwarded": 0,
            "tokens_backwarded": 0,
        }

    import torch
    from get_important_segments import select_important_segments
    from grad_analyze import IntegratedGradientsAttribution, attribute_example

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    calculator = IntegratedGradientsAttribution(
        args.model, gradient_checkpointing=not args.no_ig_grad_checkpoint
    )
    prep_n = n_params_of(calculator.model)
    ig_rows = []
    try:
        for i, rec in enumerate(records):
            scores, _status = attribute_example(
                calculator, rec,
                ig_steps=args.ig_steps,
                ig_batch_size=args.ig_batch_size,
                max_input_tokens=args.ig_max_tokens,
                sample_index=i,
            )
            if getattr(args, 'require_valid_attribution', False) and _status != 'ok':
                raise RuntimeError(f'Attribution failed for sample {rec.get("id", i)}: {_status}; smoke cannot pass with zero fallback scores')
            ig_rows.append(compact_ig(scores))
            del scores
        select_important_segments(
            records, ig_rows, args.cumulative_ratio, args.coherence_max
        )
    finally:
        tokens_f = int(calculator.tokens_forwarded)
        tokens_b = int(calculator.tokens_backwarded)
        peak = cuda_peak_gb()
        del calculator
        empty_cuda()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    prep_flops = 2.0 * prep_n * tokens_f + 6.0 * prep_n * tokens_b
    return {
        "prep_wall_time_sec": wall,
        "prep_peak_vram_gb": peak,
        "prep_flops": prep_flops,
        "prep_n_params": prep_n,
        "tokens_forwarded": tokens_f,
        "tokens_backwarded": tokens_b,
    }


def stage_train(args, records):
    """Stage 2. LoRA SFT mac dinh, torchrun rieng cho ba GPU."""

    fd, tmp_name = tempfile.mkstemp(prefix="one_pass_train_", suffix=".jsonl")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                row = {
                    "question": rec["question"],
                    "solution": rec["solution"],
                    "answer": rec.get("answer", ""),
                    "selected_spans_ids": list(rec.get("selected_spans_ids") or []),
                    "segments": list(rec.get("segments") or []),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        argv = [
            "--model_name_or_path", args.model,
            "--data_names", str(tmp_path),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--learning_rate", str(args.lr),
            "--warmup_ratio", str(args.warmup_ratio),
            "--lr_scheduler_type", "cosine",
            "--optim", "adamw_torch",
            "--weight_decay", "0.0",
            "--max_grad_norm", "1.0",
            "--per_device_train_batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.grad_accum),
            "--max_seq_length", str(args.max_seq_length),
            "--save_strategy", "epoch",
            "--save_total_limit", "1",
            "--output_dir", str(args.output_dir),
            "--segment_mode", args.segment_mode,
            "--think_prefix", args.think_prefix,
        ]
        if not args.full_sft:
            argv.append("--mask")
        if args.full_finetune:
            argv.append("--full_finetune")
        subprocess.run([
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(args.nproc_per_node),
            str(ROOT / "SelectiveSFT/train_mask.py"), *argv,
        ], check=True)
        return json.loads((Path(args.output_dir) / "training_stats.json").read_text())
    finally:
        tmp_path.unlink(missing_ok=True)


def cost_record(args, prep, trained):
    raw = int(trained["total_raw_tokens"])
    retained = int(trained["retained_supervised_tokens"])
    n_params = int(trained["n_params"])
    epochs = int(args.epochs)
    train_flops = 6.0 * n_params * retained * epochs
    world = int(trained["world_size"])
    prep_wall = float(prep["prep_wall_time_sec"])
    train_wall = float(trained["train_wall_time_sec"])
    pct = (100.0 * retained / raw) if raw else 0.0
    return {
        "run_id": args.run_id,
        "model_path": args.model,
        "full_finetune": args.full_finetune,
        "selective": not args.full_sft,
        "num_epochs": epochs,
        "learning_rate": args.lr,
        "prep_wall_time_sec": prep_wall,
        "prep_peak_vram_gb": float(prep["prep_peak_vram_gb"]),
        "total_raw_tokens": raw,
        "retained_supervised_tokens": retained,
        "retained_token_pct": pct,
        "prep_flops": float(prep["prep_flops"]),
        "train_wall_time_sec": train_wall,
        "train_peak_vram_gb": float(trained["train_peak_vram_gb"]),
        "total_gpu_hours": ((prep_wall + train_wall) / 3600.0) * world,
        "flops_are_estimates": True,
        "train_flops": train_flops,
        "total_flops": float(prep["prep_flops"]) + train_flops,
        "epoch_loss_history": trained["epoch_loss_history"],
        "world_size": world,
        "n_params": n_params,
    }


def stage_eval(args, checkpoint_dir):
    """Sinh va cham diem bang luong vLLM/math-verify dung chung voi P-ALIGN."""
    result_dir = Path(args.output_dir) / "result"
    subprocess.run([
        args.eval_python, str(ROOT.parent / "palign/src/run_eval.py"),
        "--model", str(checkpoint_dir), "--data-dir", str(args.data_dir),
        "--result-dir", str(result_dir), "--n", str(args.eval_n),
        "--max-tokens", str(args.max_tokens), "--max-model-len", str(args.max_model_len),
    ], check=True)
    return json.loads((result_dir / "summary.json").read_text())


def parse_args():
    p = argparse.ArgumentParser(description="One-pass selective SFT + eval, offline, checkpoint local")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train-data", required=True, help="jsonl local: question, solution, answer")
    p.add_argument("--data-dir", default=str(ROOT.parent / "palign/data/raw"), help="Thu muc benchmark local")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--full-sft", action="store_true", help="Vanilla SFT, bo qua IG, prep_wall_time=0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nproc-per-node", type=int, default=3)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--eval-n", type=int, default=int(os.environ.get("EVAL_N", "3")))
    p.add_argument("--eval-python", default=os.environ.get("EVAL_PYTHON", sys.executable))
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=11)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=32768)
    p.add_argument("--segment-mode", default="paragraph")
    p.add_argument("--think-prefix", default="none", choices=["none", "plain", "special"])
    p.add_argument("--ig-steps", type=int, default=50)
    p.add_argument("--ig-batch-size", type=int, default=1)
    p.add_argument("--ig-max-tokens", type=int, default=0)
    p.add_argument("--no-ig-grad-checkpoint", action="store_true")
    p.add_argument("--cumulative-ratio", type=float, default=0.7)
    p.add_argument("--coherence-max", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--eval-backend", default="vllm", choices=["vllm"])
    p.add_argument("--dry-run", action="store_true", help="Chi kiem tra duong dan, khong train")
    return p.parse_args()


def main():
    args = parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    out = Path(args.output_dir) / args.run_id
    if out.exists() and any(out.iterdir()) and not args.dry_run:
        raise FileExistsError(f"{out} already contains a run; choose another run-id")
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)
    records = read_jsonl(args.train_data)
    if not records:
        raise SystemExit(f"Train data rong: {args.train_data}")
    for key in ("question", "solution"):
        if key not in records[0]:
            raise SystemExit(f"Thieu cot {key!r} trong {args.train_data}")

    if args.dry_run:
        print(f"dry-run run_id={args.run_id} selective={not args.full_sft} records={len(records)}")
        for bench in ("aime25", "aime24", "amc12", "math500"):
            n = args.eval_n
            path = Path(args.data_dir) / (bench + ".jsonl")
            print(f"  {bench} n={n} file={path} exists={path.is_file()}")
        return

    prep = stage_prep(args, records)
    trained = stage_train(args, records)
    cost = cost_record(args, prep, trained)
    cost_path = out / (f"{args.run_id}_cost.json")
    final_ckpt = Path(trained["checkpoint"])
    cost["checkpoint"] = str(final_ckpt)
    write_json(cost_path, cost)
    print(f"Da ghi {cost_path}")
    print(f"Checkpoint: {final_ckpt}")

    summaries = stage_eval(args, final_ckpt)

    summary = {
        "run_id": args.run_id,
        "cost_file": cost_path.name,
        "checkpoint": str(final_ckpt),
        "benchmarks": summaries,
    }
    write_json(out / (f"{args.run_id}_summary.json"), summary)
    print(f"Checkpoint giu lai: {final_ckpt}")


if __name__ == "__main__":
    main()
