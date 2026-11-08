"""Pipeline mot luot: attribution -> SFT full-parameter -> eval.

Offline: khong day gi len mang. Checkpoint ghi tren dia local,
{output_dir}/checkpoint-final, de vLLM doc. Eval loi thi weight van con.
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
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SelectiveSFT"))
sys.path.insert(0, str(ROOT / "Attribution"))
sys.path.insert(0, str(ROOT / "Eval"))

# N mau / bai. AIME va AMC dung 64; MATH500 va benchmark khong-toan dung 15.
BENCHES = (
    ("aime24", 64),
    ("aime25", 64),
    ("amc12", 64),
    ("math500", 15),
    ("gpqa", 15),
)


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
    from grad_analyze import IntegratedGradientsAttribution, attribute_example
    from get_important_segments import select_important_segments

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
    """Stage 2. Full-parameter SFT. Moi epoch ghi mot checkpoint, giu ban moi nhat."""
    import train_mask

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
            "--full_finetune",
            "--epochs", str(args.epochs),
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
        ns = train_mask.parse_args(argv)
        return train_mask.run(ns)
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
        "full_finetune": True,
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
        "train_flops": train_flops,
        "total_flops": float(prep["prep_flops"]) + train_flops,
        "epoch_loss_history": trained["epoch_loss_history"],
        "world_size": world,
        "n_params": n_params,
    }


def aggregate_benchmark(bench, n_samples, rows):
    """Pass@1 = tong lan dung / (n * N). Pass@k = ty le bai co it nhat 1 cau dung."""
    n_problems = len(rows)
    total_correct = 0
    any_correct = 0
    gen_tokens = 0
    gen_count = 0
    for row in rows:
        flags = [1 if int(x) else 0 for x in row["is_correct"]]
        total_correct += sum(flags)
        any_correct += 1 if any(flags) else 0
        gen_tokens += sum(int(x) for x in row["gen_token_lengths"])
        gen_count += len(flags)
    denom = n_samples * n_problems
    return {
        "benchmark": bench,
        "n_problems": n_problems,
        "n_samples": n_samples,
        "pass@1": (total_correct / denom) if denom else 0.0,
        "pass@%d" % n_samples: (any_correct / n_problems) if n_problems else 0.0,
        "total_correct": total_correct,
        "mean_gen_tokens": (gen_tokens / gen_count) if gen_count else 0.0,
    }


def build_prompts(tokenizer, examples, data_name):
    from parser import parse_ground_truth, parse_question
    from utils import construct_prompt

    class _Args:
        prompt_type = "deepseek-longcot"
        num_shots = 0
        adapt_few_shot = False

    built = []
    for ex in examples:
        ex = dict(ex)
        ex["question"] = parse_question(ex, data_name)
        _cot, gt = parse_ground_truth(ex, data_name)
        ex["gt_ans"] = gt
        prompt = construct_prompt(ex, data_name, _Args)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        built.append((ex.get("idx", len(built)), str(gt), text))
    return built


def score_generation(text, gt, data_name):
    from parser import extract_answer, strip_string
    from grader import math_equal

    pred = extract_answer(text or "", data_name)
    if pred is None:
        pred = ""
    else:
        pred = strip_string(str(pred))
    try:
        ok = bool(math_equal(pred, gt))
    except Exception:
        ok = False
    return str(pred), int(ok)


def generate_hf(model, tokenizer, prompts, n_samples, max_new_tokens, temperature, top_p, repetition_penalty):
    import torch
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    grouped = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        sample_out = []
        for _ in range(n_samples):
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad,
                    eos_token_id=eos,
                )
            new_ids = gen[0, enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            sample_out.append((text, int(new_ids.shape[0])))
        grouped.append(sample_out)
    return grouped


def generate_vllm(llm, prompts, n_samples, max_new_tokens, temperature, top_p, repetition_penalty):
    from vllm import SamplingParams
    flat = [p for p in prompts for _ in range(n_samples)]
    outputs = llm.generate(
        flat,
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            n=1,
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    grouped = []
    for i in range(len(prompts)):
        chunk = outputs[i * n_samples:(i + 1) * n_samples]
        grouped.append([
            (o.outputs[0].text, len(o.outputs[0].token_ids)) for o in chunk
        ])
    return grouped


def load_benchmark(data_dir, bench):
    path = Path(data_dir) / bench / "test.jsonl"
    if not path.is_file():
        return None, path
    from data_loader import load_data
    return load_data(bench, "test", str(data_dir)), path


def checkpoint_ready(path):
    path = Path(path)
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    return (path / "config.json").is_file() and bool(weights)


def save_final_checkpoint(model, tokenizer, output_dir):
    """Ghi weight cuoi vao output_dir/checkpoint-final. Khong xoa file nay."""
    dest = Path(output_dir) / "checkpoint-final"
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest, safe_serialization=True)
    tokenizer.save_pretrained(dest)
    if not checkpoint_ready(dest):
        raise RuntimeError("checkpoint-final thieu config.json hoac weight: %s" % dest)
    return dest


def stage_eval(args, slot, checkpoint_dir, release_fn):
    """Stage 3 + 4. vLLM nap checkpoint local. File checkpoint khong bi xoa."""
    import torch
    model = slot["model"]
    tokenizer = slot["tokenizer"]
    temperature, top_p, repetition_penalty = 0.6, 0.9, 1.05
    max_new_tokens = args.max_tokens
    backend = args.eval_backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except Exception:
            backend = "hf"

    llm = None
    summaries = []
    jobs = []
    for bench, n_samples in BENCHES:
        examples, path = load_benchmark(args.data_dir, bench)
        if examples is None:
            summaries.append({
                "benchmark": bench,
                "error": "missing %s" % path,
                "n_samples": n_samples,
            })
            print("Bo qua %s: khong co %s" % (bench, path))
            continue
        built = build_prompts(tokenizer, examples, bench)
        jobs.append((bench, n_samples, built))

    if not checkpoint_ready(checkpoint_dir):
        raise RuntimeError("Eval can checkpoint local, khong thay weight o %s" % checkpoint_dir)

    try:
        if backend == "vllm":
            # Weight da nam tren dia. Nha ban trong VRAM de vLLM nap dung checkpoint do.
            release_fn()
            del model, tokenizer
            model = tokenizer = None
            empty_cuda()
            from vllm import LLM
            llm = LLM(
                model=str(checkpoint_dir),
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                max_model_len=args.max_model_len,
                trust_remote_code=True,
            )

        for bench, n_samples, built in jobs:
            prompts = [p for _i, _gt, p in built]
            if backend == "vllm":
                grouped = generate_vllm(
                    llm, prompts, n_samples, max_new_tokens, temperature, top_p, repetition_penalty
                )
            else:
                grouped = generate_hf(
                    model, tokenizer, prompts, n_samples, max_new_tokens,
                    temperature, top_p, repetition_penalty,
                )
            raw_path = Path(args.output_dir) / ("%s_%s_raw.jsonl" % (args.run_id, bench))
            rows = []
            with open(raw_path, "w", encoding="utf-8") as f:
                for (idx, gt, _prompt), samples in zip(built, grouped):
                    answers, flags, lengths = [], [], []
                    for text, n_tok in samples:
                        pred, ok = score_generation(text, gt, bench)
                        answers.append(pred)
                        flags.append(ok)
                        lengths.append(int(n_tok))
                    row = {
                        "problem_id": "%s_%s" % (bench, idx),
                        "ground_truth": gt,
                        "is_correct": flags,
                        "extracted_answers": answers,
                        "gen_token_lengths": lengths,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
            metrics = aggregate_benchmark(bench, n_samples, rows)
            metrics["raw_file"] = raw_path.name
            write_json(Path(args.output_dir) / ("%s_%s_metrics.json" % (args.run_id, bench)), metrics)
            summaries.append(metrics)
            print("%s pass@1=%.4f n=%d problems=%d" % (
                bench, metrics["pass@1"], n_samples, metrics["n_problems"]))
        return summaries
    finally:
        if llm is not None:
            del llm
        empty_cuda()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def release_training(trained):
    if trained.get("_released"):
        return
    trainer = trained.pop("trainer", None)
    trained.pop("model", None)
    trained.pop("tokenizer", None)
    if trainer is not None:
        trainer.model = None
        trainer.optimizer = None
        trainer.lr_scheduler = None
        del trainer
    trained["_released"] = True
    empty_cuda()


def parse_args():
    p = argparse.ArgumentParser(description="One-pass selective SFT + eval, offline, checkpoint local")
    p.add_argument("--model", required=True, help="Duong dan local toi DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--train-data", required=True, help="jsonl local: question, solution, answer")
    p.add_argument("--data-dir", default=str(ROOT / "data"), help="Thu muc benchmark local")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--full-sft", action="store_true", help="Vanilla SFT, bo qua IG, prep_wall_time=0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
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
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--eval-backend", default="auto", choices=["auto", "hf", "vllm"])
    p.add_argument("--dry-run", action="store_true", help="Chi kiem tra duong dan, khong train")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.train_data)
    if not records:
        raise SystemExit("Train data rong: %s" % args.train_data)
    for key in ("question", "solution"):
        if key not in records[0]:
            raise SystemExit("Thieu cot %r trong %s" % (key, args.train_data))
    if not Path(args.model).exists():
        raise SystemExit("Khong thay model local: %s" % args.model)

    if args.dry_run:
        print("dry-run run_id=%s selective=%s records=%d" % (
            args.run_id, not args.full_sft, len(records)))
        for bench, n in BENCHES:
            path = Path(args.data_dir) / bench / "test.jsonl"
            print("  %s n=%d file=%s exists=%s" % (bench, n, path, path.is_file()))
        return

    prep = stage_prep(args, records)
    trained = stage_train(args, records)
    cost = cost_record(args, prep, trained)
    cost_path = out / ("%s_cost.json" % args.run_id)
    final_ckpt = save_final_checkpoint(trained["model"], trained["tokenizer"], out)
    cost["checkpoint"] = str(final_ckpt)
    write_json(cost_path, cost)
    print("Da ghi %s" % cost_path)
    print("Checkpoint: %s" % final_ckpt)

    trainer = trained.get("trainer")
    if trainer is not None:
        trainer.optimizer = None
        trainer.lr_scheduler = None
    slot = {"model": trained.get("model"), "tokenizer": trained.get("tokenizer")}

    def release():
        # Chi tha ban trong VRAM. checkpoint-final tren dia giu nguyen.
        slot["model"] = None
        slot["tokenizer"] = None
        release_training(trained)

    summaries = []
    try:
        summaries = stage_eval(args, slot, final_ckpt, release)
    finally:
        release()
        empty_cuda()

    summary = {
        "run_id": args.run_id,
        "cost_file": cost_path.name,
        "checkpoint": str(final_ckpt),
        "benchmarks": summaries,
    }
    write_json(out / ("%s_summary.json" % args.run_id), summary)
    print("Checkpoint giu lai: %s" % final_ckpt)


if __name__ == "__main__":
    main()
