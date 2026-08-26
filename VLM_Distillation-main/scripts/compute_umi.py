"""Compute Unit-Mismatch Index (UMI) for a teacher/student VLM pair.

UMI components (from `unit_aligned_distillation_audit_and_plan.md` §2):
  UMI = 0.5 * (1 - shared_vocab_jaccard_on_response_corpus)
      + 0.5 * (1 - min(n_v_s, n_v_t) / max(n_v_s, n_v_t) averaged over images)

Usage:
  python scripts/compute_umi.py \
      --student Qwen/Qwen2-VL-2B-Instruct \
      --teacher Qwen/Qwen2.5-VL-7B-Instruct \
      --data_path train_data/llava_v1_5_mix665k.json \
      --image_dir train_data \
      --response_sample 5000 \
      --image_sample 1000 \
      --out_path artifacts/umi_qwen25_to_qwen2.json
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="HF id or local path for student VLM")
    p.add_argument("--teacher", required=True, help="HF id or local path for teacher VLM")
    p.add_argument("--data_path", required=True, help="LLaVA-style JSON/JSONL")
    p.add_argument("--image_dir", default=None)
    p.add_argument("--response_sample", type=int, default=5000)
    p.add_argument("--image_sample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_path", default="artifacts/umi.json")
    p.add_argument("--image_resolution", default="mid", choices=["low", "mid", "high"])
    return p.parse_args()


def _load_records(path: str):
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_assistant_texts(records, n: int, seed: int):
    rng = random.Random(seed)
    pool = []
    for rec in records:
        for turn in rec.get("conversations", []):
            role = turn.get("from") or turn.get("role")
            if role in {"gpt", "assistant"}:
                value = turn.get("value") or turn.get("content")
                if isinstance(value, str) and value.strip():
                    pool.append(value)
    rng.shuffle(pool)
    return pool[:n]


def _vocab_jaccard_on_corpus(student_tok, teacher_tok, texts) -> Tuple[float, int, int]:
    """Jaccard on tokens *actually emitted* against the corpus, not raw vocab."""
    s_ids = Counter()
    t_ids = Counter()
    for txt in texts:
        s_ids.update(student_tok.encode(txt, add_special_tokens=False))
        t_ids.update(teacher_tok.encode(txt, add_special_tokens=False))

    # Map id sets back to the actual surface tokens so comparison is meaningful
    # across non-identical vocabs.
    s_strs = {student_tok.convert_ids_to_tokens(i) for i in s_ids}
    t_strs = {teacher_tok.convert_ids_to_tokens(i) for i in t_ids}
    inter = s_strs & t_strs
    union = s_strs | t_strs
    return (len(inter) / max(len(union), 1), len(s_strs), len(t_strs))


def _vision_token_counts(student_proc, teacher_proc, image_paths, image_resolution: str):
    from PIL import Image
    s_counts, t_counts = [], []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except (FileNotFoundError, OSError):
            continue
        # Each processor returns the number of vision tokens it would emit.
        try:
            s_pix = student_proc(images=img, return_tensors="pt")
            t_pix = teacher_proc(images=img, return_tensors="pt")
            s_n = int(s_pix.get("image_grid_thw", s_pix.get("pixel_values").shape).prod()) if "image_grid_thw" in s_pix else int(s_pix["pixel_values"].numel() / 3 / 14 / 14)
            t_n = int(t_pix.get("image_grid_thw", t_pix.get("pixel_values").shape).prod()) if "image_grid_thw" in t_pix else int(t_pix["pixel_values"].numel() / 3 / 14 / 14)
            s_counts.append(s_n)
            t_counts.append(t_n)
        except Exception as exc:
            print(f"[warn] vision-token probe failed for {p}: {exc}")
            continue
    return s_counts, t_counts


def main():
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy-import so the script is also usable for the vocab-only Jaccard piece
    # in a slim env without VLM processors installed.
    from transformers import AutoTokenizer

    student_tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    records = _load_records(args.data_path)
    texts = _extract_assistant_texts(records, args.response_sample, args.seed)
    jaccard, s_voc, t_voc = _vocab_jaccard_on_corpus(student_tok, teacher_tok, texts)

    # Vision token ratio
    rng = random.Random(args.seed)
    image_paths = []
    for rec in records:
        img_field = rec.get("image")
        if img_field:
            p = os.path.join(args.image_dir, img_field) if args.image_dir else img_field
            if os.path.exists(p):
                image_paths.append(p)
    rng.shuffle(image_paths)
    image_paths = image_paths[: args.image_sample]

    vision_ratio_mean = None
    if image_paths:
        try:
            from src.model.processor import load_processor
            from src.arguments import ModelArguments
            s_proc = load_processor(ModelArguments(model_name=args.student))
            t_proc = load_processor(ModelArguments(model_name=args.teacher))
            s_counts, t_counts = _vision_token_counts(s_proc, t_proc, image_paths, args.image_resolution)
            if s_counts and t_counts:
                ratios = [min(a, b) / max(a, b) for a, b in zip(s_counts, t_counts)]
                vision_ratio_mean = sum(ratios) / len(ratios)
        except Exception as exc:
            print(f"[warn] vision-token component failed ({exc}); reporting text-only UMI")

    text_component = 1.0 - jaccard
    vision_component = 1.0 - vision_ratio_mean if vision_ratio_mean is not None else None
    if vision_component is not None:
        umi = 0.5 * text_component + 0.5 * vision_component
    else:
        umi = text_component  # text-only fallback

    result = {
        "student": args.student,
        "teacher": args.teacher,
        "data_path": args.data_path,
        "response_sample": args.response_sample,
        "image_sample": len(image_paths),
        "student_vocab_used": s_voc,
        "teacher_vocab_used": t_voc,
        "vocab_jaccard_on_corpus": jaccard,
        "vision_token_ratio_mean": vision_ratio_mean,
        "umi_text_component": text_component,
        "umi_vision_component": vision_component,
        "umi": umi,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2))
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nUMI saved to {out_path}")


if __name__ == "__main__":
    main()
