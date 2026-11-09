"""Convert simplescaling/s1K-1.1 dataset to StepDataset JSONL format."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Support both `python data/prepare_s1k.py` and `python -m data.prepare_s1k`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.local_assets import load_local_rows
from core.model_utils import load_tokenizer
from core.segmentation import split_thinking_into_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="datasets/s1K-1.1")
    parser.add_argument("--out", default="data/train_s1k.jsonl")
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = load_tokenizer(model_name)

    print(f"Loading local s1K: {args.dataset}...")
    ds = load_local_rows(args.dataset, split="train")
    print(f"Total samples: {len(ds)}")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    instruction_prefix = "Please reason step by step, and put your final answer within \\boxed{}."

    kept = []
    for idx, item in enumerate(ds):
        raw_q = item["question"].strip()
        thinking = item["deepseek_thinking_trajectory"].strip()
        answer_text = item["deepseek_attempt"].strip()
        solution = str(item["solution"]).strip()

        question = f"{instruction_prefix}{raw_q}"
        full_cot = f"<think>\n{thinking}\n</think>\n\n{answer_text}"
        steps = split_thinking_into_steps(thinking, mode="paragraph")
        
        # calculate tokens
        n_tokens = len(tokenizer(full_cot + question, add_special_tokens=False)["input_ids"])

        row = {
            "id": str(item.get("id", f"s1k_{idx}")),
            "question": question,
            "raw_question": raw_q,
            "thinking": thinking,
            "steps": steps,
            "answer": solution,
            "answer_text": answer_text,
            "full_cot": full_cot,
            "n_tokens": n_tokens,
            "data_source": item.get("source_type", "s1K"),
        }
        kept.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Successfully converted {len(kept)} samples to {out_path}")
    print(f"Sample 0 steps: {len(kept[0]['steps'])}, tokens: {kept[0]['n_tokens']}")


if __name__ == "__main__":
    main()
