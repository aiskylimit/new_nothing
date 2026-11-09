"""Convert simplescaling/s1K-1.1 dataset to StepDataset JSONL format."""

from __future__ import annotations

import json
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

from core.segmentation import split_thinking_into_steps


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading simplescaling/s1K-1.1...")
    ds = load_dataset("simplescaling/s1K-1.1", split="train")
    print(f"Total samples: {len(ds)}")

    out_path = "data/train_s1k.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
            "id": f"s1k_{idx}",
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
