import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed

from reasoning_velocity_distill.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--student_device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--data_dir", type=str, default="processed_data/ace/qwen/")
    parser.add_argument("--dataset_name", type=str, default="ace")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.max_prompt_length < args.max_length:
        parser.error("Require 0 < --max_prompt_length < --max_length")
    if args.val_batch_size < 1:
        parser.error("--val_batch_size must be positive")

    set_seed(args.seed)
    dtype = torch.float32 if torch.device(args.student_device).type == "cpu" else (
        torch.bfloat16 if args.bf16 else torch.float16
    )
    evaluator = Evaluator(
        tokenizer_path=args.tokenizer or args.model_path,
        model_type=args.model_type,
        model_path=args.model_path,
        distilled_lora=args.lora_path,
        device=args.student_device,
        seeds=[args.seed],
        dtype=dtype,
    )
    evaluator.model.config.output_hidden_states = False
    evaluator.model.config.output_attentions = False
    metrics, responses = evaluator.evaluate_benchmark_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        batch_size=args.val_batch_size,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        split=args.split,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.dataset_name}_eval.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=4)
    with (output_dir / f"{args.dataset_name}_answers.jsonl").open("w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(json.dumps({"text": response}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
