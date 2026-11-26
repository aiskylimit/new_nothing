"""Ablation mask: supervise only the steps that contain the final answer.

Same prepared data and same definitions as sgl.diagnostics.answer_steps (`mention`: the answer
string occurs in the step; `boxed`: the step writes it inside \\boxed{}). Every other step is
masked out of the loss; the stop token stays supervised, as in every arm. Samples with no such
step (no \\boxed{} answer) are left out, so they do not train on the stop token alone.
"""

import argparse
import json
from pathlib import Path

from sgl.diagnostics.answer_steps import DEFINITIONS, analyze_record
from sgl.segment.sentence import record_step_spans
from sgl.selection.energy import build_loss_mask


def mask_record(record: dict, tokenizer, definition: str) -> dict | None:
    sample = analyze_record(record, tokenizer, None, None)
    selected = [step["index"] for step in sample["steps"] if step[definition]] if sample else []
    if not selected:
        return None
    loss_mask = build_loss_mask(len(record["input_ids"]), record_step_spans(record), selected)
    for position in range(record["response_token_span"][1], len(loss_mask)):
        loss_mask[position] = 1
    return {"id": record["id"], "input_ids": record["input_ids"], "loss_mask": loss_mask}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--tokenizer", required=True, help="tokenizer the data was prepared with")
    parser.add_argument("--definition", choices=DEFINITIONS, default="mention")
    parser.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = kept = supervised = response_tokens = 0
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.data_path) as source, output.open("w") as handle:
        for line in source:
            record = json.loads(line)
            samples += 1
            example = mask_record(record, tokenizer, args.definition)
            if example is None:
                continue
            kept += 1
            start, end = record["response_token_span"]
            supervised += sum(example["loss_mask"][start:end])
            response_tokens += end - start
            handle.write(json.dumps(example) + "\n")
    if not kept:
        raise ValueError(f"no sample has a step matching {args.definition!r}")
    print(
        f"{args.definition}: kept {kept}/{samples} samples (dropped {samples - kept} without an answer step); "
        f"supervising {supervised:,}/{response_tokens:,} response tokens ({supervised / response_tokens:.2%}) -> {output}"
    )


if __name__ == "__main__":
    main()
