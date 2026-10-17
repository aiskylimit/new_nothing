"""Tokenize Segment-Selective SFT rows into {id, input_ids, loss_mask} records, exactly as
SegmentSelectiveSFT/SelectiveSFT/train_mask.py builds its labels.

    text     = chat_template(user: prompt) + think_str + solution
    selected = --selective: the segments in `selected_spans_ids` plus, always, the first,
               second-to-last and last segment; --no-selective: every response token
    mask[t]  = 1 iff token t starts inside the response and its first character lies in a
               selected segment

Prompt (`--prompt-style`):  default = "<question>\\nPlease reason step by step, ...\\boxed{}."
                            palign  = "Please reason step by step, ...\\boxed{}.<question>"
Thinking (`--think-prefix`, with `--deepseek` for R1-Distill templates):
    Qwen/ChatML  none: nothing after "assistant\\n" | plain: "<think>\\n" | off: enable_thinking=False
    DeepSeek-R1  none/plain: learn inside the template's open <think> | off: "<think>\\n\\n</think>\\n\\n"
The chat template is probed once and must end the way the chosen mode expects.

Upstream behaviour kept on purpose: no EOS token is appended (the target ends with the trace),
sequences are truncated at --max-seq-length, and samples left with no supervised token are dropped.
The records train with sgl.training.train like any other masked dataset.
"""
import argparse
import bisect
import json
from pathlib import Path

from sgl.segment.rules import SEGMENT_PATTERNS, split_segments

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
THINK_PREFIXES = ("none", "off", "plain")


def prompt_content(question: str, prompt_style: str) -> str:
    if prompt_style == "palign":
        return INSTRUCTION + question.rstrip()
    return question + "\n" + INSTRUCTION


def template_mode(deepseek: bool, think_prefix: str) -> tuple[dict, str, str]:
    """(chat_template kwargs, text inserted after the generation prompt, expected prompt ending)."""
    if think_prefix not in THINK_PREFIXES:
        raise ValueError(f"think_prefix must be one of {THINK_PREFIXES} ('special' needs new tokens and "
                         "an embedding resize: use the reference trainer)")
    if not deepseek:
        if think_prefix == "off":
            return {"enable_thinking": False}, "", "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        think = "" if think_prefix == "none" else "<think>\n"
        return {}, think, "<|im_start|>assistant\n" + think
    if think_prefix == "off":
        return {}, "\n</think>\n\n", "<｜Assistant｜><think>\n\n</think>\n\n"
    return {}, "", "<｜Assistant｜><think>\n"


def check_template(tokenizer, chat_kwargs: dict, think: str, expected: str) -> None:
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}], tokenize=False, add_generation_prompt=True, **chat_kwargs
    )
    if not (probe + think).endswith(expected):
        raise SystemExit(
            f"chat template ends with {(probe + think)[-40:]!r}, not {expected!r}: "
            "check --deepseek / --think-prefix for this model"
        )


def build_record(tokenizer, row: dict, *, prompt_style: str, chat_kwargs: dict, think: str,
                 selective: bool, segment_mode: str, max_seq_length: int) -> dict | None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content(row["question"], prompt_style)}],
        tokenize=False, add_generation_prompt=True, **chat_kwargs,
    )
    output = row["solution"]
    encoding = tokenizer(prompt + think + output, add_special_tokens=False, truncation=True,
                         max_length=max_seq_length, return_offsets_mapping=True)
    response_char = len(prompt) + len(think)

    starts = None
    if selective:
        segments = split_segments(output, segment_mode)
        always = [0, len(segments) - 2, len(segments) - 1]
        keep = {i for i in always + list(row.get("selected_spans_ids") or []) if 0 <= i < len(segments)}
        starts, cursor = [], response_char
        for segment in segments:
            starts.append(cursor)
            cursor += len(segment)
        end_char = cursor

    mask = [0] * len(encoding["input_ids"])
    for position, (char_start, char_end) in enumerate(encoding["offset_mapping"]):
        if char_end <= char_start or char_start < response_char:
            continue  # special token, or prompt
        if starts is None:
            mask[position] = 1
            continue
        if char_start >= end_char:
            continue
        if bisect.bisect_right(starts, char_start) - 1 in keep:
            mask[position] = 1
    if not any(mask):
        return None  # everything truncated away: cross-entropy over zero tokens is NaN
    return {"input_ids": list(encoding["input_ids"]), "loss_mask": mask}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="rows with question / solution [/ selected_spans_ids]")
    parser.add_argument("--tokenizer", required=True, help="tokenizer of the model being trained")
    parser.add_argument("--output", required=True)
    parser.add_argument("--selective", action=argparse.BooleanOptionalAction, default=True,
                        help="mask to the selected segments (default); --no-selective = full long-CoT SFT")
    parser.add_argument("--deepseek", action=argparse.BooleanOptionalAction,
                        help="DeepSeek-R1 chat template (default: on when the name contains DeepSeek-R1)")
    parser.add_argument("--think-prefix", default="none", choices=THINK_PREFIXES)
    parser.add_argument("--prompt-style", default="default", choices=["default", "palign"])
    parser.add_argument("--segment-mode", default="paragraph", choices=sorted(SEGMENT_PATTERNS))
    parser.add_argument("--max-seq-length", type=int, default=32768)
    return parser


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise SystemExit("a fast tokenizer (offset mapping) is required to place segment boundaries")
    deepseek = args.deepseek if args.deepseek is not None else "DeepSeek-R1" in args.tokenizer
    chat_kwargs, think, expected = template_mode(deepseek, args.think_prefix)
    check_template(tokenizer, chat_kwargs, think, expected)

    rows = [json.loads(line) for line in open(args.data_path) if line.strip()]
    if args.selective and rows and "selected_spans_ids" not in rows[0]:
        raise SystemExit(f"{args.data_path} has no selected_spans_ids: run sgl.ssft.select first")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = supervised = total = 0
    with output.open("w") as handle:
        for index, row in enumerate(rows):
            record = build_record(tokenizer, row, prompt_style=args.prompt_style, chat_kwargs=chat_kwargs,
                                  think=think, selective=args.selective, segment_mode=args.segment_mode,
                                  max_seq_length=args.max_seq_length)
            if record is None:
                continue
            kept += 1
            supervised += sum(record["loss_mask"])
            total += len(record["input_ids"])
            handle.write(json.dumps({"id": index, **record}) + "\n")
    mode = "selective" if args.selective else "full long-CoT"
    print(f"{mode}: {kept}/{len(rows)} records -> {output} ({len(rows) - kept} dropped: nothing left "
          f"to supervise at max length {args.max_seq_length}); supervised {supervised}/{total} tokens")


if __name__ == "__main__":
    main()
