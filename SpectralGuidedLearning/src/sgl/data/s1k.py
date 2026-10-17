"""s1K-1.1 (or any trace dataset) -> rows {question, solution, answer[, segments]}.

The input format of both Segment-Selective SFT and P-ALIGN. Port of
SegmentSelectiveSFT/prepare_s1k.py followed by Attribution/segment_split.py:
`solution` is the long-CoT trace (default deepseek_thinking_trajectory), `answer` the last
\\boxed{} of the trace (fallback: s1K's own `solution` field, the bare ground truth), and
`segments` the trace split by the chosen rule. Rows whose answer is missing or longer than
--max-answer-chars are dropped, since an IG target that long is meaningless.

    python -m sgl.data.s1k --output data/s1k-traces/train.jsonl [--segment-mode none]
"""
import argparse
import glob
import json
import os
from pathlib import Path

from sgl.segment.rules import SEGMENT_PATTERNS, split_segments


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, brace-matched (prepare_s1k.py last_boxed)."""
    index = text.rfind("\\boxed{")
    if index == -1:
        return None
    start = index + len("\\boxed{")
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position].strip()
    return None


def convert_rows(rows, trace_field: str, answer_source: str, max_answer_chars: int) -> tuple[list[dict], dict]:
    kept, stats = [], {"no_trace": 0, "no_answer": 0, "answer_too_long": 0, "trace_disagrees_with_gt": 0}
    for row in rows:
        trace = (row.get(trace_field) or "").strip()
        if not trace:
            stats["no_trace"] += 1
            continue
        ground_truth = (row.get("solution") or "").strip()
        boxed = last_boxed(trace)
        answer = (ground_truth or boxed) if answer_source == "gt" else (boxed or ground_truth)
        if not answer:
            stats["no_answer"] += 1
            continue
        if len(answer) > max_answer_chars:
            stats["answer_too_long"] += 1
            continue
        if boxed is not None and ground_truth and boxed != ground_truth:
            stats["trace_disagrees_with_gt"] += 1
        kept.append({"question": row["question"].strip(), "solution": trace, "answer": answer})
    return kept, stats


def load_rows(dataset: str, split: str):
    from datasets import load_dataset

    if os.path.isdir(dataset):  # offline HF snapshot: <dir>/**/<split>-*.parquet (or *.jsonl)
        files = sorted(glob.glob(os.path.join(dataset, "**", "*.parquet"), recursive=True))
        fmt = "parquet"
        if not files:
            files = sorted(glob.glob(os.path.join(dataset, "**", "*.jsonl"), recursive=True))
            fmt = "json"
        if not files:
            raise SystemExit(f"no *.parquet / *.jsonl under {dataset}")
        by_split = [path for path in files if os.path.basename(path).startswith(split)]
        return load_dataset(fmt, data_files=by_split or files, split="train")
    if os.path.isfile(dataset):
        return load_dataset("json", data_files=dataset, split="train")
    return load_dataset(dataset, split=split)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1", help="HF id, snapshot dir or local jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--trace-field", default="deepseek_thinking_trajectory")
    parser.add_argument("--answer-source", default="trace", choices=["trace", "gt"],
                        help="trace = last \\boxed{} of the trace (default); gt = s1K's ground-truth field")
    parser.add_argument("--max-answer-chars", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--segment-mode", default="paragraph", choices=[*sorted(SEGMENT_PATTERNS), "none"],
                        help="paragraph = every \\n\\n (SSFT fork default); cue = backtracking cues "
                        "(SSFT paper); none = no segments field")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows, stats = convert_rows(load_rows(args.dataset, args.split), args.trace_field,
                               args.answer_source, args.max_answer_chars)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if args.segment_mode != "none":
        for row in rows:
            row["segments"] = split_segments(row["solution"], args.segment_mode)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = [len(row["segments"]) for row in rows if "segments" in row]
    print(f"kept {len(rows)} rows -> {output}; dropped: {stats}")
    if counts:
        print(f"segments/row ({args.segment_mode}): mean {sum(counts) / len(counts):.1f}, "
              f"min {min(counts)}, max {max(counts)}")


if __name__ == "__main__":
    main()
