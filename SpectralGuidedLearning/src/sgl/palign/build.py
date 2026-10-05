"""Assemble P-ALIGN's training file (step 3): teacher prefix + student continuation, kept only when
the continuation reaches the gold answer.

    output = "<Begin_of_Prefix>" + prefix + "<End_of_Prefix>\\n" + continuation.strip()

as in P-ALIGN's released palign_sft_*.json (every one of its 966 rows has exactly this marker
layout; upstream ships no script for this step, so the strip/newline join is read off that file).
Rows are alpaca records {instruction, input, output} with P-ALIGN's instruction, readable by
sgl.data.prepare and by LLaMA-Factory. Correctness is P-ALIGN's grader (math_verify on the last
\\boxed{}); the paper's filter keeps 966 of 1000 s1K-1.1 problems.
"""
import argparse
import json
from pathlib import Path

from sgl.eval.graders import palign as palign_grader

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
BEGIN, END = "<Begin_of_Prefix>", "<End_of_Prefix>"


def assemble(prefix: str, continuation: str) -> str:
    return f"{BEGIN}{prefix}{END}\n{continuation.strip()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aligned", required=True, help="sgl.palign.align output")
    parser.add_argument("--truncated", required=True, help="sgl.palign.truncate output (gold answers)")
    parser.add_argument("--output", required=True, help=".json list of alpaca rows")
    parser.add_argument("--keep-unverified", action="store_true", help="skip the answer filter")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    answers = {}
    for line in open(args.truncated, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            answers[row["question"]] = str(row.get("answer", ""))

    rows, dropped = [], {"no_gold": 0, "wrong": 0}
    for line in open(args.aligned, encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        gold = answers.get(item["question"], "")
        if not args.keep_unverified:
            if not gold:
                dropped["no_gold"] += 1
                continue
            if not palign_grader.grade_math_verify([item["output"]], gold)[0]:
                dropped["wrong"] += 1
                continue
        rows.append({"instruction": INSTRUCTION, "input": item["question"],
                     "output": assemble(item["sufficient_reasoning"], item["output"])})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"kept {len(rows)} rows -> {output}; dropped {dropped}")


if __name__ == "__main__":
    main()
