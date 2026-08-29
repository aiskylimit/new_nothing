"""Benchmark loaders returning a unified record format: {id, question, gold, task_type}.

GPQA-Diamond options are shuffled with a fixed per-question seed so the correct letter is
not always the same position, while staying identical across evaluated models.
"""

import os
import random

from datasets import load_dataset

from answer_scoring import extract_boxed

MATH_PROMPT = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
MCQ_PROMPT = (
    "{question}\n\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
    "Please reason step by step, and put the letter of your final answer within \\boxed{{}}.\n\n"
)

BENCH_DATA_ROOT = os.environ.get("BENCH_DATA_ROOT")


def _dataset_path(repo_id: str, local_dir_name: str) -> str:
    """BENCH_DATA_ROOT/<local_dir_name> on an offline server (see download.txt), else the HF repo id."""
    return f"{BENCH_DATA_ROOT}/{local_dir_name}" if BENCH_DATA_ROOT else repo_id


def _math_records(rows, question_key: str, answer_key: str) -> list[dict]:
    return [
        {
            "id": index,
            "question": row[question_key],
            "prompt": MATH_PROMPT.format(question=row[question_key]),
            "gold": str(row[answer_key]),
            "task_type": "math",
        }
        for index, row in enumerate(rows)
    ]


def load_aime24() -> list[dict]:
    """math-ai/aime24 stores the gold answer as a boxed expression in `solution`."""
    rows = load_dataset(_dataset_path("math-ai/aime24", "aime24"), split="test")
    records = _math_records(rows, "problem", "solution")
    for record in records:
        record["gold"] = extract_boxed(record["gold"]) or record["gold"]
    return records


def load_aime25() -> list[dict]:
    rows = load_dataset(_dataset_path("math-ai/aime25", "aime25"), split="test")
    return _math_records(rows, "problem", "answer")


def load_math500() -> list[dict]:
    rows = load_dataset(_dataset_path("HuggingFaceH4/MATH-500", "MATH-500"), split="test")
    return _math_records(rows, "problem", "answer")


def load_olympiadbench() -> list[dict]:
    """English open-ended text-only math competition subset."""
    rows = load_dataset(_dataset_path("Hothan/OlympiadBench", "OlympiadBench"), "OE_TO_maths_en_COMP", split="train")
    records = []
    for index, row in enumerate(rows):
        answers = row.get("final_answer") or []
        if not answers:
            continue
        records.append(
            {
                "id": index,
                "question": row["question"],
                "prompt": MATH_PROMPT.format(question=row["question"]),
                "gold": str(answers[0]).strip("$"),
                "task_type": "math",
            }
        )
    return records


def load_amc12() -> list[dict]:
    """AI-MO/aimo-validation-amc: 83 AMC 12 problems (2022-2023), matches P-ALIGN Table 3."""
    rows = load_dataset(_dataset_path("AI-MO/aimo-validation-amc", "aimo-validation-amc"), split="train")
    return _math_records(rows, "problem", "answer")


def load_gpqa_diamond() -> list[dict]:
    rows = load_dataset(_dataset_path("Idavidrein/gpqa", "gpqa"), "gpqa_diamond", split="train")
    records = []
    for index, row in enumerate(rows):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        rng = random.Random(index)  # fixed per question -> same layout for every model
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        shuffled = [options[position] for position in order]
        records.append(
            {
                "id": index,
                "question": row["Question"],
                "prompt": MCQ_PROMPT.format(
                    question=row["Question"],
                    a=shuffled[0],
                    b=shuffled[1],
                    c=shuffled[2],
                    d=shuffled[3],
                ),
                "gold": "ABCD"[order.index(0)],
                "task_type": "mcq",
            }
        )
    return records


BENCHMARKS = {
    "aime24": load_aime24,
    "aime25": load_aime25,
    "math500": load_math500,
    "olympiadbench": load_olympiadbench,
    "gpqa": load_gpqa_diamond,
    "amc12": load_amc12,
}
