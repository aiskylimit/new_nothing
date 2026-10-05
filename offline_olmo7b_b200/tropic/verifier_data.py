"""Data loader for the RLSD/SDPO baselines.

Reuses `tropic.data.OPSD_DATASET_PATH` and `tropic.data.build_student_prompt`
(both public imports) so all four methods (TROPIC-P, OPSD, RLSD, SDPO) train
on byte-identical (dataset row -> student prompt) pairs: same dataset, same
`correct==True` filter, same shuffle/seed/select, same student-side prompt
wording. The ONLY thing added here is the dataset's own `Answer` column
(verified live to exist alongside `problem`/`solution` - see tropic/data.py's
own docstring on `load_opsd_math_examples`), which neither TROPIC-P nor OPSD
ever needs (pure distillation, no verifier) but RLSD/SDPO both require for a
scalar/binary reward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from tropic.data import OPSD_DATASET_PATH, build_student_prompt


@dataclass
class VerifierExample:
    question: str
    answer: str  # gold FINAL answer only (no chain-of-thought)
    student_prompt_ids: torch.Tensor


def load_opsd_math_examples_with_answer(
    tokenizer: PreTrainedTokenizerBase,
    num_examples: int,
    split: str = "train",
    seed: int = 0,
    only_correct: bool = True,
    student_thinking: bool = False,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> list[VerifierExample]:
    """Same dataset/filtering/shuffle/select/student-prompt as
    `tropic.data.load_opsd_math_examples` - the only difference is also
    keeping (and requiring) the dataset's own `Answer` column, since RLSD/SDPO
    are meaningless without a gold answer to grade rollouts against. Rows
    with a missing/empty `Answer` are dropped."""
    # Offline deployment: same TROPIC_TRAIN_DATA_PATH override as
    # tropic.data.load_opsd_math_examples (see that function's own comment) -
    # this loader has its OWN load_dataset() call site, so the override has
    # to be duplicated here rather than inherited automatically.
    dataset_path = os.environ.get("TROPIC_TRAIN_DATA_PATH", OPSD_DATASET_PATH)
    ds = load_dataset(dataset_path, split=split)
    if only_correct and "correct" in ds.column_names:
        ds = ds.filter(lambda r: bool(r["correct"]))
    if "Answer" not in ds.column_names:
        raise ValueError(f"{dataset_path!r} has no 'Answer' column - cannot compute a verifier reward")
    ds = ds.filter(lambda r: r["Answer"] is not None and str(r["Answer"]).strip() != "")
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))

    examples: list[VerifierExample] = []
    for row in ds:
        question = row["problem"]
        student_ids = build_student_prompt(
            tokenizer, question, enable_thinking=student_thinking, max_length=max_length,
            empty_think_suffix=empty_think_suffix,
        )
        examples.append(
            VerifierExample(question=question, answer=str(row["Answer"]), student_prompt_ids=student_ids)
        )
    return examples
