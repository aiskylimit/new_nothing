"""OPSD-style dual-context data pipelines.

Two loaders are provided:
  - load_gsm8k_examples:      cheap, CPU-friendly, used by the Phase 0-8
                               smoke test on Qwen2.5-0.5B-Instruct.
  - load_opsd_math_examples:  the REAL training data OPSD itself trains on
                               (verified live against opsd_train.py:
                               `load_dataset("siyanzhao/Openthoughts_math_30k_opsd")`),
                               used for the GPU-scale TROPIC-vs-OPSD run on
                               Qwen3-1.7B.

In both cases: the student only ever sees the question. The teacher
additionally sees the reference solution as privileged context, following
OPSD ("the student sees only the problem, while the teacher additionally
sees the ground-truth solution").

The student/teacher prompt WORDING below is copied verbatim from OPSD's own
`data_collator.py` (`SelfDistillationDataCollator.__call__`, fetched live from
github.com/siyan-zhao/OPSD on 2026-09-02) rather than a paraphrase - a fair
comparison needs the two methods to see byte-identical prompts, and OPSD's
own reported numbers were produced with this exact wording.

Qwen3 is a "thinking" model: its chat template accepts `enable_thinking`
(True leaves `<think>...</think>` open for the model to fill in; False
pre-closes it with an empty block). OPSD's own defaults are
`teacher_thinking=True, student_thinking=False` (opsd_train.py
CustomScriptArguments) - the teacher reasons at length over the privileged
solution, the student is graded on its short-form answer. Qwen2.5 (used by
the earlier CPU smoke test) has no such template branch, so `enable_thinking`
is simply omitted there.

OLMo-3-7B-Think (allenai/Olmo-3-7B-Think) is a DIFFERENT case, verified live
against its real tokenizer on 2026-09-15: unlike Qwen3, its chat template
has NO `enable_thinking` branch at all - passing `enable_thinking=False` is
silently ignored (confirmed: rendered text is byte-identical with or
without it). Its template's OWN `add_generation_prompt=True` output already
ends with an OPENED `<think>` tag (not closed), so the model always starts
reasoning. To get a non-thinking rollout anyway, this file uses the
community "empty think prefill" trick already documented for DeepSeek-R1-
style reasoning models (see e.g. huggingface.co/deepseek-ai/DeepSeek-R1-
Distill-Qwen-14B/discussions/11): manually append a closing `</think>` (with
blank content) right after the template's own auto-opened `<think>`, so the
model sees an ALREADY-CLOSED, empty reasoning block and proceeds straight to
the final answer instead of generating one. This is `empty_think_suffix`
below - UNVERIFIED end-to-end (confirmed only that the template ignores
`enable_thinking` and ends on an open `<think>`; NOT confirmed that OLMo-3-
7B-Think actually skips reasoning after seeing the closed empty block the
way DeepSeek-R1 does - that pattern is well-documented for DeepSeek-R1 but
OLMo-3-Think is a different model/training recipe). Test on a real
checkpoint (inspect a few actual generations) before trusting this for a
real training run - see CONFIG's own comment in whichever script sets
`empty_think_suffix` for OLMo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

OPSD_DATASET_PATH = "siyanzhao/Openthoughts_math_30k_opsd"

# Verbatim from OPSD's data_collator.py (SelfDistillationDataCollator).
_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


@dataclass
class TropicExample:
    question: str
    reference_solution: str
    student_prompt_ids: torch.Tensor  # [Ls]
    teacher_prefix_ids: torch.Tensor  # [Lt], Lt != Ls in general


def _apply_chat_template_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """Mirrors OPSD's own two-step collator process exactly: render the chat
    template to a STRING (`tokenize=False`), then tokenize that string with a
    separate `tokenizer(...)` call - rather than tokenizing directly inside
    `apply_chat_template` - so that `max_length` truncation (OPSD's own
    `--max_length 20000`, a TOKEN cap on the whole rendered prompt, not a
    character cap on the solution alone) is applied identically to how their
    `SelfDistillationDataCollator.__call__` does it. `enable_thinking` is
    forwarded only when given explicitly (Qwen3); omitted for non-thinking
    models (Qwen2.5, used by the GSM8K CPU smoke test) AND for models with no
    native `enable_thinking` template branch at all (OLMo-3-Think - passing
    it there is a silent no-op, verified live, so it is never forwarded as a
    kwarg when `empty_think_suffix` is given; see this module's own
    docstring for why).

    `empty_think_suffix` (default None): for models like OLMo-3-Think whose
    template has NO real `enable_thinking` toggle but DOES auto-open a
    `<think>` tag at the end of `add_generation_prompt=True` output - when
    `enable_thinking is False`, this string (e.g. `"\\n\\n</think>\\n\\n"`) is
    appended directly to the rendered text, closing that already-open tag
    with empty content (the community "empty think prefill" trick - see
    module docstring). Ignored when `enable_thinking` is True/None (thinking
    stays on, nothing appended) or when this arg itself is None (Qwen3/
    Qwen2.5 path, unchanged from before)."""
    if empty_think_suffix is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if enable_thinking is False:
            text += empty_think_suffix
    else:
        kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = tokenizer(text, truncation=max_length is not None, max_length=max_length, return_tensors="pt")
    return encoded["input_ids"].squeeze(0)


def build_student_prompt(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """Verbatim wording from OPSD's data_collator.py: `f"Problem: {problem}\\n\\n{_BOXED_INSTRUCTION}"`
    as a single user message. Note this differs from OPSD's own EVAL prompt
    (no "Problem: " prefix there - see the eval helper's docstring) - that
    mismatch exists in OPSD's own code too, not a bug introduced here."""
    content = f"Problem: {question}\n\n{_BOXED_INSTRUCTION}"
    messages = [{"role": "user", "content": content}]
    return _apply_chat_template_ids(tokenizer, messages, enable_thinking, max_length, empty_think_suffix)


def build_teacher_prefix(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    reference_solution: str,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """Verbatim wording from OPSD's data_collator.py (non-`reason_first` branch,
    the one every real run script uses): a SINGLE user message (no system
    message) containing the problem, the delimited reference solution, the
    transition prompt, then the same boxed-answer instruction as the student."""
    content = (
        f"Problem: {question}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{reference_solution}\n=== Reference Solution End ===\n"
        f"{_TRANSITION_PROMPT}\n"
        f"{_BOXED_INSTRUCTION}"
    )
    messages = [{"role": "user", "content": content}]
    return _apply_chat_template_ids(tokenizer, messages, enable_thinking, max_length, empty_think_suffix)


def build_teacher_context_only_prefix(
    tokenizer: PreTrainedTokenizerBase,
    reference_solution: str,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """TROPIC-**G** only (alpha>0): Eq. 11's `z_theta(.|empty,o<t,r)` - the
    SAME teacher template as `build_teacher_prefix`, with the question `x`
    replaced by an empty string (the reference solution `r` is still
    present) - "empty" in Eq. 11 refers to the question, not the whole
    context. Reuses `build_teacher_prefix` verbatim (`question=""`) rather
    than duplicating the template, so the two prompts stay byte-identical
    apart from that one substitution. TROPIC-P (alpha=0) never calls this."""
    return build_teacher_prefix(tokenizer, "", reference_solution, enable_thinking, max_length, empty_think_suffix)


# ---------------------------------------------------------------------------
# GSM8K - cheap CPU smoke-test data (Phase 0-8, Qwen2.5-0.5B-Instruct).
# ---------------------------------------------------------------------------


def load_gsm8k_examples(
    tokenizer: PreTrainedTokenizerBase, num_examples: int, split: str = "train", seed: int = 0
) -> list[TropicExample]:
    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))

    examples: list[TropicExample] = []
    for row in ds:
        question = row["question"]
        reference_solution = row["answer"]
        student_ids = build_student_prompt(tokenizer, question)
        teacher_ids = build_teacher_prefix(tokenizer, question, reference_solution)
        examples.append(
            TropicExample(
                question=question,
                reference_solution=reference_solution,
                student_prompt_ids=student_ids,
                teacher_prefix_ids=teacher_ids,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# OPSD's real training set - siyanzhao/Openthoughts_math_30k_opsd (GPU run).
# ---------------------------------------------------------------------------


def load_opsd_math_examples(
    tokenizer: PreTrainedTokenizerBase,
    num_examples: int,
    split: str = "train",
    seed: int = 0,
    only_correct: bool = True,
    teacher_thinking: bool = True,
    student_thinking: bool = False,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> list[TropicExample]:
    """Loads OPSD's own training data. Columns verified live:
    ['source','problem','solution','messages','system','conversations',
     'generated_token_count','correct','Question','COT_Reason','Answer'].
    We use `problem` (question) and `solution` (the long, already-verified
    chain-of-thought reference) as the privileged context, and by default
    keep only rows where `correct == True` (the dataset's own signal that
    the recorded reasoning chain actually reached the right answer) - not
    filtering this would let the teacher condition on a demonstrably wrong
    demonstration.

    `max_length`, if set, truncates the RENDERED prompt (chat template +
    instructions, not just the solution text) to this many TOKENS - this
    mirrors OPSD's own `--max_length 20000` (`SelfDistillationDataCollator`
    tokenizes the whole prompt string with `truncation=True, max_length=...`,
    not a character-count trim of the solution alone; see `build_teacher_prefix`).
    Set small for a fast CPU dry run (see notebooks/TROPIC_vs_OPSD.ipynb's
    DRY_RUN flag); OPSD's own real run uses 20000.
    """
    # Offline deployment: same TROPIC_TRAIN_DATA_PATH override as
    # tropic/verifier_data.py's load_opsd_math_examples_with_answer (that
    # loader has its OWN load_dataset() call site, so the override must be
    # duplicated here too) - falls back to the real HF repo id when unset
    # (online cluster behavior, unchanged).
    dataset_path = os.environ.get("TROPIC_TRAIN_DATA_PATH", OPSD_DATASET_PATH)
    ds = load_dataset(dataset_path, split=split)
    if only_correct and "correct" in ds.column_names:
        ds = ds.filter(lambda r: bool(r["correct"]))
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))

    examples: list[TropicExample] = []
    for row in ds:
        question = row["problem"]
        reference_solution = row["solution"]
        student_ids = build_student_prompt(
            tokenizer, question, enable_thinking=student_thinking, max_length=max_length,
            empty_think_suffix=empty_think_suffix,
        )
        teacher_ids = build_teacher_prefix(
            tokenizer, question, reference_solution, enable_thinking=teacher_thinking, max_length=max_length,
            empty_think_suffix=empty_think_suffix,
        )
        examples.append(
            TropicExample(
                question=question,
                reference_solution=reference_solution,
                student_prompt_ids=student_ids,
                teacher_prefix_ids=teacher_ids,
            )
        )
    return examples


def debug_print_examples(tokenizer: PreTrainedTokenizerBase, examples: list[TropicExample], n: int = 2) -> None:
    for ex in examples[:n]:
        print("=" * 80)
        print("QUESTION:", ex.question)
        print("-" * 80)
        print("STUDENT PROMPT (decoded):")
        print(tokenizer.decode(ex.student_prompt_ids))
        print("-" * 80)
        print("TEACHER PREFIX (decoded):")
        print(tokenizer.decode(ex.teacher_prefix_ids))
        assert ex.reference_solution.split("\n")[0][:20] not in tokenizer.decode(
            ex.student_prompt_ids
        ), "reference solution must not leak into the student prompt"
