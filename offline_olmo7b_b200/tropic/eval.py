"""Evaluation on OPSD's own benchmark suite: AIME24, AIME25, HMMT25.

Dataset paths verified live against the actual OPSD eval script
(github.com/siyan-zhao/OPSD/blob/main/eval/evaluate_math.py):
    AIME24  -> HuggingFaceH4/aime_2024      (problem, solution, answer, url, year)
    AIME25  -> yentinglin/aime_2025         (problem, solution, answer, url, year)
    HMMT25  -> MathArena/hmmt_feb_2025      (problem, answer, problem_type)

CORRECTED (an earlier version of this file assumed all three benchmarks give
a bare integer `answer`, so a normalized string/int compare would "collapse
to exactly" OPSD's own math_verify-based grading - checked live against the
real HMMT25 data and this assumption is FALSE: roughly half of HMMT25's gold
answers are non-integer LaTeX expressions, e.g. `\\frac{1}{576}`,
`8\\sqrt{10}`, `14+4\\sqrt{37}`, `2^{25} \\cdot 26!`. A plain string/int
compare produces false negatives whenever a correct answer is written in a
mathematically-equivalent but textually different form (e.g. `1/576` vs
`\\frac{1}{576}`), silently under-scoring HMMT25 specifically. AIME24/AIME25
answers ARE always integers 0-999 by competition rule, so they were never
affected.) Now uses `math_verify` (the exact package OPSD's own
eval/evaluate_math.py uses: `from math_verify import parse, verify`) for
symbolic/LaTeX-aware equivalence checking, with the same string-normalize
fallback OPSD's own `grade_answer` uses if math_verify itself raises.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from datasets import load_dataset

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    from math_verify.errors import TimeoutException as _MathVerifyTimeoutException
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False

    class _MathVerifyTimeoutException(BaseException):
        """Placeholder when math_verify isn't installed - never actually
        raised (grade() only reaches the math_verify call path when
        _HAS_MATH_VERIFY is True); exists so the `except` clause below has a
        name to reference either way."""

BENCHMARKS = {
    "aime24": ("HuggingFaceH4/aime_2024", "train"),
    "aime25": ("yentinglin/aime_2025", "train"),
    "hmmt25": ("MathArena/hmmt_feb_2025", "train"),
    # Same MathArena schema as hmmt25 (problem/answer/problem_idx) - no
    # loader changes needed. Not part of OPSD's own reported benchmark
    # suite (AIME 2026 postdates that paper); added on request to try it.
    "aime26": ("MathArena/aime_2026", "train"),
}


@dataclass
class EvalProblem:
    benchmark: str
    problem_id: str
    problem: str
    answer: str


def load_benchmark(name: str, num_problems: int | None = None) -> list[EvalProblem]:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}, expected one of {list(BENCHMARKS)}")
    path, split = BENCHMARKS[name]
    # Offline deployment: TROPIC_EVAL_DATA_DIR, when set, points at a local
    # directory with one subfolder per benchmark name (each a raw download
    # of that benchmark's repo, e.g. via huggingface_hub.snapshot_download
    # onto a no-internet server) - falls back to the plain repo id (normal
    # online lookup) when unset, so this is a no-op for every other caller.
    eval_data_dir = os.environ.get("TROPIC_EVAL_DATA_DIR")
    if eval_data_dir:
        path = f"{eval_data_dir}/{name}"
    ds = load_dataset(path, split=split)
    if num_problems is not None:
        ds = ds.select(range(min(num_problems, len(ds))))
    problems = []
    for i, row in enumerate(ds):
        pid = str(row.get("id") or row.get("problem_idx") or i)
        problems.append(EvalProblem(benchmark=name, problem_id=pid, problem=row["problem"], answer=str(row["answer"])))
    return problems


_BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed_answer(text: str) -> str | None:
    """Find the LAST \\boxed{...} in `text` and return its (brace-balanced) content.

    Mirrors OPSD's own extract_boxed_answer: models sometimes emit multiple
    boxed spans while thinking out loud, and the final one is the answer.
    """
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unbalanced braces, e.g. truncated generation
    return text[start : i - 1].strip()


def _normalize_answer(ans: str) -> str:
    return ans.replace("$", "").replace(" ", "").replace("\\!", "").replace(",", "").strip().lstrip("+")


def _string_fallback_grade(pred: str, gold: str) -> bool:
    """OPSD's own fallback (`grade_answer`'s except branch): lowercase,
    strip `$`/spaces, compare as strings - used only when math_verify itself
    raises, or isn't installed at all."""
    pred_norm = pred.replace("$", "").replace(" ", "").lower().strip()
    gold_norm = gold.replace("$", "").replace(" ", "").lower().strip()
    if pred_norm == gold_norm:
        return True
    try:
        return int(_normalize_answer(pred)) == int(_normalize_answer(gold))
    except ValueError:
        return False


def grade(predicted_text: str, gold_answer: str) -> bool:
    """True iff the LAST \\boxed{...} in `predicted_text` is mathematically
    equivalent to `gold_answer`.

    Mirrors OPSD's own eval/evaluate_math.py `grade_answer` exactly: parse
    both sides with `math_verify` (wrapping bare expressions in `$...$` for
    LaTeX parsing) and check symbolic equivalence - this correctly handles
    HMMT25's non-integer answers (fractions, radicals, factorials - see
    module docstring), where a plain string/int compare would under-score
    correct-but-differently-formatted answers. Falls back to a normalized
    string/int compare only if math_verify raises or isn't installed.
    """
    predicted = extract_boxed_answer(predicted_text)
    if predicted is None:
        return False

    if _HAS_MATH_VERIFY:
        try:
            pred_wrapped = predicted if "$" in predicted else f"${predicted}$"
            gold_wrapped = gold_answer if "$" in gold_answer else f"${gold_answer}$"
            pred_parsed = _mv_parse(pred_wrapped, fallback_mode="no_fallback")
            gold_parsed = _mv_parse(gold_wrapped, fallback_mode="no_fallback")
            # raise_on_error=True: math_verify's own default (False) SWALLOWS
            # internal errors and returns a plain False instead of raising -
            # indistinguishable from "genuinely not equivalent" and defeats
            # our except-fallback below entirely (verified live: on an
            # environment where verify()'s internal multiprocessing timeout
            # mechanism itself fails, it silently returns False rather than
            # erroring). Forcing True makes that a catchable exception instead.
            return bool(_mv_verify(gold_parsed, pred_parsed, timeout_seconds=5, raise_on_error=True))
        except (Exception, _MathVerifyTimeoutException):
            # math_verify.errors.TimeoutException inherits from BaseException,
            # NOT Exception (verified live: `TimeoutException.__mro__` ==
            # (TimeoutException, BaseException, object)) - a bare `except
            # Exception:` does NOT catch it, so a single pathologically slow
            # comparison (a real crash hit mid-training, see chat history)
            # takes down the whole process instead of just failing this one
            # grade() call. Caught explicitly here alongside Exception.
            return _string_fallback_grade(predicted, gold_answer)
    return _string_fallback_grade(predicted, gold_answer)


@dataclass
class BenchmarkResult:
    benchmark: str
    num_problems: int
    k: int
    avg_at_k: float  # OPSD's Average@N: total correct / total generations
    pass_at_k: float  # fraction of problems with >=1 correct sample
    per_problem_correct_fraction: list[float] = field(default_factory=list)


def score_generations(benchmark: str, generations: list[list[str]], golds: list[str]) -> BenchmarkResult:
    """generations[i] = list of k sampled completions for problem i."""
    assert len(generations) == len(golds)
    k = len(generations[0]) if generations else 0
    total_correct = 0
    total_gen = 0
    pass_count = 0
    per_problem = []
    for gens, gold in zip(generations, golds):
        correct_flags = [grade(g, gold) for g in gens]
        n_correct = sum(correct_flags)
        total_correct += n_correct
        total_gen += len(gens)
        pass_count += int(n_correct > 0)
        per_problem.append(n_correct / len(gens) if gens else 0.0)
    return BenchmarkResult(
        benchmark=benchmark,
        num_problems=len(generations),
        k=k,
        avg_at_k=total_correct / total_gen if total_gen else 0.0,
        pass_at_k=pass_count / len(generations) if generations else 0.0,
        per_problem_correct_fraction=per_problem,
    )
