"""P-ALIGN's answer grader, ported from P-ALIGN/src/evaluation.py.

grade_math_verify() is P-ALIGN's current evaluation.py: math_verify only. grade() is the
earlier variant, correct when EITHER math_verify OR oat_math_grader accepts it (the old
`any_true=True` default) -- more lenient, so its numbers are not comparable to P-ALIGN's.
grade() reports which branches are live instead of silently degrading.
"""

import os
import signal
from contextlib import contextmanager

VERIFY_TIMEOUT_SECONDS = 10


@contextmanager
def _time_limit(seconds: int):
    """Abort a hanging sympy verification. POSIX-only, as in P-ALIGN's own decorator."""
    if os.name != "posix":
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("verification timed out")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _math_verify_labels(predictions: list[str], gold: str) -> list[int]:
    from math_verify import parse, verify

    try:
        with _time_limit(VERIFY_TIMEOUT_SECONDS):
            parsed_gold = parse("$" + gold + "$")  # P-ALIGN wraps gold in $..$ before parsing
            return [int(bool(verify(parsed_gold, parse(prediction)))) for prediction in predictions]
    except Exception:
        return [0] * len(predictions)


def _oat_label(prediction: str, gold: str) -> int:
    from sgl.eval.graders.oat import boxed_reward_fn

    try:
        _, result = boxed_reward_fn(prediction, gold, fast=False)
        return int(result == 1.0)
    except Exception:
        return 0


def active_graders() -> list[str]:
    """Grader branches whose imports resolve in this environment."""
    available = []
    for name, module in (("math_verify", "math_verify"), ("oat_math_grader", "sgl.eval.graders.oat")):
        try:
            __import__(module)
        except ImportError:
            continue
        available.append(name)
    return available


def grade_math_verify(predictions: list[str], gold: str) -> list[int]:
    """Per-generation 0/1 labels exactly as P-ALIGN/src/evaluation.py assigns them."""
    if "math_verify" not in active_graders():
        raise ImportError("math_verify grading needs the math-verify package installed.")
    return _math_verify_labels(predictions, gold)


def grade(predictions: list[str], gold: str) -> list[int]:
    """Per-generation 0/1 labels for one problem. Raises if no grader is installed."""
    available = active_graders()
    if not available:
        raise ImportError(
            "P-ALIGN grading needs math_verify (and optionally oat_math_grader); install "
            "math-verify, or pass --grader builtin to use this repo's own scorer."
        )

    labels = (
        _math_verify_labels(predictions, gold)
        if "math_verify" in available
        else [0] * len(predictions)
    )
    if "oat_math_grader" in available:
        labels = [
            int(label or _oat_label(prediction, gold))
            for label, prediction in zip(labels, predictions)
        ]
    return labels
