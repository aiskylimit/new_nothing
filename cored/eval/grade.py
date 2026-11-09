"""Chấm đáp án bằng ``math_verify`` (spec §8), có fallback string-match.

Pass@1 ở đây = accuracy trung bình trên k sample (spec §8: AIME lấy k=4 rồi
trung bình). Đây là ước lượng không chệch của xác suất một sample đúng, đúng
định nghĩa Pass@1 dùng trong các bài R1-distill.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

_BOXED = re.compile(r"\\boxed\s*{")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

try:  # math_verify là dependency chính; fallback chỉ để chạy được khi thiếu
    from math_verify import parse as mv_parse, verify as mv_verify

    HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover
    HAS_MATH_VERIFY = False


def extract_boxed(text: str) -> Optional[str]:
    """Lấy nội dung ``\\boxed{...}`` cuối cùng, khớp ngoặc lồng nhau."""
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, out = 1, []
    for char in text[start:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(char)
    return "".join(out).strip() or None


def normalize(value: str) -> str:
    value = value.strip().rstrip(".").replace(" ", "").replace(",", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "").replace("$", "")
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    if value.endswith("%"):
        value = value[:-1]
    try:
        number = float(value)
        import math
        if math.isinf(number) or math.isnan(number):
            return value
        return str(int(number)) if number == int(number) else str(number)
    except (ValueError, OverflowError):
        return value


def grade_one(prediction: str, gold: str) -> bool:
    """True nếu prediction khớp gold."""
    if HAS_MATH_VERIFY:
        try:
            gold_parsed = mv_parse(gold if "\\boxed" in gold else f"\\boxed{{{gold}}}")
            pred_parsed = mv_parse(prediction)
            if gold_parsed and pred_parsed and mv_verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass  # rơi xuống fallback
    pred_boxed = extract_boxed(prediction)
    if pred_boxed is None:
        numbers = _NUMBER.findall(prediction)
        pred_boxed = numbers[-1] if numbers else None
    if pred_boxed is None:
        return False
    gold_boxed = extract_boxed(gold) or gold
    return normalize(pred_boxed) == normalize(gold_boxed)


_MC_BOXED_LETTER = re.compile(r"\\boxed\s*{\s*\(?\s*([A-Ja-j])\s*\)?\s*}")
_MC_TAIL_LETTER = re.compile(r"\b(?:answer|Answer)\b[^A-Ja-j]{0,20}\(?([A-Ja-j])\)?\b")


def grade_multiple_choice(prediction: str, gold: str) -> bool:
    """Trắc nghiệm (GPQA / ARC-C / MMLU-Pro): gold là một chữ cái.

    Ưu tiên ``\\boxed{LETTER}`` (đúng format prompt yêu cầu), rồi tới chữ cái sau
    "answer", cuối cùng là chữ cái đứng riêng cuối văn bản.
    """
    gold_letter = gold.strip().strip("()").upper()[:1]
    if not gold_letter.isalpha():
        return grade_one(prediction, gold)
    m = _MC_BOXED_LETTER.search(prediction)
    if m:
        return m.group(1).upper() == gold_letter
    boxed = extract_boxed(prediction)
    if boxed:
        b = boxed.strip().strip("()").upper()
        if len(b) == 1 and b.isalpha():
            return b == gold_letter
    m = _MC_TAIL_LETTER.search(prediction)
    if m:
        return m.group(1).upper() == gold_letter
    tail = re.findall(r"\b([A-J])\b", prediction[-200:])
    return bool(tail) and tail[-1] == gold_letter


def pass_at_1(predictions: Sequence[str], gold: str, kind: str = "math") -> float:
    """Trung bình accuracy trên k sample của cùng một bài."""
    if not predictions:
        return 0.0
    grader = grade_multiple_choice if kind == "mc" else grade_one
    return sum(float(grader(p, gold)) for p in predictions) / len(predictions)


def aggregate(scores: List[float]) -> float:
    return float(sum(scores) / len(scores)) if scores else 0.0

def pass_at_k(predictions: Sequence[str], gold: str, kind: str = "math") -> float:
    """1.0 nếu ít nhất 1 sample đúng (Pass@k), ngược lại 0.0."""
    if not predictions:
        return 0.0
    grader = grade_multiple_choice if kind == "mc" else grade_one
    return 1.0 if any(grader(p, gold) for p in predictions) else 0.0
