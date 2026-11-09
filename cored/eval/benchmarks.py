"""Nạp benchmark eval và trích ground-truth (spec §3.1, §8).

Hai họ benchmark:
  - Main math benchmarks use the same shared local snapshots as the baselines.
  - **Kịch bản CurioSFT** (``cs_*``): đọc parquet local từ ``Hao0oWang/CurioSFT_Data``
    — đúng tập test bài dùng. ``test.math.parquet`` chứa 6 bench math (tách theo
    cột ``data_source``); OOD là 3 parquet riêng (trắc nghiệm, gold = chữ cái).
    Set ``--curiosft_data_dir`` (mặc định ``data/curiosft_raw``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.local_assets import asset_path, load_local_rows

BENCHMARKS: Dict[str, dict] = {
    "gsm8k": {"path": "datasets/gsm8k/main", "split": "test", "k": 1},
    "math500": {"path": "../palign/datasets/MATH-500", "split": "test", "k": 3},
    "aime24": {"path": "../palign/datasets/AIME_2024/aime_2024_problems.parquet", "split": "train", "k": 3},
    "aime25": {"path": "../palign/datasets/aime_2025", "split": "train", "k": 3},
    # Cùng snapshot AMC 83 bài với các baseline, thay đường dẫn repo cũ.
    "amc": {"path": "../palign/datasets/aimo-validation-amc", "split": "train", "k": 3},
}

# Prefix mà P-ALIGN gắn vào mọi câu hỏi (src/test.py). Ta strip ra khỏi các file
# local rồi để render_prompt / run_eval gắn lại đồng nhất cho mọi benchmark.
_PALIGN_PREFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# --- CurioSFT: parquet + filter data_source. k=8 mọi bench trừ mmlu_pro (n=1). ---
CURIOSFT_BENCHMARKS: Dict[str, dict] = {
    "cs_math500":  {"file": "test.math.parquet", "data_source": "math",          "k": 8, "kind": "math"},
    "cs_amc":      {"file": "test.math.parquet", "data_source": "amc",           "k": 8, "kind": "math"},
    "cs_aime24":   {"file": "test.math.parquet", "data_source": "aime",          "k": 8, "kind": "math"},
    "cs_aime25":   {"file": "test.math.parquet", "data_source": "aime25",        "k": 8, "kind": "math"},
    "cs_minerva":  {"file": "test.math.parquet", "data_source": "minerva",       "k": 8, "kind": "math"},
    "cs_olympiad": {"file": "test.math.parquet", "data_source": "olympiad_bench", "k": 8, "kind": "math"},
    "cs_gpqa":     {"file": "test.gpqa.parquet",     "data_source": None, "k": 8, "kind": "mc"},
    "cs_arc_c":    {"file": "test.arc_c.parquet",    "data_source": None, "k": 8, "kind": "mc"},
    "cs_mmlu_pro": {"file": "test.mmlu_pro.parquet", "data_source": None, "k": 1, "kind": "mc"},
}
BENCHMARKS.update(CURIOSFT_BENCHMARKS)

_GSM8K_ANSWER = re.compile(r"####\s*(.+)\s*$")


@dataclass
class EvalItem:
    index: int
    question: str
    answer: str
    benchmark: str
    kind: str = "math"  # "math" | "mc" (multiple choice, gold = chữ cái)


def load_benchmark(
    name: str, limit: Optional[int] = None, curiosft_data_dir: str = "data/curiosft_raw"
) -> List[EvalItem]:
    if name not in BENCHMARKS:
        raise ValueError(f"benchmark chưa khai báo: {name}")
    spec = BENCHMARKS[name]

    if spec.get("local"):
        return _load_local_jsonl_benchmark(name, spec, limit)

    if name in CURIOSFT_BENCHMARKS:
        return _load_curiosft_benchmark(name, spec, curiosft_data_dir, limit)

    if name == "aime25":
        dataset = _load_aime25()
    else:
        dataset = load_local_rows(spec["path"], split=spec["split"])

    items: List[EvalItem] = []
    for index, row in enumerate(dataset):
        question = _get_question(row)
        answer = _get_answer(name, row)
        if name == "amc" and answer is not None and answer.endswith(".0"):
            answer = answer[:-2]
        if not question or answer is None:
            continue
        items.append(EvalItem(index=index, question=question, answer=str(answer),
                              benchmark=name))
        if limit and len(items) >= limit:
            break
    return items


def _load_local_jsonl_benchmark(name, spec, limit) -> List[EvalItem]:
    """Đọc file jsonl local (định dạng output của P-ALIGN: ``prompt_ori`` + ``answer``).

    ``prompt_ori`` đã kèm sẵn prefix "Please reason step by step ..." — ta strip
    ra để lấy câu hỏi thô, render_prompt/run_eval sẽ gắn lại nhất quán.
    """
    import json as _json

    path = spec["local"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: thiếu file local {path}")

    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            question = str(row.get("prompt_ori") or row.get("problem")
                           or row.get("question") or "")
            if question.startswith(_PALIGN_PREFIX):
                question = question[len(_PALIGN_PREFIX):].lstrip()
            gold = row.get("answer")
            if gold is None or gold == "" or not question:
                continue
            gold = str(gold)
            # answer kiểu "142.0" -> "142" cho khớp gold số nguyên
            if gold.endswith(".0"):
                gold = gold[:-2]
            items.append(EvalItem(index=index, question=question, answer=gold,
                                  benchmark=name, kind="math"))
            if limit and len(items) >= limit:
                break
    if not items:
        raise RuntimeError(f"{name}: 0 bài từ {path}")
    return items


def _load_curiosft_benchmark(name, spec, data_dir, limit) -> List[EvalItem]:
    """Đọc parquet CurioSFT_Data. Câu hỏi = ``prompt[0].content`` (đã kèm hướng
    dẫn ``\\boxed`` cho bench trắc nghiệm), gold = ``reward_model.ground_truth``.
    """
    import pyarrow.parquet as pq

    path = str(asset_path(data_dir) / spec["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Optional CurioSFT benchmark missing: {path}. Supply these legacy local files explicitly; "
            "cored/download.txt provisions the shared baseline benchmarks."
        )
    table = pq.read_table(path)
    cols = table.column_names
    rows = table.to_pylist()
    items: List[EvalItem] = []
    for index, row in enumerate(rows):
        if spec["data_source"] is not None and row.get("data_source") != spec["data_source"]:
            continue
        prompt = row.get("prompt") or []
        if not prompt:
            continue
        question = prompt[0]["content"]
        rm = row.get("reward_model") or {}
        gold = rm.get("ground_truth")
        if not question or gold in (None, ""):
            continue
        items.append(EvalItem(index=index, question=str(question), answer=str(gold),
                              benchmark=name, kind=spec["kind"]))
        if limit and len(items) >= limit:
            break
    if not items:
        raise RuntimeError(f"{name}: 0 bài (data_source={spec['data_source']}, cols={cols})")
    return items


def _load_aime25():
    """Read the same yentinglin/aime_2025 train snapshot used by the baselines."""
    root = asset_path(BENCHMARKS["aime25"]["path"])
    rows = load_local_rows(root, split=BENCHMARKS["aime25"]["split"])
    questions = [_get_question(row) for row in rows]
    if len(rows) != 30 or len(set(questions)) != 30 or not all(questions):
        raise ValueError(f"AIME2025 requires 30 unique questions from the shared baseline snapshot: {root}")
    return rows


def _get_question(row) -> str:
    for key in ("problem", "Problem", "question", "Question", "query"):
        if key in row and row[key]:
            return str(row[key])
    return ""


def _get_answer(name: str, row) -> Optional[str]:
    if name == "gsm8k":
        match = _GSM8K_ANSWER.search(str(row.get("answer", "")))
        return match.group(1).replace(",", "").strip() if match else None
    for key in ("answer", "Answer", "solution", "expected_answer"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return None
