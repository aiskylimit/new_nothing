from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .metrics import QueryRunner

MetricResult = float | dict[str, str]
MetricFn = Callable[..., float]

VALID_METRICS = ("execution_accuracy", "psjs", "executable")
DEFAULT_METRICS = ("execution_accuracy", "psjs", "executable")
END_OF_TURN = "<end_of_turn>"


def clean_pred_cypher(pred_cypher: str | None) -> str:
    """Apply exactly the output cleanup used by CypherKD evaluation."""

    pred_cypher = pred_cypher or ""
    if pred_cypher.endswith(END_OF_TURN):
        return pred_cypher[: -len(END_OF_TURN)].strip()
    return pred_cypher


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read the project's JSONL output or CypherKD's JSON-array output."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array or JSONL objects in {path}")
    return rows


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def avg_and_round(nums: Iterable[float], n: int = 4) -> float:
    nums = list(nums)
    return round(sum(nums) / len(nums), n) if nums else math.nan


def metric_value(value: Any) -> float:
    return value if isinstance(value, int | float) else 0.0


def _get_metrics(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("metrics", record.get("cypher_metrics", {}))


def calculate_aggregates(
    result: Iterable[dict[str, Any]],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, dict[str, float]]:
    """Aggregate with CypherKD's missing/error handling and four-digit rounding."""

    records = list(result)
    return {
        "overall": {
            metric: avg_and_round(
                metric_value(_get_metrics(record).get(metric))
                for record in records
                if metric in _get_metrics(record)
            )
            for metric in metrics
        }
    }


# Compatibility name used by merge.py and callers in this repository.
aggregate_scores = calculate_aggregates


@lru_cache(maxsize=1)
def get_metric_functions() -> dict[str, MetricFn]:
    from .metrics import executable, execution_accuracy, provenance_subgraph_jaccard_similarity

    return {
        "execution_accuracy": execution_accuracy,
        "psjs": provenance_subgraph_jaccard_similarity,
        "executable": executable,
    }


def safe_compute(
    metric_name: str,
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: QueryRunner,
    *,
    timeout: int = 120,
) -> MetricResult:
    try:
        return get_metric_functions()[metric_name](
            pred_cypher=pred_cypher,
            target_cypher=target_cypher,
            neo4j_connector=neo4j_connector,
            timeout=timeout,
        )
    except Exception as error:
        return {"error": f"{metric_name} failed: {error}"}


def score_records(
    records: Iterable[dict[str, Any]],
    connector: QueryRunner,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    predicted_key: str = "predicted_cypher",
    target_key: str = "reference_cypher",
    timeout: int = 120,
    desc: str = "Evaluating Cypher",
) -> list[dict[str, Any]]:
    """Adapt project records to CypherKD scoring without changing metric inputs."""

    unknown = set(metrics).difference(VALID_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    output_records = []
    for item in tqdm(list(records), desc=desc):
        pred_cypher = clean_pred_cypher(item.get(predicted_key, ""))
        target_cypher = item.get(target_key, "")
        output_records.append(
            {
                **item,
                predicted_key: pred_cypher,
                "metrics": {
                    metric: safe_compute(
                        metric,
                        pred_cypher,
                        target_cypher,
                        connector,
                        timeout=timeout,
                    )
                    for metric in metrics
                },
            }
        )
    return output_records


__all__ = [
    "DEFAULT_METRICS",
    "VALID_METRICS",
    "aggregate_scores",
    "avg_and_round",
    "calculate_aggregates",
    "clean_pred_cypher",
    "metric_value",
    "read_records",
    "safe_compute",
    "score_records",
    "write_jsonl",
]
