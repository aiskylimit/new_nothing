"""Public metric registry backed by the behavior-compatible CypherKD ports."""

from __future__ import annotations

from typing import Any, Protocol

from .executable import executable
from .execution_accuracy import execution_accuracy
from .provenance_subgraph_jaccard_similarity import (
    get_ps_cypher,
    provenance_subgraph_jaccard_similarity,
)


class QueryRunner(Protocol):
    def run_query(self, cypher: str, *, timeout: int | None = None, **parameters: Any) -> list[dict[str, Any]]: ...


METRICS = {
    "execution_accuracy": execution_accuracy,
    "psjs": provenance_subgraph_jaccard_similarity,
    "executable": executable,
}

__all__ = [
    "METRICS",
    "QueryRunner",
    "executable",
    "execution_accuracy",
    "get_ps_cypher",
    "provenance_subgraph_jaccard_similarity",
]
