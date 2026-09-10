from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MergeResult:
    sub_schema: dict[str, list[dict[str, Any]]]
    directly_selected_unit_ids: tuple[str, ...]
    closure_added_node_ids: tuple[str, ...]


def merge_schema_units(
    units: Iterable[dict[str, Any]],
    related_unit_ids: Iterable[str],
    *,
    close_relation_endpoints: bool = True,
) -> MergeResult:
    """Merge predicted units into the canonical generator schema format.

    Input order is retained, matching ``CanonicalSchema.subset_dict``. When a
    selected relationship has an unselected endpoint, the corresponding node
    unit is included so the predicted schema remains graph-valid.
    """

    ordered_units = list(units)
    selected = set(related_unit_ids)
    directly_selected = tuple(str(unit["id"]) for unit in ordered_units if str(unit["id"]) in selected)
    nodes_by_label: dict[str, dict[str, Any]] = {}
    node_ids_by_label: dict[str, str] = {}
    for unit in ordered_units:
        if unit.get("kind") != "node":
            continue
        schema = unit.get("schema")
        if not isinstance(schema, dict) or not isinstance(schema.get("label"), str):
            raise ValueError(f"Malformed node unit: {unit!r}")
        label = schema["label"]
        nodes_by_label[label] = schema
        node_ids_by_label[label] = str(unit["id"])

    closure_added: list[str] = []
    if close_relation_endpoints:
        for unit in ordered_units:
            if unit.get("kind") != "relation" or str(unit.get("id")) not in selected:
                continue
            schema = unit.get("schema")
            if not isinstance(schema, dict):
                raise ValueError(f"Malformed relation unit: {unit!r}")
            for endpoint in (schema.get("source"), schema.get("target")):
                if not isinstance(endpoint, str) or endpoint not in node_ids_by_label:
                    raise ValueError(f"Relationship endpoint {endpoint!r} has no node unit")
                node_id = node_ids_by_label[endpoint]
                if node_id not in selected:
                    selected.add(node_id)
                    closure_added.append(node_id)

    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for unit in ordered_units:
        unit_id = str(unit.get("id"))
        if unit_id not in selected or unit_id in seen_ids:
            continue
        seen_ids.add(unit_id)
        schema = unit.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(f"Malformed schema unit: {unit!r}")
        if unit.get("kind") == "node":
            nodes.append(schema)
        elif unit.get("kind") == "relation":
            relationships.append(schema)
        else:
            raise ValueError(f"Unsupported schema-unit kind: {unit.get('kind')!r}")

    return MergeResult(
        sub_schema={"nodes": nodes, "relationships": relationships},
        directly_selected_unit_ids=directly_selected,
        closure_added_node_ids=tuple(closure_added),
    )
