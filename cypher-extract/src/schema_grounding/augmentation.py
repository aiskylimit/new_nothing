"""Distractor augmentation for generator training sub-schemas.

Generator rows are normally built from the gold sub-schema only, while inference
feeds the generator a *predicted* sub-schema that may contain irrelevant units.
This module rewrites training rows into one of three modes:

* ``gold``: the unchanged gold sub-schema;
* ``noisy``: gold plus a small number of non-gold distractor units;
* ``full``: the complete schema.

Distractors are merged with :func:`merge_schema_units`, the same function the
inference pipeline uses, so unit order, relationship-endpoint closure, and the
serialized format stay identical to predicted sub-schemas. Gold units are never
removed. This is a training-data transform only; nothing here may be applied to
inference prompts.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .inference.merge import merge_schema_units

GOLD_MODE = "gold"
NOISY_MODE = "noisy"
FULL_MODE = "full"
POOL_NAMES = ("neighbor", "lexical", "random")

_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_QUESTION_WORD_RE = re.compile(r"[a-z0-9]+")
_LEXICAL_STOPWORDS = frozenset(
    {
        "all", "and", "are", "for", "from", "has", "have", "how", "list", "many", "name", "number",
        "that", "the", "their", "them", "there", "these", "was", "were", "what", "when", "where",
        "which", "who", "whose", "with",
    }
)


@dataclass(frozen=True)
class AugmentationConfig:
    gold_ratio: float = 0.15
    full_ratio: float = 0.10
    max_distractors: int = 6
    neighbor_weight: float = 0.5
    lexical_weight: float = 0.25
    random_weight: float = 0.25
    seed: int = 42

    def __post_init__(self) -> None:
        if self.gold_ratio < 0 or self.full_ratio < 0 or self.gold_ratio + self.full_ratio > 1:
            raise ValueError("gold_ratio and full_ratio must be non-negative and sum to at most 1.")
        if self.max_distractors < 1:
            raise ValueError("max_distractors must be at least 1.")
        weights = (self.neighbor_weight, self.lexical_weight, self.random_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Distractor pool weights must be non-negative with a positive sum.")


def node_unit_id(label: str) -> str:
    return f"node:{label}"


def relation_unit_id(relationship: Mapping[str, Any]) -> str:
    return f"relation:{relationship['source']}|{relationship['type']}|{relationship['target']}"


def schema_units(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return merge-ready units in canonical order (nodes, then relationships)."""

    units = [{"id": node_unit_id(node["label"]), "kind": "node", "schema": node} for node in schema["nodes"]]
    units.extend(
        {"id": relation_unit_id(relationship), "kind": "relation", "schema": relationship}
        for relationship in schema["relationships"]
    )
    return units


def gold_unit_ids(sub_schema: Mapping[str, Any], units_by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Map a gold sub-schema back to schema unit ids, rejecting inconsistent rows."""

    ids = [node_unit_id(node["label"]) for node in sub_schema["nodes"]]
    ids.extend(relation_unit_id(relationship) for relationship in sub_schema["relationships"])
    entries = [*sub_schema["nodes"], *sub_schema["relationships"]]
    for unit_id, entry in zip(ids, entries, strict=True):
        unit = units_by_id.get(unit_id)
        if unit is None or unit["schema"] != entry:
            raise ValueError(f"Gold sub-schema entry {unit_id!r} does not match the row's schema.")
    return ids


def _endpoint_ids(unit: Mapping[str, Any]) -> set[str]:
    if unit["kind"] != "relation":
        return set()
    schema = unit["schema"]
    return {node_unit_id(schema["source"]), node_unit_id(schema["target"])}


def _words(identifier: str) -> set[str]:
    return {_normalize_word(word.lower()) for word in _WORD_RE.findall(identifier)}


def _normalize_word(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _unit_words(unit: Mapping[str, Any]) -> set[str]:
    schema = unit["schema"]
    name = schema["label"] if unit["kind"] == "node" else schema["type"]
    words = _words(name)
    for property_name in schema.get("properties", {}):
        words |= _words(property_name)
    return {word for word in words if len(word) >= 3 and word not in _LEXICAL_STOPWORDS}


def _question_words(question: str) -> set[str]:
    words = {_normalize_word(word) for word in _QUESTION_WORD_RE.findall(question.lower())}
    return {word for word in words if len(word) >= 3 and word not in _LEXICAL_STOPWORDS}


def distractor_pools(
    units: Sequence[Mapping[str, Any]], gold_ids: Sequence[str], question: str
) -> dict[str, list[str]]:
    """Build candidate pools that mimic plausible selector false positives."""

    gold = set(gold_ids)
    gold_nodes = {unit_id for unit_id in gold if unit_id.startswith("node:")}
    candidates = [unit for unit in units if unit["id"] not in gold]
    adjacent_nodes: set[str] = set()
    for unit in units:
        endpoints = _endpoint_ids(unit)
        if endpoints & gold_nodes:
            adjacent_nodes |= endpoints
    question_words = _question_words(question)
    return {
        "neighbor": [
            unit["id"]
            for unit in candidates
            if (unit["kind"] == "relation" and _endpoint_ids(unit) & gold_nodes)
            or (unit["kind"] == "node" and unit["id"] in adjacent_nodes)
        ],
        "lexical": [unit["id"] for unit in candidates if _unit_words(unit) & question_words],
        "random": [unit["id"] for unit in candidates],
    }


def example_rng(seed: int, example_id: str) -> random.Random:
    """Derive an order-independent, process-stable RNG for one example."""

    digest = sha256(str(example_id).encode("utf-8")).digest()
    return random.Random(seed ^ int.from_bytes(digest[:8], byteorder="big"))


def sample_distractors(
    units: Sequence[Mapping[str, Any]],
    gold_ids: Sequence[str],
    pools: Mapping[str, Sequence[str]],
    target: int,
    config: AugmentationConfig,
    rng: random.Random,
) -> list[str]:
    """Pick up to ``target`` added units, counting relationship-endpoint closure."""

    units_by_id = {unit["id"]: unit for unit in units}
    selected = set(gold_ids)
    added: list[str] = []
    remaining = {name: [unit_id for unit_id in pools[name] if unit_id not in selected] for name in POOL_NAMES}
    weights = dict(zip(POOL_NAMES, (config.neighbor_weight, config.lexical_weight, config.random_weight), strict=True))
    while len(added) < target:
        available = [name for name in POOL_NAMES if remaining[name] and weights[name] > 0]
        if not available:
            break
        pool = rng.choices(available, weights=[weights[name] for name in available])[0]
        unit_id = remaining[pool].pop(rng.randrange(len(remaining[pool])))
        if unit_id in selected:
            continue
        closure = [node_id for node_id in sorted(_endpoint_ids(units_by_id[unit_id])) if node_id not in selected]
        new_ids = [unit_id, *closure]
        if len(added) + len(new_ids) > target:
            continue
        selected.update(new_ids)
        added.extend(new_ids)
    return added


def augment_generation_row(
    row: Mapping[str, Any], schema: Mapping[str, Any], config: AugmentationConfig
) -> dict[str, Any]:
    """Return a copy of ``row`` whose sub-schema follows one augmentation mode."""

    if row["schema_id"] != schema["schema_id"]:
        raise ValueError(f"Row {row['id']!r} references schema {row['schema_id']!r}, got {schema['schema_id']!r}.")
    units = schema_units(schema)
    units_by_id = {unit["id"]: unit for unit in units}
    gold_ids = gold_unit_ids(row["sub_schema"], units_by_id)
    non_gold = [unit["id"] for unit in units if unit["id"] not in set(gold_ids)]

    rng = example_rng(config.seed, str(row["id"]))
    draw = rng.random()
    if draw < config.gold_ratio or not non_gold:
        mode, added = GOLD_MODE, []
    elif draw < config.gold_ratio + config.full_ratio:
        mode, added = FULL_MODE, non_gold
    else:
        mode = NOISY_MODE
        target = rng.randint(1, min(config.max_distractors, len(non_gold)))
        pools = distractor_pools(units, gold_ids, str(row["question"]))
        added = sample_distractors(units, gold_ids, pools, target, config, rng)

    merged = merge_schema_units(units, [*gold_ids, *added], close_relation_endpoints=True)
    if merged.closure_added_node_ids:
        raise AssertionError(f"Row {row['id']!r} needed unexpected closure: {merged.closure_added_node_ids}")
    augmented = dict(row)
    augmented["sub_schema"] = merged.sub_schema
    augmented["schema_augmentation"] = {
        "mode": mode,
        "gold_unit_ids": gold_ids,
        "distractor_unit_ids": [unit_id for unit_id in merged.directly_selected_unit_ids if unit_id not in gold_ids],
    }
    return augmented
