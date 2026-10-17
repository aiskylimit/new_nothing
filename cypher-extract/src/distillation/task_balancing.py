from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from schema_grounding.selector_labels import parse_selector_label

GENERATOR_TASK_ID = 0
SELECTOR_TASK_ID = 1
TASK_NAMES = {
    GENERATOR_TASK_ID: "generator",
    SELECTOR_TASK_ID: "selector",
}


def _explicit_task_id(feature: Mapping[str, Any]) -> int | None:
    value = feature.get("task_id")
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("task_id tensors must contain exactly one value.")
        value = value.item()
    task_id = int(value)
    if task_id not in TASK_NAMES:
        raise ValueError(f"Unknown task_id={task_id}; expected {sorted(TASK_NAMES)}.")
    return task_id


def infer_task_id(feature: Mapping[str, Any], tokenizer: Any) -> int:
    """Identify selector rows from their supervised selector response."""

    explicit = _explicit_task_id(feature)
    if explicit is not None:
        return explicit

    labels = feature.get("labels")
    if labels is None:
        raise ValueError("Task-aware collation requires labels or an explicit task_id.")
    if isinstance(labels, torch.Tensor):
        label_ids = labels.detach().cpu().tolist()
    else:
        label_ids = list(labels)
    target_ids = [int(token_id) for token_id in label_ids if int(token_id) != -100]
    if not target_ids:
        raise ValueError("Cannot infer a task from an example without supervised response tokens.")
    response = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
    return SELECTOR_TASK_ID if parse_selector_label(response) is not None else GENERATOR_TASK_ID


class TaskAwareDataCollator:
    """Add task IDs while delegating sequence padding to LlamaFactory."""

    def __init__(self, delegate: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]], tokenizer: Any) -> None:
        self.delegate = delegate
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        task_ids: list[int] = []
        clean_features: list[dict[str, Any]] = []
        for feature in features:
            task_ids.append(infer_task_id(feature, self.tokenizer))
            clean = dict(feature)
            clean.pop("task_id", None)
            clean_features.append(clean)

        batch = self.delegate(clean_features)
        batch["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
        return batch


def task_masks_and_weights(
    task_ids: torch.Tensor,
    selector_weight: float,
) -> list[tuple[str, torch.Tensor, float]]:
    """Return present task masks with weights normalized over present tasks."""

    if task_ids.ndim != 1:
        raise ValueError("task_ids must have shape [batch].")
    unknown = ~((task_ids == GENERATOR_TASK_ID) | (task_ids == SELECTOR_TASK_ID))
    if torch.any(unknown):
        values = sorted(set(task_ids[unknown].detach().cpu().tolist()))
        raise ValueError(f"Unknown task IDs in batch: {values}.")

    present: list[tuple[str, torch.Tensor, float]] = []
    generator_mask = task_ids.eq(GENERATOR_TASK_ID)
    selector_mask = task_ids.eq(SELECTOR_TASK_ID)
    if torch.any(generator_mask):
        present.append((TASK_NAMES[GENERATOR_TASK_ID], generator_mask, 1.0 - selector_weight))
    if torch.any(selector_mask):
        present.append((TASK_NAMES[SELECTOR_TASK_ID], selector_mask, selector_weight))
    if not present:
        raise ValueError("Cannot balance an empty task batch.")

    total_weight = sum(weight for _, _, weight in present)
    return [(name, mask, weight / total_weight) for name, mask, weight in present]


def task_balanced_loss(
    task_ids: torch.Tensor,
    selector_weight: float,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize within each task, then combine generator and selector loss."""

    components: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name, mask, weight in task_masks_and_weights(task_ids, selector_weight):
        component = loss_fn(mask)
        if component.ndim != 0:
            raise ValueError(f"{name} loss must be a scalar, got shape {tuple(component.shape)}.")
        components[name] = component
        weighted = component * weight
        total = weighted if total is None else total + weighted
    if total is None:  # guarded by task_masks_and_weights
        raise RuntimeError("Task-balanced loss accumulation failed.")
    return total, components
