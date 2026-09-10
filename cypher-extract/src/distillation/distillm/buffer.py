from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping

import torch

TensorFeature = dict[str, torch.Tensor]


class ReplayBuffer:
    """CPU replay memory containing individual student-generated examples."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = capacity
        self._memory: deque[TensorFeature] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def add(self, feature: Mapping[str, torch.Tensor]) -> None:
        required = {"input_ids", "attention_mask", "labels"}
        missing = required.difference(feature)
        if missing:
            raise KeyError(f"Replay feature is missing: {', '.join(sorted(missing))}.")
        item: TensorFeature = {}
        stored_fields = required | ({"task_id"} if "task_id" in feature else set())
        for key in stored_fields:
            value = feature[key]
            expected_ndim = 0 if key == "task_id" else 1
            if not isinstance(value, torch.Tensor) or value.ndim != expected_ndim:
                raise ValueError(f"Replay field {key!r} must be a {expected_ndim}-dimensional tensor.")
            item[key] = value.detach().to(device="cpu").clone()
        if not (item["input_ids"].shape == item["attention_mask"].shape == item["labels"].shape):
            raise ValueError("Replay input_ids, attention_mask, and labels must have equal length.")
        self._memory.append(item)

    def extend(self, features: list[Mapping[str, torch.Tensor]]) -> None:
        for feature in features:
            self.add(feature)

    def sample(self, batch_size: int) -> list[TensorFeature]:
        if batch_size <= 0:
            raise ValueError("Replay batch_size must be positive.")
        if len(self._memory) < batch_size:
            raise ValueError(f"Replay buffer has {len(self._memory)} examples but {batch_size} were requested.")
        return [dict(item) for item in self._rng.sample(list(self._memory), k=batch_size)]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "memory": [{key: value.clone() for key, value in item.items()} for item in self._memory],
            "random_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("capacity") != self.capacity:
            raise ValueError("Replay checkpoint capacity does not match the current configuration.")
        self._memory.clear()
        for feature in state.get("memory", []):
            self.add(feature)
        if "random_state" in state:
            self._rng.setstate(state["random_state"])
