from __future__ import annotations

import json

POSITIVE_SELECTOR_LABEL = "YES"
NEGATIVE_SELECTOR_LABEL = "NO"
SELECTOR_LABELS = frozenset({POSITIVE_SELECTOR_LABEL, NEGATIVE_SELECTOR_LABEL})
SELECTOR_OUTPUT_KEY = "label"


def selector_label_from_binary(label: int) -> str:
    """Map the source binary relevance label to its semantic label."""

    if label == 1:
        return POSITIVE_SELECTOR_LABEL
    if label == 0:
        return NEGATIVE_SELECTOR_LABEL
    raise ValueError(f"Selector label must be 0 or 1, got {label!r}")


def format_selector_response(label: str) -> str:
    """Serialize one selector label using the supervised JSON contract."""

    if label not in SELECTOR_LABELS:
        raise ValueError(f"Unknown selector label: {label!r}")
    return json.dumps({SELECTOR_OUTPUT_KEY: label})


def parse_selector_label(text: str) -> str | None:
    """Parse the JSON selector contract, retaining legacy bare-label support."""

    normalized = text.strip()
    if normalized in SELECTOR_LABELS:
        return normalized
    try:
        payload = json.loads(normalized)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {SELECTOR_OUTPUT_KEY}:
        return None
    label = payload[SELECTOR_OUTPUT_KEY]
    return label if isinstance(label, str) and label in SELECTOR_LABELS else None
