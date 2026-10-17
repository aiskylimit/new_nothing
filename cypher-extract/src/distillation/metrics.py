"""Task-aware generative metrics for the mixed Text-to-Cypher evaluation set."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from schema_grounding.inference.parsing import parse_selector_label

IGNORE_INDEX = -100
_TOKEN_PATTERN = re.compile(
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:``|[^`])*`|"
    r"<=|>=|<>|!=|=~|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]"
)
def _json_objects(text: str):
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _malformed_json_cypher(text: str) -> str | None:
    """Recover the cypher field from common truncated/loosely escaped model JSON."""

    match = re.search(r'["\']cypher["\']\s*:\s*"', text, flags=re.IGNORECASE)
    if not match:
        return None

    output: list[str] = []
    index = match.end()
    closed = False
    simple_escapes = {
        '"': '"',
        "'": "'",  # Models often emit invalid JSON \' for Cypher literals.
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < len(text):
        character = text[index]
        if character == '"':
            closed = True
            break
        if character != "\\":
            output.append(character)
            index += 1
            continue

        if index + 1 >= len(text):
            output.append("\\")
            break
        escaped = text[index + 1]
        if escaped == "u" and index + 5 < len(text):
            codepoint = text[index + 2 : index + 6]
            if re.fullmatch(r"[0-9A-Fa-f]{4}", codepoint):
                output.append(chr(int(codepoint, 16)))
                index += 6
                continue
        if escaped in simple_escapes:
            output.append(simple_escapes[escaped])
        else:
            # Preserve unknown escapes because they may belong to a Cypher regex.
            output.extend(("\\", escaped))
        index += 2

    cypher = "".join(output).strip()
    if not closed:
        # A truncated JSON envelope often leaves its closing brace on a new line.
        cypher = re.sub(r"\s*\}\s*$", "", cypher).strip()
    return cypher or None


def extract_cypher(text: str) -> str:
    """Extract a Cypher string from the expected JSON response, with robust fallbacks."""

    stripped = text.strip()
    for payload in _json_objects(stripped):
        cypher = payload.get("cypher")
        if isinstance(cypher, str):
            return cypher.strip()

    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    malformed = _malformed_json_cypher(stripped)
    if malformed:
        return malformed
    return stripped


def _cypher_tokens(text: str, *, lowercase: bool) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(extract_cypher(text))
    if tokens and tokens[-1] == ";":
        tokens.pop()
    return [token.lower() for token in tokens] if lowercase else tokens


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if predicted == 0 and reference == 0:
        return 1.0
    if overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2.0 * precision * recall / (precision + recall)


def _rouge_n(prediction: list[str], reference: list[str], n: int) -> float:
    predicted_ngrams = Counter(tuple(prediction[index : index + n]) for index in range(len(prediction) - n + 1))
    reference_ngrams = Counter(tuple(reference[index : index + n]) for index in range(len(reference) - n + 1))
    overlap = sum((predicted_ngrams & reference_ngrams).values())
    return _f1(overlap, sum(predicted_ngrams.values()), sum(reference_ngrams.values()))


def _rouge_l(prediction: list[str], reference: list[str]) -> float:
    previous = [0] * (len(reference) + 1)
    for predicted_token in prediction:
        current = [0]
        for index, reference_token in enumerate(reference, 1):
            if predicted_token == reference_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return _f1(previous[-1], len(prediction), len(reference))


def compute_task_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Compute selector accuracy plus token-level Cypher EM and ROUGE F1 scores."""

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    selector_correct = 0
    selector_count = 0
    generator_count = 0
    generator_exact = 0
    rouge1 = 0.0
    rouge2 = 0.0
    rouge_l = 0.0

    for prediction, reference in zip(predictions, references, strict=True):
        reference_label = parse_selector_label(reference)
        if reference_label is not None:
            selector_count += 1
            selector_correct += parse_selector_label(prediction) == reference_label
            continue

        generator_count += 1
        predicted_exact = _cypher_tokens(prediction, lowercase=False)
        reference_exact = _cypher_tokens(reference, lowercase=False)
        generator_exact += predicted_exact == reference_exact
        predicted_rouge = [token.lower() for token in predicted_exact]
        reference_rouge = [token.lower() for token in reference_exact]
        rouge1 += _rouge_n(predicted_rouge, reference_rouge, 1)
        rouge2 += _rouge_n(predicted_rouge, reference_rouge, 2)
        rouge_l += _rouge_l(predicted_rouge, reference_rouge)

    metrics: dict[str, float] = {
        "selector_count": float(selector_count),
        "generator_count": float(generator_count),
    }
    if selector_count:
        metrics["selector_accuracy"] = round(100.0 * selector_correct / selector_count, 4)
    if generator_count:
        metrics.update(
            generator_exact_match=round(100.0 * generator_exact / generator_count, 4),
            generator_rouge1=round(100.0 * rouge1 / generator_count, 4),
            generator_rouge2=round(100.0 * rouge2 / generator_count, 4),
            generator_rougeL=round(100.0 * rouge_l / generator_count, 4),
        )
    return metrics


def decode_task_outputs(tokenizer: Any, predictions: Any, label_ids: Any) -> tuple[list[str], list[str]]:
    """Decode gathered generation outputs and labels using the training metric convention."""

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]
    predictions = np.asarray(predictions)
    labels = np.asarray(label_ids)
    if predictions.ndim != 2 or labels.ndim != 2:
        raise ValueError("Task outputs require 2-D generated token IDs and labels.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    labels = np.where(labels != IGNORE_INDEX, labels, pad_token_id)
    predictions = np.where(predictions != IGNORE_INDEX, predictions, pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_predictions, decoded_labels


@dataclass
class ComputeTaskMetrics:
    """Decode generated token IDs and dispatch rows to their task metric."""

    tokenizer: Any

    def __call__(self, eval_prediction: Any) -> dict[str, float]:
        decoded_predictions, decoded_labels = decode_task_outputs(
            self.tokenizer,
            eval_prediction.predictions,
            eval_prediction.label_ids,
        )
        return compute_task_metrics(decoded_predictions, decoded_labels)
