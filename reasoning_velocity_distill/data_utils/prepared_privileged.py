"""Read a complete, row-aligned cache of teacher-generated privileged context."""

import hashlib
import json


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_context_field(field):
    if not field or field in ("prompt", "user_prompt", "instruction", "system_prompt", "response", "output", "generated_text",
                              "privileged_prompt", "privileged_preparation"):
        raise ValueError("Context field must be separate from the original prompt/response and preparation metadata")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def load_prepared_contexts(path, records, field):
    """Validate coverage before train_num/train_ratio select any subset.

    The source fingerprint binds each generated context to the exact original
    row, including its question and unchanged reference response.
    """
    validate_context_field(field)
    contexts = []
    for index, row in enumerate(read_jsonl(path)):
        if index >= len(records):
            raise ValueError("Prepared privileged data has more rows than the full training dataset")
        metadata = row.get("privileged_preparation", {})
        if (not isinstance(metadata, dict) or metadata.get("version") != 1
                or metadata.get("kind") != "context"
                or metadata.get("source_sha256") != fingerprint(records[index])):
            raise ValueError(f"Prepared privileged row {index} does not match the training source; rerun prepare")
        context = row.get(field)
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"Prepared privileged row {index} requires nonempty '{field}' context")
        contexts.append(context)
    if len(contexts) != len(records) or not contexts:
        raise ValueError("Prepared privileged data must cover the full training dataset, in the same order")
    return contexts
