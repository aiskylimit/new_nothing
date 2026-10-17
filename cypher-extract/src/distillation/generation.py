from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _token_ids(value: Any) -> Iterable[int]:
    if isinstance(value, int):
        yield value
    elif isinstance(value, list | tuple | set):
        for token_id in value:
            if isinstance(token_id, int):
                yield token_id


def resolve_eos_token_ids(tokenizer: Any, generation_config: Any | None = None) -> list[int]:
    """Return the model-declared EOS IDs plus the tokenizer EOS, in stable order."""

    candidates = []
    if generation_config is not None:
        candidates.extend(_token_ids(getattr(generation_config, "eos_token_id", None)))
    candidates.extend(_token_ids(getattr(tokenizer, "eos_token_id", None)))
    eos_token_ids = list(dict.fromkeys(token_id for token_id in candidates if token_id >= 0))
    if not eos_token_ids:
        raise ValueError("Generation requires at least one valid EOS token ID.")
    return eos_token_ids


def generation_eos_value(eos_token_ids: list[int]) -> int | list[int]:
    """Use Transformers' scalar form for one EOS and list form for multiple EOS IDs."""

    if not eos_token_ids:
        raise ValueError("Generation requires at least one EOS token ID.")
    return eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids


def selector_generation_kwargs() -> dict[str, Any]:
    """Return the deterministic JSON protocol shared by eval and inference."""

    return {
        "do_sample": False,
        "top_k": 0,
        "top_p": 1.0,
        "temperature": 1.0,
        "max_new_tokens": 16,
        "num_beams": 1,
    }


def reference_eval_generation_kwargs(
    generating_args: Any,
    tokenizer: Any,
    generation_config: Any | None = None,
) -> dict[str, Any]:
    """Build the stochastic validation decoding settings used by CypherKD."""

    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs.update(
        do_sample=True,
        top_k=0,
        top_p=0.95,
        temperature=0.5,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
    )
    eos_token_ids = resolve_eos_token_ids(tokenizer, generation_config)
    gen_kwargs["eos_token_id"] = generation_eos_value(eos_token_ids)
    return gen_kwargs
