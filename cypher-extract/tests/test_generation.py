from types import SimpleNamespace

import pytest

from distillation.generation import (
    generation_eos_value,
    reference_eval_generation_kwargs,
    resolve_eos_token_ids,
    selector_generation_kwargs,
)


def test_qwen2_5_generation_preserves_model_eos_list_without_other_special_tokens() -> None:
    tokenizer = SimpleNamespace(
        eos_token_id=151645,
        additional_special_tokens_ids=[151644, 151645, 151646, 151647],
    )
    generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

    eos_token_ids = resolve_eos_token_ids(tokenizer, generation_config)

    assert eos_token_ids == [151645, 151643]
    assert generation_eos_value(eos_token_ids) == [151645, 151643]


def test_generation_eos_falls_back_to_tokenizer_and_rejects_missing_id() -> None:
    assert resolve_eos_token_ids(SimpleNamespace(eos_token_id=2)) == [2]
    assert generation_eos_value([2]) == 2
    with pytest.raises(ValueError, match="EOS"):
        resolve_eos_token_ids(SimpleNamespace(eos_token_id=None))


def test_reference_eval_generation_uses_stochastic_cypherkd_settings() -> None:
    generating_args = SimpleNamespace(
        to_dict=lambda **_: {"do_sample": False, "top_p": 1.0, "temperature": 1.0}
    )
    tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
    generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

    kwargs = reference_eval_generation_kwargs(generating_args, tokenizer, generation_config)

    assert kwargs == {
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.95,
        "temperature": 0.5,
        "max_new_tokens": 256,
        "pad_token_id": 151643,
        "eos_token_id": [151645, 151643],
    }


def test_selector_eval_generation_matches_inference_protocol() -> None:
    assert selector_generation_kwargs() == {
        "do_sample": False,
        "top_k": 0,
        "top_p": 1.0,
        "temperature": 1.0,
        "max_new_tokens": 16,
        "num_beams": 1,
    }
