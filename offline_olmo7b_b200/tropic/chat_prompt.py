"""Minimal chat-template rendering helper shared by the RLSD/SDPO baselines'
own prompt builders (tropic/rlsd.py, tropic/sdpo.py) and their data loader
(tropic/verifier_data.py).

Deliberately NOT importing tropic.data's private `_apply_chat_template_ids`
(kept these baseline files dependency-free of any *internal* helper of the
existing TROPIC-P/OPSD code, per an explicit "don't touch/lean on my
existing files" instruction) - this is the same two-step render-then-tokenize
process OPSD's own data_collator.py uses (render the chat template to a
STRING, then tokenize that string separately, so `max_length` truncation
applies to the whole rendered prompt), duplicated here in a few lines rather
than imported. `tropic/data.py` itself is never imported by this file.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    content: str,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """`empty_think_suffix`: for models with NO native `enable_thinking`
    template branch but whose `add_generation_prompt=True` output auto-opens
    a `<think>` tag (e.g. OLMo-3-Think) - same "empty think prefill" trick as
    `tropic.data._apply_chat_template_ids` (kept as an independent copy here
    per this file's own no-cross-import docstring note, not by re-importing
    that function): when `enable_thinking is False`, this string is appended
    to close the auto-opened tag empty instead of passing `enable_thinking`
    (which such models silently ignore) as a kwarg."""
    messages = [{"role": "user", "content": content}]
    if empty_think_suffix is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if enable_thinking is False:
            text += empty_think_suffix
    else:
        kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = tokenizer(text, truncation=max_length is not None, max_length=max_length, return_tensors="pt")
    return encoded["input_ids"].squeeze(0)
