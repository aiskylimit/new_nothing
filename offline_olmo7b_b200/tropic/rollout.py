"""Phase 4: on-policy rollout generation ("student acts", Algorithm 1 line 3)."""
from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def generate_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    student_prompt_ids: Tensor,
    max_new_tokens: int = 48,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> tuple[Tensor, int]:
    """Sample o ~ pi_theta(.|x) from the student context only.

    `top_k`: OPSD's own real training-time rollout sampling uses `top_k=20`
    (scripts/run_opsd_1b.sh, verified live) alongside temperature=1.1/top_p=0.95
    - left as an optional (default None = HF's own unrestricted default) so
    older callers (the GSM8K CPU smoke test) are unaffected.

    Returns (generated_ids, gen_len) with the prompt stripped off and any
    trailing EOS token stripped as well (so gen_len only counts genuine
    generated content that later gets scored by the teacher/old/student
    forward passes).
    """
    was_training = model.training
    model.eval()
    input_ids = student_prompt_ids.to(model.device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if top_k is not None:
        generate_kwargs["top_k"] = top_k
    output = model.generate(**generate_kwargs)
    if was_training:
        model.train()

    generated = output[0, student_prompt_ids.shape[0] :]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated = generated[: eos_positions[0].item()]
    if generated.shape[0] == 0:
        # degenerate case (immediate EOS): keep at least 1 token so downstream
        # shapes never collapse to zero-length tensors.
        generated = output[0, student_prompt_ids.shape[0] : student_prompt_ids.shape[0] + 1]
    return generated, generated.shape[0]
