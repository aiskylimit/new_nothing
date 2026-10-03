"""Group-rollout sampling shared by the RLSD and SDPO baselines - both need
G i.i.d. samples from the SAME prompt in one training step (RLSD's group-
relative advantage, Eq. 1; SDPO's "successful previous rollout from the same
group" feedback, Sec 3), unlike TROPIC-P/OPSD which only ever draw one
rollout per example. A new file rather than an addition to
`tropic/rollout.py` (kept byte-for-byte unchanged, like every other
pre-existing tropic/*.py module)."""
from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def generate_rollout_group(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    student_prompt_ids: Tensor,
    group_size: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.1,
    top_p: float = 0.95,
    top_k: int | None = 20,
) -> list[tuple[Tensor, int]]:
    """Samples G rollouts {y^(1),...,y^(G)} ~ pi_theta(.|x) in one batched
    `generate()` call (input repeated G times). Defaults (temperature=1.1,
    top_p=0.95, top_k=20) match this project's own TROPIC-P/OPSD training-time
    rollout sampling (tropic/rollout.py's docstring / run_full_experiment.py's
    CONFIG, itself verified against OPSD's real launch scripts) - "the same
    experimental setting" this project already uses.

    Returns a list of (generated_ids, gen_len), EOS-stripped per-sequence
    exactly like `tropic.rollout.generate_rollout` (each rollout in the group
    can legitimately have a different length)."""
    was_training = model.training
    model.eval()
    input_ids = student_prompt_ids.to(model.device).unsqueeze(0).repeat(group_size, 1)
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

    prompt_len = student_prompt_ids.shape[0]
    eos_id = tokenizer.eos_token_id
    results: list[tuple[Tensor, int]] = []
    for i in range(group_size):
        generated = output[i, prompt_len:]
        if eos_id is not None:
            eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated = generated[: eos_positions[0].item()]
        if generated.shape[0] == 0:
            generated = output[i, prompt_len : prompt_len + 1]
        results.append((generated, generated.shape[0]))
    return results
