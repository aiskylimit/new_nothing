"""Pack prompt + response IDs with causal labels, preserving EOS == PAD."""

import torch


def pack_trajectories(prompts, responses, pad_id, model_type, max_length, device):
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("Expected equally sized nonempty prompt/response batches")
    lengths = [len(p) + len(r) - 1 for p, r in zip(prompts, responses)]
    if any(not len(p) or not len(r) for p, r in zip(prompts, responses)):
        raise ValueError("Every trajectory requires a prompt and response")
    if max(lengths) + 1 > max_length:
        raise ValueError("Trajectory exceeds context limit; increase --t-max-length "
                         "for privileged scoring (responses are never silently truncated)")
    width = max(lengths)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        sequence = torch.cat((torch.as_tensor(prompt, device=device),
                              torch.as_tensor(response, device=device))).long()
        length = lengths[i]
        ids[i, :length] = sequence[:-1]
        mask[i, :length] = 1
        labels[i, len(prompt) - 1:length] = sequence[len(prompt):]
    batch = {"input_ids": ids, "attention_mask": mask}
    if model_type == "gpt2":
        batch["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0) * mask
    return batch, {"label": labels, "loss_mask": (labels != -100).float()}
