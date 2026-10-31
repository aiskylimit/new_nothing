"""Batched rollout and reference CE, preserving the batch-1 sample weighting."""
import torch
import torch.nn.functional as F
from .bi import render_prompt


def prompt_ids(tokenizer, row, max_prompt_length=None):
    ids = tokenizer.encode(render_prompt(tokenizer, row["prompt"], row.get("prompt_format", "auto")),
                           add_special_tokens=False)
    ids = ids[:max_prompt_length] if max_prompt_length else ids
    if not ids:
        raise ValueError(f"Empty prompt: {row['id']}")
    return ids


def rollout_inputs(tokenizer, rows, device, max_prompt_length=None):
    prefixes = [prompt_ids(tokenizer, row, max_prompt_length) for row in rows]
    width = max(map(len, prefixes))
    inputs = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for i, prefix in enumerate(prefixes):
        # Decoder-only generation needs left padding, even if training uses right padding.
        inputs[i, -len(prefix):] = torch.tensor(prefix, device=device)
        mask[i, -len(prefix):] = 1
    return {"input_ids": inputs, "attention_mask": mask}


def trim_generated(generated, eos_token_id):
    """Remove batch-generation padding after first EOS, retaining the EOS itself."""
    result = []
    for ids in generated.tolist():
        if eos_token_id in ids:
            ids = ids[:ids.index(eos_token_id) + 1]
        result.append(ids)
    return result


def supervised_batch_loss(student, tokenizer, rows, device, max_length, max_prompt_length=None):
    sequences, starts = [], []
    for row in rows:
        prefix = prompt_ids(tokenizer, row, max_prompt_length)
        response = tokenizer.encode(row.get("response", ""), add_special_tokens=False)
        if not response:
            raise ValueError(f"Missing reference response for CE: {row['id']}; use --kd-rate 1 for prompt-only OPD")
        ids = (prefix + response + [tokenizer.eos_token_id])[:max_length]
        if len(ids) <= len(prefix):
            raise ValueError("No reference response fits context length")
        sequences.append(ids)
        starts.append(len(prefix))
    width = max(map(len, sequences)) - 1
    inputs = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask, labels = torch.zeros_like(inputs), torch.full_like(inputs, -100)
    for i, (ids, start) in enumerate(zip(sequences, starts)):
        n = len(ids) - 1
        inputs[i, :n] = torch.tensor(ids[:-1], device=device)
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(ids[1:], device=device)
        labels[i, :start - 1] = -100
    output = student(input_ids=inputs, attention_mask=mask, use_cache=False)
    losses = F.cross_entropy(output.logits.float().flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
    return (losses.sum(-1) / labels.ne(-100).sum(-1)).mean()
