"""Batch helper retained only for archived cross-tokenizer criterion regression tests."""
import torch
from opd.bi import render_prompt

def build_batch(student_tok, teacher_tok, prompts, responses, sampled_ids, device,
                max_length=2048):
    """Preserve sampled student IDs; teacher consumes the identical response text."""
    s_rows, t_rows, s_starts, t_starts = [], [], [], []
    for prompt, response, ids in zip(prompts, responses, sampled_ids):
        sp = student_tok.encode(render_prompt(student_tok, prompt), add_special_tokens=False)
        tp = teacher_tok.encode(render_prompt(teacher_tok, prompt), add_special_tokens=False)
        tr = teacher_tok.encode(response, add_special_tokens=False)
        if ids and ids[-1] == student_tok.eos_token_id:
            tr += [teacher_tok.eos_token_id]
        s_rows.append(sp + list(ids))
        t_rows.append(tp + tr)
        s_starts.append(len(sp))
        t_starts.append(len(tp))
    if not s_rows or any(len(r) < p + 2 for r, p in zip(s_rows, s_starts)):
        raise ValueError("Need at least two sampled tokens for response increment alignment")
    if any(len(r) < p + 2 for r, p in zip(t_rows, t_starts)):
        raise ValueError("Teacher response is too short")
    width = max(len(r) for r in s_rows + t_rows) - 1
    if width + 1 > max_length:
        raise ValueError("Retokenized rollout exceeds --max-length; increase context budget")

    def pack(rows, starts, pad):
        inputs = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        labels = torch.full_like(inputs, -100)
        for i, (ids, start) in enumerate(zip(rows, starts)):
            n = len(ids) - 1
            inputs[i, :n] = torch.tensor(ids[:-1], device=device)
            mask[i, :n] = 1  # Includes valid EOS even if EOS == PAD.
            labels[i, :n] = torch.tensor(ids[1:], device=device)
            labels[i, :start - 1] = -100
        return {"input_ids": inputs, "attention_mask": mask}, {"label": labels,
                    "loss_denom": labels.ne(-100).sum().clamp_min(1)}

    si, sl = pack(s_rows, s_starts, student_tok.pad_token_id)
    ti, tl = pack(t_rows, t_starts, teacher_tok.pad_token_id)
    return {"op_input_batch": si, "op_label_batch": sl,
            "op_teacher_input_batch": ti, "op_teacher_label_batch": tl,
            "op_raw_texts": responses}



