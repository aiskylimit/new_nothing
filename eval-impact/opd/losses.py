"""Direct full-vocabulary OPD losses on fresh student-sampled continuations.

No ETA, vocabulary remapping, or dual-space projectors are part of this objective.
"""
import json
import torch
import torch.nn.functional as F


def require_shared_tokenizer(student_tok, teacher_tok):
    if student_tok.get_vocab() != teacher_tok.get_vocab():
        raise ValueError("Direct OPD requires the same token-to-ID vocabulary; cross-tokenizer pairs need an explicit output-distribution bridge")
    for name in ("eos_token_id", "bos_token_id", "unk_token_id"):
        if getattr(student_tok, name, None) != getattr(teacher_tok, name, None):
            raise ValueError(f"Direct OPD tokenizer mismatch: {name}")
    s_backend, t_backend = getattr(student_tok, "backend_tokenizer", None), getattr(teacher_tok, "backend_tokenizer", None)
    if s_backend is not None and t_backend is not None:
        s, t = json.loads(s_backend.to_str()), json.loads(t_backend.to_str())
        for key in ("model", "normalizer", "pre_tokenizer", "decoder", "post_processor"):
            if s.get(key) != t.get(key):
                raise ValueError(f"Direct OPD tokenizer mismatch: {key}")


def direct_opd_loss(student_logits, teacher_logits, labels, objective="reverse_kl", temperature=1.0, token_ids=None, temperature_squared=True, reduction="token_mean"):
    """Mean full-vocabulary divergence at response prediction positions.

    Labels are already shifted once by the batch builder. Their values only
    specify the response mask. Reverse KL = KL(p_student || p_teacher).
    Student sampling is stop-gradient; teacher targets are always detached.
    """
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    if (student_logits.shape[:-1] != teacher_logits.shape[:-1] or
            labels.shape != student_logits.shape[:-1]):
        raise ValueError("Direct OPD requires identical sequence shapes")
    mask = labels.ne(-100)
    if not mask.any():
        raise ValueError("No response prediction positions for OPD")
    student_rows, teacher_rows = student_logits[mask], teacher_logits.detach()[mask]
    if token_ids is not None:
        # Score all actual tokenizer IDs; omit only unassigned LM-head padding.
        student_rows = student_rows.index_select(-1, token_ids)
        teacher_rows = teacher_rows.index_select(-1, token_ids)
    if student_rows.shape != teacher_rows.shape:
        raise ValueError("Direct OPD requires the same output-vocabulary support")
    s = F.log_softmax(student_rows.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_rows.float() / temperature, dim=-1)
    if objective == "reverse_kl":
        token_loss = (s.exp() * (s - t)).sum(-1)
    elif objective == "forward_kl":
        token_loss = (t.exp() * (t - s)).sum(-1)
    elif objective == "js_divergence":
        mixture = torch.logaddexp(s, t) - torch.log(s.new_tensor(2.0))
        token_loss = .5 * ((s.exp() * (s - mixture)).sum(-1) +
                           (t.exp() * (t - mixture)).sum(-1))
    else:
        raise ValueError(f"Unknown divergence {objective}")
    if reduction == "sample_mean":
        # Preserve the original batch-1 + accumulation objective: each example
        # has equal weight, regardless of its response length.
        counts = mask.sum(-1)
        if (counts == 0).any():
            raise ValueError("Every OPD sample needs a response prediction position")
        row_ids = mask.nonzero(as_tuple=True)[0]
        sums = token_loss.new_zeros(labels.shape[0]).scatter_add(0, row_ids, token_loss)
        result = (sums / counts).mean()
    elif reduction == "token_mean":
        result = token_loss.mean()
    else:
        raise ValueError(f"Unknown reduction {reduction}")
    return result * (temperature ** 2 if temperature_squared else 1.)


def build_direct_batch(tokenizer, prompts, sampled_ids, device, max_length=2048,
                       prompt_format="auto", max_prompt_length=None):
    """Teacher and student consume exactly the same sampled IDs and prompt.

    A common student chat prefix is used for both models. EOS is preserved even
    when it is also the padding token; no decoding/retokenization changes IDs.
    """
    from .bi import render_prompt
    rows, starts = [], []
    if len(prompts) != len(sampled_ids):
        raise ValueError("Prompts and continuations must have the same length")
    formats = [prompt_format] * len(prompts) if isinstance(prompt_format, str) else prompt_format
    if len(formats) != len(prompts):
        raise ValueError("Prompt formats must match batch size")
    for prompt, ids, fmt in zip(prompts, sampled_ids, formats):
        prefix = tokenizer.encode(render_prompt(tokenizer, prompt, fmt), add_special_tokens=False)
        prefix = prefix[:max_prompt_length] if max_prompt_length else prefix
        if not prefix or not ids:
            raise ValueError("OPD requires a nonempty prompt and sampled continuation")
        rows.append(prefix + list(ids))
        starts.append(len(prefix))
    if not rows:
        raise ValueError("Empty batch")
    if max(map(len, rows)) > max_length:
        raise ValueError("Sampled rollout exceeds --max-length")
    width = max(map(len, rows)) - 1
    inputs = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask, labels = torch.zeros_like(inputs), torch.full_like(inputs, -100)
    for i, (ids, start) in enumerate(zip(rows, starts)):
        n = len(ids) - 1
        inputs[i, :n] = torch.tensor(ids[:-1], device=device)
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(ids[1:], device=device)
        labels[i, :start - 1] = -100
    input_batch = {"input_ids": inputs, "attention_mask": mask}
    label_batch = {"label": labels, "loss_denom": labels.ne(-100).sum().clamp_min(1)}
    return {"op_input_batch": input_batch, "op_teacher_input_batch": input_batch,
            "op_label_batch": label_batch, "op_teacher_label_batch": label_batch}
