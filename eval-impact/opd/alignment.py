"""Align response states from actual sampled IDs, with independent chat prefixes."""
import math
import torch
import torch.nn.functional as F


def decoded_offsets(tokenizer, ids):
    """Retain sampled segmentation, including noncanonical and byte fallback tokens.

    Incomplete UTF-8 prefixes get a zero-width interval; their text is attributed
    to the next token whose decoded prefix is stable. No encode/decode round trip.
    """
    ids = [int(i) for i in ids]
    decode = lambda x: tokenizer.decode(x, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
    text = decode(ids)
    offsets, boundary = [], 0
    for i in range(len(ids)):
        prefix = decode(ids[:i + 1])
        end = len(prefix) if text.startswith(prefix) else boundary
        if end < boundary:
            raise ValueError("Tokenizer decoder has nonmonotone text boundaries")
        offsets.append((boundary, end))
        boundary = end
    if boundary != len(text):
        raise ValueError("Cannot map sampled tokens to decoded response")
    return text, offsets


def response_view(input_batch, label_batch, tokenizer, row):
    labels = label_batch["label"][row]
    targets = labels.ne(-100).nonzero().flatten()
    if not len(targets):
        return "", [], [], None
    # Labels are shifted: first response target is predicted from last prompt state.
    start = int(targets[0]) + 1
    positions = input_batch["attention_mask"][row].nonzero().flatten()
    positions = positions[positions >= start]
    ids = input_batch["input_ids"][row, positions].tolist()
    text, offsets = decoded_offsets(tokenizer, ids)
    return text, offsets, positions.tolist(), start - 1


def boundary_state(c, offsets, states):
    if c <= 0:
        return states[0]
    for i, (a, b) in enumerate(offsets):
        if b > a and a <= c <= b:
            alpha = (c - a) / (b - a)
            return states[i] * (1 - alpha) + states[i + 1] * alpha
    raise ValueError(f"Boundary {c} outside teacher response coverage")


def response_alignment_loss(student_hs, teacher_hs, batch, student_tok, teacher_tok,
                            projection, scores, top_k=6, tau=2.0,
                            selection="bi", weighting="bi", seed=42, allow_empty=False):
    if tau <= 0 or top_k < 1:
        raise ValueError("tau and top_k must be positive")
    lt, ls = len(teacher_hs) - 1, len(student_hs) - 1
    k = min(top_k, lt)
    scores = scores.to(student_hs[0].device).float()
    if scores.shape != (lt,) or not torch.isfinite(scores).all():
        raise ValueError("BI scores do not match teacher layer count")
    if selection == "bi":
        chosen = torch.argsort(scores, descending=True, stable=True)[:k]
    elif selection == "first":
        chosen = torch.arange(k, device=scores.device)
    elif selection == "last":
        chosen = torch.arange(lt - k, lt, device=scores.device)
    elif selection == "random":
        g = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(lt, generator=g)[:k].to(scores.device)
    else:
        raise ValueError(selection)
    weights = (scores[chosen] / tau).softmax(0) if weighting == "bi" else torch.ones_like(scores[chosen]) / k
    prefix = "op_" if "op_input_batch" in batch else ""
    s_input, t_input = batch[prefix + "input_batch"], batch[prefix + "teacher_input_batch"]
    s_labels, t_labels = batch[prefix + "label_batch"], batch[prefix + "teacher_label_batch"]
    views = []
    for row in range(s_input["input_ids"].shape[0]):
        sv = response_view(s_input, s_labels, student_tok, row)
        tv = response_view(t_input, t_labels, teacher_tok, row)
        # Independent truncation may shorten one response; shared prefix is valid.
        if not (sv[0].startswith(tv[0]) or tv[0].startswith(sv[0])):
            raise ValueError("Teacher/student inputs contain different response text")
        views.append((sv, tv))
    total = student_hs[-1].sum() * 0
    valid_count = 0
    for index, weight in zip(chosen.tolist(), weights):
        t_layer = index + 1
        s_layer = math.ceil(t_layer * ls / lt)  # Same rule as shipped IMPACT code.
        losses = []
        for row, (sv, tv) in enumerate(views):
            _, so, sp, sbase = sv
            ttext, to, tp, tbase = tv
            if not sp or not tp or sbase is None or tbase is None:
                continue
            sh = student_hs[s_layer][row].float()
            th = teacher_hs[t_layer][row].float() @ projection.float()
            tstates = torch.cat([th[tbase:tbase + 1], th[tp]], dim=0)
            for (a, b), pos in zip(so, sp):
                if b <= a or b > len(ttext):
                    continue
                dt = boundary_state(b, to, tstates) - boundary_state(a, to, tstates)
                ds = sh[pos] - sh[pos - 1]
                losses.append(1 - F.cosine_similarity(ds, dt, dim=0, eps=1e-6))
        if losses:
            total = total + weight * torch.stack(losses).mean()
            valid_count += len(losses)
    if not valid_count:
        if allow_empty:
            return total
        raise ValueError("No covered response spans for IMPACT; increase generation/context length")
    return total
