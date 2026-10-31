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
                            selection="bi", weighting="bi", seed=42, allow_empty=False,
                            reduction="token_mean"):
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
    # Decode offsets once on CPU, not inside GPU layer/token loops.
    shared = s_input is t_input and student_tok is teacher_tok and s_labels is t_labels
    s_input = {k: v.detach().cpu() for k, v in s_input.items()}
    s_labels = {"label": s_labels["label"].detach().cpu()}
    t_input = s_input if shared else {k: v.detach().cpu() for k, v in t_input.items()}
    t_labels = s_labels if shared else {"label": t_labels["label"].detach().cpu()}
    spans = []
    for row in range(s_input["input_ids"].shape[0]):
        sv = response_view(s_input, s_labels, student_tok, row)
        tv = sv if shared else response_view(t_input, t_labels, teacher_tok, row)
        # Independent truncation may shorten one response; shared prefix is valid.
        if not (sv[0].startswith(tv[0]) or tv[0].startswith(sv[0])):
            raise ValueError("Teacher/student inputs contain different response text")
        _, so, sp, sbase = sv
        ttext, to, tp, tbase = tv
        if not sp or not tp or sbase is None or tbase is None:
            continue
        positions = [tbase, *tp]

        def boundary(c):
            if c <= 0:
                return positions[0], positions[0], 0.
            for i, (a, b) in enumerate(to):
                if b > a and a <= c <= b:
                    return positions[i], positions[i + 1], (c - a) / (b - a)
            raise ValueError(f"Boundary {c} outside teacher response coverage")

        for (a, b), pos in zip(so, sp):
            if b > a and b <= len(ttext):
                spans.append((row, pos, *boundary(a), *boundary(b)))
    total = student_hs[-1].sum() * 0
    if not spans:
        if allow_empty:
            return total
        raise ValueError("No covered response spans for IMPACT; increase generation/context length")
    device = student_hs[0].device
    columns = list(zip(*spans))
    row_ids, pos, a0, a1, b0, b1 = [torch.tensor(columns[i], device=device, dtype=torch.long)
                                    for i in (0, 1, 2, 3, 5, 6)]
    alpha = torch.tensor(columns[4], device=device).unsqueeze(-1)
    beta = torch.tensor(columns[7], device=device).unsqueeze(-1)
    batch_size = s_input["input_ids"].shape[0]
    counts = torch.bincount(row_ids, minlength=batch_size).clamp_min(1)
    for index, weight in zip(chosen.tolist(), weights):
        t_layer = index + 1
        s_layer = math.ceil(t_layer * ls / lt)  # Same rule as shipped IMPACT code.
        sh, th = student_hs[s_layer], teacher_hs[t_layer].detach()
        # Linear projection commutes with interpolation and subtraction. Project
        # only response deltas, in one GEMM per layer (not full padded sequences).
        begin = th[row_ids, a0].float() * (1 - alpha) + th[row_ids, a1].float() * alpha
        end = th[row_ids, b0].float() * (1 - beta) + th[row_ids, b1].float() * beta
        dt = (end - begin) @ projection.float()
        ds = sh[row_ids, pos].float() - sh[row_ids, pos - 1].float()
        losses = 1 - F.cosine_similarity(ds, dt, dim=-1, eps=1e-6)
        if reduction == "sample_mean":
            # Empty (e.g. EOS-only) examples contribute zero, as in batch-1 runs.
            sums = losses.new_zeros(batch_size).scatter_add(0, row_ids, losses)
            layer_loss = (sums / counts).mean()
        elif reduction == "token_mean":
            layer_loss = losses.mean()
        else:
            raise ValueError(f"Unknown reduction {reduction}")
        total = total + weight * layer_loss
    return total
