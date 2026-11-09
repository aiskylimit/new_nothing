"""Exact step-weighted NLL/KL with chunked fp32 work and recomputed backward.

Keep bf16 logits, not a full-sequence fp32 softmax graph. The KL derivative is
analytical (p_student - p_teacher); neither vocabulary nor tokens are sampled.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


def step_token_weights(length, spans, weights, *, device, dtype=torch.float32):
    """Map step-mean weights to *prediction* positions, including causal shift."""
    weights = torch.as_tensor(weights, device=device, dtype=dtype)
    if len(spans) != weights.numel():
        raise ValueError("Segment/weight count mismatch")
    out = torch.zeros(length, device=device, dtype=dtype)
    denominator = weights.sum().clamp_min(1e-8)
    for (start, end), weight in zip(spans, weights):
        start, end = max(int(start), 1), min(int(end), length)
        if end > start:
            out[start - 1:end - 1] += weight / ((end - start) * denominator)
    return out


class _WeightedNLLKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, labels, weights, chunk):
        if student.shape[:-1] != labels.shape or labels.shape != weights.shape:
            raise ValueError("Expected logits (..., V), labels/weights (...)")
        if teacher is not None and teacher.shape != student.shape:
            raise ValueError("Teacher/student logits must match")
        if chunk <= 0:
            raise ValueError("Loss chunk must be positive")
        # Only response positions enter loss; prompt/padding have zero weights.
        indices = weights.reshape(-1).nonzero().flatten()
        s = student.reshape(-1, student.shape[-1])
        t = teacher.reshape_as(s) if teacher is not None else None
        y, w = labels.reshape(-1), weights.reshape(-1)
        nll = student.new_zeros((), dtype=torch.float32)
        kl = nll.clone()
        for pos in indices.split(chunk):
            log_s = s.index_select(0, pos).float().log_softmax(-1)
            coef = w.index_select(0, pos).float()
            targets = y.index_select(0, pos)
            nll -= (coef * log_s.gather(1, targets[:, None]).squeeze(1)).sum()
            if t is not None:
                log_t = t.index_select(0, pos).float().log_softmax(-1)
                kl += (coef * (log_t.exp() * (log_t - log_s)).sum(-1)).sum()
        ctx.save_for_backward(student, teacher if teacher is not None else student.new_empty(0),
                              labels, weights, indices)
        ctx.has_teacher, ctx.chunk = teacher is not None, chunk
        return nll, kl

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_nll, grad_kl):
        student, teacher, labels, weights, indices = ctx.saved_tensors
        s = student.reshape(-1, student.shape[-1])
        t = teacher.reshape_as(s) if ctx.has_teacher else None
        y, w = labels.reshape(-1), weights.reshape(-1)
        grad = torch.zeros_like(student)
        flat = grad.reshape_as(s)
        for pos in indices.split(ctx.chunk):
            probs = s.index_select(0, pos).float().softmax(-1)
            nll_grad = probs.clone()
            nll_grad.scatter_add_(1, y.index_select(0, pos)[:, None],
                                  -torch.ones((pos.numel(), 1), device=s.device))
            result = grad_nll * nll_grad
            if t is not None:
                teacher_probs = t.index_select(0, pos).float().softmax(-1)
                result.add_(grad_kl * (probs - teacher_probs))
            result.mul_(w.index_select(0, pos).float()[:, None])
            flat.index_copy_(0, pos, result.to(student.dtype))
        return grad, None, None, None, None


def weighted_nll_kl(student, labels, weights, *, teacher=None, chunk=256):
    """Return two scalar losses. Teacher is fixed; weights are not trainable."""
    return _WeightedNLLKL.apply(student, teacher, labels, weights, chunk)
