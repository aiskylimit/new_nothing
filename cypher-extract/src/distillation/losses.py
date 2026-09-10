from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal LM cross-entropy normalized over response tokens."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("LM logits [batch, sequence, vocab] and labels [batch, sequence] must align.")
    if labels.shape[1] < 2:
        raise ValueError("At least two tokens are required for causal LM loss.")
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    if not torch.any(shifted_labels.ne(IGNORE_INDEX)):
        raise ValueError("The LM batch contains no response tokens.")
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace non-finite logits before probability/logarithm operations."""

    if torch.isfinite(logits).all():
        return logits
    return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


def align_causal_logits_and_labels(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align causal outputs and retain only supervised response-token rows.

    The legacy dataset fed `tokens[:-1]` to the model and supplied
    `tokens[1:]` as labels. LlamaFactory supplies full sequences and relies on
    the model's internal shift, so KD must explicitly apply the same shift.
    Compacting before sanitization and probability operations avoids building
    vocabulary-sized intermediates for ignored prompt and padding positions.
    """

    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("Student and teacher logits must have shape [batch, sequence, vocabulary].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence].")
    if student_logits.shape[:2] != teacher_logits.shape[:2] or student_logits.shape[:2] != labels.shape:
        raise ValueError("Student logits, teacher logits, and labels must share batch/sequence dimensions.")
    if labels.shape[1] < 2:
        raise ValueError("At least two tokens are required for causal distillation.")

    shared_vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
    if shared_vocab_size <= 0:
        raise ValueError("Student and teacher vocabularies must be non-empty.")
    shifted_labels = labels[:, 1:].masked_fill(labels[:, 1:] >= shared_vocab_size, IGNORE_INDEX)
    response_mask = shifted_labels.ne(IGNORE_INDEX)
    return (
        _sanitize_logits(student_logits[:, :-1, :shared_vocab_size][response_mask]),
        _sanitize_logits(teacher_logits[:, :-1, :shared_vocab_size][response_mask]),
        shifted_labels[response_mask],
    )


def _masked_token_mean(token_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels.ne(IGNORE_INDEX)
    count = mask.sum()
    if not torch.any(mask):
        raise ValueError("The distillation batch contains no response tokens.")
    return token_values.masked_fill(~mask, 0.0).sum() / count


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Teacher-to-student cross-entropy used by the original FKL baseline.

    The teacher entropy constant is intentionally retained, matching the
    template and therefore its gradients and scalar logs.
    """

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    product = torch.masked_fill(teacher_probs * student_log_probs, torch.isinf(student_logits), 0)
    token_loss = -product.sum(dim=-1)
    return _masked_token_mean(token_loss, labels)


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    teacher_product = torch.masked_fill(student_probs * teacher_log_probs, inf_mask, 0)
    student_product = torch.masked_fill(student_probs * student_log_probs, inf_mask, 0)
    token_loss = (student_product - teacher_product).sum(dim=-1)
    return _masked_token_mean(token_loss, labels)


def skewed_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    mixture = alpha * teacher_probs + (1.0 - alpha) * student_probs
    mixture_log_probs = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    product = torch.masked_fill(teacher_probs * mixture_log_probs, inf_mask, 0)
    token_loss = -product.sum(dim=-1)
    return _masked_token_mean(token_loss, labels)


def skewed_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    mixture = (1.0 - alpha) * teacher_probs + alpha * student_probs
    mixture_log_probs = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    mixture_product = torch.masked_fill(student_probs * mixture_log_probs, inf_mask, 0)
    student_product = torch.masked_fill(student_probs * student_log_probs, inf_mask, 0)
    token_loss = (student_product - mixture_product).sum(dim=-1)
    return _masked_token_mean(token_loss, labels)


def bdl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lam: float = 0.9,
) -> torch.Tensor:
    """Bidirectional Discrepancy Loss between asymmetric probability mixtures.

    The teacher and student distributions are mixed in opposite directions:
    ``Pm=(1-lam)p+lam*q`` and ``Qm=lam*p+(1-lam)*q``.  BDL is then
    ``KL(Pm || Qm)`` evaluated on the response-token mask.
    """

    if not 0.0 < lam < 1.0:
        raise ValueError("BDL lam must be in (0, 1).")
    if math.isclose(lam, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("BDL lam=0.5 makes both mixture distributions identical and the loss zero.")

    student_logits = student_logits.to(torch.float32)
    teacher_logits = teacher_logits.to(torch.float32)
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    if inf_mask.any():
        student_logits = student_logits.masked_fill(torch.isinf(student_logits), 0.0)
        teacher_logits = teacher_logits.masked_fill(torch.isinf(teacher_logits), 0.0)

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    p_mix = (1.0 - lam) * teacher_probs + lam * student_probs
    q_mix = lam * teacher_probs + (1.0 - lam) * student_probs
    p_mix_log = p_mix.clamp_min(torch.finfo(p_mix.dtype).tiny).log()
    q_mix_log = q_mix.clamp_min(torch.finfo(q_mix.dtype).tiny).log()
    token_loss = p_mix * (p_mix_log - q_mix_log)
    token_loss = token_loss.masked_fill(inf_mask, 0.0).sum(dim=-1)
    return _masked_token_mean(token_loss, labels)


def csd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: str = "SS",
) -> torch.Tensor:
    """CSD surrogate with the same detached gradient coefficients as the template."""

    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    logit_delta = student_logits - teacher_logits

    if mode == "SS":
        centered = logit_delta - (student_probs * logit_delta).sum(dim=-1, keepdim=True)
        loss = centered.detach() * student_probs.detach() * student_logits
    elif mode == "TS":
        teacher_centered = logit_delta - (teacher_probs * logit_delta).sum(dim=-1, keepdim=True)
        student_centered = logit_delta - (student_probs * logit_delta).sum(dim=-1, keepdim=True)
        loss1 = teacher_centered.detach() * student_probs.detach() * student_logits
        loss2 = student_centered.detach() * teacher_probs * student_logits
        loss = (loss1 + loss2) / 2.0
    else:
        raise ValueError("CSD mode must be 'SS' or 'TS'.")

    return _masked_token_mean(loss.sum(dim=-1), labels)


def _hpd_align_shared_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align HPD to the shared student/teacher vocabulary prefix."""

    if student_logits.shape[-1] == teacher_logits.shape[-1]:
        return student_logits, teacher_logits, labels

    shared_vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logits = student_logits[..., :shared_vocab_size]
    teacher_logits = teacher_logits[..., :shared_vocab_size]
    labels = labels.masked_fill(labels >= shared_vocab_size, ignore_index)
    return student_logits, teacher_logits, labels


def _hpd_sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    return _sanitize_logits(logits).to(torch.float32)


def _hpd_masked_ratio(mask: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return 0.0
    return ((mask & valid_mask).float().sum() / valid_count.float()).item() * 100.0


def _hpd_masked_stats(prefix: str, values: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    selected = values.detach().masked_select(valid_mask)
    if selected.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_neg_pct": 0.0,
        }
    return {
        f"{prefix}_mean": selected.mean().item(),
        f"{prefix}_min": selected.min().item(),
        f"{prefix}_max": selected.max().item(),
        f"{prefix}_neg_pct": (selected < 0).float().mean().item() * 100.0,
    }


def compute_hpd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    sample_in_fp32: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Hybrid Policy Distillation loss.

    This follows the official HPD implementation: the offline expert token
    receives a hybrid forward/reverse-KL weight, while a token sampled from
    the student receives a negative-only reverse-KL penalty. Only supervised
    response prefixes are retained before vocabulary-sized probability and
    sampling operations, so ignored prompt and padding positions allocate no
    HPD intermediates.

    Adapted from the Apache-2.0 implementation in:
    https://github.com/zwhong714/Hybrid-Policy-Distillation
    """

    if student_logits.ndim != 3 or teacher_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("HPD expects student/teacher logits [batch, sequence, vocab] and labels [batch, sequence].")
    if student_logits.shape[:2] != teacher_logits.shape[:2] or student_logits.shape[:2] != labels.shape:
        raise ValueError("HPD student logits, teacher logits, and labels must share batch/sequence dimensions.")
    if labels.shape[1] < 2:
        raise ValueError("At least two tokens are required for HPD causal distillation.")

    sample_logits = student_logits[..., :-1, :]
    teacher_logits = teacher_logits[..., :-1, :]
    labels = labels[..., 1:]
    sample_logits, teacher_logits, labels = _hpd_align_shared_vocab(
        sample_logits,
        teacher_logits,
        labels,
        ignore_index,
    )
    response_mask = labels.ne(ignore_index)
    if not torch.any(response_mask):
        raise ValueError("The HPD batch contains no response tokens.")
    sample_logits = sample_logits[response_mask]
    teacher_logits = teacher_logits[response_mask]
    labels = labels[response_mask].unsqueeze(-1)

    student_logits_fp32 = _hpd_sanitize_logits(sample_logits)
    teacher_logits_fp32 = _hpd_sanitize_logits(teacher_logits)
    student_log_probs = F.log_softmax(student_logits_fp32, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_fp32, dim=-1)

    active_mask = torch.ones_like(labels, dtype=torch.bool)
    safe_labels = labels

    # Expert-token discrepancy: k1 > 0 means the student underestimates the
    # offline/expert token and should receive a forward-KL reinforcement.
    student_log_prob_expert = student_log_probs.gather(dim=-1, index=safe_labels)
    teacher_log_prob_expert = teacher_log_probs.gather(dim=-1, index=safe_labels)
    k1_gt_raw = (
        teacher_log_prob_expert - student_log_prob_expert
    ) * torch.exp(student_log_prob_expert)
    expert_underestimated = k1_gt_raw > 0
    expert_weight = torch.zeros_like(k1_gt_raw)
    expert_weight[expert_underestimated] = (
        torch.exp(teacher_log_prob_expert)[expert_underestimated] + k1_gt_raw[expert_underestimated]
    )
    expert_weight[~expert_underestimated] = k1_gt_raw[~expert_underestimated]

    # Lightweight on-policy approximation: sample one token from the student
    # at each offline prefix and only retain negative reverse-KL rewards.
    sampling_logits = student_logits_fp32
    if not sample_in_fp32:
        sampling_logits = _sanitize_logits(sample_logits)
    student_probs = torch.softmax(sampling_logits, dim=-1)
    sampled_labels = torch.multinomial(student_probs, num_samples=1)

    student_log_prob_sampled = student_log_probs.gather(dim=-1, index=sampled_labels)
    teacher_log_prob_sampled = teacher_log_probs.gather(dim=-1, index=sampled_labels)
    k1_sample = (
        teacher_log_prob_sampled - student_log_prob_sampled
    ) * torch.exp(student_log_prob_sampled)
    sampled_overestimated = k1_sample < 0
    sampled_weight = torch.zeros_like(k1_sample)
    sampled_weight[sampled_overestimated] = k1_sample[sampled_overestimated]

    # If the student samples a non-expert token that is overestimated, also
    # reinforce the expert token when it was underestimated.
    reinforce_expert = expert_underestimated & sampled_overestimated
    expert_weight[reinforce_expert] += torch.exp(teacher_log_prob_expert)[reinforce_expert]

    sampled_diff = sampled_labels.ne(safe_labels)
    sampled_penalty_active = sampled_diff & sampled_overestimated
    reinforce_expert_active = reinforce_expert

    expert_weight = expert_weight.detach()
    sampled_weight = sampled_weight.detach()
    ce_loss = -student_log_prob_expert
    hpd_loss = (
        -student_log_prob_expert * expert_weight
        - sampled_weight * student_log_prob_sampled * sampled_labels.ne(safe_labels)
    )
    normalizer = active_mask.sum() + 1e-8
    loss = hpd_loss.sum() / normalizer

    metrics = {
        "neg_student_nll_loss": (-student_log_prob_expert.detach()).sum().div(normalizer).item(),
        "ce_loss": ce_loss.sum().div(normalizer).detach().item(),
        "hpd_loss": loss.detach().item(),
        "sampled_diff_pct": _hpd_masked_ratio(sampled_labels.ne(safe_labels), active_mask),
        "adv1_neg_pct": _hpd_masked_ratio(expert_weight < 0, active_mask),
        "adv2_active_pct": _hpd_masked_ratio(sampled_penalty_active, active_mask),
        "mask3_pct": _hpd_masked_ratio(reinforce_expert_active, active_mask),
    }
    metrics.update(_hpd_masked_stats("k1_gt_raw", k1_gt_raw, active_mask))
    metrics.update(_hpd_masked_stats("k1_sample", k1_sample, sampled_diff))
    return loss, metrics


def hpd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_in_fp32: bool = True,
) -> torch.Tensor:
    """Tensor-only HPD entry point for the common loss dispatch table."""

    return compute_hpd_loss(
        student_logits,
        teacher_logits,
        labels,
        sample_in_fp32=sample_in_fp32,
    )[0]


def amid(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    div_name: str = "fkl",
    div_order: str = "pr",
    alpha: float = 0.5,
    lam: float = 0.5,
) -> torch.Tensor:
    """Assistant-mediated interpolated divergence from the reference template.

    ``p`` is the teacher distribution, ``q`` is the student distribution, and
    ``r`` is the assistant distribution constructed from both.  The template
    supports forward KL and AB divergence with any of the four pair orders:
    ``pr``, ``qr``, ``rp``, and ``rq``.
    """

    valid_divergences = {"fkl", "ab"}
    valid_orders = {"pr", "qr", "rp", "rq"}
    if div_name not in valid_divergences:
        raise ValueError(f"Unsupported AMID divergence: {div_name!r}. Expected one of: fkl, ab.")
    if div_order not in valid_orders:
        raise ValueError(f"Unsupported AMID divergence order: {div_order!r}. Expected one of: pr, qr, rp, rq.")
    if not math.isfinite(alpha):
        raise ValueError("AMID alpha must be finite.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("AMID lam must be in [0, 1].")

    # Keep the distribution calculations in float32, matching the numerical
    # precision used by the other KD losses in the active LlamaFactory path.
    p = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    q = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    logp = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    logq = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(student_logits)

    # Construct the assistant distribution.  This is the power-mean form
    # used by the paper reference implementation, including its alpha >= 1 branch.
    if lam == 0.0:
        r, logr = q, logq
    elif lam == 1.0:
        r, logr = p, logp
    elif alpha >= 1.0:
        logr_unnorm = lam * logp + (1.0 - lam) * logq
        r = F.softmax(logr_unnorm, dim=-1, dtype=torch.float32)
        logr = F.log_softmax(logr_unnorm, dim=-1, dtype=torch.float32)
    else:
        t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
        t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq
        logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
        r = F.softmax(logr_unnorm, dim=-1, dtype=torch.float32)
        logr = F.log_softmax(logr_unnorm, dim=-1, dtype=torch.float32)

    if div_name == "fkl":
        distributions = {
            "pr": (p, logp, logr),
            "qr": (q, logq, logr),
            "rp": (r, logr, logp),
            "rq": (r, logr, logq),
        }
        left, left_log, right_log = distributions[div_order]
        token_loss = torch.masked_fill(left * (left_log - right_log), inf_mask, 0).sum(dim=-1)
        return _masked_token_mean(token_loss, labels)

    # AMID's AB branch uses the paper's fixed AB parameters.
    ab_alpha, ab_beta = 0.2, 0.7
    ab_sum = ab_alpha + ab_beta
    if div_order == "pr":
        first_log, second_log = logp, logr
    elif div_order == "qr":
        first_log, second_log = logq, logr
    elif div_order == "rp":
        first_log, second_log = logr, logp
    else:  # div_order == "rq"
        first_log, second_log = logr, logq

    term1 = torch.exp(torch.logsumexp(ab_alpha * first_log + ab_beta * second_log, dim=-1))
    term2 = (ab_alpha / ab_sum) * torch.exp(torch.logsumexp(ab_sum * first_log, dim=-1))
    term3 = (ab_beta / ab_sum) * torch.exp(torch.logsumexp(ab_sum * second_log, dim=-1))
    divergence = -(term1 - term2 - term3) / (ab_alpha * ab_beta)
    safe_divergence = torch.where(torch.isfinite(divergence), divergence, torch.zeros_like(divergence))
    return _masked_token_mean(safe_divergence, labels)


LOSS_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "fkl": forward_kl,
    "rkl": reverse_kl,
    "sfkl": skewed_forward_kl,
    "srkl": skewed_reverse_kl,
    "bdl": bdl,
    "csd": csd,
    "amid": amid,
    "hpd": hpd,
}


def compute_distillation_loss(
    method: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    skew_alpha: float = 0.1,
    amid_div_name: str = "fkl",
    amid_div_order: str = "pr",
    amid_alpha: float = 0.5,
    amid_lam: float = 0.5,
    hpd_sample_in_fp32: bool = True,
    bdl_lambda: float = 0.9,
) -> torch.Tensor:
    if method == "hpd":
        # HPD performs its own causal alignment and shared-vocabulary handling
        # because it also samples from the student distribution.
        return compute_hpd_loss(
            student_logits,
            teacher_logits,
            labels,
            sample_in_fp32=hpd_sample_in_fp32,
        )[0]
    student_logits, teacher_logits, labels = align_causal_logits_and_labels(student_logits, teacher_logits, labels)
    if method not in LOSS_FUNCTIONS:
        raise ValueError(f"Unsupported base distillation method: {method}.")
    if method in {"sfkl", "srkl"}:
        return LOSS_FUNCTIONS[method](student_logits, teacher_logits, labels, alpha=skew_alpha)
    if method == "bdl":
        return bdl(student_logits, teacher_logits, labels, lam=bdl_lambda)
    if method == "amid":
        return amid(
            student_logits,
            teacher_logits,
            labels,
            div_name=amid_div_name,
            div_order=amid_div_order,
            alpha=amid_alpha,
            lam=amid_lam,
        )
    return LOSS_FUNCTIONS[method](student_logits, teacher_logits, labels)
