"""Semantic-Cluster Visual Attention (SCVA) distillation.

For each generation step t, instead of forcing the student to match the
teacher's per-patch visual attention a_T^(t) — which is overly strict because
both can attend to slightly different vision tokens while focusing on the same
semantic region — we cluster the teacher's vision tokens into M semantic groups
and align the *cluster-level* attention distributions r_T^(t), r_S^(t) in R^M.

Loss per draft.pdf §SCVA:
    L_SCVA = ( Σ_t u_t · KL(r_T^(t) || r_S^(t)) ) / Σ_t u_t
    u_t    = exp( -H(a_T^(t)) / log n )       (teacher-confidence weight)
    r_X^(t)(m) = Σ_{i ∈ C_m} a_X^(t)(i)        (cluster-level attention)

The single-method criterion below mixes L_SCVA with supervised CE following the
EM-KD / SRE convention:
    L = α · L_CE + (1 - α) · w · L_SCVA
This keeps the standalone-SCVA criterion compositionally similar to the others
in this repo. The joint draft method (SCVA + CGKD) uses the draft's additive
formula directly and calls ``SCVACriterion._scva_loss`` to access the pure KD
component.

Assumptions:
  - Teacher vision tokens are clustered by semantic + spatial distance, then
    each teacher cluster is projected onto the student patch grid by normalized
    patch coordinates. This allows teacher/student vision-token counts to differ.
  - ``output.attentions`` is populated (forced by ``_force_eager_attention`` in
    src/model/model.py).
  - Instead of a single attention layer, SCVA selects N_LAYER_PAIRS evenly-spaced
    student layers in [LAYER_LOW_PCT, LAYER_HIGH_PCT] of the stack and maps each
    to its teacher counterpart by the layer-count ratio. The per-layer losses are
    averaged into the final SCVA term.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdbscan
from sklearn.cluster import DBSCAN


IGNORE_INDEX = -100

# Default hyper-parameters for the multi-layer selection strategy.
_DEFAULT_N_LAYER_PAIRS = 4
_DEFAULT_LAYER_LOW_PCT = 0.4
_DEFAULT_LAYER_HIGH_PCT = 0.7


def _last_hidden(outputs) -> torch.Tensor:
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError("SCVA requires model outputs with hidden_states.")
    return hs[-1]


def _resolve_layer_pairs(
    s_attns,
    t_attns,
    n_pairs: int = _DEFAULT_N_LAYER_PAIRS,
    low_pct: float = _DEFAULT_LAYER_LOW_PCT,
    high_pct: float = _DEFAULT_LAYER_HIGH_PCT,
) -> list[tuple[int, int]]:
    """
    Select n_pairs evenly-spaced student layers in [low_pct, high_pct] of the
    student attention stack, then map each to its teacher counterpart by the
    ratio of total layer counts.

    The number of layers for each model is inferred from the length of its
    attention tuple (one tensor per layer, as returned by HuggingFace models
    when output_attentions=True).

    Args:
        s_attns:   Tuple of student attention tensors (one per layer).
        t_attns:   Tuple of teacher attention tensors (one per layer).
        n_pairs:   Number of (student, teacher) layer pairs to return.
        low_pct:   Lower bound of the selection range as a fraction of total
                   student layers (default 0.4 → 40%).
        high_pct:  Upper bound of the selection range as a fraction of total
                   student layers (default 0.7 → 70%).

    Returns:
        List of (student_layer_idx, teacher_layer_idx) pairs, deduplicated and
        sorted by student index.  May be shorter than n_pairs if the index
        range collapses (e.g. very shallow models).
    """
    n_s = len(s_attns)
    n_t = len(t_attns)
    if n_s == 0 or n_t == 0:
        return []

    # Inclusive index range within the student stack
    lo = int(math.floor(low_pct * n_s))
    hi = int(math.floor(high_pct * n_s))
    lo = max(0, min(lo, n_s - 1))
    hi = max(lo, min(hi, n_s - 1))

    # n_pairs evenly-spaced indices inside [lo, hi]
    if n_pairs == 1 or lo == hi:
        student_indices = [lo]
    else:
        step = (hi - lo) / (n_pairs - 1)
        student_indices = sorted(
            {min(int(round(lo + i * step)), hi) for i in range(n_pairs)}
        )

    # Map each student index to the teacher index by layer-count ratio.
    # teacher_layer = round(student_layer * n_t / n_s), clamped to [0, n_t-1].
    ratio = n_t / n_s
    pairs: list[tuple[int, int]] = []
    seen_s: set[int] = set()
    for s_idx in student_indices:
        if s_idx in seen_s:
            continue
        seen_s.add(s_idx)
        t_idx = min(int(round(s_idx * ratio)), n_t - 1)
        pairs.append((s_idx, t_idx))

    return pairs


def _factor_grid(n_tokens: int, aspect: float = 1.0) -> tuple[int, int]:
    """Infer a patch grid (rows, cols) whose area matches n_tokens."""
    if n_tokens <= 1:
        return 1, max(n_tokens, 1)

    side = int(math.sqrt(n_tokens))
    if side * side == n_tokens:
        return side, side

    aspect = max(float(aspect), 1e-6)
    best_rows, best_cols, best_score = 1, n_tokens, float("inf")
    for rows in range(1, int(math.sqrt(n_tokens)) + 1):
        if n_tokens % rows != 0:
            continue
        cols = n_tokens // rows
        score = abs(math.log((cols / rows) / aspect))
        if score < best_score:
            best_rows, best_cols, best_score = rows, cols, score
    return best_rows, best_cols


def _grid_from_inputs(inputs: Dict[str, Any], sample_idx: int, n_tokens: int) -> tuple[int, int]:
    """Best-effort image-token grid inference for Qwen-style and Llava-style inputs."""
    image_grid_thw = inputs.get("image_grid_thw")
    if torch.is_tensor(image_grid_thw) and image_grid_thw.ndim == 2 and sample_idx < image_grid_thw.shape[0]:
        t, h, w = [int(v) for v in image_grid_thw[sample_idx].detach().cpu().tolist()]
        raw = max(t * h * w, 1)
        if raw == n_tokens:
            return max(t * h, 1), max(w, 1)
        merge_sq = raw // max(n_tokens, 1)
        merge = int(math.sqrt(merge_sq))
        if merge > 0 and merge * merge == merge_sq and h % merge == 0 and w % merge == 0:
            return max(t * (h // merge), 1), max(w // merge, 1)

    aspect = 1.0
    return _factor_grid(n_tokens, aspect=aspect)


def _patch_centers(n_tokens: int, grid_rows: int, grid_cols: int, device: torch.device) -> torch.Tensor:
    """Normalized patch centers in [0, 1] x [0, 1]."""
    if grid_rows * grid_cols != n_tokens:
        grid_rows, grid_cols = _factor_grid(n_tokens)
    idx = torch.arange(n_tokens, device=device)
    rows = torch.div(idx, grid_cols, rounding_mode="floor").float()
    cols = (idx % grid_cols).float()
    x = (cols + 0.5) / max(grid_cols, 1)
    y = (rows + 0.5) / max(grid_rows, 1)
    return torch.stack([x, y], dim=-1)


def _vision_distance_matrix(
    hidden_states: torch.Tensor,
    grid_rows: int,
    grid_cols: int,
    spatial_weight: float,
) -> np.ndarray:
    """Distance matrix matching span_propose_attn: cosine distance + spatial distance."""
    hidden_norm = F.normalize(hidden_states.float(), p=2, dim=-1)
    cosine_distance = 1.0 - hidden_norm @ hidden_norm.T

    coords = _patch_centers(hidden_states.size(0), grid_rows, grid_cols, hidden_states.device)
    spatial_distance = torch.cdist(coords, coords)
    total = cosine_distance + float(spatial_weight) * spatial_distance
    total = (total + total.T) / 2
    total = total.clamp_min(0)
    total.fill_diagonal_(0)
    return total.detach().cpu().numpy().astype(np.float64)


def _cluster_vision_tokens_dbscan(
    hidden_states: torch.Tensor,
    grid_rows: int,
    grid_cols: int,
    spatial_weight: float,
    min_samples: int,
    eps_percentile: float,
) -> torch.Tensor:
    if hidden_states.size(0) <= 1:
        return torch.zeros(hidden_states.size(0), dtype=torch.long, device=hidden_states.device)

    distance_matrix = _vision_distance_matrix(hidden_states, grid_rows, grid_cols, spatial_weight)
    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    if upper.size == 0:
        labels = np.zeros(hidden_states.size(0), dtype=np.int64)
    else:
        eps = float(np.percentile(upper, eps_percentile))
        min_samples = max(1, min(int(min_samples), hidden_states.size(0)))
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distance_matrix)

        if np.all(labels == -1):
            labels = np.zeros(hidden_states.size(0), dtype=np.int64)
    return torch.tensor(labels, dtype=torch.long, device=hidden_states.device)


def _cluster_onehot_from_labels(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real_mask = labels.ge(0)
    noise_mask = labels.lt(0)

    real_unique = torch.unique(labels[real_mask], sorted=True)

    remapped = labels.new_full(labels.shape, -1)

    # Remap normal clusters to 0, 1, ..., K-1
    for new_id, old_id in enumerate(real_unique):
        remapped[labels.eq(old_id)] = new_id

    # Collect all noise tokens into a single extra cluster
    if noise_mask.any():
        noise_cluster_id = int(real_unique.numel())
        remapped[noise_mask] = noise_cluster_id

    num_clusters = int(real_unique.numel()) + int(noise_mask.any())

    if num_clusters == 0:
        empty = labels.new_zeros((labels.numel(), 0), dtype=torch.float)
        return empty, remapped

    onehot = F.one_hot(remapped, num_classes=num_clusters).float()
    return onehot, remapped


def _map_teacher_clusters_to_student_onehot(
    teacher_labels: torch.Tensor,
    teacher_grid: tuple[int, int],
    student_grid: tuple[int, int],
    n_student_tokens: int,
    n_clusters: int,
) -> torch.Tensor:
    """Map teacher cluster labels onto student patch indices by normalized coordinates."""
    device = teacher_labels.device
    student_onehot = torch.zeros(n_student_tokens, n_clusters, device=device)
    valid_teacher = teacher_labels.ge(0)
    if n_clusters == 0 or not valid_teacher.any() or n_student_tokens == 0:
        return student_onehot

    teacher_coords = _patch_centers(teacher_labels.numel(), teacher_grid[0], teacher_grid[1], device)
    student_coords = _patch_centers(n_student_tokens, student_grid[0], student_grid[1], device)
    nearest_student = torch.cdist(teacher_coords, student_coords).argmin(dim=-1)
    for teacher_idx in valid_teacher.nonzero(as_tuple=True)[0]:
        student_onehot[nearest_student[teacher_idx], teacher_labels[teacher_idx]] = 1.0
    return student_onehot


class SCVACriterion(nn.Module):
    """Standalone SCVA-only criterion. See module docstring for the loss form."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = float(getattr(args, "scva_alpha", 0.5))
        self.weight = float(getattr(args, "scva_weight", 1.0))
        # Kept for CLI/backward compatibility. SCVA now follows span_propose_attn
        # and uses density clusters instead of fixed-k Lloyd k-means.
        self.n_clusters = max(int(getattr(args, "scva_n_clusters", 16)), 1)
        self.kmeans_iters = max(int(getattr(args, "scva_kmeans_iters", 10)), 1)
        self.min_vision_tokens = max(int(getattr(args, "scva_min_vision_tokens", 4)), 1)
        self.spatial_weight = float(getattr(args, "scva_spatial_weight", 0.1))
        self.dbscan_min_samples = max(int(getattr(args, "scva_dbscan_min_samples", 8)), 1)
        self.dbscan_eps_percentile = float(getattr(args, "scva_dbscan_eps_percentile", 3.0))
        # Multi-layer alignment parameters
        self.n_layer_pairs = max(int(getattr(args, "scva_n_layer_pairs", _DEFAULT_N_LAYER_PAIRS)), 1)
        self.layer_low_pct = float(getattr(args, "scva_layer_low_pct", _DEFAULT_LAYER_LOW_PCT))
        self.layer_high_pct = float(getattr(args, "scva_layer_high_pct", _DEFAULT_LAYER_HIGH_PCT))

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running SCVA.")

        student_outputs = distiller.student(**student_inputs)
        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        return self.compute_losses(distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs)

    def compute_losses(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        ce = student_outputs.loss
        if ce is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        scva_kd = self._scva_loss(student_outputs, teacher_outputs, student_inputs, teacher_inputs)
        total = self.alpha * ce + (1.0 - self.alpha) * self.weight * scva_kd
        out: Dict[str, torch.Tensor] = {
            "loss": total,
            "hard_loss": ce.detach(),
            "scva_loss": scva_kd.detach(),
        }
        out.update(self._scalarize_counts(ce))
        return out

    def _scalarize_counts(self, like: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert the per-step validity counters into bf16-safe scalar tensors
        (so DistillTrainer.log can mean-reduce them across micro-batches)."""
        counts = getattr(self, "_last_scva_counts", None) or {}
        return {f"scva_{k}": like.new_tensor(float(v)) for k, v in counts.items()}

    def _scva_loss(
        self,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        counts: Dict[str, int] = {
            "valid": 0,
            "skipped_mask": 0,
            "skipped_attention": 0,
            "skipped_no_clusters": 0,
            "skipped_no_teacher_labels": 0,
            "skipped_no_student_labels": 0,
            "skipped_empty_teacher_response": 0,
            "skipped_empty_student_response": 0,
        }
        self._last_scva_counts = counts

        teacher_hidden = _last_hidden(teacher_outputs)        # [B, L_t, D]
        teacher_vision_mask = getattr(teacher_outputs, "vision_feature_mask", None)
        student_vision_mask = getattr(student_outputs, "vision_feature_mask", None)
        if teacher_vision_mask is None or student_vision_mask is None:
            counts["skipped_mask"] = int(teacher_hidden.shape[0])
            return teacher_hidden.new_zeros(())

        t_attns = getattr(teacher_outputs, "attentions", None)
        s_attns = getattr(student_outputs, "attentions", None)
        if not t_attns or not s_attns:
            counts["skipped_attention"] = int(teacher_hidden.shape[0])
            return teacher_hidden.new_zeros(())

        # Select n_layer_pairs evenly-spaced student layers in [layer_low_pct,
        # layer_high_pct] of the stack; map each to its teacher counterpart by
        # the ratio n_teacher_layers / n_student_layers.
        layer_pairs = _resolve_layer_pairs(
            s_attns,
            t_attns,
            n_pairs=self.n_layer_pairs,
            low_pct=self.layer_low_pct,
            high_pct=self.layer_high_pct,
        )
        if not layer_pairs:
            counts["skipped_attention"] = int(teacher_hidden.shape[0])
            return teacher_hidden.new_zeros(())

        t_labels = teacher_inputs.get("labels")
        s_labels = student_inputs.get("labels")

        sample_losses = []
        for b in range(teacher_hidden.shape[0]):
            t_v_mask = teacher_vision_mask[b].to(device=teacher_hidden.device, dtype=torch.bool)
            s_v_mask = student_vision_mask[b].to(device=teacher_hidden.device, dtype=torch.bool)
            if t_v_mask.sum().item() < self.min_vision_tokens or s_v_mask.sum().item() < self.min_vision_tokens:
                counts["skipped_mask"] += 1
                continue

            t_v_idx = t_v_mask.nonzero(as_tuple=True)[0]
            s_v_idx = s_v_mask.nonzero(as_tuple=True)[0]
            n_t_v = int(t_v_idx.numel())
            n_s_v = int(s_v_idx.numel())

            if t_labels is None:
                counts["skipped_no_teacher_labels"] += 1
                continue
            if s_labels is None:
                counts["skipped_no_student_labels"] += 1
                continue
            t_r_idx = (t_labels[b].to(teacher_hidden.device) != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            s_r_idx = (s_labels[b].to(teacher_hidden.device) != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if t_r_idx.numel() == 0:
                counts["skipped_empty_teacher_response"] += 1
                continue
            if s_r_idx.numel() == 0:
                counts["skipped_empty_student_response"] += 1
                continue

            # Cluster teacher vision-token hidden states using the same
            # semantic + spatial distance idea as span_propose_attn.py, then
            # map teacher cluster membership onto the student patch grid.
            # Clustering is done once per sample and shared across all layer pairs.
            with torch.no_grad():
                teacher_grid = _grid_from_inputs(teacher_inputs, b, n_t_v)
                student_grid = _grid_from_inputs(student_inputs, b, n_s_v)
                cluster_labels = _cluster_vision_tokens_dbscan(
                    teacher_hidden[b, t_v_idx].float(),
                    teacher_grid[0],
                    teacher_grid[1],
                    self.spatial_weight,
                    self.dbscan_min_samples,
                    self.dbscan_eps_percentile,
                )
                teacher_cluster_onehot, remapped_labels = _cluster_onehot_from_labels(cluster_labels)
                student_cluster_onehot = _map_teacher_clusters_to_student_onehot(
                    remapped_labels,
                    teacher_grid,
                    student_grid,
                    n_s_v,
                    teacher_cluster_onehot.shape[-1],
                )
            if teacher_cluster_onehot.shape[-1] == 0 or student_cluster_onehot.sum().item() == 0:
                counts["skipped_no_clusters"] += 1
                continue

            # Compute SCVA loss for each (student_layer, teacher_layer) pair and
            # average the results. Attention tensors are indexed as
            # attns[layer][batch] to avoid loading all heads into memory at once.
            pair_losses: list[torch.Tensor] = []
            for s_layer, t_layer in layer_pairs:
                # Head-averaged attention rows for this sample. Use float for KL stability.
                t_attn_b = t_attns[t_layer].float().mean(dim=1)[b]   # [L_t, L_t]
                s_attn_b = s_attns[s_layer].float().mean(dim=1)[b]   # [L_s, L_s]

                # Slice response→vision sub-matrix and renormalize to a distribution.
                t_resp_to_vis = t_attn_b[t_r_idx][:, t_v_idx]       # [T_t, n_t_v]
                t_resp_to_vis = t_resp_to_vis / t_resp_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                s_resp_to_vis = s_attn_b[s_r_idx][:, s_v_idx]       # [T_s, n_s_v]
                s_resp_to_vis = s_resp_to_vis / s_resp_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                # Crop to the shorter response length defensively.
                T_aligned = min(t_resp_to_vis.shape[0], s_resp_to_vis.shape[0])
                t_resp_to_vis = t_resp_to_vis[:T_aligned]
                s_resp_to_vis = s_resp_to_vis[:T_aligned]

                # Cluster-level distributions (T_aligned × M).
                r_T = (t_resp_to_vis @ teacher_cluster_onehot).clamp_min(1e-12)
                r_S = (s_resp_to_vis @ student_cluster_onehot).clamp_min(1e-12)
                r_T = r_T / r_T.sum(dim=-1, keepdim=True)
                r_S = r_S / r_S.sum(dim=-1, keepdim=True)

                # Teacher-attention entropy → per-step confidence weight u_t.
                h_t = -(t_resp_to_vis.clamp_min(1e-12).log() * t_resp_to_vis).sum(dim=-1)
                u = (-h_t / max(math.log(max(n_t_v, 2)), 1e-6)).exp()   # [T_aligned]

                kl = F.kl_div(r_S.log(), r_T, reduction="none").sum(dim=-1)  # [T_aligned]
                pair_losses.append((u * kl).sum() / u.sum().clamp_min(1e-6))

            # Average across the selected layer pairs for this sample.
            sample_losses.append(torch.stack(pair_losses).mean())
            counts["valid"] += 1

        if not sample_losses:
            return teacher_hidden.new_zeros(())
        return torch.stack(sample_losses).mean()