"""Align response-to-vision attention over shared semantic regions.

Cluster teacher vision-token hidden states using semantic and spatial distances.
Map those clusters to student patches by normalized patch coordinates. For each
paired decoder layer, project head-averaged response attention onto the regions
and weight the KL divergence by the teacher's attention entropy.

The standalone criterion mixes supervised CE with the SCVA loss. The joint
criteria reuse the same region and attention code with their own loss weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from src.model.scva_attention import resolve_layer_pairs


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
    *,
    half_open: bool = False,
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
    return resolve_layer_pairs(len(s_attns), len(t_attns), n_pairs, low_pct, high_pct, half_open=half_open)


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
        eps = max(float(np.percentile(upper, eps_percentile)), np.finfo(np.float64).eps)
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
    teacher_coords = _patch_centers(teacher_labels.numel(), teacher_grid[0], teacher_grid[1], device)
    student_coords = _patch_centers(n_student_tokens, student_grid[0], student_grid[1], device)
    nearest_teacher = torch.cdist(student_coords, teacher_coords).argmin(dim=-1)
    return F.one_hot(teacher_labels[nearest_teacher], num_classes=n_clusters).float()


@dataclass
class SemanticRegions:
    teacher_positions: torch.Tensor
    student_positions: torch.Tensor
    teacher_assignment: torch.Tensor
    student_assignment: torch.Tensor
    teacher_grid: tuple[int, int]


class SCVACriterion(nn.Module):
    """Standalone SCVA-only criterion. See module docstring for the loss form."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = float(getattr(args, "scva_alpha", 0.5))
        self.weight = float(getattr(args, "scva_weight", 1.0))
        # Public legacy settings retained for callers inspecting old recipes.
        self.n_clusters = max(int(getattr(args, "scva_n_clusters", 16)), 1)
        self.kmeans_iters = max(int(getattr(args, "scva_kmeans_iters", 10)), 1)
        self.attention_layer = int(getattr(args, "scva_attention_layer", -1))
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

        scva_kd = self._scva_loss(
            student_outputs,
            teacher_outputs,
            student_inputs,
            teacher_inputs,
            layer_pairs=getattr(distiller, "scva_layer_pairs", None),
        )
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

    @torch.no_grad()
    def build_regions(self, student_outputs, teacher_outputs, student_inputs, teacher_inputs):
        teacher_hidden = _last_hidden(teacher_outputs)
        t_mask = getattr(teacher_outputs, "vision_feature_mask", None)
        s_mask = getattr(student_outputs, "vision_feature_mask", None)
        if t_mask is None or s_mask is None:
            return {}

        regions = {}
        for b in range(teacher_hidden.shape[0]):
            t_idx = t_mask[b].nonzero(as_tuple=True)[0]
            s_idx = s_mask[b].nonzero(as_tuple=True)[0]
            if min(t_idx.numel(), s_idx.numel()) < self.min_vision_tokens:
                continue
            # Qwen grids are packed across the batch, without rows for text-only samples.
            image_idx = len(t_mask[:b].any(dim=-1).nonzero())
            teacher_grid = _grid_from_inputs(teacher_inputs, image_idx, t_idx.numel())
            student_grid = _grid_from_inputs(student_inputs, image_idx, s_idx.numel())
            labels = _cluster_vision_tokens_dbscan(
                teacher_hidden[b, t_idx], *teacher_grid, self.spatial_weight,
                self.dbscan_min_samples, self.dbscan_eps_percentile,
            )
            teacher_assignment, remapped = _cluster_onehot_from_labels(labels)
            student_assignment = _map_teacher_clusters_to_student_onehot(
                remapped, teacher_grid, student_grid, s_idx.numel(), teacher_assignment.shape[-1],
            )
            regions[b] = SemanticRegions(t_idx, s_idx, teacher_assignment, student_assignment, teacher_grid)
        return regions

    def _scva_loss(
        self,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
        layer_pairs: Optional[list[tuple[int, int]]] = None,
        regions: Optional[dict[int, SemanticRegions]] = None,
        smoothing: Optional[float] = None,
    ) -> torch.Tensor:
        counts: Dict[str, int] = {
            "valid": 0,
            "skipped_mask": 0,
            "skipped_attention": 0,
            "skipped_no_teacher_labels": 0,
            "skipped_no_student_labels": 0,
            "skipped_empty_teacher_response": 0,
            "skipped_empty_student_response": 0,
        }
        self._last_scva_counts = counts

        teacher_hidden = _last_hidden(teacher_outputs)        # [B, L_t, D]
        if regions is None:
            regions = self.build_regions(student_outputs, teacher_outputs, student_inputs, teacher_inputs)
        counts["skipped_mask"] = teacher_hidden.shape[0] - len(regions)
        if not regions:
            return teacher_hidden.new_zeros(())

        t_sparse = getattr(teacher_outputs, "scva_attentions", None)
        s_sparse = getattr(student_outputs, "scva_attentions", None)
        use_sparse = isinstance(t_sparse, dict) and isinstance(s_sparse, dict)
        t_attns = getattr(teacher_outputs, "attentions", None)
        s_attns = getattr(student_outputs, "attentions", None)

        if use_sparse:
            if layer_pairs is None:
                common_layers = sorted(set(s_sparse).intersection(t_sparse))
                layer_pairs = [(index, index) for index in common_layers]
        elif t_attns and s_attns:
            layer_pairs = _resolve_layer_pairs(
                s_attns,
                t_attns,
                n_pairs=self.n_layer_pairs,
                low_pct=self.layer_low_pct,
                high_pct=self.layer_high_pct,
                half_open=smoothing is not None,
            )
        else:
            layer_pairs = []
        if not layer_pairs:
            counts["skipped_attention"] = int(teacher_hidden.shape[0])
            return teacher_hidden.new_zeros(())

        t_labels = teacher_inputs.get("labels")
        s_labels = student_inputs.get("labels")

        sample_losses = []
        for b, region in regions.items():
            t_v_idx = region.teacher_positions
            s_v_idx = region.student_positions
            n_t_v = t_v_idx.numel()
            n_s_v = s_v_idx.numel()

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

            # Compute SCVA loss for each (student_layer, teacher_layer) pair and
            # average the results. Attention tensors are indexed as
            # attns[layer][batch] to avoid loading all heads into memory at once.
            pair_losses: list[torch.Tensor] = []
            for s_layer, t_layer in layer_pairs:
                # Sparse captures are already response-to-vision distributions;
                # the fallback path slices head-averaged full attention below.
                if use_sparse:
                    if s_layer not in s_sparse or t_layer not in t_sparse:
                        continue
                    if b >= len(s_sparse[s_layer]) or b >= len(t_sparse[t_layer]):
                        continue
                    t_resp_to_vis = t_sparse[t_layer][b].float()
                    s_resp_to_vis = s_sparse[s_layer][b].float()
                    if t_resp_to_vis.shape[-1] != n_t_v or s_resp_to_vis.shape[-1] != n_s_v:
                        continue
                else:
                    t_resp_to_vis = t_attns[t_layer][b][:, t_r_idx[:, None], t_v_idx].float().mean(dim=0)
                    s_resp_to_vis = s_attns[s_layer][b][:, s_r_idx[:, None], s_v_idx].float().mean(dim=0)

                # Normalize both paths before cluster aggregation.
                t_resp_to_vis = t_resp_to_vis / t_resp_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                s_resp_to_vis = s_resp_to_vis / s_resp_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                # Crop to the shorter response length defensively.
                T_aligned = min(t_resp_to_vis.shape[0], s_resp_to_vis.shape[0])
                t_resp_to_vis = t_resp_to_vis[:T_aligned]
                s_resp_to_vis = s_resp_to_vis[:T_aligned]

                # Cluster-level distributions (T_aligned × M).
                alpha = 1e-12 if smoothing is None else smoothing
                r_T = (t_resp_to_vis @ region.teacher_assignment) + alpha
                r_S = (s_resp_to_vis @ region.student_assignment) + alpha
                r_T = r_T / r_T.sum(dim=-1, keepdim=True)
                r_S = r_S / r_S.sum(dim=-1, keepdim=True)

                # Teacher-attention entropy → per-step confidence weight u_t.
                h_t = -(t_resp_to_vis.clamp_min(1e-12).log() * t_resp_to_vis).sum(dim=-1)
                u = (-h_t / max(math.log(max(n_t_v, 2)), 1e-6)).exp()   # [T_aligned]

                kl = F.kl_div(r_S.log(), r_T, reduction="none").sum(dim=-1)  # [T_aligned]
                pair_losses.append((u * kl).sum() / u.sum().clamp_min(1e-6))

            # Average across the selected layer pairs for this sample.
            if not pair_losses:
                counts["skipped_attention"] += 1
                continue
            sample_losses.append(torch.stack(pair_losses).mean())
            counts["valid"] += 1

        if not sample_losses:
            return teacher_hidden.new_zeros(())
        return torch.stack(sample_losses).mean()
