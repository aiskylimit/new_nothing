"""
SEGDLoss — 3-node semantic graph, multi-layer spectral + representation distillation.

Loss composition:
  total = contrastive_loss + λ_sim * L_sim + λ_spectral * L_spectral

Pipeline:
  Graph / spectral: relative-depth checkpoints (default N=4 → 25/50/75%).
  Graph nodes: R_txt/R_vis = mean-pool; R_all = last token of [vision | text] (both models).
  L_sim: last-layer encode_input — Teacher EOS vs Student EOS (qry + pos);
         if hidden dims differ, student reps go through distiller Linear s→t first.
  Contrastive: one-way InfoNCE q→p on Student encode_input EOS last layer (not symmetric).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.criterions.sgd_loss import (
    count_text_tokens_student,
    count_text_tokens_teacher,
    extract_text_hidden_states,
    extract_vision_hidden_states,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8
_NODE_TYPES = ("txt", "vis", "all")
_CLUSTERS = ("qry", "pos")


def get_align_layer_indices(
    num_hidden_states: int,
    num_align_layers: int = 4,
) -> List[int]:
    L = max(int(num_hidden_states) - 1, 1)
    n = max(int(num_align_layers), 2)
    idxs: List[int] = []
    seen = set()
    for i in range(1, n):
        r = i / n
        idx = round(r * L)
        idx = min(max(int(idx), 1), L)
        if idx in seen:
            continue
        seen.add(idx)
        idxs.append(idx)
    return idxs


POOL_GRAPH = {"txt": "mean", "vis": "mean", "all": "last"}


def pool_tokens(tokens: Optional[torch.Tensor], mode: str = "mean") -> Optional[torch.Tensor]:
    if tokens is None or tokens.numel() == 0:
        return None
    if mode == "last":
        return tokens[-1]
    return tokens.mean(dim=0)


def _concat_cluster_tokens(
    vision: Optional[torch.Tensor],
    text: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    parts = [p for p in (vision, text) if p is not None and p.numel() > 0]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def three_semantic_nodes(
    vision: Optional[torch.Tensor],
    text: Optional[torch.Tensor],
    pools: Dict[str, str],
) -> Dict[str, Optional[torch.Tensor]]:
    r_txt = pool_tokens(text, pools["txt"])
    r_vis = pool_tokens(vision, pools["vis"])
    r_all = pool_tokens(_concat_cluster_tokens(vision, text), pools["all"])
    return {"txt": r_txt, "vis": r_vis, "all": r_all}


def _has_image_feature(
    image_features: Optional[Sequence[Optional[torch.Tensor]]],
    sample_idx: int,
) -> bool:
    return (
        image_features is not None
        and sample_idx < len(image_features)
        and image_features[sample_idx] is not None
        and int(image_features[sample_idx].size(0)) > 0
    )


def extract_cluster_tokens(
    hidden_one_layer: torch.Tensor,
    *,
    is_teacher: bool,
    model_input: Dict[str, torch.Tensor],
    image_features: Optional[Sequence[Optional[torch.Tensor]]],
    sample_idx: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    input_ids = model_input["input_ids"][sample_idx]
    if is_teacher:
        num_text = count_text_tokens_teacher(input_ids)
    else:
        num_text = count_text_tokens_student(input_ids)

    has_image = _has_image_feature(image_features, sample_idx)
    num_vision = int(image_features[sample_idx].size(0)) if has_image else 0

    hidden_for_extract = [hidden_one_layer]
    vision = None
    text = None
    if has_image and num_vision > 0:
        vision = extract_vision_hidden_states(
            hidden_for_extract, sample_idx, num_vision, num_text, is_teacher=is_teacher,
        )[-1]
    if num_text > 0:
        text = extract_text_hidden_states(
            hidden_for_extract, sample_idx, num_text, num_vision,
            is_teacher=is_teacher, has_image=has_image,
        )[-1]
    return vision, text


def align_checkpoint_nodes(
    teacher_batch: List[Dict[str, Dict[str, Optional[torch.Tensor]]]],
    student_batch: List[Dict[str, Dict[str, Optional[torch.Tensor]]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    t_list: List[torch.Tensor] = []
    s_list: List[torch.Tensor] = []
    b = len(teacher_batch)
    for i in range(b):
        for cluster in _CLUSTERS:
            for typ in _NODE_TYPES:
                r_t = teacher_batch[i][cluster][typ]
                r_s = student_batch[i][cluster][typ]
                if r_t is None or r_s is None:
                    continue
                t_list.append(r_t)
                s_list.append(r_s)
    if not t_list:
        return None, None
    return torch.stack(t_list, dim=0), torch.stack(s_list, dim=0)


def build_full_cosine_graph(nodes: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    n = int(nodes.size(0))
    if n < 2:
        return nodes.new_zeros(n, n, dtype=torch.float32)
    xn = F.normalize(nodes.float(), dim=-1)
    logits = xn @ xn.t() / max(float(tau), _EPS)
    eye = torch.eye(n, dtype=torch.bool, device=nodes.device)
    logits = logits.masked_fill(eye, float("-inf"))
    w = torch.softmax(logits, dim=-1)
    return 0.5 * (w + w.t())


def build_normalized_laplacian(w: torch.Tensor) -> torch.Tensor:
    w = w.float()
    deg = w.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    w_norm = deg_inv_sqrt.unsqueeze(1) * w * deg_inv_sqrt.unsqueeze(0)
    eye = torch.eye(w.size(0), device=w.device, dtype=w.dtype)
    return eye - w_norm


def get_eigenspace(lap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(lap)
    return eigvals, eigvecs


def select_k_by_eigengap(
    eigvals: torch.Tensor,
    k_max: int = 0,
    k_min: int = 8,
) -> int:
    n = int(eigvals.numel())
    if n <= 1:
        return 1

    hard_max = n - 1
    if k_max > 0:
        hard_max = min(hard_max, int(k_max))
    hard_max = max(1, hard_max)
    hard_min = max(1, min(int(k_min), hard_max))

    ev = eigvals.detach().float().reshape(-1)
    gaps = ev[1:] - ev[:-1]
    lo = hard_min - 1
    hi = hard_max
    gaps_search = gaps[lo:hi]
    i_local = int(torch.argmax(gaps_search).item())
    k = (lo + i_local) + 1
    return max(hard_min, min(k, hard_max))


def spectral_projector_loss(
    u_t: torch.Tensor,
    u_s: torch.Tensor,
    k: int,
) -> torch.Tensor:
    k_use = min(int(k), u_t.size(1), u_s.size(1))
    if k_use <= 0:
        return u_s.new_zeros(())
    ut = u_t[:, :k_use]
    us = u_s[:, :k_use]
    pt = ut @ ut.t()
    ps = us @ us.t()
    return ((pt - ps) ** 2).sum() / max(ps.size(0), 1)


def representation_sim_loss(r_t: torch.Tensor, r_s: torch.Tensor) -> torch.Tensor:
    if r_t.size(-1) != r_s.size(-1):
        raise ValueError(
            f"L_sim cosine requires equal hidden dims, got teacher={r_t.size(-1)} "
            f"student={r_s.size(-1)}. Project student→teacher before calling this."
        )
    rt = F.normalize(r_t.detach().float(), dim=-1)
    rs = F.normalize(r_s.float(), dim=-1)
    return 1.0 - (rt * rs).sum(dim=-1)


def project_student_reps_for_sim(distiller, student_reps: torch.Tensor, teacher_dim: int) -> torch.Tensor:
    if student_reps.size(-1) == teacher_dim:
        return student_reps

    in_dim = int(student_reps.size(-1))
    projectors = getattr(distiller, "projectors", None)
    if projectors is None or len(projectors) == 0:
        raise ValueError(
            f"L_sim needs student→teacher projector for dims {in_dim}→{teacher_dim}, "
            "but distiller.projectors is empty. Set --num_projectors 1."
        )

    if isinstance(projectors, nn.ModuleDict):
        candidates = []
        if "s2t" in projectors:
            candidates.append(projectors["s2t"])
        candidates.extend(projectors.values())
    else:
        candidates = list(projectors)

    proj = None
    for cand in candidates:
        linear = cand if isinstance(cand, nn.Linear) else next(
            (m for m in cand.modules() if isinstance(m, nn.Linear)), None
        )
        if linear is not None and linear.in_features == in_dim and linear.out_features == teacher_dim:
            proj = cand
            break
    if proj is None:
        raise ValueError(
            f"No distiller projector maps student dim {in_dim} → teacher dim {teacher_dim}."
        )

    proj_param = next(proj.parameters())
    return proj(student_reps.to(dtype=proj_param.dtype)).to(dtype=student_reps.dtype)


def infonce_loss(
    r_q: torch.Tensor,
    r_p: torch.Tensor,
    temperature: float,
    student_model,
) -> torch.Tensor:
    scores = student_model.compute_similarity(r_q, r_p)
    scores = scores.view(r_q.size(0), -1)
    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    target = target * (r_q.size(0) // max(r_p.size(0), 1))
    return F.cross_entropy(scores / max(temperature, _EPS), target)


def _checkpoint_spectral(
    nodes_t: torch.Tensor,
    nodes_s: torch.Tensor,
    *,
    tau_graph: float,
    k_eigen_max: int,
    k_eigen_min: int,
) -> Tuple[torch.Tensor, int, int, int]:
    n = int(nodes_s.size(0))
    if n < 2:
        return nodes_s.new_zeros(()), 1, 1, 1

    with torch.no_grad():
        w_t = build_full_cosine_graph(nodes_t, tau=tau_graph)
        l_t = build_normalized_laplacian(w_t)
        evals_t, u_t_full = get_eigenspace(l_t)
        k_t = select_k_by_eigengap(evals_t, k_max=k_eigen_max, k_min=k_eigen_min)

    w_s = build_full_cosine_graph(nodes_s, tau=tau_graph)
    l_s = build_normalized_laplacian(w_s)
    evals_s, u_s_full = get_eigenspace(l_s)
    k_s = select_k_by_eigengap(evals_s, k_max=k_eigen_max, k_min=k_eigen_min)

    k_avail = min(max(u_t_full.size(1) - 1, 1), max(u_s_full.size(1) - 1, 1))
    k_use = min(k_avail, max(k_eigen_min, min(k_t, k_s)))
    k_use = max(1, k_use)

    kd = spectral_projector_loss(u_t_full[:, :k_use].detach(), u_s_full[:, :k_use], k_use)
    return kd, k_use, k_t, k_s


class SEGDLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        self.args = args
        self.lambda_sim = float(getattr(args, "segd_lambda_sim", 1.0))
        self.lambda_spectral = float(getattr(args, "segd_lambda_spectral", 1.0))
        self.tau_graph = float(getattr(args, "segd_tau_graph", 1.0))
        self.num_align_layers = int(getattr(args, "segd_num_align_layers", 4))
        self.k_eigen_max = int(getattr(args, "segd_k_eigen", 0))
        self.k_eigen_min = int(getattr(args, "segd_k_eigen_min", 8))

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    @staticmethod
    def _zero(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=dtype)

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]
        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]

        batch_size = student_qry_input["input_ids"].size(0)
        device = student_qry_input["input_ids"].device

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(
                teacher_qry_input, output_attentions=False,
            )
            teacher_pos_output = teacher_model.encode_input(
                teacher_pos_input, output_attentions=False,
            )
            (
                teacher_qry_reps,
                teacher_qry_image_features,
                _teacher_qry_attn,
                teacher_qry_hidden_states,
            ) = teacher_qry_output
            (
                teacher_pos_reps,
                teacher_pos_image_features,
                _teacher_pos_attn,
                teacher_pos_hidden_states,
            ) = teacher_pos_output

        student_qry_output = student_model.encode_input(
            student_qry_input, output_attentions=False,
        )
        student_pos_output = student_model.encode_input(
            student_pos_input, output_attentions=False,
        )
        (
            student_qry_reps,
            student_qry_image_features,
            _student_qry_attn,
            student_qry_hidden_states,
        ) = student_qry_output
        (
            student_pos_reps,
            student_pos_image_features,
            _student_pos_attn,
            student_pos_hidden_states,
        ) = student_pos_output

        t_idxs = get_align_layer_indices(len(teacher_qry_hidden_states), self.num_align_layers)
        s_idxs = get_align_layer_indices(len(student_qry_hidden_states), self.num_align_layers)
        n_cp = min(len(t_idxs), len(s_idxs))
        t_idxs, s_idxs = t_idxs[:n_cp], s_idxs[:n_cp]

        spectral_terms: List[torch.Tensor] = []
        k_uses: List[int] = []
        k_teachers: List[int] = []
        k_students: List[int] = []
        n_totals: List[int] = []
        n_vis_qry = 0.0
        n_vis_pos = 0.0

        for m, (t_idx, s_idx) in enumerate(zip(t_idxs, s_idxs)):
            t_hidden = teacher_qry_hidden_states[t_idx]
            s_hidden = student_qry_hidden_states[s_idx]
            t_hidden_pos = teacher_pos_hidden_states[t_idx]
            s_hidden_pos = student_pos_hidden_states[s_idx]
            t_graph_batch = []
            s_graph_batch = []
            for i in range(batch_size):
                t_vis_q, t_txt_q = extract_cluster_tokens(
                    t_hidden, is_teacher=True, model_input=teacher_qry_input,
                    image_features=teacher_qry_image_features, sample_idx=i,
                )
                t_vis_p, t_txt_p = extract_cluster_tokens(
                    t_hidden_pos, is_teacher=True, model_input=teacher_pos_input,
                    image_features=teacher_pos_image_features, sample_idx=i,
                )
                s_vis_q, s_txt_q = extract_cluster_tokens(
                    s_hidden, is_teacher=False, model_input=student_qry_input,
                    image_features=student_qry_image_features, sample_idx=i,
                )
                s_vis_p, s_txt_p = extract_cluster_tokens(
                    s_hidden_pos, is_teacher=False, model_input=student_pos_input,
                    image_features=student_pos_image_features, sample_idx=i,
                )
                t_graph_batch.append({
                    "qry": three_semantic_nodes(t_vis_q, t_txt_q, POOL_GRAPH),
                    "pos": three_semantic_nodes(t_vis_p, t_txt_p, POOL_GRAPH),
                })
                s_graph_batch.append({
                    "qry": three_semantic_nodes(s_vis_q, s_txt_q, POOL_GRAPH),
                    "pos": three_semantic_nodes(s_vis_p, s_txt_p, POOL_GRAPH),
                })

            if m == 0:
                for i in range(batch_size):
                    if (
                        s_graph_batch[i]["qry"]["vis"] is not None
                        and t_graph_batch[i]["qry"]["vis"] is not None
                    ):
                        n_vis_qry += 1.0
                    if (
                        s_graph_batch[i]["pos"]["vis"] is not None
                        and t_graph_batch[i]["pos"]["vis"] is not None
                    ):
                        n_vis_pos += 1.0

            nodes_t, nodes_s = align_checkpoint_nodes(t_graph_batch, s_graph_batch)
            if nodes_t is None or nodes_s is None:
                continue

            n_totals.append(int(nodes_s.size(0)))

            kd_m, k_use, k_t, k_s = _checkpoint_spectral(
                nodes_t, nodes_s,
                tau_graph=self.tau_graph,
                k_eigen_max=self.k_eigen_max,
                k_eigen_min=self.k_eigen_min,
            )
            if not torch.isfinite(kd_m):
                logger.warning("spectral loss non-finite at checkpoint %s; replacing with 0", m)
                kd_m = self._zero(device, nodes_s.dtype)
            spectral_terms.append(kd_m)
            k_uses.append(k_use)
            k_teachers.append(k_t)
            k_students.append(k_s)

        if spectral_terms:
            spectral_loss = torch.stack(spectral_terms).mean()
        else:
            spectral_loss = self._zero(device)

        t_dim = int(teacher_qry_reps.size(-1))
        s_qry_for_sim = project_student_reps_for_sim(distiller, student_qry_reps, t_dim)
        s_pos_for_sim = project_student_reps_for_sim(distiller, student_pos_reps, t_dim)
        sim_loss = torch.cat([
            representation_sim_loss(teacher_qry_reps, s_qry_for_sim),
            representation_sim_loss(teacher_pos_reps, s_pos_for_sim),
        ], dim=0).mean()

        cq, cp = student_qry_reps, student_pos_reps
        if self.world_size > 1:
            all_q = self._dist_gather_tensor(cq)
            all_p = self._dist_gather_tensor(cp)
        else:
            all_q, all_p = cq, cp

        c_loss = infonce_loss(
            all_q, all_p,
            temperature=float(distiller.temperature),
            student_model=student_model,
        )

        sim_weighted = self.lambda_sim * sim_loss
        spectral_weighted = self.lambda_spectral * spectral_loss
        total_loss = c_loss + sim_weighted + spectral_weighted

        def _metric(v: float) -> torch.Tensor:
            return torch.tensor(v, device=device, dtype=torch.float32)

        def _k_at(vals: Sequence[int], i: int) -> float:
            return float(vals[i]) if i < len(vals) else -1.0

        n_total = float(n_totals[0]) if n_totals else 0.0
        k_mean = float(sum(k_uses) / len(k_uses)) if k_uses else 0.0
        k_t_mean = float(sum(k_teachers) / len(k_teachers)) if k_teachers else 0.0
        k_s_mean = float(sum(k_students) / len(k_students)) if k_students else 0.0

        return {
            "loss": total_loss,
            "contrastive_loss": c_loss.detach(),
            "sim_loss": sim_loss.detach(),
            "segd_loss": spectral_loss.detach(),
            "spectral_kd_loss": spectral_loss.detach(),
            "sim_weighted": sim_weighted.detach(),
            "spectral_weighted": spectral_weighted.detach(),
            "segd_lambda_sim": _metric(self.lambda_sim),
            "segd_lambda_spectral": _metric(self.lambda_spectral),
            "batch_size": _metric(float(batch_size)),
            "n_total": _metric(n_total),
            "n_checkpoints": _metric(float(n_cp)),
            "n_vis_nodes_qry": _metric(n_vis_qry),
            "n_vis_nodes_pos": _metric(n_vis_pos),
            "segd_k_eigen": _metric(k_mean),
            "segd_k_eigen_teacher": _metric(k_t_mean),
            "segd_k_eigen_student": _metric(k_s_mean),
            "segd_k_eigen_0": _metric(_k_at(k_uses, 0)),
            "segd_k_eigen_1": _metric(_k_at(k_uses, 1)),
            "segd_k_eigen_2": _metric(_k_at(k_uses, 2)),
            "segd_k_eigen_3": _metric(_k_at(k_uses, 3)),
            "segd_layer_teacher_0": _metric(float(t_idxs[0]) if n_cp > 0 else -1.0),
            "segd_layer_teacher_1": _metric(float(t_idxs[1]) if n_cp > 1 else -1.0),
            "segd_layer_teacher_2": _metric(float(t_idxs[2]) if n_cp > 2 else -1.0),
            "segd_layer_teacher_3": _metric(float(t_idxs[3]) if n_cp > 3 else -1.0),
            "segd_layer_student_0": _metric(float(s_idxs[0]) if n_cp > 0 else -1.0),
            "segd_layer_student_1": _metric(float(s_idxs[1]) if n_cp > 1 else -1.0),
            "segd_layer_student_2": _metric(float(s_idxs[2]) if n_cp > 2 else -1.0),
            "segd_layer_student_3": _metric(float(s_idxs[3]) if n_cp > 3 else -1.0),
        }
