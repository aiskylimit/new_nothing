"""§5.3 (Eq 25) — khởi tạo student ˜θ bằng model merging.

F_merge ∈ {TA, TIES, DARE-TIES, TSV, Iso-C}. Thao tác trên **delta-weight** của
từng module: ΔW_m = s · B_m A_m  (s = α/r; các expert hiện tại s = 1).

Đầu ra là một adapter LoRA rank ``out_rank`` (mặc định = r) tương thích với
``eval/run_eval.py`` và ``train/phase2_distill.py`` (dùng làm --teacher_adapter):
ΔW_merged được refactor về (A, B) qua SVD cắt ngắn, alpha = rank ⇒ scaling = 1.

    python -m proposal.merge --phase1_dir outputs/proposal/phase1 \
        --strategy tsv --out_dir outputs/proposal/merge_tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file

STRATEGIES = ["ta", "ties", "dare_ties", "tsv", "iso_c", "ctm"]

_A_SUF = ".lora_A.weight"
_B_SUF = ".lora_B.weight"


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _load_expert_factors(expert_dir: str, device: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Trả về {module_prefix: (A (r,din), B (dout,r))} — đã gộp scaling s vào B."""
    cfg = json.load(open(os.path.join(expert_dir, "adapter_config.json")))
    s = float(cfg["lora_alpha"]) / float(cfg["r"])
    sd = load_file(os.path.join(expert_dir, "adapter_model.safetensors"))
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for k in sd:
        if not k.endswith(_A_SUF):
            continue
        prefix = k[: -len(_A_SUF)]
        A = sd[k].to(device=device, dtype=torch.float32)
        B = s * sd[prefix + _B_SUF].to(device=device, dtype=torch.float32)
        out[prefix] = (A, B)
    return out


def _refactor_to_lora(dW: torch.Tensor, out_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ΔW (dout, din) -> (A: (k, din), B: (dout, k)) sao cho B @ A ≈ ΔW, k=out_rank.

    alpha = rank ⇒ scaling = 1 ⇒ không cần chia s.
    """
    k = min(out_rank, min(dW.shape))
    q = min(k + 8, min(dW.shape))
    U, S, V = torch.svd_lowrank(dW, q=q, niter=4)     # dW ≈ U diag(S) Vᵀ
    U_k, S_k, V_k = U[:, :k], S[:k], V[:, :k]
    B = U_k * S_k.unsqueeze(0)                        # (dout, k)
    A = V_k.t()                                       # (k, din)
    return A.contiguous(), B.contiguous()


# --------------------------------------------------------------------------- #
# Chiến lược merge — mỗi hàm nhận list[ΔW_m] (dout, din) -> ΔW_merged
# --------------------------------------------------------------------------- #
def _merge_ta(deltas: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    out = torch.zeros_like(deltas[0])
    for w, d in zip(weights, deltas):
        out += w * d
    return out


def _refactor_ta_fast(A_list, B_list, weights, out_rank):
    """TA giữ nguyên cấu trúc low-rank: ΔW = Σ w_m B_m A_m = [√w B_1|..] @ [√w A_1;..].

    SVD chỉ trên khối (M·r × M·r) thay vì (dout × din) — nhanh hơn nhiều bậc.
    """
    Bcat = torch.cat([float(w) ** 0.5 * B for w, B in zip(weights, B_list)], dim=1)  # (dout, Mr)
    Acat = torch.cat([float(w) ** 0.5 * A for w, A in zip(weights, A_list)], dim=0)  # (Mr, din)
    # QR để rút gọn: Bcat = Qb Rb ; Acat^T = Qa Ra  -> ΔW = Qb (Rb Ra^T? ) ...
    Qb, Rb = torch.linalg.qr(Bcat)                    # Qb (dout, Mr), Rb (Mr, Mr)
    Qa, Ra = torch.linalg.qr(Acat.t())               # Qa (din, Mr), Ra (Mr, Mr)
    core = Rb @ Ra.t()                                # (Mr, Mr)
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    k = min(out_rank, S.numel())
    B = (Qb @ U[:, :k]) * S[:k].unsqueeze(0)          # (dout, k)
    A = Vh[:k, :] @ Qa.t()                            # (k, din)
    return A.contiguous(), B.contiguous()


def _trim(d: torch.Tensor, density: float) -> torch.Tensor:
    """Giữ lại ``density`` phần tử biên độ lớn nhất (per-tensor), zero phần còn lại."""
    if density >= 1.0:
        return d
    flat = d.abs().reshape(-1)
    k = max(int(flat.numel() * density), 1)
    # topk (selection O(n)) nhanh hon kthvalue/sort tren GPU voi tensor lon
    thresh = torch.topk(flat, k, sorted=False).values.amin()
    return torch.where(d.abs() >= thresh, d, torch.zeros_like(d))


def _ties_core(deltas: List[torch.Tensor], weights: List[float], density: float) -> torch.Tensor:
    trimmed = [_trim(d, density) for d in deltas]
    stack = torch.stack([w * t for w, t in zip(weights, trimmed)], dim=0)  # (M, dout, din)
    gamma = torch.sign(stack.sum(dim=0))                                   # dấu bầu chọn
    agree = (torch.sign(stack) == gamma.unsqueeze(0)) & (stack != 0)
    kept = torch.where(agree, stack, torch.zeros_like(stack))
    count = agree.sum(dim=0).clamp_min(1)
    return kept.sum(dim=0) / count


def _merge_dare_ties(deltas: List[torch.Tensor], weights: List[float], density: float,
                     gen: torch.Generator) -> torch.Tensor:
    dropped = []
    for d in deltas:
        mask = (torch.rand(d.shape, generator=gen, device=d.device) < density).float()
        dropped.append(d * mask / max(density, 1e-6))
    return _ties_core(dropped, weights, density=1.0)


def _lowrank_svd(d: torch.Tensor, r: int):
    r = min(r, min(d.shape))
    q = min(r + 8, min(d.shape))
    U, S, V = torch.svd_lowrank(d, q=q, niter=4)
    return U[:, :r], S[:r], V[:, :r].t()


def _merge_tsv(deltas: List[torch.Tensor], per_task_rank: int, weights: List[float]) -> torch.Tensor:
    """TSV-M: trực giao hóa hướng kỳ dị GIỮA các task để khử interference.

    Mỗi task: ΔW_m ≈ U_m Σ_m V_mᵀ (rank r). Interference = phần chồng lấn giữa
    {U_m} và giữa {V_m}. Ta lấy cơ sở trực chuẩn Qu, Qv của hợp các không gian
    con, chiếu ΔW_bar = Σ w_m ΔW_m lên (Qu, Qv) rồi giữ phần "sạch":
        C = Quᵀ ΔW_bar Qv ,  ΔW_tsv = Qu C Qvᵀ
    Sau đó SVD-cắt về rank ở bước refactor. (Đã chuẩn hóa theo w_m — không bị
    3× magnitude như bản 'tổng thô'.)
    """
    Us, Vs = [], []
    dW_bar = torch.zeros_like(deltas[0])
    for w, d in zip(weights, deltas):
        u, s, vh = _lowrank_svd(d, per_task_rank)
        Us.append(u); Vs.append(vh.t())
        dW_bar += float(w) * d
    Qu, _ = torch.linalg.qr(torch.cat(Us, dim=1))   # (dout, M*r) trực chuẩn
    Qv, _ = torch.linalg.qr(torch.cat(Vs, dim=1))   # (din,  M*r) trực chuẩn
    C = Qu.t() @ dW_bar @ Qv                        # (M*r, M*r)
    return Qu @ C @ Qv.t()


def _merge_ctm(factors, deltas: List[torch.Tensor], out_rank: int, weights: List[float],
               beta: float = 0.5, hooi_iters: int = 4, base: str = "mean",
               density: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """CtM — Compress-then-Merge (arXiv 2606.03723).

    Ép rank-r TRƯỚC khi merge (thay vì merge full rồi cắt SVD = mất hướng).
      1. Core-space (Eq 9): UB, VA nửa-trực-giao span hợp {B_m}, {A_m}.  Tr = M·r.
      2. Core matrices M_m = UBᵀ ΔW_m VA  (Tr×Tr).
      3. Surrogate tái cân bằng (Eq 3): M_m^tgt = λ_m · M_m/‖M_m‖,
         λ_m = β‖M_m‖ + (1-β)·mean‖M‖  (khử magnitude dominance).
      4. Tucker-2 (HOSVD init + HOOI) trong Tr×Tr -> cơ sở chung Ucore, Vcore (Tr×r).
      5. Toạ độ gốc C_m = Ucoreᵀ M_m Vcore  (r×r); merge {C_m} bằng base rule
         (mean / ties) NGAY trong không gian r×r.
      6. Lift: ΔW = (UB Ucore) C_merged (VA Vcore)ᵀ  -> rank ≤ r *by construction*.
    """
    T = len(deltas)
    A_list = [a for a, _ in factors]
    B_list = [b for _, b in factors]
    Bcat = torch.cat(B_list, dim=1)          # (dout, T*r)
    Acat = torch.cat(A_list, dim=0)          # (T*r, din)
    UB, _ = torch.linalg.qr(Bcat)            # (dout, T*r)
    VA, _ = torch.linalg.qr(Acat.t())       # (din,  T*r)

    Ms = [UB.t() @ d @ VA for d in deltas]                       # (Tr, Tr)
    fro = [m.norm().clamp_min(1e-12) for m in Ms]
    avg = sum(fro) / T
    lam = [beta * f + (1.0 - beta) * avg for f in fro]
    Mt = [(l / f) * m for l, f, m in zip(lam, fro, Ms)]          # ‖Mt_m‖ = λ_m

    r = min(out_rank, Ms[0].shape[0])
    # HOSVD init
    Ucore = torch.linalg.svd(torch.cat(Mt, dim=1), full_matrices=False)[0][:, :r]
    Vcore = torch.linalg.svd(torch.cat([m.t() for m in Mt], dim=1), full_matrices=False)[0][:, :r]
    # HOOI refine
    for _ in range(hooi_iters):
        Yu = torch.cat([m @ Vcore for m in Mt], dim=1)
        Ucore = torch.linalg.svd(Yu, full_matrices=False)[0][:, :r]
        Yv = torch.cat([m.t() @ Ucore for m in Mt], dim=1)
        Vcore = torch.linalg.svd(Yv, full_matrices=False)[0][:, :r]

    Cs = [Ucore.t() @ m @ Vcore for m in Ms]                    # (r, r) — dùng M gốc, không surrogate
    if base == "ties":
        Cm = _ties_core(Cs, weights, density)
    else:
        Cm = torch.zeros_like(Cs[0])
        for w, c in zip(weights, Cs):
            Cm += float(w) * c

    U = UB @ Ucore                                              # (dout, r)
    V = VA @ Vcore                                              # (din,  r)
    Uc, Sc, Vhc = torch.linalg.svd(Cm, full_matrices=False)
    k = min(out_rank, Sc.numel())
    B = (U @ Uc[:, :k]) * Sc[:k].unsqueeze(0)                   # (dout, k)
    A = (Vhc[:k, :] @ V.t())                                    # (k, din)
    return A.contiguous(), B.contiguous()


def _merge_iso_c(deltas: List[torch.Tensor], keep_rank: int, weights: List[float]) -> torch.Tensor:
    """Iso-C: làm phẳng phổ kỳ dị của ΔW_bar (trung bình có trọng số) về σ̄."""
    dW_bar = torch.zeros_like(deltas[0])
    for w, d in zip(weights, deltas):
        dW_bar += float(w) * d
    k = min(keep_rank, min(dW_bar.shape))
    q = min(k + 8, min(dW_bar.shape))
    U, S, V = torch.svd_lowrank(dW_bar, q=q, niter=4)
    s_bar = S[:k].mean()
    return (U[:, :k] * s_bar) @ V[:, :k].t()


# --------------------------------------------------------------------------- #
def merge_experts(
    expert_dirs: List[str],
    strategy: str,
    *,
    out_rank: int = 16,
    density: float = 0.2,
    weights: List[float] | None = None,
    per_task_rank: int = 16,
    ctm_beta: float = 0.5,
    ctm_hooi: int = 4,
    ctm_base: str = "mean",
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Trả về (state_dict adapter đã merge, out_rank thực tế)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy phải thuộc {STRATEGIES}, nhận {strategy!r}")
    M = len(expert_dirs)
    weights = weights or [1.0 / M] * M
    gen = torch.Generator(device=device).manual_seed(seed)

    per_expert = [_load_expert_factors(d, device) for d in expert_dirs]
    modules = list(per_expert[0].keys())

    merged_sd: Dict[str, torch.Tensor] = {}
    for prefix in modules:
        factors = [pe[prefix] for pe in per_expert]           # [(A_m, B_m)]
        if strategy == "ta":
            A, B = _refactor_ta_fast([a for a, _ in factors], [b for _, b in factors],
                                     weights, out_rank)
        elif strategy == "ctm":
            deltas = [b @ a for a, b in factors]
            A, B = _merge_ctm(factors, deltas, out_rank, weights,
                              beta=ctm_beta, hooi_iters=ctm_hooi, base=ctm_base, density=density)
        else:
            deltas = [b @ a for a, b in factors]              # ΔW_m (dout, din)
            if strategy == "ties":
                dW = _ties_core(deltas, weights, density)
            elif strategy == "dare_ties":
                dW = _merge_dare_ties(deltas, weights, density, gen)
            elif strategy == "tsv":
                dW = _merge_tsv(deltas, per_task_rank, weights)
            elif strategy == "iso_c":
                dW = _merge_iso_c(deltas, out_rank, weights)
            A, B = _refactor_to_lora(dW, out_rank)
        merged_sd[prefix + _A_SUF] = A.to(torch.float32).cpu()
        merged_sd[prefix + _B_SUF] = B.to(torch.float32).cpu()
    return merged_sd, out_rank


def save_merged_adapter(merged_sd: Dict[str, torch.Tensor], rank: int,
                        template_dir: str, out_dir: str) -> str:
    merged_path = os.path.join(out_dir, "merged")
    os.makedirs(merged_path, exist_ok=True)
    cfg = json.load(open(os.path.join(template_dir, "adapter_config.json")))
    cfg["r"] = rank
    cfg["lora_alpha"] = rank            # scaling = 1
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}
    cfg["inference_mode"] = True
    json.dump(cfg, open(os.path.join(merged_path, "adapter_config.json"), "w"), indent=2)
    save_file(merged_sd, os.path.join(merged_path, "adapter_model.safetensors"))
    return merged_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Proposal §5.3 model merging")
    p.add_argument("--phase1_dir", default="outputs/proposal/phase1")
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--strategy", required=True, choices=STRATEGIES)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_rank", type=int, default=16)
    p.add_argument("--density", type=float, default=0.2, help="ties / dare_ties")
    p.add_argument("--per_task_rank", type=int, default=16, help="tsv")
    p.add_argument("--ctm_beta", type=float, default=0.5, help="ctm: β rescale (Eq 3); 0=đều, 1=giữ scale gốc")
    p.add_argument("--ctm_hooi", type=int, default=4, help="ctm: số vòng HOOI")
    p.add_argument("--ctm_base", default="mean", choices=["mean", "ties"], help="ctm: rule merge trong core r×r")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--weights", default=None,
                   help="M số cách nhau bởi dấu phẩy (vd '35.66,34.09,34.91'), tự chuẩn hoá "
                        "về tổng 1; mặc định None = đều 1/M.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from core.proctitle import set_proc_title
    set_proc_title(f"merge-{args.strategy}")
    names = [f"expert_{i}" for i in range(args.M)]
    expert_dirs = [os.path.join(args.phase1_dir, n) for n in names]
    missing = [d for d in expert_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"Thiếu expert dir: {missing}")

    weights = None
    if args.weights:
        raw = [float(x) for x in args.weights.split(",")]
        if len(raw) != args.M:
            raise ValueError(f"--weights cần {args.M} số, nhận {len(raw)}")
        tot = sum(raw)
        weights = [x / tot for x in raw]

    merged_sd, rank = merge_experts(
        expert_dirs, args.strategy, out_rank=args.out_rank, density=args.density,
        per_task_rank=args.per_task_rank, ctm_beta=args.ctm_beta, ctm_hooi=args.ctm_hooi,
        ctm_base=args.ctm_base, seed=args.seed, device=args.device, weights=weights,
    )
    path = save_merged_adapter(merged_sd, rank, expert_dirs[0], args.out_dir)
    if os.environ.get("REBUTTAL_STAGE_DIR"):
        from rebuttal.telemetry import STATE, flush, install
        install()
        estimate = 0.0
        for name, factor in merged_sd.items():
            if 'lora_A' in name:
                other = merged_sd[name.replace('lora_A', 'lora_B')]
                din, dout = factor.shape[1], other.shape[0]
                estimate += din * dout * (args.M * (2 * rank + 10) + 4 * (rank + 8) * 5)
        STATE['estimated_flops'] += estimate
        STATE['merge_flops_model'] = 'TIES elementwise + low-rank SVD dense approximation, niter=4'
        flush()
    info = {"strategy": args.strategy, "out_rank": rank, "density": args.density,
            "per_task_rank": args.per_task_rank, "ctm_beta": args.ctm_beta,
            "ctm_base": args.ctm_base, "source": args.phase1_dir, "weights": weights}
    json.dump(info, open(os.path.join(args.out_dir, "merge_info.json"), "w"), indent=2)
    print(f"[merge] {args.strategy} -> {path} (rank {rank}) | {info}", flush=True)


if __name__ == "__main__":
    main()
