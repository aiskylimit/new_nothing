#!/usr/bin/env python3
"""How a fitted steering map moves the three activation groups at one layer: the stored last-prompt-token activations
(benign / refused / jailbreak, the fit sets of the extract stage) before and after the map (h + alpha g(h), exactly the
steered input of that layer), drawn on fixed linear axes so before and after are comparable:

  shift_<layer>_plane.png      x = refusal direction (mean jailbreak -> mean refused), y = harmful direction (mean
                               benign -> mean harmful, made orthogonal to x); before | after per alpha, arrows on a
                               sample of jailbreaks
  shift_<layer>_refusal_axis.png  the three distributions along the refusal direction, before vs after
  shift_<layer>_transport.png  the first two axes of the map's transport subspace (U, or P for a gated map), if any
  shift_<layer>_gate.png       gated maps (SELOT-gated) only: the gate value G(h) of the three groups
  shift_<layer>_clusters.png   gated maps with the Gaussian-mixture gate only: every activation coloured by its mixture
                               component (the one of its own set with the highest posterior; jailbreak components =
                               the transport modes), in the plane of shift_<layer>_plane.png (top row) and on the first
                               two gate features (whitened top PCs of the three sets, bottom row)

    python scripts/plot_shift.py --acts <work>/acts/<tag>/wf1/llamaguard --map <work>/maps/<run>/map.pt --layer 18 \
        --alphas 1 --out results/results_<run>/plots

Linear axes on purpose (no t-SNE / UMAP): they keep the affine move and the same axes before and after.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selot.affine_map import AffineMap  # noqa: E402

COLORS = {"benign": "#8c8c8c", "refused": "#1f77b4", "jailbreak": "#d62728"}
GROUPS = ("benign", "refused", "jailbreak")


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def load(acts_dir, layer, max_benign, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name in GROUPS:
        f = Path(acts_dir) / f"{name}.pt"
        if not f.exists():
            continue
        h = torch.load(f, map_location="cpu", weights_only=False)["acts"][layer].float()
        if name == "benign" and h.shape[0] > max_benign:
            h = h[torch.randperm(h.shape[0], generator=g)[:max_benign]]
        out[name] = h
    return out


def mixture_components(lm, x):
    """gate features z (n x m) and, per gate component, log pi_k N_k(z) (n x K) of a gated map with a GMM gate"""
    e = lm.extras
    z = (x.float() - e["gate_mu"]) @ e["gate_Q"]
    maha = torch.einsum("nkm,kmj->nkj", z.unsqueeze(1) - e["gate_means"], e["gate_chol"]).pow(2).sum(-1)
    return z, e["gate_logc"] - 0.5 * maha


CLASS_ID = {"jailbreak": 0, "benign": 1, "refused": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", default="1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-benign", type=int, default=3000)
    ap.add_argument("--arrows", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    lm = AffineMap.load(a.map).layers[a.layer]
    h = load(a.acts, a.layer, a.max_benign, a.seed)
    after = {al: {n: lm.apply(x, al) for n, x in h.items()} for al in alphas}
    mu = {n: x.mean(0) for n, x in h.items()}
    harm = torch.cat([h["refused"], h["jailbreak"]]).mean(0)
    ex = unit(mu["refused"] - mu["jailbreak"])
    ey = harm - mu["benign"]
    ey = unit(ey - (ey @ ex) * ex)
    center = torch.cat(list(h.values())).mean(0)
    proj = lambda x: torch.stack([(x - center) @ ex, (x - center) @ ey], 1).numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (a.title + " " if a.title else "") + f"layer {a.layer}"
    files = {}

    # 1. plane: before | after (per alpha)
    n = 1 + len(alphas)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4.8), sharex=True, sharey=True)
    pb = {k: proj(x) for k, x in h.items()}
    for ax, (label, pts) in zip(axs, [("before steering", pb)] + [(f"after (alpha = {al:g})",
                                                                     {k: proj(x) for k, x in after[al].items()})
                                                                    for al in alphas]):
        for k in GROUPS:
            if k in pts:
                ax.scatter(pts[k][:, 0], pts[k][:, 1], s=4, alpha=0.25 if k == "benign" else 0.4, c=COLORS[k],
                           label=f"{k} ({len(pts[k])})", rasterized=True)
                ax.scatter(*pts[k].mean(0), s=140, c=COLORS[k], marker="X", edgecolors="black", linewidths=1)
        ax.set_title(label)
        ax.set_xlabel("refusal direction (jailbreak -> refused)")
    axs[0].set_ylabel("harmful direction (benign -> harmful)")
    if alphas:
        pa = proj(after[alphas[-1]]["jailbreak"])
        idx = torch.randperm(len(pa), generator=torch.Generator().manual_seed(a.seed))[: a.arrows].numpy()
        for i in idx:
            axs[-1].annotate("", xy=pa[i], xytext=pb["jailbreak"][i],
                             arrowprops=dict(arrowstyle="->", color=COLORS["jailbreak"], alpha=0.35, lw=0.7))
    axs[0].legend(loc="best", fontsize=8, markerscale=3)
    fig.suptitle(f"{tag}: activations before and after steering (X = group mean)")
    fig.tight_layout()
    files["plane"] = out / f"shift_L{a.layer}_plane.png"
    fig.savefig(files["plane"], dpi=130)
    plt.close(fig)

    # 2. distributions along the refusal direction
    fig, axs = plt.subplots(1, len(GROUPS), figsize=(15, 3.8), sharex=True)
    xr = lambda x: ((x - center) @ ex).numpy()
    for ax, k in zip(axs, GROUPS):
        if k not in h:
            continue
        ax.hist(xr(h[k]), bins=60, density=True, histtype="step", lw=1.6, color="black", label="before")
        for al in alphas:
            ax.hist(xr(after[al][k]), bins=60, density=True, histtype="stepfilled", alpha=0.35, color=COLORS[k],
                    label=f"after (alpha = {al:g})")
        for ref in ("refused", "jailbreak"):
            ax.axvline(float((mu[ref] - center) @ ex), color=COLORS[ref], ls="--", lw=1)
        ax.set_title(f"{k}: before vs after")
        ax.set_xlabel("refusal direction (dashed: refused / jailbreak means before)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{tag}: distributions along the refusal direction (jailbreak should move right, benign stay)")
    fig.tight_layout()
    files["refusal_axis"] = out / f"shift_L{a.layer}_refusal_axis.png"
    fig.savefig(files["refusal_axis"], dpi=130)
    plt.close(fig)

    # 3. the map's transport subspace (first two columns of U; P for a gated map)
    basis = lm.U if lm.U is not None else lm.extras.get("P")
    if basis is not None and basis.shape[1] >= 2:
        u = torch.linalg.qr(basis.float())[0][:, :2]
        pt = lambda x: ((x - center) @ u).numpy()
        al = alphas[-1] if alphas else 1.0
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
        for ax, (label, pts) in zip(axs, [("before", h), (f"after (alpha = {al:g})", after.get(al, h))]):
            for k in GROUPS:
                if k in pts:
                    q = pt(pts[k])
                    ax.scatter(q[:, 0], q[:, 1], s=4, alpha=0.25 if k == "benign" else 0.4, c=COLORS[k], label=k,
                               rasterized=True)
                    ax.scatter(*q.mean(0), s=140, c=COLORS[k], marker="X", edgecolors="black", linewidths=1)
            ax.set_title(label)
            ax.set_xlabel("transport axis 1")
        axs[0].set_ylabel("transport axis 2")
        axs[0].legend(fontsize=8, markerscale=3)
        fig.suptitle(f"{tag}: the transport subspace (where the OT map acts)")
        fig.tight_layout()
        files["transport"] = out / f"shift_L{a.layer}_transport.png"
        fig.savefig(files["transport"], dpi=130)
        plt.close(fig)

    # 4. gated map: where the gate opens
    gate = {}
    if lm.gated:
        gate = {k: lm.gate(x)[1].numpy() for k, x in h.items()}
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for k in GROUPS:
            if k in gate:
                ax.hist(gate[k], bins=50, range=(0, 1), density=True, histtype="stepfilled", alpha=0.4,
                        color=COLORS[k], label=f"{k}: open {100 * (gate[k] > 0.5).mean():.1f}%")
        ax.set_yscale("log")
        ax.set_xlabel("gate G(h) (1 = full OT step)")
        ax.legend(fontsize=8)
        fig.suptitle(f"{tag}: gate value per group")
        fig.tight_layout()
        files["gate"] = out / f"shift_L{a.layer}_gate.png"
        fig.savefig(files["gate"], dpi=130)
        plt.close(fig)

    # 5. GMM gate: the mixture components of each set (jailbreak components = the transport modes)
    clusters = {}
    if lm.gated and "gate_class" in lm.extras:
        cls = lm.extras["gate_class"].long()
        zs, assign = {}, {}
        for k, x in h.items():
            z, lp = mixture_components(lm, x)
            own = (cls == CLASS_ID[k]).nonzero().flatten()
            zs[k] = z[:, :2].numpy()
            assign[k] = lp[:, own].argmax(1).numpy()
            clusters[k] = [float((assign[k] == c).mean()) for c in range(len(own))]
        palette = plt.get_cmap("tab10").colors
        fig, axs = plt.subplots(2, len(GROUPS), figsize=(5.2 * len(GROUPS), 9.2), sharex="row", sharey="row")
        for row, (pts, xl, yl) in enumerate([(pb, "refusal direction (jailbreak -> refused)",
                                              "harmful direction (benign -> harmful)"),
                                             (zs, "gate feature 1", "gate feature 2")]):
            for col, k in enumerate(GROUPS):
                ax = axs[row, col]
                if k not in h:
                    continue
                for other in GROUPS:                                     # the other sets, faint, for orientation
                    if other != k and other in pts:
                        ax.scatter(pts[other][:, 0], pts[other][:, 1], s=3, c="#d0d0d0", alpha=0.25, rasterized=True)
                for c in range(len(clusters[k])):
                    sel = assign[k] == c
                    if sel.any():
                        ax.scatter(pts[k][sel, 0], pts[k][sel, 1], s=5, alpha=0.5, color=palette[c % 10],
                                   label=f"{k[0]}{c + 1} ({100 * clusters[k][c]:.0f}%)", rasterized=True)
                        ax.scatter(*pts[k][sel].mean(0), s=150, color=palette[c % 10], marker="X",
                                   edgecolors="black", linewidths=1)
                ax.set_title(f"{k}: {len(clusters[k])} component{'s' if len(clusters[k]) > 1 else ''}"
                             + (" (one OT map each)" if k == "jailbreak" and len(clusters[k]) > 1 else ""))
                ax.set_xlabel(xl)
                ax.legend(fontsize=8, markerscale=3, loc="best")
            axs[row, 0].set_ylabel(yl)
        fig.suptitle(f"{tag}: Gaussian-mixture components of each set (before steering; grey = the other sets)")
        fig.tight_layout()
        files["clusters"] = out / f"shift_L{a.layer}_clusters.png"
        fig.savefig(files["clusters"], dpi=130)
        plt.close(fig)

    stats = {k: {"before_mean_x": float(((h[k] - center) @ ex).mean()),
                 **{f"after_{al:g}_mean_x": float(((after[al][k] - center) @ ex).mean()) for al in alphas},
                 **({"gate_open": float((gate[k] > 0.5).mean()), "gate_mean": float(gate[k].mean())} if gate else {}),
                 **({"component_shares": clusters[k]} if k in clusters else {})}
             for k in h}
    (out / f"shift_L{a.layer}.json").write_text(json.dumps(stats, indent=1))
    print("plots:", ", ".join(str(f) for f in files.values()))


if __name__ == "__main__":
    main()
