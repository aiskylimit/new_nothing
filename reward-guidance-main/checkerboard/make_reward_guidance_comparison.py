"""Render direct-training and inference-guidance checkerboard distributions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DEFAULT_REWARD_CENTER, checkerboard_density, reward_fn

REPO = Path(__file__).resolve().parents[1]
plt.style.use(str(REPO / "assets" / "paper.mplstyle"))


def load_samples(path: Path, key: str = "samples") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path)[key])


def draw(ax, samples: np.ndarray, label: str, sigma_r: float, display_n: int) -> None:
    grid = np.linspace(-3, 3, 240)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([xx, yy], axis=-1)
    density = checkerboard_density(points)
    ax.contourf(xx, yy, density, levels=[0.5, 1.5], colors=["#d7dfe3"], alpha=0.9)

    shown = samples[:display_n]
    rewards = reward_fn(
        torch.from_numpy(samples).float(), DEFAULT_REWARD_CENTER, sigma_r
    ).numpy()
    valid_mass = checkerboard_density(samples).mean()
    ax.scatter(
        shown[:, 0],
        shown[:, 1],
        s=5,
        c="#f6c945",
        edgecolors="none",
        alpha=0.72,
    )
    center = DEFAULT_REWARD_CENTER.numpy()
    ax.scatter(*center, marker="X", s=150, c="#e60000", edgecolors="black")
    ax.set_title(
        f"{label}\nreward={rewards.mean():.3f}, valid={valid_mass:.3f}", fontsize=11
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])


def main(args) -> None:
    results = Path(args.results_dir)
    beta_text = f"{args.beta:g}"
    beta_filename = str(float(args.beta))
    tilted_evaluation = Path(args.tilted_model_dir) / "evaluation"
    panels = [
        (
            "Exact tilt",
            load_samples(results / f"analytic_tilt_lam{beta_filename}.npz"),
        ),
        (
            "Direct tilted flow",
            load_samples(tilted_evaluation / "reward_tilted_samples.npz", "learned"),
        ),
        (
            "First-order",
            load_samples(results / f"guided_first_order_lam{beta_filename}.npz"),
        ),
        (
            "Second-order",
            load_samples(results / f"guided_second_order_lam{beta_filename}.npz"),
        ),
        (
            "Plugin k=1",
            load_samples(results / f"guided_k1_lam{beta_filename}.npz"),
        ),
    ]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(17, 3.75), sharex=True, sharey=True
    )
    for ax, (label, samples) in zip(axes, panels):
        draw(ax, samples, label, args.sigma_r, args.display_samples)
    fig.suptitle(
        rf"Reward-guided checkerboard comparison ($\beta={beta_text}$)", y=1.01
    )
    fig.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_dir / f"reward_guidance_comparison.{extension}"
        fig.savefig(output, dpi=200 if extension == "png" else None)
        print(f"saved -> {output}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--sigma-r", type=float, default=1.5)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tilted-model-dir", default="results/reward_tilted_beta10")
    parser.add_argument("--output-dir", default="../figures/checkerboard")
    parser.add_argument("--display-samples", type=int, default=5_000)
    parsed = parser.parse_args()
    if parsed.beta < 0.0 or parsed.sigma_r <= 0.0 or parsed.display_samples < 1:
        parser.error("beta must be non-negative; sigma/display count must be positive.")
    main(parsed)
