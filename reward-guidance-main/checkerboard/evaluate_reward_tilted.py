"""Validate a reward-tilted flow checkpoint against the exact 2D target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import VelocityMLP, checkerboard_density, reward_fn
from sample import sample_analytic_tilt
from train import generate_samples


def histogram_js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    """Jensen-Shannon divergence between two normalized 2D histograms."""

    hist_a, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=[[-3, 3], [-3, 3]])
    hist_b, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=[[-3, 3], [-3, 3]])
    eps = 1e-12
    p = (hist_a + eps) / (hist_a.sum() + eps * hist_a.size)
    q = (hist_b + eps) / (hist_b.sum() + eps * hist_b.size)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def sliced_wasserstein(
    a: np.ndarray, b: np.ndarray, num_projections: int = 128, seed: int = 0
) -> float:
    """Mean one-dimensional Wasserstein-2 distance over random projections."""

    if len(a) != len(b):
        raise ValueError("sliced_wasserstein expects equally sized samples.")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_projections, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected_a = np.sort(a @ directions.T, axis=0)
    projected_b = np.sort(b @ directions.T, axis=0)
    return float(np.sqrt(np.square(projected_a - projected_b).mean(axis=0)).mean())


def main(args) -> None:
    device = torch.device(args.device)
    checkpoint_path = Path(args.model_dir) / "velocity_net.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint["args"]

    beta = float(args.beta if args.beta is not None else saved.get("beta", 0.0))
    sigma_r = float(
        args.sigma_r if args.sigma_r is not None else saved.get("sigma_r", 1.5)
    )
    center_values = (
        args.reward_center
        if args.reward_center is not None
        else saved.get("reward_center", [0.5, 0.5])
    )
    center = torch.tensor(center_values, dtype=torch.float32)

    if saved.get("target_distribution") != "reward_tilted":
        raise ValueError(
            f"{checkpoint_path} is not a reward_tilted checkpoint; "
            "train with --target-distribution reward_tilted first."
        )

    model = VelocityMLP(
        hidden_dim=saved["hidden_dim"],
        num_layers=saved["num_layers"],
        rescale=saved["rescale"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    model.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    learned = generate_samples(
        model,
        args.num_samples,
        num_steps=args.num_ode_steps,
        device=device,
        rescale=saved["rescale"],
    ).numpy()
    exact = sample_analytic_tilt(
        args.num_samples,
        beta,
        center,
        sigma_r,
        seed=args.seed + 1,
    )

    learned_rewards = reward_fn(torch.from_numpy(learned), center, sigma_r).numpy()
    exact_rewards = reward_fn(torch.from_numpy(exact), center, sigma_r).numpy()
    metrics = {
        "beta": beta,
        "sigma_r": sigma_r,
        "reward_center": [float(x) for x in center_values],
        "num_samples": int(args.num_samples),
        "learned_reward_mean": float(learned_rewards.mean()),
        "exact_reward_mean": float(exact_rewards.mean()),
        "reward_mean_error": float(abs(learned_rewards.mean() - exact_rewards.mean())),
        "learned_valid_mass": float(checkerboard_density(learned).mean()),
        "exact_valid_mass": float(checkerboard_density(exact).mean()),
        "histogram_js_divergence": histogram_js_divergence(learned, exact),
        "sliced_wasserstein_2": sliced_wasserstein(learned, exact, seed=args.seed),
    }

    output_dir = Path(args.output_dir or args.model_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "reward_tilted_samples.npz",
        learned=learned,
        exact=exact,
        learned_rewards=learned_rewards,
        exact_rewards=exact_rewards,
    )
    with (output_dir / "reward_tilted_metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved validation to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a learned reward-tilted checkerboard flow to the exact target."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--sigma-r", type=float, default=None)
    parser.add_argument("--reward-center", type=float, nargs=2, default=None)
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--num-ode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parsed = parser.parse_args()
    if parsed.num_samples < 1 or parsed.num_ode_steps < 1:
        parser.error("--num-samples and --num-ode-steps must be positive.")
    if parsed.beta is not None and parsed.beta < 0.0:
        parser.error("--beta must be non-negative.")
    main(parsed)
