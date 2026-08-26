"""Evaluate generated comparisons with a held-out CLIP model and paired CIs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_CONDITIONS = (
    "unguided",
    "gns50",
    "2nd_order_gns50",
    "gns50_k8",
    "2nd_order_unnorm",
)


def completed_run(condition_dir: Path) -> Path:
    """Select the most recently completed run, never a partial image folder."""

    runs = [path.parent for path in condition_dir.rglob("rewards.npy")]
    if not runs:
        raise FileNotFoundError(f"No completed run under {condition_dir}")
    return max(runs, key=lambda path: (path / "rewards.npy").stat().st_mtime_ns)


def metadata_literal(metadata_path: Path, field: str) -> str:
    prefix = f"{field}:"
    for line in metadata_path.read_text().splitlines():
        if line.startswith(prefix):
            value = line.split(":", 1)[1].strip()
            if "  (" in value:
                value = value.split("  (", 1)[0]
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, str):
                raise ValueError(f"{field} is not text in {metadata_path}")
            return parsed
    raise KeyError(f"Missing {field!r} in {metadata_path}")


def confidence_interval(values: np.ndarray) -> list[float]:
    """Normal 95% confidence interval for a per-seed metric mean."""

    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    mean = float(values.mean())
    return [mean - half_width, mean + half_width]


def paired_bootstrap(
    lhs: np.ndarray, rhs: np.ndarray, *, draws: int, seed: int
) -> dict[str, float | list[float]]:
    """Bootstrap ``rhs-lhs`` using paired seed indices."""

    count = min(len(lhs), len(rhs))
    if count < 2:
        raise ValueError("Paired bootstrap requires at least two shared seeds.")
    differences = rhs[:count] - lhs[:count]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(draws, count))
    means = differences[indices].mean(axis=1)
    return {
        "n": count,
        "mean_difference": float(differences.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def encode_images(
    paths: list[Path], model, preprocess, device: torch.device, batch_size: int
) -> torch.Tensor:
    features = []
    for start in range(0, len(paths), batch_size):
        images = torch.stack(
            [
                preprocess(Image.open(path).convert("RGB"))
                for path in paths[start : start + batch_size]
            ]
        ).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(images).float()
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu())
    return torch.cat(features)


def diversity(features: torch.Tensor) -> float:
    if len(features) < 2:
        return 0.0
    similarities = features @ features.T
    upper = torch.triu_indices(len(features), len(features), offset=1)
    return float((1.0 - similarities[upper[0], upper[1]]).mean().item())


def evaluate_experiment(
    experiment_dir: Path,
    conditions: tuple[str, ...],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    bootstrap_draws: int,
    seed: int,
    clip_module,
) -> dict:
    runs = {
        condition: completed_run(experiment_dir / condition) for condition in conditions
    }
    generation_prompt = metadata_literal(runs["unguided"] / "metadata.txt", "prompt")
    reward_prompt = metadata_literal(
        runs["gns50"] / "metadata.txt", "ir_prompt (scoring)"
    )
    with torch.no_grad():
        tokens = clip_module.tokenize([generation_prompt]).to(device)
        text_feature = model.encode_text(tokens).float()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature.cpu()

    metrics = {}
    clip_scores = {}
    reward_scores = {}
    for condition, run in runs.items():
        paths = sorted(run.glob("[0-9]*.png"))
        rewards = np.asarray(np.load(run / "rewards.npy"), dtype=np.float64)
        if len(paths) != len(rewards):
            raise ValueError(
                f"Image/reward count mismatch in {run}: {len(paths)} vs {len(rewards)}"
            )
        features = encode_images(paths, model, preprocess, device, batch_size)
        similarities = (100.0 * features @ text_feature.T).squeeze(1).numpy()
        clip_scores[condition] = similarities
        reward_scores[condition] = rewards
        metrics[condition] = {
            "run": str(run),
            "n": len(paths),
            "guidance_reward_mean": float(rewards.mean()),
            "guidance_reward_ci95": confidence_interval(rewards),
            "heldout_clip_generation_mean": float(similarities.mean()),
            "heldout_clip_generation_ci95": confidence_interval(similarities),
            "clip_embedding_diversity": diversity(features),
        }

    comparisons = {
        "second_order_gns50_minus_plugin_gns50": {
            "guidance_reward": paired_bootstrap(
                reward_scores["gns50"],
                reward_scores["2nd_order_gns50"],
                draws=bootstrap_draws,
                seed=seed,
            ),
            "heldout_clip_generation": paired_bootstrap(
                clip_scores["gns50"],
                clip_scores["2nd_order_gns50"],
                draws=bootstrap_draws,
                seed=seed + 1,
            ),
        }
    }
    return {
        "generation_prompt": generation_prompt,
        "reward_prompt": reward_prompt,
        "conditions": metrics,
        "paired_comparisons": comparisons,
    }


def main(args) -> None:
    import clip

    device = torch.device(args.device)
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval().requires_grad_(False)
    data_root = Path(args.data_root)
    conditions = tuple(args.conditions)
    report = {
        "heldout_model": f"OpenAI CLIP {args.clip_model}",
        "bootstrap_draws": args.bootstrap_draws,
        "experiments": {},
    }
    for name in args.experiments:
        report["experiments"][name] = evaluate_experiment(
            data_root / name,
            conditions,
            model,
            preprocess,
            device,
            args.batch_size,
            args.bootstrap_draws,
            args.seed,
            clip,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved held-out evaluation to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "imagereward_archaeologist",
            "imagereward_miner",
            "imagereward_market",
        ],
    )
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="../data/imagereward_evaluation.json")
    parsed = parser.parse_args()
    if parsed.batch_size < 1 or parsed.bootstrap_draws < 100:
        parser.error("--batch-size must be positive and --bootstrap-draws >= 100.")
    main(parsed)
