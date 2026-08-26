"""Train a flow-matching model on the base or reward-tilted checkerboard."""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import numpy as np
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model import VelocityMLP, checkerboard_density, reward_fn, sample_checkerboard
from reward_guidance_math import effective_sample_size, exponential_tilt_probabilities

plt.style.use(str(REPO / "assets" / "default.mplstyle"))


def _checkpoint_arg(saved_args, name, default=None):
    """Read an argument from old dict checkpoints and Namespace checkpoints."""

    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def build_training_pool(data_pool, args):
    """Return endpoint samples for the requested empirical target distribution.

    For ``reward_tilted`` the endpoints are resampled with probabilities
    proportional to ``exp(+beta * reward)``.  This makes the terminal empirical
    marginal exactly the desired positive-reward tilt, up to the finite pool.
    """

    if args.target_distribution == "base":
        return data_pool, {
            "target_distribution": "base",
            "effective_sample_size": float(data_pool.shape[0]),
        }
    if args.target_distribution != "reward_tilted":
        raise ValueError(f"Unknown target distribution: {args.target_distribution}")

    center = torch.as_tensor(
        args.reward_center, device=data_pool.device, dtype=data_pool.dtype
    )
    with torch.no_grad():
        rewards = reward_fn(data_pool, center, args.sigma_r)
        probabilities = exponential_tilt_probabilities(rewards, args.beta)
        ess = effective_sample_size(probabilities)
        generator = torch.Generator(device=data_pool.device).manual_seed(args.seed + 17)
        indices = torch.multinomial(
            probabilities,
            num_samples=data_pool.shape[0],
            replacement=True,
            generator=generator,
        )
        tilted_pool = data_pool[indices]

    stats = {
        "target_distribution": "reward_tilted",
        "beta": float(args.beta),
        "base_reward_mean": float(rewards.mean().item()),
        "tilted_reward_mean": float(rewards[indices].mean().item()),
        "effective_sample_size": ess,
        "pool_size": int(data_pool.shape[0]),
    }
    print(
        "Reward tilt: "
        f"beta={args.beta:g}, reward {stats['base_reward_mean']:.4f} -> "
        f"{stats['tilted_reward_mean']:.4f}, ESS={ess:.0f}/{len(data_pool)}",
        flush=True,
    )
    if ess < 0.01 * len(data_pool):
        print(
            "WARNING: tilt ESS is below 1% of the endpoint pool; increase "
            "--pool-size or reduce --beta.",
            flush=True,
        )
    return tilted_pool, stats


def generate_samples(
    model, num_samples, num_steps=200, device="cpu", snapshot_times=None, rescale=1.0
):
    """Generate samples by integrating the learned flow ODE with Heun's method.

    Args:
        snapshot_times: optional list of times in [0, 1] at which to save snapshots.
            If provided, returns (final_samples, dict of {t: samples_at_t}).
        rescale: std of the data; initial noise is rescale * N(0, I).
    """
    model.eval()
    x = rescale * torch.randn(num_samples, 2, device=device)
    dt = 1.0 / num_steps

    snapshots = {}
    snapshot_steps = set()
    if snapshot_times is not None:
        for st in snapshot_times:
            step_idx = int(round(st / dt))
            snapshot_steps.add(step_idx)

    def vel(t_val, x_val):
        t_tensor = torch.full((num_samples,), t_val, device=device)
        return model(t_tensor, x_val)

    with torch.no_grad():
        for i in range(num_steps):
            if i in snapshot_steps:
                snapshots[i * dt] = x.detach().cpu()
            t_i = i * dt
            # Heun (2nd-order predictor-corrector), matching nmboffi/jax-interpolants
            v0 = vel(t_i, x)
            x_pred = x + dt * v0
            v1 = vel(t_i + dt, x_pred)
            x = x + 0.5 * dt * (v0 + v1)

    model.train()
    final = x.detach().cpu()
    if snapshot_times is not None:
        snapshots[1.0] = final
        return final, snapshots
    return final


def plot_loss_curve(losses, image_dir):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    fig.savefig(os.path.join(image_dir, "loss_curve.pdf"))
    plt.close(fig)


def plot_samples(
    samples, image_dir, filename="train_samples.pdf", xlim=(-3, 3), ylim=(-3, 3)
):
    fig, ax = plt.subplots(figsize=(5, 5))
    s = samples.numpy()
    # Checkerboard background
    res = 200
    gx = np.linspace(-3, 3, res)
    gy = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X, Y], axis=-1)
    density = checkerboard_density(pts)
    ax.contourf(X, Y, density, levels=[0.5, 1.5], colors=["gray"], alpha=0.15)
    # Scatter
    ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.5, c="C0", edgecolors="none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(image_dir, filename))
    plt.close(fig)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(os.path.dirname(args.output_dir), "images", "training")
    os.makedirs(image_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "velocity_net.pt")
    losses = []
    start_step = 1

    resume_checkpoint = None
    if args.resume and os.path.exists(checkpoint_path):
        resume_checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        saved_args = resume_checkpoint["args"]
        # Resuming must retain the target that produced the existing optimizer
        # state; command-line defaults must not silently change it.
        for name in (
            "target_distribution",
            "beta",
            "sigma_r",
            "reward_center",
            "pool_size",
            "num_steps",
        ):
            saved = _checkpoint_arg(saved_args, name, None)
            if saved is not None:
                setattr(args, name, saved)

    init_checkpoint = None
    if args.init_checkpoint and resume_checkpoint is None:
        init_checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )

    # Pre-generate the base pool, then construct the requested endpoint target.
    pool_size = max(args.batch_size * 100, args.pool_size)
    print(f"Pre-generating {pool_size} checkerboard samples...", flush=True)
    base_pool = sample_checkerboard(pool_size, device=device)
    data_pool, tilt_stats = build_training_pool(base_pool, args)

    # Adaptive Gaussian rescaling: match noise std to data std (nmboffi/jax-interpolants)
    # The prior remains calibrated to the base data, not to a concentrated tilt.
    rescale = float(base_pool.std().item())
    print(f"Data std (rescale): {rescale:.4f}", flush=True)

    architecture_checkpoint = resume_checkpoint or init_checkpoint
    if architecture_checkpoint is not None:
        saved_args = architecture_checkpoint["args"]
        hidden_dim = int(_checkpoint_arg(saved_args, "hidden_dim", args.hidden_dim))
        num_layers = int(_checkpoint_arg(saved_args, "num_layers", args.num_layers))
        rescale = float(_checkpoint_arg(saved_args, "rescale", rescale))
    else:
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers

    model = VelocityMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rescale=rescale,
    ).to(device)
    ema_model = copy.deepcopy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=0
    )

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        ema_model.load_state_dict(
            resume_checkpoint.get("ema_model", resume_checkpoint["model"])
        )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_step = resume_checkpoint["step"] + 1
        losses = resume_checkpoint.get("losses", [])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        else:
            # Legacy checkpoints did not save the scheduler. Set its closed-form
            # state directly instead of calling scheduler.step() before optimizer.
            scheduler.last_epoch = resume_checkpoint["step"]
            scheduler._step_count = resume_checkpoint["step"] + 1
            lrs = scheduler._get_closed_form_lr()
            for group, lr in zip(optimizer.param_groups, lrs):
                group["lr"] = lr
        print(f"Resumed from step {start_step - 1}", flush=True)
    elif init_checkpoint is not None:
        initial_state = init_checkpoint.get("ema_model", init_checkpoint["model"])
        model.load_state_dict(initial_state)
        ema_model.load_state_dict(initial_state)
        print(f"Initialized from EMA weights in {args.init_checkpoint}", flush=True)

    pbar = trange(start_step, args.num_steps + 1, desc="Training")
    for step in pbar:
        # Draw a random batch from the pool
        idx = torch.randint(0, pool_size, (args.batch_size,), device=device)
        x1 = data_pool[idx]
        # Adaptive Gaussian base: N(0, rescale^2 * I)
        x0 = rescale * torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)

        # Linear interpolant: I_t = (1 - t) x0 + t x1
        xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1

        # Target velocity: dI_t/dt = x1 - x0
        target = x1 - x0

        # Flow matching loss
        pred = model(t, xt)
        loss = torch.mean((pred - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        # EMA update (decay 0.9999, following nmboffi/jax-interpolants)
        update_ema(ema_model, model, decay=0.9999)

        losses.append(loss.item())
        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Checkpoint every save_every steps
        if step % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "losses": losses,
                    "args": {
                        **vars(args),
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "rescale": rescale,
                    },
                    "tilt_stats": tilt_stats,
                },
                checkpoint_path,
            )

    # Final save
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": args.num_steps,
            "losses": losses,
            "args": {
                **vars(args),
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "rescale": rescale,
            },
            "tilt_stats": tilt_stats,
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}", flush=True)

    # Save loss curve
    np.save(os.path.join(args.output_dir, "losses.npy"), np.array(losses))
    plot_loss_curve(losses, image_dir)
    print(f"Loss curve saved to {image_dir}/loss_curve.pdf", flush=True)

    if args.final_eval_samples == 0:
        return

    # Generate and plot samples from EMA model.
    print("Generating samples with snapshots...", flush=True)
    snapshot_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    samples, snapshots = generate_samples(
        ema_model,
        args.final_eval_samples,
        num_steps=args.final_eval_steps,
        device=device,
        snapshot_times=snapshot_times,
        rescale=rescale,
    )
    plot_samples(samples, image_dir)
    print(f"Sample plot saved to {image_dir}/train_samples.pdf", flush=True)

    # Save snapshot plots
    snap_dir = os.path.join(image_dir, "flow_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    for t_snap, x_snap in sorted(snapshots.items()):
        plot_samples(x_snap, snap_dir, filename=f"t_{t_snap:.2f}.pdf")
    print(f"Snapshot plots saved to {snap_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching on checkerboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--save-every", type=int, default=2000, help="Checkpoint every N steps"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument("--pool-size", type=int, default=500_000)
    parser.add_argument("--final-eval-samples", type=int, default=5_000)
    parser.add_argument("--final-eval-steps", type=int, default=200)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model and EMA from another checkpoint's EMA weights.",
    )
    parser.add_argument(
        "--target-distribution",
        choices=["base", "reward_tilted"],
        default="base",
    )
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--sigma-r", type=float, default=1.5)
    parser.add_argument("--reward-center", type=float, nargs=2, default=[0.5, 0.5])
    args = parser.parse_args()
    if args.beta < 0.0:
        parser.error("--beta must be non-negative.")
    if args.sigma_r <= 0.0:
        parser.error("--sigma-r must be positive.")
    if args.pool_size < 1 or args.log_every < 1 or args.save_every < 1:
        parser.error("--pool-size, --log-every and --save-every must be positive.")
    if args.final_eval_samples < 0 or args.final_eval_steps < 1:
        parser.error("--final-eval-samples must be non-negative and steps positive.")
    train(args)
