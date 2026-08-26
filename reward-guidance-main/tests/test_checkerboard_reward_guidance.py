import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKERBOARD = ROOT / "checkerboard"
for path in (str(CHECKERBOARD), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model import reward_fn, sample_checkerboard  # noqa: E402
from sample import compute_second_order_guidance  # noqa: E402
from train import build_training_pool  # noqa: E402
from reward_guidance_math import second_order_preconditioned_gradient  # noqa: E402


class ZeroVelocity(torch.nn.Module):
    def forward(self, t, x):
        return torch.zeros_like(x)


class CheckerboardRewardGuidanceTests(unittest.TestCase):
    def test_endpoint_resampling_matches_positive_reward_direction(self):
        np.random.seed(0)
        torch.manual_seed(0)
        pool = sample_checkerboard(20_000)
        args = SimpleNamespace(
            target_distribution="reward_tilted",
            reward_center=[0.5, 0.5],
            sigma_r=1.5,
            beta=5.0,
            seed=0,
        )
        tilted, stats = build_training_pool(pool, args)
        center = torch.tensor(args.reward_center)
        base_reward = reward_fn(pool, center, args.sigma_r).mean()
        tilted_reward = reward_fn(tilted, center, args.sigma_r).mean()
        self.assertGreater(float(tilted_reward), float(base_reward) + 0.2)
        self.assertGreater(stats["effective_sample_size"], 100.0)

    def test_second_order_uses_deterministic_conditional_mean(self):
        velocity = ZeroVelocity()
        x = torch.tensor([[0.2, -0.3], [1.0, 0.7]])
        center = torch.tensor([0.5, 0.5])
        beta = 2.0
        variance = 0.4
        sigma_r = 1.5

        actual_1 = compute_second_order_guidance(
            velocity,
            t=0.4,
            x=x,
            lam=beta,
            reward_center=center,
            sigma_r=sigma_r,
            sigma_t_sq=variance,
        )
        actual_2 = compute_second_order_guidance(
            velocity,
            t=0.4,
            x=x,
            lam=beta,
            reward_center=center,
            sigma_r=sigma_r,
            sigma_t_sq=variance,
        )
        torch.testing.assert_close(actual_1, actual_2)

        reward = reward_fn(x, center, sigma_r)
        diff = x - center
        gradient = -diff / sigma_r**2 * reward.unsqueeze(-1)
        eye = torch.eye(2).expand(len(x), -1, -1)
        hessian = (
            -(
                eye * reward[:, None, None]
                + torch.bmm(diff.unsqueeze(-1), gradient.unsqueeze(1))
            )
            / sigma_r**2
        )
        expected, _ = second_order_preconditioned_gradient(
            gradient, hessian, variance, beta
        )
        torch.testing.assert_close(actual_1, expected)


if __name__ == "__main__":
    unittest.main()
