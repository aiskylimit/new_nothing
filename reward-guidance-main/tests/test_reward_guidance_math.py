import unittest

import torch

from reward_guidance_math import (
    effective_sample_size,
    exponential_tilt_probabilities,
    first_order_gaussian,
    second_order_gaussian,
    second_order_preconditioned_gradient,
    stabilize_low_rank_curvatures,
)


class RewardGuidanceMathTests(unittest.TestCase):
    def test_exponential_tilt_prefers_high_reward(self):
        rewards = torch.tensor([-2.0, 0.0, 3.0])
        probabilities = exponential_tilt_probabilities(rewards, beta=4.0)
        self.assertEqual(int(probabilities.argmax()), 2)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertGreater(effective_sample_size(probabilities), 0.99)

    def test_first_order_positive_sign_moves_up_reward(self):
        mean = torch.tensor([[0.0, 0.0]])
        gradient = torch.tensor([[2.0, -1.0]])
        corrected, covariance = first_order_gaussian(mean, 0.5, 3.0, gradient)
        torch.testing.assert_close(corrected, torch.tensor([[3.0, -1.5]]))
        torch.testing.assert_close(covariance, 0.5 * torch.eye(2).unsqueeze(0))

    def test_second_order_matches_exact_quadratic_tilt(self):
        mean = torch.tensor([[0.4, -0.3]], dtype=torch.float64)
        variance = 0.7
        beta = 1.8
        curvature = torch.tensor([[1.5, 0.2], [0.2, 0.8]], dtype=torch.float64)
        target = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        gradient = -((mean - target) @ curvature)
        hessian = -curvature.unsqueeze(0)

        actual_mean, actual_cov, diagnostics = second_order_gaussian(
            mean, variance, beta, gradient, hessian
        )
        precision = torch.eye(2, dtype=torch.float64) / variance + beta * curvature
        expected_cov = torch.linalg.inv(precision).unsqueeze(0)
        expected_mean = mean + beta * torch.bmm(
            expected_cov, gradient.unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(actual_cov, expected_cov)
        torch.testing.assert_close(actual_mean, expected_mean)
        self.assertEqual(diagnostics.num_clipped_curvatures, 0)

    def test_positive_curvature_is_kept_until_precision_floor(self):
        gradient = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        hessian = torch.diag(torch.tensor([5.0, -2.0], dtype=torch.float64)).unsqueeze(
            0
        )
        corrected, diagnostics = second_order_preconditioned_gradient(
            gradient, hessian, variance=1.0, beta=1.0, precision_floor=0.1
        )
        self.assertEqual(diagnostics.num_clipped_curvatures, 1)
        self.assertGreaterEqual(diagnostics.min_eigenvalue_after, 0.1 - 1e-10)
        self.assertGreater(float(corrected[0, 0]), 1.0)

    def test_low_rank_stabilizer_keeps_negative_and_damps_positive(self):
        w = torch.tensor([[1.0, 0.3], [0.2, 1.2], [0.5, -0.4]], dtype=torch.float64)
        gram = w.T @ w
        curvatures = torch.tensor([4.0, -2.0], dtype=torch.float64)
        stabilized, diagnostics = stabilize_low_rank_curvatures(
            gram, curvatures, scale=1.0, precision_floor=0.1
        )
        dense_precision = (
            torch.eye(3, dtype=torch.float64) - w @ torch.diag(stabilized) @ w.T
        )
        self.assertAlmostEqual(float(stabilized[1]), -2.0, places=12)
        self.assertLess(float(stabilized[0]), 4.0)
        self.assertGreaterEqual(
            float(torch.linalg.eigvalsh(dense_precision).min()), 0.1 - 1e-8
        )
        self.assertLess(diagnostics.positive_curvature_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
