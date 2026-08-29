import torch

from spectral_utils import (
    analyze_gradient_matrix,
    cumulative_energy,
    effective_rank,
    step_spectral_strengths,
    token_leverage_scores,
)


def test_cumulative_energy_is_monotone_and_normalized():
    energy = cumulative_energy(torch.tensor([3.0, 2.0, 1.0]))
    assert torch.all(energy[1:] >= energy[:-1])
    assert torch.isclose(energy[-1], torch.tensor(1.0))


def test_effective_rank_on_exactly_low_rank_matrix():
    # 3 nonzero singular values -> k* must be 3 at any cutoff below 1.0
    singular_values = torch.tensor([5.0, 4.0, 3.0, 0.0, 0.0])
    assert effective_rank(singular_values, energy_cutoff=0.999) == 3


def test_effective_rank_recovers_planted_rank_via_svd():
    torch.manual_seed(0)
    rank = 4
    left, right = torch.randn(200, rank), torch.randn(rank, 64)
    matrix = left @ right  # exactly rank 4
    singular_values = torch.linalg.svdvals(matrix)
    assert effective_rank(singular_values, energy_cutoff=0.95) == rank


def test_leverage_scores_sum_to_k_star():
    # Columns of U are orthonormal, so sum_t ||U[t,:k]||^2 == k exactly.
    torch.manual_seed(0)
    u_matrix, _, _ = torch.linalg.svd(torch.randn(50, 16), full_matrices=False)
    leverage = token_leverage_scores(u_matrix, k_star=5)
    assert torch.isclose(leverage.sum(), torch.tensor(5.0), atol=1e-4)


def test_step_strength_is_mean_leverage_over_span():
    leverage = torch.tensor([1.0, 3.0, 10.0, 20.0])
    assert step_spectral_strengths(leverage, [(0, 2), (2, 4)]) == [2.0, 15.0]


def test_step_with_empty_span_gets_zero_strength():
    assert step_spectral_strengths(torch.tensor([1.0, 2.0]), [(1, 1)]) == [0.0]


def test_analyze_gradient_matrix_ranks_aligned_step_highest():
    # Step 0 rows lie in the dominant direction; step 1 rows are weak noise.
    torch.manual_seed(0)
    direction = torch.zeros(1, 32)
    direction[0, 0] = 1.0
    aligned = direction.repeat(10, 1) * 10.0
    noise = torch.randn(10, 32) * 0.01
    gradient_matrix = torch.cat([aligned, noise], dim=0)

    result = analyze_gradient_matrix(gradient_matrix, [(0, 10), (10, 20)], energy_cutoff=0.95)

    assert result.k_star >= 1
    assert result.step_strengths[0] > result.step_strengths[1]
