"""Student-aligned region weighting (SARW) for P-ALIGN's hybrid targets.

A P-ALIGN response is a teacher prefix closed by ``<End_of_Prefix>``, then a continuation. Global
IWC standardizes entropy over the whole response, and entropy is structurally higher in the teacher
prefix, so it moves loss mass from the continuation to the prefix as a side effect (+15% prefix mass
on Qwen2.5-7B at lambda 0.5). SARW factorizes the coefficient of step k in region r as

    w_k = a_k * b_r

a_k  IWC-Stable computed separately inside each region: it only ranks steps within a region and
     keeps each region's token mass, so it cannot move budget between regions.
b_r  one prefix/continuation ratio per student, ``(G_P / G_C) ** gamma`` where G_r is the frozen
     student's answer gain per token in region r, turned into per-sample coefficients that keep the
     sample's token mass. Gain per token rather than per step: gain piles up on the last steps,
     which are almost all continuation, so a per-step mean would favour the continuation for every
     student instead of measuring the student. For the same reason the last steps of each
     sample (the one stating the answer holds ~half of all continuation gain) can be left out of
     the estimate with ``tail_fraction``; they are still weighted.

The step containing the marker straddles both regions; it is its own region with a_k = b = 1.
"""

import math

from sgl.allocation.iwc import stable_iwc_step_weights

PREFIX, CONTINUATION, JUNCTION = "P", "C", "J"


def step_regions(step_spans: list[tuple[int, int]], boundary: int) -> list[str]:
    """Region of each step given the absolute position of the first continuation token."""
    regions = []
    for start, end in step_spans:
        if end <= boundary:
            regions.append(PREFIX)
        elif start >= boundary:
            regions.append(CONTINUATION)
        else:
            regions.append(JUNCTION)
    return regions


def region_stable_step_weights(
    entropies: list[float],
    lengths: list[int],
    regions: list[str],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """IWC-Stable (Eq. 13-16) run independently on the prefix and on the continuation steps."""
    if not len(entropies) == len(lengths) == len(regions):
        raise ValueError("entropies, lengths and regions must have the same number of steps")
    weights = [1.0] * len(entropies)
    for region in (PREFIX, CONTINUATION):
        indices = [index for index, value in enumerate(regions) if value == region]
        if not indices:
            continue
        region_weights = stable_iwc_step_weights(
            [entropies[index] for index in indices], [lengths[index] for index in indices],
            temperature, interpolation, clip, epsilon,
        )
        for index, weight in zip(indices, region_weights):
            weights[index] = weight
    return weights


def region_gain_ratio(
    gains: list[list[float]],
    lengths: list[list[int]],
    regions: list[list[str]],
    gamma: float = 1.0,
    tail_fraction: float = 0.0,
) -> tuple[float, float, float]:
    """Corpus answer gain per token in the prefix and the continuation, and (G_P / G_C) ** gamma.

    ``tail_fraction`` drops the last ceil(fraction * n_steps) steps of every sample from the estimate.
    """
    if not 0.0 <= tail_fraction < 1.0:
        raise ValueError("tail_fraction must be in [0, 1)")
    totals = {PREFIX: [0.0, 0], CONTINUATION: [0.0, 0]}
    for record_gains, record_lengths, record_regions in zip(gains, lengths, regions):
        keep = len(record_gains) - math.ceil(tail_fraction * len(record_gains))
        for gain, length, region in zip(record_gains[:keep], record_lengths[:keep], record_regions[:keep]):
            if region in totals:
                totals[region][0] += gain
                totals[region][1] += length
    if not totals[PREFIX][1] or not totals[CONTINUATION][1]:
        raise ValueError("region ratio needs both prefix and continuation tokens")
    gain_prefix = totals[PREFIX][0] / totals[PREFIX][1]
    gain_cont = totals[CONTINUATION][0] / totals[CONTINUATION][1]
    if gain_prefix <= 0 or gain_cont <= 0:
        # A power of a non-positive ratio has no budget meaning; better to stop than to guess.
        raise ValueError(f"answer gain per token must be positive in both regions "
                         f"(prefix {gain_prefix:.6g}, continuation {gain_cont:.6g})")
    return gain_prefix, gain_cont, (gain_prefix / gain_cont) ** gamma


def region_budget(lengths: list[int], regions: list[str], ratio: float) -> list[float]:
    """Per-step b_r with b_P / b_C = ratio that keeps the sample's prefix + continuation mass."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    prefix_mass = sum(length for length, region in zip(lengths, regions) if region == PREFIX)
    cont_mass = sum(length for length, region in zip(lengths, regions) if region == CONTINUATION)
    if not prefix_mass or not cont_mass:
        # one region only: there is no budget to move
        return [1.0] * len(lengths)
    cont_coefficient = (prefix_mass + cont_mass) / (ratio * prefix_mass + cont_mass)
    by_region = {PREFIX: ratio * cont_coefficient, CONTINUATION: cont_coefficient, JUNCTION: 1.0}
    return [by_region[region] for region in regions]
