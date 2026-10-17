"""Step coefficients for Information-Weighted Consensus (IWC).

Both variants operate only on steps already admitted by the spectral gate.  ``iwc``
is the paper's original step-mean softmax (Eq. 11); ``iwc-stable`` uses the
token-mass-preserving construction (Eq. 13-16).  The latter nests spectral SFT
exactly when ``interpolation=0``.
"""

import math
import random
import sys


def _validate(entropies: list[float], lengths: list[int], temperature: float) -> None:
    if len(entropies) != len(lengths):
        raise ValueError("entropies and lengths must have the same number of steps")
    if not entropies:
        return
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("selected step lengths must be positive")
    if not all(math.isfinite(value) for value in entropies):
        raise ValueError("entropies must be finite")


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    normalizer = sum(exponentials)
    return [value / normalizer for value in exponentials]


def vanilla_iwc_step_weights(
    entropies: list[float], lengths: list[int], temperature: float = 1.0
) -> list[float]:
    """Eq. 11: softmax entropy with unit *step*-mean coefficients.

    This intentionally does not preserve token mass when step length correlates
    with entropy.  It is retained as the direct baseline against IWC-Stable.
    """
    _validate(entropies, lengths, temperature)
    if not entropies:
        return []
    return [len(entropies) * probability for probability in _softmax(
        [entropy / temperature for entropy in entropies]
    )]


def stable_iwc_step_weights(
    entropies: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Eq. 13-16: clipped standardized entropy, normalized by token mass."""
    _validate(entropies, lengths, temperature)
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be in [0, 1]")
    if clip < 0:
        raise ValueError("clip must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not entropies:
        return []

    mean = sum(entropies) / len(entropies)
    variance = sum((entropy - mean) ** 2 for entropy in entropies) / len(entropies)
    std = math.sqrt(variance)
    z_scores = [0.0 if std < epsilon else (entropy - mean) / (std + epsilon) for entropy in entropies]
    # The common max shift leaves Eq. 14's normalized ratios unchanged while
    # remaining finite for a valid but very small positive temperature.
    scaled = [max(-clip, min(clip, z_score)) / temperature for z_score in z_scores]
    maximum = max(scaled)
    # Preserve the method's strict positivity even when a legal tiny tau makes
    # the smallest clipped ratio fall below float64's representable exponent.
    raw = [max(math.exp(value - maximum), sys.float_info.min) for value in scaled]
    token_mass = sum(lengths)
    weighted_mass = sum(length * weight for length, weight in zip(lengths, raw))
    normalized = [weight * token_mass / weighted_mass for weight in raw]
    return [(1.0 - interpolation) + interpolation * weight for weight in normalized]


def shuffled_iwc_step_weights(
    entropies: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
    seed: int = 0,
) -> list[float]:
    """Control: shuffle the entropy *vector* (not the final coefficients) before weighting.

    Permutes entropies across selected steps, then reruns Eq. 13-16's full construction
    -- including its length-weighted-mass renormalization -- on that shuffled pairing. The
    resulting coefficients are therefore NOT a permutation of the real run's coefficients:
    a different entropy-length pairing changes the renormalization constant, so both which
    step gets which value AND the exact values can differ from stable_iwc_step_weights's
    real output. What IS guaranteed unchanged: the marginal entropy distribution, the
    spectral gate (selected step set), and this control's own token-mass invariant
    (Sigma n_i alpha_i still equals Sigma n_i). Isolates whether the specific
    entropy-to-step correspondence matters, not whether reweighting in general does.
    """
    shuffled = list(entropies)
    random.Random(seed).shuffle(shuffled)
    return stable_iwc_step_weights(shuffled, lengths, temperature, interpolation, clip, epsilon)


def reverse_iwc_step_weights(
    entropies: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Control: invert the entropy ranking (negate before standardizing).

    Standardization is odd in the input (z(-x) = -z(x)), so this exactly reverses which
    steps get the largest coefficient while preserving the token-mass invariant -- tests
    the paper's "high entropy -> supervise harder" direction against its opposite.
    """
    return stable_iwc_step_weights(
        [-entropy for entropy in entropies], lengths, temperature, interpolation, clip, epsilon
    )
