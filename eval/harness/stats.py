"""Intervals for rates and latency. Pure Python, seeded, no packages.

- Rates: Wilson score interval per scenario; overall, a bootstrap over scenarios
  (whole scenarios are resampled, since runs of one scenario are correlated).
- Latency: percentiles (nearest rank) with a bootstrap over conversations.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z95 = 1.959964
BOOTSTRAP_SAMPLES = 2000


def wilson(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; ``q`` in (0, 1]."""
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _interval(estimates: list[float]) -> tuple[float, float]:
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def cluster_bootstrap_rate(
    clusters: Sequence[Sequence[int]], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> tuple[float, float]:
    """95 % interval of the pooled rate, resampling whole clusters (scenarios)."""
    rng = random.Random(seed)
    filled = [c for c in clusters if c]
    if not filled:
        return 0.0, 1.0
    rates = []
    for _ in range(samples):
        pick = [filled[rng.randrange(len(filled))] for _ in filled]
        total = sum(len(c) for c in pick)
        rates.append(sum(sum(c) for c in pick) / total)
    return _interval(rates)


def bootstrap_percentile(
    conversations: Sequence[Sequence[float]],
    q: float,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """95 % interval of a latency percentile, resampling whole conversations."""
    rng = random.Random(seed)
    filled = [c for c in conversations if c]
    if not filled:
        return math.nan, math.nan
    estimates = []
    for _ in range(samples):
        pick = [filled[rng.randrange(len(filled))] for _ in filled]
        estimates.append(percentile([x for c in pick for x in c], q))
    return _interval(estimates)
