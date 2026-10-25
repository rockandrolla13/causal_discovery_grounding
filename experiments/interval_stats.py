"""Interval statistics shared by the Exp25 and Exp32 analysis scripts.

A leaf module: it imports nothing from :mod:`experiments`, so any script can use it
without creating an import cycle.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Normal quantile for a two-sided 95% interval.
WILSON_Z = 1.959963984540054


def wilson_interval(
    successes: Sequence[float] | np.ndarray,
    total: Sequence[float] | np.ndarray,
    *,
    z: float = WILSON_Z,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wilson score intervals for binomial proportions.

    Unlike the Wald interval, it stays within the unit range for proportions near
    zero or one.

    Args:
        successes: Success counts.
        total: Trial counts. Zero-count entries yield missing bounds.
        z: Normal quantile; the default gives a 95% interval.

    Returns:
        Lower and upper bounds, clipped to ``[0, 1]``.
    """

    counts = np.asarray(total, dtype=float)
    hits = np.asarray(successes, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        proportion = hits / counts
        denominator = 1.0 + (z**2) / counts
        centre = (proportion + (z**2) / (2.0 * counts)) / denominator
        half = (
            z
            * np.sqrt(
                proportion * (1.0 - proportion) / counts
                + (z**2) / (4.0 * counts**2)
            )
            / denominator
        )
    valid = counts > 0
    lower = np.where(valid, np.clip(centre - half, 0.0, 1.0), np.nan)
    upper = np.where(valid, np.clip(centre + half, 0.0, 1.0), np.nan)
    return lower, upper


def percentile_interval(
    draws: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Return the central percentile interval of one bootstrap draw vector."""

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )
