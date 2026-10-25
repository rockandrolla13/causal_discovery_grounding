"""Paper grids: the one definition of the paper's epsilon, tau and beta grids.

Every module that needs one of these values imports it from here rather than
copying the literal. Changing a value here changes the paper's results.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Naturalness thresholds ``epsilon`` searched by the trigger grid.
EPSILON_GRID: tuple[float, ...] = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.50)
#: Relevance thresholds ``tau`` searched by the trigger grid.
TAU_GRID: tuple[float, ...] = (0.01, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
#: Smallest relevance threshold ``tau`` of the paper's experiments.
MIN_RELEVANCE_TAU = 0.075
#: OSRCT selection strengths ``beta`` of the paper's ManyLabs experiments.
PAPER_BETAS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

#: Naturalness thresholds ``epsilon`` of the paper's experiments.
PAPER_EPSILONS: tuple[float, ...] = tuple(EPSILON_GRID)
#: Relevance grid ``tau >= 0.075`` of the paper's experiments (plain ``>=``, no slack).
PAPER_TAUS: tuple[float, ...] = tuple(
    tau for tau in TAU_GRID if tau >= MIN_RELEVANCE_TAU
)


def floor_taus(taus: Sequence[float], floor: float) -> tuple[float, ...]:
    """Return the tau grid values at or above ``floor``.

    The comparison allows ``1e-12`` of slack below ``floor``, so a floor read
    back from text with rounding error still keeps its own grid value.

    Args:
        taus: Tau grid to filter.
        floor: Smallest tau to keep.

    Returns:
        The kept values as floats, in input order.

    Raises:
        ValueError: If no value is at or above ``floor``.
    """
    kept = tuple(float(t) for t in taus if float(t) >= float(floor) - 1e-12)
    if not kept:
        raise ValueError(f"No tau value is at or above floor {floor}")
    return kept


__all__ = [
    "EPSILON_GRID",
    "MIN_RELEVANCE_TAU",
    "PAPER_BETAS",
    "PAPER_EPSILONS",
    "PAPER_TAUS",
    "TAU_GRID",
    "floor_taus",
]
