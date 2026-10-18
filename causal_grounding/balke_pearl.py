"""Sharp Balke-Pearl instrumental-variable bounds for binary Z, X and Y.

This is the hard-constraint reference point for the Naturalness comparison. It
assumes exactly what the classical instrumental-variable model assumes:

* **exclusion** -- ``Z`` affects ``Y`` only through ``X``, encoded structurally
  by writing the outcome as ``Y = ty(X)`` for a latent response type ``ty``;
* **exogeneity** -- ``Z`` is independent of the latent types
  ``(Y(0), Y(1), X(0), X(1))``, encoded by giving both levels of ``Z`` one
  shared type distribution.

It is *not* the ``c = d = 0`` limit of the Naturalness linear program, which
constrains ``theta_{star,x,z} = P(Y(star)=1 | X=x, Z=z; idle)`` to vary by at
most ``2c`` across levels of the candidate. At ``c = 0`` that forbids variation
the instrumental-variable assumptions permit: ``X=1, Z=1`` mixes compliers with
always-takers while ``X=1, Z=0`` selects always-takers alone, and those strata
have different ``Y(0)`` distributions. The two constraint sets are not nested,
so neither bound dominates the other in general.

The response-type linear program is solved directly rather than through a
closed form. Defiers are retained: Balke and Pearl assume no monotonicity.

Reference: Balke and Pearl, "Bounds on Treatment Effects from Studies with
Imperfect Compliance", JASA 92(439), 1997.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

#: Latent treatment response types as ``(X(Z=0), X(Z=1))``: never-taker,
#: complier, defier, always-taker. Defiers are kept because no monotonicity
#: assumption is imposed.
TREATMENT_TYPES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

#: Latent outcome response types as ``(Y(X=0), Y(X=1))``.
OUTCOME_TYPES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class BalkePearlInterval:
    """One Balke-Pearl bound with the diagnostics needed to read it honestly.

    Attributes:
        lower: Sharp lower bound on ``E[Y(1) - Y(0)]``.
        upper: Sharp upper bound.
        feasible: Whether the observed table admits any distribution over latent
            types satisfying exclusion and exogeneity. A ``False`` here is the
            instrumental inequality firing: the data refute the assumptions, and
            the returned interval falls back to the Manski range rather than
            reporting an empty or fabricated set.
        status: ``"sharp"``, ``"infeasible"`` or ``"unsupported"``.
    """

    lower: float
    upper: float
    feasible: bool
    status: str


def _design_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Return the equality design matrix and the treatment-effect objective.

    The 16 latent types are indexed by ``(outcome type, treatment type)``. For
    an instrument level ``z`` a type realises ``X = tx[z]`` and then
    ``Y = ty[X]``, which is where the exclusion restriction enters: the outcome
    reads the treatment it actually received and never ``z`` itself.

    Returns:
        ``(A, c)`` where ``A`` has one row per ``(z, y, x)`` cell mapping type
        probabilities onto ``P(Y=y, X=x | Z=z)``, and ``c`` is the per-type
        individual treatment effect ``ty(1) - ty(0)``.
    """
    types = list(product(OUTCOME_TYPES, TREATMENT_TYPES))
    rows = []
    for z, y, x in product((0, 1), (0, 1), (0, 1)):
        rows.append([
            1.0 if (tx[z] == x and ty[tx[z]] == y) else 0.0
            for ty, tx in types
        ])
    objective = np.array([float(ty[1] - ty[0]) for ty, _ in types])
    return np.asarray(rows, dtype=float), objective


_A_EQ, _OBJECTIVE = _design_matrix()


def balke_pearl_interval(joint: np.ndarray, *, tolerance: float = 1e-9) -> BalkePearlInterval:
    """Return sharp Balke-Pearl bounds on the average treatment effect.

    Args:
        joint: Array of shape ``(2, 2, 2)`` holding ``P(Y=y, X=x | Z=z)`` indexed
            as ``joint[z, y, x]``. Each ``joint[z]`` must sum to one.
        tolerance: Slack allowed when checking that each instrument level's
            table is a probability distribution.

    Returns:
        The interval, with ``feasible=False`` and the Manski range ``[-1, 1]``
        when the instrumental inequality is violated.

    Raises:
        ValueError: If ``joint`` has the wrong shape, holds negative mass, or
            does not normalise within ``tolerance``.
    """
    joint = np.asarray(joint, dtype=float)
    if joint.shape != (2, 2, 2):
        raise ValueError(f"Balke-Pearl needs a (2,2,2) table of P(Y,X|Z); got {joint.shape}")
    if np.any(joint < -tolerance) or not np.isfinite(joint).all():
        raise ValueError("Balke-Pearl table must hold finite nonnegative mass")
    totals = joint.sum(axis=(1, 2))
    if np.any(np.abs(totals - 1.0) > 1e-6):
        raise ValueError(f"Each instrument level must sum to one; got {totals}")

    b_eq = np.array([joint[z, y, x] for z, y, x in product((0, 1), (0, 1), (0, 1))])
    bounds = [(0.0, 1.0)] * _A_EQ.shape[1]
    low = linprog(_OBJECTIVE, A_eq=_A_EQ, b_eq=b_eq, bounds=bounds, method="highs")
    high = linprog(-_OBJECTIVE, A_eq=_A_EQ, b_eq=b_eq, bounds=bounds, method="highs")
    if not (low.success and high.success):
        # The instrumental inequality has fired: no latent-type distribution
        # reproduces this table under exclusion and exogeneity. Reporting an
        # empty set would silently drop the cell, so hand back Manski instead
        # and let the caller see the refutation in the status.
        return BalkePearlInterval(-1.0, 1.0, feasible=False, status="infeasible")
    return BalkePearlInterval(float(low.fun), float(-high.fun), feasible=True, status="sharp")


def joint_table_from_counts(counts: np.ndarray) -> np.ndarray:
    """Normalise a ``(2, 2, 2)`` count array of ``(z, y, x)`` into P(Y,X|Z).

    Args:
        counts: Observed counts indexed as ``counts[z, y, x]``.

    Returns:
        The conditional table, or an array of NaN when either instrument level
        is empty, which the caller must treat as an unsupported cell.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=(1, 2), keepdims=True)
    if np.any(totals == 0):
        return np.full((2, 2, 2), np.nan)
    return counts / totals


def intersect_intervals(intervals: Sequence[BalkePearlInterval]) -> BalkePearlInterval:
    """Intersect several instruments' bounds on the same estimand.

    Each valid instrument bounds the same effect, so the intersection is valid
    whenever every contributing instrument is valid. That proviso is the point:
    an invalid instrument can drag the intersection off the truth, and Balke-
    Pearl has no way to detect this from observational data alone. A crossed
    intersection is reported as infeasible rather than silently reordered.

    Args:
        intervals: Feasible intervals to combine; unsupported ones are skipped.

    Returns:
        The intersection, or the Manski range when nothing usable was supplied.
    """
    usable = [i for i in intervals if i.feasible and np.isfinite(i.lower) and np.isfinite(i.upper)]
    if not usable:
        return BalkePearlInterval(-1.0, 1.0, feasible=False, status="unsupported")
    lower = max(i.lower for i in usable)
    upper = min(i.upper for i in usable)
    if lower > upper:
        return BalkePearlInterval(-1.0, 1.0, feasible=False, status="infeasible")
    return BalkePearlInterval(lower, upper, feasible=True, status="sharp")
