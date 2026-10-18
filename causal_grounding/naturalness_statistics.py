"""Public helpers for Naturalness support and trigger statistics."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def as_tuple(values: Any) -> Tuple[Any, ...]:
    """Normalize scalar or row-like values to a tuple.

    Args:
        values: A scalar, list, NumPy array, pandas Series, or tuple.

    Returns:
        The values represented as one tuple.
    """
    if isinstance(values, tuple):
        return values
    if isinstance(values, pd.Series):
        return tuple(values.tolist())
    if isinstance(values, np.ndarray):
        return tuple(values.tolist())
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def clip_probability(value: float) -> float:
    """Clip a probability-like scalar to the closed unit interval.

    Args:
        value: Numeric value to constrain.

    Returns:
        ``value`` clipped to ``[0, 1]``.
    """
    return float(np.clip(value, 0.0, 1.0))


def safe_sorted(values: Iterable[Any]) -> List[Any]:
    """Sort possibly mixed categorical values deterministically.

    Args:
        values: Discrete values that may not be mutually orderable.

    Returns:
        Values sorted by type then display representation.
    """
    return sorted(list(values), key=lambda value: (str(type(value)), str(value)))


def is_binary_series(series: pd.Series) -> bool:
    """Check whether every nonmissing value is binary.

    Args:
        series: Values to validate.

    Returns:
        True when values are contained in the binary ``0``/``1`` set.
    """
    values = set(series.dropna().unique().tolist())
    return values.issubset({0, 1, 0.0, 1.0, False, True})


def observed_covariate_values(idle: pd.DataFrame, covariate: str) -> List[Any]:
    """Return observed idle-regime values for one candidate covariate.

    Args:
        idle: Prepared idle-regime rows.
        covariate: Candidate coordinate in the Naturalness grid.

    Returns:
        Deterministically ordered nonmissing observed values.
    """
    return safe_sorted(idle[covariate].dropna().unique().tolist())


def assemble_covariate_state(
    j: int,
    z_j: Any,
    z_minus: Tuple[Any, ...],
    n_covariates: int,
) -> Tuple[Any, ...]:
    """Insert one coordinate into a ``Z_-j`` tuple to rebuild ``Z``.

    Args:
        j: Index of the candidate covariate.
        z_j: Value at candidate index ``j``.
        z_minus: Ordered state values excluding index ``j``.
        n_covariates: Length of the full state vector.

    Returns:
        Full ordered covariate-state tuple.
    """
    values = []
    z_minus_iterator = iter(z_minus)
    for index in range(n_covariates):
        values.append(z_j if index == j else next(z_minus_iterator))
    return tuple(values)


def compute_i_deviation(
    covariates: Sequence[str],
    j: int,
    z_minus_values: Sequence[Tuple[Any, ...]],
    probability: Callable[[int, int, Tuple[Any, ...]], float],
    idle: pd.DataFrame,
) -> float:
    """Compute the maximum observed Naturalness outcome-stability statistic.

    Args:
        covariates: Ordered full covariate vector defining ``Z``.
        j: Candidate covariate index.
        z_minus_values: Supported comparison slices of ``Z_-j``.
        probability: Callback for ``P(Y=y | X=x, Z=z; idle)``.
        idle: Idle-regime rows used to enumerate observed candidate levels.

    Returns:
        ``I_j``, or NaN when no candidate-level pair can be compared.
    """
    z_j_values = observed_covariate_values(idle, covariates[j])
    differences: List[float] = []
    for z_minus in z_minus_values:
        for z_j, z_j_prime in combinations(z_j_values, 2):
            z = assemble_covariate_state(j, z_j, z_minus, len(covariates))
            z_prime = assemble_covariate_state(
                j,
                z_j_prime,
                z_minus,
                len(covariates),
            )
            for x in [0, 1]:
                for y in [0, 1]:
                    differences.append(
                        abs(probability(y, x, z) - probability(y, x, z_prime))
                    )
    return float(np.max(differences)) if differences else float("nan")


def compute_d_strength(
    covariates: Sequence[str],
    j: int,
    z_minus_values: Sequence[Tuple[Any, ...]],
    probability: Callable[[int, Tuple[Any, ...]], float],
    idle: pd.DataFrame,
) -> float:
    """Compute the minimum observed Naturalness treatment-relevance statistic.

    Args:
        covariates: Ordered full covariate vector defining ``Z``.
        j: Candidate covariate index.
        z_minus_values: Supported comparison slices of ``Z_-j``.
        probability: Callback for ``P(X=x | Z=z; idle)``.
        idle: Idle-regime rows used to enumerate observed candidate levels.

    Returns:
        ``D_j``, or NaN when no candidate-level pair can be compared.
    """
    z_j_values = observed_covariate_values(idle, covariates[j])
    differences: List[float] = []
    for z_minus in z_minus_values:
        for z_j, z_j_prime in combinations(z_j_values, 2):
            z = assemble_covariate_state(j, z_j, z_minus, len(covariates))
            z_prime = assemble_covariate_state(
                j,
                z_j_prime,
                z_minus,
                len(covariates),
            )
            for x in [0, 1]:
                differences.append(abs(probability(x, z) - probability(x, z_prime)))
    return float(np.min(differences)) if differences else float("nan")


__all__ = [
    "as_tuple",
    "assemble_covariate_state",
    "clip_probability",
    "compute_d_strength",
    "compute_i_deviation",
    "is_binary_series",
    "observed_covariate_values",
    "safe_sorted",
]
