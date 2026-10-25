"""Assumption-free Manski bounds on a binary-outcome CATE.

With a binary outcome and treatment, the CATE in a cell lies in

``[a1 - a0 - n1 / n, a1 - a0 + n0 / n]``,

where ``n`` counts the cell's rows, ``n0`` and ``n1`` its control and treated
rows, and ``a_x`` is the number of events in arm ``x`` divided by ``n``. Each
interval has width one. No assumption beyond a binary outcome enters, so these
bounds are the fallback every other method in the comparisons is scored
against.

Reference: Manski, "Nonparametric Bounds on Treatment Effects", AER 80(2), 1990.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .loaders.schema import OUTCOME_COLUMN, TREATMENT_COLUMN


def manski_interval(cell: pd.DataFrame, *, treatment: str = TREATMENT_COLUMN,
                    outcome: str = OUTCOME_COLUMN) -> tuple[float, float]:
    """Return the Manski interval of one cell.

    A missing treatment arm counts as unobserved mass. It is neither dropped
    nor filled in.

    Args:
        cell: The cell's observational rows.
        treatment: Binary treatment column.
        outcome: Binary outcome column.

    Returns:
        ``(lower, upper)``. An empty cell gives ``(-1.0, 1.0)``.
    """
    n = len(cell)
    if not n:
        return -1., 1.
    arm0, arm1 = cell[treatment] == 0, cell[treatment] == 1
    n0, n1 = int(arm0.sum()), int(arm1.sum())
    a0 = float(cell.loc[arm0, outcome].sum()) / n
    a1 = float(cell.loc[arm1, outcome].sum()) / n
    return a1 - a0 - n1 / n, a1 - a0 + n0 / n


def manski_bounds(target_idle: pd.DataFrame, strata: pd.DataFrame, covariates: Sequence[str],
                  min_count: int, *, treatment: str = TREATMENT_COLUMN,
                  outcome: str = OUTCOME_COLUMN) -> pd.DataFrame:
    """Compute unsmoothed binary Manski bounds and a shared fitting-support mask.

    Args:
        target_idle: Target observational rows, the subset every learner fits.
        strata: The evaluation cells, one row per covariate combination. A
            cell with no fitting rows gets ``[-1, 1]``.
        covariates: Columns that define a cell.
        min_count: Rows each arm needs for the cell to be ``common_supported``.
        treatment: Binary treatment column.
        outcome: Binary outcome column.

    Returns:
        One row per stratum with the covariates, ``lower``, ``upper``,
        ``n_fit``, ``n_fit_arm0``, ``n_fit_arm1`` and ``common_supported``.
    """
    records = []
    for z in strata[list(covariates)].itertuples(index=False, name=None):
        mask = np.logical_and.reduce([target_idle[c].to_numpy() == v for c, v in zip(covariates, z)])
        cell = target_idle.loc[mask]
        n0, n1 = int((cell[treatment] == 0).sum()), int((cell[treatment] == 1).sum())
        lower, upper = manski_interval(cell, treatment=treatment, outcome=outcome)
        records.append(dict(**dict(zip(covariates, z)), lower=lower, upper=upper, n_fit=len(cell),
                            n_fit_arm0=n0, n_fit_arm1=n1,
                            common_supported=n0 >= min_count and n1 >= min_count))
    return pd.DataFrame(records)
