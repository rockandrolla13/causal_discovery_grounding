"""Naturalness bounds averaged over posterior tolerance draws.

Each tolerance draw from :meth:`NaturalnessGrounder.sample_tolerances` sets the
constraints of one target linear program. :func:`posterior_target_bounds`
solves one program per draw and summarises the draws per target stratum: the
interval is the mean of the endpoints over the draws whose program is optimal,
and ``p_optimal`` is the share of such draws, as in the paper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .lp_solver_learned_naturalness import (
    HIERARCHICAL_DIRICHLET_BACKENDS,
    HIERARCHICAL_DIRICHLET_DRAW_BACKENDS,
    TargetObservedBinaryQuantities,
    solve_target_bounds_from_observed_binary_quantities,
)
from .seeding import posterior_sample_seeds

#: Target probability backends of
#: :func:`~causal_grounding.estimate_target_observed_binary_quantities`.
TARGET_PROBABILITY_BACKENDS: frozenset[str] = frozenset(
    {
        "empirical",
        "xgboost",
        "tabicl_v2",
        *HIERARCHICAL_DIRICHLET_BACKENDS,
        *HIERARCHICAL_DIRICHLET_DRAW_BACKENDS,
    }
)
#: Backends that draw one target PMF per tolerance draw.
TARGET_PROBABILITY_DRAW_BACKENDS: frozenset[str] = frozenset(
    HIERARCHICAL_DIRICHLET_DRAW_BACKENDS
)

SolveDraw = Callable[[TargetObservedBinaryQuantities, pd.DataFrame], pd.DataFrame]
DrawQuantities = Callable[[int], TargetObservedBinaryQuantities]


def target_probability_draw_seeds(random_seed: int, n_draws: int) -> list[int]:
    """Return one independent target-PMF seed per tolerance draw.

    Args:
        random_seed: Seed of the run.
        n_draws: Number of tolerance draws.

    Returns:
        ``n_draws`` seeds, the ``posterior_sample_seed`` of draw ``k`` at
        position ``k``. Same derivation as Experiment I.
    """
    return posterior_sample_seeds(random_seed, n_draws)


def _solve_draw(
    quantities: TargetObservedBinaryQuantities, draw: pd.DataFrame
) -> pd.DataFrame:
    """Solve the target linear program of one tolerance draw."""
    return solve_target_bounds_from_observed_binary_quantities(
        quantities,
        draw,
        return_details=True,
        include_intervention_diagnostics=False,
    )


def posterior_target_bounds(
    quantities: TargetObservedBinaryQuantities,
    tolerance_draws: pd.DataFrame,
    covariates: Sequence[str],
    *,
    draw_quantities: DrawQuantities | None = None,
    solve_draw: SolveDraw | None = None,
    n_jobs: int = 1,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Solve the target program per tolerance draw and summarise the draws.

    Args:
        quantities: Target probabilities. Their strata ``z_values`` define the
            output rows, and every draw uses them unless ``draw_quantities``
            is given.
        tolerance_draws: Output of
            :meth:`~causal_grounding.NaturalnessGrounder.sample_tolerances`,
            with a ``sample_id`` column.
        covariates: Ordered covariate columns of the strata.
        draw_quantities: Optional map from ``sample_id`` to that draw's target
            probabilities, for backends that redraw the target PMF per draw.
            It must be deterministic in ``sample_id``.
        solve_draw: Optional solver ``(quantities, draw) -> rows`` returning
            one row per stratum with ``status``, ``lower`` and ``upper``.
            Defaults to
            :func:`~causal_grounding.solve_target_bounds_from_observed_binary_quantities`
            with details and without intervention diagnostics.
        n_jobs: Threads that solve the draws. ``1`` solves them in order.
            Results do not depend on ``n_jobs``: every draw is seeded before
            solving and the rows keep the draw order.
        return_samples: Also return the per-draw rows.

    Returns:
        ``bounds``: one row per stratum of ``quantities`` with the covariates,
        ``lower``, ``upper`` and ``p_optimal``. The endpoints are the means
        over the optimal draws and ``NaN`` where no draw is optimal, and
        ``p_optimal`` is 0 there. With ``return_samples``, ``(bounds,
        samples)``, where ``samples`` holds every draw's rows and a
        ``sample_id`` column.

    Raises:
        ValueError: If ``tolerance_draws`` is empty or has no ``sample_id``
            column, or ``n_jobs`` is below 1.
    """
    if "sample_id" not in tolerance_draws.columns or tolerance_draws.empty:
        raise ValueError("tolerance_draws must be non-empty with a 'sample_id' column")
    if int(n_jobs) < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    covariates = list(covariates)
    solve = _solve_draw if solve_draw is None else solve_draw
    grouped = list(tolerance_draws.groupby("sample_id", sort=True))

    def solve_one(item: tuple[object, pd.DataFrame]) -> pd.DataFrame:
        """Solve one draw with its own target probabilities."""
        sample_id, draw = item
        table = quantities if draw_quantities is None else draw_quantities(int(sample_id))
        return solve(table, draw).assign(sample_id=sample_id)

    workers = min(int(n_jobs), len(grouped))
    if workers == 1:
        frames = [solve_one(item) for item in grouped]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            frames = list(executor.map(solve_one, grouped))
    samples = pd.concat(frames, ignore_index=True)

    optimal = samples["status"].eq("optimal")
    summary = (
        samples.assign(optimal=optimal)
        .groupby(covariates, observed=True)
        .agg(p_optimal=("optimal", "mean"))
        .reset_index()
    )
    endpoints = (
        samples[optimal]
        .groupby(covariates, observed=True)[["lower", "upper"]]
        .mean()
        .reset_index()
    )
    strata = pd.DataFrame(quantities.z_values, columns=covariates)
    bounds = strata.merge(summary, on=covariates, how="left").merge(
        endpoints, on=covariates, how="left"
    )
    bounds["p_optimal"] = bounds["p_optimal"].fillna(0.0)
    bounds = bounds[[*covariates, "lower", "upper", "p_optimal"]]
    return (bounds, samples) if return_samples else bounds
