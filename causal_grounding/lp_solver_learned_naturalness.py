"""
Target-site LP solver for learned naturalness tolerances.

The solver in this module implements the prediction LP for binary treatment and outcome.  It takes
target idle-regime data, selected target-triggered naturalness choices, and the
source-learned ``2c``/``2d`` tolerances, then bounds target CATEs by solving
linear programs over ``theta_{star,x,z}`` variables.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from . import naturalness_statistics
from .naturalness_target_policy import (
    TargetNaturalnessChoice,
    TargetNaturalnessPolicyResult,
)


ChoiceLike = Union[
    TargetNaturalnessChoice,
    TargetNaturalnessPolicyResult,
    Mapping[str, Any],
    pd.Series,
]

HIERARCHICAL_DIRICHLET_BACKENDS = {
    "hierarchical_dirichlet",
}
HIERARCHICAL_DIRICHLET_DRAW_BACKENDS = {
    "hierarchical_dirichlet_draws",
    "hierarchical_dirichlet_posterior",
    "dirichlet_posterior",
}


@dataclass(frozen=True)
class TargetObservedBinaryQuantities:
    """
    Target idle-regime binary probabilities used by the prediction LP.

    The LP variables are:

        theta_{star,x,z} = P(Y(star)=1 | X=x,Z=z; idle).

    The observed factual cells are fixed by consistency:

        theta_{x,x,z} = P(Y=1 | X=x,Z=z; idle).

    Treatment probabilities ``p_x1`` and ``p_x0`` supply
    ``pi*(x | z) = P(X=x | Z=z; idle)`` for intervention means:

        mu_star(z) = sum_x theta_{star,x,z} pi*(x | z)
        CATE(z) = mu_1(z) - mu_0(z).

    Args:
        covariates: Ordered covariate columns defining ``Z``.
        z_values: Supported target strata retained for the LP.
        p_x1: ``P(X=1 | Z=z; idle)`` for each supported stratum.
        p_x0: ``P(X=0 | Z=z; idle)`` for each supported stratum.
        p_y1_given_xz: ``P(Y=1 | X=x,Z=z; idle)`` by ``(x, z)``.
        n_z: Target idle counts by stratum.
        n_xz: Target idle counts by treatment and stratum.
    """

    covariates: List[str]
    z_values: List[Tuple[Any, ...]]
    p_x1: Dict[Tuple[Any, ...], float]
    p_x0: Dict[Tuple[Any, ...], float]
    p_y1_given_xz: Dict[Tuple[int, Tuple[Any, ...]], float]
    n_z: Dict[Tuple[Any, ...], int]
    n_xz: Dict[Tuple[int, Tuple[Any, ...]], int]

    @property
    def q_y0(self) -> Dict[Tuple[Any, ...], float]:
        """Return ``P(Y=1 | X=0,Z=z; idle)`` by stratum."""
        return {z: self.p_y1_given_xz[(0, z)] for z in self.z_values}

    @property
    def q_y1(self) -> Dict[Tuple[Any, ...], float]:
        """Return ``P(Y=1 | X=1,Z=z; idle)`` by stratum."""
        return {z: self.p_y1_given_xz[(1, z)] for z in self.z_values}

    def p_x(self, x: int, z: Tuple[Any, ...]) -> float:
        """
        Return ``P(X=x | Z=z; idle)`` for a supported stratum.

        Args:
            x: Binary treatment value.
            z: Supported target covariate stratum.

        Returns:
            Target idle treatment probability for ``x`` at ``z``.
        """
        if x == 0:
            return self.p_x0[z]
        if x == 1:
            return self.p_x1[z]
        raise ValueError("x must be binary")


def estimate_target_observed_binary_quantities(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    regime_col: Optional[str] = None,
    idle_value: str = "idle",
    min_count: int = 5,
    alpha: float = 0.0,
    probability_backend: str = "empirical",
    probability_model_kwargs: Optional[Dict[str, Any]] = None,
    model_covariates: Optional[Sequence[str]] = None,
) -> TargetObservedBinaryQuantities:
    """
    Estimate target idle ``P(X | Z)`` and ``P(Y | X,Z)`` for binary LPs.

    Args:
        data: Target site data; if ``regime_col`` exists, only ``idle_value`` rows are used.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        covariates: Ordered columns defining the target stratum ``Z``.
        model_covariates: Optional subset of ``covariates`` used as features (default: all).
            The empirical backend pools over it and broadcasts to full strata.
        regime_col: Optional regime column.
        idle_value: Regime value identifying observational rows.
        min_count: Minimum support of retained full strata and, for the empirical backend,
            of treatment cells grouped by ``model_covariates``.
        alpha: Symmetric smoothing, empirical backend only; ``0`` gives exact cell proportions.
        probability_backend: ``"empirical"`` (cell counts), ``"hierarchical_dirichlet"``
            (empirical-Bayes Dirichlet shrinkage), ``"hierarchical_dirichlet_draws"`` (its
            posterior mean, or a posterior draw when ``probability_model_kwargs`` has
            ``posterior_sample_seed``), ``"xgboost"`` or ``"tabicl_v2"``.
        probability_model_kwargs: Keyword arguments forwarded to
            ``create_predictor(probability_backend, **probability_model_kwargs)``.

    Returns:
        TargetObservedBinaryQuantities with one entry per supported stratum.
    """
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    covariates = list(covariates)
    model_covariates = list(model_covariates or covariates)
    if not model_covariates:
        raise ValueError("model_covariates must not be empty")
    unknown_model_covariates = sorted(set(model_covariates) - set(covariates))
    if unknown_model_covariates:
        raise ValueError(
            "model_covariates must be a subset of covariates; unknown values: "
            f"{unknown_model_covariates}"
        )
    missing = [col for col in [treatment, outcome] + covariates if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    idle = data.copy()
    if regime_col is not None and regime_col in idle.columns:
        idle = idle[idle[regime_col] == idle_value].copy()
    if idle.empty:
        raise ValueError("target data has no usable idle rows")
    if not naturalness_statistics.is_binary_series(idle[treatment]):
        raise ValueError(f"Treatment column {treatment!r} must be binary")
    if not naturalness_statistics.is_binary_series(idle[outcome]):
        raise ValueError(f"Outcome column {outcome!r} must be binary")

    probability_backend = probability_backend.lower()
    if probability_backend == "empirical":
        return _estimate_target_observed_binary_quantities_empirical(
            idle=idle,
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            model_covariates=model_covariates,
            min_count=min_count,
            alpha=alpha,
        )
    if probability_backend in {
        *HIERARCHICAL_DIRICHLET_BACKENDS,
        *HIERARCHICAL_DIRICHLET_DRAW_BACKENDS,
        "xgboost",
        "tabicl_v2",
    }:
        return _estimate_target_observed_binary_quantities_model(
            idle=idle,
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            model_covariates=model_covariates,
            min_count=min_count,
            probability_backend=probability_backend,
            probability_model_kwargs=probability_model_kwargs,
        )
    raise ValueError(
        "probability_backend must be one of {'empirical', "
        "'hierarchical_dirichlet', 'hierarchical_dirichlet_draws', "
        "'xgboost', 'tabicl_v2'}, "
        f"got {probability_backend!r}"
    )


def _estimate_target_observed_binary_quantities_empirical(
    idle: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    model_covariates: Sequence[str],
    min_count: int,
    alpha: float,
) -> TargetObservedBinaryQuantities:
    """
    Estimate target LP probabilities with exact empirical target idle cells.

    Args:
        idle: Target idle rows only.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        covariates: Ordered covariate columns defining ``Z``.
        model_covariates: Covariate subset used to pool empirical probability
            estimates before broadcasting them to full LP strata.
        min_count: Minimum count required for each pooled treatment cell and
            each retained full stratum.
        alpha: Symmetric smoothing amount for binary probabilities.

    Returns:
        TargetObservedBinaryQuantities with empirical probabilities.
    """
    covariates = list(covariates)
    model_covariates = list(model_covariates)
    pooled_p_x1: Dict[Tuple[Any, ...], float] = {}
    pooled_p_y1_given_xz: Dict[Tuple[int, Tuple[Any, ...]], float] = {}

    pooled_groups = idle.groupby(model_covariates, observed=True, dropna=False)
    for model_z_raw, model_z_frame in pooled_groups:
        model_z = naturalness_statistics.as_tuple(model_z_raw)
        model_z_total = int(len(model_z_frame))
        counts_by_x = {
            x: int(model_z_frame[treatment].eq(x).sum())
            for x in [0, 1]
        }
        if (
            model_z_total < min_count
            or any(counts_by_x[x] < min_count for x in [0, 1])
        ):
            continue

        pooled_p_x1[model_z] = float(
            (counts_by_x[1] + alpha) / (model_z_total + 2.0 * alpha)
        )
        for x in [0, 1]:
            x_frame = model_z_frame[model_z_frame[treatment].eq(x)]
            y_success = int(x_frame[outcome].eq(1).sum())
            pooled_p_y1_given_xz[(x, model_z)] = float(
                (y_success + alpha) / (len(x_frame) + 2.0 * alpha)
            )

    p_x1: Dict[Tuple[Any, ...], float] = {}
    p_x0: Dict[Tuple[Any, ...], float] = {}
    p_y1_given_xz: Dict[Tuple[int, Tuple[Any, ...]], float] = {}
    n_z: Dict[Tuple[Any, ...], int] = {}
    n_xz: Dict[Tuple[int, Tuple[Any, ...]], int] = {}

    grouped = idle.groupby(covariates, observed=True, dropna=False)
    for z_raw, z_frame in grouped:
        z = naturalness_statistics.as_tuple(z_raw)
        z_total = int(len(z_frame))
        if z_total < min_count:
            continue
        z_by_covariate = dict(zip(covariates, z))
        model_z = tuple(z_by_covariate[col] for col in model_covariates)
        if model_z not in pooled_p_x1:
            continue

        n_z[z] = z_total
        p_x1[z] = pooled_p_x1[model_z]
        p_x0[z] = 1.0 - pooled_p_x1[model_z]

        for x in [0, 1]:
            n_xz[(x, z)] = int(z_frame[treatment].eq(x).sum())
            p_y1_given_xz[(x, z)] = pooled_p_y1_given_xz[(x, model_z)]

    z_values = naturalness_statistics.safe_sorted(n_z.keys())
    return TargetObservedBinaryQuantities(
        covariates=covariates,
        z_values=z_values,
        p_x1=p_x1,
        p_x0=p_x0,
        p_y1_given_xz=p_y1_given_xz,
        n_z=n_z,
        n_xz=n_xz,
    )


def _estimate_target_observed_binary_quantities_model(
    idle: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    model_covariates: Sequence[str],
    min_count: int,
    probability_backend: str,
    probability_model_kwargs: Optional[Dict[str, Any]] = None,
) -> TargetObservedBinaryQuantities:
    """
    Estimate target LP probabilities with fitted target idle nuisance models.

    Args:
        idle: Target idle rows only.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        covariates: Ordered covariate columns defining ``Z``.
        model_covariates: Covariate subset used to fit the nuisance models.
        min_count: Minimum count required for retained ``Z`` strata.
        probability_backend: Predictor backend name passed to
            ``create_predictor``.
        probability_model_kwargs: Keyword arguments forwarded to
            ``create_predictor(probability_backend, ...)``.

    Returns:
        TargetObservedBinaryQuantities with model-predicted target idle
        probabilities on every retained target stratum.
    """
    try:
        from .predictors import create_predictor
    except ImportError:  # pragma: no cover - fallback for direct script usage
        from predictors import create_predictor

    draw_seed: Optional[int] = None
    backend_for_model = probability_backend
    model_kwargs = dict(probability_model_kwargs or {})
    if backend_for_model in HIERARCHICAL_DIRICHLET_DRAW_BACKENDS:
        backend_for_model = "hierarchical_dirichlet"
        if "posterior_sample_seed" in model_kwargs:
            draw_seed = int(model_kwargs.pop("posterior_sample_seed"))

    if backend_for_model == "xgboost" and idle[treatment].nunique(dropna=True) < 2:
        raise ValueError("XGBoost target probability backend requires both treatment classes")
    if backend_for_model == "xgboost" and idle[outcome].nunique(dropna=True) < 2:
        raise ValueError("XGBoost target probability backend requires both outcome classes")

    covariates = list(covariates)
    model_covariates = list(model_covariates)
    feature_columns = _target_feature_columns(idle, model_covariates)
    encoded = _encode_target_covariates(
        idle,
        model_covariates,
        feature_columns,
    )

    fit_data = encoded.copy()
    fit_data[treatment] = idle[treatment].astype(int).to_numpy()
    fit_data[outcome] = idle[outcome].astype(int).to_numpy()

    if backend_for_model == "xgboost":
        model_kwargs.setdefault("use_grid_search", False)
    predictor = create_predictor(backend_for_model, **model_kwargs)
    predictor.fit_propensity(fit_data, treatment, feature_columns)
    predictor.fit_outcome(fit_data, outcome, treatment, feature_columns)
    if draw_seed is not None:
        if not hasattr(predictor, "sample_posterior"):
            raise ValueError(
                f"Backend {probability_backend!r} does not support posterior draws"
            )
        predictor = predictor.sample_posterior(random_state=draw_seed)

    p_x1: Dict[Tuple[Any, ...], float] = {}
    p_x0: Dict[Tuple[Any, ...], float] = {}
    p_y1_given_xz: Dict[Tuple[int, Tuple[Any, ...]], float] = {}
    n_z: Dict[Tuple[Any, ...], int] = {}
    n_xz: Dict[Tuple[int, Tuple[Any, ...]], int] = {}

    grouped = idle.groupby(covariates, observed=True, dropna=False)
    for z_raw, z_frame in grouped:
        z = naturalness_statistics.as_tuple(z_raw)
        z_total = int(len(z_frame))
        if z_total < min_count:
            continue

        z_encoded = _make_target_z_frame(
            z,
            covariates,
            model_covariates,
            feature_columns,
        )
        p_x1_z = naturalness_statistics.clip_probability(
            float(predictor.predict_propensity(z_encoded)[0])
        )

        n_z[z] = z_total
        p_x1[z] = p_x1_z
        p_x0[z] = 1.0 - p_x1_z

        for x in [0, 1]:
            n_xz[(x, z)] = int(z_frame[treatment].eq(x).sum())
            p_y1_given_xz[(x, z)] = naturalness_statistics.clip_probability(
                float(predictor.predict_outcome(z_encoded, x)[0])
            )

    z_values = naturalness_statistics.safe_sorted(n_z.keys())
    return TargetObservedBinaryQuantities(
        covariates=covariates,
        z_values=z_values,
        p_x1=p_x1,
        p_x0=p_x0,
        p_y1_given_xz=p_y1_given_xz,
        n_z=n_z,
        n_xz=n_xz,
    )


def _target_feature_columns(data: pd.DataFrame, covariates: Sequence[str]) -> List[str]:
    """
    Build a stable encoded feature schema for target probability models.

    Args:
        data: Target idle rows.
        covariates: Ordered covariate columns defining ``Z``.

    Returns:
        Encoded feature column names used by the target probability model.
    """
    encoded = pd.get_dummies(data[list(covariates)], dummy_na=False)
    if encoded.empty:
        raise ValueError("Target probability model produced no covariate features")
    return encoded.columns.tolist()


def _encode_target_covariates(
    data: pd.DataFrame,
    covariates: Sequence[str],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    One-hot encode target covariates and align to a fitted feature schema.

    Args:
        data: Rows containing target covariates.
        covariates: Ordered covariate columns defining ``Z``.
        feature_columns: Fitted encoded feature schema.

    Returns:
        Numeric encoded covariate frame.
    """
    encoded = pd.get_dummies(data[list(covariates)], dummy_na=False)
    return encoded.reindex(columns=list(feature_columns), fill_value=0).astype(float)


def _make_target_z_frame(
    z: Tuple[Any, ...],
    covariates: Sequence[str],
    model_covariates: Sequence[str],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Convert one target stratum into an encoded feature row.

    Args:
        z: Target stratum tuple.
        covariates: Ordered covariate columns defining ``Z``.
        model_covariates: Covariate subset used by the nuisance models.
        feature_columns: Fitted encoded feature schema.

    Returns:
        One-row encoded covariate frame.
    """
    z = naturalness_statistics.as_tuple(z)
    if len(z) != len(covariates):
        raise ValueError(f"Expected z with {len(covariates)} values, got {len(z)}")
    raw = pd.DataFrame([{col: value for col, value in zip(covariates, z)}])
    return _encode_target_covariates(raw, model_covariates, feature_columns)


def solve_target_bounds_from_observed_binary_quantities(
    quantities: TargetObservedBinaryQuantities,
    choices: Union[ChoiceLike, Sequence[ChoiceLike], pd.DataFrame],
    solver_method: str = "highs",
    z_values: Optional[Sequence[Tuple[Any, ...]]] = None,
    return_details: bool = False,
    include_intervention_diagnostics: bool = True,
) -> Union[Dict[Tuple[Any, ...], Tuple[float, float]], pd.DataFrame]:
    """
    Bound target CATEs from precomputed target idle binary probabilities.

    Several tolerance draws (e.g. sampled GP ``c``/``d``) can reuse the same target idle
    nuisance estimates. Every query stratum shares one LP over
    ``theta_{star,x,z} = P(Y(star)=1 | X=x,Z=z; idle)`` with the factual equalities
    ``theta_{x,x,z} = P(Y=1 | X=x,Z=z; idle)`` and, for each selected row
    ``(j, epsilon, tau)`` and paired strata ``z_a,z_b`` differing only in coordinate ``j``:

        |theta_{star,x,z_a} - theta_{star,x,z_b}| <= 2c_j
        |sum_x theta_{star,x,z_a} p(x | z_a) - sum_x theta_{star,x,z_b} p(x | z_b)| <= 2d_j.

    The CATE bounds at ``z0`` solve
    ``min/max [sum_x theta_{1,x,z0} p(x | z0) - sum_x theta_{0,x,z0} p(x | z0)]``.

    Args:
        quantities: Target idle ``P(X | Z)`` and ``P(Y | X,Z)`` estimates.
        choices: Selected choice, policy result, sequence of choices, or DataFrame rows
            carrying ``covariate_index``, ``two_c`` and ``two_d``.
        solver_method: SciPy ``linprog`` method, usually ``"highs"``.
        z_values: Optional query strata. Defaults to all supported strata.
        return_details: If true, return a per-stratum diagnostic DataFrame; otherwise
            ``{z: (lower, upper)}`` for optimal CATE rows.
        include_intervention_diagnostics: Also bound the intervention means ``mu0`` and
            ``mu1``. Disabling it cuts six solves per stratum to two; CATE endpoints are unchanged.

    Returns:
        Either a bounds dictionary or a detailed DataFrame with LP diagnostics.
    """
    selected_choices = _coerce_choices(choices)
    if not selected_choices:
        empty = _empty_details_frame()
        return empty if return_details else {}
    if not quantities.z_values:
        empty = _empty_details_frame()
        return empty if return_details else {}

    query_z_values = _prepare_query_z_values(z_values, quantities.z_values)
    prepared_choices = _prepare_choices(selected_choices, quantities.covariates)
    if not prepared_choices:
        empty = _empty_details_frame()
        return empty if return_details else {}

    problem = _build_lp_problem(quantities, prepared_choices)
    rows = [
        _solve_query_stratum(
            quantities=quantities,
            problem=problem,
            z=z,
            solver_method=solver_method,
            include_intervention_diagnostics=include_intervention_diagnostics,
        )
        for z in query_z_values
    ]
    details = pd.DataFrame(rows)
    if details.empty:
        return details if return_details else {}

    details["n_choices"] = int(len(prepared_choices))
    details["n_strata"] = int(len(quantities.z_values))
    details["n_variables"] = int(problem["n_variables"])
    details["n_constraints"] = int(problem["n_constraints"])
    details["solver_method"] = solver_method
    details["intervention_diagnostics_computed"] = bool(
        include_intervention_diagnostics
    )

    if return_details:
        return details

    optimal = details[details["status"] == "optimal"]
    return {
        naturalness_statistics.as_tuple(row["z_value"]): (
            float(row["lower"]),
            float(row["upper"]),
        )
        for _, row in optimal.iterrows()
    }


def solve_target_weighted_cate_bounds_from_observed_binary_quantities(
    quantities: TargetObservedBinaryQuantities,
    choices: Union[ChoiceLike, Sequence[ChoiceLike], pd.DataFrame],
    weighted_queries: Mapping[
        Tuple[Any, ...],
        Mapping[Tuple[Any, ...], float],
    ],
    query_covariates: Sequence[str],
    solver_method: str = "highs",
    return_details: bool = False,
    include_intervention_diagnostics: bool = True,
) -> Union[Dict[Tuple[Any, ...], Tuple[float, float]], pd.DataFrame]:
    """
    Bound weighted CATEs over several supported target strata.

    Each query ``v`` supplies nonnegative weights ``q_v(z)`` over full ``Z`` strata that sum
    to one; the optimized estimand is ``sum_z q_v(z) [mu_1(z) - mu_0(z)]``. Bounding this
    marginal or subgroup CATE inside the shared naturalness polytope is at least as tight as
    averaging separately optimized per-stratum endpoints.

    Args:
        quantities: Target idle probabilities defining the common LP.
        choices: Selected naturalness constraints.
        weighted_queries: Mapping from query-value tuples to ``{supported Z tuple: weight}``.
        query_covariates: Ordered names corresponding to each query tuple.
        solver_method: SciPy ``linprog`` method.
        return_details: Return diagnostic rows instead of an optimal-only bounds dictionary.
        include_intervention_diagnostics: Also bound the weighted ``mu0`` and ``mu1``.
            Disabling it cuts six solves per query to two; CATE bounds are unchanged.

    Returns:
        Per-query weighted CATE bounds or their detailed solver diagnostics.
    """

    query_covariates = list(query_covariates)
    if not query_covariates:
        raise ValueError("query_covariates must contain at least one column")
    if len(set(query_covariates)) != len(query_covariates):
        raise ValueError("query_covariates must not contain duplicates")

    selected_choices = _coerce_choices(choices)
    if not selected_choices or not quantities.z_values:
        empty = _empty_weighted_details_frame(query_covariates)
        return empty if return_details else {}

    prepared_choices = _prepare_choices(selected_choices, quantities.covariates)
    if not prepared_choices:
        empty = _empty_weighted_details_frame(query_covariates)
        return empty if return_details else {}

    prepared_queries = _prepare_weighted_queries(
        weighted_queries,
        query_covariates=query_covariates,
        supported_z_values=quantities.z_values,
    )
    if not prepared_queries:
        empty = _empty_weighted_details_frame(query_covariates)
        return empty if return_details else {}

    problem = _build_lp_problem(quantities, prepared_choices)
    rows = [
        _solve_weighted_query(
            quantities=quantities,
            problem=problem,
            query_value=query_value,
            z_weights=z_weights,
            query_covariates=query_covariates,
            solver_method=solver_method,
            include_intervention_diagnostics=include_intervention_diagnostics,
        )
        for query_value, z_weights in prepared_queries
    ]
    details = pd.DataFrame(rows)
    details["n_choices"] = int(len(prepared_choices))
    details["n_strata"] = int(len(quantities.z_values))
    details["n_variables"] = int(problem["n_variables"])
    details["n_constraints"] = int(problem["n_constraints"])
    details["solver_method"] = solver_method
    details["intervention_diagnostics_computed"] = bool(
        include_intervention_diagnostics
    )

    if return_details:
        return details

    optimal = details.loc[details["status"].eq("optimal")]
    return {
        naturalness_statistics.as_tuple(row["query_value"]): (
            float(row["lower"]),
            float(row["upper"]),
        )
        for _, row in optimal.iterrows()
    }


def solve_candidate_state_bounds_from_observed_binary_quantities(
    quantities_by_state: Mapping[Tuple[str, ...], TargetObservedBinaryQuantities],
    choices: Union[ChoiceLike, Sequence[ChoiceLike], pd.DataFrame],
    candidate_states: Mapping[str, Sequence[str]],
    reporting_covariates: Sequence[str],
    solver_method: str = "highs",
    include_intervention_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Bound reporting-stratum CATEs when candidates live in different states.

    A grid with conditioning covariates evaluates each candidate in its own state (the
    conditioning covariates plus the candidate). Choices are grouped by state and each group
    is solved as its own LP on its own ``theta_{star,x,z}`` variables. A group whose state
    equals ``reporting_covariates`` is solved per stratum by
    :func:`solve_target_bounds_from_observed_binary_quantities`; any other group bounds the
    coarser reporting CATE directly through
    :func:`solve_target_weighted_cate_bounds_from_observed_binary_quantities`, with weights
    ``q_v(z) = n(z) / sum_{z' in v} n(z')`` over its supported target idle strata ``z`` in
    reporting stratum ``v``. Every group bounds the same estimand, so the intervals are
    intersected: ``lower(v) = max_s lower_s(v)``, ``upper(v) = min_s upper_s(v)``.

    A stratum is ``optimal`` only if every group that returned it solved optimally, otherwise
    ``nonoptimal``; an intersection with ``lower > upper`` is ``empty_intersection``. A group
    without target support for a stratum constrains nothing there.

    Args:
        quantities_by_state: Target idle probability tables keyed by candidate-state tuple;
            each table's covariates must equal its key.
        choices: Selected naturalness choices carrying ``covariate``, ``two_c`` and ``two_d``.
        candidate_states: State covariates for every candidate covariate.
        reporting_covariates: Ordered covariates defining the reported CATE strata; every
            state must contain them.
        solver_method: SciPy ``linprog`` method.
        include_intervention_diagnostics: Also bound ``mu0`` and ``mu1``; the intersected
            diagnostics are reported when every group computed them.

    Returns:
        One row per reporting stratum in the per-stratum LP detail schema, plus
        ``n_candidate_states``, ``candidate_states``, ``candidate_state_statuses``,
        ``binding_lower_state`` and ``binding_upper_state``. Empty when no choice is usable.

    Raises:
        ValueError: If a choice's covariate has no state, a state has no table, or a state
            omits a reporting covariate.
    """

    reporting = list(reporting_covariates)
    if not reporting:
        raise ValueError("reporting_covariates must contain at least one column")
    if len(set(reporting)) != len(reporting):
        raise ValueError("reporting_covariates must not contain duplicates")

    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for choice in _coerce_choices(choices):
        covariate = _choice_get(choice, "covariate", None)
        if covariate is None or pd.isna(covariate):
            continue
        covariate = str(covariate)
        if covariate not in candidate_states:
            raise ValueError(f"No candidate state is defined for {covariate!r}")
        state = tuple(candidate_states[covariate])
        # Constraint indices must refer to the group's own state.
        grouped.setdefault(state, []).append({
            "covariate": covariate,
            "covariate_index": state.index(covariate),
            "two_c": _choice_get(choice, "two_c", np.nan),
            "two_d": _choice_get(choice, "two_d", np.nan),
            "epsilon": _choice_get(choice, "epsilon", np.nan),
            "tau": _choice_get(choice, "tau", np.nan),
        })

    state_frames: List[pd.DataFrame] = []
    for state, state_choices in grouped.items():
        if state not in quantities_by_state:
            raise ValueError(f"No target probability table for candidate state {state!r}")
        quantities = quantities_by_state[state]
        if list(quantities.covariates) != list(state):
            raise ValueError(
                f"Target probability table covariates {quantities.covariates!r} "
                f"do not match candidate state {state!r}"
            )
        missing = [covariate for covariate in reporting if covariate not in state]
        if missing:
            raise ValueError(
                f"Candidate state {state!r} omits reporting covariates {missing}"
            )
        if list(state) == reporting:
            details = solve_target_bounds_from_observed_binary_quantities(
                quantities,
                state_choices,
                solver_method=solver_method,
                return_details=True,
                include_intervention_diagnostics=include_intervention_diagnostics,
            )
        else:
            positions = [state.index(covariate) for covariate in reporting]
            counts: Dict[Tuple[Any, ...], Dict[Tuple[Any, ...], float]] = {}
            for z in quantities.z_values:
                query_value = tuple(z[position] for position in positions)
                counts.setdefault(query_value, {})[z] = float(quantities.n_z[z])
            weighted_queries = {
                query_value: {
                    z: count / sum(z_counts.values())
                    for z, count in z_counts.items()
                }
                for query_value, z_counts in counts.items()
                if sum(z_counts.values()) > 0
            }
            details = solve_target_weighted_cate_bounds_from_observed_binary_quantities(
                quantities,
                state_choices,
                weighted_queries,
                query_covariates=reporting,
                solver_method=solver_method,
                return_details=True,
                include_intervention_diagnostics=include_intervention_diagnostics,
            ).rename(columns={
                "query_value": "z_value",
                "query_value_str": "z_value_str",
            })
        if details.empty:
            continue
        details = details.copy()
        details["candidate_state"] = "|".join(state)
        state_frames.append(details)

    if not state_frames:
        return _empty_details_frame()
    combined = pd.concat(state_frames, ignore_index=True, sort=False)

    rows: List[Dict[str, Any]] = []
    for _, group in combined.groupby(reporting, sort=True, dropna=False):
        first = group.iloc[0]
        statuses = group["status"].astype(str).tolist()
        all_optimal = all(status == "optimal" for status in statuses)
        lower_values = pd.to_numeric(group["lower"], errors="coerce")
        upper_values = pd.to_numeric(group["upper"], errors="coerce")
        row: Dict[str, Any] = {
            "z_value": first["z_value"],
            "z_value_str": str(first["z_value"]),
            **{covariate: first[covariate] for covariate in reporting},
        }
        if len(group) == 1:
            # A single contributing state is reported exactly as it solved.
            for column in [
                "lower", "upper", "status", "lower_status", "upper_status",
                "mu0_lower", "mu0_upper", "mu0_lower_status", "mu0_upper_status",
                "mu1_lower", "mu1_upper", "mu1_lower_status", "mu1_upper_status",
            ]:
                row[column] = first.get(column, np.nan)
            state_label = str(first["candidate_state"])
            row["binding_lower_state"] = state_label if all_optimal else None
            row["binding_upper_state"] = state_label if all_optimal else None
        elif all_optimal:
            lower_position = int(np.argmax(lower_values.to_numpy()))
            upper_position = int(np.argmin(upper_values.to_numpy()))
            row["lower"] = float(lower_values.iloc[lower_position])
            row["upper"] = float(upper_values.iloc[upper_position])
            row["binding_lower_state"] = str(group["candidate_state"].iloc[lower_position])
            row["binding_upper_state"] = str(group["candidate_state"].iloc[upper_position])
            row["lower_status"] = "optimal"
            row["upper_status"] = "optimal"
            row["status"] = (
                "empty_intersection"
                if row["lower"] > row["upper"] + 1e-8
                else "optimal"
            )
        else:
            row["lower"] = np.nan
            row["upper"] = np.nan
            row["binding_lower_state"] = None
            row["binding_upper_state"] = None
            row["lower_status"] = "nonoptimal"
            row["upper_status"] = "nonoptimal"
            row["status"] = "nonoptimal"
        for star in [0, 1] if len(group) > 1 else []:
            lower_column = f"mu{star}_lower"
            upper_column = f"mu{star}_upper"
            computed = (
                include_intervention_diagnostics
                and all_optimal
                and lower_column in group
                and group[f"mu{star}_lower_status"].eq("optimal").all()
                and group[f"mu{star}_upper_status"].eq("optimal").all()
            )
            row[lower_column] = (
                float(pd.to_numeric(group[lower_column]).max()) if computed else np.nan
            )
            row[upper_column] = (
                float(pd.to_numeric(group[upper_column]).min()) if computed else np.nan
            )
            status = "optimal" if computed else (
                "not_requested" if not include_intervention_diagnostics else "nonoptimal"
            )
            row[f"mu{star}_lower_status"] = status
            row[f"mu{star}_upper_status"] = status
        row["n_choices"] = int(pd.to_numeric(group["n_choices"]).sum())
        row["n_strata"] = int(pd.to_numeric(group["n_strata"]).sum())
        row["n_variables"] = int(pd.to_numeric(group["n_variables"]).sum())
        row["n_constraints"] = int(pd.to_numeric(group["n_constraints"]).sum())
        row["solver_method"] = solver_method
        row["intervention_diagnostics_computed"] = bool(include_intervention_diagnostics)
        row["n_candidate_states"] = int(len(group))
        row["candidate_states"] = json.dumps(group["candidate_state"].astype(str).tolist())
        row["candidate_state_statuses"] = json.dumps(statuses)
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_weighted_queries(
    weighted_queries: Mapping[
        Tuple[Any, ...],
        Mapping[Tuple[Any, ...], float],
    ],
    *,
    query_covariates: Sequence[str],
    supported_z_values: Sequence[Tuple[Any, ...]],
) -> List[Tuple[Tuple[Any, ...], Dict[Tuple[Any, ...], float]]]:
    """Validate weighted query keys, support, and probability masses."""

    supported = {
        naturalness_statistics.as_tuple(z) for z in supported_z_values
    }
    prepared: List[Tuple[Tuple[Any, ...], Dict[Tuple[Any, ...], float]]] = []
    for raw_query_value, raw_weights in weighted_queries.items():
        query_value = naturalness_statistics.as_tuple(raw_query_value)
        if len(query_value) != len(query_covariates):
            raise ValueError(
                f"Query {query_value!r} has {len(query_value)} values; "
                f"expected {len(query_covariates)}"
            )
        z_weights: Dict[Tuple[Any, ...], float] = {}
        for raw_z, raw_weight in raw_weights.items():
            z = naturalness_statistics.as_tuple(raw_z)
            if z not in supported:
                raise ValueError(
                    f"Weighted query {query_value!r} references unsupported "
                    f"stratum {z!r}"
                )
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("Weighted-query masses must be finite and nonnegative")
            if weight > 0.0:
                z_weights[z] = weight
        if not z_weights:
            raise ValueError(f"Weighted query {query_value!r} has no positive mass")
        total = float(sum(z_weights.values()))
        if not np.isclose(total, 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError(
                f"Weighted query {query_value!r} masses sum to {total}, not 1"
            )
        prepared.append((query_value, z_weights))
    return prepared


def _coerce_choices(
    choices: Union[ChoiceLike, Sequence[ChoiceLike], pd.DataFrame],
) -> List[ChoiceLike]:
    """Normalize supported choice containers into a list of selected choices."""
    if isinstance(choices, TargetNaturalnessPolicyResult):
        return list(choices.selected_choices)
    if isinstance(choices, TargetNaturalnessChoice):
        return [choices]
    if isinstance(choices, pd.DataFrame):
        if choices.empty:
            return []
        return [row for _, row in choices.iterrows()]
    if isinstance(choices, (list, tuple)):
        normalized: List[ChoiceLike] = []
        for choice in choices:
            normalized.extend(_coerce_choices(choice))
        return normalized
    if isinstance(choices, (Mapping, pd.Series)):
        return [choices]
    raise TypeError("Unsupported naturalness choice container")


def _prepare_choices(
    choices: Sequence[ChoiceLike],
    covariates: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Validate selected choices and extract LP-relevant naturalness fields.

    The target policy attaches ``two_c = 2c(j,epsilon,tau)`` and
    ``two_d = 2d(j,epsilon,tau)``.  The LP consumes those right-hand sides
    directly because its constraints are stated on potential-outcome
    probabilities, not on hidden response-type probabilities.
    """
    prepared: List[Dict[str, Any]] = []
    n_non_finite = 0
    covariate_to_index = {name: index for index, name in enumerate(covariates)}
    for choice in choices:
        covariate_index = _choice_get(choice, "covariate_index", None)
        if covariate_index is None or pd.isna(covariate_index):
            covariate = _choice_get(choice, "covariate", None)
            covariate_index = covariate_to_index.get(covariate)
        if covariate_index is None or pd.isna(covariate_index):
            continue
        covariate_index = int(covariate_index)
        if covariate_index < 0 or covariate_index >= len(covariates):
            continue

        two_c = float(_choice_get(choice, "two_c", np.nan))
        two_d = float(_choice_get(choice, "two_d", np.nan))
        if not np.isfinite(two_c) or not np.isfinite(two_d):
            n_non_finite += 1
            continue
        prepared.append({
            "covariate": covariates[covariate_index],
            "covariate_index": covariate_index,
            "two_c": max(0.0, two_c),
            "two_d": max(0.0, two_d),
            "epsilon": _choice_get(choice, "epsilon", np.nan),
            "tau": _choice_get(choice, "tau", np.nan),
        })
    # A dropped choice loses both of its constraints, so the bounds are looser
    # than the selected conditions imply (or empty if every choice is dropped).
    if n_non_finite:
        warnings.warn(
            f"Dropped {n_non_finite} of {len(choices)} naturalness choices with "
            "non-finite two_c or two_d; their c and d constraints are not imposed.",
            RuntimeWarning,
            stacklevel=3,
        )
    return prepared


def _choice_get(choice: ChoiceLike, name: str, default: Any = None) -> Any:
    """Read a field from dataclass, dict-like, or pandas choice rows."""
    if isinstance(choice, Mapping):
        return choice.get(name, default)
    if isinstance(choice, pd.Series):
        return choice.get(name, default)
    return getattr(choice, name, default)


def _prepare_query_z_values(
    requested: Optional[Sequence[Tuple[Any, ...]]],
    supported: Sequence[Tuple[Any, ...]],
) -> List[Tuple[Any, ...]]:
    """Return requested query strata that are supported by the target LP."""
    supported_set = {naturalness_statistics.as_tuple(z) for z in supported}
    if requested is None:
        return list(supported)
    return [
        naturalness_statistics.as_tuple(z)
        for z in requested
        if naturalness_statistics.as_tuple(z) in supported_set
    ]


def _build_lp_problem(
    quantities: TargetObservedBinaryQuantities,
    choices: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build common LP matrices shared by all target-stratum objectives.

    Variable indexing:

        theta_{star,x,z} = P(Y(star)=1 | X=x,Z=z; idle).

    Equality rows encode factual consistency:

        theta_{x,x,z} = P(Y=1 | X=x,Z=z; idle).

    For every selected naturalness row and ordered pair ``(z_a,z_b)`` sharing
    ``Z_-j``, inequality rows encode:

        theta_{star,x,z_a} - theta_{star,x,z_b} <= 2c_j

    and:

        sum_x theta_{star,x,z_a} pi*(x | z_a)
        - sum_x theta_{star,x,z_b} pi*(x | z_b) <= 2d_j.

    Ordered pairs are generated both ways, so these one-sided inequalities are
    equivalent to absolute-difference bounds.
    """
    var_index: Dict[Tuple[int, int, Tuple[Any, ...]], int] = {}
    for z in quantities.z_values:
        for star in [0, 1]:
            for x in [0, 1]:
                var_index[(star, x, z)] = len(var_index)

    n_variables = len(var_index)
    a_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    a_eq: List[np.ndarray] = []
    b_eq: List[float] = []

    for z in quantities.z_values:
        for x in [0, 1]:
            row = np.zeros(n_variables)
            row[var_index[(x, x, z)]] = 1.0
            a_eq.append(row)
            b_eq.append(quantities.p_y1_given_xz[(x, z)])

    for choice in choices:
        j = int(choice["covariate_index"])
        for z_a, z_b in _ordered_z_pairs_by_minus(quantities.z_values, j):
            for star in [0, 1]:
                for x in [0, 1]:
                    row = np.zeros(n_variables)
                    row[var_index[(star, x, z_a)]] = 1.0
                    row[var_index[(star, x, z_b)]] = -1.0
                    a_ub.append(row)
                    b_ub.append(float(choice["two_c"]))

                row = np.zeros(n_variables)
                for x in [0, 1]:
                    row[var_index[(star, x, z_a)]] += quantities.p_x(x, z_a)
                    row[var_index[(star, x, z_b)]] -= quantities.p_x(x, z_b)
                a_ub.append(row)
                b_ub.append(float(choice["two_d"]))

    return {
        "var_index": var_index,
        "n_variables": n_variables,
        "a_ub": np.vstack(a_ub) if a_ub else None,
        "b_ub": np.asarray(b_ub, dtype=float) if b_ub else None,
        "a_eq": np.vstack(a_eq) if a_eq else None,
        "b_eq": np.asarray(b_eq, dtype=float) if b_eq else None,
        "bounds": [(0.0, 1.0)] * n_variables,
        "n_constraints": len(a_ub) + len(a_eq),
    }


def _ordered_z_pairs_by_minus(
    z_values: Sequence[Tuple[Any, ...]],
    j: int,
) -> Iterable[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]:
    """
    Yield ordered supported stratum pairs that share ``Z`` except index ``j``.

    Both ``(z_a,z_b)`` and ``(z_b,z_a)`` are emitted.  This lets the LP express
    ``|a-b| <= tolerance`` using the standard ``A_ub @ theta <= b_ub`` form.
    """
    groups: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
    for z in z_values:
        z_minus = tuple(value for index, value in enumerate(z) if index != j)
        groups.setdefault(z_minus, []).append(z)

    for group in groups.values():
        ordered_group = naturalness_statistics.safe_sorted(group)
        if len(ordered_group) < 2:
            continue
        yield from permutations(ordered_group, 2)


def _solve_query_stratum(
    quantities: TargetObservedBinaryQuantities,
    problem: Dict[str, Any],
    z: Tuple[Any, ...],
    solver_method: str,
    include_intervention_diagnostics: bool = True,
) -> Dict[str, Any]:
    """
    Solve lower/upper CATE and intervention objectives for one stratum.

    For query stratum ``z0`` the CATE objective is ``Delta(z0) = mu_1(z0) - mu_0(z0)``, with
    ``mu_star(z0) = sum_x theta_{star,x,z0} P(X=x | Z=z0; idle)``. Minimizing ``Delta`` gives
    the lower bound and minimizing ``-Delta`` the upper bound after sign reversal; the
    optional ``mu0`` and ``mu1`` diagnostic intervals are solved the same way.

    Args:
        quantities: Target probability table defining objective weights.
        problem: Common LP matrices and variable index.
        z: Target covariate stratum being queried.
        solver_method: SciPy linear-programming method.
        include_intervention_diagnostics: Whether to run the four additional
            ``mu0``/``mu1`` endpoint objectives.

    Returns:
        One query row with CATE endpoints and optional intervention-specific
        diagnostics. Skipped diagnostic endpoints are NaN with
        ``not_requested`` statuses.
    """
    row: Dict[str, Any] = {
        "z_value": z,
        "z_value_str": str(z),
    }
    for covariate, value in zip(quantities.covariates, z):
        row[covariate] = value

    cate_objective = _objective_vector(quantities, problem, z, objective="cate")
    row.update(_solve_objective_interval(problem, cate_objective, solver_method))
    row["status"] = (
        "optimal"
        if row["lower_status"] == "optimal" and row["upper_status"] == "optimal"
        else "nonoptimal"
    )

    for star in [0, 1]:
        if include_intervention_diagnostics:
            objective = _objective_vector(
                quantities,
                problem,
                z,
                objective=f"mu{star}",
            )
            interval = _solve_objective_interval(problem, objective, solver_method)
            row[f"mu{star}_lower"] = interval["lower"]
            row[f"mu{star}_upper"] = interval["upper"]
            row[f"mu{star}_lower_status"] = interval["lower_status"]
            row[f"mu{star}_upper_status"] = interval["upper_status"]
        else:
            row[f"mu{star}_lower"] = np.nan
            row[f"mu{star}_upper"] = np.nan
            row[f"mu{star}_lower_status"] = "not_requested"
            row[f"mu{star}_upper_status"] = "not_requested"

    if row["status"] == "optimal" and row["lower"] > row["upper"] + 1e-8:
        row["status"] = "invalid_interval"
    return row


def _solve_weighted_query(
    quantities: TargetObservedBinaryQuantities,
    problem: Dict[str, Any],
    query_value: Tuple[Any, ...],
    z_weights: Mapping[Tuple[Any, ...], float],
    query_covariates: Sequence[str],
    solver_method: str,
    include_intervention_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Solve one weighted subgroup-CATE query over the common LP.

    Args:
        quantities: Target probability table defining objective weights.
        problem: Common LP matrices and variable index.
        query_value: Reported subgroup value.
        z_weights: Target weights over supported full covariate strata.
        query_covariates: Names corresponding to ``query_value``.
        solver_method: SciPy linear-programming method.
        include_intervention_diagnostics: Whether to run the four additional
            weighted ``mu0``/``mu1`` endpoint objectives.

    Returns:
        One weighted-query row with CATE endpoints and optional intervention
        diagnostics. Skipped diagnostic endpoints are NaN with
        ``not_requested`` statuses.
    """

    row: Dict[str, Any] = {
        "query_value": query_value,
        "query_value_str": str(query_value),
        "n_objective_strata": int(len(z_weights)),
        "objective_weight_sum": float(sum(z_weights.values())),
    }
    for covariate, value in zip(query_covariates, query_value):
        row[covariate] = value

    cate_objective = _weighted_objective_vector(
        quantities,
        problem,
        z_weights,
        objective="cate",
    )
    row.update(_solve_objective_interval(problem, cate_objective, solver_method))
    row["status"] = (
        "optimal"
        if row["lower_status"] == "optimal" and row["upper_status"] == "optimal"
        else "nonoptimal"
    )

    for star in [0, 1]:
        if include_intervention_diagnostics:
            objective = _weighted_objective_vector(
                quantities,
                problem,
                z_weights,
                objective=f"mu{star}",
            )
            interval = _solve_objective_interval(problem, objective, solver_method)
            row[f"mu{star}_lower"] = interval["lower"]
            row[f"mu{star}_upper"] = interval["upper"]
            row[f"mu{star}_lower_status"] = interval["lower_status"]
            row[f"mu{star}_upper_status"] = interval["upper_status"]
        else:
            row[f"mu{star}_lower"] = np.nan
            row[f"mu{star}_upper"] = np.nan
            row[f"mu{star}_lower_status"] = "not_requested"
            row[f"mu{star}_upper_status"] = "not_requested"

    if row["status"] == "optimal" and row["lower"] > row["upper"] + 1e-8:
        row["status"] = "invalid_interval"
    return row


def _objective_vector(
    quantities: TargetObservedBinaryQuantities,
    problem: Dict[str, Any],
    z: Tuple[Any, ...],
    objective: str,
) -> np.ndarray:
    """
    Return a linear objective vector for CATE or one intervention mean.

    Objective rows encode:

        mu_star(z) = sum_x theta_{star,x,z} P(X=x | Z=z; idle)

    and:

        CATE(z) = mu_1(z) - mu_0(z).

    SciPy solves minimization problems, so callers negate this vector when an
    upper bound is needed.
    """
    vector = np.zeros(problem["n_variables"])
    var_index = problem["var_index"]

    stars: List[Tuple[int, float]]
    if objective == "cate":
        stars = [(1, 1.0), (0, -1.0)]
    elif objective == "mu0":
        stars = [(0, 1.0)]
    elif objective == "mu1":
        stars = [(1, 1.0)]
    else:
        raise ValueError(f"Unknown objective {objective!r}")

    for star, sign in stars:
        for x in [0, 1]:
            vector[var_index[(star, x, z)]] += sign * quantities.p_x(x, z)
    return vector


def _weighted_objective_vector(
    quantities: TargetObservedBinaryQuantities,
    problem: Dict[str, Any],
    z_weights: Mapping[Tuple[Any, ...], float],
    objective: str,
) -> np.ndarray:
    """Return a weighted sum of existing stratum-level objective vectors."""

    vector = np.zeros(problem["n_variables"])
    for z, weight in z_weights.items():
        vector += float(weight) * _objective_vector(
            quantities,
            problem,
            z,
            objective=objective,
        )
    return vector


def _solve_linear_objective(
    problem: Dict[str, Any],
    objective: np.ndarray,
    solver_method: str,
) -> Any:
    """Run SciPy ``linprog`` for one objective and common LP constraints."""
    method = solver_method if solver_method else "highs"
    return linprog(
        c=objective,
        A_ub=problem["a_ub"],
        b_ub=problem["b_ub"],
        A_eq=problem["a_eq"],
        b_eq=problem["b_eq"],
        bounds=problem["bounds"],
        method=method,
    )


def _solve_objective_interval(
    problem: Dict[str, Any],
    objective: np.ndarray,
    solver_method: str,
) -> Dict[str, Any]:
    """Solve lower and upper endpoints for one shared linear objective."""
    lower_result = _solve_linear_objective(problem, objective, solver_method)
    upper_result = _solve_linear_objective(problem, -objective, solver_method)
    return {
        "lower": _result_value(lower_result, sign=1.0),
        "upper": _result_value(upper_result, sign=-1.0),
        "lower_status": _result_status(lower_result),
        "upper_status": _result_status(upper_result),
    }


def _result_status(result: Any) -> str:
    """Convert a SciPy optimization result into a compact status string."""
    if bool(getattr(result, "success", False)):
        return "optimal"
    status_code = getattr(result, "status", None)
    if status_code == 2:
        return "infeasible"
    if status_code == 3:
        return "unbounded"
    return "failed"


def _result_value(result: Any, sign: float) -> float:
    """Return the objective value if the solve was optimal, else NaN."""
    if _result_status(result) != "optimal":
        return float("nan")
    return float(sign * result.fun)


def _empty_details_frame() -> pd.DataFrame:
    """Return an empty LP detail table with stable columns."""
    return pd.DataFrame(columns=[
        "z_value",
        "z_value_str",
        "lower",
        "upper",
        "status",
        "lower_status",
        "upper_status",
        "mu0_lower",
        "mu0_upper",
        "mu0_lower_status",
        "mu0_upper_status",
        "mu1_lower",
        "mu1_upper",
        "mu1_lower_status",
        "mu1_upper_status",
        "n_choices",
        "n_strata",
        "n_variables",
        "n_constraints",
        "solver_method",
        "intervention_diagnostics_computed",
        "probability_backend",
    ])


def _empty_weighted_details_frame(query_covariates: Sequence[str]) -> pd.DataFrame:
    """Return an empty weighted-query LP table with stable columns."""

    return pd.DataFrame(columns=[
        "query_value",
        "query_value_str",
        *query_covariates,
        "lower",
        "upper",
        "status",
        "lower_status",
        "upper_status",
        "mu0_lower",
        "mu0_upper",
        "mu0_lower_status",
        "mu0_upper_status",
        "mu1_lower",
        "mu1_upper",
        "mu1_lower_status",
        "mu1_upper_status",
        "n_objective_strata",
        "objective_weight_sum",
        "n_choices",
        "n_strata",
        "n_variables",
        "n_constraints",
        "solver_method",
        "intervention_diagnostics_computed",
    ])


__all__ = [
    "TargetObservedBinaryQuantities",
    "estimate_target_observed_binary_quantities",
    "solve_candidate_state_bounds_from_observed_binary_quantities",
    "solve_target_bounds_from_observed_binary_quantities",
    "solve_target_weighted_cate_bounds_from_observed_binary_quantities",
]
