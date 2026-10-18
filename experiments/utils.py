"""Shared data, naturalness, calibration, and output helpers for experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causal_grounding.discretize import discretize_polideo
from causal_grounding.naturalness_grounder import NaturalnessGrounder, NaturalnessGrid
from causal_grounding.train_target_split import (
    get_available_sites,
)


SUPPORTED_STAGE1_BACKENDS = (
    "xgboost",
    "hierarchical_dirichlet",
    "tabicl_v2",
)
HIERARCHICAL_DIRICHLET_LP_BACKENDS = (
    "hierarchical_dirichlet",
    "hierarchical_dirichlet_draws",
    "hierarchical_dirichlet_posterior",
    "dirichlet_posterior",
)
SUPPORTED_LP_PROBABILITY_BACKENDS = (
    "stage1",
    "empirical",
    "xgboost",
    *HIERARCHICAL_DIRICHLET_LP_BACKENDS,
    "tabicl_v2",
)
SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS = (
    "lp",
    *SUPPORTED_LP_PROBABILITY_BACKENDS,
)
MAX_BINARY_CATE_ENDPOINT_RESIDUAL = 2.0
DEFAULT_AGE_BINARY_THRESHOLD = 21.0
DEFAULT_GENDER_BINARY_THRESHOLD = 0.5
DEFAULT_POLIDEO_BINARY_THRESHOLD = 3.0
DEFAULT_LP_POSTERIOR_N_JOBS = 8
SUPPORTED_MANYLABS_STUDIES = (
    "gainloss",
    "reciprocity",
    "allowedforbidden",
)
USEFUL_COVARIATES = (
    "resp_age_bin",
    "resp_gender_bin",
    "resp_polideo_bin",
)


def find_osrct_file(config: Any) -> Path:
    """
    Find an OSRCT file for a study, pattern, beta, and seed configuration.

    Args:
        config: Object exposing ``data_root``, ``study``, ``pattern``, ``beta``,
            and ``seed`` attributes.

    Returns:
        Path to the first matching Parquet, CSV, or pickle file.

    Raises:
        FileNotFoundError: If none of the supported naming conventions exist.
    """
    data_root = Path(config.data_root)
    study_dir = data_root / config.study
    beta = config.beta
    stems = [
        f"{config.pattern}_beta{beta}_seed{config.seed}",
        f"{config.pattern}_beta{beta:g}_seed{config.seed}",
        f"{config.pattern}_beta{beta:.1f}_seed{config.seed}",
        f"{config.pattern}_beta{beta:.2f}_seed{config.seed}",
        f"{config.study}_{config.pattern}_beta_{beta}",
    ]
    candidates = [
        study_dir / f"{stem}{suffix}"
        for stem in stems
        for suffix in (".parquet", ".csv", ".pkl")
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "OSRCT file not found. Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def add_naturalness_columns(
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    age_bins: int,
    outcome: str = "dv",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Add binary outcome and discrete covariates used by naturalness experiments.

    Args:
        osrct_data: Observational OSRCT rows.
        rct_data: Randomized rows from the same study.
        age_bins: Number of quantile-based age categories.
        outcome: Binary source-outcome column.

    Returns:
        Prepared OSRCT rows, prepared RCT rows, and transformation metadata.

    Raises:
        ValueError: If the combined outcome support is not binary.
    """
    osrct = osrct_data.copy()
    rct = rct_data.copy()
    outcome_values = sorted(
        set(osrct[outcome].dropna().unique().tolist())
        | set(rct[outcome].dropna().unique().tolist())
    )
    if len(outcome_values) != 2:
        raise ValueError(
            f"Naturalness experiments require a binary source outcome; "
            f"got {outcome_values}"
        )

    positive_value = outcome_values[-1]
    combined_age = pd.concat([osrct["resp_age"], rct["resp_age"]], ignore_index=True)
    _, age_edges = pd.qcut(
        combined_age,
        q=age_bins,
        retbins=True,
        duplicates="drop",
    )
    age_edges[0] = -np.inf
    age_edges[-1] = np.inf
    age_labels = list(range(len(age_edges) - 1))

    for data in [osrct, rct]:
        data["dv_binary"] = (data[outcome] == positive_value).astype(int)
        data["dv_positive_value"] = positive_value
        data["resp_age_cat"] = pd.cut(
            data["resp_age"],
            bins=age_edges,
            labels=age_labels,
            include_lowest=True,
        ).astype(int)
        data["resp_gender_bin"] = data["resp_gender"].astype(int)
        data["resp_polideo_cat"] = discretize_polideo(
            data["resp_polideo"]
        ).astype(int)

    metadata = {
        "source_outcome": outcome,
        "binary_outcome": "dv_binary",
        "positive_value": float(positive_value),
        "age_edges": age_edges.tolist(),
        "age_bins": len(age_labels),
    }
    return osrct, rct, metadata


def add_binary_covariates(
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    age_threshold: float,
    polideo_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Add threshold-based binary age and political-ideology covariates.

    Args:
        osrct_data: Prepared OSRCT rows containing ``resp_age`` and
            ``resp_polideo``.
        rct_data: Prepared RCT rows containing the same source columns.
        age_threshold: Age value assigned to the lower bin via ``<=``.
        polideo_threshold: Ideology value assigned to the lower bin via ``<=``.

    Returns:
        Copied input frames with binary columns and threshold metadata.
    """
    osrct = osrct_data.copy()
    rct = rct_data.copy()
    for data in [osrct, rct]:
        data["resp_age_bin"] = (
            pd.to_numeric(data["resp_age"], errors="coerce") > age_threshold
        ).astype(int)
        data["resp_polideo_bin"] = (
            pd.to_numeric(data["resp_polideo"], errors="coerce") > polideo_threshold
        ).astype(int)

    combined = pd.concat(
        [
            osrct[["resp_age_bin", "resp_polideo_bin"]],
            rct[["resp_age_bin", "resp_polideo_bin"]],
        ],
        ignore_index=True,
    )
    metadata = {
        "age_binary_threshold": float(age_threshold),
        "polideo_binary_threshold": float(polideo_threshold),
        "age_binary_rule": f"resp_age_bin = 1[resp_age > {age_threshold:g}]",
        "polideo_binary_rule": (
            f"resp_polideo_bin = 1[resp_polideo > {polideo_threshold:g}]"
        ),
        "age_binary_share_one": float(combined["resp_age_bin"].mean()),
        "polideo_binary_share_one": float(combined["resp_polideo_bin"].mean()),
    }
    return osrct, rct, metadata


def expand_target_sites(
    idle_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    requested_sites: Sequence[str],
    max_target_sites: Optional[int],
) -> List[str]:
    """
    Expand target-site requests against sites available in both regimes.

    Args:
        idle_data: Observational rows containing a ``site`` column.
        rct_data: Randomized rows containing a ``site`` column.
        requested_sites: Requested sites, optionally containing ``all``.
        max_target_sites: Optional cap applied after expansion.

    Returns:
        Sorted eligible target-site labels.

    Raises:
        ValueError: If a requested site is unavailable or no site is selected.
    """
    idle_sites = set(get_available_sites(idle_data, "site"))
    rct_sites = set(get_available_sites(rct_data, "site"))
    eligible = sorted(idle_sites & rct_sites)

    if any(str(site).lower() == "all" for site in requested_sites):
        sites = eligible
    else:
        missing = sorted(set(str(site) for site in requested_sites) - set(eligible))
        if missing:
            raise ValueError(
                "Requested target sites not available in both idle and RCT "
                f"data: {missing}. Eligible sites include: {eligible[:20]}"
            )
        sites = sorted(str(site) for site in requested_sites)

    if max_target_sites is not None:
        sites = sites[:max_target_sites]
    if not sites:
        raise ValueError("No eligible target sites were selected")
    return sites


def build_configured_naturalness_grid(config: Any) -> NaturalnessGrid:
    """
    Build the naturalness grid described by an experiment configuration.

    ``naturalness_extra_candidates`` (optional, default empty) names
    covariates that are tested as candidates without joining the reporting
    covariates. When any are given, the grid covariates are the naturalness
    covariates followed by the extras, and the naturalness covariates become
    the grid's ``conditioning_covariates``: each extra candidate is evaluated
    in the state ``naturalness_covariates + [extra]`` and never enters another
    candidate's ``Z_-j``. Without extras the grid is the legacy single-state
    grid. ``naturalness_candidate_covariates`` optionally restricts which grid
    covariates are candidates; ``None`` or absence means all of them.

    Args:
        config: Configuration exposing naturalness-grid fields.

    Returns:
        Validated naturalness grid.
    """
    covariates = list(getattr(config, "naturalness_covariates", config.covariates))
    extras = list(getattr(config, "naturalness_extra_candidates", ()) or ())
    grid_covariates = covariates + extras
    candidates = getattr(config, "naturalness_candidate_covariates", None)
    return NaturalnessGrid(
        epsilons=list(config.epsilons),
        taus=list(config.taus),
        covariates=grid_covariates,
        candidate_covariates=list(candidates or grid_covariates),
        min_cell_count=config.min_cell_count,
        max_z_minus_samples=getattr(config, "max_z_minus_samples", None),
        conditioning_covariates=list(covariates) if extras else None,
    )


def build_configured_naturalness_grounder(
    config: Any,
    *,
    stage1_backend: Optional[str] = None,
    fit_gp: Optional[bool] = None,
) -> NaturalnessGrounder:
    """
    Build an unfitted grounder using full configurable predictor settings.

    Args:
        config: Configuration exposing naturalness-grid and predictor fields.
        stage1_backend: Optional backend override.
        fit_gp: Optional GP-fitting override.

    Returns:
        Configured, unfitted naturalness grounder.
    """
    backend = stage1_backend or config.stage1_backend
    grid = build_configured_naturalness_grid(config)
    return NaturalnessGrounder(
        grid=grid,
        alpha=config.alpha,
        min_overlap=config.min_overlap,
        random_seed=config.seed,
        fit_gp=config.fit_gp if fit_gp is None else fit_gp,
        stage1_backend=backend,
        stage1_kwargs=build_stage1_predictor_kwargs(backend, config),
        gp_feature_mode=getattr(config, "gp_feature_mode", "thresholds"),
    )


def fit_configured_naturalness_grounder(
    training_data: Dict[str, pd.DataFrame],
    config: Any,
    *,
    stage1_backend: Optional[str] = None,
    fit_gp: Optional[bool] = None,
) -> NaturalnessGrounder:
    """
    Fit a grounder using full configurable Stage-1 predictor settings.

    Args:
        training_data: Site-keyed rows containing idle and randomized regimes.
        config: Configuration exposing naturalness-grid and predictor fields.
        stage1_backend: Optional backend override.
        fit_gp: Optional GP-fitting override.

    Returns:
        Fitted naturalness grounder.
    """
    grounder = build_configured_naturalness_grounder(
        config,
        stage1_backend=stage1_backend,
        fit_gp=fit_gp,
    )
    grounder.fit(
        training_data,
        treatment="iv",
        outcome="dv_binary",
        regime_col="F",
    )
    return grounder


def finite_sample_conformal_quantile(
    scores: Sequence[float],
    alpha: float,
    *,
    upper_bound: float = MAX_BINARY_CATE_ENDPOINT_RESIDUAL,
) -> Tuple[float, int]:
    """
    Compute split-conformal upper quantile with bounded infinity sentinel.

    Args:
        scores: Independent nonconformity scores.
        alpha: Desired miscoverage level.
        upper_bound: Known upper bound for every score.

    Returns:
        Conformal correction and one-based quantile rank.
    """
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("At least one conformal calibration score is required")
    if not np.isfinite(values).all():
        raise ValueError("Conformal calibration scores must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    values = np.clip(values, 0.0, upper_bound)
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    if rank > len(values):
        return float(upper_bound), rank
    return float(np.sort(values)[rank - 1]), rank


def add_full_range_fallback(
    lp_bounds: pd.DataFrame,
    target_idle: pd.DataFrame,
    config: Any,
    *,
    empty_status: str,
) -> pd.DataFrame:
    """
    Make binary-CATE interval scoring total while retaining LP failures.

    Args:
        lp_bounds: Base LP rows, possibly empty or nonoptimal.
        target_idle: Target idle rows defining observed covariate strata.
        config: Configuration exposing ordered LP covariates.
        empty_status: Status assigned when no base LP rows exist.

    Returns:
        One row per observed target stratum, with total scoring endpoints and
        explicit fallback diagnostics. Strata omitted by the LP probability
        estimator are retained with ``lp_stratum_not_returned`` status.
    """
    result = lp_bounds.copy()
    covariates = list(config.covariates)
    strata = (
        target_idle[covariates]
        .dropna()
        .drop_duplicates()
        .sort_values(covariates)
        .reset_index(drop=True)
    )
    missing_status = empty_status
    if not result.empty:
        missing_columns = [
            covariate for covariate in covariates if covariate not in result
        ]
        if missing_columns:
            raise ValueError(
                "LP bounds are missing target stratum columns: "
                f"{missing_columns}"
            )
        returned_strata = result[covariates].drop_duplicates()
        strata = (
            strata.merge(
                returned_strata,
                on=covariates,
                how="left",
                indicator=True,
                validate="one_to_one",
            )
            .loc[lambda frame: frame["_merge"].eq("left_only"), covariates]
            .reset_index(drop=True)
        )
        missing_status = "lp_stratum_not_returned"

    if not strata.empty:
        rows: List[Dict[str, Any]] = []
        for _, stratum in strata.iterrows():
            z_value = tuple(stratum[col] for col in covariates)
            row: Dict[str, Any] = {
                "z_value": z_value,
                "z_value_str": str(z_value),
                "lower": np.nan,
                "upper": np.nan,
                "status": missing_status,
            }
            for covariate, value in zip(covariates, z_value):
                row[covariate] = value
            rows.append(row)
        result = pd.concat(
            [result, pd.DataFrame(rows)],
            ignore_index=True,
            sort=False,
        )

    base_available = (
        result["status"].eq("optimal")
        & pd.to_numeric(result["lower"], errors="coerce").notna()
        & pd.to_numeric(result["upper"], errors="coerce").notna()
    )
    result["base_interval_available"] = base_available
    result["used_full_range_fallback"] = ~base_available
    result["score_lower"] = np.where(base_available, result["lower"], -1.0)
    result["score_upper"] = np.where(base_available, result["upper"], 1.0)
    result["interval_procedure_status"] = np.where(
        base_available,
        "base_optimal",
        "full_range_fallback",
    )
    return result


def conformalize_lp_bounds(
    lp_bounds: pd.DataFrame,
    correction: float,
) -> pd.DataFrame:
    """
    Expand final binary-CATE endpoints by a conformal correction.

    Args:
        lp_bounds: Base per-stratum LP output.
        correction: Non-negative endpoint correction on CATE scale.

    Returns:
        Bounds with unchanged base columns and conformal endpoint columns.
    """
    if correction < 0.0 or not np.isfinite(correction):
        raise ValueError("Endpoint conformal correction must be finite and non-negative")

    result = lp_bounds.copy()
    if result.empty:
        return result
    result["base_lower"] = result["lower"]
    result["base_upper"] = result["upper"]
    result["base_width"] = result["upper"] - result["lower"]
    result["base_status"] = result["status"]
    if "covers_rct_cate" in result.columns:
        result["base_covers_rct_cate"] = result["covers_rct_cate"]

    optimal = (
        result["status"].eq("optimal")
        & result["lower"].notna()
        & result["upper"].notna()
    )
    result["endpoint_conformal_q"] = float(correction)
    result["conformal_lower"] = np.where(
        optimal,
        np.maximum(-1.0, result["lower"] - correction),
        -1.0,
    )
    result["conformal_upper"] = np.where(
        optimal,
        np.minimum(1.0, result["upper"] + correction),
        1.0,
    )
    result["conformal_width"] = (
        result["conformal_upper"] - result["conformal_lower"]
    )
    result["conformal_status"] = np.where(
        optimal,
        "optimal_expanded",
        "base_nonoptimal_full_range",
    )
    if "rct_cate" in result.columns:
        result["covers_rct_cate_conformal"] = (
            result["rct_cate"].notna()
            & result["conformal_lower"].notna()
            & result["conformal_upper"].notna()
            & (result["conformal_lower"] <= result["rct_cate"])
            & (result["rct_cate"] <= result["conformal_upper"])
        )
    if "rct_cate_for_score" in result.columns:
        result["covers_rct_cate_score_label_conformal"] = (
            result["rct_cate_for_score"].notna()
            & result["conformal_lower"].notna()
            & result["conformal_upper"].notna()
            & (result["conformal_lower"] <= result["rct_cate_for_score"])
            & (result["rct_cate_for_score"] <= result["conformal_upper"])
        )
    return result


def resolve_lp_probability_backend(config: Any) -> str:
    """
    Resolve a target-LP probability backend from experiment configuration.

    Args:
        config: Object exposing ``lp_probability_backend`` and
            ``stage1_backend``.

    Returns:
        Concrete supported backend name.

    Raises:
        ValueError: If the resolved backend is unsupported.
    """
    backend = config.lp_probability_backend.lower()
    if backend == "stage1":
        backend = config.stage1_backend.lower()
    if backend not in SUPPORTED_LP_PROBABILITY_BACKENDS:
        raise ValueError(
            "lp_probability_backend must be one of 'stage1', 'empirical', "
            "'xgboost', 'hierarchical_dirichlet', "
            "'hierarchical_dirichlet_draws', or 'tabicl_v2', got "
            f"{config.lp_probability_backend!r}"
        )
    return backend


def resolve_target_trigger_probability_backend(config: Any) -> str:
    """Resolve target-trigger probabilities independently from target LP ones.

    ``"lp"`` preserves experiments whose target trigger screen historically
    reused target-LP nuisance probabilities. ``"stage1"`` instead instantiates
    the configured Stage-1 model family on target idle rows.

    Args:
        config: Object exposing ``target_trigger_probability_backend``,
            ``lp_probability_backend``, and ``stage1_backend``.

    Returns:
        Concrete supported target probability backend name.

    Raises:
        ValueError: If the configured target-trigger backend is unsupported.
    """

    backend = str(config.target_trigger_probability_backend).lower()
    if backend == "lp":
        return resolve_lp_probability_backend(config)
    if backend == "stage1":
        backend = str(config.stage1_backend).lower()
    concrete_backends = set(SUPPORTED_LP_PROBABILITY_BACKENDS) - {"stage1"}
    if backend not in concrete_backends:
        raise ValueError(
            "target_trigger_probability_backend must be 'lp', 'stage1', or "
            "a concrete supported target probability backend, got "
            f"{config.target_trigger_probability_backend!r}"
        )
    return backend


def build_lp_probability_model_kwargs(
    config: Any,
    backend: str,
) -> Optional[Dict[str, Any]]:
    """
    Build backend-specific target-LP probability model options.

    Args:
        config: Object exposing ``seed``. A true ``tabicl_share_checkpoint``
            makes TabICLv2 fits share one loaded checkpoint.
        backend: Resolved probability backend.

    Returns:
        Model keyword arguments, or None for empirical estimation.
    """
    if backend == "xgboost":
        return {
            "use_grid_search": bool(
                getattr(config, "lp_xgboost_use_grid_search", False)
            ),
            "n_estimators": int(
                getattr(config, "lp_xgboost_n_estimators", 200)
            ),
            "max_depth": int(getattr(config, "lp_xgboost_max_depth", 5)),
            "learning_rate": float(
                getattr(config, "lp_xgboost_learning_rate", 0.10)
            ),
            "random_state": config.seed,
            "n_jobs": getattr(config, "lp_xgboost_n_jobs", None),
        }
    if backend == "tabicl_v2":
        device = getattr(config, "tabicl_device", None)
        kwargs = {
            "random_state": config.seed,
            "kv_cache": "repr",
            "device": None if device == "auto" else device,
        }
        # Opt-in only, so every existing configuration keeps identical kwargs.
        if getattr(config, "tabicl_share_checkpoint", False):
            kwargs["share_checkpoint"] = True
        return kwargs
    if backend in HIERARCHICAL_DIRICHLET_LP_BACKENDS:
        return {
            "prior_strength": float(
                getattr(config, "hierarchical_dirichlet_prior_strength", 2.0)
            ),
            "base_prior_strength": float(
                getattr(config, "hierarchical_dirichlet_base_prior_strength", 2.0)
            ),
            "global_prior": float(
                getattr(config, "hierarchical_dirichlet_global_prior", 0.5)
            ),
        }
    return None


def build_stage1_predictor_kwargs(
    backend: str,
    config: Any,
) -> Dict[str, Any]:
    """
    Build predictor options for model-based naturalness Stage-1 backends.

    Args:
        backend: Stage-1 backend name.
        config: Object exposing ``seed``. A true ``tabicl_share_checkpoint``
            makes TabICLv2 fits share one loaded checkpoint.

    Returns:
        Nested predictor keyword arguments, or an empty mapping for a backend
        with no configured options.
    """
    backend = backend.lower()
    if backend == "xgboost":
        return {
            "predictor_kwargs": {
                "use_grid_search": bool(
                    getattr(config, "stage1_xgboost_use_grid_search", True)
                ),
                "n_estimators_range": list(
                    getattr(
                        config,
                        "stage1_xgboost_n_estimators_range",
                        (50, 100, 200, 400),
                    )
                ),
                "max_depth_range": list(
                    getattr(
                        config,
                        "stage1_xgboost_max_depth_range",
                        (2, 3, 5,),
                    )
                ),
                "learning_rate_range": list(
                    getattr(
                        config,
                        "stage1_xgboost_learning_rate_range",
                        (0.05, 0.10, 0.20),
                    )
                ),
                "cv_folds": int(
                    getattr(config, "stage1_xgboost_cv_folds", 5)
                ),
                "random_state": config.seed,
                "n_jobs": getattr(config, "stage1_xgboost_n_jobs", None),
            }
        }
    if backend == "tabicl_v2":
        device = getattr(config, "tabicl_device", None)
        predictor_kwargs = {
            "random_state": config.seed,
            "device": None if device == "auto" else device,
        }
        # Opt-in only, so every existing configuration keeps identical kwargs.
        if getattr(config, "tabicl_share_checkpoint", False):
            predictor_kwargs["share_checkpoint"] = True
        return {"predictor_kwargs": predictor_kwargs}
    if backend == "hierarchical_dirichlet":
        return {
            "predictor_kwargs": {
                "prior_strength": float(
                    getattr(config, "hierarchical_dirichlet_prior_strength", 2.0)
                ),
                "base_prior_strength": float(
                    getattr(
                        config,
                        "hierarchical_dirichlet_base_prior_strength",
                        2.0,
                    )
                ),
                "global_prior": float(
                    getattr(config, "hierarchical_dirichlet_global_prior", 0.5)
                ),
            }
        }
    return {}


def uses_hierarchical_dirichlet_draws(backend: str) -> bool:
    """Return whether an LP backend requests a hierarchical-Dirichlet draw.

    Args:
        backend: Resolved target-LP probability backend.

    Returns:
        ``True`` for a posterior-draw hierarchical-Dirichlet alias.
    """
    return (
        backend in HIERARCHICAL_DIRICHLET_LP_BACKENDS
        and backend != "hierarchical_dirichlet"
    )


def json_default(value: Any) -> Any:
    """
    Convert numpy and pandas scalar values into JSON-compatible values.

    Args:
        value: Object passed to ``json.dump`` or ``json.dumps``.

    Returns:
        JSON-compatible representation.
    """
    if isinstance(value, tuple):
        return [json_default(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _manifest_safe_value(value: Any) -> Any:
    """
    Convert nested experiment values into simple YAML-serializable values.

    Args:
        value: Arbitrary experiment metadata or configuration value.

    Returns:
        Value composed only of mappings, lists, strings, numbers, booleans, and
        nulls.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return _manifest_safe_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_manifest_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_manifest_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json_default(value)
    return json_default(value)


def _yaml_scalar(value: Any) -> str:
    """
    Render one scalar value in conservative YAML syntax.

    Args:
        value: Scalar metadata value.

    Returns:
        YAML scalar text.
    """

    value = _manifest_safe_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return json.dumps(str(value))
        return json.dumps(value)
    return json.dumps(str(value))


def _append_yaml_value(lines: list[str], value: Any, indent: int) -> None:
    """
    Append a nested value to a YAML document under an existing key.

    Args:
        lines: Mutable output lines.
        value: Nested metadata value.
        indent: Number of spaces to prepend.
    """

    prefix = " " * indent
    value = _manifest_safe_value(value)
    if isinstance(value, Mapping):
        if not value:
            lines.append(f"{prefix}{{}}")
            return
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, str) and "\n" in item:
                lines.append(f"{prefix}{key_text}: |")
                for line in item.splitlines():
                    lines.append(f"{prefix}  {line}")
            elif isinstance(item, Mapping):
                lines.append(f"{prefix}{key_text}:")
                _append_yaml_value(lines, item, indent + 2)
            elif isinstance(item, list):
                if item:
                    lines.append(f"{prefix}{key_text}:")
                    _append_yaml_value(lines, item, indent + 2)
                else:
                    lines.append(f"{prefix}{key_text}: []")
            else:
                lines.append(f"{prefix}{key_text}: {_yaml_scalar(item)}")
        return
    if isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
            return
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}-")
                _append_yaml_value(lines, item, indent + 2)
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return
    lines.append(f"{prefix}{_yaml_scalar(value)}")


def _resolve_manifest_path(manifest_path: str | Path | None) -> Path | None:
    """
    Resolve a user-supplied cluster manifest path against common repo roots.

    Args:
        manifest_path: Optional absolute or relative manifest path.

    Returns:
        Existing resolved path when found, otherwise the best-effort path.
    """

    if manifest_path is None or str(manifest_path) == "":
        return None
    path = Path(manifest_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_run_manifest(
    output_dir: str | Path,
    *,
    experiment_name: str,
    script_path: str,
    config: Mapping[str, Any] | Any,
    argv: Sequence[str] | None = None,
    cluster_manifest_path: str | Path | None = None,
    filename: str = "run_manifest.yml",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Materialize a run-level YAML manifest beside experiment outputs.

    Args:
        output_dir: Directory where the manifest should be written.
        experiment_name: Stable experiment identifier.
        script_path: Python entrypoint path used by the job.
        config: Resolved experiment configuration.
        argv: Command-line arguments received by the entrypoint.
        cluster_manifest_path: Optional Kubernetes YAML template to embed.
        filename: Output manifest filename.
        extra: Optional experiment-specific metadata.

    Returns:
        Path to the written manifest.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = _resolve_manifest_path(cluster_manifest_path)
    source_info: dict[str, Any] = {
        "requested_path": str(cluster_manifest_path)
        if cluster_manifest_path is not None
        else None,
        "resolved_path": str(source_path) if source_path is not None else None,
        "exists": bool(source_path is not None and source_path.exists()),
    }
    if source_path is not None and source_path.exists():
        source_info["contents"] = source_path.read_text(encoding="utf-8")

    payload: dict[str, Any] = {
        "experiment": experiment_name,
        "script": script_path,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": list(argv or ()),
        },
        "config": _manifest_safe_value(config),
        "cluster_manifest": source_info,
    }
    if extra:
        payload["extra"] = _manifest_safe_value(extra)

    lines: list[str] = []
    _append_yaml_value(lines, payload, 0)
    manifest_output_path = output_path / filename
    manifest_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_output_path


def summarize_lp_posterior_bounds(
    samples: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize sampled LP endpoint distributions by target query stratum.

    Args:
        samples: Sample-level LP detail rows.
        covariates: Ordered covariate columns defining query strata.

    Returns:
        Endpoint and width summaries plus posterior optimal-solve frequency.
    """
    if samples.empty or "lp_posterior_sample_id" not in samples.columns:
        return pd.DataFrame()
    group_cols = [
        col for col in ["z_value_str", *covariates] if col in samples.columns
    ]
    if not group_cols:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for group_key, group in samples.groupby(group_cols, dropna=False, sort=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(group_cols, key_tuple))
        if "z_value" in group.columns:
            record["z_value"] = group["z_value"].iloc[0]
        n_total = int(group["lp_posterior_sample_id"].nunique())
        optimal = group[
            group["status"].eq("optimal")
            & group["lower"].notna()
            & group["upper"].notna()
        ].copy()
        n_optimal = int(optimal["lp_posterior_sample_id"].nunique())
        record["lp_posterior_n_samples"] = n_total
        record["lp_posterior_n_optimal"] = n_optimal
        record["lp_posterior_prob_optimal"] = (
            float(n_optimal / n_total) if n_total else float("nan")
        )
        if optimal.empty:
            for column in [
                "lower_mean",
                "lower_q05",
                "lower_q50",
                "lower_q95",
                "upper_mean",
                "upper_q05",
                "upper_q50",
                "upper_q95",
                "width_mean",
                "width_q05",
                "width_q50",
                "width_q95",
                "posterior_envelope_lower",
                "posterior_envelope_upper",
                "posterior_envelope_width",
            ]:
                record[column] = float("nan")
            records.append(record)
            continue

        width = optimal["upper"].astype(float) - optimal["lower"].astype(float)
        for prefix, values in [
            ("lower", optimal["lower"].astype(float)),
            ("upper", optimal["upper"].astype(float)),
            ("width", width),
        ]:
            record[f"{prefix}_mean"] = float(values.mean())
            record[f"{prefix}_q05"] = float(values.quantile(0.05))
            record[f"{prefix}_q50"] = float(values.quantile(0.50))
            record[f"{prefix}_q95"] = float(values.quantile(0.95))
        record["posterior_envelope_lower"] = record["lower_q05"]
        record["posterior_envelope_upper"] = record["upper_q95"]
        record["posterior_envelope_width"] = (
            record["posterior_envelope_upper"] - record["posterior_envelope_lower"]
        )
        records.append(record)
    return pd.DataFrame(records)


def concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate output frames while tolerating an empty accumulator.

    Args:
        frames: Frames to concatenate.

    Returns:
        Concatenated frame, or an empty frame when no inputs exist.
    """
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _latex_escape(value: Any) -> str:
    """Return a conservative LaTeX-safe representation of a scalar value."""

    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _numeric_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric metric column, treating booleans as 0/1 values."""

    if column not in frame.columns:
        raise ValueError(f"Missing required metric column: {column}")
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def format_latex_mean_sd(
    mean: float,
    sd: float,
    *,
    digits: int = 3,
) -> str:
    """
    Format a mean and standard deviation for a LaTeX table cell.

    Args:
        mean: Mean value to display.
        sd: Standard deviation value to display. Missing values are displayed as
            zero, which is the natural single-row group convention.
        digits: Number of decimal places.

    Returns:
        A math-mode string of the form ``$mean \\pm sd$``.
    """

    if not np.isfinite(mean):
        return ""
    display_sd = 0.0 if not np.isfinite(sd) else float(sd)
    return f"${float(mean):.{digits}f} \\pm {display_sd:.{digits}f}$"


def make_metric_summary_latex_table(
    run_summary: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    coverage_col: str = "rct_conformal_point_coverage",
    width_col: str = "conformal_weighted_width",
    digits: int = 3,
    column_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a formatted coverage/width summary table from run-summary rows.

    Args:
        run_summary: Frame containing one row per experiment run or target.
        group_columns: Columns defining rows of the output table.
        coverage_col: Numeric or boolean coverage column to summarize.
        width_col: Numeric interval-width column to summarize.
        digits: Number of decimal places in ``mean \\pm sd`` cells.
        column_labels: Optional display labels for grouping and metric columns.

    Returns:
        A display-ready DataFrame with group labels, ``N``, coverage
        ``mean \\pm sd``, and width ``mean \\pm sd``.

    Raises:
        ValueError: If the requested grouping or metric columns are missing.
    """

    if run_summary.empty:
        return pd.DataFrame()
    missing_groups = [col for col in group_columns if col not in run_summary.columns]
    if missing_groups:
        raise ValueError(f"Missing required grouping columns: {missing_groups}")

    working = run_summary.copy()
    working["_coverage_metric"] = _numeric_metric(working, coverage_col)
    working["_width_metric"] = _numeric_metric(working, width_col)
    working = working.dropna(subset=["_coverage_metric", "_width_metric"])
    if working.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for group_key, group in working.groupby(list(group_columns), dropna=False, sort=True):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = {
            column: _latex_escape(value)
            for column, value in zip(group_columns, key_tuple)
        }
        coverage = group["_coverage_metric"].astype(float)
        width = group["_width_metric"].astype(float)
        record["n"] = int(len(group))
        record["coverage"] = format_latex_mean_sd(
            float(coverage.mean()),
            float(coverage.std(ddof=1)) if len(coverage) > 1 else 0.0,
            digits=digits,
        )
        record["width"] = format_latex_mean_sd(
            float(width.mean()),
            float(width.std(ddof=1)) if len(width) > 1 else 0.0,
            digits=digits,
        )
        records.append(record)

    table = pd.DataFrame(records)
    labels = {
        "study": "Experiment",
        "pattern": "Confounding variable",
        "beta": "Beta",
        "target_site": "Target",
        "stage1_backend": "Backend",
        "lp_probability_backend": "LP backend",
        "n": "N",
        "coverage": "Coverage",
        "width": "Width",
    }
    if column_labels:
        labels.update(column_labels)
    return table.rename(columns={col: labels.get(col, col) for col in table.columns})


def make_multi_metric_summary_latex_table(
    run_summary: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    digits: int = 3,
    column_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a formatted grouped table for several numeric summary metrics.

    Args:
        run_summary: Frame containing one row per plotted stratum or run.
        group_columns: Columns defining rows of the output table.
        metrics: Pairs of ``(display_label, source_column)`` to summarize.
        digits: Number of decimal places in ``mean \\pm sd`` cells.
        column_labels: Optional display labels for grouping columns.

    Returns:
        A display-ready DataFrame with group labels, ``N``, and one
        ``mean \\pm sd`` column per requested metric.

    Raises:
        ValueError: If grouping columns or requested metric columns are missing.
    """

    if run_summary.empty:
        return pd.DataFrame()
    missing_groups = [col for col in group_columns if col not in run_summary.columns]
    if missing_groups:
        raise ValueError(f"Missing required grouping columns: {missing_groups}")
    missing_metrics = [column for _, column in metrics if column not in run_summary.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metric columns: {missing_metrics}")

    working = run_summary.copy()
    metric_work_columns: List[Tuple[str, str]] = []
    for index, (label, column) in enumerate(metrics):
        work_column = f"_metric_{index}"
        working[work_column] = _numeric_metric(working, column)
        metric_work_columns.append((label, work_column))

    records: List[Dict[str, Any]] = []
    for group_key, group in working.groupby(list(group_columns), dropna=False, sort=True):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = {
            column: _latex_escape(value)
            for column, value in zip(group_columns, key_tuple)
        }
        record["n"] = int(len(group))
        for label, work_column in metric_work_columns:
            values = group[work_column].dropna().astype(float)
            record[label] = format_latex_mean_sd(
                float(values.mean()) if len(values) else float("nan"),
                float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                digits=digits,
            )
        records.append(record)

    table = pd.DataFrame(records)
    labels = {
        "study": "Experiment",
        "pattern": "Confounding variable",
        "beta": "Beta",
        "target_site": "Target",
        "stage1_backend": "Backend",
        "lp_probability_backend": "LP backend",
        "n": "N",
    }
    if column_labels:
        labels.update(column_labels)
    return table.rename(columns={col: labels.get(col, col) for col in table.columns})


def render_latex_table(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    index: bool = False,
    column_format: str | None = None,
    midrule_before: Sequence[int] = (),
) -> str:
    """
    Render a display-ready DataFrame as a simple booktabs LaTeX table.

    Args:
        table: DataFrame whose cells and column labels are already formatted for
            display. Cell contents are emitted as-is so math-mode summaries such
            as ``$0.90 \\pm 0.10$`` remain intact.
        caption: Optional LaTeX caption. When provided, the tabular block is
            wrapped in a table environment.
        label: Optional LaTeX label. When provided, the tabular block is wrapped
            in a table environment.
        index: Whether to include the DataFrame index as the first column.
        column_format: Tabular column specification such as ``"llcc"``;
            defaults to one left-aligned column per displayed column.
        midrule_before: Zero-based body row positions preceded by a
            ``\\midrule``, which separates grouped blocks of rows.

    Returns:
        LaTeX source for the table without relying on Pandas Styler or Jinja2.
    """

    display = table.reset_index() if index else table
    if display.empty:
        return ""

    def _latex_cell(value: Any) -> str:
        """Return a raw LaTeX cell value, preserving preformatted math."""

        return "" if pd.isna(value) else str(value)

    lines: list[str] = []
    wrap_table = caption is not None or label is not None
    if wrap_table:
        lines.append(r"\begin{table}")
        if caption is not None:
            lines.append(f"\\caption{{{caption}}}")
        if label is not None:
            lines.append(f"\\label{{{label}}}")

    if column_format is None:
        column_format = "l" * len(display.columns)
    breaks = set(midrule_before)
    lines.extend(
        [
            f"\\begin{{tabular}}{{{column_format}}}",
            r"\toprule",
            " & ".join(str(column) for column in display.columns) + r" \\",
            r"\midrule",
        ]
    )
    for position, row in enumerate(display.itertuples(index=False, name=None)):
        if position in breaks and position > 0:
            lines.append(r"\midrule")
        lines.append(" & ".join(_latex_cell(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    if wrap_table:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_metric_summary_latex_table(
    run_summary: pd.DataFrame,
    output_path: str | Path,
    *,
    group_columns: Sequence[str],
    coverage_col: str = "rct_conformal_point_coverage",
    width_col: str = "conformal_weighted_width",
    digits: int = 3,
    column_labels: Mapping[str, str] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """
    Write a LaTeX coverage/width summary table and return its display frame.

    Args:
        run_summary: Frame containing run-level coverage and width metrics.
        output_path: Destination ``.tex`` path.
        group_columns: Columns defining rows of the output table.
        coverage_col: Numeric or boolean coverage column to summarize.
        width_col: Numeric interval-width column to summarize.
        digits: Number of decimal places in ``mean \\pm sd`` cells.
        column_labels: Optional display labels for output columns.
        caption: Optional LaTeX table caption.
        label: Optional LaTeX table label.

    Returns:
        The display-ready summary table that was written. If no complete metric
        rows exist, an empty file is written and an empty frame is returned.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = make_metric_summary_latex_table(
        run_summary,
        group_columns=group_columns,
        coverage_col=coverage_col,
        width_col=width_col,
        digits=digits,
        column_labels=column_labels,
    )
    if table.empty:
        output.write_text("", encoding="utf-8")
        return table
    output.write_text(
        render_latex_table(
            table,
            index=False,
            caption=caption,
            label=label,
        ),
        encoding="utf-8",
    )
    return table
