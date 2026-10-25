"""Probability-backend resolution and grounder/predictor builders from configs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from causal_grounding.naturalness_grounder import NaturalnessGrounder, NaturalnessGrid
from causal_grounding.probability_backends import probability_backend_names


# Derived from causal_grounding.probability_backends, the single backend registry.
SUPPORTED_STAGE1_BACKENDS = probability_backend_names(stage1=True, model_based=True)
HIERARCHICAL_DIRICHLET_LP_BACKENDS = probability_backend_names(
    target_lp=True, fit_backend="hierarchical_dirichlet"
)
SUPPORTED_LP_PROBABILITY_BACKENDS = (
    "stage1",
    *probability_backend_names(target_lp=True),
)
SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS = (
    "lp",
    *SUPPORTED_LP_PROBABILITY_BACKENDS,
)

DEFAULT_LP_POSTERIOR_N_JOBS = 8


# Name prefixes reserved for ProbabilityModelOptions fields. A config attribute
# with one of these prefixes that is not a declared field is treated as a typo.
PROBABILITY_MODEL_OPTION_PREFIXES = (
    "lp_xgboost_",
    "stage1_xgboost_",
    "tabicl_",
    "hierarchical_dirichlet_",
)


@dataclass(frozen=True)
class ProbabilityModelOptions:
    """Typed contract for the probability-model fields a config may set.

    The kwargs builders read exactly these fields. A config may omit any of
    them, and the omitted field takes the default declared here. A config
    attribute that carries a reserved prefix but names no field raises a
    ``ValueError`` instead of falling back to a default silently.

    Attributes:
        lp_xgboost_use_grid_search: Grid-search the target-LP XGBoost model.
        lp_xgboost_n_estimators: Target-LP XGBoost tree count.
        lp_xgboost_max_depth: Target-LP XGBoost tree depth.
        lp_xgboost_learning_rate: Target-LP XGBoost learning rate.
        lp_xgboost_n_jobs: Target-LP XGBoost worker count, or None.
        stage1_xgboost_use_grid_search: Grid-search the Stage-1 XGBoost model.
        stage1_xgboost_n_estimators_range: Stage-1 tree-count grid.
        stage1_xgboost_max_depth_range: Stage-1 tree-depth grid.
        stage1_xgboost_learning_rate_range: Stage-1 learning-rate grid.
        stage1_xgboost_cv_folds: Stage-1 grid-search fold count.
        stage1_xgboost_n_jobs: Stage-1 XGBoost worker count, or None.
        tabicl_device: TabICLv2 device; ``"auto"`` maps to None.
        tabicl_share_checkpoint: Share one loaded TabICLv2 checkpoint.
        hierarchical_dirichlet_prior_strength: Site-level prior strength.
        hierarchical_dirichlet_base_prior_strength: Base-level prior strength.
        hierarchical_dirichlet_global_prior: Global pseudo-count prior.
    """

    lp_xgboost_use_grid_search: bool = False
    lp_xgboost_n_estimators: int = 200
    lp_xgboost_max_depth: int = 5
    lp_xgboost_learning_rate: float = 0.10
    lp_xgboost_n_jobs: Optional[int] = None
    stage1_xgboost_use_grid_search: bool = True
    stage1_xgboost_n_estimators_range: Sequence[int] = (50, 100, 200, 400)
    stage1_xgboost_max_depth_range: Sequence[int] = (2, 3, 5)
    stage1_xgboost_learning_rate_range: Sequence[float] = (0.05, 0.10, 0.20)
    stage1_xgboost_cv_folds: int = 5
    stage1_xgboost_n_jobs: Optional[int] = None
    tabicl_device: Optional[str] = None
    tabicl_share_checkpoint: bool = False
    hierarchical_dirichlet_prior_strength: float = 2.0
    hierarchical_dirichlet_base_prior_strength: float = 2.0
    hierarchical_dirichlet_global_prior: float = 0.5

    @classmethod
    def from_config(cls, config: Any) -> ProbabilityModelOptions:
        """Read the declared option fields from an experiment configuration.

        Args:
            config: Any object. Each declared field it exposes overrides the
                default. Each declared field it lacks keeps the default.

        Returns:
            Options holding the config values and the defaults.

        Raises:
            ValueError: If ``config`` exposes an attribute with a reserved
                prefix that is not a declared field.
        """
        declared = {item.name for item in fields(cls)}
        unknown = sorted(
            name
            for name in dir(config)
            if name.startswith(PROBABILITY_MODEL_OPTION_PREFIXES)
            and name not in declared
        )
        if unknown:
            raise ValueError(
                f"Unknown probability-model option(s) on {type(config).__name__}: "
                f"{unknown}. Declared options: {sorted(declared)}"
            )
        return cls(
            **{name: getattr(config, name) for name in declared if hasattr(config, name)}
        )


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
        config: Object exposing ``seed`` and, optionally, any field of
            :class:`ProbabilityModelOptions`. A true ``tabicl_share_checkpoint``
            makes TabICLv2 fits share one loaded checkpoint.
        backend: Resolved probability backend.

    Returns:
        Model keyword arguments, or None for empirical estimation.

    Raises:
        ValueError: If ``config`` carries an undeclared probability-model
            option, for example a misspelled ``lp_xgboost_*`` field.
    """
    options = ProbabilityModelOptions.from_config(config)
    if backend == "xgboost":
        return {
            "use_grid_search": bool(options.lp_xgboost_use_grid_search),
            "n_estimators": int(options.lp_xgboost_n_estimators),
            "max_depth": int(options.lp_xgboost_max_depth),
            "learning_rate": float(options.lp_xgboost_learning_rate),
            "random_state": config.seed,
            "n_jobs": options.lp_xgboost_n_jobs,
        }
    if backend == "tabicl_v2":
        device = options.tabicl_device
        kwargs = {
            "random_state": config.seed,
            "kv_cache": "repr",
            "device": None if device == "auto" else device,
        }
        # Opt-in only, so every existing configuration keeps identical kwargs.
        if options.tabicl_share_checkpoint:
            kwargs["share_checkpoint"] = True
        return kwargs
    if backend in HIERARCHICAL_DIRICHLET_LP_BACKENDS:
        return {
            "prior_strength": float(options.hierarchical_dirichlet_prior_strength),
            "base_prior_strength": float(
                options.hierarchical_dirichlet_base_prior_strength
            ),
            "global_prior": float(options.hierarchical_dirichlet_global_prior),
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
        config: Object exposing ``seed`` and, optionally, any field of
            :class:`ProbabilityModelOptions`. A true ``tabicl_share_checkpoint``
            makes TabICLv2 fits share one loaded checkpoint.

    Returns:
        Nested predictor keyword arguments, or an empty mapping for a backend
        with no configured options.

    Raises:
        ValueError: If ``config`` carries an undeclared probability-model
            option, for example a misspelled ``stage1_xgboost_*`` field.
    """
    options = ProbabilityModelOptions.from_config(config)
    backend = backend.lower()
    if backend == "xgboost":
        return {
            "predictor_kwargs": {
                "use_grid_search": bool(options.stage1_xgboost_use_grid_search),
                "n_estimators_range": list(options.stage1_xgboost_n_estimators_range),
                "max_depth_range": list(options.stage1_xgboost_max_depth_range),
                "learning_rate_range": list(
                    options.stage1_xgboost_learning_rate_range
                ),
                "cv_folds": int(options.stage1_xgboost_cv_folds),
                "random_state": config.seed,
                "n_jobs": options.stage1_xgboost_n_jobs,
            }
        }
    if backend == "tabicl_v2":
        device = options.tabicl_device
        predictor_kwargs = {
            "random_state": config.seed,
            "device": None if device == "auto" else device,
        }
        # Opt-in only, so every existing configuration keeps identical kwargs.
        if options.tabicl_share_checkpoint:
            predictor_kwargs["share_checkpoint"] = True
        return {"predictor_kwargs": predictor_kwargs}
    if backend == "hierarchical_dirichlet":
        return {
            "predictor_kwargs": {
                "prior_strength": float(options.hierarchical_dirichlet_prior_strength),
                "base_prior_strength": float(
                    options.hierarchical_dirichlet_base_prior_strength
                ),
                "global_prior": float(options.hierarchical_dirichlet_global_prior),
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

