"""Experiment 18: CB and B-Learner comparison on Exp16 data.

This experiment compares the multi-environment partial-identification learners
from Schweisthal et al. (2024), notebook-faithful target-only B-Learner
sensitivity bounds, and a saved Exp16 naturalness result. It defines
exactly two environments for
the CB/WB learners:

* the target site's ``idle`` rows and a raw pool of source rows from the exact
  naturalness final-training cohort are the two fitting environments;
* WB candidates use Pi or the pooled-source environment; and
* configurable CB-PI, CB-RA, CB-IPW, and CB-DR learners use the two possible
  ordered pairs of those environments; and
* the fitting data source matches the saved naturalness reference; and
* the saved target RCT CATE labels are loaded only after all meta-learner fits
  have completed, to score the target-stratum intervals.

B-Learner is fit separately on target idle rows. Its default ``width_matched``
Gamma strategy fits the notebook's inclusive 1--5 grid and selects one Gamma
per study by mean absolute stratum-width gap from naturalness.
The opt-in ``width_matched_binary`` strategy uses a configurable coarse grid
to bracket the target-prevalence-weighted naturalness width, verifies the
fitted aggregate widths are monotone, and refines the bracket by binary search.
The ``exp_beta`` strategy instead performs one B-Learner fit with
``Gamma = exp(beta)``.
The ``oracle`` strategy uses saved OSRCT preferred-treatment probabilities to
calibrate the smallest finite-target-support marginal-sensitivity Gamma after
conditioning on the exact B-Learner covariates.
The ``dgp_oracle`` strategy instead performs that calibration over the full
pre-selection target RCT using the known OSRCT assignment mechanism.

The paper's target is environment-agnostic under its oracle-response-surface
assumption. Thus, each two-environment candidate is evaluated at the target
site's covariate strata and then intersected. This is intentionally a different
identifying assumption from naturalness's learned structural stability.

Independent WB and CB endpoint fits are dispatched to CPU processes.  Each
worker is restricted to a small number of Torch/BLAS threads so that process
parallelism does not oversubscribe the Kubernetes CPU allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit


from causal_grounding.loaders.schema import OUTCOME_COLUMN, TREATMENT_COLUMN
from causal_grounding.naturalness_grounder import SUPPORTED_GP_FEATURE_MODES
from experiments.osrct_target_pipeline import (
    DEFAULT_EXP16_STUDIES,
    Exp16Config,
)
from causal_grounding.loaders.cohort import OBSERVED_GENDER_COHORT_RULE
from experiments.benchmarks import (
    BLEARNER_BINARY_SEARCH_GAMMA_TOLERANCE,
    BLEARNER_BINARY_SEARCH_GRID,
    BLEARNER_BINARY_SEARCH_MAX_ITERATIONS,
    BLEARNER_BINARY_SEARCH_WIDTH_TOLERANCE,
    BLEARNER_COMMIT,
    BLEARNER_GAMMA_STRATEGIES,
    DGP_ORACLE_GAMMA,
    EXP_BETA_GAMMA,
    FitResult,
    FitTask,
    META_LEARNER_COLORS,
    META_LEARNER_LABELS,
    ORACLE_GAMMA,
    WIDTH_MATCHED_BINARY_GAMMA,
    WIDTH_MATCHED_GAMMA,
    BLearnerFitResult,
    BLearnerFitTask,
    BenchmarkFitConfig,
    blearner_gammas as _shared_blearner_gammas,
    make_blearner as _shared_make_blearner,
    parse_gamma as _shared_parse_gamma,
    parse_gamma_search_grid as _shared_parse_gamma_search_grid,
    resolve_n_jobs as _resolve_n_jobs,
    run_blearner_fit_tasks as _shared_run_blearner_fit_tasks,
    run_fit_tasks as _shared_run_fit_tasks,
    validate_blearner_repository as _shared_validate_blearner_repository,
)
from experiments.naturalness_plotting import (
    _format_stratum_tick,
    _prepare_study_panel_plot_data,
    _set_horizontal_study_panel_axis_labels,
    _set_study_panel_axis_labels,
)
from experiments.osrct_data_utils import load_prepared_data
from experiments.osrct_files import load_osrct_benchmark_module
from experiments.run_manifest import json_default, write_run_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


PAPER_METHODS = ("cb_pi", "cb_ra", "cb_ipw", "cb_dr")
DEFAULT_META_LEARNERS = ("cb_pi", "cb_ra")
ENDPOINTS = ("lower", "upper")
PAIR_SCOPES = ("target_anchored", "all_ordered")
SOURCE_REGIMES = ("idle", "on")
DATA_SOURCES = ("auto", "exp16")
SOURCE_SITE_POLICIES = (
    "random_sample",
    "naturalness_final_training",
    "all_non_target",
)
NATURALNESS_FINAL_TRAINING_ROLES = ("proper_training", "policy_tuning")
POOLED_SOURCE_ENVIRONMENT = "sources_pooled"
EXP16_REFERENCE = "exp16"
OSRCT_SELECTION_PROBABILITY_COLUMNS = (
    "_selection_prob",
    "osrct_preferred_treatment_probability",
)
ORACLE_GAMMA_ASSUMPTION = (
    "balanced_rct_osrct_propensity_equals_preferred_probability"
)
DGP_ORACLE_GAMMA_ASSUMPTION = (
    "known_balanced_rct_assignment_and_known_osrct_selection_dgp"
)
UNBALANCED_ORACLE_GAMMA_ASSUMPTION = (
    "known_osrct_selection_dgp_and_stratum_constant_assignment"
)
STRATUM_GROUP_SPACING = 1.15
HORIZONTAL_STRATUM_GROUP_SPACING = 1.90
HORIZONTAL_REFERENCE_LANE_SPACING = 0.24
HORIZONTAL_LEGEND_FONTSIZE = 14
HORIZONTAL_AXIS_LABEL_FONTSIZE = 16
DEFAULT_BLEARNER_REPOSITORY = Path(
    os.environ.get(
        "B_LEARNER_REPOSITORY",
        "/opt/BLearner" if Path("/opt/BLearner").exists() else "/tmp/BLearner",
    )
)
@dataclass(frozen=True)
class Exp18Config:
    """Configuration for matched-data CB, B-Learner, and naturalness comparison.

    ``data_source='auto'`` matches fitting rows to the Exp16 reference schema.
    The default reference selects Exp16 with ``target=pi``,
    ``pattern=gender``, and ``beta=1.0``. Target idle rows and all selected-
    source rows from one configured regime pooled into one raw-row environment
    are the only learner inputs. Target ``F='on'`` rows never enter a fit.
    """

    studies: tuple[str, ...] = DEFAULT_EXP16_STUDIES
    data_source: Literal["auto", "exp16"] = "auto"
    data_root: str = Exp16Config().data_root
    rct_path: str = Exp16Config().rct_path
    require_observed_gender: bool = True
    comparison_covariates: tuple[str, ...] | None = None
    pattern: str = "gender"
    beta: float = 1.0
    target_site: str = "pi"
    n_source_sites: int = 25
    expected_non_target_sites: int = 35
    source_site_policy: Literal[
        "random_sample", "naturalness_final_training", "all_non_target"
    ] = "random_sample"
    source_regime: Literal["idle", "on"] = "idle"
    pair_scope: str = "target_anchored"
    seed: int = 42
    n_jobs: int = -1
    worker_threads: int = 1
    n_layers_out: int = 2
    n_units_out: int = 100
    n_iter: int = 10_000
    batch_size: int = 100
    validation_fraction: float = 0.30
    n_iter_print: int = 50
    meta_learners: tuple[
        Literal["cb_pi", "cb_ra", "cb_ipw", "cb_dr"], ...
    ] = DEFAULT_META_LEARNERS
    gamma: float | Literal[
        "width_matched",
        "width_matched_binary",
        "exp_beta",
        "oracle",
        "dgp_oracle",
    ] = WIDTH_MATCHED_GAMMA
    gamma_search_grid: tuple[float, ...] = BLEARNER_BINARY_SEARCH_GRID
    gamma_search_tolerance: float = BLEARNER_BINARY_SEARCH_GAMMA_TOLERANCE
    gamma_search_width_tolerance: float = (
        BLEARNER_BINARY_SEARCH_WIDTH_TOLERANCE
    )
    gamma_search_max_iterations: int = BLEARNER_BINARY_SEARCH_MAX_ITERATIONS
    blearner_repository: str = str(DEFAULT_BLEARNER_REPOSITORY)
    naturalness_backend: str = "xgboost"
    naturalness_trigger_backend: str = "empirical"
    naturalness_lp_backend: str = "stage1"
    naturalness_gp_feature_mode: str = "thresholds"
    naturalness_pre_all_triggered_folder: str = ""
    naturalness_reference: str | None = None
    output_dir: str = str(PROJECT_ROOT / "results" / "exp18_cb_meta_learner")
    cluster_manifest_path: str | None = None
    command_argv: tuple[str, ...] = ()
    make_plots: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate the comparison setting before loading any source data."""

        if not self.studies:
            raise ValueError("At least one study must be requested")
        if self.comparison_covariates is not None:
            if not self.comparison_covariates:
                raise ValueError("comparison_covariates must not be empty")
            if len(set(self.comparison_covariates)) != len(
                self.comparison_covariates
            ):
                raise ValueError(
                    "comparison_covariates must not contain duplicates"
                )
        if self.data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {DATA_SOURCES}")
        if self.source_site_policy not in SOURCE_SITE_POLICIES:
            raise ValueError(
                f"source_site_policy must be one of {SOURCE_SITE_POLICIES}"
            )
        if self.n_source_sites < 1:
            raise ValueError("n_source_sites must be positive")
        if (
            self.source_site_policy == "random_sample"
            and self.expected_non_target_sites < self.n_source_sites
        ):
            raise ValueError(
                "expected_non_target_sites must be at least n_source_sites"
            )
        if self.source_regime not in SOURCE_REGIMES:
            raise ValueError(f"source_regime must be one of {SOURCE_REGIMES}")
        if self.pair_scope not in PAIR_SCOPES:
            raise ValueError(f"pair_scope must be one of {PAIR_SCOPES}")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.worker_threads < 1:
            raise ValueError("worker_threads must be positive")
        if self.n_layers_out < 1 or self.n_units_out < 1:
            raise ValueError("network layers and units must be positive")
        if self.n_iter < 1 or self.batch_size < 1:
            raise ValueError("n_iter and batch_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if not self.meta_learners:
            raise ValueError("At least one CB meta-learner must be requested")
        unsupported_meta_learners = tuple(
            method for method in self.meta_learners if method not in PAPER_METHODS
        )
        if unsupported_meta_learners:
            raise ValueError(
                "meta_learners must be drawn from "
                f"{PAPER_METHODS}; received {unsupported_meta_learners}"
            )
        if len(set(self.meta_learners)) != len(self.meta_learners):
            raise ValueError("meta_learners must not contain duplicates")
        if self.gamma not in BLEARNER_GAMMA_STRATEGIES and (
            not isinstance(self.gamma, (int, float))
            or not np.isfinite(float(self.gamma))
            or float(self.gamma) < 1.0
        ):
            raise ValueError(
                "gamma must be 'width_matched', 'width_matched_binary', "
                "'exp_beta', 'oracle', 'dgp_oracle', or a number at least one"
            )
        if self.gamma == EXP_BETA_GAMMA and (
            not np.isfinite(self.beta)
            or self.beta < 0.0
            or self.beta > np.log(np.finfo(float).max)
        ):
            raise ValueError(
                "gamma='exp_beta' requires a finite nonnegative beta whose "
                "exponential is representable"
            )
        gamma_search_grid = np.asarray(self.gamma_search_grid, dtype=float)
        if (
            len(gamma_search_grid) < 2
            or not np.isfinite(gamma_search_grid).all()
            or (gamma_search_grid < 1.0).any()
            or (np.diff(gamma_search_grid) <= 0.0).any()
        ):
            raise ValueError(
                "gamma_search_grid must contain at least two strictly "
                "increasing finite values at least one"
            )
        if (
            not np.isfinite(self.gamma_search_tolerance)
            or self.gamma_search_tolerance <= 0.0
        ):
            raise ValueError("gamma_search_tolerance must be positive and finite")
        if (
            not np.isfinite(self.gamma_search_width_tolerance)
            or self.gamma_search_width_tolerance < 0.0
        ):
            raise ValueError(
                "gamma_search_width_tolerance must be nonnegative and finite"
            )
        if self.gamma_search_max_iterations < 1:
            raise ValueError("gamma_search_max_iterations must be positive")
        if not self.naturalness_backend or any(
            separator in self.naturalness_backend for separator in ("/", "\\")
        ):
            raise ValueError(
                "naturalness_backend must be a non-empty path component"
            )
        for field_name, component in (
            ("naturalness_trigger_backend", self.naturalness_trigger_backend),
            ("naturalness_lp_backend", self.naturalness_lp_backend),
        ):
            if not component or any(
                separator in component for separator in ("/", "\\")
            ):
                raise ValueError(f"{field_name} must be a non-empty path component")
        if self.naturalness_gp_feature_mode not in SUPPORTED_GP_FEATURE_MODES:
            raise ValueError(
                "naturalness_gp_feature_mode must be one of "
                f"{SUPPORTED_GP_FEATURE_MODES}; got "
                f"{self.naturalness_gp_feature_mode!r}"
            )
        if self.naturalness_pre_all_triggered_folder in {".", ".."} or any(
            separator in self.naturalness_pre_all_triggered_folder
            for separator in ("/", "\\")
        ):
            raise ValueError(
                "naturalness_pre_all_triggered_folder must be one path component"
            )


@dataclass(frozen=True)
class PreparedStudyData:
    """Idle and randomized fitting rows prepared for one study.

    Args:
        idle: Observational rows used for the target and, when requested,
            pooled sources.
        rct: Original randomized rows available only to pooled sources when
            ``source_regime='on'``.
    """

    idle: pd.DataFrame
    rct: pd.DataFrame


def _format_beta_path(beta: float) -> str:
    """Format beta as a stable directory component."""

    value = float(beta)
    return f"{value:.1f}" if value.is_integer() else f"{value:g}"


def _comparison_covariates(config: Exp18Config) -> list[str]:
    """Return reporting covariates for the configured benchmark."""

    if config.comparison_covariates is not None:
        return list(config.comparison_covariates)
    return list(Exp16Config().covariates)


def _study_target_site(config: Exp18Config, study: str) -> str:
    """Return the target environment label for one comparison panel."""

    return str(config.target_site)


def _covariate_matrix(
    data: pd.DataFrame,
    covariates: Sequence[str],
) -> np.ndarray:
    """Convert reporting covariates to numeric learner inputs."""

    encoded = data.loc[:, list(covariates)].copy()
    return encoded.to_numpy(dtype=np.float32)


def _parse_gamma(
    value: str,
) -> float | Literal[
    "width_matched",
    "width_matched_binary",
    "exp_beta",
    "oracle",
    "dgp_oracle",
]:
    """Parse one fixed B-Learner Gamma or a width-matched strategy.

    Args:
        value: Command-line value supplied to ``--gamma``.

    Returns:
        A supported width-matching strategy or floating-point Gamma at least
        one.
    """

    return _shared_parse_gamma(value)


def _parse_gamma_search_grid(value: str) -> tuple[float, ...]:
    """Parse one comma-separated coarse Gamma search grid.

    Args:
        value: Comma-separated Gamma values supplied to
            ``--gamma-search-grid``.

    Returns:
        Parsed Gamma values. Configuration validation checks ordering and
        admissible range.

    Raises:
        argparse.ArgumentTypeError: If any grid value is not numeric.
    """

    return _shared_parse_gamma_search_grid(value)


def _blearner_gammas(config: Exp18Config) -> tuple[float, ...]:
    """Return Gamma values required by the configured B-Learner strategy.

    Args:
        config: Exp18 configuration with fixed or width-matched Gamma.

    Returns:
        Grid needed by the selected width-matching strategy or one fixed
        Gamma. Oracle calibration returns an empty tuple because its
        study-specific values require loaded target OSRCT rows.
    """

    return _shared_blearner_gammas(
        config.gamma,
        beta=config.beta,
        gamma_search_grid=config.gamma_search_grid,
    )


def _identify_naturalness_reference_experiment(
    columns: Sequence[str],
) -> Literal["exp16"]:
    """Identify an Exp16 posterior artifact from columns.

    Args:
        columns: CSV column names without loading any reference values.

    Returns:
        Benchmark identifier matching the posterior-summary schema.

    Raises:
        ValueError: If columns do not match the supported result schema.
    """

    column_set = set(columns)
    if {"pattern", "beta"}.issubset(column_set):
        return EXP16_REFERENCE
    raise ValueError(
        "Naturalness reference is not a recognized Exp16 posterior summary"
    )


def _naturalness_results_roots() -> tuple[Path, ...]:
    """Return mounted and repository-local roots that can hold results.

    Returns:
        Ceph results root first for cluster jobs, followed by the repository's
        local results directory for development runs.
    """

    return (
        Path("/mnt/cephfs/predictive-shift/results"),
        PROJECT_ROOT / "results",
    )


def _resolved_naturalness_backend_signature(
    config: Exp18Config,
) -> tuple[str, str, str]:
    """Resolve Stage-1, trigger, and LP aliases for routing and output paths.

    Args:
        config: Exp18 configuration containing the saved-reference backend
            selectors.

    Returns:
        Concrete ``(stage1, trigger, lp)`` backend names. ``stage1`` LP aliases
        resolve to the Stage-1 backend, while ``lp`` trigger aliases resolve to
        the concrete LP backend.
    """

    stage1_backend = config.naturalness_backend
    lp_backend = (
        stage1_backend
        if config.naturalness_lp_backend == "stage1"
        else config.naturalness_lp_backend
    )
    if config.naturalness_trigger_backend == "stage1":
        trigger_backend = stage1_backend
    elif config.naturalness_trigger_backend == "lp":
        trigger_backend = lp_backend
    else:
        trigger_backend = config.naturalness_trigger_backend
    return stage1_backend, trigger_backend, lp_backend


def _naturalness_reference_path(config: Exp18Config) -> Path:
    """Return an explicit reference or derive one from its backend selector.

    Args:
        config: Exp18 configuration with a backend and target setting.

    Returns:
        Existing posterior-summary artifact for the selected benchmark
        backend. When no derived artifact exists yet, returns the canonical
        cluster path used by current Exp16 output routing.
    """

    if config.naturalness_reference is not None:
        return Path(config.naturalness_reference)
    candidates: list[Path] = []
    for results_root in _naturalness_results_roots():
        pre_trigger_root = (
            Path(config.naturalness_pre_all_triggered_folder)
            if config.naturalness_pre_all_triggered_folder
            else Path()
        )
        stage1_backend, trigger_backend, lp_backend = (
            _resolved_naturalness_backend_signature(config)
        )
        backend_signatures = list(
            dict.fromkeys(
                (
                    (stage1_backend, trigger_backend, lp_backend),
                    (
                        config.naturalness_backend,
                        config.naturalness_trigger_backend,
                        config.naturalness_lp_backend,
                    ),
                )
            )
        )
        for backend_signature in backend_signatures:
            exp16_mode_root = (
                results_root
                / "exp16_osrct_calibrated_policy"
                / backend_signature[0]
                / backend_signature[1]
                / backend_signature[2]
                / "all_triggered"
            )
            if config.naturalness_gp_feature_mode != "thresholds":
                exp16_mode_root /= config.naturalness_gp_feature_mode
            exp16_backend_root = exp16_mode_root / str(config.target_site)
            canonical_exp16_reference = (
                exp16_backend_root
                / str(config.pattern)
                / _format_beta_path(config.beta)
                / "target_gp_posterior_summary.csv"
            )
            candidates.append(canonical_exp16_reference)
            if exp16_mode_root.exists() and not canonical_exp16_reference.exists():
                variant_pattern = str(
                    Path("**")
                    / str(config.target_site)
                    / str(config.pattern)
                    / _format_beta_path(config.beta)
                    / "target_gp_posterior_summary.csv"
                )
                variant_references = sorted(
                    path
                    for path in exp16_mode_root.glob(variant_pattern)
                    if path != canonical_exp16_reference
                    and not any(
                        path.is_relative_to(exp16_mode_root / feature_mode)
                        for feature_mode in SUPPORTED_GP_FEATURE_MODES
                        if feature_mode != "thresholds"
                        and config.naturalness_gp_feature_mode == "thresholds"
                    )
                )
                if len(variant_references) > 1:
                    raise ValueError(
                        "Multiple Exp16 naturalness references match the "
                        "configured Stage-1, trigger, and LP backends. Supply "
                        "--naturalness-reference explicitly: "
                        f"{variant_references}"
                    )
                candidates.extend(variant_references)
        legacy_exp16_mode_root = (
            results_root
            / "exp16_osrct_calibrated_policy"
            / pre_trigger_root
            / "all_triggered"
        )
        if config.naturalness_gp_feature_mode != "thresholds":
            legacy_exp16_mode_root /= config.naturalness_gp_feature_mode
        legacy_exp16_root = (
            legacy_exp16_mode_root
            / config.naturalness_backend
            / str(config.target_site)
        )
        candidates.extend(
            (
                legacy_exp16_root
                / str(config.pattern)
                / _format_beta_path(config.beta)
                / "target_gp_posterior_summary.csv",
                # Before Exp16's backend-root output normalization, the
                # backend was repeated below the target site.
                legacy_exp16_root
                / config.naturalness_backend
                / str(config.pattern)
                / _format_beta_path(config.beta)
                / "target_gp_posterior_summary.csv",
            )
        )
    return next((path for path in candidates if path.exists()), candidates[0])


def _inspect_naturalness_reference(
    config: Exp18Config,
) -> Literal["exp16"]:
    """Validate the reference path and identify its table schema."""

    path = _naturalness_reference_path(config)
    if not path.exists():
        raise FileNotFoundError(
            "Missing Exp16 naturalness reference: "
            f"{path}. Supply --naturalness-reference if it lives elsewhere."
        )
    columns = _read_table(path).columns.tolist()
    return _identify_naturalness_reference_experiment(columns)


def _read_table(path: Path) -> pd.DataFrame:
    """Read one supported experiment table from CSV or Parquet."""

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_data_source(
    config: Exp18Config,
    reference_experiment: str,
) -> Literal["exp16"]:
    """Resolve and validate the fitting-data source against the reference.

    Args:
        config: Exp18 configuration with automatic or explicit data source.
        reference_experiment: Experiment identified from the reference CSV.

    Returns:
        Matching fitting-data source.

    Raises:
        ValueError: If an explicit data source differs from the reference.
    """

    data_source = (
        reference_experiment if config.data_source == "auto" else config.data_source
    )
    if data_source != reference_experiment:
        raise ValueError(
            "Fitting data and naturalness reference must come from the same "
            f"experiment; got data_source={data_source!r} and "
            f"reference={reference_experiment!r}"
        )
    return data_source


def _leaf_output_dir(
    config: Exp18Config,
    *,
    data_source: str | None = None,
) -> Path:
    """Return the unique output leaf for one Exp18 comparison setting.

    Exp16 references use the compact layout
    ``experiment/stage1/trigger/lp/[gp_feature_mode/]gamma/setting_beta``.
    The GP feature component is omitted for the default threshold mode. For
    Setting is the OSRCT pattern. Other references use the legacy layout with the selected
    naturalness backend and optional folder before
    ``all_triggered``, is placed below the supplied output root. This prevents
    comparisons against different Exp16 result trees from overwriting one
    another. Exp16 retains the target/pattern/beta hierarchy. Binary Gamma searches add a strategy
    suffix so they cannot overwrite existing grid-only outputs. A canonical
    meta-learner component above the setting leaf isolates different CB method
    subsets.

    Args:
        config: Exp18 configuration defining the target and source setting.
        data_source: Resolved fitting-data source. ``None`` preserves the
            original Exp16-compatible path for helper callers.

    Returns:
        Setting-specific output directory.

    Raises:
        ValueError: If a non-null data source is unsupported.
    """

    if data_source not in (
        None,
        EXP16_REFERENCE,
    ):
        raise ValueError("Unsupported fitting-data source")
    if data_source == EXP16_REFERENCE:
        stage1_backend, trigger_backend, lp_backend = (
            _resolved_naturalness_backend_signature(config)
        )
        gamma_label = (
            f"{float(config.gamma):g}"
            if isinstance(config.gamma, (float, int))
            else str(config.gamma)
        )
        setting = config.pattern
        setting_label = str(setting).replace("/", "_").replace("\\", "_")
        setting_beta_label = f"{setting_label}_{_format_beta_path(config.beta)}"
        output_components = [
            data_source,
            stage1_backend,
            trigger_backend,
            lp_backend,
        ]
        if config.naturalness_gp_feature_mode != "thresholds":
            output_components.append(config.naturalness_gp_feature_mode)
        output_components.extend((gamma_label, setting_beta_label))
        return Path(config.output_dir).joinpath(*output_components)
    source_label = {
        "random_sample": f"source{config.n_source_sites}",
        "naturalness_final_training": "sources_naturalness_final_training",
        "all_non_target": "sources_all_non_target",
    }[config.source_site_policy]
    leaf_name = (
        f"{source_label}_pooled_{config.source_regime}_{config.pair_scope}"
    )
    if config.gamma == WIDTH_MATCHED_BINARY_GAMMA:
        leaf_name = f"{leaf_name}_gamma_{WIDTH_MATCHED_BINARY_GAMMA}"
    elif config.gamma == EXP_BETA_GAMMA:
        leaf_name = f"{leaf_name}_gamma_{EXP_BETA_GAMMA}"
    elif config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA):
        leaf_name = f"{leaf_name}_gamma_{config.gamma}"
    target_root = Path(config.output_dir) / config.naturalness_backend
    if config.naturalness_pre_all_triggered_folder:
        target_root /= config.naturalness_pre_all_triggered_folder
    meta_learner_leaf = "_".join(
        method.replace("_", "")
        for method in PAPER_METHODS
        if method in config.meta_learners
    )
    target_root /= str(config.target_site)
    return (
        target_root
        / str(config.pattern)
        / _format_beta_path(config.beta)
        / meta_learner_leaf
        / leaf_name
    )


def _exp16_config(config: Exp18Config, study: str) -> Exp16Config:
    """Build the data-only Exp16 configuration for one requested study."""

    return Exp16Config(
        study=study,
        studies=(study,),
        pattern=config.pattern,
        patterns=(config.pattern,),
        beta=float(config.beta),
        betas=(float(config.beta),),
        target_sites=(config.target_site,),
        data_root=config.data_root,
        rct_path=config.rct_path,
        require_observed_gender=bool(config.require_observed_gender),
        seed=config.seed,
        make_plots=False,
        verbose=False,
    )


def _load_fitting_data(
    config: Exp18Config,
    data_source: Literal["exp16"],
) -> tuple[dict[str, PreparedStudyData], pd.DataFrame]:
    """Load or reconstruct all fitting rows for the resolved experiment.

    Args:
        config: Exp18 comparison configuration.
        data_source: Matched Exp16 data source.

    Returns:
        Study-indexed prepared rows plus saved provenance diagnostics.
    """

    prepared: dict[str, PreparedStudyData] = {}
    provenance_rows: list[dict[str, Any]] = []
    for study in config.studies:
        osrct, rct, metadata = load_prepared_data(_exp16_config(config, study))
        prepared[str(study)] = PreparedStudyData(idle=osrct, rct=rct)
        provenance_rows.append(
            {
                "study": str(study),
                "data_source": EXP16_REFERENCE,
                "pattern": config.pattern,
                "beta": float(config.beta),
                "n_idle_rows": len(osrct),
                "n_rct_rows": len(rct),
                "loader_metadata": json.dumps(metadata, default=json_default),
            }
        )
    return prepared, pd.DataFrame(provenance_rows)


def _run_fit_tasks(
    tasks: Sequence[FitTask],
    config: Exp18Config,
) -> tuple[dict[int, FitResult], pd.DataFrame]:
    """Compatibility wrapper for shared CATENets task execution."""

    return _shared_run_fit_tasks(tasks, _benchmark_fit_config(config))


def _load_naturalness_reference(config: Exp18Config) -> pd.DataFrame:
    """Load matching naturalness bounds and evaluation CATEs.

    Exp16 posterior summaries are identified by their ``pattern`` and ``beta``
    metadata and must match the OSRCT setting configured for Exp18.

    Args:
        config: Exp18 configuration containing the reference path and target
            setting.

    Returns:
        Validated reference rows with normalized naturalness endpoint, target
        row-count, blackbox-reference, and source-experiment columns.

    Raises:
        FileNotFoundError: If the configured reference does not exist.
        ValueError: If the schema, metadata, studies, or strata do not match.
    """

    reference_experiment = _inspect_naturalness_reference(config)
    path = _naturalness_reference_path(config)
    reference = _read_table(path)
    covariates = list(Exp16Config().covariates)
    required = {
        "study",
        *covariates,
        "lower_mean",
        "upper_mean",
        "n_target_rows",
        "rct_cate",
        "rct_cate_ci90_lower",
        "rct_cate_ci90_upper",
        "rct_cate_available",
        "lp_posterior_prob_optimal",
    }
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"Naturalness reference is missing columns: {sorted(missing)}")
    metadata_expectations: list[tuple[str, Any]] = []
    metadata_expectations.append(("target_site", config.target_site))
    metadata_expectations.extend(
        (("pattern", config.pattern), ("beta", float(config.beta)))
    )
    for column, expected in metadata_expectations:
        if column not in reference:
            raise ValueError(
                "Naturalness reference is missing metadata column: "
                f"{column}"
            )
        values = reference[column].dropna().unique()
        if len(values) != 1 or values[0] != expected:
            raise ValueError(
                f"Expected naturalness {column}={expected!r}; found {values!r}"
            )
    reference = reference.loc[reference["study"].isin(config.studies)].copy()
    if reference.empty:
        raise ValueError("Naturalness reference has no rows for the requested studies")
    if reference.duplicated(["study", *covariates]).any():
        raise ValueError("Naturalness reference has duplicate study/stratum rows")
    reference = reference.rename(
        columns={
            "lower_mean": "naturalness_lower",
            "upper_mean": "naturalness_upper",
            "posterior_envelope_lower": "naturalness_envelope_lower",
            "posterior_envelope_upper": "naturalness_envelope_upper",
            "posterior_envelope_width": "naturalness_envelope_width",
            "n_target_rows": "n_target_idle_rows",
        }
    )
    for column in (
        "naturalness_envelope_lower",
        "naturalness_envelope_upper",
    ):
        if column not in reference:
            reference[column] = np.nan
    if "naturalness_envelope_width" not in reference:
        reference["naturalness_envelope_width"] = (
            reference["naturalness_envelope_upper"]
            - reference["naturalness_envelope_lower"]
        )
    reference["naturalness_width"] = (
        reference["naturalness_upper"] - reference["naturalness_lower"]
    )
    if "blackbox_reference_cate" not in reference:
        reference["blackbox_reference_cate"] = np.nan
    reference["naturalness_reference_experiment"] = reference_experiment
    return reference


def _benchmark_fit_config(config: Exp18Config) -> BenchmarkFitConfig:
    """Convert Exp18's public configuration into shared fit settings."""

    return BenchmarkFitConfig(
        seed=config.seed,
        n_jobs=config.n_jobs,
        worker_threads=config.worker_threads,
        n_layers_out=config.n_layers_out,
        n_units_out=config.n_units_out,
        n_iter=config.n_iter,
        batch_size=config.batch_size,
        validation_fraction=config.validation_fraction,
        n_iter_print=config.n_iter_print,
        blearner_repository=config.blearner_repository,
        verbose=config.verbose,
    )


def _validate_blearner_repository(repository: Path) -> None:
    """Compatibility wrapper for shared checkout validation."""

    _shared_validate_blearner_repository(repository)


def _make_blearner(config: Exp18Config, gamma: float) -> Any:
    """Compatibility wrapper for shared B-Learner construction."""

    return _shared_make_blearner(_benchmark_fit_config(config), gamma)


@dataclass(frozen=True)
class _BLearnerStudyInputs:
    """Validated fitting arrays and stratum tables for one study."""

    covariates: tuple[str, ...]
    x: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    strata: pd.DataFrame
    strata_x: np.ndarray
    target_counts: pd.DataFrame


def _blearner_study_inputs(
    config: Exp18Config,
    study: str,
    target_idle: pd.DataFrame,
) -> _BLearnerStudyInputs:
    """Extract and validate every array one study's B-Learner fits need.

    Args:
        config: Exp18 configuration defining the comparison covariates.
        study: Study label used in the treatment-arm error message.
        target_idle: Matched target-site idle rows used for B-Learner fitting.

    Returns:
        Fitting arrays, the sorted unique strata and their covariate matrix,
        and per-stratum target row counts.

    Raises:
        ValueError: If the target idle rows lack either treatment arm.
    """

    covariates = _comparison_covariates(config)
    treatment = target_idle[TREATMENT_COLUMN].to_numpy(dtype=int)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError(
            f"{study}/{_study_target_site(config, study)} target idle data "
            "lack both treatment arms"
        )
    strata = (
        target_idle[covariates]
        .drop_duplicates()
        .sort_values(covariates)
        .reset_index(drop=True)
    )
    return _BLearnerStudyInputs(
        covariates=tuple(covariates),
        x=_covariate_matrix(target_idle, covariates).astype(float),
        treatment=treatment,
        outcome=target_idle[OUTCOME_COLUMN].to_numpy(dtype=float),
        strata=strata,
        strata_x=_covariate_matrix(strata, covariates).astype(float),
        target_counts=(
            target_idle.groupby(covariates, observed=True)
            .size()
            .rename("n_target_idle_rows")
            .reset_index()
        ),
    )


def _fit_blearner_study_bounds(
    config: Exp18Config,
    study: str,
    target_idle: pd.DataFrame,
    gammas: Sequence[float],
    *,
    fitted_endpoints: Mapping[float, BLearnerFitResult] | None = None,
) -> pd.DataFrame:
    """Fit notebook-faithful B-Learner bounds on one target idle dataset.

    Args:
        config: Exp18 configuration.
        study: Study label stored with fitted bounds.
        target_idle: Matched target-site idle rows used for B-Learner fitting.
        gammas: Gamma values fitted independently.
        fitted_endpoints: Raw endpoints already fitted for this study, keyed by
            Gamma. Callers that fit a whole Gamma grid in parallel pass them so
            no fit is repeated here; when omitted every Gamma is fitted in
            process, one after another.

    Returns:
        One row per target stratum and Gamma with fitted sensitivity bounds.
    """

    inputs = _blearner_study_inputs(config, study, target_idle)
    covariates = list(inputs.covariates)
    strata = inputs.strata
    frames: list[pd.DataFrame] = []
    for gamma in gammas:
        fit_start = time.perf_counter()
        if fitted_endpoints is None:
            learner = _make_blearner(config, float(gamma))
            learner.fit(inputs.x, inputs.treatment, inputs.outcome)
            lower, upper = learner.effect(inputs.strata_x)
            fit_runtime_seconds = time.perf_counter() - fit_start
        else:
            fitted = fitted_endpoints[float(gamma)]
            lower, upper = fitted.lower, fitted.upper
            fit_runtime_seconds = fitted.runtime_seconds
        lower_raw = np.asarray(lower, dtype=float)
        upper_raw = np.asarray(upper, dtype=float)
        bounds_rearranged = lower_raw > upper_raw
        if np.isclose(gamma, 1.0):
            point_estimate = (lower_raw + upper_raw) / 2.0
            lower = point_estimate
            upper = point_estimate
        else:
            # B-Learner estimates its endpoint forests independently, so
            # finite-sample predictions are not constrained to preserve their
            # theoretical order. Pointwise rearrangement restores a coherent
            # interval while retaining both endpoint estimates.
            lower = np.minimum(lower_raw, upper_raw)
            upper = np.maximum(lower_raw, upper_raw)
        frame = strata.copy()
        frame["study"] = study
        frame["pattern"] = (
            config.pattern if config.data_source == EXP16_REFERENCE else None
        )
        frame["beta"] = config.beta
        frame["target_site"] = _study_target_site(config, study)
        frame["gamma"] = float(gamma)
        frame["blearner_lower_raw"] = lower_raw
        frame["blearner_upper_raw"] = upper_raw
        frame["blearner_bounds_rearranged"] = (
            bounds_rearranged & ~np.isclose(gamma, 1.0)
        )
        frame["blearner_lower"] = lower
        frame["blearner_upper"] = upper
        frame["blearner_width"] = upper - lower
        frame["outside_binary_cate_support"] = (lower < -1.0) | (upper > 1.0)
        frame["fit_runtime_seconds"] = fit_runtime_seconds
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.merge(
        inputs.target_counts,
        on=covariates,
        how="left",
        validate="many_to_one",
    )


def _post_selection_propensity(
    preferred_probability: np.ndarray,
    randomized_treatment_probability: float,
) -> np.ndarray:
    """Convert preferred-treatment probabilities to post-OSRCT propensities.

    An OSRCT retains a randomized row when its assignment equals the drawn
    preferred treatment, so the retained treatment propensity is
    ``p*s / (p*s + (1-p)*(1-s))`` for randomization probability ``p`` and
    preferred-treatment probability ``s``. Balanced assignment (``p = 0.5``)
    makes this the identity, which is why the Exp16 OSRCT benchmark can use
    the saved preferred probability directly; the general form supports
    unbalanced assignment too.

    Args:
        preferred_probability: Per-row preferred-treatment probabilities.
        randomized_treatment_probability: Randomization probability governing
            the rows, assumed constant across the supplied rows.

    Returns:
        Post-selection treatment propensities, one per supplied row.

    Raises:
        ValueError: If the randomization probability is outside ``(0, 1)``.
    """

    probability = float(randomized_treatment_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("randomized_treatment_probability must lie in (0, 1)")
    preferred = np.asarray(preferred_probability, dtype=float)
    # Balanced assignment is an exact identity; short-circuiting keeps Exp16
    # results bit-identical rather than round-tripping through the ratio. The
    # copy keeps callers from aliasing their own input array.
    if probability == 0.5:
        return preferred.copy()
    treated = probability * preferred
    return treated / (treated + (1.0 - probability) * (1.0 - preferred))


def _osrct_pattern_preferred_probability(
    full_rct: pd.DataFrame,
    *,
    pattern: str,
    beta: float,
) -> np.ndarray:
    """Rebuild Exp16 OSRCT preferred-treatment probabilities from its DGP.

    The benchmark's own sampler is reconstructed with the configured pattern
    and biasing strength, exactly as in dataset generation, so the returned
    probabilities are the true selection probabilities rather than estimates.

    Args:
        full_rct: Complete pre-selection RCT rows carrying every biasing
            covariate. Standardization is fit on this table, so callers must
            pass the whole study rather than one site.
        pattern: OSRCT biasing pattern name from the benchmark grid.
        beta: OSRCT biasing strength.

    Returns:
        Preferred-treatment probability per row, in ``full_rct`` row order.

    Raises:
        ValueError: If the pattern is absent from the benchmark grid.
    """

    pattern_configs = load_osrct_benchmark_module("experimental_grid").COVARIATE_PATTERNS
    if pattern not in pattern_configs:
        raise ValueError(f"Unknown OSRCT pattern for DGP Gamma: {pattern!r}")
    pattern_config = pattern_configs[pattern]
    sampler_class = load_osrct_benchmark_module("osrct").OSRCTSampler
    sampler = sampler_class(
        biasing_covariates=pattern_config["covariates"],
        biasing_coefficients=pattern_config["coefficients"](float(beta)),
        intercept=0.0,
        standardize=True,
        random_seed=0,
    )
    return np.asarray(sampler.selection_probability(full_rct), dtype=float)


def _known_preferred_probability(
    config: Exp18Config,
    full_rct: pd.DataFrame,
) -> np.ndarray:
    """Return Exp16's known preferred-treatment probability.

    Args:
        config: Resolved Exp18 configuration naming the data source.
        full_rct: Complete pre-selection RCT rows for one study.

    Returns:
        Preferred-treatment probability per row, in ``full_rct`` row order.
    """

    return _osrct_pattern_preferred_probability(
        full_rct,
        pattern=str(config.pattern),
        beta=float(config.beta),
    )


def _study_randomized_treatment_probability(
    config: Exp18Config,
    full_rct: pd.DataFrame,
    *,
    target_site: str,
) -> float:
    """Return the randomization probability governing one target site.

    Exp16's benchmark RCTs are balanced by design, so the known value is used
    directly.

    Args:
        config: Resolved Exp18 configuration naming the data source.
        full_rct: Complete pre-selection RCT rows for one study.
        target_site: Target site whose assignment rate is required.

    Returns:
        Randomization probability for the target site.

    The data arguments remain part of the shared calibration interface but do
    not affect this known design constant.
    """

    del config, full_rct, target_site
    return 0.5


def _oracle_gamma_calibration(
    target_idle: pd.DataFrame,
    covariates: Sequence[str],
    *,
    study: str,
    randomized_treatment_probability: float = 0.5,
) -> dict[str, Any]:
    """Calibrate finite-support B-Learner Gamma from the OSRCT mechanism.

    The saved OSRCT preferred-treatment probability is converted to the full
    treatment propensity after observational sampling; under balanced RCT
    assignment the two coincide. The observed B-Learner propensity is its
    empirical mean within each exact B-Learner covariate stratum. The returned
    Gamma is the maximum full-versus-observed treatment-odds ratio, in either
    direction, across retained target rows. It is therefore the smallest
    marginal-sensitivity Gamma covering the target sample under this known
    OSRCT mechanism.

    Args:
        target_idle: Target OSRCT rows containing a saved preferred-treatment
            probability and every B-Learner conditioning covariate.
        covariates: Exact covariates supplied to B-Learner.
        study: Study label stored in calibration diagnostics.
        randomized_treatment_probability: Randomization probability governing
            the target rows. Defaults to balanced Exp16 assignment.

    Returns:
        Scalar calibration diagnostics including ``gamma`` and
        ``oracle_log_gamma``.

    Raises:
        ValueError: If required columns are missing, probabilities are outside
            ``(0, 1)``, covariates are missing, or Gamma is not finite.
    """

    missing_covariates = [
        covariate for covariate in covariates if covariate not in target_idle
    ]
    if missing_covariates:
        raise ValueError(
            "Oracle Gamma requires every B-Learner covariate; missing "
            f"{missing_covariates}"
        )
    probability_columns = [
        column
        for column in OSRCT_SELECTION_PROBABILITY_COLUMNS
        if column in target_idle
    ]
    if not probability_columns:
        raise ValueError(
            "gamma='oracle' requires saved OSRCT preferred-treatment "
            "probabilities in one of "
            f"{OSRCT_SELECTION_PROBABILITY_COLUMNS}"
        )
    probability_column = probability_columns[0]
    preferred_probability = pd.to_numeric(
        target_idle[probability_column],
        errors="coerce",
    )
    if (
        preferred_probability.isna().any()
        or not np.isfinite(preferred_probability.to_numpy()).all()
        or preferred_probability.le(0.0).any()
        or preferred_probability.ge(1.0).any()
    ):
        raise ValueError(
            f"Oracle Gamma column {probability_column!r} must contain finite "
            "probabilities strictly between zero and one"
        )
    if target_idle[list(covariates)].isna().any().any():
        raise ValueError(
            "Oracle Gamma conditioning covariates must not be missing"
        )

    calibration = target_idle[list(covariates)].copy()
    calibration["_oracle_full_propensity"] = _post_selection_propensity(
        preferred_probability.to_numpy(),
        randomized_treatment_probability,
    )
    calibration["_oracle_observed_propensity"] = calibration.groupby(
        list(covariates),
        observed=True,
        dropna=False,
    )["_oracle_full_propensity"].transform("mean")
    observed_probability = calibration["_oracle_observed_propensity"]
    if observed_probability.le(0.0).any() or observed_probability.ge(1.0).any():
        raise ValueError(
            "Oracle Gamma marginalized propensities must lie strictly between "
            "zero and one"
        )

    log_odds_deviation = np.abs(
        logit(calibration["_oracle_full_propensity"].to_numpy(dtype=float))
        - logit(observed_probability.to_numpy(dtype=float))
    )
    oracle_log_gamma = float(np.max(log_odds_deviation))
    if np.isclose(oracle_log_gamma, 0.0, atol=1e-12, rtol=0.0):
        oracle_log_gamma = 0.0
    gamma = float(np.exp(oracle_log_gamma))
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("Oracle Gamma calibration produced an invalid value")

    return {
        "study": str(study),
        "gamma": gamma,
        "oracle_log_gamma": oracle_log_gamma,
        "oracle_selection_probability_column": probability_column,
        "oracle_n_target_idle_rows": int(len(calibration)),
        "oracle_n_blearner_strata": int(
            calibration[list(covariates)].drop_duplicates().shape[0]
        ),
        "oracle_full_propensity_min": float(
            calibration["_oracle_full_propensity"].min()
        ),
        "oracle_full_propensity_max": float(
            calibration["_oracle_full_propensity"].max()
        ),
        "oracle_observed_propensity_min": float(observed_probability.min()),
        "oracle_observed_propensity_max": float(observed_probability.max()),
        "oracle_gamma_assumption": (
            ORACLE_GAMMA_ASSUMPTION
            if randomized_treatment_probability == 0.5
            else UNBALANCED_ORACLE_GAMMA_ASSUMPTION
        ),
        "oracle_randomized_treatment_probability": float(
            randomized_treatment_probability
        ),
    }


def _dgp_oracle_gamma_calibration(
    full_rct: pd.DataFrame,
    covariates: Sequence[str],
    *,
    study: str,
    target_site: str,
    pattern: str,
    beta: float,
    randomized_treatment_probability: float = 0.5,
    preferred_probability: np.ndarray | None = None,
    selection_mechanism: str = "known_osrct_dgp",
) -> dict[str, Any]:
    """Calibrate Gamma from the known OSRCT DGP on pre-selection target rows.

    For each target unit, the known preferred-treatment probability is
    converted to its post-selection treatment propensity. The observed
    propensity is its retention-weighted mean within the exact B-Learner
    covariate stratum. Gamma is the maximum absolute log-odds deviation over
    the full pre-selection target cohort.

    The randomization probability cancels exactly whenever it is constant
    within a B-Learner stratum, because both propensities then carry the same
    ``logit(p)`` offset. It is retained as a parameter for diagnostics and for
    designs where that constancy does not hold.

    Args:
        full_rct: Complete pre-selection RCT rows for one study.
        covariates: Exact covariates supplied to B-Learner.
        study: Study label stored in diagnostics.
        target_site: Target site used by the B-Learner fit.
        pattern: OSRCT biasing pattern name, used to rebuild the Exp16
            mechanism when ``preferred_probability`` is not supplied and
            recorded in diagnostics either way.
        beta: OSRCT biasing strength.
        randomized_treatment_probability: Known RCT assignment probability.
        preferred_probability: Known preferred-treatment probability per row
            of ``full_rct``. Defaults to rebuilding the Exp16 benchmark
            sampler from ``pattern``.
        selection_mechanism: Label describing where the probabilities came
            from, stored in diagnostics.

    Returns:
        Study-level DGP calibration diagnostics including ``gamma``.

    Raises:
        ValueError: If required columns are missing, the supplied
            probabilities do not align with ``full_rct``, the target cohort is
            empty, or the calibration produces an invalid Gamma.
    """

    if not 0.0 < randomized_treatment_probability < 1.0:
        raise ValueError("randomized_treatment_probability must lie in (0, 1)")
    required = {"site", *covariates}
    missing = required.difference(full_rct.columns)
    if missing:
        raise ValueError(
            f"DGP-oracle Gamma RCT data are missing columns: {sorted(missing)}"
        )

    if preferred_probability is None:
        preferred_probability = _osrct_pattern_preferred_probability(
            full_rct,
            pattern=pattern,
            beta=float(beta),
        )
    else:
        preferred_probability = np.asarray(preferred_probability, dtype=float)
        if preferred_probability.shape != (len(full_rct),):
            raise ValueError(
                "DGP-oracle Gamma preferred probabilities must have one entry "
                f"per pre-selection row; got {preferred_probability.shape} for "
                f"{len(full_rct)} rows"
            )
    randomized_probability = float(randomized_treatment_probability)
    retention_probability = (
        randomized_probability * preferred_probability
        + (1.0 - randomized_probability) * (1.0 - preferred_probability)
    )
    full_propensity = (
        randomized_probability
        * preferred_probability
        / retention_probability
    )

    target_mask = full_rct["site"].astype(str).eq(str(target_site)).to_numpy()
    target = full_rct.loc[target_mask, list(covariates)].copy()
    if target.empty:
        raise ValueError(
            f"DGP-oracle Gamma found no pre-selection rows for site {target_site!r}"
        )
    if target.isna().any().any():
        raise ValueError("DGP-oracle Gamma covariates must not be missing")
    target["_oracle_full_propensity"] = full_propensity[target_mask]
    target["_oracle_retention_probability"] = retention_probability[target_mask]
    target["_oracle_treated_retention"] = (
        randomized_probability * preferred_probability[target_mask]
    )
    grouping = target.groupby(list(covariates), observed=True, dropna=False)
    target["_oracle_observed_propensity"] = (
        grouping["_oracle_treated_retention"].transform("sum")
        / grouping["_oracle_retention_probability"].transform("sum")
    )
    observed_propensity = target["_oracle_observed_propensity"].to_numpy(
        dtype=float
    )
    full_target_propensity = target["_oracle_full_propensity"].to_numpy(
        dtype=float
    )
    if (
        np.any(full_target_propensity <= 0.0)
        or np.any(full_target_propensity >= 1.0)
        or np.any(observed_propensity <= 0.0)
        or np.any(observed_propensity >= 1.0)
    ):
        raise ValueError("DGP-oracle propensities must lie strictly in (0, 1)")
    log_odds_deviation = np.abs(
        logit(full_target_propensity) - logit(observed_propensity)
    )
    oracle_log_gamma = float(np.max(log_odds_deviation))
    if np.isclose(oracle_log_gamma, 0.0, atol=1e-12, rtol=0.0):
        oracle_log_gamma = 0.0
    gamma = float(np.exp(oracle_log_gamma))
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("DGP-oracle Gamma calibration produced an invalid value")
    return {
        "study": str(study),
        "gamma": gamma,
        "oracle_log_gamma": oracle_log_gamma,
        "oracle_selection_probability_column": str(selection_mechanism),
        "oracle_n_target_idle_rows": int(len(target)),
        "oracle_n_blearner_strata": int(
            target[list(covariates)].drop_duplicates().shape[0]
        ),
        "oracle_full_propensity_min": float(np.min(full_target_propensity)),
        "oracle_full_propensity_max": float(np.max(full_target_propensity)),
        "oracle_observed_propensity_min": float(np.min(observed_propensity)),
        "oracle_observed_propensity_max": float(np.max(observed_propensity)),
        "oracle_gamma_assumption": (
            DGP_ORACLE_GAMMA_ASSUMPTION
            if randomized_probability == 0.5
            else UNBALANCED_ORACLE_GAMMA_ASSUMPTION
        ),
        "oracle_calibration_population": "full_pre_selection_target_rct",
        "oracle_randomized_treatment_probability": randomized_probability,
        "oracle_pattern": str(pattern),
        "oracle_beta": float(beta),
    }


def _score_blearner_widths(
    blearner_bounds: pd.DataFrame,
    naturalness_reference: pd.DataFrame,
    covariates: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match strata and compute unweighted and prevalence-weighted width gaps.

    Args:
        blearner_bounds: B-Learner intervals for one or more studies and Gamma
            values.
        naturalness_reference: Naturalness intervals for matching target
            strata.
        covariates: Matching stratum columns. Defaults to ManyLabs strata.

    Returns:
        Stratum-level matches and aggregate scores for each study/Gamma.
    """

    covariates = list(covariates or Exp16Config().covariates)
    keys = ["study", *covariates]
    width_matches = blearner_bounds.merge(
        naturalness_reference[[*keys, "naturalness_width"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if len(width_matches) != len(blearner_bounds):
        raise ValueError(
            "Naturalness reference does not cover every B-Learner stratum"
        )
    width_matches["stratum_width_gap"] = (
        width_matches["blearner_width"] - width_matches["naturalness_width"]
    )
    width_matches["absolute_stratum_width_gap"] = width_matches[
        "stratum_width_gap"
    ].abs()
    if "n_target_idle_rows" in width_matches:
        weights = pd.to_numeric(
            width_matches["n_target_idle_rows"], errors="coerce"
        )
        if weights.isna().any() or (weights <= 0.0).any():
            raise ValueError("B-Learner target-stratum weights must be positive")
    else:
        weights = pd.Series(1.0, index=width_matches.index)
    width_matches["_target_weight"] = weights.astype(float)
    width_matches["_width_comparison_available"] = (
        pd.to_numeric(
            width_matches["blearner_width"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            width_matches["naturalness_width"],
            errors="coerce",
        ).notna()
    )
    width_matches["_comparison_weight"] = width_matches[
        "_target_weight"
    ].where(width_matches["_width_comparison_available"])
    width_matches["_comparison_blearner_width"] = width_matches[
        "blearner_width"
    ].where(width_matches["_width_comparison_available"])
    width_matches["_weighted_blearner_width"] = (
        width_matches["blearner_width"] * width_matches["_comparison_weight"]
    )
    width_matches["_weighted_naturalness_width"] = (
        width_matches["naturalness_width"]
        * width_matches["_comparison_weight"]
    )
    scores = (
        width_matches.groupby(["study", "gamma"], as_index=False)
        .agg(
            n_width_matched_strata=(
                "_width_comparison_available",
                "sum",
            ),
            mean_absolute_stratum_width_gap=(
                "absolute_stratum_width_gap",
                "mean",
            ),
            mean_blearner_width=("_comparison_blearner_width", "mean"),
            mean_naturalness_width=("naturalness_width", "mean"),
            target_weight=(
                "_comparison_weight",
                lambda values: values.sum(min_count=1),
            ),
            weighted_blearner_width_sum=(
                "_weighted_blearner_width",
                lambda values: values.sum(min_count=1),
            ),
            weighted_naturalness_width_sum=(
                "_weighted_naturalness_width",
                lambda values: values.sum(min_count=1),
            ),
        )
        .sort_values(["study", "gamma"])
        .reset_index(drop=True)
    )
    scores["target_prevalence_weighted_blearner_width"] = (
        scores["weighted_blearner_width_sum"] / scores["target_weight"]
    )
    scores["target_prevalence_weighted_naturalness_width"] = (
        scores["weighted_naturalness_width_sum"] / scores["target_weight"]
    )
    scores["target_prevalence_weighted_width_gap"] = (
        scores["target_prevalence_weighted_blearner_width"]
        - scores["target_prevalence_weighted_naturalness_width"]
    )
    scores["absolute_target_prevalence_weighted_width_gap"] = scores[
        "target_prevalence_weighted_width_gap"
    ].abs()
    scores = scores.drop(
        columns=[
            "target_weight",
            "weighted_blearner_width_sum",
            "weighted_naturalness_width_sum",
        ]
    )
    width_matches = width_matches.drop(
        columns=[
            "_target_weight",
            "_width_comparison_available",
            "_comparison_weight",
            "_comparison_blearner_width",
            "_weighted_blearner_width",
            "_weighted_naturalness_width",
        ]
    )
    return width_matches, scores


def _gamma_scores_are_monotonic(scores: pd.DataFrame) -> bool:
    """Test whether fitted weighted mean width is nondecreasing in Gamma.

    Args:
        scores: Aggregate score rows for one study.

    Returns:
        Whether widths are nondecreasing up to numerical tolerance.
    """

    widths = scores.sort_values("gamma")[
        "target_prevalence_weighted_blearner_width"
    ].to_numpy(dtype=float)
    return bool(np.all(np.diff(widths) >= -1e-10))


def _fit_blearner_binary_search_study_bounds(
    config: Exp18Config,
    study: str,
    target_idle: pd.DataFrame,
    naturalness_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Fit a coarse Gamma grid and refine its width bracket by bisection.

    Search uses signed target-prevalence-weighted width difference. It stops
    without refinement when target width falls outside grid range or fitted
    aggregate B-Learner widths violate monotonicity.

    Args:
        config: Exp18 binary-search configuration.
        study: Study label stored in fit and diagnostic rows.
        target_idle: Target idle sample used by every B-Learner fit.
        naturalness_reference: Naturalness target intervals for this study.

    Returns:
        All coarse and midpoint fits annotated with final search diagnostics.
    """

    if naturalness_reference.empty:
        raise ValueError(f"Naturalness reference has no rows for study {study!r}")
    evaluated = _fit_blearner_study_bounds(
        config,
        study,
        target_idle,
        config.gamma_search_grid,
    )
    evaluated["gamma_search_stage"] = "initial_grid"
    evaluated["gamma_search_iteration"] = 0
    _, scores = _score_blearner_widths(
        evaluated,
        naturalness_reference,
        _comparison_covariates(config),
    )
    monotonic = _gamma_scores_are_monotonic(scores)
    lower_gamma = np.nan
    upper_gamma = np.nan
    iterations = 0

    if not monotonic:
        status = "nonmonotonic_initial_grid"
    else:
        gaps = scores.set_index("gamma")[
            "target_prevalence_weighted_width_gap"
        ].sort_index()
        close = gaps.loc[
            gaps.abs() <= config.gamma_search_width_tolerance
        ]
        below = gaps.loc[gaps < 0.0]
        above = gaps.loc[gaps > 0.0]
        if not close.empty:
            closest_gamma = float(close.abs().sort_values().index[0])
            lower_gamma = upper_gamma = closest_gamma
            status = "initial_width_tolerance_reached"
        elif below.empty:
            lower_gamma = upper_gamma = float(gaps.index.min())
            status = "naturalness_width_below_grid"
        elif above.empty:
            lower_gamma = upper_gamma = float(gaps.index.max())
            status = "naturalness_width_above_grid"
        else:
            lower_gamma = float(below.index.max())
            upper_gamma = float(above.index.min())
            status = "searching"
            while (
                upper_gamma - lower_gamma > config.gamma_search_tolerance
                and iterations < config.gamma_search_max_iterations
            ):
                iterations += 1
                midpoint = (lower_gamma + upper_gamma) / 2.0
                midpoint_bounds = _fit_blearner_study_bounds(
                    config,
                    study,
                    target_idle,
                    (midpoint,),
                )
                midpoint_bounds["gamma_search_stage"] = "binary_search"
                midpoint_bounds["gamma_search_iteration"] = iterations
                evaluated = pd.concat(
                    [evaluated, midpoint_bounds], ignore_index=True
                )
                _, scores = _score_blearner_widths(
                    evaluated,
                    naturalness_reference,
                    _comparison_covariates(config),
                )
                monotonic = _gamma_scores_are_monotonic(scores)
                if not monotonic:
                    status = "nonmonotonic_refinement"
                    break
                midpoint_gap = float(
                    scores.loc[
                        np.isclose(scores["gamma"], midpoint),
                        "target_prevalence_weighted_width_gap",
                    ].iloc[0]
                )
                if (
                    abs(midpoint_gap)
                    <= config.gamma_search_width_tolerance
                ):
                    lower_gamma = upper_gamma = midpoint
                    status = "binary_width_tolerance_reached"
                    break
                if midpoint_gap < 0.0:
                    lower_gamma = midpoint
                else:
                    upper_gamma = midpoint
            if status == "searching":
                status = (
                    "gamma_tolerance_reached"
                    if upper_gamma - lower_gamma
                    <= config.gamma_search_tolerance
                    else "max_iterations_reached"
                )

    evaluated["gamma_search_status"] = status
    evaluated["gamma_search_monotonic"] = monotonic
    evaluated["gamma_search_bracket_lower"] = lower_gamma
    evaluated["gamma_search_bracket_upper"] = upper_gamma
    evaluated["gamma_search_bracket_width"] = upper_gamma - lower_gamma
    evaluated["gamma_search_iterations"] = iterations
    return evaluated


def _fit_blearner_bounds(
    config: Exp18Config,
    environments_by_study: Mapping[str, Mapping[str, pd.DataFrame]],
    naturalness_reference: pd.DataFrame | None = None,
    fitting_data_by_study: Mapping[str, PreparedStudyData] | None = None,
) -> pd.DataFrame:
    """Fit configured B-Learner Gamma values for every requested study.

    Args:
        config: Exp18 configuration.
        environments_by_study: Loaded fitting environments keyed by study.
        naturalness_reference: Naturalness intervals needed by binary search.
        fitting_data_by_study: Full pre-selection RCT rows required by the
            DGP-oracle Gamma strategy.

    Returns:
        Concatenated raw B-Learner bounds across studies and Gamma values.
    """

    if config.gamma == WIDTH_MATCHED_BINARY_GAMMA:
        if naturalness_reference is None:
            raise ValueError(
                "naturalness_reference is required for binary Gamma search"
            )
        frames = [
            _fit_blearner_binary_search_study_bounds(
                config,
                study,
                environments_by_study[study][
                    _study_target_site(config, study)
                ],
                naturalness_reference.loc[
                    naturalness_reference["study"].eq(study)
                ],
            )
            for study in config.studies
        ]
    elif config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA):
        # The DGP oracle calibrates on complete pre-selection rows. The
        # realized oracle uses saved probabilities on selected rows.
        needs_full_rct = config.gamma == DGP_ORACLE_GAMMA
        if needs_full_rct and fitting_data_by_study is None:
            raise ValueError(
                f"gamma={config.gamma!r} requires full pre-selection RCT data"
            )
        frames = []
        for study in config.studies:
            target_site = _study_target_site(config, study)
            target_idle = environments_by_study[study][target_site]
            full_rct = (
                fitting_data_by_study[str(study)].rct
                if needs_full_rct
                else None
            )
            randomized_probability = (
                _study_randomized_treatment_probability(
                    config,
                    full_rct,
                    target_site=target_site,
                )
                if full_rct is not None
                else 0.5
            )
            if config.gamma == DGP_ORACLE_GAMMA:
                calibration = _dgp_oracle_gamma_calibration(
                    full_rct,
                    _comparison_covariates(config),
                    study=str(study),
                    target_site=target_site,
                    pattern=str(config.pattern),
                    beta=float(config.beta),
                    randomized_treatment_probability=randomized_probability,
                    preferred_probability=_known_preferred_probability(
                        config,
                        full_rct,
                    ),
                    selection_mechanism="known_osrct_dgp",
                )
                if config.require_observed_gender:
                    calibration["oracle_calibration_population"] = (
                        "observed_gender_complete_case_pre_selection_target_rct"
                    )
                    calibration["oracle_cohort_filter"] = (
                        OBSERVED_GENDER_COHORT_RULE
                    )
            else:
                calibration = _oracle_gamma_calibration(
                    target_idle,
                    _comparison_covariates(config),
                    study=str(study),
                    randomized_treatment_probability=randomized_probability,
                )
            bounds = _fit_blearner_study_bounds(
                config,
                study,
                target_idle,
                (float(calibration["gamma"]),),
            )
            for column, value in calibration.items():
                if column not in {"study", "gamma"}:
                    bounds[column] = value
            frames.append(bounds)
    else:
        gammas = _blearner_gammas(config)
        target_idle_by_study = {
            str(study): environments_by_study[study][
                _study_target_site(config, study)
            ]
            for study in config.studies
        }
        endpoints = _fit_blearner_gamma_grid(
            config,
            target_idle_by_study,
            gammas,
        )
        frames = [
            _fit_blearner_study_bounds(
                config,
                study,
                target_idle_by_study[str(study)],
                gammas,
                fitted_endpoints=endpoints[str(study)],
            )
            for study in config.studies
        ]
    return pd.concat(frames, ignore_index=True)


def _fit_blearner_gamma_grid(
    config: Exp18Config,
    target_idle_by_study: Mapping[str, pd.DataFrame],
    gammas: Sequence[float],
) -> dict[str, dict[float, BLearnerFitResult]]:
    """Fit every study-by-Gamma B-Learner of one grid across worker processes.

    Each Gamma is an independent refit of the same target sample, so a
    width-matching grid over several studies is a batch of independent fits
    rather than a sequence. Dispatching the whole batch at once keeps every
    configured worker busy and pays the process start-up cost only once per
    grid, instead of once per study.

    Args:
        config: Exp18 configuration supplying the worker counts and the seed
            shared by every fit.
        target_idle_by_study: Target-site idle rows keyed by study.
        gammas: Gamma grid fitted for every study.

    Returns:
        Raw fitted endpoints keyed by study and then by Gamma.
    """

    inputs_by_study = {
        str(study): _blearner_study_inputs(
            config,
            str(study),
            target_idle_by_study[str(study)],
        )
        for study in config.studies
    }
    tasks: list[BLearnerFitTask] = []
    task_keys: dict[int, tuple[str, float]] = {}
    for study in config.studies:
        inputs = inputs_by_study[str(study)]
        for gamma in gammas:
            task_keys[len(tasks)] = (str(study), float(gamma))
            tasks.append(
                BLearnerFitTask(
                    task_id=len(tasks),
                    study=str(study),
                    gamma=float(gamma),
                    x=inputs.x,
                    treatment=inputs.treatment,
                    outcome=inputs.outcome,
                    x_target=inputs.strata_x,
                )
            )
    results = _run_blearner_fit_tasks(tasks, _benchmark_fit_config(config))
    endpoints: dict[str, dict[float, BLearnerFitResult]] = {
        str(study): {} for study in config.studies
    }
    for task_id, (study, gamma) in task_keys.items():
        endpoints[study][gamma] = results[task_id]
    return endpoints


def _run_blearner_fit_tasks(
    tasks: Sequence[BLearnerFitTask],
    config: BenchmarkFitConfig,
) -> dict[int, BLearnerFitResult]:
    """Compatibility wrapper for shared parallel B-Learner execution."""

    return _shared_run_blearner_fit_tasks(tasks, config)


def _select_blearner_bounds(
    config: Exp18Config,
    blearner_bounds: pd.DataFrame,
    naturalness_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select fixed or per-study width-matched B-Learner bounds.

    Args:
        config: Exp18 configuration with requested Gamma strategy.
        blearner_bounds: Bounds for every fitted study and Gamma.
        naturalness_reference: Matching naturalness posterior-mean intervals.

    Returns:
        Selected stratum-level bounds and per-study/Gamma selection scores.
    """

    width_matches, scores = _score_blearner_widths(
        blearner_bounds,
        naturalness_reference,
        _comparison_covariates(config),
    )
    search_columns = [
        "gamma_search_stage",
        "gamma_search_iteration",
        "gamma_search_status",
        "gamma_search_monotonic",
        "gamma_search_bracket_lower",
        "gamma_search_bracket_upper",
        "gamma_search_bracket_width",
        "gamma_search_iterations",
    ]
    present_search_columns = [
        column for column in search_columns if column in width_matches
    ]
    if present_search_columns:
        search_metadata = width_matches.groupby(
            ["study", "gamma"], as_index=False
        )[present_search_columns].first()
        scores = scores.merge(
            search_metadata,
            on=["study", "gamma"],
            how="left",
            validate="one_to_one",
        )
    if config.gamma == WIDTH_MATCHED_GAMMA:
        scores = scores.sort_values(
            ["study", "mean_absolute_stratum_width_gap", "gamma"]
        ).reset_index(drop=True)
        selected_gamma = scores.drop_duplicates("study", keep="first")[
            ["study", "gamma"]
        ]
        selection_metric = "mean_absolute_stratum_width_gap"
    elif config.gamma == WIDTH_MATCHED_BINARY_GAMMA:
        scores = scores.sort_values(
            [
                "study",
                "absolute_target_prevalence_weighted_width_gap",
                "gamma",
            ]
        ).reset_index(drop=True)
        selected_gamma = scores.drop_duplicates("study", keep="first")[
            ["study", "gamma"]
        ]
        selection_metric = "absolute_target_prevalence_weighted_width_gap"
    elif config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA):
        selected_gamma = blearner_bounds[["study", "gamma"]].drop_duplicates()
        if len(selected_gamma) != len(config.studies):
            raise ValueError(
                "Oracle Gamma calibration must produce one value per study"
            )
        oracle_columns = [
            column
            for column in (
                "oracle_log_gamma",
                "oracle_selection_probability_column",
                "oracle_n_target_idle_rows",
                "oracle_n_blearner_strata",
                "oracle_full_propensity_min",
                "oracle_full_propensity_max",
                "oracle_observed_propensity_min",
                "oracle_observed_propensity_max",
                "oracle_gamma_assumption",
                "oracle_calibration_population",
                "oracle_cohort_filter",
                "oracle_randomized_treatment_probability",
                "oracle_pattern",
                "oracle_beta",
            )
            if column in blearner_bounds
        ]
        if oracle_columns:
            oracle_metadata = blearner_bounds.groupby(
                ["study", "gamma"],
                as_index=False,
            )[oracle_columns].first()
            scores = scores.merge(
                oracle_metadata,
                on=["study", "gamma"],
                how="left",
                validate="one_to_one",
            )
        selection_metric = (
            "dgp_oracle_msm_max_log_odds_deviation"
            if config.gamma == DGP_ORACLE_GAMMA
            else "oracle_msm_max_log_odds_deviation"
        )
    else:
        resolved_gamma = _blearner_gammas(config)[0]
        selected_gamma = scores.loc[
            np.isclose(scores["gamma"], resolved_gamma),
            ["study", "gamma"],
        ]
        selection_metric = (
            EXP_BETA_GAMMA
            if config.gamma == EXP_BETA_GAMMA
            else "fixed_gamma"
        )
    selected_gamma = selected_gamma.rename(columns={"gamma": "selected_gamma"})
    if len(selected_gamma) != len(config.studies):
        raise ValueError("B-Learner Gamma selection did not cover every study")
    scores = scores.merge(
        selected_gamma,
        on="study",
        how="left",
        validate="many_to_one",
    )
    scores["selected"] = np.isclose(scores["gamma"], scores["selected_gamma"])
    scores["gamma_strategy"] = (
        config.gamma
        if config.gamma in BLEARNER_GAMMA_STRATEGIES
        else "fixed"
    )
    scores["requested_gamma"] = config.gamma
    scores["selection_metric"] = selection_metric
    selected = width_matches.merge(
        selected_gamma,
        on="study",
        how="inner",
        validate="many_to_one",
    )
    selected = selected.loc[
        np.isclose(selected["gamma"], selected["selected_gamma"])
    ].copy()
    selected_counts = selected.groupby("study", sort=True).size().to_dict()
    reference_counts = (
        naturalness_reference.groupby("study", sort=True).size().to_dict()
    )
    if selected_counts != reference_counts:
        raise ValueError(
            "Selected B-Learner Gamma must retain every naturalness stratum"
        )
    return selected, scores


def _select_source_sites(
    config: Exp18Config,
    study_data: PreparedStudyData | None = None,
) -> tuple[str, ...]:
    """Sample one reproducible source-site set shared by all studies.

    Args:
        config: Exp18 setting controlling target, source count, and seed.
        study_data: Already prepared first-study data. When omitted, load the
            legacy Exp16 setting directly for helper compatibility.

    Returns:
        Sorted sampled source-site labels.
    """

    if study_data is None:
        osrct, rct, _ = load_prepared_data(
            _exp16_config(config, config.studies[0])
        )
    else:
        osrct, rct = study_data.idle, study_data.rct
    eligible = tuple(
        sorted(
            set(osrct["site"].astype(str))
            & set(rct["site"].astype(str))
            - {str(config.target_site)}
        )
    )
    if len(eligible) != config.expected_non_target_sites:
        raise ValueError(
            "Expected "
            f"{config.expected_non_target_sites} matched source sites, found "
            f"{len(eligible)}."
        )
    generator = np.random.default_rng(config.seed)
    return tuple(
        sorted(
            generator.choice(
                eligible,
                size=config.n_source_sites,
                replace=False,
            ).tolist()
        )
    )


def _load_naturalness_final_training_sources(
    config: Exp18Config,
    fitting_data_by_study: Mapping[str, PreparedStudyData],
) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    """Load the exact final-grounder source cohort from naturalness roles.

    The compared posterior-mean naturalness endpoints use the union of proper
    training and policy-tuning sites. Endpoint-calibration sites affect other
    conformal artifacts but not the ``lower_mean``/``upper_mean`` endpoints
    loaded by Exp18.

    Args:
        config: Exp18 configuration pointing to a naturalness result leaf.
        fitting_data_by_study: Matched data used to validate site availability.

    Returns:
        Study-specific selected source sites and their role diagnostics.

    Raises:
        FileNotFoundError: If the adjacent source-role artifact is missing.
        ValueError: If roles are malformed or selected sites lack matched data.
    """

    reference_path = _naturalness_reference_path(config)
    roles_path = reference_path.with_name(
        "source_site_roles" + reference_path.suffix
    )
    if not roles_path.exists():
        raise FileNotFoundError(
            "Naturalness final-training source selection requires "
            f"source_site_roles table beside the reference: {roles_path}"
        )
    roles = _read_table(roles_path)
    required = {"study", "site", "role"}
    missing = required.difference(roles.columns)
    if missing:
        raise ValueError(
            f"Naturalness source roles are missing columns: {sorted(missing)}"
        )
    if "target_site" in roles:
        target_values = roles["target_site"].dropna().astype(str).unique()
        if len(target_values) != 1 or target_values[0] != str(config.target_site):
            raise ValueError(
                "Naturalness source roles do not match requested target: "
                f"{target_values!r}"
            )
    selected = roles.loc[
        roles["study"].astype(str).isin(map(str, config.studies))
        & roles["role"].isin(NATURALNESS_FINAL_TRAINING_ROLES)
    ].copy()
    selected["study"] = selected["study"].astype(str)
    selected["source_site"] = selected["site"].astype(str)
    if selected.duplicated(["study", "source_site"]).any():
        raise ValueError("Naturalness source roles contain duplicate study/site rows")

    source_sites_by_study: dict[str, tuple[str, ...]] = {}
    diagnostics: list[pd.DataFrame] = []
    for study in config.studies:
        study_name = str(study)
        study_roles = selected.loc[selected["study"].eq(study_name)].copy()
        observed_roles = set(study_roles["role"].astype(str))
        if observed_roles != set(NATURALNESS_FINAL_TRAINING_ROLES):
            raise ValueError(
                f"{study_name} naturalness roles lack final-training groups: "
                f"{sorted(observed_roles)}"
            )
        sites = tuple(sorted(study_roles["source_site"].unique()))
        if str(config.target_site) in sites:
            raise ValueError("Naturalness final-training sources include target site")
        study_data = fitting_data_by_study[study_name]
        available = set(study_data.idle["site"].astype(str)) & set(
            study_data.rct["site"].astype(str)
        )
        unavailable = sorted(set(sites).difference(available))
        if unavailable:
            raise ValueError(
                f"{study_name} final-training sources lack matched data: "
                f"{unavailable}"
            )
        source_sites_by_study[study_name] = sites
        study_roles["source_site_policy"] = "naturalness_final_training"
        study_roles["roles_path"] = str(roles_path)
        diagnostics.append(
            study_roles[
                [
                    "study",
                    "source_site",
                    "role",
                    "source_site_policy",
                    "roles_path",
                ]
            ]
        )
    return source_sites_by_study, pd.concat(diagnostics, ignore_index=True)


def _select_source_sites_by_study(
    config: Exp18Config,
    fitting_data_by_study: Mapping[str, PreparedStudyData],
) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    """Resolve configured source cohorts for every comparison panel.

    Args:
        config: Exp18 source-site selection configuration.
        fitting_data_by_study: Matched prepared data for availability checks.

    Returns:
        Study-specific source tuples and source-selection diagnostics.
    """

    if config.source_site_policy == "naturalness_final_training":
        return _load_naturalness_final_training_sources(
            config,
            fitting_data_by_study,
        )
    if config.source_site_policy == "all_non_target":
        source_sites_by_study: dict[str, tuple[str, ...]] = {}
        rows: list[dict[str, Any]] = []
        for study in config.studies:
            study_name = str(study)
            study_data = fitting_data_by_study[study_name]
            target_site = _study_target_site(config, study_name)
            sites = tuple(
                sorted(
                    (
                        set(study_data.idle["site"].astype(str))
                        & set(study_data.rct["site"].astype(str))
                    )
                    - {target_site}
                )
            )
            if not sites:
                raise ValueError(
                    f"{study_name} has no matched non-target source sites"
                )
            if len(sites) != config.expected_non_target_sites:
                raise ValueError(
                    f"{study_name} expected "
                    f"{config.expected_non_target_sites} matched non-target "
                    f"source sites, found {len(sites)}"
                )
            source_sites_by_study[study_name] = sites
            rows.extend(
                {
                    "study": study_name,
                    "source_site": site,
                    "role": "all_non_target",
                    "source_site_policy": "all_non_target",
                    "roles_path": None,
                }
                for site in sites
            )
        return source_sites_by_study, pd.DataFrame(rows)
    shared_sites = _select_source_sites(
        config,
        fitting_data_by_study[str(config.studies[0])],
    )
    source_sites_by_study = {
        str(study): shared_sites for study in config.studies
    }
    diagnostics = pd.DataFrame(
        [
            {
                "study": str(study),
                "source_site": site,
                "role": "random_sample",
                "source_site_policy": "random_sample",
                "roles_path": None,
            }
            for study, sites in source_sites_by_study.items()
            for site in sites
        ]
    )
    return source_sites_by_study, diagnostics


def _load_site_environments(
    config: Exp18Config,
    study: str,
    source_sites: Sequence[str],
    study_data: PreparedStudyData | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Build target idle and one raw pooled selected-source environment.

    All selected source rows form one environment labelled
    ``sources_pooled``. The source rows are matched synthetic/OSRCT idle rows
    when ``source_regime='idle'`` and original source RCT rows when
    ``source_regime='on'``. Consequently, a source site with more rows has
    proportionally more influence on the pooled learner. The returned
    composition table makes that raw-row weighting explicit in saved results.

    Args:
        config: Exp18 target, source-regime, and covariate setting.
        study: Study label recorded in diagnostics.
        source_sites: Selected sites pooled into the source environment.
        study_data: Matching Exp16 rows. When omitted, load Exp16
            directly for helper compatibility.

    Returns:
        Fitting environments, stratum-support diagnostics, and source-pool
        composition.
    """

    if study_data is None:
        osrct, rct, _ = load_prepared_data(_exp16_config(config, study))
    else:
        osrct, rct = study_data.idle, study_data.rct
    covariates = _comparison_covariates(config)
    target_site = _study_target_site(config, study)
    target_data = osrct.loc[
        osrct["site"].astype(str).eq(target_site)
    ].copy()
    source_frame = osrct if config.source_regime == "idle" else rct
    source_data = source_frame.loc[
        source_frame["site"].astype(str).isin(source_sites)
    ].copy()
    available_source_sites = set(source_data["site"].astype(str))
    empty_sites = sorted(
        set(map(str, source_sites)).difference(available_source_sites)
    )
    if target_data.empty:
        empty_sites = [target_site, *empty_sites]
    if empty_sites:
        raise ValueError(
            f"{study} has empty requested target/source sites: {empty_sites}"
        )
    environments = {
        target_site: target_data,
        POOLED_SOURCE_ENVIRONMENT: source_data,
    }
    target_strata = (
        environments[target_site][covariates]
        .drop_duplicates()
        .sort_values(covariates)
        .reset_index(drop=True)
    )
    support_frames: list[pd.DataFrame] = []
    for site, data in environments.items():
        counts = (
            data.groupby(covariates, observed=True)
            .size()
            .rename("n_environment_rows")
            .reset_index()
        )
        support = target_strata.merge(counts, on=covariates, how="left")
        support["n_environment_rows"] = support["n_environment_rows"].fillna(
            0
        ).astype(int)
        support["study"] = study
        support["environment"] = site
        support["environment_regime"] = (
            "idle" if site == target_site else config.source_regime
        )
        support["observed_support"] = support[
            "n_environment_rows"
        ].gt(0)
        support_frames.append(support)
    source_pool_composition = (
        source_data.assign(source_site=source_data["site"].astype(str))
        .groupby("source_site", as_index=False, observed=True)
        .size()
        .rename(columns={"size": "n_source_site_rows"})
        .sort_values("source_site")
        .reset_index(drop=True)
    )
    source_pool_composition["study"] = study
    source_pool_composition["source_regime"] = config.source_regime
    source_pool_composition["source_pool_row_weight"] = (
        source_pool_composition["n_source_site_rows"]
        / source_pool_composition["n_source_site_rows"].sum()
    )
    return (
        environments,
        pd.concat(support_frames, ignore_index=True),
        source_pool_composition,
    )


def _ordered_pairs(
    config: Exp18Config,
    environment_names: Sequence[str],
    non_target_environment_names: Sequence[str],
    target_environment: str | None = None,
) -> list[tuple[str, str]]:
    """Return the configured ordered CB pairs of fitting environments."""

    target = str(target_environment or config.target_site)
    if config.pair_scope == "target_anchored":
        return [
            *((target, name) for name in non_target_environment_names),
            *((name, target) for name in non_target_environment_names),
        ]
    if config.pair_scope == "all_ordered":
        return [
            (left, right)
            for left in environment_names
            for right in environment_names
            if left != right
        ]
    raise ValueError(f"Unsupported pair scope: {config.pair_scope}")


def _environment_arrays(
    data: pd.DataFrame,
    covariates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a site's learner arrays with the CATENets float32 dtype."""

    return (
        _covariate_matrix(data, covariates),
        data[TREATMENT_COLUMN].to_numpy(dtype=np.float32),
        data[OUTCOME_COLUMN].to_numpy(dtype=np.float32),
    )


def _build_study_tasks(
    config: Exp18Config,
    study: str,
    environments: Mapping[str, pd.DataFrame],
    non_target_environment_names: Sequence[str],
    first_task_id: int,
) -> list[FitTask]:
    """Create every independent two-environment WB/CB endpoint task."""

    covariates = _comparison_covariates(config)
    target_site = _study_target_site(config, study)
    target_strata = (
        environments[target_site][covariates]
        .drop_duplicates()
        .sort_values(covariates)
    )
    target_matrix = _covariate_matrix(target_strata, covariates)
    arrays = {
        name: _environment_arrays(data, covariates)
        for name, data in environments.items()
    }
    tasks: list[FitTask] = []
    next_task_id = first_task_id
    for environment, (x, a, y) in arrays.items():
        for endpoint in ENDPOINTS:
            tasks.append(
                FitTask(
                    task_id=next_task_id,
                    study=study,
                    kind="wb",
                    endpoint=endpoint,
                    seed=config.seed + next_task_id,
                    x_left=x,
                    a_left=a,
                    y_left=y,
                    x_target=target_matrix,
                    environment=environment,
                )
            )
            next_task_id += 1
    for left, right in _ordered_pairs(
        config,
        tuple(environments),
        non_target_environment_names,
        target_site,
    ):
        x_left, a_left, y_left = arrays[left]
        x_right, a_right, y_right = arrays[right]
        for method in config.meta_learners:
            for endpoint in ENDPOINTS:
                tasks.append(
                    FitTask(
                        task_id=next_task_id,
                        study=study,
                        kind="cb",
                        endpoint=endpoint,
                        seed=config.seed + next_task_id,
                        x_left=x_left,
                        a_left=a_left,
                        y_left=y_left,
                        x_target=target_matrix,
                        method=method,
                        left_environment=left,
                        right_environment=right,
                        x_right=x_right,
                        a_right=a_right,
                        y_right=y_right,
                    )
                )
                next_task_id += 1
    return tasks


def _pair_support(
    support: pd.DataFrame,
    covariates: Sequence[str],
    left: str,
    right: str,
) -> pd.DataFrame:
    """Join two site-support tables for one ordered CB pair."""

    left_support = support.loc[
        support["environment"].eq(left),
        [*covariates, "n_environment_rows", "environment_regime"],
    ].rename(
        columns={
            "n_environment_rows": "n_left_environment_rows",
            "environment_regime": "left_environment_regime",
        }
    )
    right_support = support.loc[
        support["environment"].eq(right),
        [*covariates, "n_environment_rows", "environment_regime"],
    ].rename(
        columns={
            "n_environment_rows": "n_right_environment_rows",
            "environment_regime": "right_environment_regime",
        }
    )
    result = left_support.merge(
        right_support,
        on=list(covariates),
        how="inner",
        validate="one_to_one",
    )
    result["pair_observed_support"] = (
        result["n_left_environment_rows"].gt(0)
        & result["n_right_environment_rows"].gt(0)
    )
    return result


def _finalize_binary_cate_intersection(
    lower_raw: pd.Series,
    upper_raw: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coherent binary-CATE endpoints and empty-intersection flags.

    Finite-sample endpoint estimates can yield an empty max-lower/min-upper
    intersection even when every theoretical candidate is valid.  Such a row
    does not justify swapping learned endpoints.  It instead receives the
    valid binary-CATE fallback ``[-1, 1]`` and an explicit diagnostic flag.

    Args:
        lower_raw: Unclipped maximum lower endpoint for each target stratum.
        upper_raw: Unclipped minimum upper endpoint for each target stratum.

    Returns:
        Final lower endpoints, final upper endpoints, and Boolean flags for
        empty estimated intersections.
    """

    lower_values = pd.to_numeric(lower_raw, errors="coerce").to_numpy(float)
    upper_values = pd.to_numeric(upper_raw, errors="coerce").to_numpy(float)
    empty_intersection = (~np.isfinite(lower_values)) | (~np.isfinite(upper_values))
    empty_intersection |= lower_values > upper_values
    lower = np.where(empty_intersection, -1.0, np.clip(lower_values, -1.0, 1.0))
    upper = np.where(empty_intersection, 1.0, np.clip(upper_values, -1.0, 1.0))
    return lower, upper, empty_intersection


def _assemble_study_bounds(
    config: Exp18Config,
    study: str,
    environments: Mapping[str, pd.DataFrame],
    support: pd.DataFrame,
    tasks: Sequence[FitTask],
    fit_results: Mapping[int, FitResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble parallel fit outputs and intersect paper-style candidates."""

    covariates = _comparison_covariates(config)
    target_site = _study_target_site(config, study)
    target_strata = (
        environments[target_site][covariates]
        .drop_duplicates()
        .sort_values(covariates)
        .reset_index(drop=True)
    )
    study_tasks = [task for task in tasks if task.study == study]
    wb_frames: list[pd.DataFrame] = []
    for environment in environments:
        frame = target_strata.copy()
        frame["study"] = study
        frame["environment"] = environment
        for endpoint in ENDPOINTS:
            task = next(
                item
                for item in study_tasks
                if item.kind == "wb"
                and item.environment == environment
                and item.endpoint == endpoint
            )
            frame[f"wb_{endpoint}"] = fit_results[task.task_id].predictions
        frame["wb_interval_inverted"] = frame["wb_lower"] > frame["wb_upper"]
        wb_frames.append(frame)
    wb_bounds = pd.concat(wb_frames, ignore_index=True).merge(
        support[
            [
                "study",
                "environment",
                *covariates,
                "n_environment_rows",
                "environment_regime",
                "observed_support",
            ]
        ],
        on=["study", "environment", *covariates],
        how="left",
        validate="one_to_one",
    )
    wb_bounds["eligible_wb_candidate"] = (
        wb_bounds["observed_support"]
        & wb_bounds["wb_interval_inverted"].eq(False)
    )

    pair_frames: list[pd.DataFrame] = []
    pairs = _ordered_pairs(
        config,
        tuple(environments),
        tuple(site for site in environments if site != target_site),
        target_site,
    )
    for left, right in pairs:
        support_frame = _pair_support(support, covariates, left, right)
        for method in config.meta_learners:
            frame = target_strata.copy()
            frame["study"] = study
            frame["method"] = method
            frame["left_environment"] = left
            frame["right_environment"] = right
            for endpoint in ENDPOINTS:
                task = next(
                    item
                    for item in study_tasks
                    if item.kind == "cb"
                    and item.method == method
                    and item.endpoint == endpoint
                    and item.left_environment == left
                    and item.right_environment == right
                )
                frame[f"{method}_{endpoint}_raw"] = fit_results[
                    task.task_id
                ].predictions
                frame[f"{method}_{endpoint}"] = np.clip(
                    frame[f"{method}_{endpoint}_raw"],
                    -1.0,
                    1.0,
                )
            frame[f"{method}_interval_inverted"] = (
                frame[f"{method}_lower"] > frame[f"{method}_upper"]
            )
            pair_frames.append(
                frame.merge(
                    support_frame,
                    on=covariates,
                    how="left",
                    validate="one_to_one",
                )
            )
    cb_pair_bounds = pd.concat(pair_frames, ignore_index=True)
    cb_pair_bounds["candidate_interval_inverted"] = False
    for method in config.meta_learners:
        method_mask = cb_pair_bounds["method"].eq(method)
        cb_pair_bounds.loc[method_mask, "candidate_interval_inverted"] = (
            cb_pair_bounds.loc[
                method_mask,
                f"{method}_interval_inverted",
            ]
            .astype(bool)
            .to_numpy()
        )
    cb_pair_bounds["candidate_interval_inverted"] = cb_pair_bounds[
        "candidate_interval_inverted"
    ].astype(bool)
    cb_pair_bounds["eligible_cb_candidate"] = (
        cb_pair_bounds["pair_observed_support"]
        & cb_pair_bounds["candidate_interval_inverted"].eq(False)
    )

    wb_eligible = wb_bounds.loc[wb_bounds["eligible_wb_candidate"]].copy()
    wb_lower = (
        wb_eligible.groupby(covariates, as_index=False)["wb_lower"]
        .max()
        .rename(columns={"wb_lower": "wb_lower_intersection"})
    )
    wb_upper = (
        wb_eligible.groupby(covariates, as_index=False)["wb_upper"]
        .min()
        .rename(columns={"wb_upper": "wb_upper_intersection"})
    )
    wb_count = (
        wb_eligible.groupby(covariates, as_index=False)
        .size()
        .rename(columns={"size": "n_eligible_wb_environments"})
    )
    final = (
        target_strata.merge(
            wb_lower,
            on=covariates,
            how="left",
            validate="one_to_one",
        )
        .merge(
            wb_upper,
            on=covariates,
            how="left",
            validate="one_to_one",
        )
        .merge(
            wb_count,
            on=covariates,
            how="left",
            validate="one_to_one",
        )
    )
    final["study"] = study
    final["pattern"] = (
        config.pattern if config.data_source == EXP16_REFERENCE else None
    )
    final["beta"] = config.beta
    final["target_site"] = target_site
    final["n_source_sites"] = config.n_source_sites
    final["pair_scope"] = config.pair_scope
    for method in config.meta_learners:
        method_pairs = cb_pair_bounds.loc[
            cb_pair_bounds["method"].eq(method)
        ].copy()
        eligible = method_pairs.loc[method_pairs["eligible_cb_candidate"]]
        cb_lower = (
            eligible.groupby(covariates, as_index=False)[f"{method}_lower"]
            .max()
            .rename(columns={f"{method}_lower": f"{method}_lower_intersection"})
        )
        cb_upper = (
            eligible.groupby(covariates, as_index=False)[f"{method}_upper"]
            .min()
            .rename(columns={f"{method}_upper": f"{method}_upper_intersection"})
        )
        cb_count = (
            eligible.groupby(covariates, as_index=False)
            .size()
            .rename(columns={"size": f"{method}_n_eligible_cb_pairs"})
        )
        cb_inverted = (
            method_pairs.loc[method_pairs["pair_observed_support"]]
            .groupby(covariates, as_index=False)["candidate_interval_inverted"]
            .sum()
            .rename(
                columns={
                    "candidate_interval_inverted": f"{method}_n_inverted_cb_pairs"
                }
            )
        )
        final = (
            final.merge(cb_lower, on=covariates, how="left", validate="one_to_one")
            .merge(cb_upper, on=covariates, how="left", validate="one_to_one")
            .merge(cb_count, on=covariates, how="left", validate="one_to_one")
            .merge(cb_inverted, on=covariates, how="left", validate="one_to_one")
        )
        final[f"{method}_lower_raw"] = np.maximum(
            final["wb_lower_intersection"].fillna(-np.inf),
            final[f"{method}_lower_intersection"].fillna(-np.inf),
        )
        final[f"{method}_upper_raw"] = np.minimum(
            final["wb_upper_intersection"].fillna(np.inf),
            final[f"{method}_upper_intersection"].fillna(np.inf),
        )
        lower, upper, empty_intersection = _finalize_binary_cate_intersection(
            final[f"{method}_lower_raw"],
            final[f"{method}_upper_raw"],
        )
        final[f"{method}_lower"] = lower
        final[f"{method}_upper"] = upper
        final[f"{method}_raw_intersection_inverted"] = empty_intersection
        final[f"{method}_used_full_range_fallback"] = empty_intersection
        final[f"{method}_interval_inverted"] = (
            final[f"{method}_lower"] > final[f"{method}_upper"]
        )
        final[f"{method}_width"] = np.where(
            final[f"{method}_interval_inverted"],
            np.nan,
            final[f"{method}_upper"] - final[f"{method}_lower"],
        )
    return final, cb_pair_bounds, wb_bounds


def fit_prepared_study_bounds(
    config: Exp18Config,
    *,
    study: str,
    study_data: PreparedStudyData,
    source_sites: Sequence[str],
    blearner_gammas: Sequence[float] = (),
) -> dict[str, pd.DataFrame]:
    """Fit Exp18 baselines directly on caller-prepared site tables.

    This adapter exposes the paper-faithful WB/CB and B-Learner implementation
    without coupling a synthetic or external benchmark to Exp16 artifact
    loaders. The target environment is determined by the same
    Exp18 convention as the configured data source, and target randomized
    rows remain unavailable to every fitted learner.

    Args:
        config: Exp18 learner and environment configuration. Set
            ``comparison_covariates`` when the prepared data use a custom
            reporting state.
        study: Study label.
        study_data: Prepared observational and randomized rows.
        source_sites: Source sites pooled into the configured source regime.
        blearner_gammas: Optional fixed Gamma values to fit on target idle
            rows. An empty sequence skips B-Learner fitting.

    Returns:
        Named tables for final CB bounds, pair-level CB candidates, WB
        candidates, optional B-Learner bounds, fit diagnostics, support, and
        source-pool composition.
    """

    if not source_sites:
        raise ValueError("source_sites must not be empty")
    local_config = replace(
        config,
        n_source_sites=len(tuple(source_sites)),
    )
    environments, support, composition = _load_site_environments(
        local_config,
        str(study),
        tuple(map(str, source_sites)),
        study_data,
    )
    target_site = _study_target_site(local_config, str(study))
    tasks = _build_study_tasks(
        local_config,
        str(study),
        environments,
        tuple(name for name in environments if name != target_site),
        0,
    )
    fit_results, fit_diagnostics = _run_fit_tasks(tasks, local_config)
    cb_final, cb_pairs, wb_bounds = _assemble_study_bounds(
        local_config,
        str(study),
        environments,
        support,
        tasks,
        fit_results,
    )
    if blearner_gammas:
        blearner_bounds = _fit_blearner_study_bounds(
            local_config,
            str(study),
            environments[target_site],
            tuple(float(value) for value in blearner_gammas),
        )
    else:
        blearner_bounds = pd.DataFrame()
    return {
        "cb_final_bounds": cb_final,
        "cb_pair_bounds": cb_pairs,
        "wb_bounds": wb_bounds,
        "blearner_bounds": blearner_bounds,
        "fit_diagnostics": fit_diagnostics,
        "support": support,
        "source_pool_composition": composition,
    }


def _build_comparison(
    cb_final_bounds: pd.DataFrame,
    naturalness_reference: pd.DataFrame,
    selected_blearner_bounds: pd.DataFrame,
    meta_learners: Sequence[str] = DEFAULT_META_LEARNERS,
    covariates: Sequence[str] | None = None,
    naturalness_method_label: str = "naturalness_xgboost_posterior_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge CB, naturalness, and selected B-Learner target bounds.

    RCT point coverage is available for Exp16 references.

    Args:
        cb_final_bounds: Intersected WB/CB intervals for every selected method.
        naturalness_reference: Matched naturalness and evaluation rows.
        selected_blearner_bounds: Selected B-Learner interval per target row.
        meta_learners: CB method names whose coverage and summary fields should
            be computed.
        covariates: Matching stratum columns. Defaults to ManyLabs strata.
        naturalness_method_label: Method label saved in summary rows.

    Returns:
        Merged target-stratum comparison and per-study method summary.
    """

    covariates = list(covariates or Exp16Config().covariates)
    naturalness_reference = naturalness_reference.copy()
    for column in (
        "naturalness_envelope_lower",
        "naturalness_envelope_upper",
        "naturalness_envelope_width",
    ):
        if column not in naturalness_reference:
            naturalness_reference[column] = np.nan
    comparison = cb_final_bounds.merge(
        naturalness_reference[
            [
                "study",
                *covariates,
                "naturalness_lower",
                "naturalness_upper",
                "naturalness_width",
                "naturalness_envelope_lower",
                "naturalness_envelope_upper",
                "naturalness_envelope_width",
                "n_target_idle_rows",
                "rct_cate",
                "rct_cate_ci90_lower",
                "rct_cate_ci90_upper",
                "rct_cate_available",
                "blackbox_reference_cate",
                "naturalness_reference_experiment",
                "lp_posterior_prob_optimal",
            ]
        ],
        on=["study", *covariates],
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(naturalness_reference):
        raise ValueError(
            "Exp18 and naturalness target-stratum rows do not match exactly"
        )
    blearner_columns = [
        "study",
        *covariates,
        "gamma",
        "blearner_lower",
        "blearner_upper",
        "blearner_width",
        "outside_binary_cate_support",
        "fit_runtime_seconds",
        "absolute_stratum_width_gap",
    ]
    comparison = comparison.merge(
        selected_blearner_bounds[blearner_columns].rename(
            columns={"gamma": "blearner_gamma"}
        ),
        on=["study", *covariates],
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(naturalness_reference):
        raise ValueError(
            "Selected B-Learner and naturalness target strata do not match exactly"
        )
    for method in meta_learners:
        comparison[f"{method}_covers_rct_cate"] = (
            ~comparison[f"{method}_interval_inverted"]
            & comparison["rct_cate_available"]
            & comparison["rct_cate"].ge(comparison[f"{method}_lower"])
            & comparison["rct_cate"].le(comparison[f"{method}_upper"])
        )
    comparison["blearner_covers_rct_cate"] = (
        comparison["rct_cate_available"]
        & comparison["rct_cate"].ge(comparison["blearner_lower"])
        & comparison["rct_cate"].le(comparison["blearner_upper"])
    )
    comparison["blackbox_cate_available"] = pd.to_numeric(
        comparison["blackbox_reference_cate"], errors="coerce"
    ).notna()
    comparison["naturalness_covers_blackbox_cate"] = (
        comparison["blackbox_cate_available"]
        & comparison["blackbox_reference_cate"].ge(
            comparison["naturalness_lower"]
        )
        & comparison["blackbox_reference_cate"].le(
            comparison["naturalness_upper"]
        )
    )
    comparison["blearner_covers_blackbox_cate"] = (
        comparison["blackbox_cate_available"]
        & comparison["blackbox_reference_cate"].ge(
            comparison["blearner_lower"]
        )
        & comparison["blackbox_reference_cate"].le(
            comparison["blearner_upper"]
        )
    )
    for method in meta_learners:
        comparison[f"{method}_covers_blackbox_cate"] = (
            ~comparison[f"{method}_interval_inverted"]
            & comparison["blackbox_cate_available"]
            & comparison["blackbox_reference_cate"].ge(
                comparison[f"{method}_lower"]
            )
            & comparison["blackbox_reference_cate"].le(
                comparison[f"{method}_upper"]
            )
        )

    summary_rows: list[dict[str, Any]] = []
    for study, rows in comparison.groupby("study", sort=True):
        labeled = rows.loc[rows["rct_cate_available"]]
        blackbox_labeled = rows.loc[rows["blackbox_cate_available"]]
        summary_rows.append(
            {
                "study": study,
                "method": naturalness_method_label,
                "n_target_strata": len(rows),
                "n_rct_labeled_strata": len(labeled),
                "n_blackbox_labeled_strata": len(blackbox_labeled),
                "n_inverted_intervals": 0,
                "n_raw_empty_intersections": 0,
                "n_full_range_fallbacks": 0,
                "target_prevalence_weighted_width": np.average(
                    rows["naturalness_width"],
                    weights=rows["n_target_idle_rows"],
                ),
                "rct_point_coverage": (
                    labeled["rct_cate"].ge(labeled["naturalness_lower"])
                    & labeled["rct_cate"].le(labeled["naturalness_upper"])
                ).mean(),
                "blackbox_point_coverage": (
                    blackbox_labeled[
                        "naturalness_covers_blackbox_cate"
                    ].mean()
                    if not blackbox_labeled.empty
                    else np.nan
                ),
                "mean_lp_posterior_prob_optimal": rows[
                    "lp_posterior_prob_optimal"
                ].mean(),
            }
        )
        summary_rows.append(
            {
                "study": study,
                "method": "blearner",
                "n_target_strata": len(rows),
                "n_rct_labeled_strata": len(labeled),
                "n_blackbox_labeled_strata": len(blackbox_labeled),
                "n_inverted_intervals": 0,
                "n_raw_empty_intersections": 0,
                "n_full_range_fallbacks": 0,
                "target_prevalence_weighted_width": np.average(
                    rows["blearner_width"],
                    weights=rows["n_target_idle_rows"],
                ),
                "rct_point_coverage": (
                    labeled["blearner_covers_rct_cate"].mean()
                    if not labeled.empty
                    else np.nan
                ),
                "blackbox_point_coverage": (
                    blackbox_labeled[
                        "blearner_covers_blackbox_cate"
                    ].mean()
                    if not blackbox_labeled.empty
                    else np.nan
                ),
                "mean_lp_posterior_prob_optimal": np.nan,
            }
        )
        for method in meta_learners:
            valid = rows.loc[~rows[f"{method}_interval_inverted"]]
            labeled_valid = valid.loc[valid["rct_cate_available"]]
            blackbox_labeled_valid = valid.loc[
                valid["blackbox_cate_available"]
            ]
            summary_rows.append(
                {
                    "study": study,
                    "method": method,
                    "n_target_strata": len(rows),
                    "n_rct_labeled_strata": len(labeled),
                    "n_blackbox_labeled_strata": len(blackbox_labeled),
                    "n_inverted_intervals": int(
                        rows[f"{method}_interval_inverted"].sum()
                    ),
                    "n_raw_empty_intersections": int(
                        rows[f"{method}_raw_intersection_inverted"].sum()
                    ),
                    "n_full_range_fallbacks": int(
                        rows[f"{method}_used_full_range_fallback"].sum()
                    ),
                    "target_prevalence_weighted_width": (
                        np.average(
                            valid[f"{method}_width"],
                            weights=valid["n_target_idle_rows"],
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "rct_point_coverage": (
                        labeled_valid[f"{method}_covers_rct_cate"].mean()
                        if not labeled_valid.empty
                        else np.nan
                    ),
                    "blackbox_point_coverage": (
                        blackbox_labeled_valid[
                            f"{method}_covers_blackbox_cate"
                        ].mean()
                        if not blackbox_labeled_valid.empty
                        else np.nan
                    ),
                    "mean_lp_posterior_prob_optimal": np.nan,
                }
            )
    return comparison, pd.DataFrame(summary_rows)


def _make_comparison_plot(
    comparison: pd.DataFrame,
    config: Exp18Config,
    output_path: Path,
    *,
    horizontal: bool = False,
    truth: Literal["rct", "blackbox"] = "rct",
    naturalness_envelope: bool = False,
    figure_title: str | None = None,
) -> None:
    """Plot Exp18 intervals against an RCT or model-implied CATE reference.

    Args:
        comparison: Final CB and naturalness intervals with held-out RCT CATEs.
        config: Exp18 configuration supplying the preferred study order.
        output_path: Destination PNG path.
        horizontal: Whether to place study panels side by side with CATE on
            the x-axis. The default stacks panels with CATE on the y-axis.
        truth: Reference point to plot. ``"rct"`` adds held-out RCT CATEs and
            their 90% Wald intervals; ``"blackbox"`` adds the benchmark's
            model-implied CATE points without sampling intervals.
        naturalness_envelope: Whether to draw the Naturalness posterior
            endpoint envelope behind its posterior-mean interval.
        figure_title: Optional figure-level title. Horizontal callers using a
            title receive extra header space above the shared legend.

    Raises:
        ValueError: If ``truth`` is unsupported or blackbox CATEs are absent.
    """

    if truth not in ("rct", "blackbox"):
        raise ValueError("truth must be rct or blackbox")
    if truth == "blackbox" and (
        "blackbox_reference_cate" not in comparison
        or not pd.to_numeric(
            comparison["blackbox_reference_cate"], errors="coerce"
        ).notna().any()
    ):
        raise ValueError(
            "Model-implied comparison plot requires reference CATE values"
        )
    envelope_columns = {
        "naturalness_envelope_lower",
        "naturalness_envelope_upper",
    }
    if naturalness_envelope and not envelope_columns.issubset(comparison.columns):
        raise ValueError(
            "Naturalness envelope plot requires posterior envelope endpoints"
        )

    covariates = _comparison_covariates(config)
    plotted, stratum_columns, stratum_order = _prepare_study_panel_plot_data(
        comparison,
        covariates,
    )
    stratum_group_spacing = (
        HORIZONTAL_STRATUM_GROUP_SPACING
        if horizontal
        else STRATUM_GROUP_SPACING
    )
    plotted["_plot_x"] = plotted["_plot_x"] * stratum_group_spacing
    present_studies = list(dict.fromkeys(plotted["study"].astype(str)))
    studies = [study for study in config.studies if study in present_studies]
    studies.extend(study for study in present_studies if study not in studies)
    plotted_methods = [*config.meta_learners, "naturalness", "blearner"]
    model_implied_label = "Model-implied blackbox CATE"
    if horizontal:
        # Reserve the center lane for the RCT confidence interval and point.
        # Methods occupy the remaining evenly spaced lanes above and below it.
        lower_lane_count = len(plotted_methods) // 2
        method_lanes = np.arange(len(plotted_methods)) - lower_lane_count
        method_lanes[method_lanes >= 0] += 1
        offsets = method_lanes * HORIZONTAL_REFERENCE_LANE_SPACING
    else:
        offsets = np.linspace(-0.40, 0.40, num=len(plotted_methods))
    specifications = [
        (
            method,
            (
                META_LEARNER_LABELS[method]
                if method in META_LEARNER_LABELS
                else "Naturalness" if method == "naturalness" else "B-Learner"
            ),
            (
                META_LEARNER_COLORS[method]
                if method in META_LEARNER_COLORS
                else "tab:purple" if method == "naturalness" else "tab:red"
            ),
            float(offset),
        )
        for method, offset in zip(plotted_methods, offsets)
    ]
    if horizontal:
        figure, axes = plt.subplots(
            1,
            len(studies),
            figsize=(
                max(9.0, 5.2 * len(studies)),
                max(4.0, 0.64 * max(1, len(stratum_order)) + 1.8),
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        panel_axes = axes[0, :]
    else:
        figure, axes = plt.subplots(
            len(studies),
            1,
            figsize=(9.0, max(3.8, 3.4 * len(studies) - 0.4)),
            squeeze=False,
        )
        panel_axes = axes[:, 0]
    for axis, study in zip(panel_axes, studies):
        rows = (
            plotted.loc[plotted["study"].astype(str).eq(study)]
            .sort_values("_plot_x")
            .reset_index(drop=True)
        )
        stratum_positions = rows["_plot_x"].to_numpy(dtype=float)
        for method, label, color, offset in specifications:
            lower, upper = (
                ("naturalness_lower", "naturalness_upper")
                if method == "naturalness"
                else (
                    ("blearner_lower", "blearner_upper")
                    if method == "blearner"
                    else (f"{method}_lower", f"{method}_upper")
                )
            )
            valid = rows[[lower, upper]].notna().all(axis=1)
            if method in config.meta_learners:
                valid &= ~rows[f"{method}_interval_inverted"]
            positions = stratum_positions[valid] + offset
            if method == "naturalness" and naturalness_envelope:
                envelope_valid = rows[
                    [
                        "naturalness_envelope_lower",
                        "naturalness_envelope_upper",
                    ]
                ].notna().all(axis=1)
                envelope_positions = (
                    stratum_positions[envelope_valid] + offset
                )
                if horizontal:
                    axis.hlines(
                        envelope_positions,
                        rows.loc[
                            envelope_valid,
                            "naturalness_envelope_lower",
                        ],
                        rows.loc[
                            envelope_valid,
                            "naturalness_envelope_upper",
                        ],
                        color="#d8b4fe",
                        alpha=0.55,
                        linewidth=10.0,
                        label="Naturalness posterior envelope",
                        zorder=0,
                    )
                else:
                    axis.vlines(
                        envelope_positions,
                        rows.loc[
                            envelope_valid,
                            "naturalness_envelope_lower",
                        ],
                        rows.loc[
                            envelope_valid,
                            "naturalness_envelope_upper",
                        ],
                        color="#d8b4fe",
                        alpha=0.55,
                        linewidth=10.0,
                        label="Naturalness posterior envelope",
                        zorder=0,
                    )
            if horizontal:
                axis.hlines(
                    positions,
                    rows.loc[valid, lower],
                    rows.loc[valid, upper],
                    color=color,
                    alpha=0.60,
                    linewidth=5.0,
                    label=label,
                    zorder=1,
                )
            else:
                axis.vlines(
                    positions,
                    rows.loc[valid, lower],
                    rows.loc[valid, upper],
                    color=color,
                    alpha=0.60,
                    linewidth=5.0,
                    label=label,
                    zorder=1,
                )
            point_identified = valid & np.isclose(
                rows[lower],
                rows[upper],
                atol=1e-12,
                rtol=0.0,
            )
            if method == "blearner" and point_identified.any():
                point_positions = stratum_positions[point_identified] + offset
                point_estimates = rows.loc[point_identified, lower]
                if horizontal:
                    axis.scatter(
                        point_estimates,
                        point_positions,
                        color=color,
                        marker="o",
                        s=34,
                        label="_nolegend_",
                        zorder=2,
                    )
                else:
                    axis.scatter(
                        point_positions,
                        point_estimates,
                        color=color,
                        marker="o",
                        s=34,
                        label="_nolegend_",
                        zorder=2,
                    )
            if method in config.meta_learners and (~valid).any():
                invalid_positions = stratum_positions[~valid] + offset
                if horizontal:
                    axis.scatter(
                        np.zeros((~valid).sum()),
                        invalid_positions,
                        color=color,
                        marker="x",
                        s=40,
                        label=label,
                        zorder=2,
                    )
                else:
                    axis.scatter(
                        invalid_positions,
                        np.zeros((~valid).sum()),
                        color=color,
                        marker="x",
                        s=40,
                        label=label,
                        zorder=2,
                    )
        if truth == "rct":
            labeled = rows["rct_cate_available"] & rows[
                ["rct_cate", "rct_cate_ci90_lower", "rct_cate_ci90_upper"]
            ].notna().all(axis=1)
            truth_column = "rct_cate"
            truth_label = "Held-out RCT CATE"
        else:
            labeled = pd.to_numeric(
                rows["blackbox_reference_cate"], errors="coerce"
            ).notna()
            truth_column = "blackbox_reference_cate"
            truth_label = model_implied_label
        truth_positions = stratum_positions[labeled]
        if horizontal:
            if truth == "rct":
                axis.hlines(
                    truth_positions,
                    rows.loc[labeled, "rct_cate_ci90_lower"],
                    rows.loc[labeled, "rct_cate_ci90_upper"],
                    color="tab:orange",
                    linewidth=2.5,
                    label="RCT CATE 90% Wald CI",
                    zorder=2,
                )
            axis.scatter(
                rows.loc[labeled, truth_column],
                truth_positions,
                color="black",
                s=34,
                label=truth_label,
                zorder=3,
            )
            axis.axvline(0.0, color="grey", linewidth=0.8)
            axis.set_xlim(-1.0, 1.0)
        else:
            if truth == "rct":
                axis.vlines(
                    truth_positions,
                    rows.loc[labeled, "rct_cate_ci90_lower"],
                    rows.loc[labeled, "rct_cate_ci90_upper"],
                    color="tab:orange",
                    linewidth=2.5,
                    label="RCT CATE 90% Wald CI",
                    zorder=2,
                )
            axis.scatter(
                truth_positions,
                rows.loc[labeled, truth_column],
                color="black",
                s=34,
                label=truth_label,
                zorder=3,
            )
            axis.axhline(0.0, color="grey", linewidth=0.8)
            axis.set_ylim(-1.0, 1.0)
        axis.set_title(str(study), fontweight="bold")
        if not horizontal:
            axis.set_ylabel("CATE")
    legend_order = [
        *(META_LEARNER_LABELS[method] for method in config.meta_learners),
    ]
    if naturalness_envelope:
        legend_order.append("Naturalness posterior envelope")
    legend_order.extend(("Naturalness", "B-Learner"))
    if truth == "rct":
        legend_order.extend(("RCT CATE 90% Wald CI", "Held-out RCT CATE"))
    else:
        legend_order.append(model_implied_label)
    if horizontal:
        horizontal_legend_order = legend_order.copy()
        if truth == "rct":
            horizontal_legend_order[-2:] = (
                "Held-out RCT CATE",
                "RCT CATE 90% Wald CI",
            )
        _set_horizontal_study_panel_axis_labels(
            figure,
            axes,
            stratum_columns=stratum_columns,
            stratum_order=stratum_order,
            legend_order=horizontal_legend_order,
            legend_ncol=min(3, len(horizontal_legend_order)),
        )
        if figure.legends:
            for text in figure.legends[0].get_texts():
                text.set_fontsize(HORIZONTAL_LEGEND_FONTSIZE)
            if figure_title is not None:
                figure.legends[0].set_bbox_to_anchor((0.5, 0.945))
        figure._supxlabel.set_fontsize(HORIZONTAL_AXIS_LABEL_FONTSIZE)
        panel_axes[0].yaxis.label.set_size(HORIZONTAL_AXIS_LABEL_FONTSIZE)
        if stratum_order:
            ticks = np.arange(len(stratum_order)) * stratum_group_spacing
            for axis in panel_axes:
                axis.set_ylim(-0.5, ticks[-1] + 0.5)
                axis.set_yticks(ticks)
            panel_axes[0].set_yticklabels(
                [_format_stratum_tick(stratum) for stratum in stratum_order]
            )
        if figure_title is not None:
            figure.suptitle(figure_title, fontweight="bold", y=0.995)
        figure.tight_layout(
            rect=(0.0, 0.075, 1.0, 0.86 if figure_title is not None else 0.90)
        )
    else:
        _set_study_panel_axis_labels(
            figure,
            axes,
            stratum_columns=stratum_columns,
            stratum_order=stratum_order,
            legend_order=legend_order,
            legend_ncol=min(4, len(legend_order)),
        )
        if stratum_order:
            ticks = np.arange(len(stratum_order)) * stratum_group_spacing
            for axis in panel_axes:
                axis.set_xlim(-0.5, ticks[-1] + 0.5)
                axis.set_xticks(ticks)
                axis.set_xticklabels(
                    [_format_stratum_tick(stratum) for stratum in stratum_order],
                    rotation=45,
                    ha="right",
                )
        figure.tight_layout(rect=(0.0, 0.095, 1.0, 0.90))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def make_comparison_plot(
    comparison: pd.DataFrame,
    config: Exp18Config,
    output_path: Path,
    *,
    horizontal: bool = False,
    truth: Literal["rct", "blackbox"] = "rct",
    naturalness_envelope: bool = False,
    figure_title: str | None = None,
) -> None:
    """Write an Exp18 comparison plot for a reusable benchmark result."""

    _make_comparison_plot(
        comparison,
        config,
        output_path,
        horizontal=horizontal,
        truth=truth,
        naturalness_envelope=naturalness_envelope,
        figure_title=figure_title,
    )


def _fit_and_select_blearner_bounds(
    config: Exp18Config,
    environments_by_study: Mapping[str, Mapping[str, pd.DataFrame]],
    naturalness_reference: pd.DataFrame,
    fitting_data_by_study: Mapping[str, PreparedStudyData],
) -> dict[str, pd.DataFrame]:
    """Fit B-Learner bounds and apply the configured Gamma selection rule.

    Args:
        config: Resolved Exp18 configuration.
        environments_by_study: Target-idle environments keyed by study.
        naturalness_reference: Matched Naturalness rows used for Gamma
            selection and stratum validation.
        fitting_data_by_study: Prepared data needed by DGP-oracle Gamma.

    Returns:
        Raw and selected B-Learner bounds, Gamma-selection diagnostics, and
        optional oracle-Gamma calibration rows.
    """

    blearner_bounds = _fit_blearner_bounds(
        config,
        environments_by_study,
        naturalness_reference,
        fitting_data_by_study,
    )
    oracle_gamma_columns = [
        "study",
        "gamma",
        "oracle_log_gamma",
        "oracle_selection_probability_column",
        "oracle_n_target_idle_rows",
        "oracle_n_blearner_strata",
        "oracle_full_propensity_min",
        "oracle_full_propensity_max",
        "oracle_observed_propensity_min",
        "oracle_observed_propensity_max",
        "oracle_gamma_assumption",
        "oracle_calibration_population",
        "oracle_cohort_filter",
        "oracle_randomized_treatment_probability",
        "oracle_pattern",
        "oracle_beta",
    ]
    available_oracle_gamma_columns = [
        column for column in oracle_gamma_columns if column in blearner_bounds
    ]
    if config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA):
        oracle_gamma_calibration = (
            blearner_bounds[available_oracle_gamma_columns]
            .drop_duplicates()
            .sort_values("study")
            .reset_index(drop=True)
        )
    else:
        oracle_gamma_calibration = pd.DataFrame(columns=oracle_gamma_columns)
    blearner_bounds["data_source"] = config.data_source
    selected_blearner_bounds, blearner_gamma_selection = _select_blearner_bounds(
        config,
        blearner_bounds,
        naturalness_reference,
    )
    blearner_gamma_selection["data_source"] = config.data_source
    return {
        "blearner_bounds": blearner_bounds,
        "selected_blearner_bounds": selected_blearner_bounds,
        "blearner_gamma_selection": blearner_gamma_selection,
        "blearner_oracle_gamma": oracle_gamma_calibration,
    }


def fit_blearner_only(config: Exp18Config) -> dict[str, pd.DataFrame]:
    """Fit only B-Learner against matched persisted experiment data.

    This entry point loads the same target-idle rows and Naturalness reference
    used by the complete Exp18 comparison, but does not create or execute any
    WB, CB-IPW, or CB-DR fit tasks. It supports fixed, width-selected, and
    oracle Gamma rules through the existing B-Learner implementation.

    Args:
        config: Exp18 configuration identifying the fitting data, target,
            Naturalness reference, and Gamma rule. Configured meta-learners are
            ignored because this entry point never fits them.

    Returns:
        Raw and selected B-Learner bounds, Gamma-selection diagnostics,
        optional oracle-Gamma calibration rows, and the validated Naturalness
        reference used for matching.
    """

    reference_experiment = _inspect_naturalness_reference(config)
    data_source = _resolve_data_source(config, reference_experiment)
    config = replace(config, data_source=data_source)
    fitting_data_by_study, _ = _load_fitting_data(config, data_source)
    environments_by_study: dict[str, dict[str, pd.DataFrame]] = {}
    for study in config.studies:
        study_name = str(study)
        target_site = _study_target_site(config, study_name)
        target_idle = fitting_data_by_study[study_name].idle.loc[
            fitting_data_by_study[study_name].idle["site"]
            .astype(str)
            .eq(target_site)
        ].copy()
        if target_idle.empty:
            raise ValueError(
                f"{study_name} has no target-idle rows for {target_site}"
            )
        environments_by_study[study_name] = {target_site: target_idle}
    naturalness_reference = _load_naturalness_reference(config)
    result = _fit_and_select_blearner_bounds(
        config,
        environments_by_study,
        naturalness_reference,
        fitting_data_by_study,
    )
    result["naturalness_reference"] = naturalness_reference
    return result


def run_experiment(config: Exp18Config) -> dict[str, Any]:
    """Run the parallel Exp18 comparison and write its complete artifact set."""

    start = time.perf_counter()
    reference_experiment = _inspect_naturalness_reference(config)
    data_source = _resolve_data_source(config, reference_experiment)
    config = replace(config, data_source=data_source)
    output_dir = _leaf_output_dir(
        config,
        data_source=data_source,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir,
        experiment_name="exp18_cb_meta_learner_comparison",
        script_path="experiments/exp18_cb_meta_learner_comparison.py",
        config=asdict(config),
        argv=config.command_argv,
        cluster_manifest_path=config.cluster_manifest_path,
    )
    fitting_data_by_study, data_provenance = _load_fitting_data(
        config,
        data_source,
    )
    source_sites_by_study, source_selection = _select_source_sites_by_study(
        config,
        fitting_data_by_study,
    )
    environments_by_study: dict[str, dict[str, pd.DataFrame]] = {}
    support_frames: list[pd.DataFrame] = []
    source_pool_composition_frames: list[pd.DataFrame] = []
    tasks: list[FitTask] = []
    next_task_id = 0
    for study in config.studies:
        environments, support, source_pool_composition = _load_site_environments(
            config,
            study,
            source_sites_by_study[str(study)],
            fitting_data_by_study[str(study)],
        )
        environments_by_study[study] = environments
        support_frames.append(support)
        source_pool_composition_frames.append(source_pool_composition)
        study_tasks = _build_study_tasks(
            config,
            study,
            environments,
            tuple(
                name
                for name in environments
                if name != _study_target_site(config, study)
            ),
            next_task_id,
        )
        tasks.extend(study_tasks)
        next_task_id += len(study_tasks)
    support_diagnostics = pd.concat(support_frames, ignore_index=True)
    source_pool_composition = pd.concat(
        source_pool_composition_frames,
        ignore_index=True,
    )
    source_pool_composition = source_pool_composition.merge(
        source_selection[["study", "source_site", "role"]],
        on=["study", "source_site"],
        how="left",
        validate="one_to_one",
    )
    if config.verbose:
        print(
            f"Exp18 dispatching {len(tasks)} independent fits with "
            f"n_jobs={min(_resolve_n_jobs(config.n_jobs), len(tasks))} and "
            f"worker_threads={config.worker_threads}."
        )
        for study in config.studies:
            print(
                f"{study} source sites: "
                + ", ".join(source_sites_by_study[str(study)])
            )
    fit_results, fit_diagnostics = _run_fit_tasks(tasks, config)
    # The schema header was inspected for output routing, but reference CATE
    # values remain unloaded until every CB/WB fit has finished.
    naturalness_reference = _load_naturalness_reference(config)
    if config.verbose:
        print(
            "Exp18 fitting B-Learner with Gamma="
            f"{config.gamma} on target idle rows."
        )
    blearner_result = _fit_and_select_blearner_bounds(
        config,
        environments_by_study,
        naturalness_reference,
        fitting_data_by_study,
    )
    blearner_bounds = blearner_result["blearner_bounds"]
    selected_blearner_bounds = blearner_result["selected_blearner_bounds"]
    blearner_gamma_selection = blearner_result["blearner_gamma_selection"]
    oracle_gamma_calibration = blearner_result["blearner_oracle_gamma"]
    selected_gamma_by_study = {
        str(row.study): float(row.selected_gamma)
        for row in blearner_gamma_selection.loc[
            blearner_gamma_selection["selected"]
        ].itertuples(index=False)
    }

    study_outputs = [
        _assemble_study_bounds(
            config,
            study,
            environments_by_study[study],
            support_diagnostics.loc[support_diagnostics["study"].eq(study)],
            tasks,
            fit_results,
        )
        for study in config.studies
    ]
    cb_final_bounds = pd.concat(
        [output[0] for output in study_outputs],
        ignore_index=True,
    )
    cb_pair_bounds = pd.concat(
        [output[1] for output in study_outputs],
        ignore_index=True,
    )
    wb_bounds = pd.concat(
        [output[2] for output in study_outputs],
        ignore_index=True,
    )
    for frame in (cb_final_bounds, cb_pair_bounds, wb_bounds):
        frame["data_source"] = data_source
    n_source_sites_by_study = {
        study: len(sites) for study, sites in source_sites_by_study.items()
    }
    cb_final_bounds["n_source_sites"] = cb_final_bounds["study"].map(
        n_source_sites_by_study
    )
    support_diagnostics["data_source"] = data_source
    source_pool_composition["data_source"] = data_source
    comparison, summary = _build_comparison(
        cb_final_bounds,
        naturalness_reference,
        selected_blearner_bounds,
        meta_learners=config.meta_learners,
        covariates=_comparison_covariates(config),
        naturalness_method_label=(
            f"naturalness_{config.naturalness_backend}_posterior_mean"
        ),
    )
    metadata = pd.DataFrame(
        [
            {
                "target_site": config.target_site,
                "data_source": data_source,
                "pattern": config.pattern,
                "beta": config.beta,
                "seed": config.seed,
                "source_site_policy": config.source_site_policy,
                "n_source_sites_by_study": json.dumps(
                    n_source_sites_by_study,
                    sort_keys=True,
                ),
                "source_sites_by_study": json.dumps(
                    source_sites_by_study,
                    sort_keys=True,
                ),
                "source_environment": POOLED_SOURCE_ENVIRONMENT,
                "source_regime": config.source_regime,
                "source_pooling": "raw_row_concatenation",
                "n_fitting_environments": 2,
                "source_pool_rows_by_study": json.dumps(
                    source_pool_composition.groupby("study")[
                        "n_source_site_rows"
                    ].sum().to_dict(),
                    sort_keys=True,
                ),
                "pair_scope": config.pair_scope,
                "meta_learners": "|".join(config.meta_learners),
                "n_jobs": min(_resolve_n_jobs(config.n_jobs), len(tasks)),
                "worker_threads": config.worker_threads,
                "n_fit_tasks": len(tasks),
                "blearner_gamma_strategy": config.gamma,
                "blearner_gamma_grid": (
                    "|".join(
                        f"{gamma:g}" for gamma in _blearner_gammas(config)
                    )
                    if config.gamma not in (ORACLE_GAMMA, DGP_ORACLE_GAMMA)
                    else None
                ),
                "blearner_gamma_search_tolerance": (
                    config.gamma_search_tolerance
                ),
                "blearner_gamma_search_width_tolerance": (
                    config.gamma_search_width_tolerance
                ),
                "blearner_gamma_search_max_iterations": (
                    config.gamma_search_max_iterations
                ),
                "blearner_selected_gamma_by_study": json.dumps(
                    selected_gamma_by_study,
                    sort_keys=True,
                ),
                "blearner_oracle_gamma_assumption": (
                    (
                        DGP_ORACLE_GAMMA_ASSUMPTION
                        if config.gamma == DGP_ORACLE_GAMMA
                        else ORACLE_GAMMA_ASSUMPTION
                    )
                    if config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA)
                    else None
                ),
                "blearner_repository": str(config.blearner_repository),
                "blearner_commit": BLEARNER_COMMIT,
                "naturalness_backend": config.naturalness_backend,
                "naturalness_trigger_backend": (
                    config.naturalness_trigger_backend
                ),
                "naturalness_lp_backend": config.naturalness_lp_backend,
                "naturalness_gp_feature_mode": (
                    config.naturalness_gp_feature_mode
                ),
                "naturalness_pre_all_triggered_folder": (
                    config.naturalness_pre_all_triggered_folder
                ),
                "naturalness_reference": str(_naturalness_reference_path(config)),
                "naturalness_reference_experiment": str(
                    naturalness_reference[
                        "naturalness_reference_experiment"
                    ].iloc[0]
                ),
                "assumption": "site_invariant_oracle_response_surfaces",
                "target_rct_role": "evaluation_only",
            }
        ]
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    cb_pair_bounds.to_csv(output_dir / "cb_pair_bounds.csv", index=False)
    wb_bounds.to_csv(output_dir / "wb_bounds.csv", index=False)
    support_diagnostics.to_csv(output_dir / "support.csv", index=False)
    source_pool_composition.to_csv(
        output_dir / "source_pool_composition.csv",
        index=False,
    )
    source_selection.to_csv(output_dir / "source_selection.csv", index=False)
    data_provenance.to_csv(output_dir / "data_provenance.csv", index=False)
    blearner_bounds.to_csv(output_dir / "blearner_bounds.csv", index=False)
    blearner_gamma_selection.to_csv(
        output_dir / "blearner_gamma_selection.csv",
        index=False,
    )
    if config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA):
        oracle_gamma_calibration.to_csv(
            output_dir / "blearner_oracle_gamma.csv",
            index=False,
        )
    fit_diagnostics.to_csv(output_dir / "fit_diagnostics.csv", index=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    if config.make_plots:
        _make_comparison_plot(comparison, config, output_dir / "comparison.png")
        _make_comparison_plot(
            comparison,
            config,
            output_dir / "comparison_horizontal.png",
            horizontal=True,
        )
        for suffix in (".png", ".svg"):
            _make_comparison_plot(
                comparison,
                config,
                output_dir
                / f"comparison_horizontal_with_naturalness_envelope{suffix}",
                horizontal=True,
                naturalness_envelope=True,
            )
        if comparison["blackbox_cate_available"].any():
            model_reference_stem = "comparison_vs_blackbox_cate"
            _make_comparison_plot(
                comparison,
                config,
                output_dir / f"{model_reference_stem}.png",
                truth="blackbox",
            )
            _make_comparison_plot(
                comparison,
                config,
                output_dir / f"{model_reference_stem}_horizontal.png",
                horizontal=True,
                truth="blackbox",
            )
    diagnostics = {
        "experiment": "exp18_cb_meta_learner_comparison",
        "config": asdict(config),
        "output_dir": str(output_dir),
        "runtime_seconds": time.perf_counter() - start,
        "n_fit_tasks": len(tasks),
        "n_completed_fits": len(fit_results),
        "n_blearner_fits": int(
            blearner_bounds[["study", "gamma"]].drop_duplicates().shape[0]
        ),
        "blearner_gamma_strategy": config.gamma,
        "blearner_selected_gamma_by_study": selected_gamma_by_study,
        "blearner_oracle_gamma": (
            oracle_gamma_calibration.to_dict(orient="records")
            if config.gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA)
            else []
        ),
        "blearner_gamma_search_status_by_study": (
            blearner_bounds.groupby("study")["gamma_search_status"]
            .first()
            .to_dict()
            if "gamma_search_status" in blearner_bounds
            else {}
        ),
        "blearner_repository": str(config.blearner_repository),
        "blearner_commit": BLEARNER_COMMIT,
        "meta_learners": list(config.meta_learners),
        "source_site_policy": config.source_site_policy,
        "n_source_sites_by_study": n_source_sites_by_study,
        "source_sites_by_study": {
            study: list(sites) for study, sites in source_sites_by_study.items()
        },
        "source_environment": POOLED_SOURCE_ENVIRONMENT,
        "source_regime": config.source_regime,
        "source_pooling": "raw_row_concatenation",
        "data_source": data_source,
        "naturalness_reference_experiment": str(
            naturalness_reference["naturalness_reference_experiment"].iloc[0]
        ),
        "has_blackbox_cate_reference": bool(
            comparison["blackbox_cate_available"].any()
        ),
        "interpretation": (
            f"All CB/WB and B-Learner fits use matched {data_source} data. "
            f"The target uses idle rows and the raw pooled source uses the "
            f"{config.source_regime} regime. Predictions are intersected at "
            "target-site covariate strata. Saved target reference CATEs are "
            "merged only after fitting for evaluation against naturalness."
        ),
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, default=json_default)
    if config.verbose:
        print(
            f"Exp18 completed {len(fit_results)} CB/WB fits and "
            f"{diagnostics['n_blearner_fits']} B-Learner fits in "
            f"{diagnostics['runtime_seconds']:.1f}s."
        )
        print(f"Output written to {output_dir}")
    return {
        "comparison": comparison,
        "summary": summary,
        "cb_pair_bounds": cb_pair_bounds,
        "wb_bounds": wb_bounds,
        "support": support_diagnostics,
        "source_pool_composition": source_pool_composition,
        "source_selection": source_selection,
        "data_provenance": data_provenance,
        "blearner_bounds": blearner_bounds,
        "blearner_gamma_selection": blearner_gamma_selection,
        "blearner_oracle_gamma": oracle_gamma_calibration,
        "fit_diagnostics": fit_diagnostics,
        "metadata": metadata,
        "diagnostics": diagnostics,
        "output_dir": output_dir,
    }


def parse_args(argv: Sequence[str] | None = None) -> Exp18Config:
    """Parse command-line arguments for one Exp18 comparison setting."""

    defaults = Exp18Config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--studies", nargs="+", default=list(defaults.studies))
    parser.add_argument(
        "--data-source",
        choices=DATA_SOURCES,
        default=defaults.data_source,
        help=(
            "Fitting-data experiment. 'auto' matches the naturalness "
            "reference schema."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=defaults.data_root,
        help="Seed-specific Exp16 OSRCT dataset root.",
    )
    parser.add_argument(
        "--rct-path",
        default=defaults.rct_path,
        help="ManyLabs RCT pickle matched to the Exp16 reference.",
    )
    parser.add_argument(
        "--require-observed-gender",
        action=argparse.BooleanOptionalAction,
        default=defaults.require_observed_gender,
        help=(
            "For Exp16, exclude rows with missing original resp_sex from "
            "OSRCT, RCT, B-Learner, and meta-learner inputs."
        ),
    )
    parser.add_argument("--pattern", default=defaults.pattern)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--target-site", default=defaults.target_site)
    parser.add_argument(
        "--n-source-sites",
        type=int,
        default=defaults.n_source_sites,
        help="Source count used only by --source-site-policy random_sample.",
    )
    parser.add_argument(
        "--expected-non-target-sites",
        type=int,
        default=defaults.expected_non_target_sites,
    )
    parser.add_argument(
        "--source-site-policy",
        choices=SOURCE_SITE_POLICIES,
        default=defaults.source_site_policy,
        help=(
            "Random shared source cohort, each study's exact naturalness "
            "final-training cohort, or every matched non-target site."
        ),
    )
    parser.add_argument(
        "--source-regime",
        choices=SOURCE_REGIMES,
        default=defaults.source_regime,
        help="Use matched selected-source idle rows or original RCT on rows.",
    )
    parser.add_argument(
        "--pair-scope",
        choices=PAIR_SCOPES,
        default=defaults.pair_scope,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=defaults.worker_threads,
        help="Torch/BLAS threads assigned to each process-pool worker.",
    )
    parser.add_argument("--n-layers-out", type=int, default=defaults.n_layers_out)
    parser.add_argument("--n-units-out", type=int, default=defaults.n_units_out)
    parser.add_argument("--n-iter", type=int, default=defaults.n_iter)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=defaults.validation_fraction,
    )
    parser.add_argument(
        "--n-iter-print",
        type=int,
        default=defaults.n_iter_print,
    )
    parser.add_argument(
        "--meta-learners",
        nargs="+",
        choices=PAPER_METHODS,
        default=list(defaults.meta_learners),
        help=(
            "CB meta-learners to fit. Available methods map to CATENets "
            "T-, RA-, propensity-weighted, and DR learners."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=_parse_gamma,
        default=defaults.gamma,
        help=(
            "B-Learner Gamma. Use 'width_matched' for grid-only matching or "
            "'width_matched_binary' for bracketed binary search by study; "
            "use 'exp_beta' for Gamma=exp(beta), or 'oracle' for finite-support "
            "calibration from saved OSRCT propensities. Use 'dgp_oracle' for "
            "calibration from the known OSRCT DGP on the full pre-selection "
            "target RCT."
        ),
    )
    parser.add_argument(
        "--gamma-search-grid",
        type=_parse_gamma_search_grid,
        default=defaults.gamma_search_grid,
        help=(
            "Comma-separated increasing coarse Gamma grid used to bracket "
            "width_matched_binary search."
        ),
    )
    parser.add_argument(
        "--gamma-search-tolerance",
        type=float,
        default=defaults.gamma_search_tolerance,
        help="Stop binary search when its Gamma bracket is no wider than this.",
    )
    parser.add_argument(
        "--gamma-search-width-tolerance",
        type=float,
        default=defaults.gamma_search_width_tolerance,
        help=(
            "Stop binary search when absolute prevalence-weighted width gap "
            "is no larger than this."
        ),
    )
    parser.add_argument(
        "--gamma-search-max-iterations",
        type=int,
        default=defaults.gamma_search_max_iterations,
        help="Maximum midpoint fits per study for binary Gamma search.",
    )
    parser.add_argument(
        "--blearner-repository",
        default=defaults.blearner_repository,
        help=f"Pinned B-Learner checkout at commit {BLEARNER_COMMIT}.",
    )
    parser.add_argument(
        "--naturalness-backend",
        default=defaults.naturalness_backend,
        help=(
            "Naturalness backend used to derive the reference "
            "path and isolate the Exp18 output directory."
        ),
    )
    parser.add_argument(
        "--naturalness-trigger-backend",
        default=defaults.naturalness_trigger_backend,
        help="Exp16 target-trigger backend used to derive the reference.",
    )
    parser.add_argument(
        "--naturalness-lp-backend",
        default=defaults.naturalness_lp_backend,
        help="Exp16 target-LP backend used to derive the reference.",
    )
    parser.add_argument(
        "--naturalness-gp-feature-mode",
        choices=SUPPORTED_GP_FEATURE_MODES,
        default=defaults.naturalness_gp_feature_mode,
        help=(
            "Exp16 naturalness GP feature mode used to derive the "
            "saved reference path and isolate Exp18 outputs."
        ),
    )
    parser.add_argument(
        "--naturalness-pre-all-triggered-folder",
        default=defaults.naturalness_pre_all_triggered_folder,
        help=(
            "Optional single output-folder component between the Exp16 "
            "target root and all_triggered (for example, 'latent')."
        ),
    )
    parser.add_argument(
        "--naturalness-reference",
        default=defaults.naturalness_reference,
        help=(
            "Optional explicit Exp16 target_gp_posterior_summary.csv "
            "override."
        ),
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--cluster-manifest-path",
        default=defaults.cluster_manifest_path,
        help="Kubernetes YAML template embedded in run_manifest.yml.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    return Exp18Config(
        studies=tuple(args.studies),
        data_source=args.data_source,
        data_root=args.data_root,
        rct_path=args.rct_path,
        require_observed_gender=args.require_observed_gender,
        pattern=args.pattern,
        beta=args.beta,
        target_site=args.target_site,
        n_source_sites=args.n_source_sites,
        expected_non_target_sites=args.expected_non_target_sites,
        source_site_policy=args.source_site_policy,
        source_regime=args.source_regime,
        pair_scope=args.pair_scope,
        seed=args.seed,
        n_jobs=args.n_jobs,
        worker_threads=args.worker_threads,
        n_layers_out=args.n_layers_out,
        n_units_out=args.n_units_out,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        n_iter_print=args.n_iter_print,
        meta_learners=tuple(args.meta_learners),
        gamma=args.gamma,
        gamma_search_grid=args.gamma_search_grid,
        gamma_search_tolerance=args.gamma_search_tolerance,
        gamma_search_width_tolerance=args.gamma_search_width_tolerance,
        gamma_search_max_iterations=args.gamma_search_max_iterations,
        blearner_repository=args.blearner_repository,
        naturalness_backend=args.naturalness_backend,
        naturalness_trigger_backend=args.naturalness_trigger_backend,
        naturalness_lp_backend=args.naturalness_lp_backend,
        naturalness_gp_feature_mode=args.naturalness_gp_feature_mode,
        naturalness_pre_all_triggered_folder=(
            args.naturalness_pre_all_triggered_folder
        ),
        naturalness_reference=args.naturalness_reference,
        output_dir=args.output_dir,
        cluster_manifest_path=args.cluster_manifest_path,
        command_argv=tuple(sys.argv[1:] if argv is None else argv),
        verbose=not args.quiet,
        make_plots=not args.no_plots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp18 from the command line."""

    run_experiment(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
