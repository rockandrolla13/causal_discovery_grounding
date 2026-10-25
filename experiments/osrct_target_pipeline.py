"""Per-target calibrated-policy pipeline on OSRCT/RCT data.

Exp16's library half: its configuration (:class:`Exp16Config`), the
per-target pipeline (:func:`run_one_target`) and the sweep over studies,
confounding patterns, betas and target sites (:func:`run_experiment`).
Experiment scripts import from here instead of from one another.

* OSRCT rows are the observational ``F='idle'`` regime;
* original ManyLabs rows are the randomized ``F='on'`` regime; and
* runs vary by study, confounding variable, beta, and target site.

The per-target pipeline is an independent calibrated-policy procedure:

1. proper source sites learn naturalness tolerances;
2. policy-tuning sites choose top-k naturalness constraints;
3. endpoint-calibration sites learn a simultaneous conformal correction; and
4. the held-out target RCT CATEs evaluate the final intervals.

The LP and RCT validation use the same three binary covariates, so no
fine-to-useful-stratum marginalization is required.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


from causal_grounding.loaders.schema import MANYLABS_COVARIATES, OUTCOME_COLUMN, TREATMENT_COLUMN
from causal_grounding.lp_solver_learned_naturalness import (
    TargetObservedBinaryQuantities,
    estimate_target_observed_binary_quantities,
    solve_candidate_state_bounds_from_observed_binary_quantities,
    solve_target_bounds_from_observed_binary_quantities,
)
from causal_grounding.naturalness_grounder import NaturalnessGrounder
from causal_grounding.naturalness_grounder import SUPPORTED_GP_FEATURE_MODES
from causal_grounding.naturalness_target_policy import TargetNaturalnessPolicy
from causal_grounding.seeding import posterior_sample_seeds
from causal_grounding.train_target_split import create_train_target_split
from experiments.backdoor_cate_error_analysis import (
    add_backdoor_cate_correction_outputs,
)
from experiments.naturalness_plotting import (
    build_exp16_plot_metric_summary_frame,
    make_exp16_plots,
    write_exp16_plot_metric_summary_tables,
)
from experiments.covariate_preparation import expand_target_sites
from experiments.osrct_data_utils import OSRCTConfig, iter_base_configs, load_prepared_data
from experiments.conformal_bounds import (
    add_full_range_fallback,
    conformalize_lp_bounds,
    finite_sample_conformal_quantile,
)
from experiments.lp_summaries import summarize_lp_posterior_bounds
from experiments.run_config import (
    DEFAULT_LP_POSTERIOR_N_JOBS,
    SUPPORTED_STAGE1_BACKENDS,
    SUPPORTED_LP_PROBABILITY_BACKENDS,
    SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS,
    build_configured_naturalness_grid,
    build_lp_probability_model_kwargs,
    fit_configured_naturalness_grounder,
    resolve_lp_probability_backend,
    resolve_target_trigger_probability_backend,
    uses_hierarchical_dirichlet_draws,
)
from experiments.run_manifest import json_default, write_run_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_EXP16_STUDIES = ("allowedforbidden", "gainloss", "reciprocity")
DEFAULT_EXP16_PATTERNS = ("age", "gender", "polideo")
DEFAULT_EXP16_BETAS = (0.1, 0.25, 0.5, 1.0)
DEFAULT_POLICY_TOP_K_VALUES = (1, 2, 3)
DEFAULT_GP_POSTERIOR_SAMPLES = 1_000
SUPPORTED_CANDIDATE_SELECTION_MODES = (
    "best_per_covariate",
    "all_triggered",
    "all_for_top_k_covariates",
)
SUPPORTED_GP_POLICY_UNCERTAINTIES = (
    "posterior_mean",
    "posterior_lower_05",
    "posterior_upper_95",
    "posterior_simultaneous_upper",
    "posterior_simultaneous_upper_95",
)
MAX_BINARY_CATE_ENDPOINT_RESIDUAL = 2.0

def split_source_roles(
    source_sites: Sequence[str],
    *,
    policy_tuning_fraction: float,
    min_policy_tuning_sites: int,
    calibration_fraction: float,
    min_calibration_sites: int,
    seed: int,
    scoreable_sites: Sequence[str] = (),
    min_scoreable_policy_tuning_sites: int = 0,
    min_scoreable_calibration_sites: int = 0,
) -> dict[str, tuple[str, ...]]:
    """Split source sites into proper-training, tuning, and calibration roles.

    The requested minimum sizes are honored when possible while always
    retaining at least one proper-training site and one tuning site. Optional
    scoreable-site requirements stratify only on RCT support, never outcomes.
    """

    sites = sorted({str(site) for site in source_sites})
    if len(sites) < 3:
        raise ValueError("At least three source sites are required.")
    if not 0.0 < policy_tuning_fraction < 1.0:
        raise ValueError("policy_tuning_fraction must be strictly between 0 and 1.")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1.")

    requested_calibration = max(
        int(min_calibration_sites),
        int(np.ceil(float(calibration_fraction) * len(sites))),
    )
    n_calibration = min(requested_calibration, len(sites) - 2)
    requested_tuning = max(
        int(min_policy_tuning_sites),
        int(np.ceil(float(policy_tuning_fraction) * len(sites))),
    )
    n_tuning = min(requested_tuning, len(sites) - n_calibration - 1)

    scoreable = sorted(set(str(site) for site in scoreable_sites) & set(sites))
    required_calibration = min(
        int(min_scoreable_calibration_sites),
        n_calibration,
    )
    required_tuning = min(
        int(min_scoreable_policy_tuning_sites),
        n_tuning,
    )

    rng = np.random.default_rng(int(seed))
    scoreable = list(
        np.asarray(scoreable, dtype=object)[rng.permutation(len(scoreable))]
    )
    unscoreable = sorted(set(sites) - set(scoreable))
    unscoreable = list(
        np.asarray(unscoreable, dtype=object)[rng.permutation(len(unscoreable))]
    )
    while required_calibration + required_tuning > len(scoreable):
        if required_calibration >= required_tuning and required_calibration > 0:
            required_calibration -= 1
        elif required_tuning > 0:
            required_tuning -= 1
        else:
            break

    calibration = scoreable[:required_calibration]
    scoreable = scoreable[required_calibration:]
    tuning = scoreable[:required_tuning]
    scoreable = scoreable[required_tuning:]
    remaining_pool = [*scoreable, *unscoreable]
    remaining_pool = list(
        np.asarray(remaining_pool, dtype=object)[rng.permutation(len(remaining_pool))]
    )
    calibration.extend(remaining_pool[: n_calibration - len(calibration)])
    used = set(calibration)
    remaining_pool = [site for site in remaining_pool if site not in used]
    tuning.extend(remaining_pool[: n_tuning - len(tuning)])
    used.update(tuning)
    proper = sorted(set(sites) - used)

    return {
        "proper_training": tuple(proper),
        "policy_tuning": tuple(sorted(str(site) for site in tuning)),
        "endpoint_calibration": tuple(sorted(str(site) for site in calibration)),
    }


def _rank_policy_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return target-triggered naturalness candidates in policy preference order.

    Args:
        candidates: Naturalness policy candidate table.

    Returns:
        Selected, finite-tolerance rows sorted by the calibrated-policy ranking.
    """

    if candidates.empty:
        return candidates.copy()
    if "status" in candidates and not candidates["status"].eq("selected").any():
        return candidates.iloc[0:0].copy()

    required = {"status", "covariate", "score", "two_c", "two_d"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {sorted(missing)}")

    selected = candidates.loc[
        candidates["status"].eq("selected")
        & candidates["covariate"].notna()
        & pd.to_numeric(candidates["score"], errors="coerce").notna()
        & pd.to_numeric(candidates["two_c"], errors="coerce").notna()
        & pd.to_numeric(candidates["two_d"], errors="coerce").notna()
    ].copy()
    if selected.empty:
        return selected

    ordering = [
        column
        for column in (
            "score",
            "epsilon",
            "tau",
            "i_margin",
            "d_margin",
            "covariate_index",
        )
        if column in selected.columns
    ]
    ascending = {
        "score": True,
        "epsilon": True,
        "tau": False,
        "i_margin": False,
        "d_margin": False,
        "covariate_index": True,
    }
    selected = selected.sort_values(
        ordering,
        ascending=[ascending[column] for column in ordering],
        kind="mergesort",
    )
    return selected.reset_index(drop=True)


def finalize_policy_choices(
    selected: pd.DataFrame,
    *,
    top_k: int,
    candidate_selection_mode: str,
) -> pd.DataFrame:
    """Attach stable metadata columns to a selected policy-choice table."""

    result = selected.copy().reset_index(drop=True)
    result["is_primary_choice"] = False
    if not result.empty:
        result.loc[result.index[:1], "is_primary_choice"] = True
    result["policy_top_k"] = int(top_k)
    result["candidate_selection_mode"] = str(candidate_selection_mode)
    return result


def select_policy_choices(
    candidates: pd.DataFrame,
    top_k: int,
    *,
    mode: str = "best_per_covariate",
) -> pd.DataFrame:
    """Select naturalness candidates according to the requested policy mode.

    Args:
        candidates: Naturalness policy candidate table.
        top_k: Requested number of covariates for modes that use a covariate
            budget. ``all_triggered`` records the value but keeps every
            triggered candidate.
        mode: Candidate-selection rule. ``best_per_covariate`` preserves the
            historical behavior: keep one best row per covariate and then the
            first ``top_k`` covariates. ``all_triggered`` keeps every triggered
            row. ``all_for_top_k_covariates`` chooses the top ``top_k``
            covariates by their best row and keeps every triggered row for
            those covariates.

    Returns:
        Selected candidates in policy ranking order with policy metadata.
    """

    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")
    if mode not in SUPPORTED_CANDIDATE_SELECTION_MODES:
        raise ValueError(
            f"mode must be one of {SUPPORTED_CANDIDATE_SELECTION_MODES}"
        )

    ranked = _rank_policy_candidates(candidates)
    if ranked.empty:
        return finalize_policy_choices(
            ranked,
            top_k=top_k,
            candidate_selection_mode=mode,
        )
    if mode == "best_per_covariate":
        selected = ranked.drop_duplicates(subset=["covariate"], keep="first")
        selected = selected.head(int(top_k))
    elif mode == "all_triggered":
        selected = ranked
    else:
        covariate_keys = ranked["covariate"].map(str)
        selected_covariates = list(dict.fromkeys(covariate_keys))[: int(top_k)]
        selected = ranked.loc[covariate_keys.isin(selected_covariates)]
    return finalize_policy_choices(
        selected,
        top_k=top_k,
        candidate_selection_mode=mode,
    )


def select_top_k_choices(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Select at most one best naturalness choice per covariate, then keep top-k."""

    return select_policy_choices(
        candidates,
        top_k,
        mode="best_per_covariate",
    )


def grounder_gp_diagnostics(
    grounder: NaturalnessGrounder,
    *,
    training_stage: str,
) -> pd.DataFrame:
    """Return one serializable diagnostics row for each Stage-3 GP surface."""

    rows: list[dict[str, Any]] = []
    for surface_name, surface in (
        ("c", grounder.c_surface_),
        ("d", grounder.d_surface_),
    ):
        if surface is None:
            continue
        diagnostics = surface.diagnostics()
        diagnostics["surface"] = surface_name
        diagnostics["training_stage"] = str(training_stage)
        diagnostics["learned_hyperparameters"] = json.dumps(
            diagnostics.get("learned_hyperparameters", {}),
            sort_keys=True,
        )
        rows.append(diagnostics)
    return pd.DataFrame(rows)


def _grounder_stage1_diagnostics(
    grounder: NaturalnessGrounder,
    *,
    training_stage: str,
) -> pd.DataFrame:
    """
    Return serializable Stage-1 backend diagnostics for one fitted grounder.

    Args:
        grounder: Fitted naturalness grounder.
        training_stage: Human-readable source-role stage label.

    Returns:
        Backend-specific diagnostics with a ``training_stage`` column.
    """
    frame = grounder.get_stage1_diagnostics_frame()
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["training_stage"] = str(training_stage)
    return frame


def _summarize_stage1_diagnostics(stage1_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize shrinkage strength and cross-site heterogeneity diagnostics.

    Args:
        stage1_diagnostics: Row-wise Stage-1 diagnostics from
            ``_grounder_stage1_diagnostics``.

    Returns:
        One row per training stage, backend, and regime with means of shrinkage
        weights and standard deviations of site-level rates.
    """
    if stage1_diagnostics.empty:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ["training_stage", "stage1_backend", "regime"]
        if column in stage1_diagnostics.columns
    ]
    if not group_columns:
        return pd.DataFrame()
    metrics = [
        "propensity_site_parent_weight",
        "propensity_mean_cell_parent_weight",
        "propensity_raw_rate",
        "propensity_site_base_rate",
        "outcome_site_parent_weight",
        "outcome_raw_rate",
        "outcome_site_base_rate",
        "outcome_arm0_site_parent_weight",
        "outcome_arm1_site_parent_weight",
        "outcome_arm0_mean_cell_parent_weight",
        "outcome_arm1_mean_cell_parent_weight",
        "outcome_arm0_raw_rate",
        "outcome_arm1_raw_rate",
        "outcome_arm0_site_base_rate",
        "outcome_arm1_site_base_rate",
        "propensity_prior_strength",
        "outcome_prior_strength",
        "propensity_n_coefficients",
        "outcome_n_coefficients",
        "propensity_prediction_fits",
        "outcome_prediction_fits",
        "propensity_converged",
        "outcome_converged",
        "posterior_samples",
        "posterior_quantile",
    ]
    available_metrics = [
        column for column in metrics if column in stage1_diagnostics.columns
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in stage1_diagnostics.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_sites"] = int(group["site"].nunique()) if "site" in group else len(group)
        for metric in available_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = float(values.std(ddof=0))
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_empirical_rct_cate_labels(
    rct_rows: pd.DataFrame,
    *,
    covariates: Sequence[str],
    min_arm_count: int,
) -> pd.DataFrame:
    """Estimate held-out empirical RCT CATEs on supported useful strata.

    Args:
        rct_rows: Randomized rows containing ``iv`` and ``dv_binary``.
        covariates: Covariates defining the coarser validation strata.
        min_arm_count: Required treated and control observations per stratum.

    Returns:
        One row per supported stratum with arm counts, means, CATE, standard
        error, and a 90% Wald interval.
    """

    required = {TREATMENT_COLUMN, OUTCOME_COLUMN, *covariates}
    missing = required - set(rct_rows.columns)
    if missing:
        raise ValueError(f"RCT rows are missing required columns: {sorted(missing)}")
    if int(min_arm_count) < 1:
        raise ValueError("min_arm_count must be positive.")

    rows: list[dict[str, Any]] = []
    for key, group in rct_rows.groupby(
        list(covariates),
        observed=True,
        dropna=False,
    ):
        key_tuple = key if isinstance(key, tuple) else (key,)
        treated = group.loc[group[TREATMENT_COLUMN].eq(1), OUTCOME_COLUMN]
        control = group.loc[group[TREATMENT_COLUMN].eq(0), OUTCOME_COLUMN]
        if len(treated) < int(min_arm_count) or len(control) < int(min_arm_count):
            continue
        mu1 = float(treated.mean())
        mu0 = float(control.mean())
        standard_error = float(
            np.sqrt(
                mu1 * (1.0 - mu1) / len(treated)
                + mu0 * (1.0 - mu0) / len(control)
            )
        )
        row = {
            covariate: value
            for covariate, value in zip(covariates, key_tuple)
        }
        row.update(
            {
                "rct_mu1": mu1,
                "rct_mu0": mu0,
                "rct_cate": mu1 - mu0,
                "rct_cate_se": standard_error,
                "rct_cate_ci90_lower": max(
                    -1.0,
                    mu1 - mu0 - 1.6448536269514722 * standard_error,
                ),
                "rct_cate_ci90_upper": min(
                    1.0,
                    mu1 - mu0 + 1.6448536269514722 * standard_error,
                ),
                "rct_n_treated": int(len(treated)),
                "rct_n_control": int(len(control)),
            }
        )
        rows.append(row)
    columns = [
        *covariates,
        "rct_mu1",
        "rct_mu0",
        "rct_cate",
        "rct_cate_se",
        "rct_cate_ci90_lower",
        "rct_cate_ci90_upper",
        "rct_n_treated",
        "rct_n_control",
    ]
    return pd.DataFrame(rows, columns=columns)


def _endpoint_conformal_quantile(
    scores: Iterable[float],
    *,
    alpha: float,
) -> tuple[float, int, bool]:
    """Return the finite-sample endpoint correction with an empty-set sentinel."""

    values = pd.to_numeric(pd.Series(list(scores)), errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return MAX_BINARY_CATE_ENDPOINT_RESIDUAL, 1, True
    q, rank = finite_sample_conformal_quantile(values, alpha=alpha)
    return float(q), int(rank), bool(rank > len(values))


def _minimum_non_sentinel_score_count(alpha: float) -> int:
    """Return minimum calibration scores needed to avoid bounded sentinel q."""

    n_scores = 1
    while int(np.ceil((n_scores + 1) * (1.0 - float(alpha)))) > n_scores:
        n_scores += 1
    return n_scores


@dataclass
class Exp16Config(OSRCTConfig):
    """Configuration for calibrated policies on OSRCT/RCT site data."""

    studies: tuple[str, ...] | None = DEFAULT_EXP16_STUDIES
    patterns: tuple[str, ...] | None = DEFAULT_EXP16_PATTERNS
    betas: tuple[float, ...] | None = DEFAULT_EXP16_BETAS
    target_sites: tuple[str, ...] = ("pi",)
    naturalness_covariates: tuple[str, ...] = MANYLABS_COVARIATES
    lp_probability_covariates: tuple[str, ...] = MANYLABS_COVARIATES
    stage1_backend: str = "tabicl_v2"
    validation_stage1_backend: str | None = "tabicl_v2"
    lp_probability_backend: str = "stage1"
    target_trigger_probability_backend: str = "empirical"
    lp_posterior_n_jobs: int = DEFAULT_LP_POSTERIOR_N_JOBS
    skip_lp_intervention_diagnostics: bool = False
    policy_top_k_values: tuple[int, ...] = DEFAULT_POLICY_TOP_K_VALUES
    candidate_selection_mode: str = "best_per_covariate"
    gp_feature_mode: str = "thresholds"
    target_policy_confidence_level: float = 0.95
    policy_tuning_fraction: float = 0.30
    min_policy_tuning_sites: int = 10
    calibration_fraction: float = 0.30
    min_calibration_sites: int = 10
    conformal_alpha: float = 0.10
    skip_calibration: bool = False
    rct_min_arm_count: int = 14
    max_source_sites: int | None = None
    #: Candidates tested against ``covariates`` without joining the reporting
    #: strata. Each is evaluated in the state ``covariates + (extra,)``, and
    #: its target LP bounds the reporting-stratum CATE through weighted
    #: queries (``build_configured_naturalness_grid``).
    naturalness_extra_candidates: tuple[str, ...] = ()
    #: Optional subset of ``covariates + naturalness_extra_candidates`` that
    #: are naturalness candidates; ``None`` tests all of them.
    naturalness_candidate_covariates: tuple[str, ...] | None = None
    #: Share one loaded TabICLv2 checkpoint across fits; predictions are
    #: unchanged, only repeated checkpoint loading is avoided.
    tabicl_share_checkpoint: bool = False
    #: When target choices exist but the Stage-3 GP surfaces are not fitted,
    #: return full-range fallback posterior rows instead of raising.
    fallback_when_gp_unfitted: bool = False
    output_dir: str = str(
        PROJECT_ROOT / "results" / "exp16_osrct_calibrated_policy"
    )
    cluster_manifest_path: str | None = None
    command_argv: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Fill inherited sweep defaults and validate calibrated-policy fields."""

        super().__post_init__()
        if tuple(self.naturalness_covariates) != tuple(self.covariates):
            raise ValueError(
                "Exp16 requires naturalness_covariates to match LP covariates"
            )
        if not set(self.lp_probability_covariates).issubset(self.covariates):
            raise ValueError(
                "Every LP probability covariate must also be an LP covariate"
            )
        if self.stage1_backend not in SUPPORTED_STAGE1_BACKENDS:
            raise ValueError(f"Unsupported stage1_backend: {self.stage1_backend}")
        if self.lp_probability_backend not in SUPPORTED_LP_PROBABILITY_BACKENDS:
            raise ValueError(
                f"Unsupported lp_probability_backend: {self.lp_probability_backend}"
            )
        if (
            self.target_trigger_probability_backend
            not in SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS
        ):
            raise ValueError(
                "Unsupported target_trigger_probability_backend: "
                f"{self.target_trigger_probability_backend}"
            )
        if self.target_policy_uncertainty not in SUPPORTED_GP_POLICY_UNCERTAINTIES:
            raise ValueError(
                "target_policy_uncertainty must be one of "
                f"{SUPPORTED_GP_POLICY_UNCERTAINTIES}"
            )
        if not 0.0 < self.target_policy_confidence_level < 1.0:
            raise ValueError(
                "target_policy_confidence_level must be strictly between 0 and 1"
            )
        if (
            self.target_policy_uncertainty
            in {
                "posterior_simultaneous_upper",
                "posterior_simultaneous_upper_95",
            }
            and not self.fit_gp
        ):
            raise ValueError(
                "simultaneous upper target policy requires fit_gp=True"
            )
        if self.gp_feature_mode not in SUPPORTED_GP_FEATURE_MODES:
            raise ValueError(
                "gp_feature_mode must be one of "
                f"{SUPPORTED_GP_FEATURE_MODES}"
            )
        if self.lp_posterior_samples > 0 and not self.fit_gp:
            raise ValueError(
                "lp_posterior_samples requires fit_gp=True because posterior "
                "tolerance draws come from Stage-3 GP surfaces"
            )
        if self.lp_posterior_n_jobs < 1:
            raise ValueError("lp_posterior_n_jobs must be positive")
        if not self.policy_top_k_values or any(
            int(value) <= 0 for value in self.policy_top_k_values
        ):
            raise ValueError("policy_top_k_values must contain positive integers")
        if self.candidate_selection_mode not in SUPPORTED_CANDIDATE_SELECTION_MODES:
            raise ValueError(
                "candidate_selection_mode must be one of "
                f"{SUPPORTED_CANDIDATE_SELECTION_MODES}"
            )
        if not 0.0 < self.policy_tuning_fraction < 1.0:
            raise ValueError("policy_tuning_fraction must lie in (0, 1)")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        if self.min_policy_tuning_sites < 1 or self.min_calibration_sites < 1:
            raise ValueError("Minimum tuning and calibration sizes must be positive")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must lie in (0, 1)")
        if self.rct_min_arm_count < 1:
            raise ValueError("rct_min_arm_count must be positive")
        if self.max_source_sites is not None and self.max_source_sites < 3:
            raise ValueError("max_source_sites must be at least 3")
        extras = tuple(self.naturalness_extra_candidates)
        if len(set(extras)) != len(extras):
            raise ValueError("naturalness_extra_candidates must not contain duplicates")
        repeated = sorted(set(extras) & set(self.covariates))
        if repeated:
            raise ValueError(
                "naturalness_extra_candidates must not repeat LP covariates: "
                f"{repeated}"
            )
        if extras and tuple(self.lp_probability_covariates) != tuple(self.covariates):
            raise ValueError(
                "naturalness_extra_candidates requires lp_probability_covariates "
                "to equal covariates"
            )
        if self.naturalness_candidate_covariates is not None:
            candidates = tuple(self.naturalness_candidate_covariates)
            if not candidates or len(set(candidates)) != len(candidates):
                raise ValueError(
                    "naturalness_candidate_covariates must be nonempty and unique"
                )
            unknown = sorted(set(candidates) - set(self.covariates) - set(extras))
            if unknown:
                raise ValueError(
                    "naturalness_candidate_covariates must be LP covariates or "
                    f"extra candidates; unknown values: {unknown}"
                )


def _format_beta_path(beta: float) -> str:
    """Format a beta value as a stable output-directory component.

    Args:
        beta: Numeric confounding-strength value.

    Returns:
        String such as ``0.25``, ``0.5``, or ``1.0``.
    """

    beta_value = float(beta)
    if beta_value.is_integer():
        return f"{beta_value:.1f}"
    return f"{beta_value:g}"


def _safe_path_component(value: Any) -> str:
    """Convert a sweep label into a single safe path component.

    Args:
        value: Label to use in an output path.

    Returns:
        Path-safe string with separators replaced.
    """

    return str(value).replace("/", "_").replace("\\", "_")


def _run_output_dir(root: Path, config: Exp16Config) -> Path:
    """Return the backend- and mode-specific Exp16 run directory.

    Args:
        root: Base directory supplied through ``--output-dir``.
        config: Experiment configuration providing the candidate-selection
            mode and Stage-1 backend.

    Returns:
        Run directory under ``root`` beginning with the requested Stage-1,
        target-trigger, and target-LP backend labels, followed by candidate
        selection mode. Opt-in no-calibration and CATE-only LP ablations
        receive additional leaves.
    """

    run_dir = (
        root
        / _safe_path_component(config.stage1_backend)
        / _safe_path_component(
            resolve_target_trigger_probability_backend(config)
        )
        / _safe_path_component(resolve_lp_probability_backend(config))
        / _safe_path_component(config.candidate_selection_mode)
    )
    if config.gp_feature_mode != "thresholds":
        run_dir /= _safe_path_component(config.gp_feature_mode)
    if config.skip_calibration:
        run_dir /= "no_calibration"
    if config.skip_lp_intervention_diagnostics:
        run_dir /= "cate_only_lp"
    return run_dir


def _output_group_key(
    *,
    target_site: str,
    config: Exp16Config,
) -> tuple[str, str, str]:
    """Return the output grouping key for one Exp16 target run.

    Args:
        target_site: Concrete target site label.
        config: Concrete study/pattern/beta configuration.

    Returns:
        ``(target, pattern, beta_label)`` within the mode/backend run root.
    """

    return (
        _safe_path_component(target_site),
        _safe_path_component(config.pattern),
        _format_beta_path(float(config.beta)),
    )


def _leaf_output_dir(root: Path, key: tuple[str, str, str]) -> Path:
    """Return the leaf directory for a target/pattern/beta grouping key.

    Args:
        root: Mode- and backend-specific Exp16 run directory.
        key: Output grouping key from :func:`_output_group_key`.

    Returns:
        Leaf directory path under ``root``.
    """

    target_site, pattern, beta_label = key
    return root / target_site / pattern / beta_label


def _add_metadata(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach run metadata columns to an output table."""

    result = frame.copy()
    for key, value in metadata.items():
        result[key] = value
    return result


def split_site_regimes(
    site_data: pd.DataFrame,
    config: Exp16Config,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return idle and RCT rows, optionally enforcing participant disjointness."""

    idle = site_data.loc[site_data["F"].astype(str).eq("idle")].copy()
    rct = site_data.loc[site_data["F"].astype(str).eq("on")].copy()
    removed = 0
    if (
        config.disjoint_validation_ids
        and config.id_col in idle.columns
        and config.id_col in rct.columns
    ):
        idle_ids = set(idle[config.id_col].dropna())
        before = len(rct)
        rct = rct.loc[~rct[config.id_col].isin(idle_ids)].copy()
        removed = before - len(rct)
    return idle, rct, int(removed)


def _fit_grounder_for_sites(
    source_training_data: Mapping[str, pd.DataFrame],
    sites: Sequence[str],
    config: Exp16Config,
) -> NaturalnessGrounder:
    """Fit the configured naturalness grounder on a source-site subset."""

    subset = {str(site): source_training_data[str(site)] for site in sites}
    if not subset:
        raise ValueError("At least one source site is required")
    return fit_configured_naturalness_grounder(subset, config)


def _uses_candidate_states(config: Exp16Config) -> bool:
    """Return whether extra candidates put candidates in separate states."""

    return bool(tuple(config.naturalness_extra_candidates))


def _candidate_state_map(config: Exp16Config) -> dict[str, tuple[str, ...]]:
    """Return every candidate's evaluation state from the configured grid.

    Args:
        config: Exp16 configuration defining covariates and extra candidates.

    Returns:
        Mapping from candidate covariate to its ordered state covariates.
    """

    grid = build_configured_naturalness_grid(config)
    return {
        covariate: tuple(grid.candidate_state(j)[0])
        for j, covariate in grid.candidate_items
    }


def _estimate_candidate_state_quantities(
    target_idle: pd.DataFrame,
    config: Exp16Config,
    *,
    backend: str,
    probability_kwargs: Mapping[str, Any] | None,
) -> dict[tuple[str, ...], TargetObservedBinaryQuantities]:
    """Fit one target probability table per distinct candidate state.

    Each table is estimated on exactly its state's covariates, matching the
    per-state Stage-1 models of the grounder, so a candidate's target
    triggers and LP never see covariates outside its state.

    Args:
        target_idle: Target idle-regime rows.
        config: Exp16 configuration with extra candidates.
        backend: Resolved target probability backend.
        probability_kwargs: Backend model options.

    Returns:
        Target probability tables keyed by state tuple.
    """

    states = list(dict.fromkeys(_candidate_state_map(config).values()))
    return {
        state: estimate_target_observed_binary_quantities(
            target_idle,
            treatment=TREATMENT_COLUMN,
            outcome=OUTCOME_COLUMN,
            covariates=list(state),
            model_covariates=list(state),
            probability_backend=backend,
            min_count=int(config.min_cell_count),
            probability_model_kwargs=(
                dict(probability_kwargs) if probability_kwargs is not None else None
            ),
        )
        for state in states
    }


def _summarize_target_trigger_grid(trigger_grid: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse the target trigger grid to one row per candidate.

    ``I*_j`` and ``D*_j`` do not depend on the grid point, so each candidate
    keeps its statistics and the number of ``(epsilon, tau)`` rows it fired.

    Args:
        trigger_grid: Complete target trigger evaluation, including rejections.

    Returns:
        One row per candidate covariate; empty when no grid was computed.
    """

    if trigger_grid is None or trigger_grid.empty:
        return pd.DataFrame()
    return (
        trigger_grid.groupby(["covariate", "covariate_index"], sort=False)
        .agg(
            i_deviation=("i_deviation", "first"),
            d_strength=("d_strength", "first"),
            n_z_minus_evaluated=("n_z_minus_evaluated", "first"),
            n_target_rows=("n_target_rows", "first"),
            target_probability_source=("target_probability_source", "first"),
            n_grid_points=("target_triggered", "size"),
            n_triggered_grid_points=("target_triggered", "sum"),
        )
        .reset_index()
        .assign(
            n_triggered_grid_points=lambda frame: frame[
                "n_triggered_grid_points"
            ].astype(int),
            target_triggered_any=lambda frame: frame["n_triggered_grid_points"].gt(0),
        )
    )


def build_policy_candidates(
    grounder: NaturalnessGrounder,
    target_idle: pd.DataFrame,
    config: Exp16Config,
    *,
    trigger_quantities: TargetObservedBinaryQuantities | None = None,
    with_trigger_grid: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame | None]:
    """Generate target-triggered conditions, optionally reusing probabilities.

    Args:
        grounder: Fitted source naturalness grounder.
        target_idle: Held-out idle-regime rows being screened.
        config: Exp16 experiment configuration.
        trigger_quantities: Optional precomputed target probabilities from an
            identical deterministic target-LP fit; a state-keyed mapping when
            extra candidates are configured.
        with_trigger_grid: Also return the complete target trigger grid,
            including rejected rows.

    Returns:
        Full target candidate-policy audit table, and the trigger grid when
        ``with_trigger_grid`` is true.
    """

    trigger_backend = resolve_target_trigger_probability_backend(config)
    if trigger_quantities is None:
        trigger_quantities = estimate_target_trigger_quantities(
            target_idle,
            config,
            backend=trigger_backend,
        )
    policy = TargetNaturalnessPolicy(
        policy="all_triggered_intersection",
        uncertainty=config.target_policy_uncertainty,
        confidence_level=config.target_policy_confidence_level,
    )
    result = policy.choose(
        target_idle,
        grounder,
        treatment=TREATMENT_COLUMN,
        outcome=OUTCOME_COLUMN,
        regime_col="F",
        trigger_quantities=trigger_quantities,
    )
    frame = result.to_frame(selected_only=False)
    frame["target_probability_source"] = trigger_backend
    frame["target_trigger_probability_backend"] = trigger_backend
    if with_trigger_grid:
        return frame, result.trigger_grid
    return frame


def estimate_target_trigger_quantities(
    target_idle: pd.DataFrame,
    config: Exp16Config,
    *,
    backend: str,
) -> TargetObservedBinaryQuantities | None:
    """Fit optional model-backed target probabilities for Exp16 trigger tests.

    Exp16's legacy target screen uses symmetric-smoothed empirical proportions
    inside :class:`TargetNaturalnessPolicy`; returning ``None`` preserves that
    exact path. Other configured backends fit target-only nuisance models on
    the same covariate state used by naturalness discovery.

    Args:
        target_idle: Held-out target idle-regime rows.
        config: Exp16 experiment configuration.
        backend: Resolved concrete target-trigger probability backend.

    Returns:
        Model-backed target probability table, or ``None`` for legacy
        empirical trigger evaluation. With extra candidates, a mapping with
        one table per candidate state.
    """

    if backend == "empirical":
        return None
    probability_kwargs = build_lp_probability_model_kwargs(config, backend)
    if _uses_candidate_states(config):
        return _estimate_candidate_state_quantities(
            target_idle,
            config,
            backend=backend,
            probability_kwargs=probability_kwargs,
        )
    return estimate_target_observed_binary_quantities(
        target_idle,
        treatment=TREATMENT_COLUMN,
        outcome=OUTCOME_COLUMN,
        covariates=config.covariates,
        model_covariates=config.covariates,
        probability_backend=backend,
        min_count=int(config.min_cell_count),
        probability_model_kwargs=probability_kwargs,
    )


def fit_reusable_tabicl_target_quantities(
    target_idle: pd.DataFrame,
    config: Exp16Config,
) -> TargetObservedBinaryQuantities | None:
    """Fit one safely reusable TabICLv2 target probability table.

    Reuse is deliberately limited to TabICLv2 calls whose resolved trigger
    and LP backends, modeled covariates, data, seed, and model options are
    identical. TabICLv2 receives the configured fixed seed, so replacing its
    repeated identical fits with the same materialized probability table does
    not change candidate screening or LP inputs. Stochastic posterior-draw
    backends and reduced LP predictor covariate sets are never reused here.

    Args:
        target_idle: Held-out idle-regime rows used by both target operations.
        config: Exp16 probability-backend and covariate configuration.

    Returns:
        One materialized TabICLv2 probability table when exact reuse is safe;
        otherwise ``None`` so each existing backend path remains unchanged.
    """

    trigger_backend = resolve_target_trigger_probability_backend(config)
    lp_backend = resolve_lp_probability_backend(config)
    if (
        trigger_backend != "tabicl_v2"
        or trigger_backend != lp_backend
        or tuple(config.covariates) != tuple(config.lp_probability_covariates)
    ):
        return None
    return estimate_target_trigger_quantities(
        target_idle,
        config,
        backend=trigger_backend,
    )


def solve_choices_lp(
    target_idle: pd.DataFrame,
    choices: pd.DataFrame,
    config: Exp16Config,
    *,
    observed_quantities: TargetObservedBinaryQuantities | None = None,
) -> pd.DataFrame:
    """Solve direct target LP bounds, optionally reusing fitted probabilities.

    Args:
        target_idle: Held-out idle-regime rows.
        choices: Selected target naturalness conditions.
        config: Exp16 experiment configuration.
        observed_quantities: Optional precomputed target probabilities from an
            identical deterministic TabICLv2 fit.

    Returns:
        Per-stratum direct LP bound and solver diagnostics. With extra
        candidates, one LP per candidate state is intersected on the
        reporting strata (``solve_candidate_state_bounds_from_observed_binary_quantities``).
    """

    if choices.empty:
        return pd.DataFrame()
    probability_backend = resolve_lp_probability_backend(config)
    probability_kwargs = build_lp_probability_model_kwargs(
        config,
        probability_backend,
    )
    if _uses_candidate_states(config):
        state_observed = observed_quantities
        if state_observed is None:
            state_observed = _estimate_candidate_state_quantities(
                target_idle,
                config,
                backend=probability_backend,
                probability_kwargs=probability_kwargs,
            )
        bounds = solve_candidate_state_bounds_from_observed_binary_quantities(
            state_observed,
            choices,
            candidate_states=_candidate_state_map(config),
            reporting_covariates=config.covariates,
            solver_method=config.lp_solver_method,
            include_intervention_diagnostics=(
                not config.skip_lp_intervention_diagnostics
            ),
        )
        bounds["probability_backend"] = probability_backend
        return bounds
    observed = observed_quantities
    if observed is None:
        observed = estimate_target_observed_binary_quantities(
            target_idle,
            treatment=TREATMENT_COLUMN,
            outcome=OUTCOME_COLUMN,
            covariates=config.covariates,
            model_covariates=config.lp_probability_covariates,
            probability_backend=probability_backend,
            min_count=int(config.min_cell_count),
            probability_model_kwargs=probability_kwargs,
        )
    bounds = solve_target_bounds_from_observed_binary_quantities(
        observed,
        choices,
        solver_method=config.lp_solver_method,
        return_details=True,
        include_intervention_diagnostics=(
            not config.skip_lp_intervention_diagnostics
        ),
    )
    bounds["probability_backend"] = probability_backend
    return bounds


def complete_lp_strata(
    bounds: pd.DataFrame,
    target_idle: pd.DataFrame,
    config: Exp16Config,
) -> pd.DataFrame:
    """Add total fallback endpoints and target-idle prevalence counts."""

    completed = add_full_range_fallback(
        bounds,
        target_idle,
        config,
        empty_status="no_selected_naturalness_choices",
    )
    counts = (
        target_idle.groupby(
            list(config.covariates),
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_target_rows")
        .reset_index()
    )
    return completed.merge(
        counts,
        on=list(config.covariates),
        how="left",
        validate="one_to_one",
    )


def score_rct_intervals(
    bounds: pd.DataFrame,
    rct_rows: pd.DataFrame,
    config: Exp16Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score direct LP intervals against supported empirical RCT CATEs."""

    labels = build_empirical_rct_cate_labels(
        rct_rows,
        covariates=config.covariates,
        min_arm_count=int(config.rct_min_arm_count),
    )
    scored = bounds.copy()
    label_columns = (
        "rct_mu1",
        "rct_mu0",
        "rct_cate",
        "rct_cate_se",
        "rct_cate_ci90_lower",
        "rct_cate_ci90_upper",
        "rct_n_treated",
        "rct_n_control",
    )
    if labels.empty:
        for column in label_columns:
            scored[column] = np.nan
    else:
        scored = scored.merge(
            labels,
            on=list(config.covariates),
            how="left",
            validate="one_to_one",
        )

    lower = pd.to_numeric(scored["score_lower"], errors="coerce")
    upper = pd.to_numeric(scored["score_upper"], errors="coerce")
    truth = pd.to_numeric(scored["rct_cate"], errors="coerce")
    eligible = lower.notna() & upper.notna() & truth.notna()
    scored["endpoint_lower_miss"] = np.where(
        eligible,
        np.maximum(lower - truth, 0.0),
        np.nan,
    )
    scored["endpoint_upper_miss"] = np.where(
        eligible,
        np.maximum(truth - upper, 0.0),
        np.nan,
    )
    scored["endpoint_nonconformity"] = scored[
        ["endpoint_lower_miss", "endpoint_upper_miss"]
    ].max(axis=1, skipna=False)
    scored["rct_cate_available"] = truth.notna()
    scored["covers_rct_cate"] = (
        eligible & truth.ge(lower) & truth.le(upper)
    )
    return scored, {
        "n_lp_rows": int(len(scored)),
        "n_rct_labeled_rows": int(truth.notna().sum()),
        "n_scored_rows": int(eligible.sum()),
        "site_score": (
            float(scored.loc[eligible, "endpoint_nonconformity"].max())
            if eligible.any()
            else np.nan
        ),
        "score_status": (
            "complete_supported_rct_strata"
            if eligible.any()
            else "no_supported_rct_strata"
        ),
        "rct_min_arm_count": int(config.rct_min_arm_count),
    }


def weighted_interval_width(
    frame: pd.DataFrame,
    *,
    lower_col: str,
    upper_col: str,
) -> float:
    """Return target-idle-prevalence-weighted mean interval width."""

    if frame.empty:
        return float("nan")
    lower = pd.to_numeric(frame[lower_col], errors="coerce")
    upper = pd.to_numeric(frame[upper_col], errors="coerce")
    weights = pd.to_numeric(frame["n_target_rows"], errors="coerce").fillna(0.0)
    valid = lower.notna() & upper.notna() & weights.gt(0.0)
    if not valid.any():
        return float("nan")
    return float(
        np.average(
            (upper.loc[valid] - lower.loc[valid]).to_numpy(),
            weights=weights.loc[valid].to_numpy(),
        )
    )


def _scoreable_rct_sites(
    source_training_data: Mapping[str, pd.DataFrame],
    config: Exp16Config,
) -> tuple[str, ...]:
    """Return source sites with at least one supported empirical RCT stratum."""

    scoreable: list[str] = []
    for site, site_data in source_training_data.items():
        _, rct, _ = split_site_regimes(site_data, config)
        labels = build_empirical_rct_cate_labels(
            rct,
            covariates=config.covariates,
            min_arm_count=int(config.rct_min_arm_count),
        )
        if not labels.empty:
            scoreable.append(str(site))
    return tuple(sorted(scoreable))


def evaluate_policy_site(
    *,
    site: str,
    site_data: pd.DataFrame,
    grounder: NaturalnessGrounder,
    top_k: int,
    config: Exp16Config,
    reusable_quantities: TargetObservedBinaryQuantities | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one fixed top-k policy on a held-out source site.

    Args:
        site: Held-out source-site label.
        site_data: Combined idle and randomized rows for that site.
        grounder: Fitted source naturalness grounder.
        top_k: Candidate covariate budget for this evaluation.
        config: Exp16 experiment configuration.
        reusable_quantities: Optional deterministic TabICLv2 table shared
            across repeated top-k evaluations of the same held-out site.

    Returns:
        Scored LP bounds, selected choices, and one site-level score summary.
    """

    idle, rct, removed = split_site_regimes(site_data, config)
    if reusable_quantities is None:
        reusable_quantities = fit_reusable_tabicl_target_quantities(idle, config)
    candidates = build_policy_candidates(
        grounder,
        idle,
        config,
        trigger_quantities=reusable_quantities,
    )
    choices = select_policy_choices(
        candidates,
        top_k,
        mode=config.candidate_selection_mode,
    )
    bounds = complete_lp_strata(
        solve_choices_lp(
            idle,
            choices,
            config,
            observed_quantities=reusable_quantities,
        ),
        idle,
        config,
    )
    scored, summary = score_rct_intervals(bounds, rct, config)
    scored["site"] = str(site)
    scored["policy_top_k"] = int(top_k)
    choices = choices.copy()
    choices["site"] = str(site)
    choices["policy_top_k"] = int(top_k)
    summary.update(
        {
            "site": str(site),
            "policy_top_k": int(top_k),
            "candidate_selection_mode": str(config.candidate_selection_mode),
            "n_selected_choices": int(len(choices)),
            "n_rct_removed_for_disjoint_validation": int(removed),
            "base_weighted_width": weighted_interval_width(
                scored,
                lower_col="score_lower",
                upper_col="score_upper",
            ),
        }
    )
    return scored, choices, summary


def tune_policy_top_k(
    *,
    tuning_sites: Sequence[str],
    source_training_data: Mapping[str, pd.DataFrame],
    grounder: NaturalnessGrounder,
    config: Exp16Config,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose top-k by leave-one-tuning-site-out coverage and width."""

    reusable_quantities_by_site: dict[str, TargetObservedBinaryQuantities] = {}
    for site in tuning_sites:
        idle, _, _ = split_site_regimes(source_training_data[str(site)], config)
        quantities = fit_reusable_tabicl_target_quantities(idle, config)
        if quantities is not None:
            reusable_quantities_by_site[str(site)] = quantities

    site_bounds: list[pd.DataFrame] = []
    site_choices: list[pd.DataFrame] = []
    site_scores: list[dict[str, Any]] = []
    for top_k in sorted({int(value) for value in config.policy_top_k_values}):
        for site in tuning_sites:
            bounds, choices, summary = evaluate_policy_site(
                site=str(site),
                site_data=source_training_data[str(site)],
                grounder=grounder,
                top_k=top_k,
                config=config,
                reusable_quantities=reusable_quantities_by_site.get(str(site)),
            )
            site_bounds.append(bounds)
            site_choices.append(choices)
            site_scores.append(summary)

    scores = pd.DataFrame(site_scores)
    all_bounds = pd.concat(site_bounds, ignore_index=True)
    all_choices = (
        pd.concat(site_choices, ignore_index=True)
        if site_choices
        else pd.DataFrame()
    )
    loo_rows: list[dict[str, Any]] = []
    conformal_bounds: list[pd.DataFrame] = []
    for top_k in sorted(scores["policy_top_k"].unique()):
        top_scores = scores.loc[scores["policy_top_k"].eq(top_k)].copy()
        for site in top_scores["site"].astype(str):
            calibration_scores = top_scores.loc[
                ~top_scores["site"].astype(str).eq(site),
                "site_score",
            ]
            q, rank, sentinel_used = _endpoint_conformal_quantile(
                calibration_scores,
                alpha=float(config.conformal_alpha),
            )
            heldout = all_bounds.loc[
                all_bounds["policy_top_k"].eq(top_k)
                & all_bounds["site"].astype(str).eq(site)
            ].copy()
            heldout = conformalize_lp_bounds(heldout, correction=q)
            truth = pd.to_numeric(heldout["rct_cate"], errors="coerce")
            available = truth.notna()
            covered = (
                available
                & truth.ge(heldout["conformal_lower"])
                & truth.le(heldout["conformal_upper"])
            )
            heldout["covers_rct_cate_conformal"] = covered
            heldout["policy_loo_q"] = float(q)
            heldout["policy_loo_rank"] = int(rank)
            heldout["policy_loo_sentinel_used"] = bool(sentinel_used)
            conformal_bounds.append(heldout)
            loo_rows.append(
                {
                    "site": str(site),
                    "policy_top_k": int(top_k),
                    "loo_q": float(q),
                    "loo_rank": int(rank),
                    "loo_sentinel_used": bool(sentinel_used),
                    "site_scoreable": bool(available.any()),
                    "site_covered": (
                        bool(covered.loc[available].all())
                        if available.any()
                        else np.nan
                    ),
                    "weighted_width": weighted_interval_width(
                        heldout,
                        lower_col="conformal_lower",
                        upper_col="conformal_upper",
                    ),
                }
            )

    loo = pd.DataFrame(loo_rows)
    target_coverage = 1.0 - float(config.conformal_alpha)
    policy_summary = (
        loo.groupby("policy_top_k", as_index=False)
        .agg(
            n_tuning_sites_total=("site", "nunique"),
            n_tuning_sites_scored=("site_scoreable", "sum"),
            loo_site_coverage=("site_covered", "mean"),
            loo_mean_weighted_width=("weighted_width", "mean"),
            loo_mean_q=("loo_q", "mean"),
            loo_sentinel_rate=("loo_sentinel_used", "mean"),
        )
        .sort_values("policy_top_k")
        .reset_index(drop=True)
    )
    policy_summary["target_coverage"] = target_coverage
    policy_summary["coverage_eligible"] = policy_summary[
        "loo_site_coverage"
    ].ge(target_coverage - 1e-12)
    eligible = policy_summary.loc[policy_summary["coverage_eligible"]].copy()
    if not eligible.empty:
        selected = eligible.sort_values(
            ["loo_mean_weighted_width", "policy_top_k"],
            ascending=[True, True],
            kind="mergesort",
        ).iloc[0]
        reason = "minimum_width_at_target_coverage"
    else:
        selected = policy_summary.sort_values(
            ["loo_site_coverage", "loo_mean_weighted_width", "policy_top_k"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        reason = "best_available_coverage_then_width"
    selected_top_k = int(selected["policy_top_k"])
    policy_summary["selected"] = policy_summary["policy_top_k"].eq(
        selected_top_k
    )
    policy_summary["selection_reason"] = reason
    return (
        selected_top_k,
        policy_summary,
        pd.concat(conformal_bounds, ignore_index=True),
        all_choices,
    )


def calibrate_endpoint_q(
    *,
    calibration_sites: Sequence[str],
    source_training_data: Mapping[str, pd.DataFrame],
    grounder: NaturalnessGrounder,
    selected_top_k: int,
    config: Exp16Config,
) -> tuple[float, int, bool, pd.DataFrame, pd.DataFrame]:
    """Estimate the final simultaneous endpoint-conformal correction."""

    bounds_rows: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    for site in calibration_sites:
        bounds, _, summary = evaluate_policy_site(
            site=str(site),
            site_data=source_training_data[str(site)],
            grounder=grounder,
            top_k=selected_top_k,
            config=config,
        )
        bounds_rows.append(bounds)
        score_rows.append(summary)
    scores = pd.DataFrame(score_rows)
    q, rank, sentinel_used = _endpoint_conformal_quantile(
        scores["site_score"],
        alpha=float(config.conformal_alpha),
    )
    calibration_bounds = conformalize_lp_bounds(
        pd.concat(bounds_rows, ignore_index=True),
        correction=q,
    )
    truth = pd.to_numeric(calibration_bounds["rct_cate"], errors="coerce")
    calibration_bounds["covers_rct_cate_conformal"] = (
        truth.notna()
        & truth.ge(calibration_bounds["conformal_lower"])
        & truth.le(calibration_bounds["conformal_upper"])
    )
    scores["conformal_q"] = float(q)
    scores["conformal_rank"] = int(rank)
    scores["sentinel_used"] = bool(sentinel_used)
    return q, rank, sentinel_used, scores, calibration_bounds


def _calibrate_endpoint_q_if_enabled(
    *,
    calibration_sites: Sequence[str],
    source_training_data: Mapping[str, pd.DataFrame],
    grounder: NaturalnessGrounder,
    selected_top_k: int,
    config: Exp16Config,
) -> tuple[float, int, bool, pd.DataFrame, pd.DataFrame]:
    """Run endpoint calibration unless the explicit ablation flag disables it.

    Skipped calibration keeps the original source-role split unchanged, uses
    zero endpoint widening, and returns no calibration diagnostics. This makes
    uncalibrated and calibrated runs differ only through the calibration step,
    rather than silently adding the held-out calibration sites to training.

    Args:
        calibration_sites: Held-out source sites reserved for endpoint scoring.
        source_training_data: Site-keyed source rows.
        grounder: Final source grounder used by the target policy.
        selected_top_k: Selected policy label passed to calibration sites.
        config: Exp16 configuration containing ``skip_calibration``.

    Returns:
        Correction, conformal rank, sentinel flag, calibration scores, and
        calibration bounds. The disabled result is ``(0, 0, False, empty,
        empty)``.
    """

    if config.skip_calibration:
        return 0.0, 0, False, pd.DataFrame(), pd.DataFrame()
    return calibrate_endpoint_q(
        calibration_sites=calibration_sites,
        source_training_data=source_training_data,
        grounder=grounder,
        selected_top_k=selected_top_k,
        config=config,
    )


def _fixed_all_triggered_policy_summary(
    config: Exp16Config,
    *,
    selected_top_k: int,
) -> pd.DataFrame:
    """Describe the fixed all-triggered policy without tuning diagnostics.

    Args:
        config: Experiment configuration defining the conformal target.
        selected_top_k: Positive compatibility label attached to choice rows.
            The all-triggered selector does not use this value.

    Returns:
        One selected policy row with tuning metrics marked unavailable.
    """

    return pd.DataFrame(
        [
            {
                "policy_top_k": int(selected_top_k),
                "n_tuning_sites_total": 0,
                "n_tuning_sites_scored": 0,
                "loo_site_coverage": np.nan,
                "loo_mean_weighted_width": np.nan,
                "loo_mean_q": np.nan,
                "loo_sentinel_rate": np.nan,
                "target_coverage": 1.0 - float(config.conformal_alpha),
                "coverage_eligible": np.nan,
                "selected": True,
                "selection_reason": "fixed_all_triggered_policy",
                "policy_tuning_performed": False,
                "policy_top_k_applicable": False,
            }
        ]
    )


def _fit_grounders_and_select_policy(
    *,
    source_training_data: Mapping[str, pd.DataFrame],
    roles: Mapping[str, Sequence[str]],
    config: Exp16Config,
) -> tuple[
    NaturalnessGrounder | None,
    NaturalnessGrounder,
    int,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit policy grounders and tune top-k only when the mode requires it.

    The all-triggered policy has no covariate budget: every triggered row is
    retained regardless of ``top_k``. In that mode, fit one final grounder on
    the union of nominal proper-training and policy-tuning sites. Other modes
    preserve the proper-fit, held-out tuning, and final-refit procedure.

    Args:
        source_training_data: Site-keyed source rows.
        roles: Proper-training, policy-tuning, and calibration site cohorts.
        config: Exp16 policy and grounding configuration.

    Returns:
        Optional proper-only grounder, final grounder, selected compatibility
        top-k label, policy summary, tuning bounds, and tuning choices.
    """

    final_training_sites = (
        *roles["proper_training"],
        *roles["policy_tuning"],
    )
    if config.candidate_selection_mode == "all_triggered":
        selected_top_k = min(int(value) for value in config.policy_top_k_values)
        final_grounder = _fit_grounder_for_sites(
            source_training_data,
            final_training_sites,
            config,
        )
        return (
            None,
            final_grounder,
            selected_top_k,
            _fixed_all_triggered_policy_summary(
                config,
                selected_top_k=selected_top_k,
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    proper_grounder = _fit_grounder_for_sites(
        source_training_data,
        roles["proper_training"],
        config,
    )
    (
        selected_top_k,
        policy_summary,
        tuning_bounds,
        tuning_choices,
    ) = tune_policy_top_k(
        tuning_sites=roles["policy_tuning"],
        source_training_data=source_training_data,
        grounder=proper_grounder,
        config=config,
    )
    policy_summary = policy_summary.copy()
    policy_summary["policy_tuning_performed"] = True
    policy_summary["policy_top_k_applicable"] = True
    final_grounder = _fit_grounder_for_sites(
        source_training_data,
        final_training_sites,
        config,
    )
    return (
        proper_grounder,
        final_grounder,
        selected_top_k,
        policy_summary,
        tuning_bounds,
        tuning_choices,
    )


def solve_choices_lp_posterior_samples(
    target_idle: pd.DataFrame,
    choices: pd.DataFrame,
    grounder: NaturalnessGrounder,
    config: Exp16Config,
    *,
    random_seed: int,
    observed_quantities: TargetObservedBinaryQuantities | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Propagate joint GP tolerance draws through direct target LP solves.

    When no target condition triggers, retain every observed target stratum
    for every requested posterior draw with the explicit no-choice fallback
    status. This keeps posterior summaries total over target strata while
    leaving their optimal-draw probability at zero and their endpoints
    unavailable. Deterministic TabICLv2 target probabilities may be supplied
    from the identical trigger/direct-LP call and are then reused across all
    posterior tolerance draws. Posterior probability-draw backends always
    retain their independent per-sample estimation path.

    Args:
        target_idle: Held-out idle-regime target rows.
        choices: Selected target naturalness conditions.
        grounder: Fitted source naturalness grounder.
        config: Exp16 posterior and LP configuration.
        random_seed: Seed for joint Stage-3 tolerance draws.
        observed_quantities: Optional precomputed deterministic target
            probability table from an identical TabICLv2 fit; a state-keyed
            mapping when extra candidates are configured.

    Returns:
        Tolerance samples, sample-level LP diagnostics, and posterior summary.
        With ``config.fallback_when_gp_unfitted``, choices whose GP surfaces
        are not fitted yield full-range fallback draws with status
        ``gp_surface_not_fitted`` instead of an error.
    """

    if int(config.lp_posterior_samples) <= 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    gp_unfitted = (not choices.empty) and any(
        surface is None or not getattr(surface, "is_fitted", False)
        for surface in (
            getattr(grounder, "c_surface_", None),
            getattr(grounder, "d_surface_", None),
        )
    )
    if choices.empty or (gp_unfitted and config.fallback_when_gp_unfitted):
        probability_backend = resolve_lp_probability_backend(config)
        completed = complete_lp_strata(
            pd.DataFrame(),
            target_idle,
            config,
        )
        if not choices.empty:
            # Choices triggered, but no fitted surface can sample tolerances.
            completed["status"] = "gp_surface_not_fitted"
        sample_frames: list[pd.DataFrame] = []
        for sample_id in range(int(config.lp_posterior_samples)):
            sample = completed.copy()
            sample["lp_posterior_sample_id"] = sample_id
            sample["probability_backend"] = probability_backend
            sample_frames.append(sample)
        samples = pd.concat(sample_frames, ignore_index=True)
        summary = summarize_lp_posterior_bounds(samples, config.covariates)
        return pd.DataFrame(), samples, summary
    if (
        grounder.c_surface_ is None
        or grounder.d_surface_ is None
        or not grounder.c_surface_.is_fitted
        or not grounder.d_surface_.is_fitted
    ):
        c_status = (
            "missing"
            if grounder.c_surface_ is None
            else grounder.c_surface_.status
        )
        d_status = (
            "missing"
            if grounder.d_surface_ is None
            else grounder.d_surface_.status
        )
        raise RuntimeError(
            "GP posterior LP sampling requires fitted c/d surfaces; "
            f"got c={c_status}, d={d_status}. Increase source sites or use a "
            "less restrictive trigger grid."
        )

    probability_backend = resolve_lp_probability_backend(config)
    probability_kwargs = build_lp_probability_model_kwargs(
        config,
        probability_backend,
    )
    sample_target_probabilities = (
        uses_hierarchical_dirichlet_draws(probability_backend)
    )
    observed = None
    probability_sample_seeds: list[int] = []
    if sample_target_probabilities:
        probability_sample_seeds = posterior_sample_seeds(
            random_seed, config.lp_posterior_samples
        )
    else:
        observed = observed_quantities
        if observed is None and _uses_candidate_states(config):
            observed = _estimate_candidate_state_quantities(
                target_idle,
                config,
                backend=probability_backend,
                probability_kwargs=probability_kwargs,
            )
        elif observed is None:
            observed = estimate_target_observed_binary_quantities(
                target_idle,
                treatment=TREATMENT_COLUMN,
                outcome=OUTCOME_COLUMN,
                covariates=config.covariates,
                model_covariates=config.lp_probability_covariates,
                probability_backend=probability_backend,
                min_count=int(config.min_cell_count),
                probability_model_kwargs=probability_kwargs,
            )
    candidate_states = (
        _candidate_state_map(config) if _uses_candidate_states(config) else None
    )
    tolerance_samples = grounder.sample_tolerances(
        choices,
        n_samples=int(config.lp_posterior_samples),
        random_seed=int(random_seed),
        clip=True,
    )
    grouped_samples = list(tolerance_samples.groupby("sample_id", sort=True))

    def solve_sample(
        grouped_sample: tuple[Any, pd.DataFrame],
    ) -> pd.DataFrame:
        """Solve one independent posterior tolerance/probability draw."""

        sample_id, sampled_choices = grouped_sample
        current_observed = observed
        if sample_target_probabilities:
            sample_kwargs = dict(probability_kwargs or {})
            sample_kwargs["posterior_sample_seed"] = probability_sample_seeds[
                int(sample_id)
            ]
            if candidate_states is not None:
                current_observed = _estimate_candidate_state_quantities(
                    target_idle,
                    config,
                    backend=probability_backend,
                    probability_kwargs=sample_kwargs,
                )
            else:
                current_observed = estimate_target_observed_binary_quantities(
                    target_idle,
                    treatment=TREATMENT_COLUMN,
                    outcome=OUTCOME_COLUMN,
                    covariates=config.covariates,
                    model_covariates=config.lp_probability_covariates,
                    probability_backend=probability_backend,
                    min_count=int(config.min_cell_count),
                    probability_model_kwargs=sample_kwargs,
                )
        assert current_observed is not None
        if candidate_states is not None:
            bounds = solve_candidate_state_bounds_from_observed_binary_quantities(
                current_observed,
                sampled_choices,
                candidate_states=candidate_states,
                reporting_covariates=config.covariates,
                solver_method=config.lp_solver_method,
                include_intervention_diagnostics=(
                    not config.skip_lp_intervention_diagnostics
                ),
            )
        else:
            bounds = solve_target_bounds_from_observed_binary_quantities(
                current_observed,
                sampled_choices,
                solver_method=config.lp_solver_method,
                return_details=True,
                include_intervention_diagnostics=(
                    not config.skip_lp_intervention_diagnostics
                ),
            )
        completed = complete_lp_strata(bounds, target_idle, config)
        completed["lp_posterior_sample_id"] = int(sample_id)
        completed["probability_backend"] = probability_backend
        if sample_target_probabilities:
            completed["target_probability_sample_id"] = int(sample_id)
        return completed

    posterior_workers = min(
        int(config.lp_posterior_n_jobs),
        len(grouped_samples),
    )
    if posterior_workers == 1:
        sample_frames = [solve_sample(sample) for sample in grouped_samples]
    else:
        with ThreadPoolExecutor(max_workers=posterior_workers) as executor:
            sample_frames = list(executor.map(solve_sample, grouped_samples))
    samples = pd.concat(sample_frames, ignore_index=True)
    summary = summarize_lp_posterior_bounds(samples, config.covariates)
    return tolerance_samples, samples, summary


def score_gp_posterior_summary(
    posterior_summary: pd.DataFrame,
    target_bounds: pd.DataFrame,
    *,
    conformal_q: float,
    config: Exp16Config,
) -> pd.DataFrame:
    """Attach RCT labels and sensitivity coverage to GP endpoint summaries."""

    if posterior_summary.empty:
        return posterior_summary.copy()
    reference_columns = [
        *config.covariates,
        "n_target_rows",
        "rct_cate",
        "rct_cate_se",
        "rct_cate_ci90_lower",
        "rct_cate_ci90_upper",
        "rct_n_treated",
        "rct_n_control",
        "rct_cate_available",
    ]
    scored = posterior_summary.merge(
        target_bounds[reference_columns],
        on=list(config.covariates),
        how="left",
        validate="one_to_one",
    )
    scored["gp_posterior_conformal_lower"] = np.maximum(
        -1.0,
        scored["posterior_envelope_lower"] - float(conformal_q),
    )
    scored["gp_posterior_conformal_upper"] = np.minimum(
        1.0,
        scored["posterior_envelope_upper"] + float(conformal_q),
    )
    truth = pd.to_numeric(scored["rct_cate"], errors="coerce")
    scored["covers_rct_cate_posterior_mean_interval"] = (
        truth.notna()
        & truth.ge(scored["lower_mean"])
        & truth.le(scored["upper_mean"])
    )
    scored["covers_rct_cate_posterior_envelope"] = (
        truth.notna()
        & truth.ge(scored["posterior_envelope_lower"])
        & truth.le(scored["posterior_envelope_upper"])
    )
    scored["covers_rct_cate_posterior_conformal_sensitivity"] = (
        truth.notna()
        & truth.ge(scored["gp_posterior_conformal_lower"])
        & truth.le(scored["gp_posterior_conformal_upper"])
    )
    scored["conformal_q_calibrated_on"] = (
        "none_calibration_skipped"
        if config.skip_calibration
        else f"gp_{config.target_policy_uncertainty}_lp_intervals"
    )
    return scored


def summarize_target(
    *,
    metadata: Mapping[str, Any],
    selected_top_k: int,
    conformal_q: float,
    conformal_rank: int,
    sentinel_used: bool,
    policy_summary: pd.DataFrame,
    target_bounds: pd.DataFrame,
    gp_posterior_summary: pd.DataFrame,
    config: Exp16Config,
) -> pd.DataFrame:
    """Create one target-level coverage and width summary row."""

    selected_policy = policy_summary.loc[policy_summary["selected"]].iloc[0]
    available = target_bounds["rct_cate_available"].fillna(False)
    base_covered = target_bounds["covers_rct_cate"].fillna(False)
    conformal_covered = target_bounds[
        "covers_rct_cate_conformal"
    ].fillna(False)
    row: dict[str, Any] = {
        **metadata,
        "selected_top_k": int(selected_top_k),
        "candidate_selection_mode": str(config.candidate_selection_mode),
        "policy_tuning_performed": bool(
            selected_policy.get("policy_tuning_performed", True)
        ),
        "policy_top_k_applicable": bool(
            selected_policy.get("policy_top_k_applicable", True)
        ),
        "fit_gp": bool(config.fit_gp),
        "gp_feature_mode": str(config.gp_feature_mode),
        "lp_posterior_samples": int(config.lp_posterior_samples),
        "policy_tuning_site_coverage": float(
            selected_policy["loo_site_coverage"]
        ),
        "policy_tuning_mean_weighted_width": float(
            selected_policy["loo_mean_weighted_width"]
        ),
        "conformal_q": float(conformal_q),
        "conformal_rank": int(conformal_rank),
        "conformal_sentinel_used": bool(sentinel_used),
        "endpoint_calibration_performed": bool(not config.skip_calibration),
        "n_target_strata": int(len(target_bounds)),
        "n_rct_supported_strata": int(available.sum()),
        "rct_base_point_coverage": (
            float(base_covered.loc[available].mean())
            if available.any()
            else np.nan
        ),
        "rct_conformal_point_coverage": (
            float(conformal_covered.loc[available].mean())
            if available.any()
            else np.nan
        ),
        "rct_base_simultaneous_coverage": (
            bool(base_covered.loc[available].all())
            if available.any()
            else False
        ),
        "rct_conformal_simultaneous_coverage": (
            bool(conformal_covered.loc[available].all())
            if available.any()
            else False
        ),
        "base_optimal_rate": float(
            target_bounds["base_interval_available"].mean()
        ),
        "base_weighted_width": weighted_interval_width(
            target_bounds,
            lower_col="score_lower",
            upper_col="score_upper",
        ),
        "conformal_weighted_width": weighted_interval_width(
            target_bounds,
            lower_col="conformal_lower",
            upper_col="conformal_upper",
        ),
    }
    if not gp_posterior_summary.empty:
        gp_available = gp_posterior_summary["rct_cate_available"].fillna(False)
        row.update(
            {
                "gp_posterior_envelope_weighted_width": weighted_interval_width(
                    gp_posterior_summary,
                    lower_col="posterior_envelope_lower",
                    upper_col="posterior_envelope_upper",
                ),
                "gp_posterior_envelope_rct_point_coverage": (
                    float(
                        gp_posterior_summary.loc[
                            gp_available,
                            "covers_rct_cate_posterior_envelope",
                        ].mean()
                    )
                    if gp_available.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame([row])


def run_one_target(
    *,
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    target_site: str,
    target_index: int,
    config: Exp16Config,
    base_metadata: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Run policy tuning and calibration for one OSRCT target."""

    source_training_data, target_idle = create_train_target_split(
        osrct_data=osrct_data,
        rct_data=rct_data,
        target_site=target_site,
        site_col="site",
        regime_col="F",
    )
    if config.max_source_sites is not None:
        selected_sites = sorted(source_training_data)[: int(config.max_source_sites)]
        source_training_data = {
            site: source_training_data[site] for site in selected_sites
        }
    if len(source_training_data) < 3:
        raise ValueError("Exp16 requires at least three non-target source sites")

    target_rct = rct_data.loc[
        rct_data["site"].astype(str).eq(str(target_site))
    ].copy()
    target_combined = pd.concat(
        [
            target_idle.assign(F="idle"),
            target_rct.assign(F="on"),
        ],
        ignore_index=True,
        sort=False,
    )
    _, scoring_target_rct, target_rct_removed = split_site_regimes(
        target_combined,
        config,
    )

    scoreable_sites = _scoreable_rct_sites(source_training_data, config)
    minimum_scores = _minimum_non_sentinel_score_count(
        float(config.conformal_alpha)
    )
    roles = split_source_roles(
        list(source_training_data),
        policy_tuning_fraction=float(config.policy_tuning_fraction),
        min_policy_tuning_sites=int(config.min_policy_tuning_sites),
        calibration_fraction=float(config.calibration_fraction),
        min_calibration_sites=int(config.min_calibration_sites),
        seed=int(config.seed + target_index),
        scoreable_sites=scoreable_sites,
        min_scoreable_policy_tuning_sites=(
            0
            if config.candidate_selection_mode == "all_triggered"
            else minimum_scores + 1
        ),
        min_scoreable_calibration_sites=minimum_scores,
    )
    (
        proper_grounder,
        final_grounder,
        selected_top_k,
        policy_summary,
        tuning_bounds,
        tuning_choices,
    ) = _fit_grounders_and_select_policy(
        source_training_data=source_training_data,
        roles=roles,
        config=config,
    )
    (
        q,
        rank,
        sentinel_used,
        calibration_scores,
        calibration_bounds,
    ) = _calibrate_endpoint_q_if_enabled(
        calibration_sites=roles["endpoint_calibration"],
        source_training_data=source_training_data,
        grounder=final_grounder,
        selected_top_k=selected_top_k,
        config=config,
    )

    reusable_target_quantities = fit_reusable_tabicl_target_quantities(
        target_idle,
        config,
    )
    target_candidates, target_trigger_grid = build_policy_candidates(
        final_grounder,
        target_idle,
        config,
        trigger_quantities=reusable_target_quantities,
        with_trigger_grid=True,
    )
    target_choices = select_policy_choices(
        target_candidates,
        selected_top_k,
        mode=config.candidate_selection_mode,
    )
    target_bounds = complete_lp_strata(
        solve_choices_lp(
            target_idle,
            target_choices,
            config,
            observed_quantities=reusable_target_quantities,
        ),
        target_idle,
        config,
    )
    target_bounds, _ = score_rct_intervals(
        target_bounds,
        scoring_target_rct,
        config,
    )
    target_bounds = conformalize_lp_bounds(target_bounds, correction=q)
    truth = pd.to_numeric(target_bounds["rct_cate"], errors="coerce")
    target_bounds["covers_rct_cate_conformal"] = (
        truth.notna()
        & truth.ge(target_bounds["conformal_lower"])
        & truth.le(target_bounds["conformal_upper"])
    )
    target_bounds["selected_top_k"] = int(selected_top_k)
    target_bounds["n_rct_removed_for_disjoint_validation"] = int(
        target_rct_removed
    )

    tolerance_samples, posterior_samples, posterior_summary = (
        solve_choices_lp_posterior_samples(
            target_idle,
            target_choices,
            final_grounder,
            config,
            random_seed=int(config.seed + 20_000 + target_index),
            observed_quantities=reusable_target_quantities,
        )
    )
    posterior_summary = score_gp_posterior_summary(
        posterior_summary,
        target_bounds,
        conformal_q=q,
        config=config,
    )

    metadata = {
        "study": str(config.study),
        "pattern": str(config.pattern),
        "beta": float(config.beta),
        "target_site": str(target_site),
        "stage1_backend": str(config.stage1_backend),
        "target_trigger_probability_backend": str(
            resolve_target_trigger_probability_backend(config)
        ),
        "lp_probability_backend": str(resolve_lp_probability_backend(config)),
        "fit_gp": bool(config.fit_gp),
        "target_policy_uncertainty": str(config.target_policy_uncertainty),
        "target_policy_confidence_level": float(
            config.target_policy_confidence_level
        ),
        "policy_tuning_performed": bool(
            config.candidate_selection_mode != "all_triggered"
        ),
        "endpoint_calibration_performed": bool(not config.skip_calibration),
        "lp_intervention_diagnostics_computed": bool(
            not config.skip_lp_intervention_diagnostics
        ),
        "rct_min_arm_count": int(config.rct_min_arm_count),
    }
    role_rows = pd.DataFrame(
        [
            {"site": site, "role": role}
            for role, sites in roles.items()
            for site in sites
        ]
    )
    role_rows["rct_scoreable"] = role_rows["site"].isin(scoreable_sites)
    role_rows["policy_tuning_performed"] = bool(
        config.candidate_selection_mode != "all_triggered"
    )
    role_rows["endpoint_calibration_performed"] = bool(
        not config.skip_calibration
    )
    proper_observations = (
        proper_grounder.observations_.copy()
        if (
            proper_grounder is not None
            and proper_grounder.observations_ is not None
        )
        else pd.DataFrame()
    )
    final_observations = (
        final_grounder.observations_.copy()
        if final_grounder.observations_ is not None
        else pd.DataFrame()
    )
    gp_diagnostic_frames = []
    stage1_diagnostic_frames = []
    if proper_grounder is not None:
        gp_diagnostic_frames.append(
            grounder_gp_diagnostics(
                proper_grounder,
                training_stage="proper_training",
            )
        )
        stage1_diagnostic_frames.append(
            _grounder_stage1_diagnostics(
                proper_grounder,
                training_stage="proper_training",
            )
        )
    final_training_stage = (
        "all_triggered_final_training"
        if config.candidate_selection_mode == "all_triggered"
        else "proper_plus_policy_tuning"
    )
    gp_diagnostic_frames.append(
        grounder_gp_diagnostics(
            final_grounder,
            training_stage=final_training_stage,
        )
    )
    stage1_diagnostic_frames.append(
        _grounder_stage1_diagnostics(
            final_grounder,
            training_stage=final_training_stage,
        )
    )
    gp_diagnostics = pd.concat(
        gp_diagnostic_frames,
        ignore_index=True,
        sort=False,
    )
    stage1_diagnostics = pd.concat(
        stage1_diagnostic_frames,
        ignore_index=True,
        sort=False,
    )
    stage1_diagnostic_summary = _summarize_stage1_diagnostics(
        stage1_diagnostics
    )
    calibration_scores["selected_top_k"] = int(selected_top_k)
    calibration_scores["conformal_q"] = float(q)
    summary = summarize_target(
        metadata=metadata,
        selected_top_k=selected_top_k,
        conformal_q=q,
        conformal_rank=rank,
        sentinel_used=sentinel_used,
        policy_summary=policy_summary,
        target_bounds=target_bounds,
        gp_posterior_summary=posterior_summary,
        config=config,
    )
    selected_covariates = (
        target_choices["covariate"].dropna().astype(str).tolist()
        if "covariate" in target_choices.columns
        else []
    )
    summary["selected_naturalness_covariates"] = json.dumps(selected_covariates)
    summary["selected_naturalness_covariate_signature"] = "|".join(
        selected_covariates
    )

    outputs = {
        "data_metadata": pd.DataFrame(
            [
                {
                    key: (
                        json.dumps(value, default=json_default)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in base_metadata.items()
                }
            ]
        ),
        "run_summary": summary,
        "source_site_roles": role_rows,
        "proper_source_naturalness_observations": proper_observations,
        "final_source_naturalness_observations": final_observations,
        "policy_tuning_summary": policy_summary,
        "policy_tuning_bounds": tuning_bounds,
        "policy_tuning_choices": tuning_choices,
        "endpoint_calibration_scores": calibration_scores,
        "endpoint_calibration_bounds": calibration_bounds,
        "target_policy_candidates": target_candidates,
        "target_trigger_statistics": _summarize_target_trigger_grid(
            target_trigger_grid
        ),
        "target_policy_choices": target_choices,
        "target_lp_bounds": target_bounds,
        "gp_diagnostics": gp_diagnostics,
        "stage1_diagnostics": stage1_diagnostics,
        "stage1_diagnostic_summary": stage1_diagnostic_summary,
        "target_gp_tolerance_samples": tolerance_samples,
        "target_gp_posterior_samples": posterior_samples,
        "target_gp_posterior_summary": posterior_summary,
    }
    add_backdoor_cate_correction_outputs(
        outputs,
        source_training_data=source_training_data,
        target_idle=target_idle,
        target_reference=target_bounds,
        covariates=config.covariates,
        rct_min_arm_count=int(config.rct_min_arm_count),
        backdoor_min_arm_count=int(config.min_cell_count),
        disjoint_validation_ids=bool(config.disjoint_validation_ids),
        id_col=str(config.id_col),
        metadata={"selected_top_k": int(selected_top_k)},
    )
    return {
        name: _add_metadata(frame, metadata)
        if name != "run_summary"
        else frame
        for name, frame in outputs.items()
    }


def run_experiment(config: Exp16Config) -> dict[str, Any]:
    """Run Exp16 over every configured study, confounding pattern, and beta."""

    start = time.time()
    root_output_dir = _run_output_dir(Path(config.output_dir), config)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        root_output_dir,
        experiment_name="exp16_osrct_calibrated_policy",
        script_path="experiments/exp16_osrct_calibrated_policy.py",
        config=asdict(config),
        argv=config.command_argv,
        cluster_manifest_path=config.cluster_manifest_path,
    )
    group_accumulators: dict[
        tuple[str, str, str],
        dict[str, list[pd.DataFrame]],
    ] = {}
    group_errors: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for base_index, base_config in enumerate(iter_base_configs(config)):
        assert isinstance(base_config, Exp16Config)
        osrct_data, rct_data, base_metadata = load_prepared_data(base_config)
        targets = expand_target_sites(
            osrct_data,
            rct_data,
            base_config.target_sites,
            base_config.max_target_sites,
        )
        for target_index, target_site in enumerate(targets):
            group_key = _output_group_key(
                target_site=str(target_site),
                config=base_config,
            )
            group_accumulators.setdefault(group_key, {})
            try:
                outputs = run_one_target(
                    osrct_data=osrct_data,
                    rct_data=rct_data,
                    target_site=str(target_site),
                    target_index=base_index * 10_000 + target_index,
                    config=base_config,
                    base_metadata=base_metadata,
                )
            except Exception as exc:
                error = {
                    "study": base_config.study,
                    "pattern": base_config.pattern,
                    "beta": float(base_config.beta),
                    "target_site": str(target_site),
                    "stage1_backend": str(base_config.stage1_backend),
                    "target_trigger_probability_backend": str(
                        resolve_target_trigger_probability_backend(base_config)
                    ),
                    "lp_probability_backend": str(
                        resolve_lp_probability_backend(base_config)
                    ),
                    "output_dir": str(_leaf_output_dir(root_output_dir, group_key)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(error)
                group_errors.setdefault(group_key, []).append(error)
                if base_config.fail_fast:
                    raise
                continue
            for name, frame in outputs.items():
                if not frame.empty:
                    group_accumulators[group_key].setdefault(name, []).append(frame)

    aggregate_accumulators: dict[str, list[pd.DataFrame]] = {}
    leaf_rows: list[dict[str, Any]] = []
    group_keys = sorted(set(group_accumulators) | set(group_errors))
    for group_key in group_keys:
        leaf_dir = _leaf_output_dir(root_output_dir, group_key)
        leaf_dir.mkdir(parents=True, exist_ok=True)
        leaf_accumulators = group_accumulators.get(group_key, {})
        leaf_results = {
            name: pd.concat(frames, ignore_index=True, sort=False)
            for name, frames in leaf_accumulators.items()
        }
        leaf_results["run_errors"] = pd.DataFrame(group_errors.get(group_key, []))
        for name, frame in leaf_results.items():
            frame.to_csv(leaf_dir / f"{name}.csv", index=False)
            if not frame.empty:
                aggregate_accumulators.setdefault(name, []).append(frame)
        target_site, pattern, beta_label = group_key
        output_group = {
            "candidate_selection_mode": str(config.candidate_selection_mode),
            "target_site": target_site,
            "stage1_backend": str(config.stage1_backend),
            "pattern": pattern,
            "beta": beta_label,
        }
        write_run_manifest(
            leaf_dir,
            experiment_name="exp16_osrct_calibrated_policy",
            script_path="experiments/exp16_osrct_calibrated_policy.py",
            config=asdict(config),
            argv=config.command_argv,
            cluster_manifest_path=config.cluster_manifest_path,
            extra={"output_group": output_group},
        )
        leaf_diagnostics = {
            "config": asdict(config),
            "output_group": output_group,
            "runtime_seconds": time.time() - start,
            "n_completed_runs": int(
                len(leaf_results.get("run_summary", pd.DataFrame()))
            ),
            "n_errors": int(len(group_errors.get(group_key, []))),
            "interpretation": (
                "OSRCT rows define F=idle. Proper source sites learn "
                "naturalness, policy-tuning sites choose top-k, "
                "endpoint-calibration sites learn q, and target-site RCT "
                "rows are validation only. This leaf groups all configured "
                "studies for one target/backend/pattern/beta setting."
            ),
        }
        with (leaf_dir / "diagnostics.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(leaf_diagnostics, handle, indent=2, default=json_default)
        if config.make_plots:
            make_exp16_plots(leaf_dir, leaf_results, config.covariates)
        leaf_rows.append(
            {
                "candidate_selection_mode": str(
                    config.candidate_selection_mode
                ),
                "target_site": target_site,
                "stage1_backend": str(config.stage1_backend),
                "pattern": pattern,
                "beta": beta_label,
                "output_dir": str(leaf_dir),
                "n_completed_runs": leaf_diagnostics["n_completed_runs"],
                "n_errors": leaf_diagnostics["n_errors"],
            }
        )

    results = {
        name: pd.concat(frames, ignore_index=True, sort=False)
        for name, frames in aggregate_accumulators.items()
    }
    results["run_errors"] = pd.DataFrame(errors)
    run_index = pd.DataFrame(leaf_rows)
    run_index.to_csv(root_output_dir / "run_index.csv", index=False)
    results["run_errors"].to_csv(root_output_dir / "run_errors.csv", index=False)
    plot_metric_summary = build_exp16_plot_metric_summary_frame(results)
    if not plot_metric_summary.empty:
        write_exp16_plot_metric_summary_tables(plot_metric_summary, root_output_dir)
    diagnostics = {
        "config": asdict(config),
        "runtime_seconds": time.time() - start,
        "n_completed_runs": int(len(results.get("run_summary", pd.DataFrame()))),
        "n_errors": int(len(errors)),
        "n_output_groups": int(len(leaf_rows)),
        "interpretation": (
            "OSRCT rows define F=idle. Proper source sites learn naturalness, "
            "policy-tuning sites choose top-k, endpoint-calibration sites learn "
            "q, and target-site RCT rows are validation only. Per-group outputs "
            "are written under target/pattern/beta leaves within the "
            "candidate-selection-mode/stage1-backend run root."
        ),
    }
    with (root_output_dir / "diagnostics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(diagnostics, handle, indent=2, default=json_default)
    if config.verbose:
        print(
            f"Exp16 completed {diagnostics['n_completed_runs']} run(s) with "
            f"{diagnostics['n_errors']} error(s) in "
            f"{diagnostics['runtime_seconds']:.2f}s."
        )
        print(f"Output groups written under {root_output_dir}")
    return {
        "outputs": results,
        "diagnostics": diagnostics,
        "output_dir": root_output_dir,
        "run_index": run_index,
    }
