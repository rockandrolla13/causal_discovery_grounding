"""Monte Carlo comparison of Exp16 naturalness and Exp18 baseline bounds.

Each replication generates a fresh Parquet OSRCT grid, runs one full Exp16
sweep, persists the complete reusable Naturalness result bundle as Parquet,
and then runs Exp18 for every bias-covariate/beta combination; Manski and
Balke-Pearl reference bounds follow from the saved target idle rows. Each
B-Learner Gamma writes its own result leaf under ``baseline_variants/``.

A run can be split into a GPU pass, ``--naturalness-only`` (OSRCT data and
Naturalness bundles, sharded with ``--replications``), and a CPU pass,
``--comparators-only`` (Exp18 methods and reference bounds from the bundles).
``merge SOURCE SOURCE ... --output-dir DEST`` combines runs over disjoint betas.
``paper/reproduce_experiment1_manylabs.sh`` runs both passes and the merge for
the paper's Experiment I.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.osrct_target_pipeline import (
    DEFAULT_EXP16_BETAS,
    DEFAULT_EXP16_PATTERNS,
    DEFAULT_EXP16_STUDIES,
    Exp16Config,
    run_experiment as run_exp16,
)
from experiments.exp18_cb_meta_learner_comparison import (
    Exp18Config,
    _comparison_covariates,
    _load_fitting_data,
    _study_target_site,
    fit_blearner_only,
    make_comparison_plot,
    run_experiment as run_exp18,
)
from experiments.benchmarks import (
    BLEARNER_GAMMA_GRID,
    BLEARNER_GAMMA_STRATEGIES,
    DGP_ORACLE_GAMMA,
    META_LEARNER_COLORS,
    META_LEARNER_LABELS,
    ORACLE_GAMMA,
    atomic_write_parquet as _atomic_write_parquet,
    build_mean_bounds_comparison as _shared_build_mean_bounds_comparison,
    build_run_diagnostics as _shared_build_run_diagnostics,
    commit_replication as _shared_commit_replication,
    completed_replications as _shared_completed_replications,
    comparison_to_long_intervals as _shared_comparison_to_long_intervals,
    draw_width_violin_panel as _draw_width_violin_panel,
    format_beta as _format_beta,
    gamma_output_label as _gamma_output_label,
    load_existing_outputs as _shared_load_existing_outputs,
    parse_gamma,
    require_parquet_engine as _shared_require_parquet_engine,
)
from experiments.naturalness_comparison import balke_pearl_bounds, manski_bounds
from experiments.utils import (
    resolve_lp_probability_backend,
    resolve_target_trigger_probability_backend,
    write_run_manifest,
)


EXP25_NAME = "exp25_osrct_mc_baseline_comparison"
EXP25_METHODS = ("naturalness", "blearner", "cb_ipw", "cb_dr")
EXP25_REUSED_BASELINE_METHODS = ("naturalness", "cb_ipw", "cb_dr")
# Reference bounds computed from the saved target idle rows alone. They live in
# their own leaf so the primary tables and their resume contract are untouched.
EXP25_IDENTIFICATION_METHODS = ("manski", "balke_pearl")
EXP25_IDENTIFICATION_DIRNAME = "identification_bounds"
EXP25_METHOD_LABELS = {
    "naturalness": "Naturalness",
    "blearner": "B-Learner",
    "cb_ipw": META_LEARNER_LABELS["cb_ipw"],
    "cb_dr": META_LEARNER_LABELS["cb_dr"],
}
EXP25_METHOD_COLORS = {
    "naturalness": "tab:purple",
    "blearner": "tab:red",
    "cb_ipw": META_LEARNER_COLORS["cb_ipw"],
    "cb_dr": META_LEARNER_COLORS["cb_dr"],
}
EXP25_STRATUM_COLUMNS = (
    "resp_age_bin",
    "resp_gender_bin",
    "resp_polideo_bin",
)
EXP25_IDENTITY_COLUMNS = (
    "study",
    *EXP25_STRATUM_COLUMNS,
    "target_site",
    "n_target_idle_rows",
    "rct_cate",
    "rct_cate_ci90_lower",
    "rct_cate_ci90_upper",
    "rct_cate_available",
)
EXP25_INTERVAL_COLUMNS = (
    "config_fingerprint", "replication", "seed", "bias_covariate",
    "beta", "study", "target_site", *EXP25_STRATUM_COLUMNS, "method",
    "raw_lower", "raw_upper", "lower", "upper", "width", "is_valid",
    "exclusion_reason", "used_full_range_fallback", "support_clipped",
    "n_target_idle_rows", "blearner_gamma", "lp_posterior_prob_optimal",
    "rct_cate", "rct_cate_ci90_lower", "rct_cate_ci90_upper",
    "rct_cate_available",
)
EXP25_INTERVALS_FILENAME = "mc_intervals.parquet"
EXP25_DIAGNOSTICS_FILENAME = "mc_run_diagnostics.parquet"
EXP25_GAMMA_FILENAME = "dgp_gamma_calibration.parquet"
EXP25_REALIZED_GAMMA_FILENAME = "realized_gamma_calibration.parquet"
EXP25_REPLAY_SOURCE_FILENAME = "replay_source_intervals.parquet"
EXP25_FULL_MODE_OWNER_FILENAME = "full_mode_owner.txt"
# How long a Naturalness shard waits for the shard owning replication 0 to
# initialize the shared store; it covers a slow start of that pod.
BUNDLE_MANIFEST_WAIT_SECONDS = 6 * 3600
GRID_GAMMA = "grid"
EXP25_LP_PROBABILITY_BACKENDS = (
    "tabicl_v2",
    "hierarchical_dirichlet",
    "hierarchical_dirichlet_draws",
)


@dataclass(frozen=True)
class Exp25Config:
    """Configuration for the OSRCT Monte Carlo comparison.

    Args:
        n_replications: Number of independent OSRCT/estimator realizations.
        base_seed: First seed; replication ``r`` uses ``base_seed + r``.
        studies: ManyLabs studies included in every Exp16/Exp18 run.
        bias_covariates: OSRCT selection patterns.
        betas: OSRCT selection strengths.
        target_site: Held-out target environment.
        rct_path: Fixed ManyLabs RCT pickle used to generate every realization.
        require_observed_gender: Drop rows with missing original gender before OSRCT and every fit.
        output_dir: Compact result root.
        scratch_dir: Optional parent for temporary upstream artifacts.
        stage1_backend: Exp16 Stage-1 backend; must be site-local TabICLv2.
        target_trigger_probability_backend: Exp16 target trigger backend.
        tabicl_device: TabICLv2 device.
        lp_probability_backend: Exp16 target-LP nuisance backend.
        lp_posterior_samples: Joint GP/probability posterior LP draws.
        lp_posterior_n_jobs: CPU threads used across posterior LP draws.
        exp18_n_jobs: Spawned Exp18 WB/CB fit workers.
        exp18_worker_threads: Numerical-library threads per Exp18 worker.
        blearner_gamma: Fixed Gamma or Exp18 Gamma strategy; each writes its own result leaf.
        baseline_only: Refit only B-Learner, reusing the store's Naturalness/CB-IPW/CB-DR rows.
        identification_bounds_only: Compute only the Manski and Balke-Pearl bounds from the
            saved bundle store; a full run computes them after its last replication.
        identification_min_count: Rows a full cell needs before the Balke-Pearl trigger and
            instrument arms use it, and the Manski support mask; defaults to Exp16's
            ``min_cell_count``, the floor Exp25's Naturalness uses.
        identification_tau: Relevance threshold ``D_j(tau)`` a Balke-Pearl instrument must
            clear; defaults to the smallest Exp16 ``tau``.
        naturalness_only: GPU pass of a split run: fit and persist the OSRCT/Naturalness bundles
            for ``replications`` only. Not fingerprinted, so both passes share one store and leaf.
        comparators_only: CPU pass of a split run: fit Exp18 and the Manski/Balke-Pearl bounds
            from the persisted bundles, as the tail of a full run does.
        replications: Replication indices one ``naturalness_only`` shard fits; ``None`` fits all.
        resume: Resume complete replications from matching Parquet outputs.
        make_plots: Build all final figures after aggregation.
        dummy: Generate simulated compact outputs without fitting any model.
        cluster_manifest_path: Optional cluster manifest embedded in provenance.
        command_argv: Exact command-line arguments recorded in provenance.
        verbose: Emit progress messages.
    """

    n_replications: int = 20
    base_seed: int = 42
    studies: tuple[str, ...] = DEFAULT_EXP16_STUDIES
    bias_covariates: tuple[str, ...] = DEFAULT_EXP16_PATTERNS
    betas: tuple[float, ...] = DEFAULT_EXP16_BETAS
    target_site: str = "pi"
    rct_path: str = Exp16Config().rct_path
    require_observed_gender: bool = True
    output_dir: str = str(PROJECT_ROOT / "results" / EXP25_NAME)
    scratch_dir: str | None = None
    stage1_backend: str = "tabicl_v2"
    target_trigger_probability_backend: str = "tabicl_v2"
    tabicl_device: str = "cuda"
    lp_probability_backend: str = "tabicl_v2"
    lp_posterior_samples: int = 1_000
    lp_posterior_n_jobs: int = 16
    exp18_n_jobs: int = 36
    exp18_worker_threads: int = 1
    blearner_gamma: float | str = DGP_ORACLE_GAMMA
    baseline_only: bool = False
    identification_bounds_only: bool = False
    identification_min_count: int = Exp16Config().min_cell_count
    identification_tau: float = min(Exp16Config().taus)
    naturalness_only: bool = False
    comparators_only: bool = False
    replications: tuple[int, ...] | None = None
    resume: bool = True
    make_plots: bool = True
    dummy: bool = False
    cluster_manifest_path: str | None = None
    command_argv: tuple[str, ...] = ()
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate fixed Monte Carlo grid and worker settings."""

        if self.n_replications < 1:
            raise ValueError("n_replications must be positive")
        if not self.studies or not self.bias_covariates or not self.betas:
            raise ValueError("studies, bias_covariates, and betas must be nonempty")
        if len(set(self.studies)) != len(self.studies):
            raise ValueError("studies must not contain duplicates")
        if len(set(self.bias_covariates)) != len(self.bias_covariates):
            raise ValueError("bias_covariates must not contain duplicates")
        if len(set(map(float, self.betas))) != len(self.betas):
            raise ValueError("betas must not contain duplicates")
        if self.stage1_backend != "tabicl_v2":
            raise ValueError("Exp25 requires site-local stage1_backend='tabicl_v2'")
        if self.target_trigger_probability_backend != "tabicl_v2":
            raise ValueError(
                "Exp25 requires target_trigger_probability_backend='tabicl_v2'"
            )
        if self.lp_probability_backend not in EXP25_LP_PROBABILITY_BACKENDS:
            raise ValueError(
                "Exp25 lp_probability_backend must be one of "
                f"{EXP25_LP_PROBABILITY_BACKENDS}"
            )
        if self.lp_posterior_samples < 1:
            raise ValueError("lp_posterior_samples must be positive")
        if self.lp_posterior_n_jobs < 1:
            raise ValueError("lp_posterior_n_jobs must be positive")
        if self.exp18_n_jobs == 0 or self.exp18_n_jobs < -1:
            raise ValueError("exp18_n_jobs must be -1 or positive")
        if self.identification_bounds_only and (self.baseline_only or self.dummy):
            raise ValueError(
                "identification_bounds_only cannot be combined with baseline_only or dummy"
            )
        modes = {
            "naturalness_only": self.naturalness_only,
            "comparators_only": self.comparators_only,
            "identification_bounds_only": self.identification_bounds_only,
            "baseline_only": self.baseline_only,
            "dummy": self.dummy,
        }
        if (self.naturalness_only or self.comparators_only) and sum(modes.values()) > 1:
            raise ValueError(
                "naturalness_only and comparators_only are split-run passes and "
                "combine with no other mode"
            )
        if self.replications is not None:
            if not self.naturalness_only:
                raise ValueError("replications selects shards of the naturalness_only pass")
            if not self.replications or len(set(self.replications)) != len(self.replications) or any(
                not 0 <= index < self.n_replications for index in self.replications
            ):
                raise ValueError("replications must be distinct indices below n_replications")
        if self.identification_min_count < 1:
            raise ValueError("identification_min_count must be positive")
        if not (np.isfinite(self.identification_tau) and self.identification_tau >= 0):
            raise ValueError("identification_tau must be finite and nonnegative")
        if self.exp18_worker_threads < 1:
            raise ValueError("exp18_worker_threads must be positive")
        if self.blearner_gamma not in BLEARNER_GAMMA_STRATEGIES and (
            not isinstance(self.blearner_gamma, (int, float))
            or not np.isfinite(float(self.blearner_gamma))
            or float(self.blearner_gamma) < 1.0
        ):
            raise ValueError(
                "blearner_gamma must be an Exp18 Gamma strategy or a finite "
                "number at least one"
            )
        # No Gamma is canonical: any arm may run in full mode and populate the
        # shared store, and any arm may replay from it.

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return ordered replication seeds."""

        return tuple(
            int(self.base_seed + replication)
            for replication in range(self.n_replications)
        )


def _naturalness_bundle_leaf(config: Exp25Config) -> Path:
    """Return the backend leaf holding persisted Naturalness/OSRCT bundles.

    Shared by every Gamma arm: the store owner writes the per-replication ``osrct/`` and
    ``naturalness/`` bundles and its manifest here, and B-Learner-only replays read them.
    """

    return Path(config.output_dir).joinpath(
        str(config.stage1_backend),
        resolve_target_trigger_probability_backend(config),
        resolve_lp_probability_backend(config),
    )


def _output_leaf(config: Exp25Config) -> Path:
    """Return the Gamma-isolated compact-result leaf for one configuration.

    Every arm, including the store owner, writes beneath ``baseline_variants/gamma_<rule>/``;
    a full run and a B-Learner-only replay of the same Gamma share one leaf.

    Args:
        config: Exp25 configuration naming the backends and the Gamma rule.

    Returns:
        Directory receiving the compact tables, manifest, and figures.
    """

    leaf = _naturalness_bundle_leaf(config)
    label = f"gamma_{_gamma_output_label(config.blearner_gamma)}"
    # Dummy leaves are Gamma-separated too, so dummy arms sharing a root keep separate manifests.
    if config.dummy:
        return leaf / "dummy" / label
    return leaf / "baseline_variants" / label


def _scientific_config(config: Exp25Config) -> dict[str, Any]:
    """Return stable fields that must agree before resuming outputs.

    ``baseline_only`` is deliberately absent: a B-Learner-only replay produces the same quantity
    as a full run of its Gamma and resumes into the same leaf; the replay mode is recorded as
    manifest provenance instead.
    """

    return {
        "experiment": EXP25_NAME,
        "dummy": bool(config.dummy),
        "n_replications": int(config.n_replications),
        "base_seed": int(config.base_seed),
        "seeds": list(config.seeds),
        "studies": list(config.studies),
        "bias_covariates": list(config.bias_covariates),
        "betas": [float(value) for value in config.betas],
        "target_site": str(config.target_site),
        "rct_path": str(Path(config.rct_path).resolve()),
        "cohort_filter": (
            "resp_sex_not_missing"
            if config.require_observed_gender
            else "none"
        ),
        "require_observed_gender": bool(config.require_observed_gender),
        "stage1_backend": str(config.stage1_backend),
        "target_trigger_probability_backend": str(
            config.target_trigger_probability_backend
        ),
        "tabicl_device": str(config.tabicl_device),
        "lp_probability_backend": str(config.lp_probability_backend),
        "lp_posterior_samples": int(config.lp_posterior_samples),
        "lp_posterior_n_jobs": int(config.lp_posterior_n_jobs),
        "meta_learners": ["cb_ipw", "cb_dr"],
        "blearner_gamma_rule": config.blearner_gamma,
        "blearner_gamma_population": _gamma_population(config),
        "skip_lp_intervention_diagnostics": True,
        "fallback_handling": "method_specific_exclusion",
        "blearner_support": "clip_to_binary_cate_support_retain_raw",
    }


def _baseline_replay_provenance() -> dict[str, Any]:
    """Return the non-fingerprinted description of B-Learner-only replay."""

    return {
        "baseline_replay_mode": "blearner_only_v1",
        "baseline_replay_refit_methods": ["blearner"],
        "baseline_replay_reused_methods": list(EXP25_REUSED_BASELINE_METHODS),
    }


def _gamma_population(config: Exp25Config) -> str:
    """Describe the population or status of the configured Gamma rule."""

    if config.blearner_gamma == DGP_ORACLE_GAMMA:
        return (
            "observed_gender_complete_case_pre_selection_target_rct"
            if config.require_observed_gender
            else "full_pre_selection_target_rct"
        )
    if config.blearner_gamma == ORACLE_GAMMA:
        return "retained_target_idle_empirical_support"
    return "fixed_or_width_selected_sensitivity_parameter"


def _gamma_calibration_filename(config: Exp25Config) -> str | None:
    """Return calibration artifact name for oracle Gamma strategies."""

    if config.blearner_gamma == DGP_ORACLE_GAMMA:
        return EXP25_GAMMA_FILENAME
    if config.blearner_gamma == ORACLE_GAMMA:
        return EXP25_REALIZED_GAMMA_FILENAME
    return None


def config_fingerprint(config: Exp25Config) -> str:
    """Return SHA-256 fingerprint for resume compatibility."""

    payload = json.dumps(
        _scientific_config(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _naturalness_scientific_config(config: Exp25Config) -> dict[str, Any]:
    """Return the settings the shared OSRCT/Naturalness bundles depend on.

    Gamma never reaches Exp16, so the Gamma fields are dropped; the rest identifies the OSRCT
    generation and Naturalness computation a replay must agree with.

    Args:
        config: Exp25 configuration.

    Returns:
        Gamma-independent subset of the scientific configuration.
    """

    payload = dict(_scientific_config(replace(config, dummy=False)))
    for key in ("blearner_gamma_rule", "blearner_gamma_population"):
        payload.pop(key, None)
    return payload


def naturalness_bundle_fingerprint(config: Exp25Config) -> str:
    """Return the Gamma-independent identity of the shared bundle store.

    No Gamma setting is hashed, so whichever arm runs first populates the store and every other
    arm replays from it.

    Args:
        config: Exp25 configuration.

    Returns:
        Fingerprint the shared bundle manifest must carry.
    """

    payload = json.dumps(
        _naturalness_scientific_config(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_parquet_engine() -> None:
    """Raise a targeted error when PyArrow is unavailable."""

    _shared_require_parquet_engine("Exp25")


def _load_existing_outputs(
    output_dir: Path,
    *,
    fingerprint: str,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and reconcile previously committed replication tables."""

    return _shared_load_existing_outputs(
        output_dir,
        intervals_filename=EXP25_INTERVALS_FILENAME,
        diagnostics_filename=EXP25_DIAGNOSTICS_FILENAME,
        fingerprint=fingerprint,
        resume=resume,
        experiment_name="Exp25",
    )


def _write_replay_source_intervals(
    bundle_root: Path,
    intervals: pd.DataFrame,
    *,
    bundle_fingerprint: str,
) -> Path:
    """Publish the store owner's compact rows for later replays to reuse.

    Written beside the bundles, not in the owner's leaf, so a replay need not know which Gamma
    ran first. Rows carry the bundle fingerprint; replays copy the reusable methods verbatim and
    take only stratum identities from the B-Learner rows.

    Args:
        bundle_root: Shared bundle store.
        intervals: Complete method-long compact table of the owning arm.
        bundle_fingerprint: Gamma-independent store identity.

    Returns:
        Path of the written artifact.
    """

    published = intervals.copy()
    published["config_fingerprint"] = str(bundle_fingerprint)
    path = bundle_root / EXP25_REPLAY_SOURCE_FILENAME
    _atomic_write_parquet(published, path)
    return path


def _load_replay_source_intervals(
    config: Exp25Config,
    bundle_root: Path,
) -> pd.DataFrame:
    """Load and validate the store rows reused by B-Learner-only replay.

    Args:
        config: Alternate-Gamma replay configuration.
        bundle_root: Fingerprint-matched shared bundle store.

    Returns:
        Complete method-long interval table published by the store owner.

    Raises:
        FileNotFoundError: If the store holds no replay source table.
        ValueError: If its schema or fingerprint differs from the store.
    """

    path = bundle_root / EXP25_REPLAY_SOURCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            "B-Learner-only replay requires the store's replay source table: "
            f"{path}"
        )
    intervals = pd.read_parquet(path, engine="pyarrow")
    required = {
        "config_fingerprint",
        "replication",
        "seed",
        "bias_covariate",
        "beta",
        "method",
        *EXP25_IDENTITY_COLUMNS,
    }
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(
            "Exp25 replay source intervals lack replay fields: "
            f"{sorted(missing)}"
        )
    observed_fingerprints = set(
        intervals["config_fingerprint"].dropna().astype(str)
    )
    expected_fingerprint = naturalness_bundle_fingerprint(config)
    if observed_fingerprints != {expected_fingerprint}:
        raise ValueError(
            "Exp25 replay source intervals have a different scientific "
            "fingerprint"
        )
    return intervals


def _primary_condition_replay_rows(
    primary_intervals: pd.DataFrame,
    *,
    replication: int,
    seed: int,
    pattern: str,
    beta: float,
    variant_fingerprint: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select reusable methods and B-Learner identities for one condition.

    Args:
        primary_intervals: Validated primary Exp25 compact intervals.
        replication: Requested Monte Carlo replication.
        seed: Expected replication seed.
        pattern: OSRCT bias covariate.
        beta: OSRCT selection strength.
        variant_fingerprint: Fingerprint assigned to copied variant rows.

    Returns:
        Reused non-B-Learner rows and one identity row per B-Learner stratum.

    Raises:
        ValueError: If the primary condition is incomplete, duplicated, or
            inconsistent across methods.
    """

    condition = primary_intervals.loc[
        primary_intervals["replication"].astype(int).eq(int(replication))
        & primary_intervals["seed"].astype(int).eq(int(seed))
        & primary_intervals["bias_covariate"].astype(str).eq(str(pattern))
        & np.isclose(
            pd.to_numeric(primary_intervals["beta"], errors="coerce"),
            float(beta),
        )
    ].copy()
    if condition.empty:
        raise ValueError(
            "Primary Exp25 intervals lack replay condition "
            f"replication={replication}, seed={seed}, pattern={pattern}, "
            f"beta={beta}"
        )
    observed_methods = set(condition["method"].astype(str))
    if observed_methods != set(EXP25_METHODS):
        raise ValueError(
            "Primary Exp25 replay condition has incomplete methods: "
            f"{sorted(observed_methods)}"
        )
    stratum_keys = ["study", "target_site", *EXP25_STRATUM_COLUMNS]
    if condition.duplicated(["method", *stratum_keys]).any():
        raise ValueError("Primary Exp25 replay condition has duplicate strata")
    reference_keys = set(
        condition.loc[
            condition["method"].astype(str).eq("naturalness"),
            stratum_keys,
        ].itertuples(index=False, name=None)
    )
    for method in EXP25_METHODS:
        method_keys = set(
            condition.loc[
                condition["method"].astype(str).eq(method),
                stratum_keys,
            ].itertuples(index=False, name=None)
        )
        if method_keys != reference_keys:
            raise ValueError(
                "Primary Exp25 replay strata differ across methods for "
                f"{method}"
            )
    reused = condition.loc[
        condition["method"].astype(str).isin(EXP25_REUSED_BASELINE_METHODS)
    ].copy()
    reused["config_fingerprint"] = str(variant_fingerprint)
    blearner_identities = condition.loc[
        condition["method"].astype(str).eq("blearner"),
        list(EXP25_IDENTITY_COLUMNS),
    ].copy()
    return reused, blearner_identities


def _completed_replications(
    diagnostics: pd.DataFrame,
    config: Exp25Config,
) -> set[int]:
    """Return replications with every expected condition/study/method row."""

    expected = (
        len(config.bias_covariates)
        * len(config.betas)
        * len(config.studies)
        * len(EXP25_METHODS)
    )
    return _shared_completed_replications(
        diagnostics,
        condition_columns=("bias_covariate", "beta", "study", "method"),
        expected_conditions=expected,
    )


def _write_or_validate_manifest(
    output_dir: Path,
    config: Exp25Config,
    fingerprint: str,
) -> None:
    """Create a manifest or validate its embedded resume fingerprint."""

    manifest_path = output_dir / "run_manifest.yml"
    if manifest_path.exists():
        contents = manifest_path.read_text(encoding="utf-8")
        if fingerprint not in contents:
            raise ValueError("Existing run_manifest.yml has a different config")
        return
    cohort_description = (
        "observed-gender complete-case "
        if config.require_observed_gender
        else "unrestricted "
    )
    calibration_population = (
        "filtered" if config.require_observed_gender else "full"
    )
    interpretation = (
        "B-Learner-only replay over the shared Exp25 bundle store. "
        "Naturalness, CB-IPW, and CB-DR interval rows are reused exactly "
        "from the store's replay source; only B-Learner is refit "
        "under the requested Gamma rule."
        if config.baseline_only
        else (
            "Conditional Monte Carlo over OSRCT selection and estimator "
            f"randomness on one fixed, {cohort_description}ManyLabs RCT. "
            "B-Learner Gamma is calibrated from the known OSRCT DGP over "
            f"the {calibration_population} pre-selection target RCT, "
            "conditional on its exact fitting covariates."
        )
    )
    write_run_manifest(
        output_dir,
        experiment_name=EXP25_NAME,
        script_path="experiments/exp25_osrct_mc_baseline_comparison.py",
        config=asdict(config),
        argv=config.command_argv,
        cluster_manifest_path=config.cluster_manifest_path,
        extra={
            "config_fingerprint": fingerprint,
            "naturalness_bundle_fingerprint": naturalness_bundle_fingerprint(
                config
            ),
            "scientific_config": _scientific_config(config),
            # Replay mode is provenance, not identity, so a replay shares the
            # leaf of a full run of the same Gamma.
            **(
                {"baseline_replay": _baseline_replay_provenance()}
                if config.baseline_only
                else {}
            ),
            "interpretation": interpretation,
        },
    )


def _claim_full_mode_owner(config: Exp25Config, bundle_root: Path) -> Path:
    """Atomically claim a shared bundle store for one full-mode configuration.

    Args:
        config: Full-mode Exp25 configuration claiming the store.
        bundle_root: Shared OSRCT/Naturalness bundle root.

    Returns:
        Path containing the owner's Gamma-specific configuration fingerprint.

    Raises:
        ValueError: If another full-mode configuration already owns the store.
    """

    owner_path = bundle_root / EXP25_FULL_MODE_OWNER_FILENAME
    expected = config_fingerprint(config)
    try:
        descriptor = os.open(
            owner_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        observed = owner_path.read_text(encoding="utf-8").strip()
        if observed != expected:
            raise ValueError(
                "Exp25 bundle store already has a different full-mode owner; "
                "rerun this Gamma with --baseline-only"
            )
        return owner_path
    with os.fdopen(descriptor, "w", encoding="utf-8") as owner_file:
        owner_file.write(f"{expected}\n")
    return owner_path


def _write_or_validate_bundle_manifest(config: Exp25Config) -> Path:
    """Record or check the shared bundle store's Gamma-independent identity.

    The store keeps its own manifest so that later replays can validate the
    provenance of the bundles they read.

    Args:
        config: Exp25 configuration for a run that populates the store.

    Returns:
        The shared bundle root.

    Raises:
        ValueError: If an existing store was built from different OSRCT or
            Naturalness settings, or belongs to another full-mode owner.
    """

    bundle_root = _naturalness_bundle_leaf(config)
    bundle_root.mkdir(parents=True, exist_ok=True)
    expected = naturalness_bundle_fingerprint(config)
    manifest_path = bundle_root / "run_manifest.yml"
    if manifest_path.exists():
        contents = manifest_path.read_text(encoding="utf-8")
        if expected not in contents:
            raise ValueError(
                "Existing Exp25 bundle store was built from different OSRCT "
                "or Naturalness settings"
            )
        if config_fingerprint(config) not in contents:
            raise ValueError(
                "Exp25 bundle store already has a different full-mode owner; "
                "rerun this Gamma with --baseline-only"
            )
        _claim_full_mode_owner(config, bundle_root)
        return bundle_root
    _claim_full_mode_owner(config, bundle_root)
    write_run_manifest(
        bundle_root,
        experiment_name=EXP25_NAME,
        script_path="experiments/exp25_osrct_mc_baseline_comparison.py",
        config=asdict(config),
        argv=config.command_argv,
        cluster_manifest_path=config.cluster_manifest_path,
        extra={
            "naturalness_bundle_fingerprint": expected,
            "config_fingerprint": config_fingerprint(config),
            "scientific_config": _naturalness_scientific_config(config),
            "interpretation": (
                "Shared OSRCT and Naturalness bundle store plus the replay "
                "source rows. Gamma never reaches Exp16, so every B-Learner "
                "Gamma arm under baseline_variants/ replays these same "
                "bundles."
            ),
        },
    )
    return bundle_root


def _load_osrct_grid_generator() -> Any:
    """Load the existing OSRCT benchmark grid generator lazily."""

    code_dir = PROJECT_ROOT / "osrct_benchmark_v1.0" / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    module = importlib.import_module("generate_confounded_datasets")
    return module.generate_full_grid


def generate_replication_osrct_grid(
    config: Exp25Config,
    *,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate one seed-specific OSRCT grid as reusable Parquet tables."""

    generator = _load_osrct_grid_generator()
    return generator(
        data_path=str(config.rct_path),
        output_dir=str(output_dir),
        studies=list(config.studies),
        beta_values=[float(value) for value in config.betas],
        patterns=list(config.bias_covariates),
        seeds=[int(seed)],
        include_site_stratified=False,
        verbose=bool(config.verbose),
        output_format="parquet",
        require_observed_gender=bool(config.require_observed_gender),
    )


def _exp16_config_for_replication(
    config: Exp25Config,
    *,
    seed: int,
    data_root: Path,
    output_dir: Path,
) -> Exp16Config:
    """Build exact full-grid Exp16 configuration for one replication."""

    return Exp16Config(
        studies=tuple(config.studies),
        patterns=tuple(config.bias_covariates),
        betas=tuple(map(float, config.betas)),
        target_sites=(str(config.target_site),),
        data_root=str(data_root),
        rct_path=str(config.rct_path),
        require_observed_gender=bool(config.require_observed_gender),
        output_dir=str(output_dir),
        stage1_backend=str(config.stage1_backend),
        validation_stage1_backend=str(config.stage1_backend),
        tabicl_device=str(config.tabicl_device),
        target_trigger_probability_backend=str(
            config.target_trigger_probability_backend
        ),
        lp_probability_backend=str(config.lp_probability_backend),
        fit_gp=True,
        lp_posterior_samples=int(config.lp_posterior_samples),
        lp_posterior_n_jobs=int(config.lp_posterior_n_jobs),
        target_policy_uncertainty="posterior_mean",
        policy_top_k_values=(1, 2, 3),
        candidate_selection_mode="all_triggered",
        skip_lp_intervention_diagnostics=True,
        policy_tuning_fraction=0.30,
        min_policy_tuning_sites=10,
        calibration_fraction=0.25,
        min_calibration_sites=9,
        conformal_alpha=0.10,
        rct_min_arm_count=8,
        seed=int(seed),
        fail_fast=True,
        make_plots=False,
        verbose=bool(config.verbose),
    )


def _naturalness_references(
    run_index: pd.DataFrame,
    config: Exp25Config,
) -> dict[tuple[str, float], Path]:
    """Resolve every Exp16 posterior-summary leaf from its run index."""

    required = {"target_site", "pattern", "beta", "output_dir", "n_errors"}
    missing = required - set(run_index.columns)
    if missing:
        raise ValueError(f"Exp16 run_index is missing columns: {sorted(missing)}")
    target_rows = run_index.loc[
        run_index["target_site"].astype(str).eq(str(config.target_site))
    ].copy()
    target_rows["beta_numeric"] = pd.to_numeric(
        target_rows["beta"], errors="raise"
    )
    references: dict[tuple[str, float], Path] = {}
    for pattern in config.bias_covariates:
        for beta in config.betas:
            matched = target_rows.loc[
                target_rows["pattern"].astype(str).eq(str(pattern))
                & np.isclose(target_rows["beta_numeric"], float(beta))
            ]
            if len(matched) != 1:
                raise ValueError(
                    "Expected one Exp16 reference for "
                    f"pattern={pattern}, beta={beta}; found {len(matched)}"
                )
            row = matched.iloc[0]
            if int(row["n_errors"]) != 0:
                raise RuntimeError(
                    f"Exp16 reference has errors for {pattern}/beta={beta}"
                )
            path = Path(str(row["output_dir"])) / (
                "target_gp_posterior_summary.csv"
            )
            roles_path = path.with_name("source_site_roles.csv")
            if not path.exists() or not roles_path.exists():
                raise FileNotFoundError(
                    "Exp18 requires Exp16 posterior summary and source roles: "
                    f"{path}, {roles_path}"
                )
            references[(str(pattern), float(beta))] = path
    return references


def _persist_naturalness_artifacts(
    run_index: pd.DataFrame,
    config: Exp25Config,
    replication_dir: Path,
) -> dict[tuple[str, float], Path]:
    """Convert every Exp16 leaf table to a reusable Parquet bundle.

    Args:
        run_index: Exp16 index containing one output directory per condition.
        config: Exp25 configuration defining the expected target/grid.
        replication_dir: Persistent directory for this Monte Carlo replication.

    Returns:
        Condition-indexed persistent posterior-summary paths.
    """

    scratch_references = _naturalness_references(run_index, config)
    index_rows: list[dict[str, Any]] = []
    references: dict[tuple[str, float], Path] = {}
    persistent_output_dirs: dict[tuple[str, float], str] = {}
    for (pattern, beta), scratch_reference in scratch_references.items():
        destination = replication_dir.joinpath(
            "naturalness",
            "all_triggered",
            str(config.target_site),
            str(pattern),
            _format_beta(float(beta)),
        )
        destination.mkdir(parents=True, exist_ok=True)
        persistent_output_dirs[(str(pattern), float(beta))] = str(destination)
        for csv_path in sorted(scratch_reference.parent.glob("*.csv")):
            try:
                frame = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                index_rows.append(
                    {
                        "bias_covariate": str(pattern),
                        "beta": float(beta),
                        "source_name": csv_path.name,
                        "artifact_path": None,
                        "status": "empty_source_skipped",
                        "n_rows": 0,
                    }
                )
                continue
            parquet_path = destination / f"{csv_path.stem}.parquet"
            _atomic_write_parquet(frame, parquet_path)
            index_rows.append(
                {
                    "bias_covariate": str(pattern),
                    "beta": float(beta),
                    "source_name": csv_path.name,
                    "artifact_path": str(
                        parquet_path.relative_to(replication_dir)
                    ),
                    "status": "persisted",
                    "n_rows": int(len(frame)),
                }
            )
        persistent_reference = destination / (
            "target_gp_posterior_summary.parquet"
        )
        persistent_roles = destination / "source_site_roles.parquet"
        if not persistent_reference.exists() or not persistent_roles.exists():
            raise FileNotFoundError(
                "Persisted Naturalness bundle lacks the Exp18 reference or "
                f"source roles: {persistent_reference}, {persistent_roles}"
            )
        references[(str(pattern), float(beta))] = persistent_reference
    _atomic_write_parquet(
        pd.DataFrame(index_rows),
        replication_dir / "naturalness_artifact_index.parquet",
    )
    persistent_run_index = run_index.copy()
    persistent_run_index["output_dir"] = [
        persistent_output_dirs[(str(row.pattern), float(row.beta))]
        for row in persistent_run_index.itertuples(index=False)
    ]
    _atomic_write_parquet(
        persistent_run_index,
        replication_dir / "naturalness_run_index.parquet",
    )
    return references


def _load_persisted_naturalness_references(
    config: Exp25Config,
    replication_dir: Path,
) -> dict[tuple[str, float], Path]:
    """Resolve and validate a previously persisted Naturalness bundle."""

    required_indexes = (
        replication_dir / "naturalness_artifact_index.parquet",
        replication_dir / "naturalness_run_index.parquet",
    )
    missing_indexes = [path for path in required_indexes if not path.exists()]
    if missing_indexes:
        raise FileNotFoundError(
            "Baseline-only rerun requires persisted Naturalness artifacts: "
            f"{missing_indexes}"
        )
    references: dict[tuple[str, float], Path] = {}
    for pattern in config.bias_covariates:
        for beta in config.betas:
            destination = replication_dir.joinpath(
                "naturalness",
                "all_triggered",
                str(config.target_site),
                str(pattern),
                _format_beta(float(beta)),
            )
            reference = destination / "target_gp_posterior_summary.parquet"
            roles = destination / "source_site_roles.parquet"
            if not reference.exists() or not roles.exists():
                raise FileNotFoundError(
                    "Incomplete persisted Naturalness bundle for "
                    f"{pattern}/beta={beta}: {reference}, {roles}"
                )
            references[(str(pattern), float(beta))] = reference
    return references


def _exp18_config_for_condition(
    config: Exp25Config,
    *,
    seed: int,
    pattern: str,
    beta: float,
    data_root: Path,
    naturalness_reference: Path,
    output_dir: Path,
) -> Exp18Config:
    """Build matched Exp18 configuration for one Monte Carlo condition."""

    return Exp18Config(
        studies=tuple(config.studies),
        data_source="exp16",
        data_root=str(data_root),
        rct_path=str(config.rct_path),
        require_observed_gender=bool(config.require_observed_gender),
        pattern=str(pattern),
        beta=float(beta),
        target_site=str(config.target_site),
        source_site_policy="naturalness_final_training",
        source_regime="idle",
        pair_scope="target_anchored",
        seed=int(seed),
        n_jobs=int(config.exp18_n_jobs),
        worker_threads=int(config.exp18_worker_threads),
        meta_learners=("cb_ipw", "cb_dr"),
        gamma=config.blearner_gamma,
        naturalness_backend=str(config.stage1_backend),
        naturalness_trigger_backend=resolve_target_trigger_probability_backend(
            config
        ),
        naturalness_lp_backend=resolve_lp_probability_backend(config),
        naturalness_reference=str(naturalness_reference),
        output_dir=str(output_dir),
        make_plots=False,
        verbose=bool(config.verbose),
    )


def _validity_reason(
    *,
    finite: pd.Series,
    fallback: pd.Series,
    no_optimal_draw: pd.Series,
) -> pd.Series:
    """Return one explicit method-row exclusion reason."""

    reason = pd.Series("", index=finite.index, dtype="object")
    reason.loc[~finite] = "missing_or_nonfinite_endpoints"
    reason.loc[no_optimal_draw] = "no_optimal_lp_draw"
    reason.loc[fallback] = "full_range_fallback"
    return reason


def comparison_to_long_intervals(
    comparison: pd.DataFrame,
    *,
    replication: int,
    seed: int,
    pattern: str,
    beta: float,
    config_fingerprint_value: str,
    methods: Sequence[str] = EXP25_METHODS,
) -> pd.DataFrame:
    """Convert selected Exp18 methods into tidy method-level intervals."""

    normalized = _shared_comparison_to_long_intervals(
        comparison,
        methods=methods,
        identity_columns=EXP25_IDENTITY_COLUMNS,
        replication=replication,
        seed=seed,
        bias_covariate=pattern,
        beta=beta,
        config_fingerprint_value=config_fingerprint_value,
        experiment_name="Exp25",
    )
    return normalized.loc[:, list(EXP25_INTERVAL_COLUMNS)]


def _blearner_replay_comparison(
    identities: pd.DataFrame,
    selected_blearner_bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Merge new B-Learner endpoints onto primary target identities.

    Args:
        identities: Primary B-Learner identity and evaluation columns.
        selected_blearner_bounds: Newly selected B-Learner bounds.

    Returns:
        Minimal wide comparison accepted by the shared long converter.
    """

    keys = ["study", *EXP25_STRATUM_COLUMNS]
    blearner_columns = [
        *keys,
        "gamma",
        "blearner_lower",
        "blearner_upper",
        "outside_binary_cate_support",
    ]
    missing = set(blearner_columns).difference(selected_blearner_bounds.columns)
    if missing:
        raise ValueError(
            "B-Learner-only replay lacks selected bound fields: "
            f"{sorted(missing)}"
        )
    if selected_blearner_bounds.duplicated(keys).any():
        raise ValueError("B-Learner-only replay produced duplicate strata")
    comparison = identities.merge(
        selected_blearner_bounds.loc[:, blearner_columns].rename(
            columns={"gamma": "blearner_gamma"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(identities):
        raise ValueError(
            "B-Learner-only replay does not cover every primary stratum"
        )
    return comparison


def _balke_pearl_exclusion_reason(bounds: pd.DataFrame) -> pd.Series:
    """Name why a stratum has no Balke-Pearl interval, or return ``""``.

    Args:
        bounds: Rows from ``balke_pearl_bounds`` with ``bp_status`` and
            ``bp_candidates``.

    Returns:
        ``no_supported_instrument`` when no candidate cleared relevance and the
        support floor, ``instrument_refuted`` when every one that did violated
        the instrumental inequality, ``instrument_bounds_disjoint`` when the
        surviving instruments' intervals do not overlap, else ``""``.
    """

    reason = pd.Series("", index=bounds.index, dtype="object")
    unsupported = bounds["bp_status"].eq("unsupported")
    reason.loc[unsupported & bounds["bp_candidates"].eq(0)] = "no_supported_instrument"
    reason.loc[unsupported & bounds["bp_candidates"].gt(0)] = "instrument_refuted"
    reason.loc[bounds["bp_status"].eq("infeasible")] = "instrument_bounds_disjoint"
    return reason


def identification_bound_intervals(
    target_idle_by_study: dict[str, pd.DataFrame],
    identities: pd.DataFrame,
    *,
    min_count: int,
    tau: float,
    replication: int,
    seed: int,
    pattern: str,
    beta: float,
    config_fingerprint_value: str,
) -> pd.DataFrame:
    """Score Manski and Balke-Pearl bounds on one condition's primary strata.

    Both bounds read only the target-site idle rows Exp18 hands B-Learner and use the Exp28
    bound functions. Manski is the no-assumption bound; for a binary outcome it has width one
    and contains zero, so it is a reference floor, not a competitor. Balke-Pearl treats each
    stratum covariate that clears the empirical relevance trigger ``D_j(tau)`` as a binary
    instrument within its ``Z_-j`` block, intersects the sharp IV bounds on
    ``E[Y(1)-Y(0) | Z_-j]`` and assigns them to every full cell of the block (valid under its
    exogeneity assumption). A stratum with no usable instrument is ``is_valid=False`` with the
    reason from :func:`_balke_pearl_exclusion_reason`; unlike Exp28, there is no Manski fallback.

    Args:
        target_idle_by_study: Target-site idle rows keyed by study.
        identities: One row per primary stratum with the :data:`EXP25_IDENTITY_COLUMNS`, from
            :func:`_primary_condition_replay_rows`.
        min_count: Balke-Pearl arm support floor and Manski support mask.
        tau: Relevance threshold for Balke-Pearl candidates.
        replication: Monte Carlo replication index.
        seed: Replication seed.
        pattern: OSRCT bias covariate.
        beta: OSRCT selection strength.
        config_fingerprint_value: Fingerprint written on every row.

    Returns:
        Method-long rows in the :data:`EXP25_INTERVAL_COLUMNS` contract, plus
        ``bp_candidates`` and ``bp_status`` for Balke-Pearl diagnostics.

    Raises:
        ValueError: If a study's target idle rows are missing or a bound
            does not cover every primary stratum.
    """

    covariates = list(EXP25_STRATUM_COLUMNS)
    frames: list[pd.DataFrame] = []
    for study, study_identities in identities.groupby("study", sort=True):
        if str(study) not in target_idle_by_study:
            raise ValueError(f"No target idle rows for study {study}")
        idle = target_idle_by_study[str(study)]
        # The bound helpers compare and bit-pack the covariates as integers.
        idle = idle.assign(**{name: idle[name].astype(int) for name in covariates},
                           iv=idle["iv"].astype(int), dv_binary=idle["dv_binary"].astype(int))
        strata = study_identities[covariates].astype(int).drop_duplicates()
        strata = strata.reset_index(drop=True)
        # The recomputed bins must reproduce the primary run's target counts;
        # different pooled age edges or rows would silently shift the strata.
        counts = idle.groupby(covariates).size().rename("n_rows").reset_index()
        expected = study_identities.assign(
            **{name: study_identities[name].astype(int) for name in covariates}
        ).merge(counts, on=covariates, how="left")
        if not expected["n_rows"].fillna(-1).astype(int).eq(
            expected["n_target_idle_rows"].astype(int)
        ).all():
            raise ValueError(
                f"{study} target idle rows do not reproduce the primary strata counts"
            )
        manski = manski_bounds(idle, strata, covariates, min_count)
        manski = manski.assign(is_valid=True, exclusion_reason="",
                               bp_candidates=np.nan, bp_status="")
        balke_pearl = balke_pearl_bounds(idle, strata, covariates, min_count, tau)
        reason = _balke_pearl_exclusion_reason(balke_pearl)
        valid = reason.eq("")
        balke_pearl = balke_pearl.assign(
            lower=balke_pearl["lower"].where(valid),
            upper=balke_pearl["upper"].where(valid),
            is_valid=valid,
            exclusion_reason=reason,
        )
        keyed = study_identities.assign(
            **{name: study_identities[name].astype(int) for name in covariates}
        )
        for method, bounds in (("manski", manski), ("balke_pearl", balke_pearl)):
            rows = keyed.merge(
                bounds[[*covariates, "lower", "upper", "is_valid", "exclusion_reason",
                        "bp_candidates", "bp_status"]],
                on=covariates,
                how="left",
                validate="one_to_one",
            )
            if rows["is_valid"].isna().any():
                raise ValueError(f"{method} does not cover every {study} primary stratum")
            # Restore the primary key dtypes so the analysis merges cleanly.
            for name in covariates:
                rows[name] = study_identities[name].to_numpy()
            rows = rows.assign(
                config_fingerprint=str(config_fingerprint_value),
                replication=int(replication),
                seed=int(seed),
                bias_covariate=str(pattern),
                beta=float(beta),
                method=method,
                raw_lower=rows["lower"],
                raw_upper=rows["upper"],
                width=rows["upper"] - rows["lower"],
                is_valid=rows["is_valid"].astype(bool),
                used_full_range_fallback=False,
                support_clipped=False,
                blearner_gamma=np.nan,
                lp_posterior_prob_optimal=np.nan,
            )
            frames.append(rows.loc[:, [*EXP25_INTERVAL_COLUMNS, "bp_candidates", "bp_status"]])
    return pd.concat(frames, ignore_index=True, sort=False)


def build_run_diagnostics(
    intervals: pd.DataFrame,
    *,
    runtime_seconds: float,
) -> pd.DataFrame:
    """Summarize one completed condition by study and method."""

    return _shared_build_run_diagnostics(
        intervals,
        grouping_columns=(
            "config_fingerprint", "replication", "seed", "bias_covariate",
            "beta", "study", "method",
        ),
        total_label="strata",
        runtime_seconds=runtime_seconds,
    )


def _commit_replication(
    output_dir: Path,
    existing_intervals: pd.DataFrame,
    existing_diagnostics: pd.DataFrame,
    replication_intervals: pd.DataFrame,
    replication_diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atomically insert or replace one complete replication."""

    return _shared_commit_replication(
        output_dir,
        existing_intervals,
        existing_diagnostics,
        replication_intervals,
        replication_diagnostics,
        interval_sort_columns=(
            "replication",
            "bias_covariate",
            "beta",
            "study",
            *EXP25_STRATUM_COLUMNS,
            "method",
        ),
        diagnostics_sort_columns=(
            "replication",
            "bias_covariate",
            "beta",
            "study",
            "method",
        ),
        intervals_filename=EXP25_INTERVALS_FILENAME,
        diagnostics_filename=EXP25_DIAGNOSTICS_FILENAME,
    )


def _aggregate_gamma_calibrations(
    output_dir: Path,
    filename: str | None,
) -> pd.DataFrame:
    """Collect persisted per-replication oracle Gamma tables."""

    if filename is None:
        return pd.DataFrame()
    paths = sorted(
        (output_dir / "replications").glob(
            f"replication=*/" + filename
        )
    )
    if not paths:
        return pd.DataFrame()
    calibration = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
        sort=False,
    )
    calibration = calibration.sort_values(
        ["replication", "bias_covariate", "beta", "study"]
    ).reset_index(drop=True)
    _atomic_write_parquet(calibration, output_dir / filename)
    return calibration


def _validate_naturalness_bundle_root(config: Exp25Config) -> Path:
    """Validate the shared bundle store before a baseline-only replay.

    Args:
        config: Exp25 configuration for the replay.

    Returns:
        Validated shared bundle store holding the replication bundles.

    Raises:
        FileNotFoundError: If the store's manifest is absent.
        ValueError: If the store was built from different OSRCT or Naturalness
            settings.
    """

    bundle_root = _naturalness_bundle_leaf(config)
    manifest_path = bundle_root / "run_manifest.yml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Baseline-only replay requires the shared Exp25 bundle manifest: "
            f"{manifest_path}"
        )
    expected = naturalness_bundle_fingerprint(config)
    if expected not in manifest_path.read_text(encoding="utf-8"):
        raise ValueError(
            "Saved Exp25 bundle store has different OSRCT or Naturalness "
            "settings"
        )
    return bundle_root


def _build_naturalness_bundle(
    config: Exp25Config,
    *,
    seed: int,
    bundle_replication_dir: Path,
    scratch_root: Path,
) -> dict[tuple[str, float], Path]:
    """Generate one replication's OSRCT grid, fit Exp16 and persist it.

    This is the GPU part of a replication, shared by a full run and the
    split run's Naturalness pass.

    Args:
        config: Exp25 configuration.
        seed: Replication seed for OSRCT generation and Exp16.
        bundle_replication_dir: Persistent ``replication=XXX`` directory.
        scratch_root: Temporary directory for Exp16's own outputs.

    Returns:
        Condition-indexed persistent Naturalness reference paths.

    Raises:
        RuntimeError: If any OSRCT condition fails to generate.
    """

    data_root = bundle_replication_dir / "osrct"
    generation = generate_replication_osrct_grid(
        config,
        seed=seed,
        output_dir=data_root,
    )
    if "success" in generation and not generation["success"].astype(bool).all():
        failures = generation.loc[~generation["success"].astype(bool)]
        raise RuntimeError(
            f"OSRCT generation failed for {len(failures)} condition(s)"
        )
    exp16_result = run_exp16(
        _exp16_config_for_replication(
            config,
            seed=seed,
            data_root=data_root,
            output_dir=scratch_root / "exp16",
        )
    )
    return _persist_naturalness_artifacts(
        exp16_result["run_index"],
        config,
        bundle_replication_dir,
    )


def run_one_replication(
    config: Exp25Config,
    *,
    replication: int,
    seed: int,
    scratch_root: Path,
    bundle_replication_dir: Path,
    result_replication_dir: Path,
    fingerprint: str,
    primary_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full Exp18 or replay only B-Learner from primary intervals.

    ``comparators_only`` reads the persisted OSRCT and Naturalness bundle, as
    ``baseline_only`` does, but then fits the complete Exp18 comparison.
    """

    result_replication_dir.mkdir(parents=True, exist_ok=True)
    bundle_replication_dir.mkdir(parents=True, exist_ok=True)
    data_root = bundle_replication_dir / "osrct"
    exp18_output = scratch_root / "exp18"
    if config.baseline_only or config.comparators_only:
        if not data_root.exists():
            raise FileNotFoundError(
                f"Baseline-only rerun requires persisted OSRCT data: {data_root}"
            )
        if config.baseline_only and primary_intervals is None:
            primary_intervals = _load_replay_source_intervals(
                config,
                _naturalness_bundle_leaf(config),
            )
        references = _load_persisted_naturalness_references(
            config,
            bundle_replication_dir,
        )
    else:
        references = _build_naturalness_bundle(
            config,
            seed=seed,
            bundle_replication_dir=bundle_replication_dir,
            scratch_root=scratch_root,
        )
    condition_intervals: list[pd.DataFrame] = []
    condition_diagnostics: list[pd.DataFrame] = []
    gamma_calibrations: list[pd.DataFrame] = []
    for pattern in config.bias_covariates:
        for beta in config.betas:
            condition_start = time.perf_counter()
            exp18_config = _exp18_config_for_condition(
                config,
                seed=seed,
                pattern=str(pattern),
                beta=float(beta),
                data_root=data_root,
                naturalness_reference=references[
                    (str(pattern), float(beta))
                ],
                output_dir=exp18_output,
            )
            if config.baseline_only:
                reused_intervals, blearner_identities = (
                    _primary_condition_replay_rows(
                        primary_intervals,
                        replication=replication,
                        seed=seed,
                        pattern=str(pattern),
                        beta=float(beta),
                        variant_fingerprint=fingerprint,
                    )
                )
                exp18_result = fit_blearner_only(exp18_config)
                blearner_comparison = _blearner_replay_comparison(
                    blearner_identities,
                    exp18_result["selected_blearner_bounds"],
                )
                blearner_intervals = comparison_to_long_intervals(
                    blearner_comparison,
                    replication=replication,
                    seed=seed,
                    pattern=str(pattern),
                    beta=float(beta),
                    config_fingerprint_value=fingerprint,
                    methods=("blearner",),
                )
                intervals = pd.concat(
                    [reused_intervals, blearner_intervals],
                    ignore_index=True,
                    sort=False,
                )
            else:
                exp18_result = run_exp18(exp18_config)
                intervals = comparison_to_long_intervals(
                    exp18_result["comparison"],
                    replication=replication,
                    seed=seed,
                    pattern=str(pattern),
                    beta=float(beta),
                    config_fingerprint_value=fingerprint,
                )
            diagnostics = build_run_diagnostics(
                intervals,
                runtime_seconds=time.perf_counter() - condition_start,
            )
            condition_intervals.append(intervals)
            condition_diagnostics.append(diagnostics)
            gamma_frame = exp18_result["blearner_oracle_gamma"].copy()
            gamma_frame["replication"] = int(replication)
            gamma_frame["seed"] = int(seed)
            gamma_frame["bias_covariate"] = str(pattern)
            gamma_frame["beta"] = float(beta)
            gamma_frame["config_fingerprint"] = fingerprint
            gamma_calibrations.append(gamma_frame)
    gamma_filename = _gamma_calibration_filename(config)
    if gamma_filename is not None:
        _atomic_write_parquet(
            pd.concat(gamma_calibrations, ignore_index=True, sort=False),
            result_replication_dir / gamma_filename,
        )
    return (
        pd.concat(condition_intervals, ignore_index=True, sort=False),
        pd.concat(condition_diagnostics, ignore_index=True, sort=False),
    )


def make_width_distribution_plots(
    intervals: pd.DataFrame,
    output_dir: Path,
    *,
    studies: Sequence[str],
    bias_covariates: Sequence[str],
    betas: Sequence[float],
) -> list[Path]:
    """Create one dynamic beta-grid width violin plot per pattern and study."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ordered_betas = tuple(map(float, betas))
    n_columns = min(2, len(ordered_betas))
    n_rows = math.ceil(len(ordered_betas) / n_columns)
    for pattern in bias_covariates:
        for study in studies:
            selected = intervals.loc[
                intervals["bias_covariate"].astype(str).eq(str(pattern))
                & intervals["study"].astype(str).eq(str(study))
            ].copy()
            valid_widths = pd.to_numeric(
                selected.loc[selected["is_valid"].astype(bool), "width"],
                errors="coerce",
            ).dropna()
            maximum = max(
                2.0,
                float(valid_widths.max()) if not valid_widths.empty else 2.0,
            )
            figure, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(6.0 * n_columns, 4.2 * n_rows),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for axis, beta in zip(axes.flat, ordered_betas):
                panel = selected.loc[
                    np.isclose(
                        pd.to_numeric(selected["beta"], errors="coerce"),
                        beta,
                    )
                ]
                _draw_width_violin_panel(
                    axis,
                    panel,
                    methods=EXP25_METHODS,
                    method_labels=EXP25_METHOD_LABELS,
                    method_colors=EXP25_METHOD_COLORS,
                    maximum=maximum,
                    axis_label="CATE interval width",
                )
                axis.set_title(
                    rf"$\beta = {_format_beta(beta)}$",
                    fontweight="bold",
                )
            for axis in axes.flat[len(ordered_betas):]:
                axis.set_visible(False)
            handles = [
                Patch(
                    facecolor=EXP25_METHOD_COLORS[method],
                    edgecolor=EXP25_METHOD_COLORS[method],
                    alpha=0.6,
                )
                for method in EXP25_METHODS
            ]
            labels = [EXP25_METHOD_LABELS[method] for method in EXP25_METHODS]
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.945),
                ncol=4,
                frameon=False,
            )
            figure.suptitle(
                f"{str(pattern).replace('_', ' ').title()} bias — "
                f"{study}: interval-width distributions",
                fontweight="bold",
                y=0.992,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.875))
            path = (
                figures_dir
                / f"interval_width_distributions_{pattern}_{study}.png"
            )
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def make_side_by_side_width_distribution_plots(
    intervals: pd.DataFrame,
    output_dir: Path,
    *,
    studies: Sequence[str],
    bias_covariates: Sequence[str],
    betas: Sequence[float],
) -> list[Path]:
    """Place all study-specific interval-width distributions side by side.

    Args:
        intervals: Method-long Monte Carlo interval rows.
        output_dir: Exp25 compact output root.
        studies: Ordered ManyLabs study panels.
        bias_covariates: OSRCT selection patterns.
        betas: OSRCT selection strengths.

    Returns:
        One combined figure path per bias-covariate/beta condition.
    """

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pattern in bias_covariates:
        for beta in betas:
            selected = intervals.loc[
                intervals["bias_covariate"].astype(str).eq(str(pattern))
                & np.isclose(
                    pd.to_numeric(intervals["beta"], errors="coerce"),
                    float(beta),
                )
            ].copy()
            valid_widths = pd.to_numeric(
                selected.loc[selected["is_valid"].astype(bool), "width"],
                errors="coerce",
            ).dropna()
            maximum = max(
                2.0,
                float(valid_widths.max()) if not valid_widths.empty else 2.0,
            )
            figure, axes = plt.subplots(
                1,
                len(studies),
                figsize=(5.4 * len(studies), 4.8),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for axis, study in zip(axes.flat, studies):
                panel = selected.loc[
                    selected["study"].astype(str).eq(str(study))
                ]
                _draw_width_violin_panel(
                    axis,
                    panel,
                    methods=EXP25_METHODS,
                    method_labels=EXP25_METHOD_LABELS,
                    method_colors=EXP25_METHOD_COLORS,
                    maximum=maximum,
                    axis_label="CATE interval width",
                )
                axis.set_title(str(study), fontweight="bold")
            handles = [
                Patch(
                    facecolor=EXP25_METHOD_COLORS[method],
                    edgecolor=EXP25_METHOD_COLORS[method],
                    alpha=0.6,
                )
                for method in EXP25_METHODS
            ]
            labels = [EXP25_METHOD_LABELS[method] for method in EXP25_METHODS]
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=4,
                frameon=False,
            )
            figure.suptitle(
                f"{str(pattern).replace('_', ' ').title()} bias, "
                rf"$\beta = {_format_beta(float(beta))}$"
                " — interval-width distributions by study",
                fontweight="bold",
                y=0.995,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
            path = (
                figures_dir
                / "interval_width_distributions_side_by_side_"
                f"{pattern}_beta_{_format_beta(float(beta))}.png"
            )
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def build_mean_comparison(
    intervals: pd.DataFrame,
    *,
    pattern: str,
    beta: float,
) -> pd.DataFrame:
    """Build Exp18-compatible wide mean endpoints from valid MC rows."""

    keys = ["study", *EXP25_STRATUM_COLUMNS]
    result = _shared_build_mean_bounds_comparison(
        intervals,
        bias_covariate=pattern,
        beta=beta,
        keys=keys,
        reference_columns=(
            "rct_cate",
            "rct_cate_ci90_lower",
            "rct_cate_ci90_upper",
            "rct_cate_available",
        ),
        reference_label="RCT reference",
        methods=EXP25_METHODS,
    )
    return result.sort_values(keys).reset_index(drop=True)


def make_mean_comparison_plots(
    intervals: pd.DataFrame,
    output_dir: Path,
    *,
    studies: Sequence[str],
    bias_covariates: Sequence[str],
    betas: Sequence[float],
) -> list[Path]:
    """Create one Exp18-style horizontal mean-bound plot per condition."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pattern in bias_covariates:
        for beta in betas:
            comparison = build_mean_comparison(
                intervals,
                pattern=str(pattern),
                beta=float(beta),
            )
            plot_config = Exp18Config(
                studies=tuple(studies),
                pattern=str(pattern),
                beta=float(beta),
                target_site=str(
                    intervals["target_site"].dropna().astype(str).iloc[0]
                ),
                meta_learners=("cb_ipw", "cb_dr"),
                gamma=DGP_ORACLE_GAMMA,
                make_plots=False,
                verbose=False,
            )
            path = (
                figures_dir
                / "comparison_horizontal_"
                f"{pattern}_beta_{_format_beta(float(beta))}.png"
            )
            make_comparison_plot(
                comparison,
                plot_config,
                path,
                horizontal=True,
                figure_title=(
                    f"{str(pattern).replace('_', ' ').title()} bias, "
                    rf"$\beta = {_format_beta(float(beta))}$"
                    " — MC mean bounds"
                ),
            )
            paths.append(path)
    return paths


def make_all_plots(
    intervals: pd.DataFrame,
    output_dir: Path,
    config: Exp25Config,
) -> list[Path]:
    """Create all aggregate Exp25 figures."""

    paths = make_width_distribution_plots(
        intervals,
        output_dir,
        studies=config.studies,
        bias_covariates=config.bias_covariates,
        betas=config.betas,
    )
    paths.extend(
        make_side_by_side_width_distribution_plots(
            intervals,
            output_dir,
            studies=config.studies,
            bias_covariates=config.bias_covariates,
            betas=config.betas,
        )
    )
    paths.extend(
        make_mean_comparison_plots(
            intervals,
            output_dir,
            studies=config.studies,
            bias_covariates=config.bias_covariates,
            betas=config.betas,
        )
    )
    return paths


def generate_dummy_intervals(
    config: Exp25Config,
    *,
    fingerprint: str,
) -> pd.DataFrame:
    """Generate deterministic simulated method intervals for plot inspection."""

    generator = np.random.default_rng(config.base_seed)
    rows: list[dict[str, Any]] = []
    method_width = {
        "naturalness": 0.50,
        "blearner": 0.58,
        "cb_ipw": 0.72,
        "cb_dr": 0.64,
    }
    strata = tuple(itertools.product((0, 1), repeat=3))
    for replication, seed in enumerate(config.seeds):
        for pattern_index, pattern in enumerate(config.bias_covariates):
            for beta in config.betas:
                for study_index, study in enumerate(config.studies):
                    for stratum_index, stratum in enumerate(strata):
                        age, gender, polideo = stratum
                        truth = float(
                            np.clip(
                                -0.12
                                + 0.07 * age
                                - 0.05 * gender
                                + 0.04 * polideo
                                + 0.015 * study_index,
                                -0.8,
                                0.8,
                            )
                        )
                        target_count = int(
                            18
                            + 3 * stratum_index
                            + (replication + study_index) % 7
                        )
                        for method_index, method in enumerate(EXP25_METHODS):
                            center = truth + generator.normal(0.0, 0.035)
                            width = (
                                method_width[method]
                                + 0.13 * float(beta)
                                + 0.025 * pattern_index
                                + generator.normal(0.0, 0.055)
                            )
                            width = float(np.clip(width, 0.05, 1.9))
                            raw_lower = center - width / 2.0
                            raw_upper = center + width / 2.0
                            fallback = bool(
                                method in {"cb_ipw", "cb_dr"}
                                and replication % 11 == 0
                                and stratum_index == method_index
                            )
                            no_optimal = bool(
                                method == "naturalness"
                                and replication == 7
                                and stratum_index == 7
                            )
                            if (
                                method == "blearner"
                                and replication % 9 == 0
                                and stratum_index == 0
                            ):
                                raw_lower -= 0.8
                            lower = float(np.clip(raw_lower, -1.0, 1.0))
                            upper = float(np.clip(raw_upper, -1.0, 1.0))
                            support_clipped = bool(
                                method == "blearner"
                                and (
                                    not np.isclose(lower, raw_lower)
                                    or not np.isclose(upper, raw_upper)
                                )
                            )
                            is_valid = not fallback and not no_optimal
                            exclusion_reason = (
                                "full_range_fallback"
                                if fallback
                                else "no_optimal_lp_draw" if no_optimal else ""
                            )
                            if fallback:
                                lower, upper = -1.0, 1.0
                            rows.append(
                                {
                                    "config_fingerprint": fingerprint,
                                    "replication": replication,
                                    "seed": seed,
                                    "bias_covariate": pattern,
                                    "beta": float(beta),
                                    "study": study,
                                    "target_site": config.target_site,
                                    "resp_age_bin": age,
                                    "resp_gender_bin": gender,
                                    "resp_polideo_bin": polideo,
                                    "method": method,
                                    "raw_lower": raw_lower,
                                    "raw_upper": raw_upper,
                                    "lower": lower,
                                    "upper": upper,
                                    "width": upper - lower,
                                    "is_valid": is_valid,
                                    "exclusion_reason": exclusion_reason,
                                    "used_full_range_fallback": fallback,
                                    "support_clipped": support_clipped,
                                    "n_target_idle_rows": target_count,
                                    "blearner_gamma": (
                                        1.0 + float(beta)
                                        if method == "blearner"
                                        else np.nan
                                    ),
                                    "lp_posterior_prob_optimal": (
                                        0.0
                                        if no_optimal
                                        else 0.92 + 0.03 * generator.random()
                                        if method == "naturalness"
                                        else np.nan
                                    ),
                                    "rct_cate": truth,
                                    "rct_cate_ci90_lower": truth - 0.09,
                                    "rct_cate_ci90_upper": truth + 0.09,
                                    "rct_cate_available": True,
                                }
                            )
    return pd.DataFrame(rows)


def _dummy_diagnostics(intervals: pd.DataFrame) -> pd.DataFrame:
    """Build run diagnostics for all deterministic dummy conditions."""

    frames = []
    for _, condition in intervals.groupby(
        ["replication", "bias_covariate", "beta"],
        sort=True,
    ):
        frames.append(build_run_diagnostics(condition, runtime_seconds=0.0))
    return pd.concat(frames, ignore_index=True, sort=False)


def run_dummy(config: Exp25Config) -> dict[str, Any]:
    """Write simulated compact outputs and all requested figures."""

    _require_parquet_engine()
    output_dir = _output_leaf(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config)
    _write_or_validate_manifest(output_dir, config, fingerprint)
    intervals = generate_dummy_intervals(config, fingerprint=fingerprint)
    diagnostics = _dummy_diagnostics(intervals)
    _atomic_write_parquet(
        intervals,
        output_dir / EXP25_INTERVALS_FILENAME,
    )
    _atomic_write_parquet(
        diagnostics,
        output_dir / EXP25_DIAGNOSTICS_FILENAME,
    )
    figure_paths = make_all_plots(intervals, output_dir, config)
    return {
        "output_dir": output_dir,
        "intervals": intervals,
        "diagnostics": diagnostics,
        "figure_paths": figure_paths,
    }


def _identification_scientific_config(config: Exp25Config) -> dict[str, Any]:
    """Return the settings the Manski/Balke-Pearl leaf depends on.

    Args:
        config: Exp25 configuration naming the bundle store and bound settings.

    Returns:
        The bundle identity plus the bound methods and their parameters.
    """

    return {
        "naturalness_bundle_fingerprint": naturalness_bundle_fingerprint(config),
        "methods": list(EXP25_IDENTIFICATION_METHODS),
        "min_count": int(config.identification_min_count),
        "tau": float(config.identification_tau),
        "candidate_trigger": "empirical_d_strength",
        "failure_handling": "invalid_no_manski_fallback",
    }


def identification_bounds_fingerprint(config: Exp25Config) -> str:
    """Return the SHA-256 identity of one Manski/Balke-Pearl leaf."""

    payload = json.dumps(
        _identification_scientific_config(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_identification_bounds_replay(
    config: Exp25Config,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute Manski and Balke-Pearl bounds for every saved replication.

    Reads each replication's persisted OSRCT data and the store's replay source intervals (which
    fix the primary strata and their held-out RCT references) and by default writes
    ``baseline_variants/identification_bounds/`` beside the Gamma leaves. No existing table is
    modified, and nothing runs on the GPU or in R.

    Args:
        config: Exp25 configuration matching the store's scientific settings.
        output_dir: Leaf to write instead of the default, e.g. for other bound settings such as
            another ``identification_tau``.

    Returns:
        The output leaf, interval rows, per-condition diagnostics, and an empty figure list.

    Raises:
        ValueError: If the store or an existing leaf was built with different
            settings, or the saved data disagree with the primary strata.
    """

    _require_parquet_engine()
    bundle_root = _validate_naturalness_bundle_root(config)
    primary_intervals = _load_replay_source_intervals(config, bundle_root)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else bundle_root / "baseline_variants" / EXP25_IDENTIFICATION_DIRNAME
    )
    fingerprint = identification_bounds_fingerprint(config)
    manifest_path = output_dir / "run_manifest.yml"
    if manifest_path.exists() and fingerprint not in manifest_path.read_text(encoding="utf-8"):
        raise ValueError(
            "Existing identification_bounds leaf has different settings; "
            "use a separate output root"
        )
    interval_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for replication, seed in enumerate(config.seeds):
        replication_dir = bundle_root / "replications" / f"replication={replication:03d}"
        references = _load_persisted_naturalness_references(config, replication_dir)
        for pattern in config.bias_covariates:
            for beta in config.betas:
                start = time.perf_counter()
                _, identities = _primary_condition_replay_rows(
                    primary_intervals,
                    replication=replication,
                    seed=seed,
                    pattern=str(pattern),
                    beta=float(beta),
                    variant_fingerprint=fingerprint,
                )
                exp18_config = _exp18_config_for_condition(
                    config,
                    seed=seed,
                    pattern=str(pattern),
                    beta=float(beta),
                    data_root=replication_dir / "osrct",
                    naturalness_reference=references[(str(pattern), float(beta))],
                    output_dir=output_dir,
                )
                if tuple(_comparison_covariates(exp18_config)) != EXP25_STRATUM_COLUMNS:
                    raise ValueError("Exp18 reporting covariates differ from Exp25 strata")
                prepared, _ = _load_fitting_data(exp18_config, "exp16")
                # The exact target rows Exp18 hands B-Learner.
                target_idle = {
                    study: data.idle.loc[
                        data.idle["site"].astype(str).eq(
                            _study_target_site(exp18_config, study)
                        )
                    ]
                    for study, data in prepared.items()
                }
                intervals = identification_bound_intervals(
                    target_idle,
                    identities,
                    min_count=config.identification_min_count,
                    tau=config.identification_tau,
                    replication=replication,
                    seed=seed,
                    pattern=str(pattern),
                    beta=float(beta),
                    config_fingerprint_value=fingerprint,
                )
                interval_frames.append(intervals)
                diagnostic_frames.append(
                    build_run_diagnostics(
                        intervals,
                        runtime_seconds=time.perf_counter() - start,
                    )
                )
        if config.verbose:
            print(
                f"[Exp25 bounds] replication {replication + 1}/"
                f"{config.n_replications} done",
                flush=True,
            )
    intervals = pd.concat(interval_frames, ignore_index=True, sort=False)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True, sort=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        write_run_manifest(
            output_dir,
            experiment_name=EXP25_NAME,
            script_path="experiments/exp25_osrct_mc_baseline_comparison.py",
            config=asdict(config),
            argv=config.command_argv,
            cluster_manifest_path=config.cluster_manifest_path,
            extra={
                "identification_bounds_fingerprint": fingerprint,
                "identification_config": _identification_scientific_config(config),
                "interpretation": (
                    "Manski and Balke-Pearl bounds from each replication's "
                    "saved target idle rows, on the primary strata. A stratum "
                    "without a usable Balke-Pearl instrument is invalid, not "
                    "a Manski fallback."
                ),
            },
        )
    _atomic_write_parquet(intervals, output_dir / EXP25_INTERVALS_FILENAME)
    _atomic_write_parquet(diagnostics, output_dir / EXP25_DIAGNOSTICS_FILENAME)
    return {
        "output_dir": output_dir,
        "intervals": intervals,
        "diagnostics": diagnostics,
        "figure_paths": [],
    }


def _wait_for_bundle_manifest(
    config: Exp25Config,
    *,
    timeout_seconds: float = BUNDLE_MANIFEST_WAIT_SECONDS,
    poll_seconds: float = 30.0,
) -> None:
    """Block until the store's manifest carries this run's bundle fingerprint.

    Only the shard owning replication 0 initializes the store, because the manifest write is not
    atomic and the owner claim can be read before it is written; every other shard waits here.

    Raises:
        TimeoutError: If the manifest does not appear in time.
    """

    manifest_path = _naturalness_bundle_leaf(config) / "run_manifest.yml"
    expected = naturalness_bundle_fingerprint(config)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if manifest_path.exists() and expected in manifest_path.read_text(encoding="utf-8"):
            return
        time.sleep(poll_seconds)
    raise TimeoutError(f"No initialized Exp25 bundle store at {manifest_path}")


def _naturalness_bundle_complete(config: Exp25Config, replication_dir: Path) -> bool:
    """Report whether one replication's persisted Naturalness bundle is complete."""

    try:
        _load_persisted_naturalness_references(config, replication_dir)
    except FileNotFoundError:
        return False
    return True


def run_naturalness_pass(config: Exp25Config) -> dict[str, Any]:
    """Fill the bundle store for the requested replications: the GPU pass.

    Each replication generates its OSRCT grid, fits Exp16 and persists the bundle, as a full run
    does before Exp18; under ``resume`` a replication with a complete bundle is skipped. Nothing
    under ``baseline_variants/`` is written; the ``comparators_only`` pass builds the result leaf.

    Args:
        config: Exp25 configuration with ``naturalness_only`` set.

    Returns:
        The bundle root, the replications fitted and those skipped.
    """

    selected = (
        list(config.replications)
        if config.replications is not None
        else list(range(config.n_replications))
    )
    if 0 not in selected:
        _wait_for_bundle_manifest(config)
    bundle_root = _write_or_validate_bundle_manifest(config)
    scratch_parent = Path(config.scratch_dir) if config.scratch_dir is not None else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    fitted, skipped = [], []
    for replication in sorted(selected):
        seed = config.seeds[replication]
        replication_dir = bundle_root / "replications" / f"replication={replication:03d}"
        if config.resume and _naturalness_bundle_complete(config, replication_dir):
            skipped.append(replication)
            if config.verbose:
                print(f"[Exp25 naturalness] skip complete replication={replication}", flush=True)
            continue
        if config.verbose:
            print(f"[Exp25 naturalness] start replication={replication} seed={seed}", flush=True)
        replication_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"exp25_nat_{replication:02d}_", dir=scratch_parent
        ) as temporary_name:
            _build_naturalness_bundle(
                config,
                seed=seed,
                bundle_replication_dir=replication_dir,
                scratch_root=Path(temporary_name),
            )
        fitted.append(replication)
        if config.verbose:
            print(f"[Exp25 naturalness] done replication={replication}", flush=True)
    return {"output_dir": bundle_root, "fitted": fitted, "skipped": skipped,
            "figure_paths": []}


# Leaves and tables a merge combines.
EXP25_MERGED_LEAVES = {
    "gamma_dgp_oracle": (EXP25_INTERVALS_FILENAME, EXP25_DIAGNOSTICS_FILENAME,
                         EXP25_GAMMA_FILENAME),
    EXP25_IDENTIFICATION_DIRNAME: (EXP25_INTERVALS_FILENAME, EXP25_DIAGNOSTICS_FILENAME),
}


def _backend_leaf_of(root: Path) -> Path:
    """Return the single ``<stage1>/<trigger>/<lp>`` bundle leaf under a root."""

    leaves = sorted(path.parent for path in root.glob("*/*/*/run_manifest.yml"))
    if len(leaves) != 1:
        raise ValueError(f"Expected one Exp25 bundle leaf under {root}, found {leaves}")
    return leaves[0]


def merge_result_roots(sources: Sequence[str | Path], destination: str | Path) -> Path:
    """Merge Exp25 runs over disjoint betas into one fresh analysis tree.

    Every source must share all scientific settings except ``betas``, which must not overlap, so
    seeds pair replication ``r`` across runs. The ``gamma_dgp_oracle`` and Manski/Balke-Pearl
    tables are concatenated under the same backend path, so
    :mod:`experiments.exp25_hit_ratio_analysis` and :mod:`experiments.site_average_rollup` read
    the merged tree unchanged. Sources are only read; the per-replication bundles stay in place.

    Args:
        sources: Result roots, each holding one backend leaf.
        destination: New result root; it must not already exist.

    Returns:
        The merged backend leaf.

    Raises:
        FileExistsError: If ``destination`` exists.
        ValueError: If settings differ, betas overlap, a leaf or table is
            missing, or two sources supply the same interval row.
    """

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to merge into an existing path: {destination}")
    if len(sources) < 2:
        raise ValueError("A merge needs at least two result roots")
    leaves = [_backend_leaf_of(Path(source).expanduser().resolve()) for source in sources]
    if len({leaf.relative_to(leaf.parents[2]) for leaf in leaves}) != 1:
        raise ValueError("Sources were run with different backends")
    scientific, identification, betas = [], [], []
    for leaf in leaves:
        # write_run_manifest nests experiment-specific fields under ``extra``.
        manifest = yaml.safe_load((leaf / "run_manifest.yml").read_text(encoding="utf-8"))
        config = dict(manifest["extra"]["scientific_config"])
        betas.append({float(value) for value in config.pop("betas")})
        scientific.append(config)
        bounds_manifest = leaf / "baseline_variants" / EXP25_IDENTIFICATION_DIRNAME / "run_manifest.yml"
        if not bounds_manifest.exists():
            raise ValueError(f"{leaf} has no Manski/Balke-Pearl leaf; run --identification-bounds-only")
        identification.append(
            {key: value for key, value in yaml.safe_load(bounds_manifest.read_text(encoding="utf-8"))[
                "extra"]["identification_config"].items() if key != "naturalness_bundle_fingerprint"})
    for index in range(1, len(leaves)):
        if scientific[index] != scientific[0]:
            differing = sorted(key for key in set(scientific[0]) | set(scientific[index])
                               if scientific[0].get(key) != scientific[index].get(key))
            raise ValueError(f"{leaves[index]} differs from {leaves[0]} in {differing}")
        if identification[index] != identification[0]:
            raise ValueError(f"{leaves[index]} used different Manski/Balke-Pearl settings")
    if sum(len(values) for values in betas) != len(set().union(*betas)):
        raise ValueError(f"Source betas overlap: {[sorted(values) for values in betas]}")

    merged_leaf = destination / leaves[0].relative_to(leaves[0].parents[2])
    keys = ["replication", "bias_covariate", "beta", "study", "target_site",
            *EXP25_STRATUM_COLUMNS, "method"]
    for name, tables in EXP25_MERGED_LEAVES.items():
        for table in tables:
            paths = [leaf / "baseline_variants" / name / table for leaf in leaves]
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise ValueError(f"Missing tables for the merge: {missing}")
            frame = pd.concat([pd.read_parquet(path) for path in paths],
                              ignore_index=True, sort=False)
            if table == EXP25_INTERVALS_FILENAME and frame.duplicated(keys).any():
                raise ValueError(f"Merged {name}/{table} has duplicate interval rows")
            _atomic_write_parquet(frame, merged_leaf / "baseline_variants" / name / table)
    provenance = {
        "experiment": EXP25_NAME,
        "merged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [str(leaf) for leaf in leaves],
        "betas": [sorted(values) for values in betas],
        "scientific_config": scientific[0],
        "identification_config": identification[0],
        "not_merged": "per-replication bundles",
    }
    (merged_leaf / "merge_manifest.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    return merged_leaf


def _merge_main(argv: Sequence[str]) -> int:
    """Command-line entry for ``merge SOURCE SOURCE ... --output-dir DEST``."""

    parser = argparse.ArgumentParser(
        prog="exp25_osrct_mc_baseline_comparison merge",
        description=merge_result_roots.__doc__.splitlines()[0],
    )
    parser.add_argument("sources", nargs="+", help="Exp25 result roots to merge")
    parser.add_argument("--output-dir", required=True, help="New merged result root")
    args = parser.parse_args(argv)
    merged = merge_result_roots(args.sources, args.output_dir)
    print(f"Exp25 merged leaf: {merged}")
    return 0


def run_experiment(config: Exp25Config) -> dict[str, Any]:
    """Run or resume the Exp25 workflow selected by the configuration's mode flags."""

    if config.naturalness_only:
        return run_naturalness_pass(config)
    if config.identification_bounds_only:
        return run_identification_bounds_replay(config)
    if config.dummy:
        return run_dummy(config)
    return _run_main_experiment(config)


def _run_main_experiment(config: Exp25Config) -> dict[str, Any]:
    """Run the main methods, resuming replications that are already committed."""
    _require_parquet_engine()
    output_dir = _output_leaf(config)
    bundle_output_dir = (
        _validate_naturalness_bundle_root(config)
        if config.baseline_only
        else _write_or_validate_bundle_manifest(config)
    )
    bundle_fingerprint = naturalness_bundle_fingerprint(config)
    primary_intervals = (
        _load_replay_source_intervals(config, bundle_output_dir)
        if config.baseline_only
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config)
    _write_or_validate_manifest(output_dir, config, fingerprint)
    intervals, diagnostics = _load_existing_outputs(
        output_dir,
        fingerprint=fingerprint,
        resume=config.resume,
    )
    completed = _completed_replications(diagnostics, config)
    gamma_filename = _gamma_calibration_filename(config)
    scratch_parent = (
        Path(config.scratch_dir) if config.scratch_dir is not None else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)

    for replication, seed in enumerate(config.seeds):
        if replication in completed and not config.baseline_only:
            if config.verbose:
                print(
                    f"[Exp25] skip committed replication={replication} "
                    f"seed={seed}",
                    flush=True,
                )
            continue
        if config.verbose:
            print(
                f"[Exp25] start replication={replication} seed={seed}",
                flush=True,
            )
        with tempfile.TemporaryDirectory(
            prefix=f"exp25_rep_{replication:02d}_",
            dir=scratch_parent,
        ) as temporary_name:
            replication_intervals, replication_diagnostics = (
                run_one_replication(
                    config,
                    replication=replication,
                    seed=seed,
                    scratch_root=Path(temporary_name),
                    bundle_replication_dir=(
                        bundle_output_dir
                        / "replications"
                        / f"replication={replication:03d}"
                    ),
                    result_replication_dir=(
                        output_dir
                        / "replications"
                        / f"replication={replication:03d}"
                    ),
                    fingerprint=fingerprint,
                    primary_intervals=primary_intervals,
                )
            )
            intervals, diagnostics = _commit_replication(
                output_dir,
                intervals,
                diagnostics,
                replication_intervals,
                replication_diagnostics,
            )
            _aggregate_gamma_calibrations(output_dir, gamma_filename)
            if not config.baseline_only:
                _write_replay_source_intervals(
                    bundle_output_dir,
                    intervals,
                    bundle_fingerprint=bundle_fingerprint,
                )
        if config.verbose:
            print(
                f"[Exp25] MC iteration {replication + 1}/"
                f"{config.n_replications} done "
                f"(replication={replication}, seed={seed})",
                flush=True,
            )

    if not config.baseline_only and not intervals.empty:
        # Republished after the loop so a fully resumed owner still refreshes
        # the store's replay source.
        _write_replay_source_intervals(
            bundle_output_dir,
            intervals,
            bundle_fingerprint=bundle_fingerprint,
        )
        # Cheap CPU bounds that read only the store just published.
        run_identification_bounds_replay(config)

    figure_paths = (
        make_all_plots(intervals, output_dir, config)
        if config.make_plots
        else []
    )
    return {
        "output_dir": output_dir,
        "intervals": intervals,
        "diagnostics": diagnostics,
        "gamma_calibration": _aggregate_gamma_calibrations(
            output_dir,
            gamma_filename,
        ),
        "figure_paths": figure_paths,
    }


def expand_blearner_gammas(values: Sequence[str]) -> list[float | str]:
    """Expand the ``grid`` token and parse every requested B-Learner Gamma.

    ``grid`` is an Exp25 command-line shorthand, not an Exp18 Gamma strategy: it expands to the
    fixed :data:`BLEARNER_GAMMA_GRID` sensitivity grid and never reaches an :class:`Exp25Config`,
    because each expanded Gamma is fit and stored on its own.

    Args:
        values: Raw ``--blearner-gamma`` tokens, in request order.

    Returns:
        Parsed Gammas in request order, with ``grid`` replaced in place by the fixed grid.

    Raises:
        argparse.ArgumentTypeError: If a token is neither ``grid``, an Exp18
            Gamma strategy, nor a finite number at least one.
    """

    expanded: list[float | str] = []
    for value in values:
        if value == GRID_GAMMA:
            expanded.extend(float(gamma) for gamma in BLEARNER_GAMMA_GRID)
        else:
            expanded.append(parse_gamma(value))
    return expanded


def parse_configs(argv: Sequence[str] | None = None) -> tuple[Exp25Config, ...]:
    """Parse Exp25 arguments into one configuration per requested Gamma.

    The configurations differ only in ``blearner_gamma``. Each Gamma writes its own result leaf,
    so they never contend for the same outputs and may be run in any order.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        One configuration per ``--blearner-gamma`` value, in the order given.

    Raises:
        SystemExit: If ``--blearner-gamma`` repeats a value.
    """

    defaults = Exp25Config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-replications",
        type=int,
        default=defaults.n_replications,
    )
    parser.add_argument("--base-seed", type=int, default=defaults.base_seed)
    parser.add_argument("--studies", nargs="+", default=list(defaults.studies))
    parser.add_argument(
        "--bias-covariates",
        nargs="+",
        default=list(defaults.bias_covariates),
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=list(defaults.betas),
    )
    parser.add_argument("--target-site", default=defaults.target_site)
    parser.add_argument("--rct-path", default=defaults.rct_path)
    parser.add_argument(
        "--require-observed-gender",
        action=argparse.BooleanOptionalAction,
        default=defaults.require_observed_gender,
        help=(
            "Exclude rows with missing original resp_sex before OSRCT "
            "generation, Naturalness, baselines, and RCT evaluation."
        ),
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--scratch-dir", default=defaults.scratch_dir)
    parser.add_argument(
        "--stage1-backend",
        choices=("tabicl_v2",),
        default=defaults.stage1_backend,
    )
    parser.add_argument("--tabicl-device", default=defaults.tabicl_device)
    parser.add_argument(
        "--target-trigger-probability-backend",
        choices=("tabicl_v2",),
        default=defaults.target_trigger_probability_backend,
    )
    parser.add_argument(
        "--lp-probability-backend",
        choices=EXP25_LP_PROBABILITY_BACKENDS,
        default=defaults.lp_probability_backend,
    )
    parser.add_argument(
        "--lp-posterior-samples",
        type=int,
        default=defaults.lp_posterior_samples,
    )
    parser.add_argument(
        "--lp-posterior-n-jobs",
        type=int,
        default=defaults.lp_posterior_n_jobs,
    )
    parser.add_argument(
        "--exp18-n-jobs",
        type=int,
        default=defaults.exp18_n_jobs,
    )
    parser.add_argument(
        "--exp18-worker-threads",
        type=int,
        default=defaults.exp18_worker_threads,
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Fit only B-Learner from persisted Parquet OSRCT/Naturalness "
            "bundles and reuse primary Naturalness, CB-IPW, and CB-DR rows."
        ),
    )
    parser.add_argument("--identification-bounds-only", action="store_true",
                        help="Compute only Manski and Balke-Pearl bounds from the saved bundle store")
    parser.add_argument("--identification-min-count", type=int,
                        default=defaults.identification_min_count,
                        help="Balke-Pearl instrument-arm support floor and Manski support mask")
    parser.add_argument("--identification-tau", type=float,
                        default=defaults.identification_tau,
                        help="Relevance threshold D_j(tau) for Balke-Pearl candidates")
    parser.add_argument("--naturalness-only", action="store_true",
                        help="Split-run GPU pass: fit and persist Naturalness bundles only")
    parser.add_argument("--comparators-only", action="store_true",
                        help="Split-run CPU pass: fit Exp18 and the bounds from persisted bundles")
    parser.add_argument("--replications", type=int, nargs="+", default=None,
                        help="Replication indices for one --naturalness-only shard")
    parser.add_argument(
        "--blearner-gamma",
        nargs="+",
        default=[str(defaults.blearner_gamma)],
        help=(
            "One or more B-Learner Gammas or selection strategies, or 'grid' "
            "for the fixed sensitivity grid. Each Gamma is run in turn and "
            "writes its own result leaf. Once any arm has populated the "
            "shared bundle store, later arms should add --baseline-only."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=defaults.resume,
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate simulated Parquets and plots without fitting models.",
    )
    parser.add_argument(
        "--cluster-manifest-path",
        default=defaults.cluster_manifest_path,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        gammas = expand_blearner_gammas(args.blearner_gamma)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if len({str(gamma) for gamma in gammas}) != len(gammas):
        parser.error("--blearner-gamma must not contain duplicates")
    base = Exp25Config(
        n_replications=args.n_replications,
        base_seed=args.base_seed,
        studies=tuple(args.studies),
        bias_covariates=tuple(args.bias_covariates),
        betas=tuple(args.betas),
        target_site=args.target_site,
        rct_path=args.rct_path,
        require_observed_gender=args.require_observed_gender,
        output_dir=args.output_dir,
        scratch_dir=args.scratch_dir,
        stage1_backend=args.stage1_backend,
        target_trigger_probability_backend=(
            args.target_trigger_probability_backend
        ),
        tabicl_device=args.tabicl_device,
        lp_probability_backend=args.lp_probability_backend,
        lp_posterior_samples=args.lp_posterior_samples,
        lp_posterior_n_jobs=args.lp_posterior_n_jobs,
        exp18_n_jobs=args.exp18_n_jobs,
        exp18_worker_threads=args.exp18_worker_threads,
        blearner_gamma=gammas[0],
        baseline_only=args.baseline_only,
        identification_bounds_only=args.identification_bounds_only,
        identification_min_count=args.identification_min_count,
        identification_tau=args.identification_tau,
        naturalness_only=args.naturalness_only,
        comparators_only=args.comparators_only,
        replications=tuple(args.replications) if args.replications is not None else None,
        resume=args.resume,
        make_plots=not args.no_plots,
        dummy=args.dummy,
        cluster_manifest_path=args.cluster_manifest_path,
        command_argv=tuple(sys.argv[1:] if argv is None else argv),
        verbose=not args.quiet,
    )
    return tuple(replace(base, blearner_gamma=gamma) for gamma in gammas)


def parse_args(argv: Sequence[str] | None = None) -> Exp25Config:
    """Parse Exp25 command-line arguments for a single B-Learner Gamma.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The configuration for the first requested ``--blearner-gamma`` value.
    """

    return parse_configs(argv)[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp25 from the command line, once per requested B-Learner Gamma.

    ``merge SOURCE SOURCE ... --output-dir DEST`` instead runs
    :func:`merge_result_roots`.
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["merge"]:
        return _merge_main(arguments[1:])
    for config in parse_configs(argv):
        result = run_experiment(config)
        print(f"Exp25 Gamma {config.blearner_gamma} outputs: {result['output_dir']}")
        print(f"Exp25 figures: {len(result['figure_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
