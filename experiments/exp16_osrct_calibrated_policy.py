"""Experiment 16: calibrated policies on OSRCT/RCT data.

Exp16 keeps the OSRCT/RCT construction and sweep:

* OSRCT rows are the observational ``F='idle'`` regime;
* original ManyLabs rows are the randomized ``F='on'`` regime; and
* runs vary by study, confounding variable, beta, and target site.

It uses an independent calibrated-policy procedure:

1. proper source sites learn naturalness tolerances;
2. policy-tuning sites choose top-k naturalness constraints;
3. endpoint-calibration sites learn a simultaneous conformal correction; and
4. the held-out target RCT CATEs evaluate the final intervals.

The LP and RCT validation use the same three binary covariates, so no
fine-to-useful-stratum marginalization is required.

This script is the command line. The configuration, the per-target pipeline
and the sweep live in ``experiments/osrct_target_pipeline.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from causal_grounding.naturalness_grounder import SUPPORTED_GP_FEATURE_MODES
from experiments.utils import (
    SUPPORTED_STAGE1_BACKENDS,
    SUPPORTED_LP_PROBABILITY_BACKENDS,
    SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS,
)

# The library half moved to experiments.osrct_target_pipeline. These
# re-exports keep ``from experiments.exp16_osrct_calibrated_policy import X``
# working for existing callers, tests and notebooks.
from experiments.osrct_target_pipeline import (  # noqa: F401
    DEFAULT_EXP16_BETAS,
    DEFAULT_EXP16_PATTERNS,
    DEFAULT_EXP16_STUDIES,
    DEFAULT_GP_POSTERIOR_SAMPLES,
    DEFAULT_POLICY_TOP_K_VALUES,
    MAX_BINARY_CATE_ENDPOINT_RESIDUAL,
    SUPPORTED_CANDIDATE_SELECTION_MODES,
    SUPPORTED_GP_POLICY_UNCERTAINTIES,
    Exp16Config,
    _calibrate_endpoint_q_if_enabled,
    _candidate_state_map,
    _fit_grounder_for_sites,
    _fit_grounders_and_select_policy,
    _leaf_output_dir,
    _output_group_key,
    _run_output_dir,
    _summarize_target_trigger_grid,
    _uses_candidate_states,
    build_empirical_rct_cate_labels,
    build_policy_candidates,
    calibrate_endpoint_q,
    complete_lp_strata,
    estimate_target_trigger_quantities,
    evaluate_policy_site,
    finalize_policy_choices,
    fit_reusable_tabicl_target_quantities,
    grounder_gp_diagnostics,
    run_experiment,
    run_one_target,
    score_gp_posterior_summary,
    score_rct_intervals,
    select_policy_choices,
    select_top_k_choices,
    solve_choices_lp,
    solve_choices_lp_posterior_samples,
    split_site_regimes,
    split_source_roles,
    summarize_target,
    tune_policy_top_k,
    weighted_interval_width,
)
from experiments.osrct_target_pipeline import (  # noqa: F401
    build_policy_candidates as _policy_candidates,
    estimate_target_trigger_quantities as _estimate_target_trigger_quantities,
    finalize_policy_choices as _finalize_policy_choices,
    fit_reusable_tabicl_target_quantities as _fit_reusable_tabicl_target_quantities,
    grounder_gp_diagnostics as _grounder_gp_diagnostics,
    split_site_regimes as _split_site_regimes,
)


def _default_lp_probability_backend(
    *,
    stage1_backend: str,
    fit_gp: bool,
) -> str:
    """Return the default target-LP backend implied by Stage-1 settings.

    Args:
        stage1_backend: Configured Stage-1 backend name.
        fit_gp: Whether Stage-3 GP posterior propagation is requested.

    Returns:
        ``hierarchical_dirichlet_draws`` for hierarchical Stage-1 GP analyses,
        otherwise ``stage1``.
    """

    backend = str(stage1_backend).lower()
    if fit_gp and backend == "hierarchical_dirichlet":
        return "hierarchical_dirichlet_draws"
    return "stage1"


def parse_args(argv: Sequence[str] | None = None) -> Exp16Config:
    """Parse Exp16 command-line arguments."""

    defaults = Exp16Config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--studies", nargs="+", default=list(defaults.studies or ()))
    parser.add_argument("--patterns", nargs="+", default=list(defaults.patterns or ()))
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(defaults.betas or ()),
    )
    parser.add_argument("--target-sites", nargs="+", default=list(defaults.target_sites))
    parser.add_argument("--max-target-sites", type=int, default=defaults.max_target_sites)
    parser.add_argument("--max-source-sites", type=int, default=defaults.max_source_sites)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--rct-path", default=defaults.rct_path)
    parser.add_argument(
        "--require-observed-gender",
        action=argparse.BooleanOptionalAction,
        default=defaults.require_observed_gender,
        help=(
            "Exclude rows with missing original resp_sex from both OSRCT "
            "and RCT analysis cohorts."
        ),
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(defaults.epsilons))
    parser.add_argument("--taus", type=float, nargs="+", default=list(defaults.taus))
    parser.add_argument("--min-cell-count", type=int, default=defaults.min_cell_count)
    parser.add_argument(
        "--max-z-minus-samples",
        type=int,
        default=defaults.max_z_minus_samples,
    )
    parser.add_argument(
        "--age-binary-threshold",
        type=float,
        default=defaults.age_binary_threshold,
    )
    parser.add_argument(
        "--polideo-binary-threshold",
        type=float,
        default=defaults.polideo_binary_threshold,
    )
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--min-overlap", type=float, default=defaults.min_overlap)
    parser.add_argument(
        "--stage1-backend",
        choices=SUPPORTED_STAGE1_BACKENDS,
        default=defaults.stage1_backend,
    )
    parser.add_argument(
        "--tabicl-device",
        default=defaults.tabicl_device or "auto",
        help=(
            "TabICLv2 execution device: auto, cpu, cuda, cuda:<index>, or mps. "
            "Auto uses CUDA when available."
        ),
    )
    parser.add_argument(
        "--lp-probability-backend",
        choices=SUPPORTED_LP_PROBABILITY_BACKENDS,
        default=None,
        help=(
            "Target-LP probability backend. Defaults to hierarchical "
            "Dirichlet posterior draws for hierarchical Stage-1 GP analyses "
            "and stage1 otherwise."
        ),
    )
    parser.add_argument(
        "--target-trigger-probability-backend",
        choices=SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS,
        default=defaults.target_trigger_probability_backend,
        help=(
            "Backend for target I/D trigger probabilities. Exp16 defaults to "
            "its legacy smoothed-empirical trigger screen; use stage1 to fit "
            "the configured Stage-1 model family on target idle rows, or lp "
            "to reuse the target-LP backend's posterior mean."
        ),
    )
    parser.add_argument("--fit-gp", action="store_true")
    parser.add_argument(
        "--lp-posterior-samples",
        type=int,
        default=None,
        help=(
            f"Defaults to {DEFAULT_GP_POSTERIOR_SAMPLES} with --fit-gp and "
            "0 otherwise."
        ),
    )
    parser.add_argument(
        "--lp-posterior-n-jobs",
        type=int,
        default=defaults.lp_posterior_n_jobs,
        help=(
            "CPU threads for independent posterior target-LP draws. "
            "TabICLv2 Stage 1 remains serial."
        ),
    )
    parser.add_argument(
        "--skip-lp-intervention-diagnostics",
        action="store_true",
        help=(
            "Solve only lower/upper CATE objectives and skip the four "
            "mu0/mu1 diagnostic objectives in every direct target LP."
        ),
    )
    parser.add_argument(
        "--target-policy-uncertainty",
        choices=SUPPORTED_GP_POLICY_UNCERTAINTIES,
        default=defaults.target_policy_uncertainty,
    )
    parser.add_argument(
        "--target-policy-confidence-level",
        type=float,
        default=defaults.target_policy_confidence_level,
        help=(
            "Joint confidence level used by "
            "posterior_simultaneous_upper target policy."
        ),
    )
    parser.add_argument(
        "--gp-feature-mode",
        choices=SUPPORTED_GP_FEATURE_MODES,
        default=defaults.gp_feature_mode,
        help=(
            "Stage-3 GP input features: legacy epsilon/tau thresholds or "
            "realized i_deviation/d_strength trigger statistics."
        ),
    )
    parser.add_argument(
        "--policy-top-k-values",
        type=int,
        nargs="+",
        default=list(defaults.policy_top_k_values),
    )
    parser.add_argument(
        "--candidate-selection-mode",
        choices=SUPPORTED_CANDIDATE_SELECTION_MODES,
        default=defaults.candidate_selection_mode,
        help=(
            "Naturalness candidate set passed to the LP: legacy one-best-row "
            "per top-k covariate, all triggered rows, or all rows for the "
            "top-k covariates."
        ),
    )
    parser.add_argument(
        "--policy-tuning-fraction",
        type=float,
        default=defaults.policy_tuning_fraction,
    )
    parser.add_argument(
        "--min-policy-tuning-sites",
        type=int,
        default=defaults.min_policy_tuning_sites,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=defaults.calibration_fraction,
    )
    parser.add_argument(
        "--min-calibration-sites",
        type=int,
        default=defaults.min_calibration_sites,
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=defaults.conformal_alpha,
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help=(
            "Skip endpoint-conformal calibration, retain its source sites as "
            "held out, and use zero endpoint widening (q=0)."
        ),
    )
    parser.add_argument(
        "--rct-min-arm-count",
        type=int,
        default=defaults.rct_min_arm_count,
    )
    parser.add_argument("--disjoint-validation-ids", action="store_true")
    parser.add_argument("--id-col", default=defaults.id_col)
    parser.add_argument(
        "--cluster-manifest-path",
        default=defaults.cluster_manifest_path,
        help=(
            "Kubernetes YAML template to embed in run_manifest.yml when "
            "materializing run provenance."
        ),
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    posterior_samples = (
        int(args.lp_posterior_samples)
        if args.lp_posterior_samples is not None
        else (DEFAULT_GP_POSTERIOR_SAMPLES if args.fit_gp else 0)
    )
    lp_probability_backend = (
        args.lp_probability_backend
        or _default_lp_probability_backend(
            stage1_backend=args.stage1_backend,
            fit_gp=bool(args.fit_gp),
        )
    )
    return Exp16Config(
        studies=tuple(args.studies),
        patterns=tuple(args.patterns),
        betas=tuple(args.betas),
        target_sites=tuple(args.target_sites),
        max_target_sites=args.max_target_sites,
        max_source_sites=args.max_source_sites,
        data_root=args.data_root,
        rct_path=args.rct_path,
        require_observed_gender=args.require_observed_gender,
        output_dir=args.output_dir,
        epsilons=tuple(args.epsilons),
        taus=tuple(args.taus),
        min_cell_count=args.min_cell_count,
        max_z_minus_samples=args.max_z_minus_samples,
        age_binary_threshold=args.age_binary_threshold,
        polideo_binary_threshold=args.polideo_binary_threshold,
        alpha=args.alpha,
        min_overlap=args.min_overlap,
        stage1_backend=args.stage1_backend,
        validation_stage1_backend=args.stage1_backend,
        tabicl_device=(
            None if args.tabicl_device == "auto" else args.tabicl_device
        ),
        lp_probability_backend=lp_probability_backend,
        target_trigger_probability_backend=(
            args.target_trigger_probability_backend
        ),
        fit_gp=args.fit_gp,
        gp_feature_mode=args.gp_feature_mode,
        lp_posterior_samples=posterior_samples,
        lp_posterior_n_jobs=args.lp_posterior_n_jobs,
        skip_lp_intervention_diagnostics=(
            args.skip_lp_intervention_diagnostics
        ),
        target_policy_uncertainty=args.target_policy_uncertainty,
        target_policy_confidence_level=args.target_policy_confidence_level,
        policy_top_k_values=tuple(args.policy_top_k_values),
        candidate_selection_mode=args.candidate_selection_mode,
        policy_tuning_fraction=args.policy_tuning_fraction,
        min_policy_tuning_sites=args.min_policy_tuning_sites,
        calibration_fraction=args.calibration_fraction,
        min_calibration_sites=args.min_calibration_sites,
        conformal_alpha=args.conformal_alpha,
        skip_calibration=args.skip_calibration,
        rct_min_arm_count=args.rct_min_arm_count,
        disjoint_validation_ids=args.disjoint_validation_ids,
        id_col=args.id_col,
        cluster_manifest_path=args.cluster_manifest_path,
        command_argv=tuple(sys.argv[1:] if argv is None else argv),
        seed=args.seed,
        fail_fast=args.fail_fast,
        verbose=not args.quiet,
        make_plots=not args.no_plots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp16 from the command line."""

    run_experiment(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
