"""Tests for Exp16 OSRCT calibrated-policy helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.osrct_target_pipeline as target_pipeline
import experiments.naturalness_plotting as plotting
from experiments.osrct_target_pipeline import (
    DEFAULT_GP_POSTERIOR_SAMPLES,
    Exp16Config,
    _leaf_output_dir,
    _output_group_key,
    _run_output_dir,
    complete_lp_strata,
    score_rct_intervals,
    solve_choices_lp_posterior_samples,
)
from experiments.exp16_osrct_calibrated_policy import parse_args
from experiments.naturalness_plotting import make_exp16_plots
from experiments.utils import DEFAULT_LP_POSTERIOR_N_JOBS, write_run_manifest


def test_exp16_does_not_import_removed_experiment_modules() -> None:
    """Exp16 should survive after removing retired experiment scripts."""

    source = Path("experiments/exp16_osrct_calibrated_policy.py").read_text()

    forbidden = f"experiments.exp{12}_osrct_cd_calibration"
    assert forbidden not in source


def test_exp16_defaults_merge_osrct_data_with_exp15_policy() -> None:
    """Defaults should retain OSRCT sweeps and calibrated top-k selection."""

    config = Exp16Config()

    assert config.studies == ("allowedforbidden", "gainloss", "reciprocity")
    assert config.patterns == ("age", "gender", "polideo")
    assert config.betas == (0.1, 0.25, 0.5, 1.0)
    assert config.target_sites == ("pi",)
    assert config.policy_top_k_values == (1, 2, 3)
    assert config.candidate_selection_mode == "best_per_covariate"
    assert config.conformal_alpha == 0.10
    assert config.rct_min_arm_count == 14
    assert config.covariates == config.naturalness_covariates
    assert config.target_trigger_probability_backend == "empirical"
    assert config.lp_posterior_n_jobs == DEFAULT_LP_POSTERIOR_N_JOBS
    assert not config.skip_lp_intervention_diagnostics
    assert not config.skip_calibration
    assert config.age_binary_threshold == 21.0


def test_fit_gp_cli_enables_posterior_lp_samples() -> None:
    """The GP scenario should inherit Exp15's joint target-draw default."""

    config = parse_args(["--fit-gp", "--no-plots"])

    assert config.fit_gp
    assert config.lp_posterior_samples == DEFAULT_GP_POSTERIOR_SAMPLES


def test_exp16_cli_accepts_simultaneous_upper_tolerance_policy() -> None:
    """Exp16 forwards model-based joint upper tolerance selection."""

    config = parse_args([
        "--fit-gp",
        "--target-policy-uncertainty",
        "posterior_simultaneous_upper",
        "--target-policy-confidence-level",
        "0.90",
        "--no-plots",
    ])

    assert (
        config.target_policy_uncertainty
        == "posterior_simultaneous_upper"
    )
    assert config.target_policy_confidence_level == 0.90
    with pytest.raises(ValueError, match="requires fit_gp=True"):
        Exp16Config(
            fit_gp=False,
            target_policy_uncertainty="posterior_simultaneous_upper",
        )


def test_cli_can_pass_only_target_and_hierarchical_backend() -> None:
    """Cluster configs may omit studies, patterns, betas, and LP backend."""

    config = parse_args(
        [
            "--target-sites",
            "pi",
            "--stage1-backend",
            "hierarchical_dirichlet",
            "--cluster-manifest-path",
            "cluster_scripts/job_exp16_osrct_calibrated_policy.yml",
            "--fit-gp",
            "--no-plots",
        ]
    )

    assert config.studies == ("allowedforbidden", "gainloss", "reciprocity")
    assert config.patterns == ("age", "gender", "polideo")
    assert config.betas == (0.1, 0.25, 0.5, 1.0)
    assert config.lp_probability_backend == "hierarchical_dirichlet_draws"
    assert config.cluster_manifest_path == (
        "cluster_scripts/job_exp16_osrct_calibrated_policy.yml"
    )
    assert "--cluster-manifest-path" in config.command_argv


def test_exp16_cli_accepts_candidate_selection_mode() -> None:
    """Exp16 CLI should pass the candidate-selection mode into the config."""

    config = parse_args([
        "--candidate-selection-mode",
        "all_triggered",
        "--no-plots",
    ])

    assert config.candidate_selection_mode == "all_triggered"
    assert "--candidate-selection-mode" in config.command_argv


def test_exp16_cli_can_skip_endpoint_calibration() -> None:
    """The calibration ablation should be explicit and disabled by default."""

    config = parse_args(["--skip-calibration", "--no-plots"])

    assert config.skip_calibration
    assert "--skip-calibration" in config.command_argv


def test_exp16_cli_can_skip_lp_intervention_diagnostics() -> None:
    """The CATE-only LP optimization should be explicit and opt-in."""

    config = parse_args([
        "--skip-lp-intervention-diagnostics",
        "--no-plots",
    ])

    assert config.skip_lp_intervention_diagnostics
    assert "--skip-lp-intervention-diagnostics" in config.command_argv


def test_skip_calibration_returns_zero_correction_without_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping calibration should preserve held-out roles and avoid fitting."""

    monkeypatch.setattr(
        target_pipeline,
        "calibrate_endpoint_q",
        lambda **_kwargs: pytest.fail("calibration must be skipped"),
    )
    q, rank, sentinel, scores, bounds = (
        target_pipeline._calibrate_endpoint_q_if_enabled(
            calibration_sites=("held_out",),
            source_training_data={"held_out": pd.DataFrame()},
            grounder=object(),
            selected_top_k=1,
            config=Exp16Config(skip_calibration=True),
        )
    )

    assert q == 0.0
    assert rank == 0
    assert not sentinel
    assert scores.empty
    assert bounds.empty


def test_tabicl_target_probabilities_are_reused_only_for_identical_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache gate must reject backend or predictor-covariate differences."""

    marker = object()
    calls: list[str] = []

    def fake_estimate(
        _target_idle: pd.DataFrame,
        _config: Exp16Config,
        *,
        backend: str,
    ) -> object:
        """Return a marker while recording the single expected TabICL fit."""

        calls.append(backend)
        return marker

    monkeypatch.setattr(
        target_pipeline,
        "estimate_target_trigger_quantities",
        fake_estimate,
    )
    idle = pd.DataFrame()
    reusable = target_pipeline.fit_reusable_tabicl_target_quantities(
        idle,
        Exp16Config(
            stage1_backend="tabicl_v2",
            target_trigger_probability_backend="stage1",
            lp_probability_backend="stage1",
        ),
    )

    assert reusable is marker
    assert calls == ["tabicl_v2"]

    not_reusable = target_pipeline.fit_reusable_tabicl_target_quantities(
        idle,
        Exp16Config(
            stage1_backend="tabicl_v2",
            target_trigger_probability_backend="stage1",
            lp_probability_backend="stage1",
            lp_probability_covariates=("resp_age_bin",),
        ),
    )
    different_backend = target_pipeline.fit_reusable_tabicl_target_quantities(
        idle,
        Exp16Config(
            stage1_backend="tabicl_v2",
            target_trigger_probability_backend="stage1",
            lp_probability_backend="hierarchical_dirichlet_draws",
        ),
    )

    assert not_reusable is None
    assert different_backend is None
    assert calls == ["tabicl_v2"]


def test_policy_site_reuses_one_safe_tabicl_probability_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger screening and the direct LP should receive the same table."""

    marker = object()
    received: dict[str, object] = {}
    idle = pd.DataFrame({"F": ["idle"], "iv": [0], "dv_binary": [0]})
    choices = pd.DataFrame({"covariate": ["z"]})
    bounds = pd.DataFrame(
        {"score_lower": [-1.0], "score_upper": [1.0], "n_target_rows": [1]}
    )

    monkeypatch.setattr(
        target_pipeline,
        "fit_reusable_tabicl_target_quantities",
        lambda *_args: marker,
    )

    def fake_candidates(
        _grounder: object,
        _idle: pd.DataFrame,
        _config: Exp16Config,
        *,
        trigger_quantities: object,
    ) -> pd.DataFrame:
        """Record the target-trigger table."""

        received["trigger"] = trigger_quantities
        return choices

    def fake_solve(
        _idle: pd.DataFrame,
        _choices: pd.DataFrame,
        _config: Exp16Config,
        *,
        observed_quantities: object,
    ) -> pd.DataFrame:
        """Record the target-LP table."""

        received["lp"] = observed_quantities
        return bounds

    monkeypatch.setattr(target_pipeline, "build_policy_candidates", fake_candidates)
    monkeypatch.setattr(
        target_pipeline,
        "select_policy_choices",
        lambda candidates, _top_k, mode: candidates.copy(),
    )
    monkeypatch.setattr(target_pipeline, "solve_choices_lp", fake_solve)
    monkeypatch.setattr(
        target_pipeline,
        "complete_lp_strata",
        lambda current, _idle, _config: current,
    )
    monkeypatch.setattr(
        target_pipeline,
        "score_rct_intervals",
        lambda current, _rct, _config: (
            current,
            {"site_score": 0.0, "n_scored_rows": 0},
        ),
    )

    target_pipeline.evaluate_policy_site(
        site="held_out",
        site_data=idle,
        grounder=object(),
        top_k=1,
        config=Exp16Config(),
    )

    assert received == {"trigger": marker, "lp": marker}


def test_top_k_tuning_reuses_tabicl_table_across_k_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tuning site should share one safe TabICL table across k values."""

    sites = ("site_a", "site_b")
    source_data = {
        site: pd.DataFrame(
            {
                "site": [site],
                "F": ["idle"],
                "iv": [0],
                "dv_binary": [0],
            }
        )
        for site in sites
    }
    markers = {site: object() for site in sites}
    fit_calls: list[str] = []
    evaluation_calls: list[tuple[str, int, object]] = []

    def fake_fit(idle: pd.DataFrame, _config: Exp16Config) -> object:
        """Return a stable site-specific table and record cache construction."""

        site = str(idle["site"].iloc[0])
        fit_calls.append(site)
        return markers[site]

    def fake_evaluate(
        *,
        site: str,
        top_k: int,
        reusable_quantities: object,
        **_kwargs: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        """Return minimal tuning diagnostics while recording cache identity."""

        evaluation_calls.append((site, top_k, reusable_quantities))
        bounds = pd.DataFrame(
            {
                "site": [site],
                "policy_top_k": [top_k],
                "rct_cate": [0.0],
                "lower": [-0.5],
                "upper": [0.5],
                "status": ["optimal"],
                "score_lower": [-0.5],
                "score_upper": [0.5],
                "n_target_rows": [1],
            }
        )
        choices = pd.DataFrame({"site": [site], "policy_top_k": [top_k]})
        summary = {
            "site": site,
            "policy_top_k": top_k,
            "site_score": 0.0,
        }
        return bounds, choices, summary

    monkeypatch.setattr(
        target_pipeline,
        "fit_reusable_tabicl_target_quantities",
        fake_fit,
    )
    monkeypatch.setattr(target_pipeline, "evaluate_policy_site", fake_evaluate)

    target_pipeline.tune_policy_top_k(
        tuning_sites=sites,
        source_training_data=source_data,
        grounder=object(),
        config=Exp16Config(
            policy_top_k_values=(1, 2),
            conformal_alpha=0.5,
        ),
    )

    assert fit_calls == ["site_a", "site_b"]
    assert evaluation_calls == [
        ("site_a", 1, markers["site_a"]),
        ("site_b", 1, markers["site_b"]),
        ("site_a", 2, markers["site_a"]),
        ("site_b", 2, markers["site_b"]),
    ]


def test_posterior_lp_reuses_supplied_deterministic_tabicl_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Posterior tolerance draws must not refit fixed TabICL probabilities."""

    marker = object()
    received: list[tuple[object, bool]] = []

    class FittedSurface:
        """Minimal fitted Stage-3 surface marker for posterior validation."""

        is_fitted = True

    class Grounder:
        """Return two deterministic tolerance samples for the focused test."""

        c_surface_ = FittedSurface()
        d_surface_ = FittedSurface()

        def sample_tolerances(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
            """Return one selected condition in each of two posterior draws."""

            return pd.DataFrame(
                {
                    "sample_id": [0, 1],
                    "covariate": ["z", "z"],
                    "two_c": [0.1, 0.2],
                    "two_d": [0.1, 0.2],
                }
            )

    monkeypatch.setattr(
        target_pipeline,
        "estimate_target_observed_binary_quantities",
        lambda *_args, **_kwargs: pytest.fail("TabICL must not be refitted"),
    )

    def fake_solve(
        quantities: object,
        _choices: pd.DataFrame,
        **_kwargs: object,
    ) -> pd.DataFrame:
        """Record the fixed probability table used by each tolerance draw."""

        received.append(
            (
                quantities,
                bool(_kwargs["include_intervention_diagnostics"]),
            )
        )
        return pd.DataFrame({"status": ["optimal"]})

    monkeypatch.setattr(
        target_pipeline,
        "solve_target_bounds_from_observed_binary_quantities",
        fake_solve,
    )
    monkeypatch.setattr(
        target_pipeline,
        "complete_lp_strata",
        lambda current, _idle, _config: current,
    )
    monkeypatch.setattr(
        target_pipeline,
        "summarize_lp_posterior_bounds",
        lambda _samples, _covariates: pd.DataFrame({"summary": [1]}),
    )
    config = Exp16Config(
        covariates=("z",),
        naturalness_covariates=("z",),
        lp_probability_covariates=("z",),
        stage1_backend="tabicl_v2",
        lp_probability_backend="stage1",
        fit_gp=True,
        lp_posterior_samples=2,
        lp_posterior_n_jobs=1,
        skip_lp_intervention_diagnostics=True,
    )

    _, samples, summary = solve_choices_lp_posterior_samples(
        pd.DataFrame({"z": [0], "iv": [0], "dv_binary": [0]}),
        pd.DataFrame({"covariate": ["z"]}),
        Grounder(),
        config,
        random_seed=42,
        observed_quantities=marker,
    )

    assert received == [(marker, False), (marker, False)]
    assert samples["lp_posterior_sample_id"].tolist() == [0, 1]
    assert summary["summary"].tolist() == [1]


def test_all_triggered_fits_combined_training_cohort_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed all-triggered policy should bypass tuning and the proper-only fit."""

    sites = tuple(f"site_{index:02d}" for index in range(35))
    source_data = {
        site: pd.DataFrame({"site": [site]})
        for site in sites
    }
    roles = {
        "proper_training": sites[:15],
        "policy_tuning": sites[15:26],
        "endpoint_calibration": sites[26:],
    }
    fit_calls: list[tuple[str, ...]] = []
    final_grounder = object()

    def fake_fit_grounder(
        _source_data: object,
        training_sites: tuple[str, ...],
        _config: Exp16Config,
    ) -> object:
        """Record the sites used by the only expected grounder fit."""

        fit_calls.append(tuple(training_sites))
        return final_grounder

    monkeypatch.setattr(target_pipeline, "_fit_grounder_for_sites", fake_fit_grounder)
    monkeypatch.setattr(
        target_pipeline,
        "tune_policy_top_k",
        lambda **_kwargs: pytest.fail("all_triggered must not tune top_k"),
    )

    (
        proper_grounder,
        fitted_final_grounder,
        selected_top_k,
        policy_summary,
        tuning_bounds,
        tuning_choices,
    ) = target_pipeline._fit_grounders_and_select_policy(
        source_training_data=source_data,
        roles=roles,
        config=Exp16Config(candidate_selection_mode="all_triggered"),
    )

    assert proper_grounder is None
    assert fitted_final_grounder is final_grounder
    assert fit_calls == [sites[:26]]
    assert selected_top_k == 1
    assert policy_summary["selection_reason"].tolist() == [
        "fixed_all_triggered_policy"
    ]
    assert not policy_summary["policy_tuning_performed"].iloc[0]
    assert not policy_summary["policy_top_k_applicable"].iloc[0]
    assert tuning_bounds.empty
    assert tuning_choices.empty


def test_top_k_mode_retains_proper_fit_tuning_and_final_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-k-dependent policies should retain the two-stage fitting design."""

    source_data = {
        site: pd.DataFrame({"site": [site]})
        for site in ("proper", "tuning", "calibration")
    }
    roles = {
        "proper_training": ("proper",),
        "policy_tuning": ("tuning",),
        "endpoint_calibration": ("calibration",),
    }
    fit_calls: list[tuple[str, ...]] = []
    grounders = [object(), object()]

    def fake_fit_grounder(
        _source_data: object,
        training_sites: tuple[str, ...],
        _config: Exp16Config,
    ) -> object:
        """Return one marker object for each expected fitting stage."""

        fit_calls.append(tuple(training_sites))
        return grounders[len(fit_calls) - 1]

    def fake_tune_policy_top_k(**kwargs: object) -> tuple[
        int,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Return a minimal selected top-k result after checking held-out use."""

        assert kwargs["grounder"] is grounders[0]
        assert kwargs["tuning_sites"] == ("tuning",)
        return (
            2,
            pd.DataFrame({"policy_top_k": [2], "selected": [True]}),
            pd.DataFrame({"bound": [1]}),
            pd.DataFrame({"choice": [1]}),
        )

    monkeypatch.setattr(target_pipeline, "_fit_grounder_for_sites", fake_fit_grounder)
    monkeypatch.setattr(target_pipeline, "tune_policy_top_k", fake_tune_policy_top_k)

    (
        proper_grounder,
        final_grounder,
        selected_top_k,
        policy_summary,
        tuning_bounds,
        tuning_choices,
    ) = target_pipeline._fit_grounders_and_select_policy(
        source_training_data=source_data,
        roles=roles,
        config=Exp16Config(candidate_selection_mode="best_per_covariate"),
    )

    assert proper_grounder is grounders[0]
    assert final_grounder is grounders[1]
    assert fit_calls == [("proper",), ("proper", "tuning")]
    assert selected_top_k == 2
    assert policy_summary["policy_tuning_performed"].iloc[0]
    assert policy_summary["policy_top_k_applicable"].iloc[0]
    assert not tuning_bounds.empty
    assert not tuning_choices.empty


def test_exp16_cli_accepts_gp_feature_mode() -> None:
    """Exp16 CLI should pass the Stage-3 GP feature mode into the config."""

    config = parse_args([
        "--fit-gp",
        "--gp-feature-mode",
        "observed_stats",
        "--no-plots",
    ])

    assert config.gp_feature_mode == "observed_stats"
    assert "--gp-feature-mode" in config.command_argv


def test_write_run_manifest_embeds_cluster_yaml(tmp_path: Path) -> None:
    """Run manifest materialization should copy the source YAML text."""

    source = tmp_path / "job.yml"
    source.write_text("apiVersion: batch/v1\nkind: Job\n", encoding="utf-8")

    output = write_run_manifest(
        tmp_path / "out",
        experiment_name="exp16_osrct_calibrated_policy",
        script_path="experiments/exp16_osrct_calibrated_policy.py",
        config={"stage1_backend": "xgboost"},
        argv=["--stage1-backend", "xgboost"],
        cluster_manifest_path=source,
    )

    text = output.read_text(encoding="utf-8")
    assert "experiment: \"exp16_osrct_calibrated_policy\"" in text
    assert "apiVersion: batch/v1" in text
    assert "- \"--stage1-backend\"" in text


def test_posterior_samples_require_gp() -> None:
    """Posterior LP propagation must not run without Stage-3 GP fitting."""

    with pytest.raises(ValueError, match="requires fit_gp=True"):
        Exp16Config(lp_posterior_samples=2)


def test_exp16_accepts_hierarchical_dirichlet_draw_lp_backend() -> None:
    """Exp16 allows posterior target-probability draws as an LP backend."""

    config = Exp16Config(lp_probability_backend="hierarchical_dirichlet_draws")

    assert config.lp_probability_backend == "hierarchical_dirichlet_draws"


def test_exp16_cli_rejects_removed_hierarchical_saturated_logistic_backend() -> None:
    """Exp16 exposes only maintained Stage-1 and LP backends."""

    with pytest.raises(SystemExit):
        parse_args([
            "--stage1-backend",
            "hierarchical_saturated_logistic",
            "--no-plots",
        ])


def test_exp16_cli_accepts_tabicl_v2_stage1_backend() -> None:
    """Exp16 exposes TabICLv2 for Stage 1 and stage1-resolved target fitting."""
    config = parse_args([
        "--stage1-backend",
        "tabicl_v2",
        "--tabicl-device",
        "cuda",
        "--no-plots",
    ])

    assert config.stage1_backend == "tabicl_v2"
    assert config.lp_probability_backend == "stage1"
    assert config.tabicl_device == "cuda"


def test_exp16_cli_separates_tabicl_triggers_from_dirichlet_lp() -> None:
    """Exp16 can preserve Stage-1 target triggers while drawing LP nuisances."""

    config = parse_args([
        "--stage1-backend",
        "tabicl_v2",
        "--target-trigger-probability-backend",
        "stage1",
        "--lp-probability-backend",
        "hierarchical_dirichlet_draws",
        "--no-plots",
    ])

    assert config.target_trigger_probability_backend == "stage1"
    assert config.lp_probability_backend == "hierarchical_dirichlet_draws"


def test_exp16_nondefault_trigger_backend_gets_isolated_output_leaf() -> None:
    """Trigger-backend comparisons must not overwrite legacy Exp16 outputs."""

    config = Exp16Config(
        candidate_selection_mode="all_triggered",
        stage1_backend="tabicl_v2",
        target_trigger_probability_backend="stage1",
    )

    assert _run_output_dir(Path("/tmp/exp16"), config) == (
        Path("/tmp/exp16")
        / "tabicl_v2"
        / "tabicl_v2"
        / "tabicl_v2"
        / "all_triggered"
    )


def test_exp16_output_root_starts_with_all_three_backend_configs() -> None:
    """Stage-1, trigger, and LP settings should lead every Exp16 run path."""

    config = Exp16Config(
        candidate_selection_mode="all_triggered",
        stage1_backend="tabicl_v2",
        target_trigger_probability_backend="stage1",
        lp_probability_backend="hierarchical_dirichlet_draws",
        gp_feature_mode="observed_stats",
    )

    assert _run_output_dir(Path("/tmp/exp16"), config) == (
        Path("/tmp/exp16")
        / "tabicl_v2"
        / "tabicl_v2"
        / "hierarchical_dirichlet_draws"
        / "all_triggered"
        / "observed_stats"
    )


def test_exp16_skip_calibration_gets_isolated_output_leaf() -> None:
    """Uncalibrated ablations must not overwrite calibrated outputs."""

    config = Exp16Config(
        candidate_selection_mode="all_triggered",
        stage1_backend="tabicl_v2",
        target_trigger_probability_backend="stage1",
        skip_calibration=True,
    )

    assert _run_output_dir(Path("/tmp/exp16"), config) == (
        Path("/tmp/exp16")
        / "tabicl_v2"
        / "tabicl_v2"
        / "tabicl_v2"
        / "all_triggered"
        / "no_calibration"
    )


def test_exp16_cate_only_lp_gets_isolated_output_leaf() -> None:
    """CATE-only LP runs must not overwrite full diagnostic outputs."""

    config = Exp16Config(
        candidate_selection_mode="all_triggered",
        stage1_backend="tabicl_v2",
        target_trigger_probability_backend="stage1",
        skip_lp_intervention_diagnostics=True,
    )

    assert _run_output_dir(Path("/tmp/exp16"), config) == (
        Path("/tmp/exp16")
        / "tabicl_v2"
        / "tabicl_v2"
        / "tabicl_v2"
        / "all_triggered"
        / "cate_only_lp"
    )


def test_complete_lp_strata_keeps_direct_three_covariate_queries() -> None:
    """Exp16 must not marginalize over an additional fine-stratum layer."""

    config = Exp16Config(
        covariates=("z",),
        naturalness_covariates=("z",),
        lp_probability_covariates=("z",),
    )
    idle = pd.DataFrame(
        {
            "z": [0, 0, 0, 1],
            "F": ["idle"] * 4,
        }
    )
    bounds = pd.DataFrame(
        {
            "z": [0, 1],
            "lower": [-0.2, np.nan],
            "upper": [0.3, np.nan],
            "status": ["optimal", "infeasible"],
        }
    )

    completed = complete_lp_strata(bounds, idle, config)

    assert completed["z"].tolist() == [0, 1]
    assert completed["n_target_rows"].tolist() == [3, 1]
    assert completed["score_lower"].tolist() == [-0.2, -1.0]
    assert completed["score_upper"].tolist() == [0.3, 1.0]


def test_complete_lp_strata_restores_omitted_low_support_query() -> None:
    """Target strata below the nuisance threshold must remain explicit."""

    config = Exp16Config(
        covariates=("z",),
        naturalness_covariates=("z",),
        lp_probability_covariates=("z",),
    )
    idle = pd.DataFrame(
        {
            "z": [0, 0, 0, 1],
            "F": ["idle"] * 4,
        }
    )
    bounds = pd.DataFrame(
        {
            "z": [0],
            "lower": [-0.2],
            "upper": [0.3],
            "status": ["optimal"],
        }
    )

    completed = complete_lp_strata(bounds, idle, config).sort_values("z")

    assert completed["z"].tolist() == [0, 1]
    assert completed["n_target_rows"].tolist() == [3, 1]
    assert completed["status"].tolist() == [
        "optimal",
        "lp_stratum_not_returned",
    ]
    assert completed["score_lower"].tolist() == [-0.2, -1.0]
    assert completed["score_upper"].tolist() == [0.3, 1.0]
    assert completed["used_full_range_fallback"].tolist() == [False, True]


def test_posterior_summary_retains_strata_when_no_choice_triggers() -> None:
    """No-choice targets must remain explicit zero-optimal posterior rows."""

    config = Exp16Config(
        covariates=("z",),
        naturalness_covariates=("z",),
        lp_probability_covariates=("z",),
        fit_gp=True,
        lp_posterior_samples=3,
        lp_probability_backend="empirical",
    )
    idle = pd.DataFrame(
        {
            "z": [0, 0, 1],
            "iv": [0, 1, 0],
            "dv_binary": [0, 1, 0],
            "F": ["idle"] * 3,
        }
    )

    tolerances, samples, summary = solve_choices_lp_posterior_samples(
        idle,
        pd.DataFrame(),
        object(),
        config,
        random_seed=42,
    )

    assert tolerances.empty
    assert len(samples) == 6
    assert samples["lp_posterior_sample_id"].unique().tolist() == [0, 1, 2]
    assert samples["status"].eq(
        "no_selected_naturalness_choices"
    ).all()
    assert samples["used_full_range_fallback"].all()
    assert summary["z"].tolist() == [0, 1]
    assert summary["lp_posterior_n_samples"].tolist() == [3, 3]
    assert summary["lp_posterior_n_optimal"].tolist() == [0, 0]
    assert summary["lp_posterior_prob_optimal"].tolist() == [0.0, 0.0]
    assert summary[["lower_mean", "upper_mean"]].isna().all().all()


def test_rct_scoring_requires_fourteen_rows_in_each_arm() -> None:
    """Only RCT strata meeting the Exp15 arm-count rule should be scored."""

    config = Exp16Config(
        covariates=("z",),
        naturalness_covariates=("z",),
        lp_probability_covariates=("z",),
        rct_min_arm_count=14,
    )
    bounds = pd.DataFrame(
        {
            "z": [0, 1],
            "lower": [-0.2, -0.2],
            "upper": [0.2, 0.2],
            "score_lower": [-0.2, -0.2],
            "score_upper": [0.2, 0.2],
            "status": ["optimal", "optimal"],
            "n_target_rows": [20, 20],
        }
    )
    supported = pd.DataFrame(
        {
            "z": [0] * 28,
            "iv": [0] * 14 + [1] * 14,
            "dv_binary": [0] * 14 + [1] * 14,
        }
    )
    unsupported = pd.DataFrame(
        {
            "z": [1] * 27,
            "iv": [0] * 13 + [1] * 14,
            "dv_binary": [0] * 27,
        }
    )

    scored, summary = score_rct_intervals(
        bounds,
        pd.concat([supported, unsupported], ignore_index=True),
        config,
    )

    assert scored["rct_cate_available"].tolist() == [True, False]
    assert summary["n_scored_rows"] == 1
    assert np.isclose(summary["site_score"], 0.8)


def test_make_plots_writes_osrct_cate_forest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The merged experiment should retain pattern/beta CATE forest labels."""

    monkeypatch.setattr(
        plotting.matplotlib.figure.Figure,
        "suptitle",
        lambda *_args, **_kwargs: pytest.fail(
            "Exp16 plots should not add a figure-level title"
        ),
    )

    bounds = pd.DataFrame(
        {
            "study": ["gainloss", "allowedforbidden", "reciprocity"],
            "pattern": ["age", "age", "age"],
            "beta": [0.5, 0.5, 0.5],
            "target_site": ["pi", "pi", "pi"],
            "z_value_str": ["(0, 1, 0)", "(0, 1, 0)", "(0, 1, 0)"],
            "resp_age_bin": [0, 0, 0],
            "resp_gender_bin": [1, 1, 1],
            "resp_polideo_bin": [0, 0, 0],
            "conformal_lower": [-0.3, -0.4, -0.2],
            "conformal_upper": [0.4, 0.5, 0.3],
            "rct_cate": [0.1, 0.0, 0.2],
            "rct_cate_ci90_lower": [-0.1, -0.2, 0.0],
            "rct_cate_ci90_upper": [0.3, 0.2, 0.4],
            "base_interval_available": [True, False, True],
            "used_full_range_fallback": [False, True, False],
        }
    )
    posterior = bounds.assign(
        posterior_envelope_lower=[-0.5, -0.6, -0.4],
        posterior_envelope_upper=[0.5, 0.6, 0.4],
        lower_mean=[-0.2, -0.3, -0.1],
        upper_mean=[0.3, 0.4, 0.2],
        lp_posterior_prob_optimal=[1.0, 0.9, 0.8],
    )

    make_exp16_plots(
        tmp_path,
        {
            "target_lp_bounds": bounds,
            "target_gp_posterior_summary": posterior,
            "target_policy_choices": pd.DataFrame(
                {
                    "study": ["gainloss", "allowedforbidden", "reciprocity"],
                    "pattern": ["age", "age", "age"],
                    "beta": [0.5, 0.5, 0.5],
                    "target_site": ["pi", "pi", "pi"],
                    "status": ["selected", "selected", "selected"],
                    "covariate": [
                        "resp_age_bin",
                        "resp_gender_bin",
                        "resp_polideo_bin",
                    ],
                    "epsilon": [0.1, 0.2, 0.5],
                    "c": [0.05, 0.10, 0.20],
                    "d": [0.04, 0.08, 0.16],
                }
            ),
        },
        ("resp_age_bin", "resp_gender_bin", "resp_polideo_bin"),
    )

    assert (
        tmp_path
        / "figures"
        / "exp16_prediction_conformalized_lp_bounds_vs_rct_cate.png"
    ).exists()
    assert (
        tmp_path / "figures" / "exp16_target_tolerance_vs_epsilon.png"
    ).exists()
    assert (
        tmp_path
        / "figures"
        / "exp16_prediction_posterior_mean_lp_bounds_vs_rct_cate.png"
    ).exists()
    horizontal_mean_path = (
        tmp_path
        / "figures"
        / "exp16_prediction_posterior_mean_lp_bounds_vs_rct_cate_horizontal.png"
    )
    assert horizontal_mean_path.exists()
    horizontal_mean_image = plotting.plt.imread(horizontal_mean_path)
    assert horizontal_mean_image.shape[1] > horizontal_mean_image.shape[0]
    probability_colorbar_path = (
        tmp_path
        / "figures"
        / "exp16_prediction_posterior_mean_with_envelope_lp_bounds_vs_rct_cate.png"
    )
    assert probability_colorbar_path.exists()
    assert "LP posterior optimal probability" in (
        probability_colorbar_path.with_suffix(".svg").read_text()
    )
    fixed_color_path = (
        tmp_path
        / "figures"
        / (
            "exp16_prediction_posterior_mean_with_envelope_"
            "no_optimality_rate_lp_bounds_vs_rct_cate.png"
        )
    )
    assert fixed_color_path.exists()
    assert "LP posterior optimal probability" not in (
        fixed_color_path.with_suffix(".svg").read_text()
    )
    horizontal_fixed_color_path = (
        tmp_path
        / "figures"
        / (
            "exp16_prediction_posterior_mean_with_envelope_"
            "no_optimality_rate_lp_bounds_vs_rct_cate_horizontal.png"
        )
    )
    assert horizontal_fixed_color_path.exists()
    assert "LP posterior optimal probability" not in (
        horizontal_fixed_color_path.with_suffix(".svg").read_text()
    )
    assert (
        tmp_path
        / "figures"
        / "exp16_prediction_posterior_envelope_lp_bounds_vs_rct_cate.png"
    ).exists()


def test_exp16_envelope_plot_without_optimality_rate_has_exp15_legend(
    tmp_path: Path,
) -> None:
    """The fixed-color envelope plot should not require an optimality rate."""

    posterior = pd.DataFrame(
        {
            "study": ["gainloss"],
            "resp_age_bin": [0],
            "resp_gender_bin": [1],
            "resp_polideo_bin": [0],
            "posterior_envelope_lower": [-0.5],
            "posterior_envelope_upper": [0.6],
            "lower_mean": [-0.2],
            "upper_mean": [0.4],
            "rct_cate": [0.1],
            "rct_cate_ci90_lower": [-0.1],
            "rct_cate_ci90_upper": [0.3],
        }
    )

    make_exp16_plots(
        tmp_path,
        {"target_gp_posterior_summary": posterior},
        ("resp_age_bin", "resp_gender_bin", "resp_polideo_bin"),
    )

    figures_dir = tmp_path / "figures"
    fixed_color_path = figures_dir / (
        "exp16_prediction_posterior_mean_with_envelope_"
        "no_optimality_rate_lp_bounds_vs_rct_cate.png"
    )
    svg_text = fixed_color_path.with_suffix(".svg").read_text()
    expected_legend_order = [
        "Posterior mean LP interval (optimal draws only)",
        "Posterior endpoint envelope (optimal draws only)",
        "RCT CATE 90% Wald CI",
        "Held-out RCT CATE",
    ]

    assert fixed_color_path.exists()
    assert "LP posterior optimal probability" not in svg_text
    assert all(label in svg_text for label in expected_legend_order)
    assert [svg_text.index(label) for label in expected_legend_order] == sorted(
        svg_text.index(label) for label in expected_legend_order
    )
    assert not (
        figures_dir
        / "exp16_prediction_posterior_mean_with_envelope_lp_bounds_vs_rct_cate.png"
    ).exists()


def test_exp16_leaf_output_path_uses_mode_backend_target_pattern_beta() -> None:
    """Leaf directories should isolate candidate mode and Stage-1 backend."""

    config = Exp16Config(
        candidate_selection_mode="all_triggered",
        stage1_backend="tabicl_v2",
        pattern="age",
        beta=0.5,
    )
    key = _output_group_key(target_site="pi", config=config)
    run_root = _run_output_dir(Path("/tmp/exp16"), config)

    assert _leaf_output_dir(run_root, key) == (
        Path("/tmp/exp16")
        / "tabicl_v2"
        / "empirical"
        / "tabicl_v2"
        / "all_triggered"
        / "pi"
        / "age"
        / "0.5"
    )


def test_exp16_validates_extra_candidate_settings() -> None:
    """Extra candidates must be new, unique and use full-state LP features."""

    config = Exp16Config(naturalness_extra_candidates=("noise_01",))
    assert config.naturalness_extra_candidates == ("noise_01",)
    with pytest.raises(ValueError, match="duplicates"):
        Exp16Config(naturalness_extra_candidates=("noise_01", "noise_01"))
    with pytest.raises(ValueError, match="must not repeat LP covariates"):
        Exp16Config(naturalness_extra_candidates=("resp_age_bin",))
    with pytest.raises(ValueError, match="lp_probability_covariates"):
        Exp16Config(
            naturalness_extra_candidates=("noise_01",),
            lp_probability_covariates=("resp_age_bin",),
        )
    with pytest.raises(ValueError, match="unknown values"):
        Exp16Config(naturalness_candidate_covariates=("noise_01",))


def test_candidate_state_map_follows_extra_candidates() -> None:
    """Real candidates keep the reporting state; extras get their own."""

    config = Exp16Config(naturalness_extra_candidates=("noise_01", "noise_02"))
    states = target_pipeline._candidate_state_map(config)
    base = tuple(config.covariates)

    assert states["resp_age_bin"] == base
    assert states["noise_02"] == (*base, "noise_02")
    assert target_pipeline._uses_candidate_states(config)
    assert not target_pipeline._uses_candidate_states(Exp16Config())


def test_solve_choices_lp_routes_candidate_states_to_state_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra candidates are bounded per state on the reporting strata."""

    received: dict[str, object] = {}

    def fake_state_solver(quantities, _choices, **kwargs):
        """Record the per-state inputs of the combined solve."""
        received.update(quantities=quantities, **kwargs)
        return pd.DataFrame({"status": ["optimal"]})

    monkeypatch.setattr(
        target_pipeline,
        "solve_candidate_state_bounds_from_observed_binary_quantities",
        fake_state_solver,
    )
    monkeypatch.setattr(
        target_pipeline,
        "solve_target_bounds_from_observed_binary_quantities",
        lambda *_args, **_kwargs: pytest.fail("the single-state solver must not run"),
    )
    marker = {("state",): object()}
    config = Exp16Config(naturalness_extra_candidates=("noise_01",))

    bounds = target_pipeline.solve_choices_lp(
        pd.DataFrame(),
        pd.DataFrame({"covariate": ["noise_01"]}),
        config,
        observed_quantities=marker,
    )

    assert received["quantities"] is marker
    assert list(received["reporting_covariates"]) == list(config.covariates)
    assert received["candidate_states"]["noise_01"] == (*config.covariates, "noise_01")
    assert bounds["probability_backend"].tolist() == ["tabicl_v2"]


def test_posterior_draws_fall_back_when_gp_surfaces_are_unfitted() -> None:
    """The opt-in fallback keeps posterior summaries total instead of raising."""

    class UnfittedSurface:
        """Stage-3 surface that never reached enough observations."""

        is_fitted = False
        status = "insufficient_observations"

    class Grounder:
        """Grounder double whose GP surfaces are both unfitted."""

        c_surface_ = UnfittedSurface()
        d_surface_ = UnfittedSurface()

    covariates = Exp16Config().covariates
    idle = pd.DataFrame(
        {**{covariate: [0, 1] for covariate in covariates}, "iv": [0, 1], "dv_binary": [0, 1]}
    )
    config = Exp16Config(
        fit_gp=True,
        lp_posterior_samples=2,
        lp_posterior_n_jobs=1,
        fallback_when_gp_unfitted=True,
    )
    choices = pd.DataFrame({"covariate": ["resp_age_bin"]})

    tolerances, samples, summary = solve_choices_lp_posterior_samples(
        idle, choices, Grounder(), config, random_seed=1
    )

    assert tolerances.empty
    assert set(samples["status"]) == {"gp_surface_not_fitted"}
    assert summary["lp_posterior_prob_optimal"].eq(0.0).all()
    with pytest.raises(RuntimeError, match="requires fitted c/d surfaces"):
        solve_choices_lp_posterior_samples(
            idle,
            choices,
            Grounder(),
            Exp16Config(fit_gp=True, lp_posterior_samples=2, lp_posterior_n_jobs=1),
            random_seed=1,
        )


def test_target_trigger_grid_collapses_to_one_row_per_candidate() -> None:
    """Trigger statistics are constant over the grid, so one row suffices."""

    grid = pd.DataFrame({
        "covariate": ["a", "a", "noise_01", "noise_01"],
        "covariate_index": [0, 0, 3, 3],
        "i_deviation": [0.1, 0.1, 0.3, 0.3],
        "d_strength": [0.2, 0.2, 0.01, 0.01],
        "n_z_minus_evaluated": [4, 4, 8, 8],
        "n_target_rows": [100] * 4,
        "target_probability_source": ["model"] * 4,
        "target_triggered": [True, False, False, False],
    })

    summary = target_pipeline._summarize_target_trigger_grid(grid)

    assert summary["covariate"].tolist() == ["a", "noise_01"]
    assert summary["n_triggered_grid_points"].tolist() == [1, 0]
    assert summary["target_triggered_any"].tolist() == [True, False]
    assert summary["n_grid_points"].tolist() == [2, 2]
    assert target_pipeline._summarize_target_trigger_grid(None).empty
