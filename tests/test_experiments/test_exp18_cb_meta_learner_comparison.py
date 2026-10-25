"""Tests for the parallel Exp18 CB meta-learner comparison helpers."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# exp18 imports experiments.benchmarks, which needs the Exp16-Exp25 cluster ML
# stack from requirements.txt. Skip instead of aborting collection.
pytest.importorskip("torch")
pytest.importorskip("catenets")
pytest.importorskip("sklearn_quantile")

import torch

import experiments.benchmarks as benchmarks
import experiments.exp18_cb_meta_learner_comparison as exp18

from dataclasses import replace

from scipy.special import expit

from experiments.benchmarks import BLEARNER_GAMMA_GRID, WIDTH_MATCHED_GAMMA
from experiments.exp18_cb_meta_learner_comparison import (
    Exp18Config,
    FitResult,
    POOLED_SOURCE_ENVIRONMENT,
    _assemble_study_bounds,
    _build_study_tasks,
    _finalize_binary_cate_intersection,
    _leaf_output_dir,
    _ordered_pairs,
    cb_pseudo_outcome,
    parse_args,
    wb_pseudo_outcome,
)


def _site_data(values: list[tuple[int, int, int]]) -> pd.DataFrame:
    """Build a small idle-data site frame for bound-assembly tests."""

    return pd.DataFrame(
        values,
        columns=["resp_age_bin", "resp_gender_bin", "resp_polideo_bin"],
    ).assign(iv=[0, 1], dv_binary=[0, 1], F="idle")


def test_exp18_defaults_match_the_existing_exp16_comparison_setting() -> None:
    """Defaults should reproduce the notebook's Pi gender/beta-one scenario."""

    config = Exp18Config()

    assert config.target_site == "pi"
    assert config.data_source == "auto"
    assert config.pattern == "gender"
    assert config.beta == 1.0
    assert config.n_source_sites == 25
    assert config.source_site_policy == "random_sample"
    assert config.source_regime == "idle"
    assert config.pair_scope == "target_anchored"
    assert config.worker_threads == 1
    assert config.meta_learners == ("cb_pi", "cb_ra")
    assert config.naturalness_gp_feature_mode == "thresholds"
    assert config.naturalness_pre_all_triggered_folder == ""
    assert config.gamma == "width_matched"
    assert exp18._blearner_gammas(config) == (
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    )
    default_output = _leaf_output_dir(config)
    assert default_output.name == "source25_pooled_idle_target_anchored"
    assert default_output.parent.name == "cbpi_cbra"
    assert default_output.parents[4].name == "xgboost"

    aligned_output = _leaf_output_dir(
        Exp18Config(source_site_policy="naturalness_final_training"),
        data_source="exp16",
    )
    assert aligned_output.name == "gender_1.0"
    assert aligned_output.parent.name == "width_matched"
    binary_output = _leaf_output_dir(
        Exp18Config(
            source_site_policy="naturalness_final_training",
            gamma="width_matched_binary",
        ),
        data_source="exp16",
    )
    assert binary_output.name == "gender_1.0"
    assert binary_output.parent.name == "width_matched_binary"
    selected_output = _leaf_output_dir(
        Exp18Config(meta_learners=("cb_ipw", "cb_pi")),
        data_source="exp16",
    )
    assert selected_output.name == "gender_1.0"


def test_blearner_only_entry_point_skips_meta_learner_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-Learner-only fitting must never dispatch WB or CB tasks."""

    config = Exp18Config(
        studies=("gainloss",),
        data_source="exp16",
        target_site="pi",
        make_plots=False,
        verbose=False,
    )
    prepared = exp18.PreparedStudyData(
        idle=pd.DataFrame({"site": ["pi"], "value": [1]}),
        rct=pd.DataFrame({"site": ["pi"], "value": [1]}),
    )
    naturalness_reference = pd.DataFrame({"study": ["gainloss"]})
    captured_environments: dict[str, object] = {}
    monkeypatch.setattr(
        exp18,
        "_inspect_naturalness_reference",
        lambda _: "exp16",
    )
    monkeypatch.setattr(
        exp18,
        "_load_fitting_data",
        lambda *_: ({"gainloss": prepared}, pd.DataFrame()),
    )
    monkeypatch.setattr(
        exp18,
        "_load_naturalness_reference",
        lambda _: naturalness_reference,
    )
    monkeypatch.setattr(
        exp18,
        "_run_fit_tasks",
        lambda *args, **kwargs: pytest.fail("Meta-learner tasks were dispatched"),
    )

    def fake_fit_and_select(
        _config: Exp18Config,
        environments: object,
        _reference: pd.DataFrame,
        _prepared: object,
    ) -> dict[str, pd.DataFrame]:
        """Capture target-only environments and return empty artifacts."""

        captured_environments.update(environments)
        empty = pd.DataFrame()
        return {
            "blearner_bounds": empty,
            "selected_blearner_bounds": empty,
            "blearner_gamma_selection": empty,
            "blearner_oracle_gamma": empty,
        }

    monkeypatch.setattr(
        exp18,
        "_fit_and_select_blearner_bounds",
        fake_fit_and_select,
    )

    result = exp18.fit_blearner_only(config)

    assert set(captured_environments) == {"gainloss"}
    assert set(captured_environments["gainloss"]) == {"pi"}
    assert result["naturalness_reference"] is naturalness_reference


def test_backend_selects_reference_artifact_and_output_root() -> None:
    """Backend selection should keep Exp18 artifacts separate by backend."""

    config = Exp18Config(
        naturalness_backend="tabicl_v2",
        output_dir="/tmp/exp18",
    )

    assert exp18._naturalness_reference_path(config) == (
        Path("/mnt/cephfs/predictive-shift/results")
        / "exp16_osrct_calibrated_policy"
        / "tabicl_v2"
        / "empirical"
        / "tabicl_v2"
        / "all_triggered"
        / "pi"
        / "gender"
        / "1.0"
        / "target_gp_posterior_summary.csv"
    )
    assert _leaf_output_dir(config, data_source="exp16").is_relative_to(
        Path("/tmp/exp18/exp16/tabicl_v2/empirical/tabicl_v2")
    )


def test_backend_signature_output_layout() -> None:
    """Explicit backend labels should produce the compact requested layout."""

    config = parse_args(
        [
            "--data-source",
            "exp16",
            "--gamma",
            "exp_beta",
            "--naturalness-backend",
            "tabicl_v2",
            "--naturalness-trigger-backend",
            "stage1",
            "--naturalness-lp-backend",
            "stage1",
            "--output-dir",
            "/tmp/exp18",
        ]
    )

    assert _leaf_output_dir(config, data_source="exp16") == Path(
        "/tmp/exp18/exp16/tabicl_v2/tabicl_v2/tabicl_v2/exp_beta/gender_1.0"
    )


def test_observed_stats_cli_and_output_layout() -> None:
    """Observed-stat references should be selectable without output collisions."""

    config = parse_args(
        [
            "--data-source",
            "exp16",
            "--naturalness-gp-feature-mode",
            "observed_stats",
            "--output-dir",
            "/tmp/exp18",
        ]
    )

    assert config.naturalness_gp_feature_mode == "observed_stats"
    assert _leaf_output_dir(config, data_source="exp16").is_relative_to(
        Path("/tmp/exp18/exp16/xgboost/empirical/xgboost/observed_stats")
    )


def test_naturalness_gp_feature_mode_must_be_supported() -> None:
    """Unknown GP feature modes should fail before reference discovery."""

    with pytest.raises(ValueError, match="naturalness_gp_feature_mode"):
        Exp18Config(naturalness_gp_feature_mode="unknown")


def test_backend_signature_output_labels_must_be_path_components() -> None:
    """Backend labels should fail before a run when they contain separators."""

    with pytest.raises(ValueError, match="naturalness_trigger_backend"):
        Exp18Config(naturalness_trigger_backend="target/stage1")


def test_backend_configs_find_one_nested_exp16_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backend-only selection should find one nested Exp16 ablation leaf."""

    reference = (
        tmp_path
        / "exp16_osrct_calibrated_policy"
        / "tabicl_v2"
        / "tabicl_v2"
        / "hierarchical_dirichlet_draws"
        / "all_triggered"
        / "no_calibration"
        / "cate_only_lp"
        / "pi"
        / "polideo"
        / "1.0"
        / "target_gp_posterior_summary.csv"
    )
    reference.parent.mkdir(parents=True)
    reference.touch()
    monkeypatch.setattr(exp18, "_naturalness_results_roots", lambda: (tmp_path,))
    config = Exp18Config(
        pattern="polideo",
        naturalness_backend="tabicl_v2",
        naturalness_trigger_backend="stage1",
        naturalness_lp_backend="hierarchical_dirichlet_draws",
    )

    assert exp18._naturalness_reference_path(config) == reference


def test_backend_configs_find_literal_stage1_alias_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reference lookup should support Exp16 paths preserving stage1 aliases."""

    reference = (
        tmp_path
        / "exp16_osrct_calibrated_policy"
        / "tabicl_v2"
        / "stage1"
        / "hierarchical_dirichlet_draws"
        / "all_triggered"
        / "no_calibration"
        / "cate_only_lp"
        / "pi"
        / "gender"
        / "1.0"
        / "target_gp_posterior_summary.csv"
    )
    reference.parent.mkdir(parents=True)
    reference.touch()
    monkeypatch.setattr(exp18, "_naturalness_results_roots", lambda: (tmp_path,))

    config = Exp18Config(
        naturalness_backend="tabicl_v2",
        naturalness_trigger_backend="stage1",
        naturalness_lp_backend="hierarchical_dirichlet_draws",
    )

    assert exp18._naturalness_reference_path(config) == reference


def test_exp16_observed_stats_reference_uses_saved_mode_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exp16 observed-stat selection should resolve its canonical saved leaf."""

    reference = (
        tmp_path
        / "exp16_osrct_calibrated_policy"
        / "tabicl_v2"
        / "tabicl_v2"
        / "tabicl_v2"
        / "all_triggered"
        / "observed_stats"
        / "pi"
        / "gender"
        / "1.0"
        / "target_gp_posterior_summary.csv"
    )
    reference.parent.mkdir(parents=True)
    reference.touch()
    monkeypatch.setattr(exp18, "_naturalness_results_roots", lambda: (tmp_path,))

    config = Exp18Config(
        data_source="exp16",
        naturalness_backend="tabicl_v2",
        naturalness_trigger_backend="stage1",
        naturalness_lp_backend="stage1",
        naturalness_gp_feature_mode="observed_stats",
    )

    assert exp18._naturalness_reference_path(config) == reference


def test_threshold_reference_discovery_ignores_observed_stats_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default threshold lookup must not silently consume observed-stat output."""

    observed_reference = (
        tmp_path
        / "exp16_osrct_calibrated_policy"
        / "tabicl_v2"
        / "tabicl_v2"
        / "tabicl_v2"
        / "all_triggered"
        / "observed_stats"
        / "pi"
        / "gender"
        / "1.0"
        / "target_gp_posterior_summary.csv"
    )
    observed_reference.parent.mkdir(parents=True)
    observed_reference.touch()
    monkeypatch.setattr(exp18, "_naturalness_results_roots", lambda: (tmp_path,))

    config = Exp18Config(
        data_source="exp16",
        naturalness_backend="tabicl_v2",
        naturalness_trigger_backend="stage1",
        naturalness_lp_backend="stage1",
    )

    resolved = exp18._naturalness_reference_path(config)
    assert resolved != observed_reference
    assert "observed_stats" not in resolved.parts


def test_backend_configs_reject_multiple_nested_exp16_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backend-only selection should reject ambiguous Exp16 ablation leaves."""

    mode_root = (
        tmp_path
        / "exp16_osrct_calibrated_policy"
        / "tabicl_v2"
        / "tabicl_v2"
        / "tabicl_v2"
        / "all_triggered"
    )
    for variant in ("no_calibration", "cate_only_lp"):
        reference = (
            mode_root
            / variant
            / "pi"
            / "gender"
            / "1.0"
            / "target_gp_posterior_summary.csv"
        )
        reference.parent.mkdir(parents=True)
        reference.touch()
    monkeypatch.setattr(exp18, "_naturalness_results_roots", lambda: (tmp_path,))

    with pytest.raises(ValueError, match="Multiple Exp16 naturalness references"):
        exp18._naturalness_reference_path(
            Exp18Config(
                naturalness_backend="tabicl_v2",
                naturalness_trigger_backend="stage1",
                naturalness_lp_backend="stage1",
            )
        )


def test_blearner_commit_validation_allows_the_exact_non_owned_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    """Commit validation should safely support a root-owned cluster checkout."""

    repository = tmp_path / "BLearner"
    expected_module = repository / "models" / "blearner" / "BLearner.py"
    expected_module.parent.mkdir(parents=True)
    expected_module.touch()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        """Record the Git validation command and return the pinned commit."""

        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=f"{exp18.BLEARNER_COMMIT}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    benchmarks._VALIDATED_BLEARNER_REPOSITORIES.discard(repository)

    exp18._validate_blearner_repository(repository)

    assert calls == [
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ]
    ]
    assert repository in benchmarks._VALIDATED_BLEARNER_REPOSITORIES


def test_wb_pseudo_outcome_matches_binary_manski_endpoints() -> None:
    """WB endpoint targets should be Eq. 11 specialized to support [0, 1]."""

    treatment = np.array([0, 0, 1, 1])
    outcome = np.array([0, 1, 0, 1])

    assert wb_pseudo_outcome(treatment, outcome, "upper").tolist() == [1, 0, 0, 1]
    assert wb_pseudo_outcome(treatment, outcome, "lower").tolist() == [0, -1, -1, 0]


def test_cb_pseudo_outcome_uses_the_ordered_environment_direction() -> None:
    """Leading and trailing environments must supply different CB components."""

    treatment = np.array([0, 1, 0, 1])
    outcome = np.array([0, 0, 1, 1])
    leading = np.array([True, True, False, False])

    # CATENets subtracts the trailing response itself.  Its transformed
    # outcome must therefore contain the positive response-function term.
    assert cb_pseudo_outcome(treatment, outcome, leading, "upper").tolist() == [1, 0, 1, 0]
    assert cb_pseudo_outcome(treatment, outcome, leading, "lower").tolist() == [0, 0, 1, 1]


def test_paper_pw_learner_keeps_catenets_propensity_first_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CB-IPW compatibility patch should return usable Torch placeholders."""

    def fake_propensity(
        _self: object,
        _x: torch.Tensor,
        _w: torch.Tensor,
        _fit_mask: torch.Tensor,
        _pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return fixed propensity predictions without fitting a network."""

        return torch.tensor([0.25, 0.75])

    monkeypatch.setattr(
        benchmarks._PaperPWLearner,
        "_impute_propensity",
        fake_propensity,
    )
    learner = benchmarks._PaperPWLearner(
        n_unit_in=1,
        binary_y=False,
        n_iter=1,
        batch_size=2,
        val_split_prop=0.0,
        batch_norm=False,
    )
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([0.0, 1.0])
    environment = torch.tensor([0.0, 1.0])
    mask = torch.tensor([True, True])

    response_0, response_1, propensity = learner._first_step(
        x,
        y,
        environment,
        mask,
        mask,
    )

    assert torch.isnan(response_0).all()
    assert torch.isnan(response_1).all()
    assert torch.allclose(propensity.cpu(), torch.tensor([0.25, 0.75]))


def test_target_anchored_pairs_use_both_two_environment_directions() -> None:
    """The default scope should create exactly the two pooled-source directions."""

    config = Exp18Config(n_source_sites=2, expected_non_target_sites=2)
    pairs = _ordered_pairs(
        config,
        ("pi", POOLED_SOURCE_ENVIRONMENT),
        (POOLED_SOURCE_ENVIRONMENT,),
    )

    assert pairs == [
        ("pi", POOLED_SOURCE_ENVIRONMENT),
        (POOLED_SOURCE_ENVIRONMENT, "pi"),
    ]
    all_ordered = Exp18Config(
        n_source_sites=2,
        expected_non_target_sites=2,
        pair_scope="all_ordered",
    )
    assert _ordered_pairs(
        all_ordered,
        ("pi", POOLED_SOURCE_ENVIRONMENT),
        (POOLED_SOURCE_ENVIRONMENT,),
    ) == pairs


def test_source_rows_are_concatenated_into_one_raw_pooled_environment(
    monkeypatch,
) -> None:
    """Selected source rows should become one environment with recorded weights."""

    osrct = pd.concat(
        [
            _site_data([(0, 0, 0), (1, 1, 1)]).assign(site="pi"),
            _site_data([(0, 0, 0), (1, 1, 1)]).assign(site="a"),
            _site_data([(0, 0, 0), (0, 0, 0)]).assign(site="b"),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(
        exp18,
        "load_prepared_data",
        lambda _: (osrct, pd.DataFrame(), {}),
    )
    config = Exp18Config(
        studies=("gainloss",),
        n_source_sites=2,
        expected_non_target_sites=2,
    )

    environments, support, composition = exp18._load_site_environments(
        config,
        "gainloss",
        ("a", "b"),
    )

    assert set(environments) == {"pi", POOLED_SOURCE_ENVIRONMENT}
    assert len(environments[POOLED_SOURCE_ENVIRONMENT]) == 4
    assert set(support["environment"]) == {"pi", POOLED_SOURCE_ENVIRONMENT}
    assert composition["source_site"].tolist() == ["a", "b"]
    assert composition["n_source_site_rows"].tolist() == [2, 2]
    assert composition["source_regime"].tolist() == ["idle", "idle"]
    assert composition["source_pool_row_weight"].tolist() == [0.5, 0.5]


def test_source_on_uses_rct_rows_while_pi_remains_idle(monkeypatch) -> None:
    """The source-RCT baseline must not provide target RCT rows to a learner."""

    osrct = pd.concat(
        [
            _site_data([(0, 0, 0), (1, 1, 1)]).assign(site="pi", F="idle"),
            _site_data([(0, 0, 0), (1, 1, 1)]).assign(site="a", F="idle"),
            _site_data([(0, 0, 0), (0, 0, 0)]).assign(site="b", F="idle"),
        ],
        ignore_index=True,
    )
    rct = pd.concat(
        [
            _site_data([(0, 0, 0), (1, 1, 1)]).assign(site="a", F="on"),
            _site_data([(0, 0, 0), (0, 0, 0)]).assign(site="b", F="on"),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(
        exp18,
        "load_prepared_data",
        lambda _: (osrct, rct, {}),
    )
    config = Exp18Config(
        studies=("gainloss",),
        n_source_sites=2,
        expected_non_target_sites=2,
        source_regime="on",
    )

    environments, support, composition = exp18._load_site_environments(
        config,
        "gainloss",
        ("a", "b"),
    )

    assert set(environments["pi"]["F"]) == {"idle"}
    assert set(environments[POOLED_SOURCE_ENVIRONMENT]["F"]) == {"on"}
    assert set(
        support.loc[
            support["environment"].eq(POOLED_SOURCE_ENVIRONMENT),
            "environment_regime",
        ]
    ) == {"on"}
    assert composition["source_regime"].tolist() == ["on", "on"]
    output = _leaf_output_dir(config)
    assert output.name == "source2_pooled_on_target_anchored"
    assert output.parent.name == "cbpi_cbra"


def test_naturalness_final_training_policy_uses_study_specific_role_union(
    tmp_path: Path,
) -> None:
    """CB sources should equal each study's proper plus tuning cohort."""

    reference_path = tmp_path / "target_gp_posterior_bounds.csv"
    reference_path.touch()
    pd.DataFrame(
        {
            "study": ["a", "a", "a", "b", "b", "b"],
            "site": ["a_proper", "a_tuning", "a_cal", "b_proper", "b_tuning", "b_cal"],
            "role": [
                "proper_training",
                "policy_tuning",
                "endpoint_calibration",
                "proper_training",
                "policy_tuning",
                "endpoint_calibration",
            ],
            "target_site": ["pi"] * 6,
        }
    ).to_csv(tmp_path / "source_site_roles.csv", index=False)

    def prepared(sites: list[str]) -> exp18.PreparedStudyData:
        """Build matched idle/RCT frames for source-selection validation."""

        frame = pd.concat(
            [
                _site_data([(0, 0, 0), (1, 1, 1)]).assign(site=site)
                for site in sites
            ],
            ignore_index=True,
        )
        return exp18.PreparedStudyData(idle=frame.copy(), rct=frame.copy())

    fitting_data = {
        "a": prepared(["pi", "a_proper", "a_tuning", "a_cal"]),
        "b": prepared(["pi", "b_proper", "b_tuning", "b_cal"]),
    }
    config = Exp18Config(
        studies=("a", "b"),
        source_site_policy="naturalness_final_training",
        naturalness_reference=str(reference_path),
    )

    selected, diagnostics = exp18._select_source_sites_by_study(
        config,
        fitting_data,
    )

    assert selected == {
        "a": ("a_proper", "a_tuning"),
        "b": ("b_proper", "b_tuning"),
    }
    assert set(diagnostics["role"]) == {"proper_training", "policy_tuning"}
    assert not diagnostics["source_site"].str.endswith("_cal").any()


def test_parallel_task_builder_creates_one_task_per_independent_endpoint_fit() -> None:
    """WB and CB endpoint fits should be separable before execution."""

    config = Exp18Config(n_source_sites=2, expected_non_target_sites=2)
    environments = {
        "pi": _site_data([(0, 0, 0), (1, 1, 1)]),
        POOLED_SOURCE_ENVIRONMENT: _site_data([(0, 0, 0), (1, 1, 1)]),
    }

    tasks = _build_study_tasks(
        config,
        "gainloss",
        environments,
        (POOLED_SOURCE_ENVIRONMENT,),
        10,
    )

    # 2 environments * 2 WB endpoints + 2 ordered pairs * 2 methods * 2 endpoints.
    assert len(tasks) == 12
    assert [task.task_id for task in tasks] == list(range(10, 22))
    assert sum(task.kind == "wb" for task in tasks) == 4
    assert sum(task.kind == "cb" for task in tasks) == 8


def test_task_builder_adds_ipw_and_dr_only_when_selected() -> None:
    """Each requested paper learner should add both pairs and endpoints."""

    config = Exp18Config(
        n_source_sites=2,
        expected_non_target_sites=2,
        meta_learners=("cb_pi", "cb_ra", "cb_ipw", "cb_dr"),
    )
    environments = {
        "pi": _site_data([(0, 0, 0), (1, 1, 1)]),
        POOLED_SOURCE_ENVIRONMENT: _site_data([(0, 0, 0), (1, 1, 1)]),
    }

    tasks = _build_study_tasks(
        config,
        "gainloss",
        environments,
        (POOLED_SOURCE_ENVIRONMENT,),
        0,
    )

    assert len(tasks) == 20
    assert sum(task.kind == "wb" for task in tasks) == 4
    assert sum(task.kind == "cb" for task in tasks) == 16
    assert {task.method for task in tasks if task.kind == "cb"} == {
        "cb_pi",
        "cb_ra",
        "cb_ipw",
        "cb_dr",
    }


def test_assembly_intersects_wb_and_cb_candidates_at_target_strata() -> None:
    """Final endpoints should be max lower/min upper, never target-RCT inputs."""

    config = Exp18Config(n_source_sites=2, expected_non_target_sites=2)
    environments = {
        "pi": _site_data([(0, 0, 0), (1, 1, 1)]),
        POOLED_SOURCE_ENVIRONMENT: _site_data([(0, 0, 0), (1, 1, 1)]),
    }
    covariates = ["resp_age_bin", "resp_gender_bin", "resp_polideo_bin"]
    support = pd.concat(
        [
            data[covariates]
            .assign(
                study="gainloss",
                environment=site,
                n_environment_rows=1,
                environment_regime="idle",
                observed_support=True,
            )
            for site, data in environments.items()
        ],
        ignore_index=True,
    )
    tasks = _build_study_tasks(
        config,
        "gainloss",
        environments,
        (POOLED_SOURCE_ENVIRONMENT,),
        0,
    )
    results: dict[int, FitResult] = {}
    wb_lower = {
        "pi": [-0.5, -0.4],
        POOLED_SOURCE_ENVIRONMENT: [-0.1, -0.15],
    }
    wb_upper = {
        "pi": [0.5, 0.5],
        POOLED_SOURCE_ENVIRONMENT: [0.2, 0.3],
    }
    for task in tasks:
        if task.kind == "wb":
            assert task.environment is not None
            values = (
                wb_lower[task.environment]
                if task.endpoint == "lower"
                else wb_upper[task.environment]
            )
        elif task.method == "cb_pi":
            if (
                task.left_environment == "pi"
                and task.right_environment == POOLED_SOURCE_ENVIRONMENT
            ):
                values = [0.9, 0.9] if task.endpoint == "lower" else [-0.9, -0.9]
            else:
                values = [0.0, 0.0] if task.endpoint == "lower" else [0.1, 0.1]
        else:
            values = [-0.05, -0.05] if task.endpoint == "lower" else [0.15, 0.15]
        results[task.task_id] = FitResult(
            task_id=task.task_id,
            predictions=np.asarray(values, dtype=float),
            runtime_seconds=0.01,
        )

    final, cb_pairs, wb_bounds = _assemble_study_bounds(
        config,
        "gainloss",
        environments,
        support,
        tasks,
        results,
    )

    assert len(cb_pairs) == 8
    assert len(wb_bounds) == 2 * 2
    assert final["cb_pi_lower"].tolist() == [0.0, 0.0]
    assert final["cb_pi_upper"].tolist() == [0.1, 0.1]
    assert final["cb_ra_lower"].tolist() == [-0.05, -0.05]
    assert final["cb_ra_upper"].tolist() == [0.15, 0.15]
    excluded_pair = cb_pairs.loc[
        cb_pairs["method"].eq("cb_pi")
        & cb_pairs["left_environment"].eq("pi")
        & cb_pairs["right_environment"].eq(POOLED_SOURCE_ENVIRONMENT)
    ]
    assert excluded_pair["candidate_interval_inverted"].all()
    assert not excluded_pair["eligible_cb_candidate"].any()


def test_empty_estimated_intersection_uses_a_transparent_full_range_fallback() -> None:
    """Estimated endpoint conflicts must not be repaired by swapping endpoints."""

    lower, upper, empty = _finalize_binary_cate_intersection(
        pd.Series([0.4, np.nan, -0.2]),
        pd.Series([0.3, 0.1, 0.2]),
    )

    assert empty.tolist() == [True, True, False]
    assert lower.tolist() == [-1.0, -1.0, -0.2]
    assert upper.tolist() == [1.0, 1.0, 0.2]


def test_exp18_cli_exposes_cpu_parallelism_and_manifest_provenance() -> None:
    """The cluster entrypoint should make worker and provenance controls explicit."""

    config = parse_args(
        [
            "--n-jobs",
            "8",
            "--data-source",
            "exp16",
            "--source-site-policy",
            "naturalness_final_training",
            "--source-regime",
            "on",
            "--worker-threads",
            "1",
            "--meta-learners",
            "cb_ipw",
            "cb_dr",
            "--naturalness-pre-all-triggered-folder",
            "latent",
            "--gamma",
            "2.5",
            "--cluster-manifest-path",
            "cluster_scripts/job_exp18_cb_meta_learner_comparison.yml",
            "--no-plots",
        ]
    )

    assert config.n_jobs == 8
    assert config.data_source == "exp16"
    assert config.source_site_policy == "naturalness_final_training"
    assert config.source_regime == "on"
    assert config.worker_threads == 1
    assert config.meta_learners == ("cb_ipw", "cb_dr")
    assert config.gamma == 2.5
    assert config.make_plots is False
    assert config.cluster_manifest_path == (
        "cluster_scripts/job_exp18_cb_meta_learner_comparison.yml"
    )
    assert config.naturalness_backend == "xgboost"
    assert config.naturalness_pre_all_triggered_folder == "latent"


def test_exp18_cli_accepts_binary_gamma_search_controls() -> None:
    """Binary width matching should expose its bracket and stop controls."""

    config = parse_args(
        [
            "--gamma",
            "width_matched_binary",
            "--gamma-search-grid",
            "1,1.5,3,5",
            "--gamma-search-tolerance",
            "0.02",
            "--gamma-search-width-tolerance",
            "0.004",
            "--gamma-search-max-iterations",
            "7",
        ]
    )

    assert config.gamma == "width_matched_binary"
    assert config.gamma_search_grid == (1.0, 1.5, 3.0, 5.0)
    assert config.gamma_search_tolerance == 0.02
    assert config.gamma_search_width_tolerance == 0.004
    assert config.gamma_search_max_iterations == 7
    assert exp18._blearner_gammas(config) == (1.0, 1.5, 3.0, 5.0)


def test_exp_beta_gamma_resolves_selects_and_isolates_output() -> None:
    """The exp-beta strategy should perform one fixed-Gamma comparison."""

    config = parse_args(["--gamma", "exp_beta", "--beta", "1.25"])
    resolved_gamma = float(np.exp(1.25))
    assert config.gamma == "exp_beta"
    assert exp18._blearner_gammas(config) == pytest.approx((resolved_gamma,))
    output = _leaf_output_dir(config, data_source="exp16")
    assert output.name == "gender_1.25"
    assert output.parent.name == "exp_beta"

    blearner_bounds = pd.DataFrame(
        {
            "study": ["a"],
            "resp_age_bin": [0],
            "resp_gender_bin": [0],
            "resp_polideo_bin": [0],
            "gamma": [resolved_gamma],
            "blearner_width": [0.4],
        }
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a"],
            "resp_age_bin": [0],
            "resp_gender_bin": [0],
            "resp_polideo_bin": [0],
            "naturalness_width": [0.5],
        }
    )
    selected, scores = exp18._select_blearner_bounds(
        Exp18Config(studies=("a",), gamma="exp_beta", beta=1.25),
        blearner_bounds,
        naturalness_reference,
    )

    assert selected["gamma"].iloc[0] == pytest.approx(resolved_gamma)
    assert scores["selection_metric"].iloc[0] == "exp_beta"
    assert scores["gamma_strategy"].iloc[0] == "exp_beta"
    assert scores["selected"].all()


def test_exp_beta_gamma_rejects_negative_beta() -> None:
    """B-Learner Gamma cannot fall below one."""

    with pytest.raises(ValueError, match="finite nonnegative beta"):
        Exp18Config(gamma="exp_beta", beta=-0.1)


def test_oracle_gamma_matches_exp_beta_for_balanced_hidden_binary_driver() -> None:
    """Symmetric omitted binary selection should yield Gamma equal exp(beta)."""

    beta = 0.75
    probability_low = 1.0 / (1.0 + np.exp(beta))
    probability_high = 1.0 / (1.0 + np.exp(-beta))
    target_idle = pd.DataFrame(
        {
            "reported_stratum": [0, 0, 0, 0],
            "_selection_prob": [
                probability_low,
                probability_low,
                probability_high,
                probability_high,
            ],
        }
    )

    calibration = exp18._oracle_gamma_calibration(
        target_idle,
        ["reported_stratum"],
        study="a",
    )

    assert calibration["gamma"] == pytest.approx(np.exp(beta))
    assert calibration["oracle_log_gamma"] == pytest.approx(beta)
    assert calibration["oracle_n_blearner_strata"] == 1


def test_oracle_gamma_is_one_when_binary_driver_is_fully_observed() -> None:
    """Conditioning on the exact binary driver should remove hidden variation."""

    target_idle = pd.DataFrame(
        {
            "binary_driver": [0, 0, 1, 1],
            "_selection_prob": [0.2, 0.2, 0.8, 0.8],
        }
    )

    calibration = exp18._oracle_gamma_calibration(
        target_idle,
        ["binary_driver"],
        study="a",
    )

    assert calibration["gamma"] == 1.0
    assert calibration["oracle_log_gamma"] == 0.0
    assert calibration["oracle_n_blearner_strata"] == 2


def test_dgp_oracle_gamma_uses_full_pre_selection_target_rct() -> None:
    """DGP calibration should recover a hidden standardized binary driver."""

    beta = 0.75
    full_rct = pd.DataFrame(
        {
            "site": ["pi", "pi", "pi", "pi"],
            "resp_gender": [-1.0, -1.0, 1.0, 1.0],
            "reported_stratum": [0, 0, 0, 0],
        }
    )

    calibration = exp18._dgp_oracle_gamma_calibration(
        full_rct,
        ["reported_stratum"],
        study="a",
        target_site="pi",
        pattern="gender",
        beta=beta,
    )

    assert calibration["gamma"] == pytest.approx(np.exp(beta))
    assert calibration["oracle_log_gamma"] == pytest.approx(beta)
    assert calibration["oracle_n_target_idle_rows"] == 4
    assert calibration["oracle_calibration_population"] == (
        "full_pre_selection_target_rct"
    )


def test_dgp_oracle_gamma_is_one_when_gender_driver_is_fully_observed() -> None:
    """Complete-case binary gender should leave no within-stratum deviation."""

    full_rct = pd.DataFrame(
        {
            "site": ["pi", "pi", "pi", "pi"],
            "resp_gender": [0.0, 0.0, 1.0, 1.0],
            "resp_gender_bin": [0, 0, 1, 1],
        }
    )

    calibration = exp18._dgp_oracle_gamma_calibration(
        full_rct,
        ["resp_gender_bin"],
        study="a",
        target_site="pi",
        pattern="gender",
        beta=1.0,
    )

    assert calibration["oracle_log_gamma"] == 0.0
    assert calibration["gamma"] == 1.0


def test_post_selection_propensity_is_identity_under_balanced_assignment() -> None:
    """Balanced assignment must leave preferred probabilities untouched."""

    preferred = np.array([0.05, 0.3124, 0.5, 0.7803, 0.95])

    balanced = exp18._post_selection_propensity(preferred, 0.5)
    unbalanced = exp18._post_selection_propensity(preferred, 0.67)

    assert balanced is not preferred
    np.testing.assert_array_equal(balanced, preferred)
    # Unbalanced assignment shifts the odds by exactly p/(1-p).
    expected_odds = (preferred / (1.0 - preferred)) * (0.67 / 0.33)
    np.testing.assert_allclose(
        unbalanced / (1.0 - unbalanced), expected_odds, rtol=1e-12
    )


def test_dgp_oracle_gamma_is_invariant_to_stratum_constant_assignment() -> None:
    """The randomization probability must cancel from the DGP-oracle Gamma.

    Both propensities carry the same ``logit(p)`` offset when assignment is
    constant inside a stratum, so Gamma may not depend on it. This is why the
    unbalanced designs can reuse the Exp16 calibration formula. The cancellation
    is algebraically exact but travels different floating-point paths, so
    equality is asserted to tolerance rather than bitwise.
    """

    full_rct = pd.DataFrame(
        {
            "site": ["pi"] * 6,
            "stratum": [0, 0, 0, 1, 1, 1],
            "driver": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    preferred = expit(0.8 * full_rct["driver"].to_numpy())

    gammas = {
        probability: exp18._dgp_oracle_gamma_calibration(
            full_rct,
            ["stratum"],
            study="a",
            target_site="pi",
            pattern="external_driver",
            beta=0.8,
            randomized_treatment_probability=probability,
            preferred_probability=preferred,
        )["gamma"]
        for probability in (0.5, 0.6, 0.67, 0.75)
    }

    np.testing.assert_allclose(
        list(gammas.values()), gammas[0.5], rtol=1e-12
    )
    assert gammas[0.5] > 1.0


def test_dgp_oracle_gamma_rejects_misaligned_preferred_probability() -> None:
    """A mechanism vector must cover every pre-selection row exactly once."""

    full_rct = pd.DataFrame(
        {"site": ["pi", "pi", "pi"], "stratum": [0, 0, 1]}
    )

    with pytest.raises(ValueError, match="one entry per pre-selection row"):
        exp18._dgp_oracle_gamma_calibration(
            full_rct,
            ["stratum"],
            study="a",
            target_site="pi",
            pattern="external_driver",
            beta=1.0,
            preferred_probability=np.array([0.4, 0.6]),
        )


def test_realized_oracle_gamma_responds_to_unbalanced_assignment() -> None:
    """Unlike the DGP oracle, the realized oracle does not cancel ``p``.

    Its observed propensity is an arithmetic mean of probabilities rather than
    an odds-preserving aggregate, so the assignment probability must be
    supplied rather than assumed.
    """

    target_idle = pd.DataFrame(
        {
            "stratum": [0, 0, 0, 0],
            "osrct_preferred_treatment_probability": [0.2, 0.2, 0.8, 0.8],
        }
    )

    balanced = exp18._oracle_gamma_calibration(
        target_idle, ["stratum"], study="a"
    )
    unbalanced = exp18._oracle_gamma_calibration(
        target_idle,
        ["stratum"],
        study="a",
        randomized_treatment_probability=0.67,
    )

    assert balanced["oracle_gamma_assumption"] == exp18.ORACLE_GAMMA_ASSUMPTION
    assert unbalanced["oracle_gamma_assumption"] == (
        exp18.UNBALANCED_ORACLE_GAMMA_ASSUMPTION
    )
    assert balanced["oracle_randomized_treatment_probability"] == 0.5
    assert unbalanced["oracle_randomized_treatment_probability"] == 0.67
    assert unbalanced["gamma"] != pytest.approx(balanced["gamma"])


def test_dgp_oracle_strategy_parses_and_isolates_output() -> None:
    """The DGP strategy should be CLI-selectable and get its own output leaf."""

    config = parse_args(["--gamma", "dgp_oracle"])

    assert config.gamma == "dgp_oracle"
    assert config.require_observed_gender is True
    assert exp18._blearner_gammas(config) == ()
    assert _leaf_output_dir(config, data_source="exp16").parent.name == (
        "dgp_oracle"
    )


def test_oracle_gamma_resolves_selects_and_isolates_output() -> None:
    """Oracle strategy should select one data-dependent Gamma per study."""

    config = parse_args(["--gamma", "oracle"])
    assert config.gamma == "oracle"
    assert exp18._blearner_gammas(config) == ()
    output = _leaf_output_dir(config, data_source="exp16")
    assert output.name == "gender_1.0"
    assert output.parent.name == "oracle"

    blearner_bounds = pd.DataFrame(
        {
            "study": ["a"],
            "resp_age_bin": [0],
            "resp_gender_bin": [0],
            "resp_polideo_bin": [0],
            "gamma": [2.25],
            "blearner_width": [0.4],
            "oracle_log_gamma": [np.log(2.25)],
            "oracle_selection_probability_column": ["_selection_prob"],
            "oracle_n_target_idle_rows": [20],
            "oracle_n_blearner_strata": [1],
            "oracle_full_propensity_min": [0.2],
            "oracle_full_propensity_max": [0.8],
            "oracle_observed_propensity_min": [0.5],
            "oracle_observed_propensity_max": [0.5],
            "oracle_gamma_assumption": [exp18.ORACLE_GAMMA_ASSUMPTION],
        }
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a"],
            "resp_age_bin": [0],
            "resp_gender_bin": [0],
            "resp_polideo_bin": [0],
            "naturalness_width": [0.5],
        }
    )

    selected, scores = exp18._select_blearner_bounds(
        Exp18Config(studies=("a",), gamma="oracle"),
        blearner_bounds,
        naturalness_reference,
    )

    assert selected["gamma"].iloc[0] == pytest.approx(2.25)
    assert scores["selection_metric"].iloc[0] == (
        "oracle_msm_max_log_odds_deviation"
    )
    assert scores["gamma_strategy"].iloc[0] == "oracle"
    assert scores["oracle_log_gamma"].iloc[0] == pytest.approx(np.log(2.25))
    assert scores["selected"].all()


def test_oracle_gamma_requires_saved_osrct_probabilities() -> None:
    """Oracle calibration should reject non-OSRCT target tables."""

    with pytest.raises(ValueError, match="preferred-treatment probabilities"):
        exp18._oracle_gamma_calibration(
            pd.DataFrame({"reported_stratum": [0, 1]}),
            ["reported_stratum"],
            study="a",
        )


class _GammaScaledBLearner:
    """Learner whose endpoints depend only on Gamma and the target rows."""

    def __init__(self, gamma: float) -> None:
        self.gamma = float(gamma)

    def fit(
        self,
        x: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> None:
        """Accept the target fitting arrays without fitting a model."""

        del x, treatment, outcome

    def effect(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one Gamma-scaled interval per requested target row."""

        first = np.asarray(x, dtype=float)[:, 0]
        return -self.gamma * (first + 1.0), self.gamma * (first + 2.0)


def _grid_target_idle(offset: int) -> pd.DataFrame:
    """Return one study's target idle rows with study-specific strata."""

    return pd.DataFrame(
        {
            "resp_age_bin": [offset, offset, offset + 1, offset + 1],
            "resp_gender_bin": [0, 0, 1, 1],
            "resp_polideo_bin": [0, 0, 0, 0],
            "iv": [0, 1, 0, 1],
            "dv_binary": [0, 1, 1, 0],
        }
    )


def test_width_matched_grid_matches_gamma_by_gamma_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched grid fitting reproduces the serial per-Gamma bounds exactly."""

    config = replace(
        Exp18Config(studies=("a", "b")),
        gamma=WIDTH_MATCHED_GAMMA,
        n_jobs=1,
        worker_threads=1,
        verbose=False,
    )
    target_idle_by_study = {"a": _grid_target_idle(0), "b": _grid_target_idle(4)}
    environments = {
        study: {exp18._study_target_site(config, study): frame}
        for study, frame in target_idle_by_study.items()
    }

    monkeypatch.setattr(
        exp18,
        "_make_blearner",
        lambda config, gamma: _GammaScaledBLearner(gamma),
    )
    serial = pd.concat(
        [
            exp18._fit_blearner_study_bounds(
                config,
                study,
                target_idle_by_study[study],
                BLEARNER_GAMMA_GRID,
            )
            for study in config.studies
        ],
        ignore_index=True,
    )

    fitted_gammas: list[float] = []

    def fake_make_blearner(
        fit_config: object,
        gamma: float,
        *,
        model_n_jobs: int = -1,
    ) -> _GammaScaledBLearner:
        """Record one batched fit and return the shared fake learner."""

        del fit_config
        assert model_n_jobs == 1
        fitted_gammas.append(float(gamma))
        return _GammaScaledBLearner(gamma)

    monkeypatch.setattr(benchmarks, "make_blearner", fake_make_blearner)
    batched = exp18._fit_blearner_bounds(config, environments)

    assert len(fitted_gammas) == len(config.studies) * len(BLEARNER_GAMMA_GRID)
    assert sorted(set(fitted_gammas)) == sorted(BLEARNER_GAMMA_GRID)
    pd.testing.assert_frame_equal(
        batched.drop(columns=["fit_runtime_seconds"]),
        serial.drop(columns=["fit_runtime_seconds"]),
    )


def test_grid_fits_are_keyed_by_study_and_gamma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every batched fit reaches the study and Gamma row that requested it."""

    config = replace(
        Exp18Config(studies=("a", "b")),
        gamma=WIDTH_MATCHED_GAMMA,
        n_jobs=1,
        worker_threads=1,
        verbose=False,
    )
    target_idle_by_study = {"a": _grid_target_idle(0), "b": _grid_target_idle(4)}

    monkeypatch.setattr(
        benchmarks,
        "make_blearner",
        lambda fit_config, gamma, model_n_jobs=-1: _GammaScaledBLearner(gamma),
    )
    endpoints = exp18._fit_blearner_gamma_grid(
        config,
        target_idle_by_study,
        BLEARNER_GAMMA_GRID,
    )

    assert set(endpoints) == {"a", "b"}
    for study, target_idle in target_idle_by_study.items():
        strata = exp18._blearner_study_inputs(config, study, target_idle)
        assert set(endpoints[study]) == set(BLEARNER_GAMMA_GRID)
        for gamma, result in endpoints[study].items():
            expected_lower, expected_upper = _GammaScaledBLearner(gamma).effect(
                strata.strata_x
            )
            assert result.lower.tolist() == expected_lower.tolist()
            assert result.upper.tolist() == expected_upper.tolist()


def test_blearner_rearranges_crossed_finite_sample_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent endpoint forests should yield coherent saved intervals."""

    class FakeBLearner:
        """Return fixed crossed and ordered endpoint predictions."""

        def fit(
            self,
            x: np.ndarray,
            treatment: np.ndarray,
            outcome: np.ndarray,
        ) -> None:
            """Accept the target fitting arrays without fitting a model."""

            del x, treatment, outcome

        def effect(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return one crossed and one already ordered interval."""

            assert len(x) == 2
            return np.array([0.3, -0.4]), np.array([0.1, 0.2])

    monkeypatch.setattr(
        exp18,
        "_make_blearner",
        lambda config, gamma: FakeBLearner(),
    )
    target_idle = pd.DataFrame(
        {
            "resp_age_bin": [0, 0, 1, 1],
            "resp_gender_bin": [0, 0, 0, 0],
            "resp_polideo_bin": [0, 0, 0, 0],
            "iv": [0, 1, 0, 1],
            "dv_binary": [0, 1, 1, 0],
        }
    )

    bounds = exp18._fit_blearner_study_bounds(
        Exp18Config(studies=("a",)),
        "a",
        target_idle,
        (float(np.exp(0.25)),),
    )

    assert bounds["blearner_lower_raw"].tolist() == [0.3, -0.4]
    assert bounds["blearner_upper_raw"].tolist() == [0.1, 0.2]
    assert bounds["blearner_bounds_rearranged"].tolist() == [True, False]
    assert bounds["blearner_lower"].tolist() == [0.1, -0.4]
    assert bounds["blearner_upper"].tolist() == [0.3, 0.2]
    assert bounds["blearner_width"].tolist() == pytest.approx([0.2, 0.6])


def test_exp_beta_gamma_retains_zero_optimal_naturalness_strata() -> None:
    """Fixed Gamma must not drop strata lacking optimal naturalness draws."""

    gamma = float(np.exp(0.5))
    blearner_bounds = pd.DataFrame(
        {
            "study": ["a", "a"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 0],
            "resp_polideo_bin": [0, 0],
            "gamma": [gamma, gamma],
            "blearner_width": [0.2, 0.4],
            "n_target_idle_rows": [10, 20],
        }
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a", "a"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 0],
            "resp_polideo_bin": [0, 0],
            "naturalness_width": [np.nan, np.nan],
        }
    )

    selected, scores = exp18._select_blearner_bounds(
        Exp18Config(studies=("a",), gamma="exp_beta", beta=0.5),
        blearner_bounds,
        naturalness_reference,
    )

    assert len(selected) == 2
    assert selected["absolute_stratum_width_gap"].isna().all()
    assert scores["n_width_matched_strata"].tolist() == [0]
    assert scores[
        "target_prevalence_weighted_naturalness_width"
    ].isna().all()
    assert scores[
        "target_prevalence_weighted_blearner_width"
    ].isna().all()


def test_width_matched_gamma_selection_is_per_study_and_persistable() -> None:
    """Width matching should select one Gamma per study from saved scores."""

    blearner_bounds = pd.DataFrame(
        {
            "study": ["a", "a", "b", "b"],
            "resp_age_bin": [0, 0, 1, 1],
            "resp_gender_bin": [0, 0, 1, 1],
            "resp_polideo_bin": [0, 0, 1, 1],
            "gamma": [1.0, 2.0, 1.0, 2.0],
            "blearner_width": [0.1, 0.4, 0.1, 0.5],
        }
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a", "b"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 1],
            "resp_polideo_bin": [0, 1],
            "naturalness_width": [0.4, 0.1],
        }
    )

    selected, scores = exp18._select_blearner_bounds(
        Exp18Config(studies=("a", "b")),
        blearner_bounds,
        naturalness_reference,
    )

    assert selected.set_index("study")["gamma"].to_dict() == {"a": 2.0, "b": 1.0}
    assert scores.loc[scores["selected"]].set_index("study")[
        "selected_gamma"
    ].to_dict() == {"a": 2.0, "b": 1.0}
    assert set(scores["gamma_strategy"]) == {"width_matched"}
    assert set(scores["requested_gamma"]) == {"width_matched"}


def test_binary_width_score_uses_target_stratum_prevalence() -> None:
    """Aggregate search gap should weight widths by target stratum counts."""

    blearner_bounds = pd.DataFrame(
        {
            "study": ["a", "a"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 0],
            "resp_polideo_bin": [0, 0],
            "gamma": [2.0, 2.0],
            "blearner_width": [0.2, 0.8],
            "n_target_idle_rows": [90, 10],
        }
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a", "a"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 0],
            "resp_polideo_bin": [0, 0],
            "naturalness_width": [0.1, 0.3],
        }
    )

    _, scores = exp18._score_blearner_widths(
        blearner_bounds,
        naturalness_reference,
    )

    assert scores["target_prevalence_weighted_blearner_width"].iloc[
        0
    ] == pytest.approx(0.26)
    assert scores["target_prevalence_weighted_naturalness_width"].iloc[
        0
    ] == pytest.approx(0.12)
    assert scores["target_prevalence_weighted_width_gap"].iloc[
        0
    ] == pytest.approx(0.14)


def test_binary_gamma_search_brackets_and_refines_weighted_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary mode should fit midpoint Gammas until matching target width."""

    def fake_fit(
        config: Exp18Config,
        study: str,
        target_idle: pd.DataFrame,
        gammas: tuple[float, ...],
    ) -> pd.DataFrame:
        """Return deterministic widths increasing linearly with Gamma."""

        del config, target_idle
        return pd.DataFrame(
            {
                "study": study,
                "resp_age_bin": 0,
                "resp_gender_bin": 1,
                "resp_polideo_bin": 0,
                "gamma": [float(gamma) for gamma in gammas],
                "blearner_width": [float(gamma) / 10.0 for gamma in gammas],
                "n_target_idle_rows": 20,
            }
        )

    monkeypatch.setattr(exp18, "_fit_blearner_study_bounds", fake_fit)
    config = Exp18Config(
        studies=("a",),
        gamma="width_matched_binary",
        gamma_search_grid=(1.0, 2.0, 3.0, 4.0, 5.0),
        gamma_search_tolerance=0.01,
        gamma_search_width_tolerance=1e-12,
        gamma_search_max_iterations=10,
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a"],
            "resp_age_bin": [0],
            "resp_gender_bin": [1],
            "resp_polideo_bin": [0],
            "naturalness_width": [0.275],
        }
    )

    bounds = exp18._fit_blearner_binary_search_study_bounds(
        config,
        "a",
        pd.DataFrame(),
        naturalness_reference,
    )
    selected, scores = exp18._select_blearner_bounds(
        config,
        bounds,
        naturalness_reference,
    )

    assert set(bounds["gamma_search_stage"]) == {
        "initial_grid",
        "binary_search",
    }
    assert set(bounds["gamma_search_status"]) == {
        "binary_width_tolerance_reached"
    }
    assert bounds["gamma_search_iterations"].iloc[0] == 2
    assert selected["gamma"].iloc[0] == pytest.approx(2.75)
    assert scores.loc[scores["selected"], "selection_metric"].iloc[0] == (
        "absolute_target_prevalence_weighted_width_gap"
    )


def test_binary_gamma_search_stops_on_nonmonotonic_initial_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Estimated width reversals should disable midpoint refinement."""

    widths = {1.0: 0.1, 2.0: 0.3, 3.0: 0.2, 4.0: 0.4, 5.0: 0.5}

    def fake_fit(
        config: Exp18Config,
        study: str,
        target_idle: pd.DataFrame,
        gammas: tuple[float, ...],
    ) -> pd.DataFrame:
        """Return one deliberately nonmonotone coarse-grid fit."""

        del config, target_idle
        return pd.DataFrame(
            {
                "study": study,
                "resp_age_bin": 0,
                "resp_gender_bin": 1,
                "resp_polideo_bin": 0,
                "gamma": [float(gamma) for gamma in gammas],
                "blearner_width": [widths[float(gamma)] for gamma in gammas],
                "n_target_idle_rows": 20,
            }
        )

    monkeypatch.setattr(exp18, "_fit_blearner_study_bounds", fake_fit)
    config = Exp18Config(
        studies=("a",),
        gamma="width_matched_binary",
    )
    naturalness_reference = pd.DataFrame(
        {
            "study": ["a"],
            "resp_age_bin": [0],
            "resp_gender_bin": [1],
            "resp_polideo_bin": [0],
            "naturalness_width": [0.38],
        }
    )

    bounds = exp18._fit_blearner_binary_search_study_bounds(
        config,
        "a",
        pd.DataFrame(),
        naturalness_reference,
    )

    assert bounds["gamma"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert set(bounds["gamma_search_status"]) == {
        "nonmonotonic_initial_grid"
    }
    assert not bounds["gamma_search_monotonic"].any()
    assert bounds["gamma_search_iterations"].eq(0).all()


def test_data_source_must_match_naturalness_reference() -> None:
    """Exp18 should reject cross-benchmark fitting/reference combinations."""

    auto = Exp18Config(data_source="auto")
    assert exp18._resolve_data_source(auto, "exp15") == "exp15"
    assert exp18._resolve_data_source(auto, "exp16") == "exp16"

    with pytest.raises(ValueError, match="must come from the same experiment"):
        exp18._resolve_data_source(
            Exp18Config(data_source="exp16"),
            "exp15",
        )


def test_comparison_merges_selected_blearner_bounds_and_summary() -> None:
    """Selected Gamma and B-Learner metrics should reach saved comparison rows."""

    keys = {
        "study": ["gainloss"],
        "resp_age_bin": [0],
        "resp_gender_bin": [0],
        "resp_polideo_bin": [0],
    }
    cb_final_bounds = pd.DataFrame(keys).assign(
        cb_pi_lower=-0.5,
        cb_pi_upper=0.5,
        cb_pi_width=1.0,
        cb_pi_interval_inverted=False,
        cb_pi_raw_intersection_inverted=False,
        cb_pi_used_full_range_fallback=False,
        cb_ra_lower=-0.4,
        cb_ra_upper=0.4,
        cb_ra_width=0.8,
        cb_ra_interval_inverted=False,
        cb_ra_raw_intersection_inverted=False,
        cb_ra_used_full_range_fallback=False,
    )
    naturalness_reference = pd.DataFrame(keys).assign(
        naturalness_lower=-0.3,
        naturalness_upper=0.3,
        naturalness_width=0.6,
        n_target_idle_rows=20,
        rct_cate=0.1,
        rct_cate_ci90_lower=0.0,
        rct_cate_ci90_upper=0.2,
        rct_cate_available=True,
        blackbox_reference_cate=0.15,
        naturalness_reference_experiment="exp15",
        lp_posterior_prob_optimal=0.9,
    )
    selected_blearner_bounds = pd.DataFrame(keys).assign(
        gamma=2.5,
        blearner_lower=-0.2,
        blearner_upper=0.2,
        blearner_width=0.4,
        outside_binary_cate_support=False,
        fit_runtime_seconds=1.5,
        absolute_stratum_width_gap=0.2,
    )

    comparison, summary = exp18._build_comparison(
        cb_final_bounds,
        naturalness_reference,
        selected_blearner_bounds,
    )

    assert comparison.loc[0, "blearner_gamma"] == 2.5
    assert bool(comparison.loc[0, "blearner_covers_rct_cate"])
    assert bool(comparison.loc[0, "blearner_covers_blackbox_cate"])
    assert summary["n_blackbox_labeled_strata"].eq(1).all()
    assert summary["blackbox_point_coverage"].notna().all()
    assert "blearner" in set(summary["method"])


def test_comparison_plots_keep_vertical_and_horizontal_aesthetics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Comparison artifacts should preserve both requested plot orientations."""

    comparison = pd.DataFrame(
        {
            "study": ["gainloss", "gainloss"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 1],
            "resp_polideo_bin": [0, 1],
            "cb_pi_lower": [-0.5, -0.4],
            "cb_pi_upper": [0.4, 0.5],
            "cb_pi_interval_inverted": [False, False],
            "cb_ra_lower": [-0.6, -0.3],
            "cb_ra_upper": [0.3, 0.6],
            "cb_ra_interval_inverted": [False, False],
            "naturalness_lower": [-0.4, -0.2],
            "naturalness_upper": [0.2, 0.4],
            "naturalness_envelope_lower": [-0.55, -0.35],
            "naturalness_envelope_upper": [0.35, 0.55],
            "blearner_lower": [-0.3, -0.1],
            "blearner_upper": [0.1, 0.3],
            "blearner_gamma": [2.5, 2.5],
            "rct_cate_available": [True, True],
            "rct_cate": [-0.1, 0.1],
            "rct_cate_ci90_lower": [-0.2, 0.0],
            "rct_cate_ci90_upper": [0.0, 0.2],
        }
    )
    captured = []
    monkeypatch.setattr(
        exp18.plt,
        "close",
        captured.append,
    )
    output_path = tmp_path / "comparison.png"

    exp18._make_comparison_plot(
        comparison,
        Exp18Config(studies=("gainloss",)),
        output_path,
    )

    figure = captured[0]
    axis = figure.axes[0]
    assert output_path.exists()
    assert figure._suptitle is None
    assert figure._supxlabel.get_text() == "Z=(age, gender, polideo)"
    assert axis.get_ylabel() == "CATE"
    assert axis.get_ylim() == (-1.0, 1.0)
    assert axis.get_title() == "gainloss"
    assert axis.title.get_fontweight() == "bold"
    assert np.allclose(
        [
            collection.get_segments()[0][0, 0]
            for collection in axis.collections[:5]
        ],
        [-0.4, -0.4 / 3.0, 0.4 / 3.0, 0.4, 0.0],
    )
    assert np.allclose(
        axis.collections[5].get_offsets()[:, 0],
        [0.0, exp18.STRATUM_GROUP_SPACING],
    )
    assert np.allclose(
        axis.get_xticks(),
        [0.0, exp18.STRATUM_GROUP_SPACING],
    )
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "(0,0,0)",
        "(1,1,1)",
    ]
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "CB-PI",
        "CB-RA",
        "Naturalness",
        "B-Learner",
        "RCT CATE 90% Wald CI",
        "Held-out RCT CATE",
    ]
    assert all("Gamma" not in text.get_text() for text in legend.get_texts())
    assert legend.get_frame_on() is False

    horizontal_output_path = tmp_path / "comparison_horizontal.png"
    exp18._make_comparison_plot(
        comparison,
        Exp18Config(studies=("gainloss",)),
        horizontal_output_path,
        horizontal=True,
    )

    horizontal_figure = captured[1]
    horizontal_axis = horizontal_figure.axes[0]
    assert horizontal_output_path.exists()
    assert horizontal_figure._suptitle is None
    assert horizontal_figure._supxlabel.get_text() == "CATE"
    assert horizontal_figure._supxlabel.get_fontsize() == (
        exp18.HORIZONTAL_AXIS_LABEL_FONTSIZE
    )
    assert horizontal_axis.get_ylabel() == "Z=(age, gender, polideo)"
    assert horizontal_axis.yaxis.label.get_size() == (
        exp18.HORIZONTAL_AXIS_LABEL_FONTSIZE
    )
    assert horizontal_axis.get_xlim() == (-1.0, 1.0)
    assert horizontal_axis.get_title() == "gainloss"
    assert horizontal_axis.title.get_fontweight() == "bold"
    assert np.allclose(
        [
            collection.get_segments()[0][0, 1]
            for collection in horizontal_axis.collections[:5]
        ],
        [-0.48, -0.24, 0.24, 0.48, 0.0],
    )
    assert np.allclose(
        horizontal_axis.collections[5].get_offsets()[:, 1],
        [0.0, exp18.HORIZONTAL_STRATUM_GROUP_SPACING],
    )
    assert np.allclose(
        horizontal_axis.get_yticks(),
        [0.0, exp18.HORIZONTAL_STRATUM_GROUP_SPACING],
    )
    assert [tick.get_text() for tick in horizontal_axis.get_yticklabels()] == [
        "(0,0,0)",
        "(1,1,1)",
    ]
    horizontal_legend = horizontal_figure.legends[0]
    assert [text.get_text() for text in horizontal_legend.get_texts()] == [
        "CB-PI",
        "CB-RA",
        "Naturalness",
        "B-Learner",
        "Held-out RCT CATE",
        "RCT CATE 90% Wald CI",
    ]
    assert horizontal_legend._ncols == 3
    assert all(
        text.get_fontsize() == exp18.HORIZONTAL_LEGEND_FONTSIZE
        for text in horizontal_legend.get_texts()
    )
    assert all(
        "Gamma" not in text.get_text()
        for text in horizontal_legend.get_texts()
    )
    assert horizontal_legend.get_frame_on() is False

    envelope_output_path = (
        tmp_path / "comparison_horizontal_with_naturalness_envelope.png"
    )
    exp18._make_comparison_plot(
        comparison,
        Exp18Config(studies=("gainloss",)),
        envelope_output_path,
        horizontal=True,
        naturalness_envelope=True,
    )

    envelope_figure = captured[2]
    envelope_axis = envelope_figure.axes[0]
    assert envelope_output_path.exists()
    assert envelope_axis.get_xlim() == (-1.0, 1.0)
    assert np.allclose(
        envelope_axis.collections[2].get_segments()[0][:, 0],
        [-0.55, 0.35],
    )
    assert [
        text.get_text() for text in envelope_figure.legends[0].get_texts()
    ] == [
        "CB-PI",
        "CB-RA",
        "Naturalness posterior envelope",
        "Naturalness",
        "B-Learner",
        "Held-out RCT CATE",
        "RCT CATE 90% Wald CI",
    ]


def test_comparison_plots_show_point_identified_blearner_estimates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Zero-width B-Learner intervals should remain visible as point markers."""

    comparison = pd.DataFrame(
        {
            "study": ["gainloss"],
            "resp_age_bin": [0],
            "resp_gender_bin": [0],
            "resp_polideo_bin": [0],
            "cb_pi_lower": [-0.5],
            "cb_pi_upper": [0.5],
            "cb_pi_interval_inverted": [False],
            "cb_ra_lower": [-0.4],
            "cb_ra_upper": [0.4],
            "cb_ra_interval_inverted": [False],
            "naturalness_lower": [-0.3],
            "naturalness_upper": [0.3],
            "blearner_lower": [0.37],
            "blearner_upper": [0.37],
            "blearner_gamma": [1.0],
            "rct_cate_available": [True],
            "rct_cate": [-0.2],
            "rct_cate_ci90_lower": [-0.3],
            "rct_cate_ci90_upper": [-0.1],
        }
    )
    captured = []
    monkeypatch.setattr(exp18.plt, "close", captured.append)
    config = Exp18Config(studies=("gainloss",))

    exp18._make_comparison_plot(
        comparison,
        config,
        tmp_path / "point_vertical.png",
    )
    exp18._make_comparison_plot(
        comparison,
        config,
        tmp_path / "point_horizontal.png",
        horizontal=True,
    )

    vertical_offsets = [
        collection.get_offsets()
        for collection in captured[0].axes[0].collections
        if hasattr(collection, "get_offsets")
    ]
    horizontal_offsets = [
        collection.get_offsets()
        for collection in captured[1].axes[0].collections
        if hasattr(collection, "get_offsets")
    ]
    assert any(
        np.allclose(offsets, [[0.4, 0.37]]) for offsets in vertical_offsets
    )
    assert any(
        np.allclose(offsets, [[0.37, 0.48]]) for offsets in horizontal_offsets
    )


def test_comparison_plot_includes_selected_ipw_and_dr_learners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Plot series and legend should follow configured four-learner selection."""

    comparison = pd.DataFrame(
        {
            "study": ["gainloss"],
            "resp_age_bin": [0],
            "resp_gender_bin": [1],
            "resp_polideo_bin": [0],
            **{
                f"{method}_{suffix}": [value]
                for method in exp18.PAPER_METHODS
                for suffix, value in (
                    ("lower", -0.5),
                    ("upper", 0.5),
                    ("interval_inverted", False),
                )
            },
            "naturalness_lower": [-0.4],
            "naturalness_upper": [0.4],
            "blearner_lower": [-0.3],
            "blearner_upper": [0.3],
            "rct_cate_available": [True],
            "rct_cate": [0.1],
            "rct_cate_ci90_lower": [0.0],
            "rct_cate_ci90_upper": [0.2],
        }
    )
    captured = []
    monkeypatch.setattr(exp18.plt, "close", captured.append)
    output_path = tmp_path / "comparison_all_meta_learners.png"
    config = Exp18Config(
        studies=("gainloss",),
        meta_learners=exp18.PAPER_METHODS,
    )

    exp18._make_comparison_plot(comparison, config, output_path)

    assert output_path.exists()
    legend = captured[0].legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "CB-PI",
        "CB-RA",
        "CB-IPW",
        "CB-DR",
        "Naturalness",
        "B-Learner",
        "RCT CATE 90% Wald CI",
        "Held-out RCT CATE",
    ]


def test_blackbox_comparison_plot_uses_exp15_model_implied_cate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Blackbox comparison should plot model-implied points, not RCT points."""

    comparison = pd.DataFrame(
        {
            "study": ["gainloss", "gainloss"],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 1],
            "resp_polideo_bin": [0, 1],
            "cb_pi_lower": [-0.5, -0.4],
            "cb_pi_upper": [0.4, 0.5],
            "cb_pi_interval_inverted": [False, False],
            "cb_ra_lower": [-0.6, -0.3],
            "cb_ra_upper": [0.3, 0.6],
            "cb_ra_interval_inverted": [False, False],
            "naturalness_lower": [-0.4, -0.2],
            "naturalness_upper": [0.2, 0.4],
            "blearner_lower": [-0.3, -0.1],
            "blearner_upper": [0.1, 0.3],
            "rct_cate_available": [True, True],
            "rct_cate": [-0.8, 0.8],
            "rct_cate_ci90_lower": [-0.9, 0.7],
            "rct_cate_ci90_upper": [-0.7, 0.9],
            "blackbox_reference_cate": [-0.1, 0.2],
        }
    )
    captured = []
    monkeypatch.setattr(exp18.plt, "close", captured.append)
    output_path = tmp_path / "comparison_vs_blackbox_cate.png"

    exp18._make_comparison_plot(
        comparison,
        Exp18Config(studies=("gainloss",)),
        output_path,
        truth="blackbox",
    )

    axis = captured[0].axes[0]
    model_truth = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "Model-implied blackbox CATE"
    )
    assert output_path.exists()
    assert np.allclose(model_truth.get_offsets()[:, 1], [-0.1, 0.2])
    assert "RCT CATE 90% Wald CI" not in axis.get_legend_handles_labels()[1]


