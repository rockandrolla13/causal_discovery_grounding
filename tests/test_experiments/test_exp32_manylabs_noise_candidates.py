"""Tests for the Exp32 noise-candidate ablation driver.

The end-to-end test fits Exp16 with the CPU hierarchical-Dirichlet Stage-1
backend on synthetic data. It checks wiring only; no number it produces is a
result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# The Exp32 driver imports the Exp25 driver, which reaches
# experiments.benchmarks and needs the Exp16-Exp25 cluster ML stack from
# requirements.txt. Skip, do not abort collection.
pytest.importorskip("torch")
pytest.importorskip("catenets")
pytest.importorskip("sklearn_quantile")

import experiments.exp32_manylabs_noise_candidates as exp32
from experiments.exp16_osrct_calibrated_policy import Exp16Config
from experiments.exp32_manylabs_noise_candidates import (
    Exp32Config,
    add_noise_candidates,
    exp25_target_index,
    load_exp25_store_config,
    noise_candidate_names,
    route_run_configs,
    run_labelled_config,
    summarize_results,
)

COVARIATES = list(Exp16Config().covariates)


def _synthetic_osrct(seed=0, n_sites=8, n_per_site=150):
    """Return OSRCT-style idle and randomized rows with a hidden confounder."""
    rng = np.random.default_rng(seed)
    frames = []
    participant = 0
    for site in [f"s{index:02d}" for index in range(n_sites - 1)] + ["pi"]:
        n_rows = n_per_site * (2 if site == "pi" else 1)
        hidden = rng.normal(size=n_rows)
        age = rng.binomial(1, 0.5, n_rows)
        treatment = rng.binomial(1, 0.5, n_rows)
        frames.append(pd.DataFrame({
            "site": site,
            "id": np.arange(participant, participant + n_rows, dtype=float),
            "iv": treatment,
            "dv_binary": rng.binomial(
                1, 1 / (1 + np.exp(-(-0.5 + 0.8 * treatment + 0.4 * age + 0.5 * hidden)))
            ),
            "resp_age_bin": age,
            "resp_gender_bin": rng.binomial(1, 0.5, n_rows),
            "resp_polideo_bin": rng.binomial(1, 0.4, n_rows),
            "hidden": hidden,
        }))
        participant += n_rows
    rct = pd.concat(frames, ignore_index=True)
    selection = 1 / (1 + np.exp(-(-0.2 + 1.2 * rct["resp_age_bin"] + 0.8 * rct["hidden"])))
    keep = rng.binomial(1, selection) == rct["iv"]
    rct = rct.drop(columns="hidden")
    return rct.loc[keep].reset_index(drop=True), rct


def _wiring_config():
    """Return a fast CPU Exp16 configuration for wiring checks."""
    return Exp16Config(
        study="s", studies=("s",), pattern="p", patterns=("p",), beta=1.0, betas=(1.0,),
        target_sites=("pi",),
        stage1_backend="hierarchical_dirichlet",
        validation_stage1_backend="hierarchical_dirichlet",
        target_trigger_probability_backend="hierarchical_dirichlet",
        lp_probability_backend="hierarchical_dirichlet",
        fit_gp=True, lp_posterior_samples=2, lp_posterior_n_jobs=1,
        candidate_selection_mode="all_triggered",
        policy_tuning_fraction=0.3, min_policy_tuning_sites=2,
        calibration_fraction=0.25, min_calibration_sites=2,
        conformal_alpha=0.5, rct_min_arm_count=5, min_cell_count=2,
        skip_lp_intervention_diagnostics=True, make_plots=False, verbose=False,
    )


def _write_store(root, *, betas=("1.5", "2.0"), n_replications=2):
    """Write a minimal Exp25 backend-leaf manifest and replication tree."""
    (root / "replications").mkdir(parents=True)
    manifest = {"config": {
        "n_replications": str(n_replications),
        "base_seed": "202608",
        "studies": ["allowedforbidden", "gainloss", "reciprocity"],
        "bias_covariates": ["age", "gender", "polideo"],
        "betas": list(betas),
        "target_site": "pi",
        "require_observed_gender": "True",
        "stage1_backend": "tabicl_v2",
        "target_trigger_probability_backend": "tabicl_v2",
        "tabicl_device": "cuda",
        "lp_probability_backend": "tabicl_v2",
        "lp_posterior_samples": "1000",
        "lp_posterior_n_jobs": "16",
    }}
    (root / "run_manifest.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return root


def test_noise_candidate_names_are_zero_padded():
    """Names sort in candidate order."""
    assert noise_candidate_names(3) == ("noise_01", "noise_02", "noise_03")
    assert noise_candidate_names(100)[0] == "noise_001"


def test_noise_is_keyed_on_participants_and_balanced():
    """A participant keeps its noise across frames, row order and reruns."""
    rows = 4000
    frame = pd.DataFrame({
        "site": np.repeat(["a", "b"], rows // 2),
        "id": np.tile(np.arange(rows // 2, dtype=float), 2),
        "x": np.arange(rows),
    })
    names = noise_candidate_names(3)
    first, shuffled = add_noise_candidates(
        [frame, frame.sample(frac=1.0, random_state=1)], names, prevalence=0.3, seed=11
    )
    again = add_noise_candidates([frame], names, prevalence=0.3, seed=11)[0]
    other = add_noise_candidates([frame], names, prevalence=0.3, seed=12)[0]

    realigned = shuffled.set_index("x").loc[first["x"], list(names)].reset_index(drop=True)
    pd.testing.assert_frame_equal(first[list(names)], realigned)
    pd.testing.assert_frame_equal(first[list(names)], again[list(names)])
    assert not first[list(names)].equals(other[list(names)])
    assert np.allclose(first[list(names)].mean(), 0.3, atol=0.03)
    assert abs(np.corrcoef(first["noise_01"], first["noise_02"])[0, 1]) < 0.05
    with pytest.raises(ValueError, match="already exist"):
        add_noise_candidates([first], names, prevalence=0.3, seed=11)
    with pytest.raises(ValueError, match="uniquely"):
        add_noise_candidates([pd.concat([frame, frame])], names, prevalence=0.3, seed=11)


def test_store_manifest_rebuilds_the_exp25_grid(tmp_path):
    """The store's recorded grid, seeds and backends are recovered as typed values."""
    store = _write_store(tmp_path / "store")
    exp25 = load_exp25_store_config(store)

    assert exp25.base_seed == 202608
    assert exp25.betas == (1.5, 2.0)
    assert exp25.require_observed_gender is True
    assert exp25.lp_posterior_samples == 1000
    with pytest.raises(FileNotFoundError):
        load_exp25_store_config(tmp_path)


def test_target_index_matches_exp16_grid_enumeration(tmp_path):
    """Pairing uses Exp16's study-pattern-beta enumeration over the full store grid."""
    exp25 = load_exp25_store_config(_write_store(tmp_path / "store"))

    assert exp25_target_index(exp25, study="allowedforbidden", pattern="age", beta=2.0) == 10_000
    assert exp25_target_index(exp25, study="allowedforbidden", pattern="gender", beta=2.0) == 30_000
    assert exp25_target_index(exp25, study="allowedforbidden", pattern="polideo", beta=2.0) == 50_000
    assert exp25_target_index(exp25, study="gainloss", pattern="age", beta=1.5) == 60_000
    with pytest.raises(ValueError, match="not in the Exp25 store"):
        exp25_target_index(exp25, study="allowedforbidden", pattern="age", beta=0.5)


def test_routes_define_separate_joint_and_baseline_runs():
    """Route a tests one noise column per run; route b tests all jointly."""
    base = Exp16Config()
    names = noise_candidate_names(3)

    separate = route_run_configs("separate", base, names)
    joint = route_run_configs("joint", base, names)
    baseline = route_run_configs("baseline", base, names)

    assert [label for label, _ in separate] == list(names)
    assert all(config.naturalness_extra_candidates == (label,) for label, config in separate)
    assert all(config.naturalness_candidate_covariates == (label,) for label, config in separate)
    assert joint[0][1].naturalness_extra_candidates == names
    assert joint[0][1].naturalness_candidate_covariates is None
    assert all(config.fallback_when_gp_unfitted for _, config in separate + joint)
    assert baseline == [("baseline", base)]


def test_route_root_refuses_a_different_configuration(tmp_path):
    """A result root is never reused for another design."""
    exp32._claim_route_root(tmp_path / "separate", {"n_noise": 15})
    exp32._claim_route_root(tmp_path / "separate", {"n_noise": 15})
    with pytest.raises(ValueError, match="fresh output root"):
        exp32._claim_route_root(tmp_path / "separate", {"n_noise": 5})


def test_run_experiment_pairs_runs_and_resumes(tmp_path, monkeypatch):
    """Orchestration pairs seeds and target indices, persists runs and resumes."""
    store = _write_store(tmp_path / "store")
    osrct, rct = _synthetic_osrct(n_sites=3, n_per_site=20)
    calls = []

    def fake_run_one_target(**kwargs):
        """Record the paired inputs and return one minimal output table."""
        calls.append(kwargs)
        return {"run_summary": pd.DataFrame({"seed": [kwargs["config"].seed]})}

    monkeypatch.setattr(exp32, "load_prepared_data", lambda _config: (osrct, rct, {"source": "test"}))
    monkeypatch.setattr(exp32, "run_one_target", fake_run_one_target)
    config = Exp32Config(
        exp25_store=str(store),
        output_dir=str(tmp_path / "out"),
        patterns=("gender",),
        replications=(1,),
        route="separate",
        n_noise=2,
        summarize=False,
        verbose=False,
    )

    result = exp32.run_experiment(config)
    exp32.run_experiment(config)

    assert len(calls) == 2
    assert {call["target_index"] for call in calls} == {30_000}
    assert {call["config"].seed for call in calls} == {202609}
    assert {call["config"].beta for call in calls} == {2.0}
    assert {call["config"].study for call in calls} == {"allowedforbidden"}
    assert all(call["config"].tabicl_share_checkpoint for call in calls)
    assert set(calls[0]["osrct_data"].columns) >= {"noise_01", "noise_02"}
    assert len(result["records"]) == 2
    markers = sorted((tmp_path / "out" / "separate").glob("replication=001/gender/runs/*/complete.json"))
    assert [marker.parent.name for marker in markers] == ["noise_01", "noise_02"]


def test_tau_floor_trims_the_grid_and_pairs_with_the_floored_baseline(tmp_path, monkeypatch):
    """A floored design runs on the trimmed tau grid, owns its root and pairs with its own baseline."""
    store = _write_store(tmp_path / "store")
    osrct, rct = _synthetic_osrct(n_sites=3, n_per_site=20)
    calls = []

    def fake_run_one_target(**kwargs):
        """Record the run and return a one-stratum summary whose lower end tells the routes apart."""
        calls.append(kwargs)
        lower = -0.4 if kwargs["config"].naturalness_extra_candidates else -0.5
        return {"target_gp_posterior_summary": pd.DataFrame({
            **{column: [0] for column in COVARIATES}, "lower_mean": [lower], "upper_mean": [0.5],
            "lp_posterior_prob_optimal": [1.0], "rct_cate": [0.1], "rct_cate_available": [True],
            "n_target_rows": [10],
        })}

    monkeypatch.setattr(exp32, "load_prepared_data", lambda _config: (osrct, rct, {"source": "test"}))
    monkeypatch.setattr(exp32, "run_one_target", fake_run_one_target)
    common = dict(exp25_store=str(store), output_dir=str(tmp_path / "out"), patterns=("gender",),
                  replications=(1,), n_noise=1, summarize=False, verbose=False, tau_floor=0.075)
    exp32.run_experiment(Exp32Config(route="separate", **common))
    with pytest.raises(ValueError, match="same floor"):
        summarize_results(tmp_path / "out", store, study="allowedforbidden", beta=2.0, tau_floor=0.075)
    exp32.run_experiment(Exp32Config(route="baseline", **common))

    tables = summarize_results(tmp_path / "out", store, study="allowedforbidden", beta=2.0, tau_floor=0.075)

    assert {call["config"].taus for call in calls} == {(0.075, 0.1, 0.15, 0.2, 0.3)}
    bounds = tables["summary_bounds"].set_index("route")
    assert bounds.loc["baseline", "exp25_lower"] == -0.5 and bounds.loc["separate", "exp25_lower"] == -0.5
    assert np.isclose(bounds.loc["separate", "width_minus_exp25"], -0.1)
    with pytest.raises(ValueError, match="fresh output root"):
        exp32.run_experiment(Exp32Config(route="separate", **{**common, "tau_floor": None}))
    with pytest.raises(ValueError, match="No tau value"):
        exp32.run_experiment(Exp32Config(route="joint", **{**common, "tau_floor": 0.5}))
    with pytest.raises(ValueError, match="tau_floor"):
        Exp32Config(exp25_store="store", output_dir="out", tau_floor=-0.1)


def test_analysis_reports_zero_noise_fires_and_pairs_the_joint_route(tmp_path, monkeypatch):
    """The analysis counts a silent noise candidate as zero fires and pairs tolerances and widths."""
    store = _write_store(tmp_path / "store")
    osrct, rct = _synthetic_osrct(n_sites=3, n_per_site=20)

    def fake_run_one_target(**kwargs):
        """Return minimal discovery and bound tables; the joint run adds a noise column that never fires."""
        joint = bool(kwargs["config"].naturalness_extra_candidates)
        covariates = ["resp_age_bin", "noise_01"] if joint else ["resp_age_bin"]
        fired = [True, False][:len(covariates)]
        return {
            "final_source_naturalness_observations": pd.DataFrame({
                "covariate": covariates, "tau": 0.075, "trigger_fired": fired,
                "d_strength": 0.2, "i_deviation": 0.01}),
            "target_trigger_statistics": pd.DataFrame({
                "covariate": covariates, "target_triggered_any": fired, "d_strength": 0.2, "i_deviation": 0.01}),
            "target_policy_choices": pd.DataFrame({
                "covariate": ["resp_age_bin"], "epsilon": [0.05], "tau": [0.075],
                "two_c": [0.31 if joint else 0.30], "two_d": [0.05]}),
            "target_gp_posterior_summary": pd.DataFrame({
                **{column: [0] for column in COVARIATES}, "lower_mean": [-0.4 if joint else -0.5],
                "upper_mean": [0.5], "lp_posterior_prob_optimal": [1.0], "rct_cate": [0.1],
                "rct_cate_available": [True], "n_target_rows": [10]}),
        }

    monkeypatch.setattr(exp32, "load_prepared_data", lambda _config: (osrct, rct, {"source": "test"}))
    monkeypatch.setattr(exp32, "run_one_target", fake_run_one_target)
    common = dict(exp25_store=str(store), output_dir=str(tmp_path / "out"), patterns=("gender",),
                  replications=(1,), n_noise=1, summarize=False, verbose=False, tau_floor=0.075)
    for route in ("baseline", "joint"):
        exp32.run_experiment(Exp32Config(route=route, **common))

    tables = exp32.analyze_results(tmp_path / "out", store, study="allowedforbidden", beta=2.0,
                                   destination=tmp_path / "analysis")

    target = tables["target_trigger_rates"].set_index(["route", "candidate_kind"])
    assert target.loc[("joint", "noise"), "n_fired"] == 0 and target.loc[("joint", "real"), "n_fired"] == 1
    assert target.loc[("joint", "noise"), "fire_rate_upper95"] > 0
    lp = tables["lp_composition"].set_index(["route", "candidate_kind"])
    assert lp.loc[("joint", "noise"), "n_lp_rows"] == 0
    shift = tables["real_tolerance_shift"].iloc[0]
    assert shift["n_matched_rows"] == 1 and np.isclose(shift["mean_two_c_change"], 0.01)
    overall = tables["joint_bounds_change"].set_index("pattern").loc["all"]
    assert np.isclose(overall["mean_width_change"], -0.1) and overall["n_both_valid"] == 1
    assert set(tables["target_trigger_rates"]["design"]) == {"tau>=0.075"}
    assert "tau>=0.075" in (tmp_path / "analysis" / "exp32_analysis_report.md").read_text()
    assert not (tmp_path / "out" / "summary_bounds.csv").exists()

    grid_parent = tmp_path / "grid"
    second = grid_parent / "allowedforbidden_beta2.0"
    second.mkdir(parents=True)
    for route in ("baseline", "joint"):
        (tmp_path / "out" / route).rename(second / route)
    # A cell whose joint route has not run yet must not break the grid, nor enter its totals.
    cell = {**common, "output_dir": str(grid_parent / "allowedforbidden_beta1.5"), "beta": 1.5}
    exp32.run_experiment(Exp32Config(route="baseline", **cell))
    partial = exp32.analyze_grid(grid_parent, tmp_path / "partial_analysis")
    bounds = partial["joint_bounds_change"].query("pattern == 'all'").set_index("beta")
    assert bounds.loc[1.5, "n_both_valid"] == 0 and bounds.loc[2.0, "n_both_valid"] == 1
    assert partial["real_tolerance_shift"].set_index("beta").loc[1.5, "n_matched_rows"] == 0
    assert "| all | all | 1 |" in (tmp_path / "partial_analysis" / "exp32_grid_analysis_report.md").read_text()
    exp32.run_experiment(Exp32Config(route="joint", **cell))

    grid = exp32.analyze_grid(grid_parent, tmp_path / "grid_analysis")

    assert sorted(grid["joint_bounds_change"].query("pattern == 'all'")["beta"]) == [1.5, 2.0]
    noise = grid["target_trigger_rates"].query("candidate_kind == 'noise'")
    assert noise["n_fired"].sum() == 0 and len(noise) == 2
    report = (tmp_path / "grid_analysis" / "exp32_grid_analysis_report.md").read_text()
    assert "| all | all |" in report and "| allowedforbidden | 1.5 |" in report
    unfloored = grid_parent / "unfloored"
    for route in ("baseline", "joint"):
        exp32.run_experiment(Exp32Config(route=route, **{**common, "output_dir": str(unfloored), "tau_floor": None}))
    with pytest.raises(ValueError, match="no tau floor"):
        exp32.analyze_grid(grid_parent, tmp_path / "grid_analysis")


def test_joint_run_wiring_end_to_end(tmp_path):
    """A joint run persists discovery tables that the summary pairs with Exp25."""
    osrct, rct = _synthetic_osrct()
    names = noise_candidate_names(2)
    osrct, rct = add_noise_candidates([osrct, rct], names, prevalence=0.5, seed=3)
    label, config = route_run_configs("joint", _wiring_config(), names)[0]
    run_dir = tmp_path / "out" / "joint" / "replication=000" / "age" / "runs" / label

    record = run_labelled_config(
        run_label=label,
        run_config=config,
        osrct_data=osrct,
        rct_data=rct,
        base_metadata={"source": "synthetic"},
        target_index=0,
        run_dir=run_dir,
        extra={"route": "joint", "replication": 0, "seed": 0, "study": "s",
               "pattern": "age", "beta": 2.0},
    )

    assert (run_dir / "complete.json").exists()
    assert record["naturalness_extra_candidates"] == list(names)
    source = pd.read_parquet(run_dir / "source_observations.parquet")
    assert set(source["covariate"]) == {*COVARIATES, *names}
    statistics = pd.read_parquet(run_dir / "target_trigger_statistics.parquet")
    assert set(statistics["covariate"]) == {*COVARIATES, *names}
    posterior = pd.read_parquet(run_dir / "target_posterior_summary.parquet")
    assert len(posterior.drop_duplicates(COVARIATES)) == len(posterior)

    baseline_dir = tmp_path.joinpath(
        "store", "replications", "replication=000", "naturalness", "all_triggered",
        "pi", "age", "2.0",
    )
    baseline_dir.mkdir(parents=True)
    baseline = posterior[COVARIATES].copy()
    baseline["study"] = "s"
    baseline["lower_mean"] = -0.5
    baseline["upper_mean"] = 0.5
    baseline["lp_posterior_prob_optimal"] = 1.0
    baseline.to_parquet(baseline_dir / "target_gp_posterior_summary.parquet", index=False)

    tables = summarize_results(tmp_path / "out", tmp_path / "store", study="s", beta=2.0)

    assert set(tables["summary_target_triggers"]["candidate_kind"]) == {"noise", "real"}
    assert set(tables["summary_source_triggers"]["candidate_kind"]) == {"noise", "real"}
    bounds = tables["summary_bounds"]
    assert bounds["exp25_width"].eq(1.0).all()
    assert (tmp_path / "out" / "summary_bounds.csv").exists()
