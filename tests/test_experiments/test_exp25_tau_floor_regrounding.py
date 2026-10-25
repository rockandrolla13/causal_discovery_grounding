"""Tests for the Exp25 tau-floor regrounding script.

The replay tests fit Exp16 with the CPU hierarchical-Dirichlet backend on
synthetic data. They check that regrounding reproduces Exp16's own grounding;
no number they produce is a result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# The regrounding module imports the Exp25 driver, which reaches
# experiments.benchmarks and needs the Exp16-Exp25 cluster ML stack from
# requirements.txt. Skip, do not abort collection.
pytest.importorskip("torch")
pytest.importorskip("catenets")
pytest.importorskip("sklearn_quantile")

import experiments.osrct_target_pipeline as target_pipeline
import experiments.exp25_tau_floor_regrounding as reg
from causal_grounding.paper_grids import floor_taus
from causal_grounding.train_target_split import create_train_target_split
from experiments.osrct_target_pipeline import Exp16Config


def _synthetic_osrct(seed=0, n_sites=8, n_per_site=150):
    """Return OSRCT-style idle and randomized rows with a hidden confounder."""
    rng = np.random.default_rng(seed)
    frames, participant = [], 0
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
    """Return a fast CPU Exp16 configuration with GP posterior draws."""
    return Exp16Config(
        study="s", studies=("s",), pattern="p", patterns=("p",), beta=1.0, betas=(1.0,),
        target_sites=("pi",),
        stage1_backend="hierarchical_dirichlet",
        validation_stage1_backend="hierarchical_dirichlet",
        target_trigger_probability_backend="hierarchical_dirichlet",
        lp_probability_backend="hierarchical_dirichlet",
        fit_gp=True, lp_posterior_samples=20, lp_posterior_n_jobs=1,
        candidate_selection_mode="all_triggered",
        policy_tuning_fraction=0.3, min_policy_tuning_sites=2,
        calibration_fraction=0.25, min_calibration_sites=2,
        conformal_alpha=0.5, rct_min_arm_count=5, min_cell_count=2,
        skip_lp_intervention_diagnostics=True, make_plots=False, verbose=False,
    )


@pytest.fixture(scope="module")
def exp16_run(tmp_path_factory):
    """Fit Exp16 once and persist its Stage-2 observations as Exp25 does."""
    osrct, rct = _synthetic_osrct()
    config = _wiring_config()
    outputs = target_pipeline.run_one_target(
        osrct_data=osrct, rct_data=rct, target_site="pi", target_index=7,
        config=config, base_metadata={},
    )
    in_memory = outputs["final_source_naturalness_observations"][list(reg.STAGE2_COLUMNS)].copy()
    in_memory["trigger_fired"] = in_memory["trigger_fired"].astype(bool)
    path = tmp_path_factory.mktemp("exp25") / "final_source_naturalness_observations.csv"
    outputs["final_source_naturalness_observations"].to_csv(path, index=False)
    stored = pd.read_csv(path)[list(reg.STAGE2_COLUMNS)]
    _, target_idle = create_train_target_split(
        osrct_data=osrct, rct_data=rct, target_site="pi", site_col="site", regime_col="F"
    )
    target_idle = target_idle[[*config.covariates, *reg.TARGET_IDLE_COLUMNS]]
    tables = reg.compute_target_tables(target_idle, config)
    return config, outputs, {"in_memory": in_memory, "stored": stored}, target_idle, tables


def _replay_against_exp16(exp16_run, which, atol):
    """Reground on Exp16's own grid and compare with Exp16's interval."""
    config, outputs, observations, target_idle, tables = exp16_run
    replay = reg.reground_condition(
        config=config, target_index=7, observations=observations[which],
        target_idle=target_idle, tables=tables, floor=0.01,
    )
    columns = [*config.covariates, "lower_mean", "upper_mean", "lp_posterior_prob_optimal"]
    merged = outputs["target_gp_posterior_summary"][columns].merge(
        replay["posterior_summary"][columns], on=list(config.covariates), suffixes=("_exp16", "_replay"),
    )
    assert len(merged) == len(outputs["target_gp_posterior_summary"])
    for column in ("lower_mean", "upper_mean", "lp_posterior_prob_optimal"):
        np.testing.assert_allclose(
            merged[f"{column}_replay"], merged[f"{column}_exp16"], atol=atol, equal_nan=True
        )
    return replay, outputs


def test_floor_at_the_smallest_tau_reproduces_exp16_exactly(exp16_run):
    """With Exp16's own observations the replay returns its interval exactly."""
    replay, outputs = _replay_against_exp16(exp16_run, "in_memory", atol=1e-12)
    expected = outputs["target_policy_choices"]
    assert len(replay["choices"]) == len(expected)
    np.testing.assert_allclose(
        np.sort(replay["choices"]["two_c"].to_numpy()), np.sort(expected["two_c"].to_numpy()), atol=1e-12
    )


def test_stored_observations_reproduce_exp16_within_csv_precision(exp16_run):
    """Observations persisted through CSV, as Exp25 stored them, keep Exp16's values to CSV precision.

    Dtypes and missing values survive the round trip, and values move by at most
    about 1e-13 relative (d_strength; the other columns by a few ulps). The replay
    of the stored observations is not compared with Exp16 endpoint by endpoint:
    a change that small can move the GP hyperparameter optimum between near-tied
    optima, and which one the optimizer reaches also depends on BLAS threading,
    so replayed endpoints moved by 2e-5 to 7e-3 across environments. The replay
    itself is checked exactly on in-memory observations above.
    """
    _, _, observations, _, _ = exp16_run
    pd.testing.assert_frame_equal(
        observations["stored"].reset_index(drop=True),
        observations["in_memory"].reset_index(drop=True),
        check_exact=False, rtol=1e-12, atol=0.0,
    )


def test_raised_floor_drops_small_tau_from_gp_and_target(exp16_run):
    """A floor removes small-tau rows from the GP data and from the target choices."""
    config, _, observations, target_idle, tables = exp16_run
    observations = observations["stored"]
    replay = reg.reground_condition(
        config=config, target_index=7, observations=observations,
        target_idle=target_idle, tables=tables, floor=0.075,
    )
    choices = replay["choices"]
    if "tau" in choices:
        assert choices["tau"].dropna().ge(0.075 - 1e-12).all()
    fired = observations["trigger_fired"] & observations["tau"].ge(0.075 - 1e-12) & observations["c_hat"].notna()
    c_surface = replay["gp"].loc[replay["gp"]["surface"].eq("c")]
    assert int(c_surface["n_training_observations"].iloc[0]) == int(fired.sum())


def test_floor_taus_and_shards():
    """Floors keep the grid's upper values; shards partition the conditions."""
    assert floor_taus((0.01, 0.05, 0.075, 0.1), 0.05) == (0.05, 0.075, 0.1)
    assert floor_taus((0.01, 0.05, 0.075, 0.1), 0.075) == (0.075, 0.1)
    with pytest.raises(ValueError):
        floor_taus((0.01, 0.05), 0.5)
    items = list(range(10))
    shards = [reg._shard(items, index, 3) for index in range(3)]
    assert sorted(sum(shards, [])) == items


def _write_store(root, betas):
    """Write a minimal Exp25 backend-leaf manifest."""
    (root / "replications").mkdir(parents=True)
    manifest = {"config": {
        "n_replications": "2", "base_seed": "202608",
        "studies": ["allowedforbidden", "gainloss"], "bias_covariates": ["age"],
        "betas": list(betas), "target_site": "pi", "require_observed_gender": "True",
        "stage1_backend": "tabicl_v2", "target_trigger_probability_backend": "tabicl_v2",
        "tabicl_device": "cuda", "lp_probability_backend": "tabicl_v2",
        "lp_posterior_samples": "1000", "lp_posterior_n_jobs": "16",
    }}
    (root / "run_manifest.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return root


def test_conditions_span_both_stores_and_keep_paper_betas(tmp_path):
    """Conditions come from every store, filtered to the requested betas."""
    low = _write_store(tmp_path / "low", ("0.5", "1.0"))
    high = _write_store(tmp_path / "high", ("1.5", "2.0"))
    conditions = reg.enumerate_conditions([low, high], betas=(1.0, 2.0))

    assert {c.beta for c in conditions} == {1.0, 2.0}
    assert len(conditions) == 2 * 2 * 1 * 2
    assert conditions[0].label == "replication=000/allowedforbidden/age/beta1.0"
    with pytest.raises(ValueError, match="both"):
        reg.enumerate_conditions([low, _write_store(tmp_path / "dup", ("1.0",))], betas=(1.0,))


def test_missing_table_waits_then_fails(tmp_path):
    """A reground shard waits for its table and fails clearly if none appears."""
    missing = tmp_path / "target_tables.pkl"
    start = __import__("time").time()
    with pytest.raises(FileNotFoundError, match="tables pass"):
        reg._wait_for(missing, wait_seconds=0.2, poll_seconds=0.05)
    assert __import__("time").time() - start >= 0.2
    missing.write_bytes(b"x")
    reg._wait_for(missing, wait_seconds=0.0)


def _merged_leaf(root):
    """Write a one-condition merged Exp25 leaf with Naturalness and B-Learner rows."""
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    strata = list(exp25.EXP25_STRATUM_COLUMNS)
    comparison = pd.DataFrame({
        "study": ["allowedforbidden"] * 2, strata[0]: [0, 1], strata[1]: [0, 0], strata[2]: [1, 1],
        "target_site": ["pi"] * 2, "n_target_idle_rows": [40, 60], "rct_cate": [0.1, -0.2],
        "rct_cate_ci90_lower": [0.0, -0.3], "rct_cate_ci90_upper": [0.2, -0.1], "rct_cate_available": [True, True],
        "naturalness_lower": [-0.3, -0.5], "naturalness_upper": [0.3, 0.1], "lp_posterior_prob_optimal": [1.0, 0.8],
        "blearner_lower": [-0.6, -0.7], "blearner_upper": [0.6, 0.4], "blearner_gamma": [2.0, 2.0],
        "outside_binary_cate_support": [False, False],
    })
    intervals = exp25.comparison_to_long_intervals(
        comparison, replication=0, seed=202608, pattern="age", beta=2.0,
        config_fingerprint_value="fp", methods=("naturalness", "blearner"))
    leaf = root / "tabicl_v2"
    gamma = leaf / "baseline_variants" / "gamma_dgp_oracle"
    gamma.mkdir(parents=True)
    intervals.to_parquet(gamma / "mc_intervals.parquet", index=False)
    exp25.build_run_diagnostics(intervals, runtime_seconds=12.0).to_parquet(
        gamma / "mc_run_diagnostics.parquet", index=False)
    pd.DataFrame({"gamma": [1.5]}).to_parquet(gamma / "dgp_gamma_calibration.parquet", index=False)
    ident = leaf / "baseline_variants" / "identification_bounds"
    ident.mkdir(parents=True)
    intervals.loc[intervals.method.eq("blearner")].assign(method="manski").to_parquet(
        ident / "mc_intervals.parquet", index=False)
    (leaf / "merge_manifest.json").write_text("{}", encoding="utf-8")
    return leaf, intervals


def test_assemble_replaces_only_naturalness_rows(tmp_path):
    """The assembled tree keeps competitors and rebuilds Naturalness with Exp25's rules."""
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    leaf, original = _merged_leaf(tmp_path / "merged")
    strata = list(exp25.EXP25_STRATUM_COLUMNS)
    summary_dir = tmp_path / "regrounding" / "floor=0.05" / "replication=000" / "allowedforbidden" / "age" / "beta2.0"
    summary_dir.mkdir(parents=True)
    pd.DataFrame({strata[0]: [0, 1], strata[1]: [0, 0], strata[2]: [1, 1],
                  "lower_mean": [-0.35, np.nan], "upper_mean": [0.32, np.nan],
                  "lp_posterior_prob_optimal": [1.0, 0.0]}).to_parquet(summary_dir / "posterior_summary.parquet", index=False)
    destination = tmp_path / "assembled" / "tabicl_v2"

    record = reg.assemble_tree(merged_leaf=leaf, regrounding_root=tmp_path / "regrounding",
                               floor=0.05, destination_leaf=destination)

    gamma = destination / "baseline_variants" / "gamma_dgp_oracle"
    new = pd.read_parquet(gamma / "mc_intervals.parquet")
    assert list(new.columns) == list(original.columns)
    pd.testing.assert_frame_equal(
        new.loc[new.method.eq("blearner")].reset_index(drop=True),
        original.loc[original.method.eq("blearner")].reset_index(drop=True))
    nat = new.loc[new.method.eq("naturalness")].reset_index(drop=True)
    assert nat.raw_lower.iloc[0] == -0.35 and nat.upper.iloc[0] == 0.32 and bool(nat.is_valid.iloc[0])
    assert not bool(nat.is_valid.iloc[1]) and nat.exclusion_reason.iloc[1] == "no_optimal_lp_draw"
    assert nat.rct_cate.tolist() == [0.1, -0.2]
    diagnostics = pd.read_parquet(gamma / "mc_run_diagnostics.parquet")
    nat_diag = diagnostics.loc[diagnostics.method.eq("naturalness")].iloc[0]
    assert nat_diag.n_valid_strata == 1 and nat_diag.condition_runtime_seconds == 12.0
    assert np.isclose(nat_diag.mean_valid_width, 0.67)
    pd.testing.assert_frame_equal(
        pd.read_parquet(destination / "baseline_variants" / "identification_bounds" / "mc_intervals.parquet"),
        pd.read_parquet(leaf / "baseline_variants" / "identification_bounds" / "mc_intervals.parquet"))
    assert record["rows_replaced"] == {"gamma_dgp_oracle/mc_intervals.parquet": 2,
                                       "gamma_dgp_oracle/mc_run_diagnostics.parquet": 1}
    assert (destination / "regrounding_manifest.json").exists()
    with pytest.raises(FileExistsError):
        reg.assemble_tree(merged_leaf=leaf, regrounding_root=tmp_path / "regrounding",
                          floor=0.05, destination_leaf=destination)


def test_bounds_pass_puts_balke_pearl_tau_at_the_floor(tmp_path, monkeypatch):
    """Each task replays a store with tau at the floor's smallest grid value and its published support floor."""
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    root = tmp_path / "20260917"
    store = _write_store(root / "tabicl_v2" / "tabicl_v2" / "tabicl_v2", ("0.5", "1.0"))
    published = store / "baseline_variants" / "identification_bounds"
    published.mkdir(parents=True)
    (published / "run_manifest.yml").write_text(
        yaml.safe_dump({"extra": {"identification_config": {"min_count": "3", "tau": "0.01"}}}), encoding="utf-8")
    calls = []

    def fake_replay(config, output_dir=None):
        calls.append((config, output_dir))
        output_dir.mkdir(parents=True)
        pd.DataFrame({"n": [1]}).to_parquet(output_dir / exp25.EXP25_INTERVALS_FILENAME)

    monkeypatch.setattr(exp25, "run_identification_bounds_replay", fake_replay)
    # The repeated task finds its leaf finished and is skipped.
    reg.run_bounds([(store, 0.01), (store, 0.06), (store, 0.06)], tmp_path / "bounds")

    assert [round(config.identification_tau, 6) for config, _ in calls] == [0.01, 0.075]
    assert [path for _, path in calls] == [tmp_path / "bounds" / "floor=0.01" / "20260917",
                                           tmp_path / "bounds" / "floor=0.06" / "20260917"]
    config = calls[0][0]
    assert config.identification_bounds_only and config.identification_min_count == 3
    assert config.output_dir == str(root) and config.betas == (0.5, 1.0)


def test_assemble_swaps_in_recomputed_identification_bounds(tmp_path):
    """A bounds root replaces the identification rows in the merged order; a mismatch writes nothing."""
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    leaf, _ = _merged_leaf(tmp_path / "merged")
    strata = list(exp25.EXP25_STRATUM_COLUMNS)
    summary_dir = tmp_path / "regrounding" / "floor=0.05" / "replication=000" / "allowedforbidden" / "age" / "beta2.0"
    summary_dir.mkdir(parents=True)
    pd.DataFrame({strata[0]: [0, 1], strata[1]: [0, 0], strata[2]: [1, 1],
                  "lower_mean": [-0.35, -0.4], "upper_mean": [0.32, 0.2],
                  "lp_posterior_prob_optimal": [1.0, 1.0]}).to_parquet(summary_dir / "posterior_summary.parquet", index=False)
    merged = pd.read_parquet(leaf / "baseline_variants" / "identification_bounds" / "mc_intervals.parquet")
    recomputed = merged.assign(lower=merged["lower"] - 0.1)
    store_leaf = tmp_path / "bounds" / "floor=0.05" / "20260923_beta_1p5_2"
    store_leaf.mkdir(parents=True)
    recomputed.iloc[::-1].to_parquet(store_leaf / "mc_intervals.parquet", index=False)

    record = reg.assemble_tree(merged_leaf=leaf, regrounding_root=tmp_path / "regrounding", floor=0.05,
                               destination_leaf=tmp_path / "assembled" / "tabicl_v2", bounds_root=tmp_path / "bounds")

    new = pd.read_parquet(tmp_path / "assembled" / "tabicl_v2" / "baseline_variants"
                          / "identification_bounds" / "mc_intervals.parquet")
    pd.testing.assert_frame_equal(new, recomputed)
    assert record["rows_replaced"]["identification_bounds/mc_intervals.parquet"] == len(merged)
    assert record["bounds_root"] == str((tmp_path / "bounds").resolve())
    recomputed.iloc[1:].to_parquet(store_leaf / "mc_intervals.parquet", index=False)
    with pytest.raises(ValueError, match="does not match"):
        reg.assemble_tree(merged_leaf=leaf, regrounding_root=tmp_path / "regrounding", floor=0.05,
                          destination_leaf=tmp_path / "again" / "tabicl_v2", bounds_root=tmp_path / "bounds")
    assert not (tmp_path / "again").exists()


def test_admissible_choices_keep_the_pareto_frontier_per_candidate():
    """Each candidate keeps its triggered pairs that no smaller-epsilon, larger-tau pair dominates."""
    rows = []
    for covariate, smallest_epsilon, largest_tau in [("a", 0.05, 0.2), ("b", 0.10, 0.1)]:
        for epsilon in (0.05, 0.10, 0.20):
            for tau in (0.075, 0.1, 0.2):
                if epsilon >= smallest_epsilon and tau <= largest_tau:
                    rows.append({"covariate": covariate, "epsilon": epsilon, "tau": tau})
    rows += [{"covariate": "c", "epsilon": 0.05, "tau": 0.075}, {"covariate": "c", "epsilon": 0.10, "tau": 0.2}]
    choices = target_pipeline.finalize_policy_choices(pd.DataFrame(rows), top_k=1, candidate_selection_mode="all_triggered")

    kept = reg.admissible_choices(choices)

    assert list(zip(kept.covariate, kept.epsilon, kept.tau)) == [
        ("a", 0.05, 0.2), ("b", 0.10, 0.1), ("c", 0.05, 0.075), ("c", 0.10, 0.2),
    ]
    assert kept["is_primary_choice"].tolist() == [True, False, False, False]
    assert kept["candidate_selection_mode"].eq("admissible_triggered").all()
    assert reg.admissible_choices(choices.iloc[0:0]).empty


def test_theory_grounding_runs_end_to_end(exp16_run):
    """Envelope GP, logit(2c) and admissible pairs together; checks wiring, not results."""
    config, _, observations, target_idle, tables = exp16_run
    observations = observations["stored"]
    default = reg.reground_condition(
        config=config, target_index=7, observations=observations,
        target_idle=target_idle, tables=tables, floor=0.01,
    )
    theory = reg.reground_condition(
        config=config, target_index=7, observations=observations,
        target_idle=target_idle, tables=tables, floor=0.01,
        gp_training_target="envelope", gp_target_scale="tilde", target_pairs="admissible",
    )

    gp = theory["gp"]
    assert gp["gp_training_target"].eq("envelope").all() and gp["gp_target_scale"].eq("tilde").all()
    fired = observations[observations.trigger_fired & observations.c_hat.notna()]
    n_grid_points = fired.groupby(["covariate_index", "epsilon", "tau"]).ngroups
    assert int(gp.loc[gp.surface.eq("c"), "n_training_observations"].iloc[0]) == n_grid_points
    choices = theory["choices"]
    assert not choices.empty
    assert choices.groupby("covariate").size().eq(1).all()
    assert len(theory["posterior_summary"]) == len(default["posterior_summary"])
    with pytest.raises(ValueError, match="target_pairs"):
        reg.reground_condition(
            config=config, target_index=7, observations=observations,
            target_idle=target_idle, tables=tables, floor=0.01, target_pairs="corner",
        )


def _trial(seed=0, n_per_stratum=40):
    """Return a two-stratum randomized trial with unique participant ids."""
    rng = np.random.default_rng(seed)
    rows = []
    for stratum, (p1, p0) in enumerate([(0.3, 0.5), (0.6, 0.4)]):
        for unit in range(n_per_stratum):
            treated = unit % 2
            rows.append({"id": float(stratum * 1000 + unit), "site": "pi", "z": stratum, "iv": treated,
                         "dv_binary": int(rng.random() < (p1 if treated else p0))})
    return pd.DataFrame(rows)


def _cate(frame):
    """Return the difference in means per stratum."""
    means = frame.groupby(["z", "iv"])["dv_binary"].mean().unstack("iv")
    return means[1] - means[0]


@pytest.mark.parametrize("delta", [0.2, -0.3])
def test_effect_shift_moves_every_stratum_cate_by_delta(delta):
    """Flips move each stratum CATE by delta, up to one participant per arm, reproducibly."""
    trial = _trial()
    flips, log = reg.effect_shift_flips(trial, delta, covariates=["z"], study="s")
    again, _ = reg.effect_shift_flips(trial, delta, covariates=["z"], study="s")
    pd.testing.assert_frame_equal(flips, again)
    shifted = reg.apply_effect_shift(trial, flips, target_site="pi")
    change = (_cate(shifted) - _cate(trial)).to_numpy()
    np.testing.assert_allclose(change, delta, atol=2 / 20 + 1e-12)
    np.testing.assert_allclose(log.set_index("z")["cate_shift"].to_numpy(), change, atol=1e-12)
    assert set(shifted.columns) == set(trial.columns) and len(shifted) == len(trial)


def test_effect_shift_caps_an_arm_without_enough_eligible_outcomes():
    """An arm that cannot move flips what it has, and the log reports the shortfall."""
    trial = _trial()
    trial.loc[trial.iv.eq(1) & trial.z.eq(0), "dv_binary"] = 1
    _, log = reg.effect_shift_flips(trial, 0.4, covariates=["z"], study="s")
    first = log.set_index("z").loc[0]
    assert first["flipped_arm1"] == 0 and first["wanted_arm1"] > 0
    assert np.isclose(first["cate_shift"], first["flipped_arm0"] / first["n_arm0"])
    with pytest.raises(ValueError, match="delta"):
        reg.effect_shift_flips(trial, 1.0, covariates=["z"], study="s")


def test_shift_target_data_carries_trial_flips_to_the_sample_by_participant():
    """The observational sample gets its participants' shifted outcomes; other sites never change."""
    trial = _trial()
    other = trial.assign(site="other")  # same ids, another site
    rct = pd.concat([trial, other], ignore_index=True)
    sample = rct.loc[rct.index % 3 != 0].reset_index(drop=True)
    osrct, shifted_rct, log = reg.shift_target_data(
        sample, rct, target_site="pi", delta=0.3, covariates=["z"], study="s",
    )
    at_pi = shifted_rct.site.eq("pi")
    outcome_by_id = shifted_rct.loc[at_pi].set_index("id")["dv_binary"]
    pi_rows = osrct.loc[osrct.site.eq("pi")]
    np.testing.assert_array_equal(pi_rows["dv_binary"].to_numpy(), pi_rows["id"].map(outcome_by_id).to_numpy())
    pd.testing.assert_frame_equal(osrct.loc[osrct.site.eq("other")], sample.loc[sample.site.eq("other")])
    pd.testing.assert_frame_equal(shifted_rct.loc[~at_pi], rct.loc[~at_pi])
    assert len(log) == 2


def test_mismatched_source_condition_maps_beta_and_covariate():
    """The D3 arm takes sources at the farthest paper beta, or selected on another covariate."""
    lookup = {(0, "s", pattern, beta): reg.Condition("store", 0, "s", pattern, beta)
              for pattern in ("age", "gender", "polideo") for beta in (0.5, 1.0, 1.5, 2.0)}
    target = lookup[(0, "s", "age", 1.0)]
    assert reg.mismatched_source_condition(target, "none", {}) is target
    assert reg.mismatched_source_condition(target, "beta", lookup).beta == 2.0
    assert reg.mismatched_source_condition(lookup[(0, "s", "age", 1.5)], "beta", lookup).beta == 0.5
    assert reg.mismatched_source_condition(target, "covariate", lookup).pattern == "polideo"
    assert reg.mismatched_source_condition(lookup[(0, "s", "gender", 2.0)], "covariate", lookup).pattern == "age"
    with pytest.raises(ValueError, match="mismatch"):
        reg.mismatched_source_condition(target, "site", lookup)


def test_transport_intervals_pool_the_named_sources_only():
    """Sites sharing one effect give a small between-site variance and an interval around it."""
    rng = np.random.default_rng(3)
    rows = []
    for site in [f"s{index}" for index in range(12)] + ["pi"]:
        for unit in range(400):
            treated = unit % 2
            effect = 0.6 if site == "pi" else 0.2  # the target is excluded, so its effect never enters
            rows.append({"site": site, "z": 0, "iv": treated,
                         "dv_binary": int(rng.random() < 0.3 + effect * treated)})
    rows.append({"site": "s0", "z": 1, "iv": 1, "dv_binary": 1})  # a stratum no site supports
    trials = pd.DataFrame(rows)
    sources = [f"s{index}" for index in range(12)]
    out = reg.transport_intervals(trials, source_sites=sources, covariates=["z"]).set_index("z")
    assert out.loc[0, "k_sites"] == 12
    assert out.loc[0, "lower"] < 0.2 < out.loc[0, "upper"] and out.loc[0, "upper"] < 0.6
    assert out.loc[0, "tau2"] < 0.01
    assert out.loc[1, "k_sites"] == 0 and np.isnan(out.loc[1, "lower"])


def _write(frame, path):
    """Write a parquet file, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_summarize_arms_scores_every_method_with_manski_fallback(tmp_path):
    """Synthetic leaves: Manski fills a stratum without an optimal draw, transport is scored at shift 0."""
    strata = pd.DataFrame({"resp_age_bin": [0, 1], "resp_gender_bin": [0, 0], "resp_polideo_bin": [0, 0]})
    store = tmp_path / "20260917" / "tabicl_v2" / "tabicl_v2" / "tabicl_v2"
    condition = reg.Condition(str(store), 0, "s", "age", 1.0)
    leaf = Path("floor=0.075") / condition.label
    reference = strata.assign(rct_cate=[0.1, -0.2], rct_cate_ci90_lower=[0.02, -0.3], rct_cate_ci90_upper=[0.2, -0.1])
    published = pd.concat([
        reference.assign(method="manski", lower=[-0.6, -0.7], upper=[0.4, 0.3]),
        reference.assign(method="balke_pearl", lower=[0.05, np.nan], upper=[0.3, np.nan]),
    ]).assign(replication=0, study="s", bias_covariate="age", beta=1.0)
    _write(published, tmp_path / "ids" / "20260917" / "mc_intervals.parquet")
    summary = strata.assign(lower_mean=[0.0, -0.4], upper_mean=[0.3, 0.1], lp_posterior_prob_optimal=[1.0, 0.0])
    choices = pd.DataFrame({"covariate": ["resp_age_bin"], "tau": [0.1], "two_c": [0.2], "two_d": [0.05]})
    for root in (tmp_path / "published", tmp_path / "arms" / "matched" / "theory",
                 tmp_path / "arms" / "d3_beta" / "per_site"):
        _write(summary, root / leaf / "posterior_summary.parquet")
        _write(choices, root / leaf / "choices.parquet")
    shifted = tmp_path / "arms" / "delta=0.1" / "per_site" / leaf
    _write(summary, shifted / "posterior_summary.parquet")
    _write(choices, shifted / "choices.parquet")
    _write(reference.assign(rct_cate=reference.rct_cate + 0.1), shifted / "reference.parquet")
    _write(published.drop(columns=[c for c in reference.columns if c.startswith("rct_")]), shifted / "identification_bounds.parquet")
    _write(strata.assign(lower=[-0.1, np.nan], upper=[0.15, np.nan]), shifted / "transport.parquet")
    _write(strata.assign(cate_shift=[0.1, 0.1]), shifted / "effect_shift_log.parquet")

    tables = reg.summarize_arms([condition], tmp_path / "arms", published_root=tmp_path / "published",
                                identities_root=tmp_path / "ids")

    rows = tables["arms_strata"]
    shift_rows = rows[rows.arm == "effect_shift"]
    nat = shift_rows[(shift_rows.method == "naturalness") & (shift_rows.delta == 0.0)
                     & (shift_rows.grounding == "per_site")].set_index("resp_age_bin")
    assert nat.loc[1, "no_interval"] and (nat.loc[1, "lower"], nat.loc[1, "upper"]) == (-0.7, 0.3)
    assert nat.loc[0, "hit"] and nat.loc[0, "width"] == pytest.approx(0.3)
    transport = rows[(rows.method == "transport") & (rows.delta == 0.0)].set_index("resp_age_bin")
    assert transport.loc[1, "no_interval"] and transport.loc[0, "hit"]
    bp = rows[(rows.method == "balke_pearl") & (rows.delta == 0.0)].set_index("resp_age_bin")
    assert bp.loc[0, "decisive"] and bp.loc[0, "confirmed"] and bp.loc[1, "no_interval"]
    shifted_bp = rows[(rows.method == "balke_pearl") & (rows.delta == 0.1)].set_index("resp_age_bin")
    assert shifted_bp.loc[0, "rct_cate"] == pytest.approx(0.2)
    mismatch = tables["family_mismatch_by_beta"].set_index(["grounding", "mismatch"])
    assert mismatch.loc[("per_site", "beta"), "width_change"] == pytest.approx(0.0)
    assert set(tables["effect_shift_by_beta"]["delta"]) == {0.0, 0.1}
    assert (tmp_path / "arms" / "summary" / "effect_shift_by_beta.csv").exists()
    assert "d3_covariate" in (tmp_path / "arms" / "summary" / "missing_leaves.txt").read_text()


def test_family_mismatch_table_sets_the_manuscript_rows():
    """Each cell is mean width (hit ratio) of one grounding; the pooled row follows a midrule."""
    rows = []
    for beta in (0.5, 2.0):
        for arm, mismatch, method, grounding, width, hit in (
            ("effect_shift", "none", "naturalness", "per_site", 0.4, True),
            ("effect_shift", "none", "balke_pearl", "", 0.6, True),
            ("family_mismatch", "beta", "naturalness", "per_site", 0.5, False),
            ("family_mismatch", "covariate", "naturalness", "per_site", 0.3, True),
            ("family_mismatch", "beta", "naturalness", "theory", 0.9, True),
        ):
            rows.append({"arm": arm, "delta": 0.0, "mismatch": mismatch, "method": method, "grounding": grounding,
                         "beta": beta, "width": width + (0.1 if beta == 2.0 else 0.0), "hit": hit})

    lines = reg.family_mismatch_table_latex(pd.DataFrame(rows)).splitlines()

    assert lines[2] == r"$\beta$ & Matched & Another $\beta$ & Another covariate & Balke--Pearl \\"
    assert r"0.5 & 0.40 (1.00) & 0.50 (0.00) & 0.30 (1.00) & 0.60 (1.00) \\" in lines
    assert lines[-4] == r"\midrule"
    assert lines[-3] == r"Pooled & 0.45 (1.00) & 0.55 (0.00) & 0.35 (1.00) & 0.65 (1.00) \\"
    assert reg.family_mismatch_table_latex(pd.DataFrame(rows).query("arm != 'family_mismatch'")) == ""


def test_direct_effect_opens_a_gap_between_candidate_levels_without_moving_cates():
    """Both arms rise by gamma/2 where the candidate is 1 and fall by gamma/2 where it is 0."""
    trial = _trial(n_per_stratum=80)
    flips, log = reg.direct_effect_flips(trial, 0.2, covariates=["z"], candidate="z", study="s")
    shifted = reg.apply_effect_shift(trial, flips, target_site="pi")
    rates = shifted.groupby(["z", "iv"]).dv_binary.mean() - trial.groupby(["z", "iv"]).dv_binary.mean()
    np.testing.assert_allclose(rates.loc[1].to_numpy(), 0.1, atol=1 / 40 + 1e-12)
    np.testing.assert_allclose(rates.loc[0].to_numpy(), -0.1, atol=1 / 40 + 1e-12)
    np.testing.assert_allclose(log.cate_shift.to_numpy(), 0.0, atol=1 / 40 + 1e-12)
    again, _ = reg.direct_effect_flips(trial, 0.2, covariates=["z"], candidate="z", study="s")
    pd.testing.assert_frame_equal(flips, again)
    with pytest.raises(ValueError, match="gamma"):
        reg.direct_effect_flips(trial, -0.1, covariates=["z"], candidate="z", study="s")
    with pytest.raises(ValueError, match="stratum covariates"):
        reg.direct_effect_flips(trial, 0.1, covariates=["z"], candidate="w", study="s")


def test_direct_effect_moves_both_arms_by_what_the_saturated_arm_can():
    """A stratum whose treated arm cannot rise moves neither arm, so its CATE stays put."""
    trial = _trial(n_per_stratum=80)
    trial.loc[trial.iv.eq(1) & trial.z.eq(1), "dv_binary"] = 1
    _, log = reg.direct_effect_flips(trial, 0.2, covariates=["z"], candidate="z", study="s")
    saturated = log.set_index("z").loc[1]
    assert saturated["flipped_arm1"] == 0 and saturated["flipped_arm0"] == 0
    assert saturated["cate_shift"] == 0.0
    with pytest.raises(ValueError, match="exactly one"):
        reg.shift_target_data(trial, trial, target_site="pi", covariates=["z"], study="s")


def test_rule_tolerances_follow_their_rule():
    """Fixed, exact and prior rules fill 2c and 2d without reading any source surface."""
    class _Rows:
        """Minimal grounder exposing only the row check the rules call."""

        @staticmethod
        def _prepare_tolerance_sample_rows(rows):
            return rows.copy()

    rows = pd.DataFrame({"covariate": ["a", "a", "b"], "epsilon": [0.1, 0.2, 0.1], "tau": [0.1, 0.1, 0.2],
                         "i_deviation": [0.12, 0.12, 0.3]})
    fixed = reg._RuleTolerances(_Rows(), "fixed", two_c=0.3, two_d=0.05).sample_tolerances(rows, 2)
    assert len(fixed) == 6 and fixed.two_c.eq(0.3).all() and fixed.two_d.eq(0.05).all()
    assert np.allclose(fixed.c, 0.15)
    exact = reg._RuleTolerances(_Rows(), "exact").sample_tolerances(rows, 1)
    np.testing.assert_allclose(exact.two_c, [0.12, 0.12, 0.3], atol=1e-8)
    assert exact.two_d.eq(0.0).all()
    prior = reg._RuleTolerances(_Rows(), "prior").sample_tolerances(rows, 50, random_seed=0)
    per_candidate = prior.groupby(["sample_id", "covariate"]).two_c.nunique()
    assert per_candidate.eq(1).all() and prior.two_c.between(0, 1).all() and prior.two_c.std() > 0.1
    with pytest.raises(ValueError, match="two_c and two_d"):
        reg._RuleTolerances(_Rows(), "fixed")


def test_ablation_solves_every_rule_on_the_grounded_choices(exp16_run):
    """Vacuous fixed tolerances give width one; every rule shares one admissible pair per candidate."""
    config, _, observations, target_idle, tables = exp16_run
    out = reg.ablate_condition(
        config=config, target_index=7, observations=observations["stored"], target_idle=target_idle,
        tables=tables, floor=0.01, two_c_grid=(0.3, 1.0), two_d_grid=(0.05, 1.0),
    )
    summary = out["ablation_summary"]
    assert set(summary.rule) == {"exact", "prior", "fixed_0.3_0.05", "fixed_0.3_1", "fixed_1_0.05", "fixed_1_1"}
    assert out["choices"].groupby("covariate").size().eq(1).all()
    vacuous = summary[summary.rule.eq("fixed_1_1") & summary.lp_posterior_prob_optimal.gt(0)]
    np.testing.assert_allclose(vacuous.upper_mean - vacuous.lower_mean, 1.0, atol=1e-6)
    prior = summary[summary.rule.eq("prior")]
    assert prior.lp_posterior_n_samples.eq(config.lp_posterior_samples).all()


def test_summarize_arms_reads_the_invalid_instrument_and_ablation_arms(tmp_path):
    """Arms 3 and 4 enter their own tables next to the matched baseline."""
    strata = pd.DataFrame({"resp_age_bin": [0, 1], "resp_gender_bin": [0, 0], "resp_polideo_bin": [0, 0]})
    store = tmp_path / "20260917" / "tabicl_v2" / "tabicl_v2" / "tabicl_v2"
    condition = reg.Condition(str(store), 0, "s", "age", 1.0)
    leaf = Path("floor=0.075") / condition.label
    reference = strata.assign(rct_cate=[0.1, -0.2], rct_cate_ci90_lower=[0.02, -0.3], rct_cate_ci90_upper=[0.2, -0.1])
    published = pd.concat([
        reference.assign(method="manski", lower=[-0.6, -0.7], upper=[0.4, 0.3]),
        reference.assign(method="balke_pearl", lower=[0.15, -0.5], upper=[0.3, -0.25]),
    ]).assign(replication=0, study="s", bias_covariate="age", beta=1.0)
    _write(published, tmp_path / "ids" / "20260917" / "mc_intervals.parquet")
    summary = strata.assign(lower_mean=[0.0, -0.4], upper_mean=[0.3, 0.1], lp_posterior_prob_optimal=[1.0, 1.0])
    _write(summary, tmp_path / "published" / leaf / "posterior_summary.parquet")
    invalid = tmp_path / "arms" / "iv_gamma=0.1" / "theory" / leaf
    _write(summary, invalid / "posterior_summary.parquet")
    _write(reference, invalid / "reference.parquet")
    _write(published.drop(columns=[c for c in reference.columns if c.startswith("rct_")]),
           invalid / "identification_bounds.parquet")
    _write(strata.assign(lower=[-0.1, -0.3], upper=[0.15, -0.1]), invalid / "transport.parquet")
    _write(strata.assign(cate_shift=[0.0, 0.0], rate_change_arm1=[-0.05, 0.05], rate_change_arm0=[-0.05, 0.05]),
           invalid / "effect_shift_log.parquet")
    ablation = pd.concat([summary.assign(rule="exact"), summary.assign(rule="fixed_0.3_0.05", lower_mean=[-0.5, -0.6])])
    _write(ablation, tmp_path / "arms" / "ablation" / leaf / "ablation_summary.parquet")

    tables = reg.summarize_arms([condition], tmp_path / "arms", published_root=tmp_path / "published",
                                identities_root=tmp_path / "ids")

    invalid_table = tables["invalid_instrument_by_beta"]
    assert set(invalid_table.gamma) == {0.0, 0.1}
    bp = invalid_table[(invalid_table.method == "balke_pearl") & (invalid_table.gamma == 0.1)]
    assert bp.hit.iloc[0] == 0.0 and bp.decisive.iloc[0] == 1.0
    ablation_table = tables["grounding_ablation_by_study"].set_index("grounding")
    assert {"exact", "fixed_0.3_0.05", "per_site"} <= set(ablation_table.index)
    assert ablation_table.loc["fixed_0.3_0.05", "width"] > ablation_table.loc["exact", "width"]
    realized = tables["modified_target_realized"].set_index("arm")
    assert realized.loc["invalid_instrument", "direct_effect"] == pytest.approx(0.1)
