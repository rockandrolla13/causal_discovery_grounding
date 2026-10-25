"""Tests for Exp25 OSRCT Monte Carlo aggregation and plotting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

# exp25 imports experiments.benchmarks (via exp18), which needs the Exp16-Exp25
# cluster ML stack from requirements.txt. Skip, do not abort collection.
pytest.importorskip("torch")
pytest.importorskip("catenets")
pytest.importorskip("sklearn_quantile")

import experiments.exp16_osrct_calibrated_policy as exp16
import experiments.exp18_cb_meta_learner_comparison as exp18
import experiments.exp25_osrct_mc_baseline_comparison as exp25
import experiments.osrct_target_pipeline as target_pipeline


def _comparison_fixture() -> pd.DataFrame:
    """Return one complete Exp18-style comparison row."""

    return pd.DataFrame(
        {
            "study": ["gainloss"],
            "target_site": ["pi"],
            "resp_age_bin": [0],
            "resp_gender_bin": [1],
            "resp_polideo_bin": [0],
            "n_target_idle_rows": [20],
            "rct_cate": [0.1],
            "rct_cate_ci90_lower": [0.0],
            "rct_cate_ci90_upper": [0.2],
            "rct_cate_available": [True],
            "naturalness_lower": [-0.3],
            "naturalness_upper": [0.4],
            "lp_posterior_prob_optimal": [0.9],
            "blearner_lower": [-1.4],
            "blearner_upper": [0.6],
            "blearner_gamma": [np.exp(0.5)],
            "outside_binary_cate_support": [True],
            "cb_ipw_lower_raw": [0.4],
            "cb_ipw_upper_raw": [-0.2],
            "cb_ipw_lower": [-1.0],
            "cb_ipw_upper": [1.0],
            "cb_ipw_used_full_range_fallback": [True],
            "cb_ipw_interval_inverted": [False],
            "cb_dr_lower_raw": [-0.5],
            "cb_dr_upper_raw": [0.7],
            "cb_dr_lower": [-0.5],
            "cb_dr_upper": [0.7],
            "cb_dr_used_full_range_fallback": [False],
            "cb_dr_interval_inverted": [False],
        }
    )


def test_defaults_and_gamma_rule() -> None:
    """Defaults should encode TabICL routing and DGP Gamma calibration."""

    config = exp25.Exp25Config()

    assert config.seeds == tuple(range(42, 62))
    assert config.stage1_backend == "tabicl_v2"
    assert config.target_trigger_probability_backend == "tabicl_v2"
    assert config.lp_probability_backend == "tabicl_v2"
    assert config.require_observed_gender is True
    assert config.blearner_gamma == exp18.DGP_ORACLE_GAMMA
    assert exp25._scientific_config(config)["cohort_filter"] == (
        "resp_sex_not_missing"
    )
    assert exp25.config_fingerprint(config) != exp25.config_fingerprint(
        replace(config, require_observed_gender=False)
    )
    assert exp25._scientific_config(config)["blearner_gamma_rule"] == (
        exp18.DGP_ORACLE_GAMMA
    )
    assert exp25._naturalness_bundle_leaf(config).parts[-3:] == (
        "tabicl_v2",
        "tabicl_v2",
        "tabicl_v2",
    )
    assert exp25._output_leaf(config).parts[-2:] == (
        "baseline_variants",
        "gamma_dgp_oracle",
    )
    assert exp25._output_leaf(config).parent.parent == (
        exp25._naturalness_bundle_leaf(config)
    )


def test_alternate_gamma_uses_isolated_leaf_without_a_canonical_primary(
) -> None:
    """Any Gamma may own the store, and replay shares its own Gamma's leaf."""

    # No Gamma is privileged: a non-dgp_oracle arm may run in full mode.
    assert exp25.Exp25Config(blearner_gamma=2.0).blearner_gamma == 2.0

    config = exp25.parse_args(
        ["--baseline-only", "--blearner-gamma", "2.5", "--no-plots"]
    )

    assert config.blearner_gamma == 2.5
    assert exp25._output_leaf(config).parts[-2:] == (
        "baseline_variants",
        "gamma_fixed_2p5",
    )
    assert exp25._naturalness_bundle_leaf(config).parts[-3:] == (
        "tabicl_v2",
        "tabicl_v2",
        "tabicl_v2",
    )
    assert exp25.config_fingerprint(config) != (
        exp25.naturalness_bundle_fingerprint(config)
    )
    # Replay mode is provenance, not identity, so it never splits the leaf.
    assert "baseline_replay_mode" not in exp25._scientific_config(config)
    assert exp25.config_fingerprint(config) == exp25.config_fingerprint(
        replace(config, baseline_only=False)
    )
    assert exp25._baseline_replay_provenance()["baseline_replay_mode"] == (
        "blearner_only_v1"
    )
    dgp_replay = exp25.parse_args(["--baseline-only", "--no-plots"])
    assert exp25._output_leaf(dgp_replay).parts[-2:] == (
        "baseline_variants",
        "gamma_dgp_oracle",
    )


def test_bundle_fingerprint_is_gamma_independent() -> None:
    """Whichever arm runs first owns a store every other arm can replay."""

    dgp_oracle = exp25.parse_args(["--no-plots"])
    fixed = exp25.parse_args(
        ["--baseline-only", "--blearner-gamma", "3.0", "--no-plots"]
    )
    width_matched = exp25.parse_args(
        ["--baseline-only", "--blearner-gamma", "width_matched", "--no-plots"]
    )

    fingerprints = {
        exp25.naturalness_bundle_fingerprint(config)
        for config in (dgp_oracle, fixed, width_matched)
    }
    assert len(fingerprints) == 1
    # The Gamma-independent store identity still tracks Naturalness settings.
    assert exp25.naturalness_bundle_fingerprint(dgp_oracle) != (
        exp25.naturalness_bundle_fingerprint(
            replace(dgp_oracle, require_observed_gender=False)
        )
    )
    assert len(
        {exp25.config_fingerprint(config) for config in (dgp_oracle, fixed)}
    ) == 2


def test_bundle_store_allows_only_one_full_mode_owner(tmp_path: Path) -> None:
    """A second Gamma must use baseline-only after one arm claims the store."""

    owner = exp25.Exp25Config(output_dir=str(tmp_path), blearner_gamma=2.0)
    challenger = replace(owner, blearner_gamma="width_matched")
    bundle_root = exp25._write_or_validate_bundle_manifest(owner)
    owner_path = bundle_root / exp25.EXP25_FULL_MODE_OWNER_FILENAME

    assert owner_path.read_text(encoding="utf-8").strip() == (
        exp25.config_fingerprint(owner)
    )
    assert exp25._write_or_validate_bundle_manifest(owner) == bundle_root
    with pytest.raises(ValueError, match="--baseline-only"):
        exp25._write_or_validate_bundle_manifest(challenger)

    # Existing stores created before the owner file use their manifest owner.
    owner_path.unlink()
    with pytest.raises(ValueError, match="--baseline-only"):
        exp25._write_or_validate_bundle_manifest(challenger)
    assert exp25._write_or_validate_bundle_manifest(owner) == bundle_root


def test_gamma_grid_expands_to_one_config_per_value() -> None:
    """A Gamma grid must yield isolated leaves and share every other field."""

    argv = [
        "--baseline-only",
        "--no-plots",
        "--blearner-gamma",
        "1.0",
        "2.5",
        "width_matched",
    ]
    configs = exp25.parse_configs(argv)

    assert [config.blearner_gamma for config in configs] == [
        1.0,
        2.5,
        "width_matched",
    ]
    assert [exp25._output_leaf(config).name for config in configs] == [
        "gamma_fixed_1",
        "gamma_fixed_2p5",
        "gamma_width_matched",
    ]
    assert len({exp25.config_fingerprint(config) for config in configs}) == 3
    assert len({config.betas for config in configs}) == 1
    assert exp25.parse_args(argv) == configs[0]


def test_grid_token_expands_to_fixed_sensitivity_grid() -> None:
    """The grid shorthand must fan out to one isolated leaf per fixed Gamma."""

    configs = exp25.parse_configs(
        ["--baseline-only", "--no-plots", "--blearner-gamma", "grid"]
    )

    assert [config.blearner_gamma for config in configs] == [
        float(gamma) for gamma in exp25.BLEARNER_GAMMA_GRID
    ]
    leaves = [exp25._output_leaf(config) for config in configs]
    assert all(leaf.parts[-2] == "baseline_variants" for leaf in leaves)
    assert leaves[0].name == "gamma_fixed_1"
    assert len({leaf.name for leaf in leaves}) == len(exp25.BLEARNER_GAMMA_GRID)


def test_grid_token_composes_with_explicit_gammas() -> None:
    """The shorthand must expand in place beside other requested Gammas."""

    configs = exp25.parse_configs(
        [
            "--baseline-only",
            "--no-plots",
            "--blearner-gamma",
            "grid",
            "width_matched",
        ]
    )

    assert configs[-1].blearner_gamma == "width_matched"
    assert len(configs) == len(exp25.BLEARNER_GAMMA_GRID) + 1

    with pytest.raises(SystemExit):
        exp25.parse_configs(
            ["--baseline-only", "--no-plots", "--blearner-gamma", "grid", "2.0"]
        )


def test_gamma_grid_rejects_duplicates() -> None:
    """A repeated Gamma would rerun the same leaf, so it must be an error."""

    with pytest.raises(SystemExit):
        exp25.parse_configs(
            [
                "--baseline-only",
                "--no-plots",
                "--blearner-gamma",
                "2.0",
                "2.0",
            ]
        )


@pytest.mark.parametrize("gamma", ("oracle", "exp_beta", "width_matched"))
def test_parser_accepts_deferred_gamma_strategies(gamma: str) -> None:
    """Existing Exp18 Gamma strategies should be available for replay."""

    config = exp25.parse_args(
        ["--baseline-only", "--blearner-gamma", gamma, "--no-plots"]
    )

    assert config.blearner_gamma == gamma
    assert exp25._output_leaf(config).name == f"gamma_{gamma}"


def test_bundle_store_manifest_is_validated_for_gamma_variant(
    tmp_path: Path,
) -> None:
    """A replay must read only a store built from matching settings."""

    pytest.importorskip("pyarrow")
    config = exp25.Exp25Config(
        n_replications=1,
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
        output_dir=str(tmp_path),
        baseline_only=True,
        blearner_gamma=2.0,
    )
    bundle_root = exp25._naturalness_bundle_leaf(config)
    bundle_root.mkdir(parents=True)
    manifest = bundle_root / "run_manifest.yml"
    manifest.write_text(
        "naturalness_bundle_fingerprint: "
        f"{exp25.naturalness_bundle_fingerprint(config)}\n"
    )

    assert exp25._validate_naturalness_bundle_root(config) == bundle_root

    source_intervals = exp25.comparison_to_long_intervals(
        _comparison_fixture(),
        replication=0,
        seed=42,
        pattern="age",
        beta=0.5,
        config_fingerprint_value=exp25.naturalness_bundle_fingerprint(config),
    )
    exp25._write_replay_source_intervals(
        bundle_root,
        source_intervals,
        bundle_fingerprint=exp25.naturalness_bundle_fingerprint(config),
    )
    loaded = exp25._load_replay_source_intervals(config, bundle_root)
    pd.testing.assert_frame_equal(loaded, source_intervals)

    manifest.write_text("naturalness_bundle_fingerprint: wrong\n")
    with pytest.raises(ValueError, match="different OSRCT or Naturalness"):
        exp25._validate_naturalness_bundle_root(config)

    exp25._write_replay_source_intervals(
        bundle_root,
        source_intervals,
        bundle_fingerprint="wrong",
    )
    with pytest.raises(ValueError, match="different scientific fingerprint"):
        exp25._load_replay_source_intervals(config, bundle_root)


def test_store_owner_publishes_replay_source_for_other_arms(
    tmp_path: Path,
) -> None:
    """A non-dgp_oracle arm may own the store that other arms replay from."""

    pytest.importorskip("pyarrow")
    owner = exp25.Exp25Config(
        n_replications=1,
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
        output_dir=str(tmp_path),
        blearner_gamma=2.0,
    )
    replay = replace(owner, blearner_gamma="width_matched", baseline_only=True)

    bundle_root = exp25._write_or_validate_bundle_manifest(owner)
    intervals = exp25.comparison_to_long_intervals(
        _comparison_fixture(),
        replication=0,
        seed=owner.seeds[0],
        pattern="age",
        beta=0.5,
        config_fingerprint_value=exp25.config_fingerprint(owner),
    )
    exp25._write_replay_source_intervals(
        bundle_root,
        intervals,
        bundle_fingerprint=exp25.naturalness_bundle_fingerprint(owner),
    )

    # The replay validates the store the fixed-Gamma arm wrote, not a leaf.
    assert exp25._validate_naturalness_bundle_root(replay) == bundle_root
    loaded = exp25._load_replay_source_intervals(replay, bundle_root)
    assert set(loaded["method"]) == set(exp25.EXP25_METHODS)
    assert exp25._output_leaf(owner) != exp25._output_leaf(replay)


def test_exp18_condition_forwards_deferred_gamma(tmp_path: Path) -> None:
    """Exp25 should pass selected Gamma strategy into matched Exp18 fits."""

    config = exp25.Exp25Config(
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
        baseline_only=True,
        blearner_gamma="oracle",
    )

    exp18_config = exp25._exp18_config_for_condition(
        config,
        seed=42,
        pattern="age",
        beta=0.5,
        data_root=tmp_path / "osrct",
        naturalness_reference=tmp_path / "reference.parquet",
        output_dir=tmp_path / "exp18",
    )

    assert exp18_config.gamma == "oracle"


def test_deferred_gamma_replay_reads_bundle_and_skips_exp16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternate Gamma replay should leave primary bundles read-only."""

    pytest.importorskip("pyarrow")
    config = exp25.Exp25Config(
        n_replications=1,
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
        baseline_only=True,
        blearner_gamma=2.0,
        make_plots=False,
        verbose=False,
    )
    bundle_replication_dir = tmp_path / "primary" / "replication=000"
    (bundle_replication_dir / "osrct").mkdir(parents=True)
    marker = bundle_replication_dir / "primary_bundle_marker"
    marker.write_text("preserve\n")
    reference = tmp_path / "reference.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(reference)
    received_gamma: list[float | str] = []
    monkeypatch.setattr(
        exp25,
        "run_exp16",
        lambda *args, **kwargs: pytest.fail("Exp16 reran during Gamma replay"),
    )
    monkeypatch.setattr(
        exp25,
        "_load_persisted_naturalness_references",
        lambda *args, **kwargs: {("age", 0.5): reference},
    )

    monkeypatch.setattr(
        exp25,
        "run_exp18",
        lambda *args, **kwargs: pytest.fail(
            "Full Exp18/meta-learners reran during Gamma replay"
        ),
    )

    def fake_blearner_only(
        exp18_config: exp18.Exp18Config,
    ) -> dict[str, pd.DataFrame]:
        """Record Gamma and return newly selected B-Learner bounds."""

        received_gamma.append(exp18_config.gamma)
        selected = _comparison_fixture().loc[
            :,
            ["study", *exp25.EXP25_STRATUM_COLUMNS],
        ].copy()
        selected["gamma"] = 2.0
        selected["blearner_lower"] = -0.8
        selected["blearner_upper"] = 0.5
        selected["outside_binary_cate_support"] = False
        return {
            "selected_blearner_bounds": selected,
            "blearner_oracle_gamma": pd.DataFrame(),
        }

    monkeypatch.setattr(exp25, "fit_blearner_only", fake_blearner_only)
    result_replication_dir = tmp_path / "variant" / "replication=000"
    primary_intervals = exp25.comparison_to_long_intervals(
        _comparison_fixture(),
        replication=0,
        seed=42,
        pattern="age",
        beta=0.5,
        config_fingerprint_value=exp25.naturalness_bundle_fingerprint(config),
    )

    intervals, diagnostics = exp25.run_one_replication(
        config,
        replication=0,
        seed=42,
        scratch_root=tmp_path / "scratch",
        bundle_replication_dir=bundle_replication_dir,
        result_replication_dir=result_replication_dir,
        fingerprint=exp25.config_fingerprint(config),
        primary_intervals=primary_intervals,
    )

    assert received_gamma == [2.0]
    assert marker.read_text() == "preserve\n"
    assert not (result_replication_dir / exp25.EXP25_GAMMA_FILENAME).exists()
    assert set(intervals["method"]) == set(exp25.EXP25_METHODS)
    by_method = intervals.set_index("method")
    assert by_method.loc["blearner", "raw_lower"] == -0.8
    assert by_method.loc["blearner", "raw_upper"] == 0.5
    assert by_method.loc["blearner", "blearner_gamma"] == 2.0
    assert by_method.loc["cb_ipw", "raw_lower"] == 0.4
    assert by_method.loc["cb_dr", "raw_lower"] == -0.5
    assert by_method.loc["naturalness", "raw_lower"] == -0.3
    assert intervals["config_fingerprint"].eq(
        exp25.config_fingerprint(config)
    ).all()
    assert diagnostics["status"].eq("completed").all()


def _iv_target_idle(rows_per_stratum: int = 100) -> pd.DataFrame:
    """Build deterministic target idle rows where only age moves treatment.

    Treatment is 90% under age bin 1 and 10% under age bin 0, independent of
    gender and polideo. The outcome depends on treatment alone, so age is a
    valid binary instrument and the other two covariates are irrelevant.
    """

    rows = []
    for age in (0, 1):
        for gender in (0, 1):
            for polideo in (0, 1):
                treated = int(rows_per_stratum * (0.9 if age else 0.1))
                for index in range(rows_per_stratum):
                    iv = int(index < treated)
                    position = index if iv else index - treated
                    arm = treated if iv else rows_per_stratum - treated
                    dv = int(position < arm * (0.6 if iv else 0.3))
                    rows.append(dict(site="pi", resp_age_bin=age, resp_gender_bin=gender,
                                     resp_polideo_bin=polideo, iv=iv, dv_binary=dv))
    return pd.DataFrame(rows)


def _iv_identities(idle: pd.DataFrame, study: str = "gainloss") -> pd.DataFrame:
    """Return one primary identity row per observed target stratum."""

    columns = list(exp25.EXP25_STRATUM_COLUMNS)
    identities = idle.groupby(columns).size().rename("n_target_idle_rows").reset_index()
    return identities.assign(study=study, target_site="pi", rct_cate=0.3,
                             rct_cate_ci90_lower=0.1, rct_cate_ci90_upper=0.5,
                             rct_cate_available=True).loc[:, list(exp25.EXP25_IDENTITY_COLUMNS)]


def test_identification_bounds_score_instruments_and_mark_missing_ones() -> None:
    """Balke-Pearl uses the relevant covariate; without support it is invalid."""

    idle = _iv_target_idle()
    identities = _iv_identities(idle)
    kwargs = dict(tau=0.01, replication=0, seed=42, pattern="age", beta=0.5,
                  config_fingerprint_value="fingerprint")
    rows = exp25.identification_bound_intervals(
        {"gainloss": idle}, identities, min_count=20, **kwargs)
    assert list(rows.columns[: len(exp25.EXP25_INTERVAL_COLUMNS)]) == list(
        exp25.EXP25_INTERVAL_COLUMNS)
    manski = rows.loc[rows["method"].eq("manski")]
    balke_pearl = rows.loc[rows["method"].eq("balke_pearl")]
    assert len(manski) == len(balke_pearl) == len(identities) == 8
    # A binary-outcome Manski interval has width one and always contains zero.
    assert manski["is_valid"].all()
    np.testing.assert_allclose(manski["width"], 1.0)
    assert (manski["lower"] <= 0).all() and (manski["upper"] >= 0).all()
    assert balke_pearl["is_valid"].all()
    assert balke_pearl["bp_candidates"].eq(1).all()
    assert (balke_pearl["width"] < 1).all()
    assert balke_pearl["rct_cate"].eq(0.3).all()
    assert rows["config_fingerprint"].eq("fingerprint").all()

    unsupported = exp25.identification_bound_intervals(
        {"gainloss": idle}, identities, min_count=10_000, **kwargs)
    balke_pearl = unsupported.loc[unsupported["method"].eq("balke_pearl")]
    assert not balke_pearl["is_valid"].any()
    assert balke_pearl["exclusion_reason"].eq("no_supported_instrument").all()
    assert balke_pearl["lower"].isna().all()
    assert unsupported.loc[unsupported["method"].eq("manski"), "is_valid"].all()


def test_identification_bounds_reject_rows_that_miss_primary_counts() -> None:
    """Target rows that do not reproduce the primary strata counts fail loudly."""

    idle = _iv_target_idle()
    identities = _iv_identities(idle)
    identities.loc[0, "n_target_idle_rows"] += 1
    with pytest.raises(ValueError, match="do not reproduce the primary strata"):
        exp25.identification_bound_intervals(
            {"gainloss": idle}, identities, min_count=20, tau=0.01, replication=0,
            seed=42, pattern="age", beta=0.5, config_fingerprint_value="fingerprint")


def test_identification_replay_writes_its_own_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replay adds baseline_variants/identification_bounds and guards it.

    An explicit ``output_dir`` takes a replay with other settings elsewhere and
    leaves the store's own leaf as it was.
    """

    pytest.importorskip("pyarrow")
    config = exp25.Exp25Config(
        n_replications=1,
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
        output_dir=str(tmp_path),
        identification_bounds_only=True,
        make_plots=False,
        verbose=False,
    )
    bundle_root = exp25._naturalness_bundle_leaf(config)
    bundle_root.mkdir(parents=True)
    (bundle_root / "run_manifest.yml").write_text(
        f"naturalness_bundle_fingerprint: {exp25.naturalness_bundle_fingerprint(config)}\n")
    idle = _iv_target_idle()
    comparison = pd.concat([_comparison_fixture()] * 8, ignore_index=True)
    comparison[list(exp25.EXP25_STRATUM_COLUMNS)] = _iv_identities(idle)[
        list(exp25.EXP25_STRATUM_COLUMNS)].to_numpy()
    comparison["n_target_idle_rows"] = 100
    exp25._write_replay_source_intervals(
        bundle_root,
        exp25.comparison_to_long_intervals(
            comparison, replication=0, seed=config.seeds[0], pattern="age", beta=0.5,
            config_fingerprint_value=exp25.naturalness_bundle_fingerprint(config)),
        bundle_fingerprint=exp25.naturalness_bundle_fingerprint(config),
    )
    monkeypatch.setattr(exp25, "_load_persisted_naturalness_references",
                        lambda *args, **kwargs: {("age", 0.5): tmp_path / "reference.parquet"})
    monkeypatch.setattr(exp25, "_load_fitting_data", lambda *args, **kwargs: (
        {"gainloss": exp18.PreparedStudyData(idle=idle, rct=pd.DataFrame())}, pd.DataFrame()))

    result = exp25.run_experiment(config)
    leaf = bundle_root / "baseline_variants" / exp25.EXP25_IDENTIFICATION_DIRNAME
    assert result["output_dir"] == leaf
    written = pd.read_parquet(leaf / exp25.EXP25_INTERVALS_FILENAME)
    assert set(written["method"]) == set(exp25.EXP25_IDENTIFICATION_METHODS)
    assert len(written) == 16
    assert (leaf / exp25.EXP25_DIAGNOSTICS_FILENAME).is_file()
    assert exp25.identification_bounds_fingerprint(config) in (
        leaf / "run_manifest.yml").read_text()
    with pytest.raises(ValueError, match="different settings"):
        exp25.run_experiment(replace(config, identification_tau=0.2))

    stricter = replace(config, identification_tau=0.2)
    elsewhere = tmp_path / "elsewhere"
    moved = exp25.run_identification_bounds_replay(stricter, output_dir=elsewhere)
    assert moved["output_dir"] == elsewhere
    assert len(pd.read_parquet(elsewhere / exp25.EXP25_INTERVALS_FILENAME)) == 16
    assert exp25.identification_bounds_fingerprint(stricter) in (
        elsewhere / "run_manifest.yml").read_text()
    assert exp25.identification_bounds_fingerprint(config) in (
        leaf / "run_manifest.yml").read_text()
    pd.testing.assert_frame_equal(pd.read_parquet(leaf / exp25.EXP25_INTERVALS_FILENAME), written)


def _split_config(tmp_path: Path, **overrides) -> exp25.Exp25Config:
    """Return a one-condition config for the split-run pass tests."""

    return exp25.Exp25Config(**{
        "n_replications": 3, "studies": ("gainloss",), "bias_covariates": ("age",),
        "betas": (0.5,), "output_dir": str(tmp_path), "make_plots": False,
        "verbose": False, **overrides})


def _fake_bundle(replication_dir: Path) -> None:
    """Write the files that mark one persisted Naturalness bundle complete."""

    pd.DataFrame({"n": [1]}).to_parquet(replication_dir / "naturalness_artifact_index.parquet")
    pd.DataFrame({"n": [1]}).to_parquet(replication_dir / "naturalness_run_index.parquet")
    leaf = replication_dir / "naturalness" / "all_triggered" / "pi" / "age" / "0.5"
    leaf.mkdir(parents=True, exist_ok=True)
    for name in ("target_gp_posterior_summary", "source_site_roles"):
        pd.DataFrame({"n": [1]}).to_parquet(leaf / f"{name}.parquet")


def test_split_passes_reject_mixed_modes() -> None:
    """Each split pass runs alone, and shards belong to the Naturalness pass."""

    with pytest.raises(ValueError, match="split-run passes"):
        exp25.Exp25Config(naturalness_only=True, comparators_only=True)
    with pytest.raises(ValueError, match="split-run passes"):
        exp25.Exp25Config(comparators_only=True, baseline_only=True)
    with pytest.raises(ValueError, match="shards of the naturalness_only"):
        exp25.Exp25Config(replications=(0,))
    with pytest.raises(ValueError, match="distinct indices"):
        exp25.Exp25Config(naturalness_only=True, n_replications=2, replications=(2,))
    config = exp25.parse_args(["--naturalness-only", "--replications", "3", "7"])
    assert config.naturalness_only and config.replications == (3, 7)
    assert exp25.parse_args(["--comparators-only"]).comparators_only
    # Both passes share one store and leaf with a full run of the same settings.
    assert exp25.config_fingerprint(config) == exp25.config_fingerprint(
        replace(config, naturalness_only=False, replications=None))


def test_naturalness_pass_fits_its_shard_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GPU pass fits only its replications, skips complete ones, writes no leaf."""

    pytest.importorskip("pyarrow")
    fitted: list[int] = []

    def fake_build(config, *, seed, bundle_replication_dir, scratch_root):
        """Record the fitted seed and leave a complete bundle behind."""
        fitted.append(seed)
        _fake_bundle(bundle_replication_dir)

    monkeypatch.setattr(exp25, "_build_naturalness_bundle", fake_build)
    config = _split_config(tmp_path, naturalness_only=True, replications=(0, 2))
    result = exp25.run_experiment(config)
    assert result["fitted"] == [0, 2] and fitted == [config.seeds[0], config.seeds[2]]
    bundle_root = exp25._naturalness_bundle_leaf(config)
    assert exp25.naturalness_bundle_fingerprint(config) in (
        bundle_root / "run_manifest.yml").read_text()
    assert not (bundle_root / "baseline_variants").exists()
    again = exp25.run_experiment(config)
    assert again["fitted"] == [] and again["skipped"] == [0, 2]
    # A shard without replication 0 validates the store replication 0 made.
    other = exp25.run_experiment(replace(config, replications=(1,)))
    assert other["fitted"] == [1]


def test_naturalness_shard_waits_for_the_initializing_shard(tmp_path: Path) -> None:
    """Only the replication-0 shard initializes; the others wait, then time out."""

    config = _split_config(tmp_path, naturalness_only=True, replications=(1,))
    with pytest.raises(TimeoutError, match="No initialized"):
        exp25._wait_for_bundle_manifest(config, timeout_seconds=0.05, poll_seconds=0.01)
    exp25._write_or_validate_bundle_manifest(config)
    exp25._wait_for_bundle_manifest(config, timeout_seconds=0.05, poll_seconds=0.01)


def test_comparators_pass_fits_full_exp18_from_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU pass reads persisted bundles and never regenerates or refits Exp16."""

    pytest.importorskip("pyarrow")
    config = _split_config(tmp_path, comparators_only=True)
    bundle_replication_dir = tmp_path / "store" / "replication=000"
    (bundle_replication_dir / "osrct").mkdir(parents=True)
    reference = tmp_path / "reference.parquet"
    monkeypatch.setattr(exp25, "generate_replication_osrct_grid",
                        lambda *args, **kwargs: pytest.fail("OSRCT regenerated in CPU pass"))
    monkeypatch.setattr(exp25, "run_exp16",
                        lambda *args, **kwargs: pytest.fail("Exp16 refit in CPU pass"))
    monkeypatch.setattr(exp25, "_load_persisted_naturalness_references",
                        lambda *args, **kwargs: {("age", 0.5): reference})
    received = []

    def fake_exp18(exp18_config):
        """Record the reference and return a complete comparison."""
        received.append(Path(exp18_config.naturalness_reference))
        return {"comparison": _comparison_fixture(), "blearner_oracle_gamma": pd.DataFrame()}

    monkeypatch.setattr(exp25, "run_exp18", fake_exp18)
    intervals, _ = exp25.run_one_replication(
        config, replication=0, seed=config.seeds[0], scratch_root=tmp_path / "scratch",
        bundle_replication_dir=bundle_replication_dir,
        result_replication_dir=tmp_path / "leaf" / "replication=000",
        fingerprint=exp25.config_fingerprint(config))
    assert received == [reference]
    assert set(intervals["method"]) == set(exp25.EXP25_METHODS)


def _fake_result_root(root: Path, **overrides) -> exp25.Exp25Config:
    """Write a minimal finished Exp25 root: store manifest plus both leaves."""

    config = _split_config(root, n_replications=1, **overrides)
    bundle_root = exp25._write_or_validate_bundle_manifest(config)
    intervals = pd.concat([
        exp25.comparison_to_long_intervals(
            _comparison_fixture(), replication=0, seed=config.seeds[0], pattern="age",
            beta=float(beta), config_fingerprint_value=exp25.config_fingerprint(config))
        for beta in config.betas], ignore_index=True)
    primary = bundle_root / "baseline_variants" / "gamma_dgp_oracle"
    primary.mkdir(parents=True)
    intervals.to_parquet(primary / exp25.EXP25_INTERVALS_FILENAME)
    exp25.build_run_diagnostics(intervals, runtime_seconds=1.0).to_parquet(
        primary / exp25.EXP25_DIAGNOSTICS_FILENAME)
    pd.DataFrame({"beta": list(config.betas)}).to_parquet(primary / exp25.EXP25_GAMMA_FILENAME)
    bounds = bundle_root / "baseline_variants" / exp25.EXP25_IDENTIFICATION_DIRNAME
    bounds.mkdir(parents=True)
    manski = intervals.loc[intervals["method"].eq("naturalness")].assign(method="manski")
    manski.to_parquet(bounds / exp25.EXP25_INTERVALS_FILENAME)
    exp25.build_run_diagnostics(manski, runtime_seconds=1.0).to_parquet(
        bounds / exp25.EXP25_DIAGNOSTICS_FILENAME)
    exp25.write_run_manifest(
        bounds, experiment_name=exp25.EXP25_NAME, script_path="test", config={}, argv=(),
        cluster_manifest_path=None,
        extra={"identification_config": exp25._identification_scientific_config(config)})
    return config


def test_merge_combines_disjoint_betas_into_a_fresh_tree(tmp_path: Path) -> None:
    """Runs differing only in beta merge; overlap, other differences, reuse fail."""

    pytest.importorskip("pyarrow")
    _fake_result_root(tmp_path / "old", betas=(0.5, 1.0))
    _fake_result_root(tmp_path / "new", betas=(1.5, 2.0))
    assert exp25.main(["merge", str(tmp_path / "old"), str(tmp_path / "new"),
                       "--output-dir", str(tmp_path / "merged")]) == 0
    leaf = exp25._backend_leaf_of(tmp_path / "old").relative_to(tmp_path / "old")
    variants = tmp_path / "merged" / leaf / "baseline_variants"
    for name in ("gamma_dgp_oracle", exp25.EXP25_IDENTIFICATION_DIRNAME):
        merged = pd.read_parquet(variants / name / exp25.EXP25_INTERVALS_FILENAME)
        assert sorted(merged["beta"].unique()) == [0.5, 1.0, 1.5, 2.0]
    assert (tmp_path / "merged" / leaf / "merge_manifest.json").is_file()

    with pytest.raises(FileExistsError):
        exp25.merge_result_roots([tmp_path / "old", tmp_path / "new"], tmp_path / "merged")
    _fake_result_root(tmp_path / "overlap", betas=(1.0, 1.5))
    with pytest.raises(ValueError, match="betas overlap"):
        exp25.merge_result_roots([tmp_path / "old", tmp_path / "overlap"], tmp_path / "m2")
    _fake_result_root(tmp_path / "other", betas=(1.5,), base_seed=7)
    with pytest.raises(ValueError, match="differs from"):
        exp25.merge_result_roots([tmp_path / "old", tmp_path / "other"], tmp_path / "m3")


def test_identification_bounds_only_is_a_separate_mode() -> None:
    """The bounds replay parses from the CLI and excludes the other replays."""

    config = exp25.parse_args(["--identification-bounds-only", "--identification-tau", "0.05",
                               "--identification-min-count", "10"])
    assert config.identification_bounds_only
    assert config.identification_tau == 0.05 and config.identification_min_count == 10
    assert exp25.Exp25Config().identification_tau == min(target_pipeline.Exp16Config().taus)
    # Balke-Pearl shares the support floor Exp25's Naturalness runs with.
    assert exp25.Exp25Config().identification_min_count == target_pipeline.Exp16Config().min_cell_count
    for other in ("baseline_only", "dummy"):
        with pytest.raises(ValueError, match="identification_bounds_only"):
            exp25.Exp25Config(identification_bounds_only=True, **{other: True})


@pytest.mark.parametrize(
    "backend",
    (
        "tabicl_v2",
        "hierarchical_dirichlet",
        "hierarchical_dirichlet_draws",
    ),
)
def test_lp_probability_backend_is_runtime_selectable(backend: str) -> None:
    """Each supported LP backend should parse into an isolated output leaf."""

    config = exp25.parse_args(
        ["--lp-probability-backend", backend, "--no-plots"]
    )

    assert config.lp_probability_backend == backend
    assert exp25._naturalness_bundle_leaf(config).name == backend
    assert exp25._output_leaf(config).parts[-3:] == (
        backend,
        "baseline_variants",
        "gamma_dgp_oracle",
    )


def test_unsupported_lp_probability_backend_is_rejected() -> None:
    """Exp25 should reject LP backends outside its declared comparison set."""

    with pytest.raises(ValueError, match="lp_probability_backend"):
        exp25.Exp25Config(lp_probability_backend="empirical")


def test_unsupported_stage1_backend_is_rejected() -> None:
    """Exp25 keeps Stage 1 on site-local TabICLv2."""

    with pytest.raises(ValueError, match="stage1_backend"):
        exp25.Exp25Config(stage1_backend="xgboost")


def test_exp18_forwards_seed_specific_data_configuration() -> None:
    """Exp18 must not silently reload default seed-42 OSRCT rows."""

    config = exp18.Exp18Config(
        studies=("gainloss",),
        data_root="/tmp/seed-specific-osrct",
        rct_path="/tmp/manylabs.pkl",
        seed=57,
    )

    forwarded = exp18._exp16_config(config, "gainloss")

    assert forwarded.data_root == "/tmp/seed-specific-osrct"
    assert forwarded.rct_path == "/tmp/manylabs.pkl"
    assert forwarded.require_observed_gender is True
    assert forwarded.seed == 57


def test_exp16_cli_exposes_posterior_lp_workers() -> None:
    """Exp16 CLI should retain serial Stage 1 and configure LP draw threads."""

    config = exp16.parse_args(
        [
            "--fit-gp",
            "--require-observed-gender",
            "--lp-posterior-samples",
            "4",
            "--lp-posterior-n-jobs",
            "3",
            "--no-plots",
        ]
    )

    assert config.lp_posterior_samples == 4
    assert config.lp_posterior_n_jobs == 3
    assert config.require_observed_gender is True


def test_exp25_exp16_config_uses_gpu_trigger_and_cate_only_lp(
    tmp_path: Path,
) -> None:
    """Exp25 should avoid unused LP objectives and route both TabICL stages."""

    config = exp25.Exp25Config(
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
    )

    exp16_config = exp25._exp16_config_for_replication(
        config,
        seed=42,
        data_root=tmp_path / "osrct",
        output_dir=tmp_path / "exp16",
    )

    assert exp16_config.stage1_backend == "tabicl_v2"
    assert exp16_config.validation_stage1_backend == "tabicl_v2"
    assert exp16_config.target_trigger_probability_backend == "tabicl_v2"
    assert exp16_config.lp_probability_backend == "tabicl_v2"
    assert exp16_config.lp_probability_covariates == exp16_config.covariates
    assert exp16_config.candidate_selection_mode == "all_triggered"
    assert exp16_config.skip_lp_intervention_diagnostics is True
    assert exp16_config.require_observed_gender is True


def test_exp25_forwards_observed_gender_filter_to_osrct_generator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The complete-case restriction must precede OSRCT standardization."""

    received: dict[str, object] = {}

    def fake_generator(**kwargs: object) -> pd.DataFrame:
        """Record generator arguments without writing a full OSRCT grid."""

        received.update(kwargs)
        return pd.DataFrame({"success": [True]})

    monkeypatch.setattr(
        exp25,
        "_load_osrct_grid_generator",
        lambda: fake_generator,
    )
    config = exp25.Exp25Config(
        studies=("gainloss",),
        bias_covariates=("gender",),
        betas=(1.0,),
    )

    exp25.generate_replication_osrct_grid(
        config,
        seed=42,
        output_dir=tmp_path,
    )

    assert received["require_observed_gender"] is True


def test_persisted_naturalness_bundle_is_parquet_and_reusable(
    tmp_path: Path,
) -> None:
    """Exp16 CSV leaves should become complete reusable Parquet bundles."""

    pytest.importorskip("pyarrow")
    source = tmp_path / "scratch_leaf"
    source.mkdir()
    pd.DataFrame(
        {
            "study": ["gainloss"],
            "pattern": ["age"],
            "beta": [0.5],
            "target_site": ["pi"],
            "resp_age_bin": [0],
            "resp_gender_bin": [0],
            "resp_polideo_bin": [0],
            "lower_mean": [-0.2],
            "upper_mean": [0.3],
        }
    ).to_csv(source / "target_gp_posterior_summary.csv", index=False)
    pd.DataFrame(
        {
            "study": ["gainloss"],
            "site": ["lse"],
            "role": ["proper_training"],
            "target_site": ["pi"],
        }
    ).to_csv(source / "source_site_roles.csv", index=False)
    pd.DataFrame({"value": [1]}).to_csv(
        source / "stage1_diagnostics.csv", index=False
    )
    pd.DataFrame().to_csv(source / "run_errors.csv", index=False)
    run_index = pd.DataFrame(
        {
            "target_site": ["pi"],
            "pattern": ["age"],
            "beta": [0.5],
            "output_dir": [str(source)],
            "n_errors": [0],
        }
    )
    config = exp25.Exp25Config(
        studies=("gainloss",),
        bias_covariates=("age",),
        betas=(0.5,),
    )
    replication_dir = tmp_path / "replication=000"

    references = exp25._persist_naturalness_artifacts(
        run_index,
        config,
        replication_dir,
    )
    reloaded = exp25._load_persisted_naturalness_references(
        config,
        replication_dir,
    )

    assert references == reloaded
    assert references[("age", 0.5)].suffix == ".parquet"
    assert references[("age", 0.5)].exists()
    assert references[("age", 0.5)].with_name(
        "source_site_roles.parquet"
    ).exists()
    assert exp18._inspect_naturalness_reference(
        exp18.Exp18Config(
            naturalness_reference=str(references[("age", 0.5)])
        )
    ) == "exp16"
    assert (replication_dir / "naturalness_artifact_index.parquet").exists()
    assert (replication_dir / "naturalness_run_index.parquet").exists()
    artifact_index = pd.read_parquet(
        replication_dir / "naturalness_artifact_index.parquet"
    ).set_index("source_name")
    assert artifact_index.loc["run_errors.csv", "status"] == (
        "empty_source_skipped"
    )
    assert artifact_index.loc["run_errors.csv", "n_rows"] == 0
    assert not list(replication_dir.rglob("*.csv"))


def test_long_conversion_excludes_fallback_and_clips_blearner_support() -> None:
    """Effective bounds should retain raw diagnostics and explicit validity."""

    result = exp25.comparison_to_long_intervals(
        _comparison_fixture(),
        replication=0,
        seed=42,
        pattern="age",
        beta=0.5,
        config_fingerprint_value="fingerprint",
    ).set_index("method")

    assert result.loc["blearner", "raw_lower"] == -1.4
    assert result.loc["blearner", "lower"] == -1.0
    assert bool(result.loc["blearner", "support_clipped"])
    assert bool(result.loc["blearner", "is_valid"])
    assert not bool(result.loc["cb_ipw", "is_valid"])
    assert (
        result.loc["cb_ipw", "exclusion_reason"]
        == "full_range_fallback"
    )
    assert bool(result.loc["cb_dr", "is_valid"])


def test_exp16_posterior_draw_threads_preserve_sample_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serial and threaded posterior LP execution should be deterministic."""

    class Surface:
        is_fitted = True
        status = "fitted"

    class Grounder:
        c_surface_ = Surface()
        d_surface_ = Surface()

        def sample_tolerances(
            self,
            choices: pd.DataFrame,
            *,
            n_samples: int,
            random_seed: int,
            clip: bool,
        ) -> pd.DataFrame:
            """Return one selected row for each requested sample."""

            assert clip
            return pd.DataFrame(
                {
                    "sample_id": np.arange(n_samples),
                    "candidate": ["age"] * n_samples,
                }
            )

    def estimate(*args, probability_model_kwargs=None, **kwargs):
        """Return posterior seed as the observed-quantity stand-in."""

        return {
            "seed": probability_model_kwargs["posterior_sample_seed"]
        }

    def solve(observed, sampled_choices, **kwargs):
        """Return deterministic bounds keyed by posterior seed."""

        value = float(observed["seed"] % 1000) / 1000.0
        return pd.DataFrame(
            {
                "status": ["optimal"],
                "lower": [-value],
                "upper": [value],
            }
        )

    monkeypatch.setattr(
        target_pipeline,
        "estimate_target_observed_binary_quantities",
        estimate,
    )
    monkeypatch.setattr(
        target_pipeline,
        "solve_target_bounds_from_observed_binary_quantities",
        solve,
    )
    monkeypatch.setattr(
        target_pipeline,
        "complete_lp_strata",
        lambda bounds, target_idle, config: bounds.copy(),
    )
    monkeypatch.setattr(
        target_pipeline,
        "summarize_lp_posterior_bounds",
        lambda samples, covariates: samples.sort_values(
            "lp_posterior_sample_id"
        ).reset_index(drop=True),
    )
    base = target_pipeline.Exp16Config(
        fit_gp=True,
        lp_posterior_samples=4,
        lp_probability_backend="hierarchical_dirichlet_draws",
        lp_posterior_n_jobs=1,
    )
    target = pd.DataFrame(
        {
            "iv": [0, 1],
            "dv_binary": [0, 1],
            "resp_age_bin": [0, 1],
            "resp_gender_bin": [0, 1],
            "resp_polideo_bin": [0, 1],
        }
    )
    choices = pd.DataFrame({"candidate": ["age"]})

    serial = target_pipeline.solve_choices_lp_posterior_samples(
        target,
        choices,
        Grounder(),
        base,
        random_seed=101,
    )[1]
    threaded = target_pipeline.solve_choices_lp_posterior_samples(
        target,
        choices,
        Grounder(),
        replace(base, lp_posterior_n_jobs=3),
        random_seed=101,
    )[1]

    pd.testing.assert_frame_equal(serial, threaded)


def test_dummy_run_writes_two_parquets_and_thirty_three_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dummy integration must never invoke Exp16 or Exp18 model fitting."""

    pytest.importorskip("pyarrow")
    monkeypatch.setattr(
        exp25,
        "run_exp16",
        lambda *args, **kwargs: pytest.fail("Exp16 called in dummy mode"),
    )
    monkeypatch.setattr(
        exp25,
        "run_exp18",
        lambda *args, **kwargs: pytest.fail("Exp18 called in dummy mode"),
    )
    title_weights: list[str | None] = []
    violin_orientations: list[str | None] = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_violinplot = matplotlib.axes.Axes.violinplot

    def record_title(self, label, *args, **kwargs):
        """Record title weight while preserving Matplotlib behavior."""

        if str(label).startswith("$\\beta"):
            title_weights.append(kwargs.get("fontweight"))
        return original_set_title(self, label, *args, **kwargs)

    def record_violinplot(self, *args, **kwargs):
        """Record violin orientation while preserving Matplotlib behavior."""

        violin_orientations.append(kwargs.get("orientation"))
        return original_violinplot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", record_title)
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "violinplot",
        record_violinplot,
    )
    config = exp25.Exp25Config(
        n_replications=2,
        output_dir=str(tmp_path / "dummy"),
        dummy=True,
        verbose=False,
    )

    result = exp25.run_experiment(config)

    output_dir = exp25._output_leaf(config)
    assert (output_dir / exp25.EXP25_INTERVALS_FILENAME).exists()
    assert (output_dir / exp25.EXP25_DIAGNOSTICS_FILENAME).exists()
    assert len(result["figure_paths"]) == 33
    assert all(path.exists() for path in result["figure_paths"])
    assert any(
        "interval_width_distributions_side_by_side" in path.name
        for path in result["figure_paths"]
    )
    assert title_weights and set(title_weights) == {"bold"}
    assert violin_orientations
    assert set(violin_orientations) == {"horizontal"}
    assert len(result["intervals"]) == 2 * 3 * 4 * 3 * 8 * 4


def test_resume_rejects_different_fingerprint(tmp_path: Path) -> None:
    """Compact tables from another scientific config must not be resumed."""

    pytest.importorskip("pyarrow")
    first = exp25.Exp25Config(
        n_replications=1,
        output_dir=str(tmp_path),
        dummy=True,
        verbose=False,
    )
    exp25.run_dummy(first)
    second = replace(first, base_seed=99)

    with pytest.raises(ValueError, match="different Exp25 config"):
        exp25._load_existing_outputs(
            exp25._output_leaf(first),
            fingerprint=exp25.config_fingerprint(second),
            resume=True,
        )


def test_resume_discards_unpaired_first_commit(tmp_path: Path) -> None:
    """One orphaned first-commit table should be safely regenerated."""

    pytest.importorskip("pyarrow")
    config = exp25.Exp25Config(
        n_replications=1,
        output_dir=str(tmp_path),
        dummy=True,
        verbose=False,
    )
    fingerprint = exp25.config_fingerprint(config)
    intervals = exp25.generate_dummy_intervals(
        config,
        fingerprint=fingerprint,
    )
    orphan_path = tmp_path / exp25.EXP25_INTERVALS_FILENAME
    exp25._atomic_write_parquet(intervals, orphan_path)

    loaded_intervals, loaded_diagnostics = exp25._load_existing_outputs(
        tmp_path,
        fingerprint=fingerprint,
        resume=True,
    )

    assert loaded_intervals.empty
    assert loaded_diagnostics.empty
    assert not orphan_path.exists()
