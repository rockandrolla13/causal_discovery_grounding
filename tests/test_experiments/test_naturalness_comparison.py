"""Fairness, roster and withholding checks for the method-comparison experiments.

These run on CPU in seconds. They deliberately avoid TabICL, the external
B-Learner checkout and the R reference: what they lock down is the contract a
reviewer would challenge -- that the selector really is withheld, that no arm
can shrink its width by dropping cells, that the sensitivity diagnostic is
evaluation-only, and that the roster cannot be silently misconfigured.
"""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from experiments import naturalness_comparison as comparison
from experiments import exp28_twins_method_comparison as exp28


def toy_prepared(names, n=6000, seed=8):
    """Return six sites over the named binaries with separate oracle arrays.

    Shaped like prepared Twins data: both potential outcomes are known and no
    latent assignment mechanism is declared, so regimes are built by revealing
    an arm on the selector covariate.

    Args:
        names: Observed covariate names.
        n: Total rows, split evenly across six sites.
        seed: Draw seed.

    Returns:
        A prepared-dataset mapping shaped like the first-stage screen's output.
    """
    rng = np.random.default_rng(seed)
    covariates = rng.integers(0, 2, (n, len(names)), dtype=np.uint8)
    y0 = rng.binomial(1, .3, n).astype(np.uint8)
    y1 = rng.binomial(1, .2, n).astype(np.uint8)
    return dict(covariates=covariates, covariate_names=list(names),
                treatment=np.zeros(n, dtype=np.uint8), outcome=np.zeros(n, dtype=np.uint8),
                sites=np.repeat(np.arange(6), n // 6), potential=(y0, y1),
                responses=(np.full(n, .3), np.full(n, .2)), provenance={})


def test_only_twins_is_supported():
    """A dataset without a specification must fail at construction, not mid-run."""
    assert comparison.SUPPORTED_DATASETS == ("twins_reduced",)
    with pytest.raises(ValueError, match="supports only"):
        comparison.ComparisonConfig(datasets=("acs_employment_support",))
    with pytest.raises(ValueError, match="Unsupported dataset"):
        comparison.condition_covariates(comparison.ComparisonConfig(), "acs_employment_support")


def test_twins_selector_may_sit_outside_the_conditioning_set():
    """The withheld-confounder design lets the selection covariate sit outside the conditioning set."""
    config = comparison.ComparisonConfig(
        datasets=("twins_reduced",), twins_covariates=("birmon", "gestat10", "mager8"),
        twins_bias_covariate="meduc6", withheld_covariates=("meduc6",))
    covariates, bias = comparison.condition_covariates(config, "twins_reduced")
    assert bias == "meduc6"
    assert bias not in covariates


def test_withheld_selector_never_reaches_any_fitting_frame():
    """No method may observe the covariate the regime was selected on.

    This is the property the whole Twins experiment rests on: the selector acts
    as an unmeasured confounder only if it is absent from every learner frame.
    """
    names = ["birmon", "gestat10", "mager8", "meduc6"]
    prepared = toy_prepared(names)
    config = comparison.ComparisonConfig(
        datasets=("twins_reduced",), twins_covariates=("birmon", "gestat10", "mager8"),
        twins_bias_covariate="meduc6", withheld_covariates=("meduc6",), max_fit_rows=100)
    condition = comparison.prepare_condition(prepared, "twins_reduced", .5, 42, config)
    assert condition["covariates"] == ("birmon", "gestat10", "mager8")
    for regime in ("on", "idle"):
        assert "meduc6" not in condition[regime].columns
    assert "meduc6" not in condition["evaluation"]["0"]["frame"].columns


def test_selector_actually_drives_assignment():
    """Changing the selector must change the regimes, or it is not a confounder."""
    names = ["birmon", "gestat10", "mager8", "meduc6"]
    prepared = toy_prepared(names)
    base = comparison.ComparisonConfig(
        datasets=("twins_reduced",), twins_covariates=("birmon", "gestat10", "mager8"),
        twins_bias_covariate="meduc6", max_fit_rows=0)
    other = comparison.ComparisonConfig(
        datasets=("twins_reduced",), twins_covariates=("birmon", "gestat10", "mager8"),
        twins_bias_covariate="dtotord_min" if "dtotord_min" in names else "birmon", max_fit_rows=0)
    first = comparison.prepare_condition(prepared, "twins_reduced", 2., 42, base)
    second = comparison.prepare_condition(prepared, "twins_reduced", 2., 42, other)
    assert not first["idle"].reset_index(drop=True).equals(second["idle"].reset_index(drop=True))


def test_missing_selector_is_rejected_rather_than_ignored():
    """A selector absent from the pool must fail loudly, not silently fall back."""
    prepared = toy_prepared(["birmon", "gestat10", "mager8"])
    config = comparison.ComparisonConfig(
        datasets=("twins_reduced",), twins_covariates=("birmon", "gestat10", "mager8"),
        twins_bias_covariate="meduc6")
    with pytest.raises(ValueError, match="absent from"):
        comparison.prepare_condition(prepared, "twins_reduced", .5, 42, config)


def test_roster_validation_rejects_silent_misconfiguration():
    """Unknown arms, a missing Manski fallback and a bad CB source regime must all raise."""
    with pytest.raises(ValueError, match="Unknown per-cell methods"):
        comparison.ComparisonConfig(methods=("manski", "not_a_method"))
    with pytest.raises(ValueError, match="Manski"):
        comparison.ComparisonConfig(methods=("naturalness",))
    with pytest.raises(ValueError, match="cb_source_regime"):
        comparison.ComparisonConfig(cb_source_regime="both")


def test_oracle_gamma_is_evaluation_only_and_detects_ignorability():
    """The required Gamma is 1 exactly when assignment is ignorable in-cell."""
    frame = pd.DataFrame({"z": [0, 0, 1, 1]})
    ignorable = dict(frame=frame, propensity=np.array([.5, .5, .2, .2]))
    assert comparison.oracle_required_gamma(ignorable, ("z",)) == pytest.approx(1., abs=1e-9)
    confounded = dict(frame=frame, propensity=np.array([.05, .95, .2, .2]))
    assert comparison.oracle_required_gamma(confounded, ("z",)) > 10.


def test_both_blearner_arms_are_reported():
    """The exp(beta) arm and the oracle arm must both be in the roster.

    Reporting exp(beta) alone would let a narrow interval fitted under an
    assumption the data-generating process violates read as a win.
    """
    assert "blearner_exp_beta" in comparison.PER_CELL_METHODS
    assert "blearner_oracle" in comparison.PER_CELL_METHODS
    assert {"blearner_exp_beta", "blearner_oracle"} <= set(exp28.default_config().methods)


def test_experiments_declare_no_calibration_and_no_conformal():
    """The entry point may not reintroduce a calibrated or conformal endpoint."""
    config = exp28.default_config()
    covariates, _ = comparison.condition_covariates(config, config.datasets[0])
    inference = comparison.naturalness_config(config, covariates, seed=config.seeds[0])
    # skip_calibration is a declaration; the mechanism is that the calibrated
    # Exp16 driver is never called at all, which the source scan below locks.
    assert inference.skip_calibration is True
    assert inference.candidate_selection_mode == "all_triggered"
    # Any other uncertainty mode widens tolerances to a GP credible quantile,
    # which would stop the endpoints being raw plug-in bounds.
    assert inference.target_policy_uncertainty == "posterior_mean"
    source = (comparison.__file__, comparison.__file__)
    for path in source:
        text = open(path).read()
        assert "conformalize_lp_bounds" not in text
        assert "finite_sample_conformal_quantile" not in text


def test_exp28_refuses_a_selector_inside_the_conditioning_set(monkeypatch):
    """Editing the selector into the reporting cells must fail, not pass quietly."""
    monkeypatch.setattr(exp28, "EXP28_BIAS_COVARIATE", "birmon")
    with pytest.raises(ValueError, match="outside the conditioning set"):
        exp28.default_config()


def test_completion_retains_every_cell_for_a_failed_arm():
    """A failed fit keeps the full target population at Manski, never drops it."""
    baseline = pd.DataFrame(dict(z=[0, 1], lower=[-1., -1.], upper=[1., 1.],
                                 n_fit=[10, 10], n_fit_arm0=[5, 5], n_fit_arm1=[5, 5],
                                 common_supported=[True, True]))
    completed = comparison._complete_bounds(pd.DataFrame(), baseline, ("z",), failure="fit_failed")
    assert len(completed) == len(baseline)
    assert completed.used_manski_fallback.all()
    assert (completed.lower == -1.).all() and (completed.upper == 1.).all()


def test_naturalness_comparisons_report_only_cb_ipw_and_cb_dr():
    """Exp28 compares against CB-IPW and CB-DR, never CB-PI or CB-RA.

    The shared ``PER_CELL_METHODS`` and ``DEFAULT_META_LEARNERS`` deliberately
    keep all four: they back the subset validator, and Exp18 exists to compare
    the CB learners against one another. The narrowing belongs to the entry
    point, so it is pinned here rather than in the shared definitions.
    """
    config = exp28.default_config()
    assert config.meta_learners == ("cb_ipw", "cb_dr")
    assert "cb_pi" not in config.methods and "cb_ra" not in config.methods
    # The non-CB arms must survive the narrowing untouched.
    assert {"manski", "naturalness", "blearner_exp_beta",
            "blearner_oracle"} <= set(config.methods)
    assert set(comparison.PER_CELL_METHODS) >= {"cb_pi", "cb_ra"}


def test_balke_pearl_is_a_reported_arm():
    """The hard-constraint IV reference must be in the roster.

    It uses strictly less information than Naturalness -- target observational
    rows only, no source trials, no transported tolerance -- so it is the arm
    that decides whether the grounding machinery pays for itself.
    """
    assert "balke_pearl" in exp28.default_config().methods
    assert comparison.METHOD_LABELS["balke_pearl"] == "Balke-Pearl IV"


def test_balke_pearl_arm_screens_candidates_on_relevance():
    """An irrelevant candidate must not be used as an instrument.

    The screen reuses the Naturalness ``D_j(tau)`` statistic so both arms pick
    candidates from the same target data.
    """
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({
        "good": rng.integers(0, 2, n),
        "useless": rng.integers(0, 2, n),
    })
    # Treatment responds only to "good"; "useless" has no relevance at all.
    frame["iv"] = (rng.random(n) < np.where(frame.good == 1, 0.8, 0.2)).astype(int)
    frame["dv_binary"] = (rng.random(n) < np.where(frame.iv == 1, 0.6, 0.4)).astype(int)
    covariates = ["good", "useless"]
    strata = frame[covariates].drop_duplicates().reset_index(drop=True)

    out = comparison.balke_pearl_bounds(frame, strata, covariates, min_count=20, tau=0.05)
    assert len(out) == len(strata)
    # Exactly one candidate clears relevance, so exactly one contributes.
    assert set(out.bp_candidates.unique()) == {1}
    assert (out.upper >= out.lower).all()
    assert set(out.bp_status.unique()) <= {"sharp", "infeasible", "unsupported"}


# ---------------------------------------------------------------------------
# Shared fitting primitives: outcome-blind fitting, the Manski fallback, the
# posterior-draw denominator and the scoring rule. These are the parts every
# arm passes through, so a regression here silently changes every reported
# width at once.
# ---------------------------------------------------------------------------

def test_response_truth_never_enters_fit_frames_and_cap_is_outcome_blind() -> None:
    """Oracle response changes preserve fits, and potential changes preserve IDs."""
    prepared = toy_prepared(("birmon", "gestat10", "mager8"))
    config = comparison.ComparisonBaseConfig(max_fit_rows=100)
    first = comparison.prepare_condition(prepared, "twins_reduced", 0.5, 42, config)
    changed = dict(prepared, responses=(np.zeros(6000), np.ones(6000)))
    second = comparison.prepare_condition(changed, "twins_reduced", 0.5, 42, config)
    pd.testing.assert_frame_equal(first["on"], second["on"])
    pd.testing.assert_frame_equal(first["idle"], second["idle"])

    changed = dict(prepared, potential=(np.ones(6000), np.zeros(6000)))
    third = comparison.prepare_condition(changed, "twins_reduced", 0.5, 42, config)
    pd.testing.assert_frame_equal(first["row_audit"], third["row_audit"])
    assert set(first["idle"].columns) == {
        "birmon", "gestat10", "mager8", "id", "site", "iv", "dv_binary", "F"
    }
    assert not set(first["on"].id) & set(first["idle"].id)


def test_latent_assignment_mechanism_is_refused_rather_than_ignored() -> None:
    """A declared latent propensity has no model here, so it must not be dropped silently."""
    prepared = dict(toy_prepared(("birmon", "gestat10", "mager8")),
                    assignment=(np.zeros(6000), np.ones(6000)))
    with pytest.raises(ValueError, match="latent assignment"):
        comparison.prepare_condition(prepared, "twins_reduced", 0.5, 42, comparison.ComparisonBaseConfig())


def test_manski_absent_arm_is_retained() -> None:
    """One-arm and missing cells retain their unobserved outcome mass."""
    target = pd.DataFrame({"z": [0, 0], "iv": [1, 1], "dv_binary": [1, 0]})
    result = comparison.manski_bounds(
        target, pd.DataFrame({"z": [0, 1]}), ("z",), 1
    )
    np.testing.assert_allclose(result.lower, [-0.5, -1.0])
    np.testing.assert_allclose(result.upper, [0.5, 1.0])
    assert not result.common_supported.any()

    raw = pd.DataFrame({"z": [0, 1], "lower": [0.1, 0.1], "upper": [0.2, 0.2]})
    completed = comparison._complete_bounds(raw, result, ("z",))
    np.testing.assert_allclose(completed.lower, result.lower)
    assert completed.used_manski_fallback.all()


def test_failed_and_missing_posterior_draws_keep_requested_denominator() -> None:
    """Optimal, failed, and absent draws all count in requested denominator."""
    baseline = pd.DataFrame(
        {"z": [0], "lower": [-0.4], "upper": [0.6], "common_supported": [True]}
    )
    samples = pd.DataFrame(
        {
            "z": [0, 0],
            "lower": [-0.1, -1.0],
            "upper": [0.1, 1.0],
            "status": ["optimal", "infeasible"],
            "lp_posterior_sample_id": [0, 1],
        }
    )
    summary, completed = comparison.posterior_bounds(samples, baseline, ("z",), 3)
    assert len(completed) == 3
    assert summary.n_requested_draws.iloc[0] == 3
    assert summary.lower.iloc[0] == pytest.approx(-0.3)
    assert summary.upper.iloc[0] == pytest.approx(1.3 / 3)
    assert summary.fallback_draw_fraction.iloc[0] == pytest.approx(2 / 3)
    assert summary.optimal_draw_fraction.iloc[0] == pytest.approx(1 / 3)


def test_scoring_retains_unsupported_population_mass() -> None:
    """Weighted scores include unsupported cells and report wrong signs."""
    baseline = pd.DataFrame(
        {
            "z": [0, 1],
            "lower": [-0.5, -0.5],
            "upper": [0.5, 0.5],
            "common_supported": [True, False],
        }
    )
    raw = pd.DataFrame({"z": [0, 1], "lower": [0.1, 0.1], "upper": [0.2, 0.2]})
    bounds = comparison._complete_bounds(raw, baseline, ("z",))
    truth = pd.DataFrame(
        {
            "z": [0, 1],
            "reference_effect": [-0.1, -0.1],
            "paired_effect": [-0.1, -0.1],
            "n_evaluation": [10, 90],
            "evaluation_weight": [0.1, 0.9],
        }
    )
    scored, metric = comparison.score_bounds(bounds, truth, ("z",), "naturalness")
    assert len(scored) == 2
    assert metric["n_evaluation"] == 100
    assert metric["weighted_width"] == pytest.approx(0.91)
    assert metric["reference_containment_mass"] == pytest.approx(0.9)
    assert metric["common_supported_mass"] == pytest.approx(0.1)
    assert metric["wrong_sign_decision_mass"] == pytest.approx(0.1)


def test_config_expansion_is_explicit() -> None:
    """All targets and independent draws require explicit settings."""
    default = comparison.ComparisonBaseConfig()
    assert default.target_sites == ("0",)
    assert len(default.dgp_seeds) == len(default.seeds) == 1
    expanded = replace(
        default,
        target_sites=("all",),
        dgp_seeds=(20260918, 20260928, 20261008),
    )
    assert expanded.target_sites == ("all",)
    with pytest.raises(ValueError):
        replace(default, betas=(0.5, 0.5))


def test_fitted_bounds_are_intersected_with_manski() -> None:
    """A fitted interval may refine the no-assumption bound but never exceed it.

    Four cells cover the cases: an overshoot on both sides is cut back to
    Manski, an interval inside Manski is left alone, an interval disjoint from
    Manski contradicts the data and falls back, and a failed endpoint stays
    failed rather than being turned into a valid fit by the cap.
    """
    baseline = pd.DataFrame({"z": [0, 1, 2, 3], "lower": [-.5] * 4, "upper": [.5] * 4,
                             "n_fit": [100] * 4, "n_fit_arm0": [50] * 4,
                             "n_fit_arm1": [50] * 4, "common_supported": [True] * 4})
    raw = pd.DataFrame({"z": [0, 1, 2, 3], "lower": [-.8, -.1, .6, np.nan],
                        "upper": [.7, .2, .9, .1], "status": ["optimal"] * 4})
    capped = comparison.intersect_with_manski(raw, baseline, ("z",))
    np.testing.assert_allclose(capped.lower[:2], [-.5, -.1])
    np.testing.assert_allclose(capped.upper[:2], [.5, .2])
    np.testing.assert_allclose(capped.uncapped_lower[:3], [-.8, -.1, .6])
    np.testing.assert_allclose(capped.uncapped_upper, [.7, .2, .9, .1])
    assert capped.manski_capped.tolist() == [True, False, True, False]
    assert capped.status.tolist() == ["optimal", "optimal", "manski_disjoint", "optimal"]
    completed = comparison._complete_bounds(capped, baseline, ("z",))
    assert completed.used_manski_fallback.tolist() == [False, False, True, True]
    np.testing.assert_allclose(completed.upper - completed.lower, [1., .3, 1., 1.])


def test_the_runner_caps_both_blearner_arms_before_scoring(tmp_path, monkeypatch) -> None:
    """Both B-Learner arms leave the runner inside Manski, with the fit kept.

    The fit is stubbed with the widest interval an effect can have, which is
    what the pseudo-outcome regressions approach at the large oracle Gamma of a
    strongly confounded condition; the completed arm must come back as Manski.
    """
    # _per_cell_bounds writes its raw frames through experiments.benchmarks,
    # which imports the full modelling stack at module level.
    pytest.importorskip("catenets")
    strata = pd.DataFrame({"z": [0, 1]})
    target_idle = pd.DataFrame({"z": [0, 0, 1, 1] * 10, "iv": [0, 1] * 20,
                                "dv_binary": [0, 1, 1, 0] * 10})
    baseline = comparison.manski_bounds(target_idle, strata, ("z",), 1)

    def widest(target_idle, strata, covariates, gamma, *args):
        """Stand in for the B-Learner with an interval spanning [-1, 1]."""
        return strata.assign(lower=-1., upper=1., status="optimal", gamma=gamma)

    monkeypatch.setattr(comparison, "blearner_bounds_at_gamma", widest)
    config = replace(exp28.default_config(),
                     methods=("manski", "blearner_exp_beta", "blearner_oracle"))
    evaluation = dict(frame=pd.DataFrame({"z": [0, 0, 1, 1]}),
                      propensity=np.array([.4, .6, .4, .6]))
    bounds, errors, _, _ = comparison._per_cell_bounds(
        {}, {}, target_idle, baseline, strata, evaluation, ("z",), "0", 1., 1, config, tmp_path)
    assert not errors
    for arm in comparison.MANSKI_CAPPED_METHODS:
        np.testing.assert_allclose(bounds[arm].lower, baseline.lower)
        np.testing.assert_allclose(bounds[arm].upper, baseline.upper)
        written = pd.read_parquet(tmp_path / f"{arm}_raw.parquet")
        assert written.uncapped_lower.eq(-1.).all() and written.uncapped_upper.eq(1.).all()


def test_parallel_grid_matches_the_serial_grid(tmp_path, monkeypatch) -> None:
    """``grid_n_jobs=2`` gives the serial grid's metrics, bounds and row order.

    The dataset is prepared in the parent process, so the stub reaches the spawned
    workers inside each job. Only ``fit_seconds``, a wall-clock time, may differ.
    """
    # run_comparison writes through experiments.benchmarks, which imports catenets.
    pytest.importorskip("catenets")
    names = ["birmon", "gestat10", "mager8", "meduc6"]
    monkeypatch.setattr(comparison.data, "prepare_dataset", lambda *args, **kwargs: toy_prepared(names))
    data_root = tmp_path / "data"
    (data_root / "twins").mkdir(parents=True)
    for letter in ("X", "T", "Y"):
        (data_root / "twins" / f"twin_pairs_{letter}_3years_samesex.csv").write_text(letter)
    base = comparison.ComparisonConfig(
        data_root=str(data_root), seeds=(1, 2), betas=(.5, 1.), target_sites=("0", "3"),
        twins_bias_covariate="meduc6", withheld_covariates=("meduc6",), max_fit_rows=400,
        methods=("manski", "balke_pearl"), meta_learners=(), min_cell_count=5)
    runs = {}
    for n_jobs in (1, 2):
        output = tmp_path / f"grid{n_jobs}"
        runs[n_jobs] = comparison.run_comparison(
            replace(base, grid_n_jobs=n_jobs, output_dir=str(output))), output
    serial, parallel = (runs[n][0].drop(columns="fit_seconds") for n in (1, 2))
    assert len(serial) == 2 * 2 * 2 * 2
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    leaves = sorted(p.relative_to(runs[1][1]) for p in runs[1][1].rglob("bounds_scored.parquet"))
    assert len(leaves) == 8
    for leaf in leaves:
        pd.testing.assert_frame_equal(pd.read_parquet(runs[1][1] / leaf),
                                      pd.read_parquet(runs[2][1] / leaf), check_exact=True)
    completion = [(json.loads((runs[n][1] / "completion.json").read_text())) for n in (1, 2)]
    assert completion[0] == completion[1]
