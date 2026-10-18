"""Tests for the Exp31 Bayesian Naturalness experiment on ManyLabs.

The numbered tests cover the experiment's fifteen required properties.  The
reference is the full target RCT, so test 1 checks an audited overlap and that
no target randomized row reaches a fitting stage.

Every fit uses a fixed stand-in for NUTS and a deterministic stand-in for the
TabICLv2 classifier (no model weights are loaded), on a small synthetic
ManyLabs-shaped pickle run through the real OSRCT generator and loaders, so no
number produced here is an Exp31 result.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_grounding import balke_pearl
from causal_grounding import bayesian_modular_naturalness as bayes
from causal_grounding import predictors as predictors_module
from causal_grounding.bayesian_modular_naturalness import (
    ToleranceSurfacePosterior,
    draw_random_identified_sets,
    draw_raw_pmfs,
    draw_tabicl_bootstrap_pmfs,
    raw_pmf_draws_from_frame,
    raw_pmf_draws_to_frame,
)
from experiments import exp31_bayesian_manylabs_naturalness as exp31


SMOKE = exp31.ConditionKey("allowedforbidden", "age", 1.0, 0)
FAST_DRAWS = 20


class _FakeTabICLClassifier:
    """Deterministic TabICLv2 stand-in whose predictions follow the fitted rows."""

    def __init__(self, **kwargs):
        """Record the constructor options TabICLv2 would receive."""

        self.kwargs = kwargs
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """Store the target rate and the feature means of the positive class."""

        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.rate_ = float(y.mean())
        self.lean_ = X[y == 1].mean(axis=0) - X.mean(axis=0) if (y == 1).any() else np.zeros(X.shape[1])
        return self

    def predict_proba(self, X):
        """Return bounded probabilities that move with the resampled rows."""

        signal = np.asarray(X, dtype=float) @ self.lean_
        probability_one = np.clip(self.rate_ + 0.5 * signal, 0.02, 0.98)
        return np.column_stack([1.0 - probability_one, probability_one])


def _patch_tabicl(patch: pytest.MonkeyPatch) -> None:
    """Route every TabICLv2 predictor to the stand-in classifier."""

    patch.setattr(predictors_module, "_TABICL_AVAILABLE", True)
    patch.setattr(predictors_module, "TabICLClassifier", _FakeTabICLClassifier)


@pytest.fixture
def fake_tabicl(monkeypatch):
    """Replace the TabICLv2 classifier for one test."""

    _patch_tabicl(monkeypatch)


def _manylabs_rows(seed: int = 0) -> pd.DataFrame:
    """Build a small ManyLabs-shaped RCT table for every study in the run.

    The target site ``pi`` and four sources carry the raw columns the OSRCT
    generator and the ManyLabs loaders read.  One row per site has a missing
    original ``resp_sex`` so the complete-case rule has something to remove.
    """

    rng = np.random.default_rng(seed)
    rows = []
    participant = 1000
    for study in exp31.ALL_STUDIES:
        for site, n_rows in (("pi", 400), ("s1", 120), ("s2", 120), ("s3", 120), ("s4", 120)):
            for index in range(n_rows):
                participant += 1
                rows.append({
                    "id": float(participant),
                    "site": site,
                    "original_study": study,
                    "iv": float(rng.integers(2)),
                    "dv": float(rng.integers(2)),
                    "resp_age": float(rng.choice([19.0, 20.0, 25.0, 40.0])),
                    "resp_gender": float(rng.integers(2)),
                    "resp_polideo": float(rng.integers(0, 7)),
                    "resp_sex": None if index == 0 else "female",
                })
    return pd.DataFrame(rows)


def _require_exp25_stack() -> None:
    """Skip where the Exp25 driver's benchmark stack is absent.

    ``prepare_inputs`` generates the OSRCT grids with the Exp25 driver, which
    reaches experiments.benchmarks, which imports torch, catenets and
    sklearn-quantile. Those live only in requirements.txt, so the
    dependency-light CI job skips the tests that need them and runs the rest.
    """
    pytest.importorskip("torch")
    pytest.importorskip("catenets")
    pytest.importorskip("sklearn_quantile")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> tuple[Path, Path]:
    """Prepare replications 0 and 1 of the fixture with the real generator.

    The pilot conditions' PMFs, which cover the smoke and leave-one-out
    conditions, are drawn beside the prepared root through the real
    ``draw-pmfs`` path with the TabICLv2 stand-in; :func:`_pmf_root` returns
    their root.
    """

    _require_exp25_stack()
    root = tmp_path_factory.mktemp("exp31")
    pickle = root / "manylabs_fixture.pkl"
    _manylabs_rows().to_pickle(pickle)
    exp31.prepare_inputs(root / "prepared", rct_path=pickle, replications=[0, 1])
    with pytest.MonkeyPatch.context() as patch:
        _patch_tabicl(patch)
        exp31.draw_pmfs(root / "pmf", root / "prepared", pickle, arm="pilot", device="cpu",
                        n_source_draws=FAST_DRAWS, n_target_draws=FAST_DRAWS)
    return root / "prepared", pickle


def _pmf_root(prepared) -> Path:
    """Return the fixture's PMF root, drawn by the ``prepared`` fixture."""

    return prepared[0].parent / "pmf"


def _unit_cube_surface(candidate: str, kind: str, imputation_id: str, latent: float = 0.0) -> ToleranceSurfacePosterior:
    """Return a fixed tolerance posterior whose support covers the unit cube."""

    features = np.asarray([[0.0], [1.0]]) if kind == "c" else np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return ToleranceSurfacePosterior(
        candidate=candidate,
        tolerance_kind=kind,
        feature_names=("i_deviation",) if kind == "c" else ("i_deviation", "d_strength"),
        source_features=features,
        source_omega=np.full(len(features), 0.01),
        posterior_samples={
            "eta": np.ones(20),
            "length_scale": np.full((20, features.shape[1]), 0.5),
            "u": np.full((20, len(features)), latent),
            "source_tolerance": np.full((20, len(features)), 0.5),
        },
        diagnostics={
            "passed": True, "attempt": "primary", "max_r_hat": 1.0, "min_bulk_ess": 500.0, "min_tail_ess": 500.0,
            "n_divergent": 0, "support_violations": 0, "num_chains": 1, "num_warmup": 1, "num_samples": 20,
            "target_accept_prob": 0.9, "n_evidence_units": len(features), "omega_clipped_count": 0,
        },
        imputation_id=imputation_id,
    )


def _fake_fit(evidence: pd.DataFrame, **kwargs) -> dict:
    """Stand in for NUTS with fixed unit-cube surfaces for every candidate.

    The surfaces report the primary budget they were asked for, as a real fit
    does, so the leaf validation of posterior-draw counts applies to them.
    """

    imputation_id = str(evidence.sample_id.iloc[0])
    surfaces = {(c, k): _unit_cube_surface(c, k, imputation_id) for c in evidence.candidate.unique() for k in ("c", "d")}
    for surface in surfaces.values():
        surface.diagnostics.update({name: kwargs[name] for name in ("num_chains", "num_warmup", "num_samples") if name in kwargs})
    return surfaces


@pytest.fixture
def fake_nuts(monkeypatch):
    """Replace every NUTS fit, in the driver and the module, by the stand-in."""

    monkeypatch.setattr(exp31, "fit_tolerance_surfaces", _fake_fit)
    monkeypatch.setattr(bayes, "fit_tolerance_surfaces", _fake_fit)


def _run_smoke(prepared, output: Path, **overrides) -> dict:
    """Run the one-condition smoke arm on the fixture with the stand-in fitter."""

    root, pickle = prepared
    settings = exp31.Exp31RunSettings(arm="smoke", imputation_counts=(FAST_DRAWS,), random_set_draws=FAST_DRAWS, **overrides)
    exp31.run_experiment(output, root, pickle, settings, pmf_root=_pmf_root(prepared), isolate_conditions=False)
    return json.loads((output / "identity.json").read_text(encoding="utf-8"))


# --- Design --------------------------------------------------------------------


def test_frozen_protocol_matches_the_plan() -> None:
    """The design constants are pinned and match Exp25's encodings."""

    _require_exp25_stack()
    from experiments.exp25_osrct_mc_baseline_comparison import EXP25_STRATUM_COLUMNS
    from experiments.exp16_osrct_calibrated_policy import Exp16Config

    assert exp31.SEEDS == tuple(range(62, 82))
    assert not set(exp31.SEEDS) & set(range(202608, 202628)) and not set(exp31.SEEDS) & set(range(42, 62))
    assert exp31.STUDIES == ("allowedforbidden", "gainloss")
    assert exp31.DEFERRED_STUDIES == ("reciprocity",)
    assert exp31.ALL_STUDIES == exp31.STUDIES + exp31.DEFERRED_STUDIES
    assert exp31.protocol_payload()["deferred_studies"]["studies"] == ["reciprocity"]
    assert exp31.PILOT_CONDITIONS == tuple(c for c in exp31.PILOT_DESIGN if c[0] != "reciprocity")
    assert len(exp31.PILOT_DESIGN) == 6 and len(exp31.PILOT_CONDITIONS) == 4
    assert exp31.PATTERNS == ("age", "gender", "polideo")
    assert exp31.BETAS == (0.5, 1.0, 1.5, 2.0)
    assert {beta for *_, beta, _ in exp31.PILOT_DESIGN} == {0.5, 1.0}
    assert exp31.PILOT_CONDITIONS[:2] == (("allowedforbidden", "age", 1.0, 0), ("gainloss", "gender", 1.0, 0))
    assert exp31.COVARIATES == tuple(EXP25_STRATUM_COLUMNS)
    assert exp31.TRIGGER_EPSILONS == tuple(Exp16Config().epsilons)
    assert exp31.TRIGGER_TAUS == tuple(Exp16Config().taus)
    # The smoke, pilot, loso and primary arms, plus the modular, predictive and loso_grid arms.
    assert set(exp31.ARMS) == {"smoke", "pilot", "loso", "primary", "modular", "predictive", "loso_grid"}
    assert exp31.ARMS["loso_grid"] == {"conditions": None, "needs_pilot_decision": False}
    assert exp31.LOSO_ARMS == {"loso", "loso_grid"} and exp31.PREDICTIVE_ROOT_ARMS == {"predictive", "loso_grid"}
    assert exp31.ARMS["modular"] == {"conditions": None, "needs_pilot_decision": False}
    assert exp31.ARMS["predictive"] == {"conditions": None, "needs_pilot_decision": False}
    assert exp31.UNGATED_ARMS == {"modular", "predictive"}
    assert exp31.PMF_MODEL == "tabicl_v2_stratified_bayesian_bootstrap"
    assert (exp31.SOURCE_PMF_DRAWS, exp31.TARGET_PMF_DRAWS) == (exp31.ESCALATED_IMPUTATIONS, exp31.RANDOM_SET_DRAWS)
    assert exp31.protocol_payload()["module_a"]["tabicl"]["checkpoint"] == predictors_module.TABICL_V2_CLASSIFIER_CHECKPOINT
    assert exp31.TABICL_PREDICTOR_KWARGS["kv_cache"] == "repr"
    assert not {"prior_eligibility_shift", "prior_endpoint_shift"} & set(exp31.CLASSIFICATION_RULES)
    assert (exp31.PRIMARY_IMPUTATIONS, exp31.RANDOM_SET_DRAWS, exp31.RCT_MIN_ARM_COUNT) == (20, 2000, 8)
    assert exp31.REQUIRE_OBSERVED_GENDER and exp31.REQUIRE_SOURCE_SUPPORT
    assert exp31.SMOKE_CONDITIONS == (("allowedforbidden", "age", 1.0, 0),)
    assert {study for study, *_ in exp31.PILOT_CONDITIONS} == set(exp31.STUDIES)
    assert len(exp31.all_conditions()) == 2 * 3 * 4 * 20
    assert not {key.study for key in exp31.all_conditions()} & set(exp31.DEFERRED_STUDIES)
    payload = exp31.protocol_payload()
    assert payload["feasibility_gate"] is None
    assert payload["candidate_roles"] == {}
    assert payload["reference"]["rows"] == "full_target_rct"


def test_a_deferred_study_is_selected_without_moving_the_frozen_identities(prepared, tmp_path) -> None:
    """``--studies`` prepares and runs a study outside the default set on its own; the defaults are unchanged.

    The study is prepared into a root of its own with the same seeds, patterns
    and betas; a run selects it through its settings, which record the selection
    only when it is made, so the identities of runs without it keep their
    fingerprints.  Fixed-condition arms and unknown studies refuse it.
    """

    root, pickle = prepared
    assert {key.study for key in exp31.all_conditions(studies=("reciprocity",))} == {"reciprocity"}
    assert len(exp31.all_conditions(studies=("reciprocity",))) == 3 * 4 * 20
    assert exp31.all_conditions() == exp31.all_conditions(studies=exp31.STUDIES)
    assert exp31.read_prepared_identity(root)["design"]["studies"] == list(exp31.STUDIES)
    deferred = exp31.prepare_inputs(tmp_path / "prepared_reciprocity", rct_path=pickle, replications=[0],
                                    studies=("reciprocity",))
    assert deferred["design"]["studies"] == ["reciprocity"] and deferred["design"]["seeds"] == [exp31.BASE_SEED]
    assert {row["study"] for row in deferred["inputs"]} == {"reciprocity"} and len(deferred["inputs"]) == 3 * 4
    with pytest.raises(ValueError, match="distinct values"):
        exp31.prepare_inputs(tmp_path / "prepared_unknown", rct_path=pickle, replications=[0], studies=("nope",))

    selected = exp31.Exp31RunSettings(arm="predictive", imputation_counts=(20,), studies=("reciprocity",))
    selected.validate()
    assert exp31._settings_payload(selected)["studies"] == ["reciprocity"]
    assert "studies" not in exp31._settings_payload(dataclasses.replace(selected, studies=None))
    with pytest.raises(ValueError, match="frozen conditions"):
        exp31.Exp31RunSettings(arm="smoke", imputation_counts=(1,), studies=("reciprocity",)).validate()
    with pytest.raises(ValueError, match="distinct values"):
        exp31.Exp31RunSettings(arm="predictive", imputation_counts=(20,), studies=("nope",)).validate()
    parser = exp31.build_parser()
    common = ["--prepared-root", "p", "--output-dir", "o"]
    assert parser.parse_args(["run", "--arm", "predictive", "--pmf-root", "m", *common,
                              "--studies", "reciprocity"]).studies == ["reciprocity"]
    assert parser.parse_args(["draw-pmfs", "--arm", "primary", *common, "--studies", "reciprocity"]).studies == ["reciprocity"]
    assert parser.parse_args(["prepare", "--output-dir", "o", "--studies", "reciprocity"]).studies == ["reciprocity"]


def test_cli_has_the_documented_commands() -> None:
    """The public workflow is the eight documented commands."""

    parser = exp31.build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert set(subparsers.choices) == {"prepare", "draw-pmfs", "adopt-pmfs", "run", "finalize", "balke-pearl", "regrid",
                                       "report"}


def test_settings_reject_protocol_violations() -> None:
    """Arms keep their fixed conditions, counts and divisible draw counts."""

    with pytest.raises(ValueError, match="frozen conditions"):
        exp31.Exp31RunSettings(arm="pilot", imputation_counts=(10, 20), replications=(0,)).validate()
    with pytest.raises(ValueError, match="pilot compares"):
        exp31.Exp31RunSettings(arm="pilot", imputation_counts=(20,)).validate()
    with pytest.raises(ValueError, match="divisible"):
        exp31.Exp31RunSettings(arm="primary", imputation_counts=(20,), random_set_draws=30).validate()
    exp31.Exp31RunSettings(arm="primary", imputation_counts=(20,), replications=(0, 3)).validate()


# --- Required properties ---------------------------------------------------------


def test_01_reference_is_full_target_rct_and_its_overlap_is_audited(prepared) -> None:
    """(1) Target idle and reference roles: full-RCT reference, audited overlap.

    The reference is the full complete-case target RCT, so no target randomized
    row may reach a fitting stage and the overlap must be exactly the
    OSRCT-selected target participants.
    """

    root, pickle = prepared
    data = exp31.load_condition_data(SMOKE, root, pickle)
    raw = pd.read_pickle(pickle)
    cohort = raw.loc[raw.original_study.eq(SMOKE.study) & raw.site.eq("pi") & raw.resp_sex.notna()]
    assert set(data.reference.id) == set(exp31._participant_ids(cohort.id))
    assert set(data.target_idle.id) <= set(data.reference.id)
    assert data.counts["n_reference_rows_shared_with_target_idle"] == len(data.target_idle)
    assert int(data.audit.shared_with_target_idle.sum()) == len(data.target_idle)
    assert "pi" not in set(data.sources.site)
    assert set(data.target_idle.F) == {"idle"}
    assert "F" not in data.reference


def test_02_role_counts_match_the_saved_audit(prepared, fake_nuts, tmp_path) -> None:
    """(2) Source, target idle and reference row counts equal the saved audit."""

    _run_smoke(prepared, tmp_path / "run")
    leaf = SMOKE.leaf(tmp_path / "run")
    audit = pd.read_parquet(leaf / "data_role_audit.parquet")
    counts = json.loads((leaf / "completion.json").read_text())["counts"]
    roles = audit.role.astype(str).value_counts()
    assert roles["source_randomized"] == counts["n_source_randomized_rows"]
    assert roles["source_idle"] == counts["n_source_idle_rows"]
    assert roles["target_idle"] == counts["n_target_idle_rows"]
    assert roles["target_reference"] == counts["n_target_reference_rows"]
    assert (audit.exclusion_reason.astype(str) == "resp_sex_missing").sum() == counts["n_excluded_resp_sex_missing"] == 5
    root, pickle = prepared
    data = exp31.load_condition_data(SMOKE, root, pickle)
    assert counts["n_source_randomized_rows"] == int(data.sources.F.eq("on").sum())
    assert counts["n_target_reference_rows"] == len(data.reference)


def _without_alpha(draw: bayes.SitePMFDraw) -> bayes.SitePMFDraw:
    """Return a draw comparable by value: a TabICLv2 draw's ``alpha`` is NaN."""

    return dataclasses.replace(draw, alpha=0.0)


def test_03_pmf_draws_lie_on_the_simplex_and_keep_the_observed_support(prepared, fake_tabicl) -> None:
    """(3) Every drawn table is a probability vector on the observed strata only.

    The TabICLv2 bootstrap draws keep each site's observed idle strata and
    physical counts, vary from draw to draw, and survive the stored table's
    round trip exactly.
    """

    root, pickle = prepared
    data = exp31.load_condition_data(SMOKE, root, pickle)
    for frame in (data.sources, data.target_idle):
        imputations = draw_tabicl_bootstrap_pmfs(frame, n_draws=3, random_seed=5, covariates=exp31.COVARIATES)
        for imputation in imputations:
            for draw in imputation.values():
                idle = frame.loc[frame.site.eq(draw.site_id) & frame.F.eq("idle")]
                observed = {tuple(int(v) for v in row) for row in idle[list(exp31.COVARIATES)].itertuples(index=False, name=None)}
                assert set(draw.z_values) == observed and set(draw.idle_z) == observed
                assert sum(draw.idle_z.values()) == pytest.approx(1.0)
                assert sum(draw.n_idle_z.values()) == len(idle)
                for z in draw.z_values:
                    cells = [draw.idle_xy_given_z[(z, x, y)] for x in (0, 1) for y in (0, 1)]
                    assert min(cells) > 0.0 and sum(cells) == pytest.approx(1.0)
                    for x in (0, 1):
                        if draw.do_y_given_xz:
                            assert sum(draw.do_y_given_xz[(z, x, y)] for y in (0, 1)) == pytest.approx(1.0)
        assert any(imputations[0][site].idle_xy_given_z != imputations[1][site].idle_xy_given_z for site in imputations[0])
        rebuilt = raw_pmf_draws_from_frame(raw_pmf_draws_to_frame(imputations), frame, covariates=exp31.COVARIATES)
        for original, copy in zip(imputations, rebuilt):
            assert {site: _without_alpha(draw) for site, draw in copy.items()} == {
                site: _without_alpha(draw) for site, draw in original.items()
            }


def test_04_nested_source_rows_are_resampled_as_physical_units(prepared, fake_tabicl) -> None:
    """(4) A nested site's idle resample is the selected part of its randomized resample.

    Every stratum keeps its idle and randomized counts, and each idle row is
    drawn exactly as often as its physical unit.
    """

    root, pickle = prepared
    data = exp31.load_condition_data(SMOKE, root, pickle)
    draws = draw_tabicl_bootstrap_pmfs(data.sources, n_draws=1, random_seed=3, covariates=exp31.COVARIATES)[0]
    assert all(draw.overlap_model for draw in draws.values())
    columns = list(exp31.COVARIATES)
    for site, rows in data.sources.groupby("site"):
        on, idle = rows.loc[rows.F.eq("on")], rows.loc[rows.F.eq("idle")]
        boot_on, boot_idle = bayes._bootstrap_site_rows(on, idle, covariates=columns, id_col="id", overlap=True,
                                                        rng=np.random.default_rng(0))
        assert set(boot_idle.id) <= set(idle.id) and set(boot_idle.id) <= set(boot_on.id)
        pd.testing.assert_series_equal(boot_idle.groupby(columns).size(), idle.groupby(columns).size())
        pd.testing.assert_series_equal(boot_on.groupby(columns).size(), on.groupby(columns).size())
        picks = boot_on.id.value_counts()
        assert boot_idle.id.value_counts().eq(picks.reindex(boot_idle.id.value_counts().index)).all()


def test_05_paired_naturalness_and_manski_use_the_same_target_draw(prepared, fake_nuts, tmp_path) -> None:
    """(5) Both methods of a draw share its sample ID and target PMF draw."""

    _run_smoke(prepared, tmp_path / "run")
    sets = pd.read_parquet(SMOKE.leaf(tmp_path / "run") / f"imputations={FAST_DRAWS}" / "random_identified_sets.parquet")
    by_draw = sets.groupby("draw_index")
    assert by_draw.sample_id.nunique().eq(1).all()
    assert by_draw.target_pmf_sample_id.nunique().eq(1).all()
    assert by_draw.target_pmf_draw_index.nunique().eq(1).all()
    assert set(sets.method) == set(exp31.METHODS)
    assert sets.groupby(["draw_index", "estimand_key"]).method.nunique().eq(2).all()


def _target_rows(covariates: tuple[str, ...] = ("z_clean", "z_bad"), strata: list[tuple[int, ...]] | None = None) -> pd.DataFrame:
    """Build fully supported target idle rows over the given binary covariates."""

    from itertools import product

    rows = []
    for z in strata or list(product((0, 1), repeat=len(covariates))):
        for treatment in (0, 1):
            for repetition in range(4):
                rows.append({
                    "id": f"t:{''.join(map(str, z))}{treatment}:{repetition}",
                    "site": "target",
                    "F": "idle",
                    "iv": treatment,
                    "dv_binary": int((treatment + sum(z) + repetition) % 2),
                    **dict(zip(covariates, z)),
                })
    return pd.DataFrame(rows)


def _module_target(n_draws: int, seed: int = 3):
    """Return two-candidate target PMF draws, as the Bayesian module tests use."""

    return [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=n_draws, random_seed=seed)]


def _module_mixture(latent: float = 0.0, far: bool = False) -> dict:
    """Return two-candidate surfaces, optionally with support far from [0, 1]."""

    mixture = {}
    for candidate in ("z_clean", "z_bad"):
        for kind in ("c", "d"):
            surface = _unit_cube_surface(candidate, kind, "imputation", latent=latent)
            if far:
                surface.source_features = surface.source_features + 2.0
            mixture[(candidate, kind)] = surface
    return mixture


def test_06_no_eligible_candidate_returns_the_matched_manski_set(fake_nuts) -> None:
    """(6) Without an eligible candidate Naturalness is Manski and is flagged."""

    mixture = _module_mixture()
    result = draw_random_identified_sets(
        _module_target(4), [mixture], n_draws=4, min_cell_count=1, epsilons=(0.0,), taus=(1.0,)
    )
    sets, triggers, _ = exp31.random_set_tables(result, [mixture])
    naturalness = sets.loc[sets.method.eq(exp31.NATURALNESS)].sort_values(["draw_index", "estimand_key"])
    manski = sets.loc[sets.method.eq(exp31.MANSKI)].sort_values(["draw_index", "estimand_key"])
    assert naturalness.no_candidate_eligible.all() and naturalness.used_manski_fallback.all()
    assert not naturalness.eligible_but_lp_infeasible.any()
    np.testing.assert_allclose(naturalness.lower, manski.lower)
    np.testing.assert_allclose(naturalness.upper, manski.upper)
    assert set(triggers.ineligibility_reason) == {"not_triggered"}


def test_07_eligible_but_infeasible_lp_stays_empty() -> None:
    """(7) Near-zero grounded tolerances empty the set; it never becomes Manski."""

    mixture = _module_mixture(latent=-8.0)
    result = draw_random_identified_sets(
        _module_target(6), [mixture], n_draws=6, min_cell_count=1, epsilons=(1.0,), taus=(0.0,)
    )
    sets, triggers, _ = exp31.random_set_tables(result, [mixture])
    naturalness = sets.loc[sets.method.eq(exp31.NATURALNESS)]
    assert triggers.eligible.all()
    assert naturalness.lp_outcome.eq("empty").all()
    assert not naturalness.nonempty.any() and naturalness.width.isna().all()
    assert naturalness.eligible_but_lp_infeasible.all()
    assert not naturalness.used_manski_fallback.any()
    assert naturalness.lower_status.eq("infeasible").all()
    assert sets.loc[sets.method.eq(exp31.MANSKI), "nonempty"].all()


def _metrics(prepared, tmp_path, fake=True) -> pd.DataFrame:
    """Run the smoke arm and return its condition metrics."""

    _run_smoke(prepared, tmp_path / "run")
    return pd.read_parquet(tmp_path / "run" / "condition_metrics.parquet")


def test_08_feasibility_uses_every_posterior_draw(prepared, fake_nuts, tmp_path) -> None:
    """(8) Feasibility divides by all draws, fallback and empty draws included."""

    metrics = _metrics(prepared, tmp_path)
    rows = metrics.loc[metrics.scope.eq("feasibility") & metrics.method.eq(exp31.NATURALNESS)].set_index("metric")
    assert rows.loc["p_nonempty", "denominator"] == FAST_DRAWS
    assert rows.loc["p_nonempty", "denominator_label"] == exp31.ALL_DRAWS
    nonempty = rows.loc["n_fallback_draws", "value"] + rows.loc["n_eligible_nonempty_draws", "value"]
    assert rows.loc["p_nonempty", "value"] == pytest.approx(nonempty / FAST_DRAWS)
    total = nonempty + rows.loc["n_eligible_but_lp_infeasible_draws", "value"] + rows.loc["n_solver_failure_draws", "value"]
    assert total == FAST_DRAWS


def test_09_conditional_width_summaries_expose_their_count(prepared, fake_nuts, tmp_path) -> None:
    """(9) Width and endpoint summaries carry the number of nonempty draws."""

    metrics = _metrics(prepared, tmp_path)
    naturalness = metrics.loc[metrics.method.eq(exp31.NATURALNESS)]
    n_nonempty = naturalness.loc[naturalness.metric.eq("n_nonempty_draws"), "value"].iloc[0]
    widths = naturalness.loc[naturalness.scope.eq("bounds") & naturalness.metric.str.startswith("width_")]
    assert not widths.empty
    assert widths.denominator.eq(n_nonempty).all()
    assert widths.denominator_label.eq(exp31.NONEMPTY).all()
    weighted = naturalness.loc[naturalness.metric.eq("weighted_width_mean")]
    assert weighted.denominator.iloc[0] == n_nonempty


def test_10_empty_sets_are_non_members_in_the_all_draw_summary() -> None:
    """(10) An empty set is a miss over all draws and absent when conditioning."""

    sets = pd.DataFrame({
        "draw_index": [0, 1], "method": exp31.NATURALNESS, "estimand_type": "cate", "estimand_key": "[0]",
        "lower": [-0.2, np.nan], "upper": [0.3, np.nan], "nonempty": [True, False],
    })
    references = pd.DataFrame({
        "reference_draw_index": [0, 1], "estimand_type": "cate", "estimand_key": "[0]", "reference_value": [0.1, 0.1],
    })
    scored = exp31.score_random_sets(sets, references)
    assert scored.reference_member.tolist() == [True, False]
    assert scored.reference_scoreable.all() and not scored.wrong_sign.any()
    assert scored.reference_member.mean() == 0.5
    assert scored.loc[sets.nonempty, "reference_member"].mean() == 1.0


def test_11_out_of_support_candidates_are_ineligible_with_a_reason() -> None:
    """(11) A triggered candidate outside the source hull is recorded and dropped."""

    mixture = _module_mixture(far=True)
    result = draw_random_identified_sets(
        _module_target(4), [mixture], n_draws=4, min_cell_count=1, epsilons=(1.0,), taus=(0.0,)
    )
    sets, triggers, tolerances = exp31.random_set_tables(result, [mixture])
    assert triggers.triggered.all() and not triggers.eligible.any()
    assert set(triggers.ineligibility_reason) == {"outside_source_support"}
    assert tolerances.empty
    assert sets.loc[sets.method.eq(exp31.NATURALNESS), "used_manski_fallback"].all()


def test_12_seeds_reproduce_identical_draw_ids_and_summaries(prepared, fake_nuts, tmp_path) -> None:
    """(12) Two runs of one condition give identical draws and metrics."""

    _run_smoke(prepared, tmp_path / "first")
    _run_smoke(prepared, tmp_path / "second")
    base = f"imputations={FAST_DRAWS}"
    for name in ("random_identified_sets.parquet", "target_trigger_draws.parquet", "scored_random_sets.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(SMOKE.leaf(tmp_path / "first") / base / name),
            pd.read_parquet(SMOKE.leaf(tmp_path / "second") / base / name),
        )
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "first" / "condition_metrics.parquet"),
        pd.read_parquet(tmp_path / "second" / "condition_metrics.parquet"),
    )


def test_13_resume_rejects_changed_data_or_configuration(prepared, fake_nuts, tmp_path) -> None:
    """(13) Resume is refused after a settings change or an input edit."""

    root, pickle = prepared
    pmf = _pmf_root(prepared)
    _run_smoke(prepared, tmp_path / "run")
    changed = exp31.Exp31RunSettings(arm="smoke", imputation_counts=(10,), random_set_draws=FAST_DRAWS)
    with pytest.raises(ValueError, match="identity differs"):
        exp31.initialize_run_root(tmp_path / "run", root, pickle, pmf, changed, resume=True)
    copy = tmp_path / "prepared_copy"
    shutil.copytree(root, copy)
    osrct = exp31._osrct_file(copy, SMOKE)
    frame = pd.read_parquet(osrct)
    frame.iloc[:-1].to_parquet(osrct, index=False)
    settings = exp31.Exp31RunSettings(arm="smoke", imputation_counts=(FAST_DRAWS,), random_set_draws=FAST_DRAWS)
    with pytest.raises(ValueError, match="changed after preparation"):
        exp31.initialize_run_root(tmp_path / "other", copy, pickle, pmf, settings, resume=False)


def _pilot_decision_file(path: Path, settings: exp31.Exp31RunSettings, count: int = 20,
                         pmf_fingerprint: str | None = None) -> Path:
    """Write a pilot decision file matching the given settings and PMF root."""

    payload = exp31._identity({
        "experiment": exp31.EXPERIMENT_NAME, "status": "frozen", "selected_num_imputations": count,
        "nuts_settings": settings.nuts_settings(), "random_set_draws": settings.random_set_draws,
        "pmf_fingerprint": pmf_fingerprint,
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_14_pmf_draws_can_be_neither_altered_nor_swapped(prepared, tmp_path) -> None:
    """(14) Altered or foreign PMF draws, and a run without every condition's draws, are refused."""

    root, pickle = prepared
    pmf = _pmf_root(prepared)
    settings = exp31.Exp31RunSettings(arm="smoke", imputation_counts=(FAST_DRAWS,), random_set_draws=FAST_DRAWS)
    identity = exp31.initialize_run_root(tmp_path / "run", root, pickle, pmf, settings, resume=False)
    assert identity["pmf_fingerprint"] == exp31.read_pmf_identity(pmf)["fingerprint"]
    data = exp31.load_condition_data(SMOKE, root, pickle)
    altered = tmp_path / "pmf_altered"
    shutil.copytree(pmf, altered)
    target = SMOKE.leaf(altered) / "target_pmf_draws.parquet"
    frame = pd.read_parquet(target)
    frame.loc[0, "probability"] = float(frame.loc[0, "probability"]) * 0.5
    frame.to_parquet(target, index=False)
    with pytest.raises(ValueError, match="changed after draw-pmfs"):
        exp31.load_condition_pmfs(SMOKE, altered, data, pmf_fingerprint=identity["pmf_fingerprint"],
                                  n_source_draws=FAST_DRAWS, n_target_draws=FAST_DRAWS)
    with pytest.raises(ValueError, match="another PMF root"):
        exp31.load_condition_pmfs(SMOKE, pmf, data, pmf_fingerprint="another-root", n_source_draws=FAST_DRAWS,
                                  n_target_draws=FAST_DRAWS)
    primary = exp31.Exp31RunSettings(arm="primary", imputation_counts=(20,), replications=(0,))
    foreign = _pilot_decision_file(tmp_path / "foreign.json", primary, pmf_fingerprint="another-root")
    with pytest.raises(ValueError, match="another draw-pmfs root"):
        exp31.initialize_run_root(tmp_path / "primary", root, pickle, pmf, primary, resume=False, pilot_decision=foreign)
    decision = _pilot_decision_file(tmp_path / "decision.json", primary, pmf_fingerprint=identity["pmf_fingerprint"])
    with pytest.raises(ValueError, match="no complete PMF draws"):
        exp31.initialize_run_root(tmp_path / "primary", root, pickle, pmf, primary, resume=False, pilot_decision=decision)


def test_adopted_pmf_draws_are_the_source_draws_under_the_new_root(prepared, fake_tabicl, tmp_path) -> None:
    """``adopt-pmfs`` copies only unchanged-input draws, never overwrites, and refuses other roots.

    A second prepared root from the same pickle reproduces the fixture's inputs
    exactly, as a new beta grid does for the betas it shares with the old one.
    """

    root, pickle = prepared
    old_pmf = _pmf_root(prepared)
    new_prepared = tmp_path / "prepared_new"
    exp31.prepare_inputs(new_prepared, rct_path=pickle, replications=[0, 1])
    new_pmf = tmp_path / "pmf_new"
    exp31.draw_pmfs(new_pmf, new_prepared, pickle, arm="smoke", device="cpu", n_source_draws=FAST_DRAWS,
                    n_target_draws=FAST_DRAWS)
    record = exp31.adopt_pmfs(new_pmf, new_prepared, source_root=old_pmf, source_prepared_root=root, arm="pilot")
    assert record["counts"] == {"adopted": 3, "already_in_target": 1, "not_complete_in_source": 0, "input_differs": 0}
    key = exp31.ConditionKey(*exp31.PILOT_CONDITIONS[1])
    for name in exp31.PMF_DATA_FILES:
        assert exp31._sha256(key.leaf(new_pmf) / name) == exp31._sha256(key.leaf(old_pmf) / name)
    completion = json.loads((key.leaf(new_pmf) / "completion.json").read_text())
    assert completion["adopted_from"]["pmf_fingerprint"] == exp31.read_pmf_identity(old_pmf)["fingerprint"]
    new_identity = exp31.read_pmf_identity(new_pmf)
    data = exp31.load_condition_data(key, new_prepared, pickle)
    sources, target, _ = exp31.load_condition_pmfs(key, new_pmf, data, pmf_fingerprint=new_identity["fingerprint"],
                                                    n_source_draws=FAST_DRAWS, n_target_draws=FAST_DRAWS)
    assert len(sources) == FAST_DRAWS and len(target) == FAST_DRAWS
    again = exp31.adopt_pmfs(new_pmf, new_prepared, source_root=old_pmf, source_prepared_root=root, arm="pilot")
    assert again["counts"]["adopted"] == 0 and again["counts"]["already_in_target"] == 4

    # A condition whose prepared input differs between the two roots is left to draw-pmfs.
    edited = tmp_path / "prepared_edited"
    shutil.copytree(root, edited)
    identity = json.loads((edited / "identity.json").read_text())
    row = next(row for row in identity["inputs"] if (row["study"], row["pattern"], float(row["beta"]), row["replication"])
               == tuple(exp31.PILOT_CONDITIONS[2]))
    row["sha256"] = "0" * 64
    (edited / "identity.json").write_text(json.dumps(identity))
    fresh = tmp_path / "pmf_fresh"
    exp31.draw_pmfs(fresh, new_prepared, pickle, arm="smoke", device="cpu", n_source_draws=FAST_DRAWS,
                    n_target_draws=FAST_DRAWS)
    record = exp31.adopt_pmfs(fresh, new_prepared, source_root=old_pmf, source_prepared_root=edited, arm="pilot")
    assert record["counts"]["adopted"] == 2 and record["counts"]["input_differs"] == 1

    # Roots drawn with other draw counts are not interchangeable.
    other = tmp_path / "pmf_other"
    exp31.draw_pmfs(other, new_prepared, pickle, arm="smoke", device="cpu", n_source_draws=FAST_DRAWS,
                    n_target_draws=FAST_DRAWS // 2)
    with pytest.raises(ValueError, match="n_target_draws"):
        exp31.adopt_pmfs(other, new_prepared, source_root=old_pmf, source_prepared_root=root, arm="pilot")


def test_15_tiny_end_to_end_fixture_writes_every_required_artifact(prepared, fake_nuts, tmp_path) -> None:
    """(15) One fixture run writes, validates and completes every artifact."""

    _run_smoke(prepared, tmp_path / "run")
    run = tmp_path / "run"
    for name in exp31.RUN_FILES:
        assert (run / name).is_file(), name
    leaf = SMOKE.leaf(run)
    for name in exp31.CONDITION_FILES + ("completion.json",):
        assert (leaf / name).is_file(), name
    for name in exp31.MIXTURE_FILES:
        assert (leaf / f"imputations={FAST_DRAWS}" / name).is_file(), name
    completion = json.loads((run / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["validated"]
    manifest = (run / "run_manifest.yml").read_text()
    for field in ("git", "source_fingerprints", "input_fingerprints", "row_counts", "frozen_seeds", "software_environment",
                  "schema_version"):
        assert field in manifest


# --- Further safeguards ----------------------------------------------------------


def test_finalize_refuses_a_root_with_a_missing_artifact(prepared, fake_nuts, tmp_path) -> None:
    """Completion is withheld and the problem reported when validation fails."""

    _run_smoke(prepared, tmp_path / "run")
    (tmp_path / "run" / "completion.json").unlink()
    (SMOKE.leaf(tmp_path / "run") / f"imputations={FAST_DRAWS}" / "scored_random_sets.parquet").unlink()
    with pytest.raises(RuntimeError, match="validation problems"):
        exp31.finalize_run(tmp_path / "run")
    assert not (tmp_path / "run" / "completion.json").exists()
    report = json.loads((tmp_path / "run" / "validation_report.json").read_text())
    assert any("scored_random_sets" in problem for problem in report["problems"])


def test_primary_arm_needs_a_matching_frozen_pilot_decision(prepared, tmp_path) -> None:
    """The primary run is refused without, or against, the pilot decision's settings."""

    root, pickle = prepared
    pmf = _pmf_root(prepared)
    settings = exp31.Exp31RunSettings(arm="primary", imputation_counts=(20,), replications=(0,))
    with pytest.raises(ValueError, match="pilot-decision"):
        exp31.initialize_run_root(tmp_path / "a", root, pickle, pmf, settings, resume=False)
    decision = _pilot_decision_file(tmp_path / "decision.json", settings, count=40)
    with pytest.raises(ValueError, match="froze 40"):
        exp31.initialize_run_root(tmp_path / "b", root, pickle, pmf, settings, resume=False, pilot_decision=decision)


def test_pilot_rule_escalates_only_above_the_threshold() -> None:
    """The imputation-count rule selects 20 at a 0.01 shift and 40 at a 0.03 shift."""

    settings = exp31.Exp31RunSettings(arm="pilot", imputation_counts=(10, 20))

    def metrics(shift: float) -> pd.DataFrame:
        """Return weighted endpoint rows for two imputation counts."""

        rows = []
        for count, offset in ((10, 0.0), (20, shift)):
            for metric in ("weighted_outer_lower", "weighted_outer_upper", "weighted_lower_q50", "weighted_upper_q50"):
                rows.append({"study": "gainloss", "pattern": "age", "beta": 1.0, "replication": 0, "num_imputations": count,
                             "scope": "weighted", "method": exp31.NATURALNESS, "estimand_type": "", "metric": metric,
                             "value": 0.1 + offset})
        return pd.DataFrame(rows)

    identity = {"fingerprint": "f", "pmf_fingerprint": "p"}
    assert exp31._pilot_decision(metrics(0.01), settings, identity)["selected_num_imputations"] == 20
    decision = exp31._pilot_decision(metrics(0.03), settings, identity)
    assert decision["selected_num_imputations"] == 40
    assert decision["maximum_shift"] == pytest.approx(0.03)
    assert decision["pmf_fingerprint"] == "p"


def test_reference_site_average_is_dropped_when_a_cell_is_unscored() -> None:
    """A site average over different cells than the random set's is never scored."""

    strata = [(0, 0, 0), (1, 0, 0)]
    target_rows = _target_rows(exp31.COVARIATES, strata)
    target = [item["target"] for item in draw_raw_pmfs(target_rows, n_draws=3, random_seed=2, covariates=exp31.COVARIATES)]
    reference = pd.DataFrame(
        [{"iv": x, "dv_binary": (x + i) % 2, "resp_age_bin": z, "resp_gender_bin": 0, "resp_polideo_bin": 0}
         for z in (0, 1) for x in (0, 1) for i in range(10)]
    )
    support = pd.DataFrame({"estimand_key": ["[0, 0, 0]", "[1, 0, 0]"], "reference_supported": [True, False]})
    key = exp31.ConditionKey("gainloss", "age", 1.0, 0)
    references = exp31.draw_reference(reference, target, support, key, n_draws=3)
    assert set(references.estimand_type) == {"cate"}
    assert set(references.estimand_key) == {"[0, 0, 0]"}
    support["reference_supported"] = True
    assert "ate" in set(exp31.draw_reference(reference, target, support, key, n_draws=3).estimand_type)


def test_loso_arm_scores_every_held_out_site(prepared, fake_nuts, tmp_path) -> None:
    """The leave-one-source-site-out arm writes one score per site, candidate and kind."""

    root, pickle = prepared
    pmf = _pmf_root(prepared)
    settings = exp31.Exp31RunSettings(arm="loso", imputation_counts=(1,), random_set_draws=FAST_DRAWS, loso_score_draws=10)
    identity = exp31.initialize_run_root(tmp_path / "run", root, pickle, pmf, settings, resume=False)
    exp31.run_condition(key=exp31.ConditionKey(*exp31.LOSO_CONDITIONS[0]), prepared_root=root, rct_path=pickle,
                        pmf_root=pmf, output_dir=tmp_path / "run", settings=settings, run_identity=identity,
                        resume=True, isolate_fits=False)
    leaf = exp31.ConditionKey(*exp31.LOSO_CONDITIONS[0]).leaf(tmp_path / "run")
    scores = pd.read_parquet(leaf / "loso_predictive_scores.parquet")
    assert len(scores) == 4 * len(exp31.COVARIATES) * 2
    scored = scores.loc[scores.scored.astype(bool)]
    assert scored.validity_probability.between(0.0, 1.0).all()
    assert scored.target_tolerance_shift.notna().any()
    assert (leaf / "condition_metrics.parquet").is_file()


def _finalized_root(path: Path, arm: str, metrics: pd.DataFrame, keys: list[exp31.ConditionKey],
                    random_sets: Path | None = None, source_fingerprint: str = "primary-fingerprint",
                    fingerprint: str | None = None) -> Path:
    """Write a minimal finalized run root around given condition metrics.

    With ``random_sets``, a smoke run's saved random identified sets, every
    condition gets a copy where the report's posterior width intervals read
    them back: under the smoke run's imputation folder, or at the leaf of a
    ``balke_pearl`` root, whose identity names ``source_fingerprint`` as its run.
    The root's own fingerprint is ``fingerprint``, ``"<arm>-fingerprint"`` by default.
    """

    path.mkdir(parents=True)
    metrics.to_parquet(path / "condition_metrics.parquet", index=False)
    identity = {"experiment": exp31.EXPERIMENT_NAME, "arm": arm, "fingerprint": fingerprint or f"{arm}-fingerprint",
                "conditions": [key.as_dict() for key in keys], "settings": {"imputation_counts": [FAST_DRAWS]}}
    if arm == "balke_pearl":
        identity["source_run"] = {"arm": "primary", "fingerprint": source_fingerprint, "num_imputations": FAST_DRAWS}
        identity["protocol"] = exp31.balke_pearl_protocol()
    (path / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (path / "completion.json").write_text(json.dumps({"status": "complete", "validated": True}), encoding="utf-8")
    for key in keys if random_sets is not None else ():
        target = key.leaf(path) if arm == "balke_pearl" else key.leaf(path) / random_sets.parent.name
        target.mkdir(parents=True)
        shutil.copy(random_sets, target / random_sets.name)
    return path


def test_report_orders_discovery_first_and_classifies_every_family(prepared, fake_nuts, tmp_path) -> None:
    """The report covers every condition, writes every table and figure, and labels families.

    One fixture condition's metrics stand in for every condition of a small
    grid, so the report plumbing is exercised without fitting 480 conditions.
    """

    _run_smoke(prepared, tmp_path / "run")
    exp31.run_balke_pearl(tmp_path / "bp_run", tmp_path / "run", prepared[0], prepared[1], _pmf_root(prepared))
    template = pd.read_parquet(tmp_path / "run" / "condition_metrics.parquet")
    bp_template = pd.read_parquet(tmp_path / "bp_run" / "condition_metrics.parquet")
    keys = [exp31.ConditionKey(study, pattern, beta, rep) for study in exp31.STUDIES for pattern in exp31.PATTERNS
            for beta in exp31.BETAS for rep in (0, 1)]

    def grid(offset: float, source: pd.DataFrame | None = None,
             grid_keys: list[exp31.ConditionKey] | None = None) -> pd.DataFrame:
        """Copy the template metrics to every key, shifting probabilities slightly."""

        frames = []
        for key in keys if grid_keys is None else grid_keys:
            frame = (template if source is None else source).copy()
            for name, value in key.as_dict().items():
                frame[name] = value
            frame["value"] = frame.value + offset * (frame.denominator_label == exp31.ALL_DRAWS)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    random_sets = next((tmp_path / "run").rglob("random_identified_sets.parquet"))
    primary = _finalized_root(tmp_path / "primary", "primary", grid(0.0), keys, random_sets)
    pilot = _finalized_root(tmp_path / "pilot", "pilot", grid(0.01), keys)
    bp_sets = SMOKE.leaf(tmp_path / "bp_run") / "bp_random_identified_sets.parquet"
    bp_root = _finalized_root(tmp_path / "balke_pearl", "balke_pearl", grid(0.0, bp_template), keys, bp_sets)
    output = tmp_path / "report"
    summary = exp31.report_results(primary_root=primary, output_dir=output, balke_pearl_root=bp_root)
    n_families = len(exp31.STUDIES) * len(exp31.PATTERNS) * len(exp31.BETAS)
    assert summary["n_conditions"] == len(keys) and summary["n_families"] == n_families
    for name in ("complete_condition_table.csv", "family_summary.csv", "family_classification.csv",
                 "cell_results_by_family.csv", "width_by_beta.csv", "width_by_beta_by_pattern.csv",
                 "discovery_and_grounding_by_candidate.csv", "report.md", "report_summary.json", "completion.json"):
        assert (output / name).is_file(), name
    assert not (output / "prior_sensitivity.csv").exists()
    figures = sorted(path.name for path in (output / "figures").glob("*.png"))
    assert figures[0].startswith("01") and any(name.startswith("05") for name in figures)
    for stem in ("04a_width_by_beta", "04b_width_by_beta_by_pattern", "04c_paired_width_gain",
                 "04d_naturalness_vs_balke_pearl", "05b_reference_by_method", "05c_width_membership_tradeoff"):
        for suffix in exp31.FIGURE_FORMATS:
            assert (output / "figures" / f"{stem}.{suffix}").is_file(), (stem, suffix)
    # Every condition carries the same metrics, so the replication bootstrap
    # interval collapses onto the mean.
    widths = pd.read_csv(output / "width_by_beta.csv")
    methods = {*exp31.METHODS, exp31.BALKE_PEARL}
    assert len(widths) == len(exp31.STUDIES) * len(exp31.BETAS) * len(methods)
    assert set(widths.method) == methods and widths.n_conditions.eq(len(exp31.PATTERNS) * 2).all()
    comparison = pd.read_csv(output / "balke_pearl_comparison_by_family.csv")
    assert len(comparison) == len(exp31.STUDIES) * len(exp31.PATTERNS) * len(exp31.BETAS)
    assert comparison.balke_pearl_weighted_width_mean.notna().all()
    assert np.allclose(widths.ci_lower, widths["mean"]) and np.allclose(widths.ci_upper, widths["mean"])
    tradeoff = pd.read_csv(output / "width_membership_tradeoff.csv")
    assert len(tradeoff) == len(exp31.STUDIES) * len(exp31.BETAS) * len(exp31.TRADEOFF_METHODS)
    for name in ("width", "member", "wrong"):
        assert np.allclose(tradeoff[f"{name}_lo"], tradeoff[name]) and np.allclose(tradeoff[f"{name}_hi"], tradeoff[name])
    table_rows = (output / "width_membership_tradeoff_table.tex").read_text().count(r"\emph{")
    assert table_rows == len(exp31.STUDIES) * len(exp31.BETAS)
    by_pattern = pd.read_csv(output / "width_by_beta_by_pattern.csv")
    assert len(by_pattern) == len(widths) * len(exp31.PATTERNS) and by_pattern.n_conditions.eq(2).all()
    # The 90% posterior interval read back from the random sets nests inside
    # the finalized 95% one.
    table = pd.read_csv(output / "complete_condition_table.csv")
    assert (table.naturalness_weighted_width_q025 <= table.naturalness_width_posterior_lower).all()
    assert (table.naturalness_width_posterior_lower <= table.naturalness_width_posterior_upper).all()
    assert (table.naturalness_width_posterior_upper <= table.naturalness_weighted_width_q975).all()
    assert widths.loc[widths.method.eq(exp31.NATURALNESS), "posterior_level"].eq(exp31.POSTERIOR_WIDTH_LEVEL).all()
    classified = pd.read_csv(output / "family_classification.csv")
    assert len(classified) == n_families and classified.label.notna().all()
    assert set(classified.label) <= {"overconfident_restriction", "empty_set_tension", "unstable_grounding",
                                     "valid_abstention", "useful_bayesian_naturalness_example", "mixed_evidence"}
    # No pilot or leave-one-out root was given, so no stability check was assessed.
    assert not classified.imputation_assessed.any() and not classified.loso_assessed.any()
    assert classified.robust.all() and "max_prior_eligibility_shift" not in classified
    report = (output / "report.md").read_text()
    assert report.index("Restriction discovery") < report.index("Grounding") < report.index("Bounds")
    assert "not a confidence or credible interval" in report
    assert exp31.FIGURE_CAPTIONS["04a_width_by_beta"] in report and "width_by_beta.csv" in report
    assert "Post hoc Bayesian Balke-Pearl" in report and exp31.BALKE_PEARL_WIDTH_CAPTION in report
    assert summary["inputs"]["balke_pearl"]["fingerprint"] == "balke_pearl-fingerprint"
    with pytest.raises(FileExistsError):
        exp31.report_results(primary_root=primary, output_dir=output)
    with pytest.raises(ValueError, match="expected 'primary'"):
        exp31.report_results(primary_root=pilot, output_dir=tmp_path / "other")
    # Random sets that no longer match the finalized metrics are refused.
    stale = grid(0.0)
    stale.loc[stale.metric.eq("weighted_width_mean") & stale.method.eq(exp31.NATURALNESS), "value"] += 0.1
    stale_root = _finalized_root(tmp_path / "stale", "primary", stale, keys, random_sets)
    with pytest.raises(ValueError, match="do not reproduce"):
        exp31.report_results(primary_root=stale_root, output_dir=tmp_path / "stale_report")
    modular = _finalized_root(tmp_path / "modular_root", "modular", grid(0.0), keys, random_sets)
    exp31.report_results(primary_root=modular, output_dir=tmp_path / "modular_report")
    assert "Arm: modular" in (tmp_path / "modular_report" / "report.md").read_text()
    foreign = _finalized_root(tmp_path / "foreign_bp", "balke_pearl", grid(0.0, bp_template), keys, bp_sets,
                              source_fingerprint="another-run")
    with pytest.raises(ValueError, match="another run"):
        exp31.report_results(primary_root=primary, output_dir=tmp_path / "foreign_report", balke_pearl_root=foreign)

    # A separate reciprocity run with identical settings is reported with the others as one run.
    deferred_keys = [exp31.ConditionKey("reciprocity", pattern, beta, rep) for pattern in exp31.PATTERNS
                     for beta in exp31.BETAS for rep in (0, 1)]
    deferred = _finalized_root(tmp_path / "primary_reciprocity", "primary", grid(0.0, grid_keys=deferred_keys),
                               deferred_keys, random_sets, fingerprint="reciprocity-fingerprint")
    deferred_bp = _finalized_root(tmp_path / "bp_reciprocity", "balke_pearl", grid(0.0, bp_template, deferred_keys),
                                  deferred_keys, bp_sets, source_fingerprint="reciprocity-fingerprint",
                                  fingerprint="bp-reciprocity-fingerprint")
    combined = exp31.report_results(primary_root=[primary, deferred], output_dir=tmp_path / "combined",
                                    balke_pearl_root=[deferred_bp, bp_root])
    assert combined["n_conditions"] == len(keys) + len(deferred_keys)
    assert combined["n_families"] == len(exp31.ALL_STUDIES) * len(exp31.PATTERNS) * len(exp31.BETAS)
    assert [entry["fingerprint"] for entry in combined["inputs"]["primary"]] == ["primary-fingerprint",
                                                                                "reciprocity-fingerprint"]
    combined_widths = pd.read_csv(tmp_path / "combined" / "width_by_beta.csv")
    assert set(combined_widths.study) == set(exp31.ALL_STUDIES)
    assert pd.read_csv(tmp_path / "combined" / "complete_condition_table.csv").balke_pearl_weighted_width_mean.notna().all()
    with pytest.raises(ValueError, match="own Balke-Pearl"):
        exp31.report_results(primary_root=[primary, deferred], output_dir=tmp_path / "missing_bp", balke_pearl_root=bp_root)
    with pytest.raises(ValueError, match="repeats conditions"):
        exp31.report_results(primary_root=[primary, primary], output_dir=tmp_path / "repeated")
    with pytest.raises(ValueError, match="another design"):
        exp31.report_results(primary_root=[primary, modular], output_dir=tmp_path / "mixed")


def test_tradeoff_table_pools_conditions_and_resamples_replications() -> None:
    """Means pool a study and strength's conditions; the bootstrap resamples replications reproducibly."""

    rows = []
    for study in ("allowedforbidden", "gainloss"):
        for replication in range(4):
            for pattern in exp31.PATTERNS:
                row = {"study": study, "beta": 1.0, "replication": replication, "pattern": pattern}
                for prefix, width in (("naturalness", 0.6 + 0.02 * replication),
                                      ("balke_pearl", 0.4 + 0.02 * replication), ("manski", 1.0)):
                    row[f"{prefix}_weighted_width_mean"] = width
                    row[f"{prefix}_membership_all"] = 0.9 - 0.02 * replication
                    row[f"{prefix}_wrong_sign_all"] = 0.002 * replication
                rows.append(row)
    conditions = pd.DataFrame(rows)

    summary = exp31.tradeoff_table(conditions)

    assert len(summary) == 2 * len(exp31.TRADEOFF_METHODS)
    pd.testing.assert_frame_equal(summary, exp31.tradeoff_table(conditions))
    naturalness = summary.loc[summary.method.eq("naturalness")].iloc[0]
    assert naturalness.width == pytest.approx(0.63)
    assert naturalness.width_lo < naturalness.width < naturalness.width_hi
    latex = exp31.tradeoff_results_latex(summary)
    assert r"\emph{allowedforbidden} & 1 & 0.63 & 0.43 & 1.00 & 0.87 & 0.87 & 0.003 & 0.003 \\" in latex
    assert latex.count(r"\emph{") == 2


def test_width_by_beta_keeps_posterior_and_replication_uncertainty_apart() -> None:
    """Posterior quantiles are averaged; the bootstrap resamples whole replications.

    Selection covariates are averaged within a replication before resampling,
    and Exp25 rows outside the Exp31 grid are dropped.
    """

    rows = []
    for replication, shift in enumerate((0.0, 0.2)):
        for pattern in ("age", "gender"):
            rows.append({"study": "gainloss", "pattern": pattern, "beta": 1.0, "replication": replication,
                         "naturalness_weighted_width_mean": 0.5 + shift + (0.1 if pattern == "gender" else 0.0),
                         "naturalness_width_posterior_lower": 0.2,
                         "naturalness_width_posterior_upper": 0.9 + shift / 2, "manski_weighted_width_mean": 1.0})
    conditions = pd.DataFrame(rows)
    exp25 = pd.DataFrame({"study": ["gainloss", "gainloss", "reciprocity"], "pattern": ["age", "age", "age"],
                          "beta": [1.0, 3.0, 1.0], "replication": [0, 0, 0], "width": [0.3, 0.1, 0.2]})
    table = exp31.width_by_beta_table(conditions, exp25).set_index("method")
    naturalness = table.loc[exp31.NATURALNESS]
    # Replication means 0.55 and 0.75: the bootstrap can only mix those two.
    assert naturalness["mean"] == pytest.approx(0.65)
    assert 0.55 - 1e-12 <= naturalness.ci_lower < naturalness.ci_upper <= 0.75 + 1e-12
    assert naturalness.posterior_lower == pytest.approx(0.2) and naturalness.posterior_upper == pytest.approx(0.95)
    assert naturalness.n_replications == 2 and naturalness.n_conditions == 4
    assert table.loc[exp31.MANSKI, "mean"] == pytest.approx(1.0) and np.isnan(table.loc[exp31.MANSKI, "posterior_lower"])
    plugin = table.loc[exp31.EXP25_PLUGIN]
    assert plugin["mean"] == pytest.approx(0.3) and plugin.n_conditions == 1
    by_pattern = exp31.width_by_beta_table(conditions, by_pattern=True)
    assert set(by_pattern.pattern) == {"age", "gender"} and by_pattern.method.isin(exp31.METHODS).all()


def test_unconverged_surface_is_excluded_for_its_imputation_only(prepared, monkeypatch, tmp_path) -> None:
    """A surface failing its second retry leaves only that imputation's draws without the candidate."""

    settings = exp31.Exp31RunSettings(arm="smoke", imputation_counts=(FAST_DRAWS,), random_set_draws=FAST_DRAWS)
    calls = {"n": 0}

    def fit_with_one_failure(evidence: pd.DataFrame, **kwargs) -> dict:
        """Exclude the first imputation's age 2c surface, as a failed second retry would."""

        surfaces = _fake_fit(evidence, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            del surfaces[("resp_age_bin", "c")]
            warmup, samples = settings.attempt_budgets()["second_retry"]
            kwargs["unconverged"].append({
                "candidate": "resp_age_bin", "tolerance_kind": "c", "imputation_id": str(evidence.sample_id.iloc[0]),
                "passed": False, "attempt": "second_retry", "max_r_hat": 1.02, "min_bulk_ess": 200.0,
                "min_tail_ess": 300.0, "n_divergent": 0, "num_chains": settings.num_chains, "num_warmup": warmup,
                "num_samples": samples,
            })
        return surfaces

    monkeypatch.setattr(exp31, "fit_tolerance_surfaces", fit_with_one_failure)
    root, pickle = prepared
    exp31.run_experiment(tmp_path / "run", root, pickle, settings, pmf_root=_pmf_root(prepared), isolate_conditions=False)
    leaf = SMOKE.leaf(tmp_path / "run")
    diagnostics = pd.read_parquet(leaf / "tolerance_surface_diagnostics.parquet")
    assert diagnostics.excluded.sum() == 1 and len(diagnostics) == FAST_DRAWS * 6
    triggers = pd.read_parquet(leaf / f"imputations={FAST_DRAWS}" / "target_trigger_draws.parquet")
    age = triggers.loc[triggers.candidate.eq("resp_age_bin")]
    first = age.loc[age.source_imputation_index.eq(0)]
    assert not first.eligible.any() and not first.surface_available.any()
    assert set(first.loc[first.triggered, "ineligibility_reason"]) <= {"surface_not_converged"}
    assert "surface_not_converged" not in set(age.loc[age.source_imputation_index.ne(0), "ineligibility_reason"])
    metrics = pd.read_parquet(tmp_path / "run" / "condition_metrics.parquet")
    excluded = metrics.loc[metrics.metric.eq("n_surfaces_excluded_unconverged"), "value"].iloc[0]
    assert excluded == 1 and json.loads((tmp_path / "run" / "completion.json").read_text())["validated"]


# --- Bayesian Balke-Pearl arm ---------------------------------------------------


def _iv_table(rng: np.random.Generator) -> np.ndarray:
    """Return a ``P(Y, X | Z)`` table, indexed ``[z, y, x]``, that a latent-type model generates."""

    types = rng.dirichlet(np.ones(balke_pearl._A_EQ.shape[1]))
    return (balke_pearl._A_EQ @ types).reshape(2, 2, 2)


def _two_covariate_draw(tables: dict[tuple[int, int], np.ndarray]) -> bayes.SitePMFDraw:
    """Build a target draw over two binary covariates from per-cell ``P(Y, X | Z=z)`` tables.

    ``tables[z][y, x]`` is the cell's joint; every cell gets ample counts.
    """

    cells = tuple(sorted(tables))
    return bayes.SitePMFDraw(
        sample_id="synthetic", site_id="pi", covariates=("z0", "z1"), z_values=cells,
        idle_xy_given_z={(z, x, y): float(tables[z][y, x]) for z in cells for x in (0, 1) for y in (0, 1)},
        do_y_given_xz={}, idle_z={z: 1.0 / len(cells) for z in cells}, n_idle_z={z: 50 for z in cells},
        n_idle_xz={(x, z): 25 for z in cells for x in (0, 1)}, overlap_model=False, alpha=1.0,
    )


def test_instrumental_inequality_matches_balke_pearl_feasibility() -> None:
    """The inequality that tells a refuted instrument from a solver failure is the LP's feasibility."""

    rng = np.random.default_rng(7)
    for concentration in (0.3, 1.0, 5.0):
        for _ in range(150):
            joint = rng.dirichlet(np.full(4, concentration), size=2).reshape(2, 2, 2)
            assert exp31._instrumental_inequality_holds(joint) == balke_pearl.balke_pearl_interval(joint).feasible


def test_balke_pearl_cells_take_their_blocks_sharp_bounds_and_empty_when_refuted() -> None:
    """Each cell gets its block's Balke-Pearl bound, the intersection over instruments, or none.

    A refuted block empties both its cells, a cell whose block partner is no LP
    cell keeps no interval from that instrument, and every bound lies inside the
    cell's Manski interval.
    """

    rng = np.random.default_rng(3)
    tables = {}
    for block in (0, 1):
        # One latent-type distribution per z1 block: z0 is a valid instrument there.
        types = rng.dirichlet(np.ones(balke_pearl._A_EQ.shape[1]))
        joint = (balke_pearl._A_EQ @ types).reshape(2, 2, 2)
        tables[(0, block)], tables[(1, block)] = joint[0], joint[1]
    draw = _two_covariate_draw(tables)
    cells = sorted(tables)
    bounds, usage = exp31.balke_pearl_cell_bounds(draw, cells, [0])
    for block in (0, 1):
        expected = balke_pearl.balke_pearl_interval(np.stack([tables[(0, block)], tables[(1, block)]]))
        for cell in ((0, block), (1, block)):
            assert bounds[cell]["outcome"] == "bounded" and bounds[cell]["n_instruments"] == 1
            assert bounds[cell]["lower"] == pytest.approx(expected.lower) and bounds[cell]["upper"] == pytest.approx(expected.upper)
            p11, p10 = tables[cell][1, 1], tables[cell][1, 0]
            p1 = tables[cell][:, 1].sum()
            assert p11 - p10 - p1 - 1e-9 <= bounds[cell]["lower"] <= bounds[cell]["upper"] <= p11 + (1 - p1) - p10 + 1e-9
    assert usage == {0: {"n_blocks": 2, "n_refuted": 0}}

    both, _ = exp31.balke_pearl_cell_bounds(draw, cells, [0, 1])
    column = balke_pearl.balke_pearl_interval(np.stack([tables[(0, 0)], tables[(0, 1)]]))
    if column.feasible:
        lower = max(bounds[(0, 0)]["lower"], column.lower)
        upper = min(bounds[(0, 0)]["upper"], column.upper)
        assert both[(0, 0)]["n_instruments"] == 2
        if lower <= upper:
            assert both[(0, 0)]["outcome"] == "bounded" and both[(0, 0)]["lower"] == pytest.approx(lower)
        else:
            assert both[(0, 0)]["outcome"] == "disjoint" and np.isnan(both[(0, 0)]["lower"])

    refuted = dict(tables)
    refuted[(0, 1)] = np.array([[0.05, 0.0], [0.05, 0.9]])  # P(Y=1, X=1 | z0=0) = 0.9
    refuted[(1, 1)] = np.array([[0.05, 0.9], [0.05, 0.0]])  # P(Y=0, X=1 | z0=1) = 0.9
    bounds, usage = exp31.balke_pearl_cell_bounds(_two_covariate_draw(refuted), cells, [0])
    assert bounds[(0, 1)]["outcome"] == bounds[(1, 1)]["outcome"] == "refuted" and np.isnan(bounds[(1, 1)]["upper"])
    assert bounds[(0, 0)]["outcome"] == "bounded" and usage[0] == {"n_blocks": 2, "n_refuted": 1}

    partial, usage = exp31.balke_pearl_cell_bounds(draw, [(0, 0), (1, 0), (0, 1)], [0])
    assert partial[(0, 1)]["outcome"] == "no_instrument" and partial[(0, 1)]["n_instruments"] == 0
    assert partial[(0, 0)]["outcome"] == "bounded" and usage[0]["n_blocks"] == 1


def test_balke_pearl_arm_builds_on_the_runs_draws_and_finalizes(prepared, fake_nuts, tmp_path) -> None:
    """The arm pairs every set with the run's draw, falls back and empties as Naturalness does, and validates.

    Cells without an instrument copy the matched Manski interval, every bound
    lies inside it, the site average is the weighted sum of the cells, the
    finalized root reads back as an arm of its own, and a changed input of the
    run or a rerun without ``resume`` is refused.
    """

    _run_smoke(prepared, tmp_path / "run")
    root, pickle = prepared
    output = tmp_path / "bp"
    exp31.run_balke_pearl(output, tmp_path / "run", root, pickle, _pmf_root(prepared))
    identity, metrics = exp31._read_finalized(output, "balke_pearl")
    assert identity["source_run"]["arm"] == "smoke" and identity["protocol"]["relevance_tau"] == 0.01
    leaf = SMOKE.leaf(output)
    sets = pd.read_parquet(leaf / "bp_random_identified_sets.parquet")
    primary = pd.read_parquet(SMOKE.leaf(tmp_path / "run") / f"imputations={FAST_DRAWS}" / "random_identified_sets.parquet")
    paired = sets.merge(primary.loc[primary.method.eq(exp31.MANSKI)], on=["draw_index", "estimand_type", "estimand_key"],
                        suffixes=("", "_manski"), validate="one_to_one")
    assert len(paired) == len(sets) == FAST_DRAWS * primary.loc[primary.method.eq(exp31.MANSKI)].groupby(
        ["estimand_type", "estimand_key"]).ngroups
    for column in ("target_pmf_draw_index", "target_pmf_sample_id", "sample_id", "estimand_weight"):
        assert paired[column].equals(paired[f"{column}_manski"]), column
    copied = paired.loc[paired.bp_cell_outcome.eq("no_instrument")]
    assert np.allclose(copied.lower, copied.lower_manski) and np.allclose(copied.upper, copied.upper_manski)
    cells = paired.loc[paired.estimand_type.eq("cate") & paired.nonempty]
    assert (cells.lower >= cells.lower_manski - 1e-7).all() and (cells.upper <= cells.upper_manski + 1e-7).all()
    assert sets.loc[~sets.nonempty, "lower"].isna().all()
    for draw_index, group in sets.loc[sets.nonempty.groupby(sets.draw_index).transform("all")].groupby("draw_index"):
        cate, ate = group.loc[group.estimand_type.eq("cate")], group.loc[group.estimand_type.eq("ate")].iloc[0]
        assert ate.lower == pytest.approx(float((cate.estimand_weight * cate.lower).sum()))
    for tag in (exp31.BP_VS_MANSKI, exp31.NATURALNESS_VS_BP):
        assert metrics.loc[metrics.method.eq(tag) & metrics.metric.eq("p_narrower_given_joint")].shape[0] >= 1
    bp = metrics.loc[metrics.method.eq(exp31.BALKE_PEARL)]
    assert {"p_nonempty", "weighted_width_mean", "weighted_membership_all", "p_any_instrument"} <= set(bp.metric)
    assert bp.loc[bp.metric.eq("p_nonempty") & bp.scope.eq("feasibility"), "denominator"].eq(FAST_DRAWS).all()

    # With no relevant instrument every draw is the matched Manski set.
    data = exp31.load_condition_data(SMOKE, root, pickle)
    _, target_draws, _ = exp31.load_condition_pmfs(SMOKE, _pmf_root(prepared), data,
                                                   pmf_fingerprint=identity["pmf_fingerprint"], n_source_draws=0,
                                                   n_target_draws=FAST_DRAWS)
    fallback, instruments = exp31.balke_pearl_random_sets(target_draws, primary, relevance_tau=2.0)
    manski = primary.loc[primary.method.eq(exp31.MANSKI)].reset_index(drop=True)
    assert fallback.used_manski_fallback.all() and not instruments.used.any()
    assert np.allclose(fallback.lower, manski.lower) and np.allclose(fallback.upper, manski.upper)

    with pytest.raises(ValueError, match="--resume"):
        exp31.initialize_balke_pearl_root(output, tmp_path / "run", root, pickle, _pmf_root(prepared), resume=False)
    # Another threshold is another arm: resuming this root with it is refused.
    with pytest.raises(ValueError, match="differs"):
        exp31.initialize_balke_pearl_root(output, tmp_path / "run", root, pickle, _pmf_root(prepared), resume=True,
                                          relevance_tau=0.075)
    strict = tmp_path / "bp_strict"
    exp31.run_balke_pearl(strict, tmp_path / "run", root, pickle, _pmf_root(prepared), relevance_tau=0.075)
    strict_instruments = pd.read_parquet(SMOKE.leaf(strict) / "bp_instrument_draws.parquet")
    assert strict_instruments.relevant.eq(strict_instruments.d_strength.ge(0.075)).all()
    assert exp31._read_finalized(strict, "balke_pearl")[0]["protocol"]["relevance_tau"] == 0.075
    changed = tmp_path / "changed_run"
    shutil.copytree(tmp_path / "run", changed)
    references = SMOKE.leaf(changed) / "rct_reference_draws.parquet"
    pd.read_parquet(references).assign(reference_value=lambda frame: frame.reference_value + 0.01).to_parquet(references)
    with pytest.raises(RuntimeError, match="conditions failed"):
        exp31.run_balke_pearl(tmp_path / "bp_changed", changed, root, pickle, _pmf_root(prepared))
    failure = json.loads((SMOKE.leaf(tmp_path / "bp_changed") / "completion.json").read_text())
    assert failure["status"] == "failed" and "changed after the run wrote it" in failure["error"]


# --- Modular arm ----------------------------------------------------------------


def test_modular_arm_grounds_on_full_data_fits_and_recentred_target_draws(prepared, fake_tabicl, fake_nuts,
                                                                          tmp_path, monkeypatch) -> None:
    """The modular arm fits one plug-in evidence imputation and bounds with target draws centred on the plug-in fit.

    Its identity records the design, one envelope is fitted per candidate and
    kind, the saved trigger statistics are those of the recentred target draws,
    whose mean logit is the target's full-data fit, and the plug-in option of
    Module A refuses more than one draw.
    """

    from scipy.special import logit

    monkeypatch.setitem(exp31.ARMS, "modular", {"conditions": exp31.SMOKE_CONDITIONS, "needs_pilot_decision": False})
    root, pickle = prepared
    settings = exp31.Exp31RunSettings(arm="modular", imputation_counts=(1,), random_set_draws=FAST_DRAWS)
    output = tmp_path / "modular"
    exp31.run_experiment(output, root, pickle, settings, pmf_root=_pmf_root(prepared), isolate_conditions=False)
    identity, _ = exp31._read_finalized(output, "modular")
    assert identity["modular_design"]["require_source_support"] is False
    leaf = SMOKE.leaf(output)
    assert (leaf / "full_data_fits.parquet").is_file()
    assert set(pd.read_parquet(leaf / "source_evidence_draws.parquet").imputation_index) == {0}
    assert len(pd.read_parquet(leaf / "tolerance_surface_diagnostics.parquet")) == len(exp31.COVARIATES) * 2

    data = exp31.load_condition_data(SMOKE, root, pickle)
    _, draws, _ = exp31.load_condition_pmfs(SMOKE, _pmf_root(prepared), data, pmf_fingerprint=identity["pmf_fingerprint"],
                                            n_source_draws=1, n_target_draws=FAST_DRAWS)
    _, center = exp31.full_data_pmfs(data, SMOKE)
    recentred = exp31.recentre_draws(draws, center)
    _, p_center, _ = exp31._pmf_conditionals(center)
    for cell in p_center:
        mean_logit = np.mean([logit(exp31._pmf_conditionals(draw)[1][cell]) for draw in recentred])
        assert mean_logit == pytest.approx(logit(p_center[cell]), abs=1e-9)
    saved = pd.read_parquet(leaf / "imputations=1" / "target_trigger_draws.parquet")
    expected = bayes.compute_trigger_statistics(recentred[0])
    first = saved.loc[saved.draw_index.eq(0)].set_index("candidate").d_strength
    assert np.allclose(first.reindex(expected.candidate).to_numpy(), expected.d_strength.to_numpy())

    with pytest.raises(ValueError, match="single draw"):
        bayes.draw_tabicl_bootstrap_pmfs(data.target_idle, n_draws=2, random_seed=0, resample=False,
                                         covariates=exp31.COVARIATES)
    with pytest.raises(ValueError, match="one evidence imputation"):
        exp31.Exp31RunSettings(arm="modular", imputation_counts=(20,)).validate()
    # Its recentred target draws are not the PMF root's, so no Balke-Pearl arm is built on it.
    with pytest.raises(ValueError, match="Balke-Pearl arm builds on"):
        exp31.run_balke_pearl(tmp_path / "bp_modular", output, root, pickle, _pmf_root(prepared))


# --- Predictive arm --------------------------------------------------------------


def test_predictive_arm_runs_on_and_only_on_a_predictive_pmf_root(prepared, fake_nuts, tmp_path, monkeypatch) -> None:
    """``draw-pmfs --module-a-scheme predictive`` writes a root that the predictive arm, and only it, runs on.

    The scheme enters the PMF root's identity, while a bootstrap root's identity
    carries no scheme and keeps its fingerprint.  The draws reproduce from the
    condition's seeds and are stored beside the full-data fits they centre on.
    The arm grounds every imputation and bounds without the source-support gate.
    Arm-root mismatches, adoption across schemes and a mixture-size comparison
    are refused.
    """

    monkeypatch.setitem(exp31.ARMS, "predictive", {"conditions": exp31.SMOKE_CONDITIONS, "needs_pilot_decision": False})
    root, pickle = prepared
    pmf = tmp_path / "pmf_predictive"
    with pytest.MonkeyPatch.context() as patch:
        _patch_tabicl(patch)
        exp31.draw_pmfs(pmf, root, pickle, arm="smoke", device="cpu", n_source_draws=2, n_target_draws=4,
                        scheme="predictive")
        data = exp31.load_condition_data(SMOKE, root, pickle)
        centers, _, targets, seconds = exp31.predictive_pmfs(data, SMOKE, n_source=2, n_target=4)
    assert exp31.read_pmf_identity(pmf)["module_a_scheme"] == "predictive"
    assert "module_a_scheme" not in exp31.read_pmf_identity(_pmf_root(prepared))
    assert set(seconds) == {"full_data_fits", "source_draws", "target_draws"}
    leaf = SMOKE.leaf(pmf)
    for name, draws in (("target_pmf_draws.parquet", targets), ("full_data_fits.parquet", [centers])):
        # Through the same parquet writer, so missing values compare alike.
        exp31._write_parquet(bayes.raw_pmf_draws_to_frame(draws), tmp_path / name)
        pd.testing.assert_frame_equal(pd.read_parquet(leaf / name), pd.read_parquet(tmp_path / name))

    settings = exp31.Exp31RunSettings(arm="predictive", imputation_counts=(2,), random_set_draws=4)
    output = tmp_path / "predictive"
    exp31.run_experiment(output, root, pickle, settings, pmf_root=pmf, isolate_conditions=False)
    identity, _ = exp31._read_finalized(output, "predictive")
    assert identity["predictive_design"]["require_source_support"] is False
    run_leaf = SMOKE.leaf(output)
    assert set(pd.read_parquet(run_leaf / "source_evidence_draws.parquet").imputation_index) == {0, 1}
    assert len(pd.read_parquet(run_leaf / "tolerance_surface_diagnostics.parquet")) == 2 * len(exp31.COVARIATES) * 2

    with pytest.raises(ValueError, match="cannot run on a 'bootstrap' PMF root"):
        exp31.run_experiment(tmp_path / "wrong_root", root, pickle, settings, pmf_root=_pmf_root(prepared),
                             isolate_conditions=False)
    smoke = exp31.Exp31RunSettings(arm="smoke", imputation_counts=(1,), random_set_draws=4)
    with pytest.raises(ValueError, match="cannot run on a 'predictive' PMF root"):
        exp31.run_experiment(tmp_path / "wrong_arm", root, pickle, smoke, pmf_root=pmf, isolate_conditions=False)
    with pytest.raises(ValueError, match="module_a_scheme"):
        exp31.adopt_pmfs(_pmf_root(prepared), root, source_root=pmf, source_prepared_root=root, arm="smoke")
    with pytest.raises(ValueError, match="one mixture"):
        exp31.Exp31RunSettings(arm="predictive", imputation_counts=(10, 20)).validate()
    assert exp31._default_imputation_counts("predictive", None) == (exp31.PREDICTIVE_SOURCE_DRAWS,)


def test_evidence_floor_restricts_module_b_and_enters_the_identity_only_when_used(prepared, fake_nuts, tmp_path,
                                                                                    monkeypatch) -> None:
    """The predictive arm's evidence subset is Module B's, run-wide, and fingerprinted only when set.

    With a floor of three randomized rows per arm the saved evidence is
    :func:`exp31._source_evidence` with that floor: no requirement exceeds the
    unfiltered one and some pairs are dropped.  The setting enters the run
    identity only when set, so runs without it keep their identities.  Only the
    ungated arms accept it, and the Balke-Pearl arm builds on the floored
    predictive root.
    """

    monkeypatch.setitem(exp31.ARMS, "predictive", {"conditions": exp31.SMOKE_CONDITIONS, "needs_pilot_decision": False})
    root, pickle = prepared
    pmf = tmp_path / "pmf_predictive"
    with pytest.MonkeyPatch.context() as patch:
        _patch_tabicl(patch)
        exp31.draw_pmfs(pmf, root, pickle, arm="smoke", device="cpu", n_source_draws=2, n_target_draws=4,
                        scheme="predictive")
    settings = exp31.Exp31RunSettings(arm="predictive", imputation_counts=(2,), random_set_draws=4,
                                      evidence_min_randomized_rows=3)
    output = tmp_path / "predictive_floor"
    exp31.run_experiment(output, root, pickle, settings, pmf_root=pmf, isolate_conditions=False)
    identity, _ = exp31._read_finalized(output, "predictive")
    assert identity["settings"]["evidence_min_randomized_rows"] == 3
    unset = exp31._settings_payload(dataclasses.replace(settings, evidence_min_randomized_rows=0))
    assert "evidence_min_randomized_rows" not in unset

    data = exp31.load_condition_data(SMOKE, root, pickle)
    pmf_identity = exp31.read_pmf_identity(pmf)
    sources, _, _ = exp31.load_condition_pmfs(SMOKE, pmf, data, pmf_fingerprint=pmf_identity["fingerprint"],
                                              n_source_draws=2, n_target_draws=4)
    saved = pd.read_parquet(SMOKE.leaf(output) / "source_evidence_draws.parquet")
    pd.testing.assert_frame_equal(saved, exp31._source_evidence(sources, data.sources, 3), check_dtype=False)
    full = exp31._source_evidence(sources, data.sources)
    keys = ["imputation_index", "site", "candidate"]
    paired = saved.merge(full, on=keys, suffixes=("", "_all"))
    assert (paired.n_comparison_pairs < paired.n_comparison_pairs_all).any()
    for column in ("required_two_c", "required_two_d"):
        finite = paired[column].notna()
        assert (paired.loc[finite, column] <= paired.loc[finite, f"{column}_all"] + 1e-12).all()
    with pytest.raises(ValueError, match="post hoc option"):
        exp31.Exp31RunSettings(arm="primary", imputation_counts=(20,), evidence_min_randomized_rows=3).validate()
    with pytest.raises(ValueError, match="non-negative"):
        exp31.Exp31RunSettings(arm="predictive", imputation_counts=(20,), evidence_min_randomized_rows=-1).validate()
    with pytest.raises(ValueError, match="source rows"):
        exp31._source_evidence(sources, None, 3)

    # The paper design's leave-one-source-site-out check runs on the same root and evidence subset.
    monkeypatch.setitem(exp31.ARMS, "loso_grid", {"conditions": exp31.SMOKE_CONDITIONS, "needs_pilot_decision": False})
    loso_settings = exp31.Exp31RunSettings(arm="loso_grid", imputation_counts=(1,), random_set_draws=4,
                                           evidence_min_randomized_rows=3)
    loso_output = tmp_path / "loso_grid"
    exp31.run_experiment(loso_output, root, pickle, loso_settings, pmf_root=pmf, isolate_conditions=False)
    loso_identity, loso_metrics = exp31._read_finalized(loso_output, "loso_grid")
    assert loso_identity["settings"]["evidence_min_randomized_rows"] == 3
    loso_leaf = SMOKE.leaf(loso_output)
    pd.testing.assert_frame_equal(pd.read_parquet(loso_leaf / "source_evidence_draws.parquet"),
                                  exp31._source_evidence(sources[:1], data.sources, 3), check_dtype=False)
    scores = pd.read_parquet(loso_leaf / "loso_predictive_scores.parquet")
    assert set(scores.held_out_site) == set(data.sources.site.astype(str))
    assert {"mean_validity_probability", "max_abs_target_tolerance_shift"} <= set(loso_metrics.metric)

    # The paper's Balke-Pearl comparator is built on the same predictive draws.
    bp = tmp_path / "bp_predictive"
    exp31.run_balke_pearl(bp, output, root, pickle, pmf)
    bp_identity, _ = exp31._read_finalized(bp, "balke_pearl")
    assert bp_identity["source_run"] == {"arm": "predictive", "fingerprint": identity["fingerprint"], "num_imputations": 2}


def _evidence_fit(evidence: pd.DataFrame, **kwargs) -> dict:
    """Stand in for NUTS with fixed surfaces on the evidence's own source units, as a real fit has them.

    Every surface's support is the units the real fit would use, so the
    source-support flags of a replay from the saved evidence are the run's.
    """

    imputation_id = str(evidence.sample_id.iloc[0])
    surfaces = {}
    for candidate, rows in evidence.groupby("candidate", sort=True):
        for kind, omega_column, feature_names in bayes._SURFACE_SPECS:
            units = rows.dropna(subset=[*feature_names, omega_column])
            base = _unit_cube_surface(candidate, kind, imputation_id)
            surface = dataclasses.replace(
                base,
                source_features=units[list(feature_names)].to_numpy(dtype=float),
                source_omega=units[omega_column].to_numpy(dtype=float),
                posterior_samples={**base.posterior_samples, "u": np.full((20, len(units)), -1.0),
                                   "source_tolerance": np.full((20, len(units)), 0.5),
                                   "length_scale": np.full((20, len(feature_names)), 0.5)},
            )
            surface.diagnostics.update(n_evidence_units=len(units),
                                       **{name: kwargs[name] for name in ("num_chains", "num_warmup", "num_samples")
                                          if name in kwargs})
            surfaces[(str(candidate), kind)] = surface
    return surfaces


def test_regrid_replays_a_run_on_the_restricted_trigger_grid(prepared, tmp_path, monkeypatch) -> None:
    """The regrid of a finalized run equals a run made on the restricted grid, and refuses anything else.

    Nothing is refitted, yet the regridded leaf's random sets, trigger draws,
    tolerance draws, evidence and metrics are those of a fresh predictive run
    whose protocol grid is ``tau >= 0.075``.  The root finalizes as a run of
    the source's arm with the restricted grid in its protocol, takes a
    Balke-Pearl arm, and does not combine with its source in one report.  A
    floor that drops no ``tau``, or a run whose saved draws the replay does not
    reproduce, is refused.
    """

    monkeypatch.setattr(exp31, "fit_tolerance_surfaces", _evidence_fit)
    monkeypatch.setitem(exp31.ARMS, "predictive", {"conditions": exp31.SMOKE_CONDITIONS, "needs_pilot_decision": False})
    root, pickle = prepared
    pmf = tmp_path / "pmf_predictive"
    with pytest.MonkeyPatch.context() as patch:
        _patch_tabicl(patch)
        exp31.draw_pmfs(pmf, root, pickle, arm="smoke", device="cpu", n_source_draws=2, n_target_draws=4,
                        scheme="predictive")
    settings = exp31.Exp31RunSettings(arm="predictive", imputation_counts=(2,), random_set_draws=8)
    run = tmp_path / "predictive"
    exp31.run_experiment(run, root, pickle, settings, pmf_root=pmf, isolate_conditions=False)
    regridded = tmp_path / "predictive_tau0p075"
    exp31.run_regrid(regridded, run, root, pickle, pmf, tau_floor=0.075)

    source_identity, _ = exp31._read_finalized(run, "predictive")
    identity, _ = exp31._read_finalized(regridded, "predictive")
    restricted = [value for value in exp31.TRIGGER_TAUS if value >= 0.075]
    assert identity["protocol"]["trigger_rule"]["taus"] == restricted
    assert identity["regrid"]["source_run"] == {"arm": "predictive", "fingerprint": source_identity["fingerprint"]}
    assert identity["settings"] == source_identity["settings"]
    triggers = pd.read_parquet(SMOKE.leaf(regridded) / "imputations=2" / "target_trigger_draws.parquet")
    saved_triggers = pd.read_parquet(SMOKE.leaf(run) / "imputations=2" / "target_trigger_draws.parquet")
    assert triggers.loc[triggers.triggered, "d_strength"].ge(0.075).all()
    assert triggers.triggered.sum() < saved_triggers.triggered.sum()

    # A fresh run made on the restricted grid: same fits and seeds, so the same draws.
    native = tmp_path / "predictive_native"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exp31, "TRIGGER_TAUS", tuple(restricted))
        exp31.run_experiment(native, root, pickle, settings, pmf_root=pmf, isolate_conditions=False)
    assert exp31._read_finalized(native, "predictive")[0]["protocol"] == identity["protocol"]
    for name in ["source_evidence_draws.parquet", *[f"imputations=2/{file}" for file in exp31.MIXTURE_FILES]]:
        pd.testing.assert_frame_equal(pd.read_parquet(SMOKE.leaf(regridded) / name),
                                      pd.read_parquet(SMOKE.leaf(native) / name), check_dtype=False, obj=name)

    bp = tmp_path / "bp_tau0p075"
    exp31.run_balke_pearl(bp, regridded, root, pickle, pmf, relevance_tau=0.075)
    assert exp31._read_finalized(bp, "balke_pearl")[0]["source_run"]["fingerprint"] == identity["fingerprint"]
    with pytest.raises(ValueError, match="another design"):
        exp31._read_combined([run, regridded], ("predictive",))

    with pytest.raises(ValueError, match="must drop some but not all"):
        exp31.initialize_regrid_root(tmp_path / "no_change", run, root, pickle, pmf, tau_floor=0.001, resume=False)
    with pytest.raises(ValueError, match="must drop some but not all"):
        exp31.initialize_regrid_root(tmp_path / "nothing_left", run, root, pickle, pmf, tau_floor=1.0, resume=False)
    # A run whose saved sets the replay cannot reproduce, with consistent hashes, is refused.
    tampered = tmp_path / "tampered"
    shutil.copytree(run, tampered)
    leaf = SMOKE.leaf(tampered)
    name = "imputations=2/random_identified_sets.parquet"
    sets = pd.read_parquet(leaf / name)
    sets.loc[sets.draw_index.eq(0) & sets.nonempty, "lower"] -= 0.01
    sets.to_parquet(leaf / name)
    completion = json.loads((leaf / "completion.json").read_text())
    completion["artifact_sha256"][name] = exp31._sha256(leaf / name)
    (leaf / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(RuntimeError, match="conditions failed"):
        exp31.run_regrid(tmp_path / "tampered_regrid", tampered, root, pickle, pmf, tau_floor=0.075)
    failure = json.loads((SMOKE.leaf(tmp_path / "tampered_regrid") / "completion.json").read_text())
    assert "does not reproduce its random sets" in failure["error"]


def test_predictive_resampling_keeps_nesting_and_the_randomized_rate() -> None:
    """A simulated source site keeps its rows, regimes and nesting, and its randomized rate in expectation.

    Idle units draw from the fit's idle rate and the other randomized units from
    the complementary rate, so the randomized arm's mean outcome is the fit's
    ``P(Y=1 | do(x), z)``.  An idle row nested in a randomized unit keeps that
    unit's outcome, and the target redraws its treatment.
    """

    z = (0, 0, 0)
    rows = []
    for unit in range(400):
        x = unit % 2
        rows.append({"id": unit, "F": "on", "iv": x, "dv_binary": 0, **dict(zip(exp31.COVARIATES, z))})
        if unit < 200:  # half of the randomized units are also idle rows
            rows.append({"id": unit, "F": "idle", "iv": x, "dv_binary": 0, **dict(zip(exp31.COVARIATES, z))})
    rows = pd.DataFrame(rows)
    p_y = {0: 0.2, 1: 0.6}
    p_do = {0: 0.3, 1: 0.7}
    center = bayes.SitePMFDraw(
        sample_id="fit", site_id="s", covariates=tuple(exp31.COVARIATES), z_values=(z,),
        idle_xy_given_z={(z, x, y): 0.5 * (p_y[x] if y else 1.0 - p_y[x]) for x in (0, 1) for y in (0, 1)},
        do_y_given_xz={(z, x, y): p_do[x] if y else 1.0 - p_do[x] for x in (0, 1) for y in (0, 1)},
        idle_z={z: 1.0}, n_idle_z={z: 200}, n_idle_xz={(z, 0): 100, (z, 1): 100}, overlap_model=True,
        alpha=float("nan"),
    )
    rng = np.random.default_rng(0)
    means = {0: [], 1: []}
    for _ in range(200):
        simulated = exp31._simulate_site_rows(rows, center, rng, redraw_treatment=False)
        on = simulated.loc[simulated.F.eq("on")].set_index("id")
        idle = simulated.loc[simulated.F.eq("idle")].set_index("id")
        assert (idle.dv_binary == on.dv_binary.reindex(idle.index)).all()
        assert simulated[["id", "F", "iv"]].equals(rows[["id", "F", "iv"]])
        for x in (0, 1):
            means[x].append(on.loc[on.iv.eq(x)].dv_binary.mean())
    for x in (0, 1):
        assert np.mean(means[x]) == pytest.approx(p_do[x], abs=0.01)
    target = exp31._simulate_site_rows(rows.loc[rows.F.eq("idle")].reset_index(drop=True), center, rng,
                                       redraw_treatment=True)
    assert not target.iv.equals(rows.loc[rows.F.eq("idle")].iv.reset_index(drop=True))
