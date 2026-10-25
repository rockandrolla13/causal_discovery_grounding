"""Tests for cut-Bayesian Naturalness probability and random-set plumbing."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from causal_grounding import bayesian_modular_naturalness as module
from causal_grounding.bayesian_modular_naturalness import (
    GP_NUGGET,
    SitePMFDraw,
    ToleranceSurfacePosterior,
    _fit_one_surface_numpyro,
    compute_source_evidence,
    draw_random_identified_sets,
    draw_raw_pmfs,
    draw_rct_reference_posterior,
    is_within_source_support,
    posterior_outer_envelopes,
    raw_pmf_draws_to_frame,
    sample_tolerance_surface,
    score_random_sets_against_rct,
)


def _overlap_rows(sites: tuple[str, ...] = ("source",)) -> pd.DataFrame:
    """Build complete binary on data with a nested idle subset."""

    rows = []
    for site in sites:
        for z_clean in (0, 1):
            for z_bad in (0, 1):
                for treatment in (0, 1):
                    for repetition in range(8):
                        outcome = int((treatment + z_clean + repetition) % 2)
                        row_id = f"{site}:{z_clean}{z_bad}{treatment}:{repetition}"
                        common = {
                            "id": row_id,
                            "site": site,
                            "iv": treatment,
                            "dv_binary": outcome,
                            "z_clean": z_clean,
                            "z_bad": z_bad,
                        }
                        rows.append({**common, "F": "on"})
                        if repetition % 2 == 0:
                            rows.append({**common, "F": "idle"})
    return pd.DataFrame(rows)


def _target_rows() -> pd.DataFrame:
    """Build supported target-only idle rows."""

    frame = _overlap_rows(("target",))
    return frame.loc[frame.F.eq("idle")].copy()


def _fake_surface(
    candidate: str,
    kind: str,
    imputation_id: str = "source-imputation",
    latent_value: float = 0.0,
) -> ToleranceSurfacePosterior:
    """Return a deterministic posterior with support covering the unit cube."""

    if kind == "c":
        features = np.asarray([[0.0], [1.0]])
    else:
        features = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    n_source = len(features)
    n_draws = 20
    return ToleranceSurfacePosterior(
        candidate=candidate,
        tolerance_kind=kind,
        feature_names=("i_deviation",) if kind == "c" else ("i_deviation", "d_strength"),
        source_features=features,
        source_omega=np.full(n_source, 0.01),
        posterior_samples={
            "eta": np.ones(n_draws),
            "length_scale": np.full((n_draws, features.shape[1]), 0.5),
            "u": np.full((n_draws, n_source), latent_value),
            "source_tolerance": np.full((n_draws, n_source), 0.5),
        },
        diagnostics={"passed": True},
        imputation_id=imputation_id,
    )


def _fake_mixture(latent_value: float = 0.0) -> dict[tuple[str, str], ToleranceSurfacePosterior]:
    """Return fake c/d surfaces for both candidates."""

    return {
        (candidate, kind): _fake_surface(candidate, kind, latent_value=latent_value)
        for candidate in ("z_clean", "z_bad")
        for kind in ("c", "d")
    }


def test_overlap_draw_is_coherent_normalized_and_reproducible() -> None:
    """Nested archive rows use one posterior draw rather than two likelihoods."""

    data = _overlap_rows(("a", "b"))
    first = draw_raw_pmfs(data, n_draws=2, random_seed=7)
    second = draw_raw_pmfs(data, n_draws=2, random_seed=7)
    assert [draw["a"].sample_id for draw in first] == [draw["a"].sample_id for draw in second]
    site = first[0]["a"]
    assert site.overlap_model
    for z in site.z_values:
        assert sum(site.idle_xy_given_z[(z, x, y)] for x in (0, 1) for y in (0, 1)) == pytest.approx(1.0)
        for x in (0, 1):
            assert sum(site.do_y_given_xz[(z, x, y)] for y in (0, 1)) == pytest.approx(1.0)
    serialized = raw_pmf_draws_to_frame(first)
    assert set(serialized["table"]) == {"idle_z", "idle_xy_given_z", "do_y_given_xz"}


def test_partial_physical_overlap_is_rejected() -> None:
    """Ambiguous mixtures of nested and independent regimes cannot be fit."""

    data = _overlap_rows()
    idle = data.loc[data.F.eq("idle")].copy()
    extra = idle.iloc[[0]].copy()
    extra["id"] = "idle-only"
    with pytest.raises(ValueError, match="Partially overlapping"):
        draw_raw_pmfs(pd.concat([data, extra], ignore_index=True), n_draws=1)


def test_source_evidence_is_one_solver_scale_row_per_candidate() -> None:
    """Evidence is not duplicated over epsilon/tau grid points or halved."""

    draw = draw_raw_pmfs(_overlap_rows(), n_draws=1, random_seed=11)[0]["source"]
    evidence = compute_source_evidence(draw)
    assert len(evidence) == 2
    assert set(evidence.candidate) == {"z_clean", "z_bad"}
    assert evidence.required_two_c.between(0.0, 1.0).all()
    assert evidence.required_two_d.between(0.0, 1.0).all()


def test_source_evidence_pair_filter_selects_the_pairs_and_keeps_the_triggers() -> None:
    """A pair filter restricts the requirements' maxima but never the trigger statistics.

    Keeping every pair reproduces the unfiltered evidence, keeping none leaves
    ``NaN`` requirements, and keeping one pair gives that pair's own maxima;
    ``I`` and ``D`` are unchanged throughout.
    """

    draw = draw_raw_pmfs(_overlap_rows(), n_draws=1, random_seed=11)[0]["source"]
    full = compute_source_evidence(draw)
    pd.testing.assert_frame_equal(compute_source_evidence(draw, lambda z, z_prime: True), full)
    none = compute_source_evidence(draw, lambda z, z_prime: False)
    assert none.required_two_c.isna().all() and none.required_two_d.isna().all()
    assert (none.n_comparison_pairs == 0).all()
    pd.testing.assert_series_equal(none.i_deviation, full.i_deviation)
    pd.testing.assert_series_equal(none.d_strength, full.d_strength)
    first = next(module._candidate_pairs(draw.z_values, 0))
    single = compute_source_evidence(draw, lambda z, z_prime: (z, z_prime) == first).iloc[0]
    expected_d = max(abs(draw.p_do_y_given_xz(y, star, first[0]) - draw.p_do_y_given_xz(y, star, first[1]))
                     for star in (0, 1) for y in (0, 1))
    assert single.required_two_d == pytest.approx(expected_d)
    assert single.required_two_d <= full.iloc[0].required_two_d
    assert single.required_two_c <= full.iloc[0].required_two_c


def test_source_support_uses_convex_hull_not_bounding_box() -> None:
    """Two-dimensional target queries outside the source hull abstain."""

    triangle = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert is_within_source_support(triangle, (0.2, 0.2))
    assert not is_within_source_support(triangle, (0.8, 0.8))


def test_random_sets_pair_ids_and_include_joint_ate() -> None:
    """Naturalness and Manski share PMF draws and the ATE is solved jointly."""

    target = [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=6, random_seed=3)]
    result = draw_random_identified_sets(
        target,
        [_fake_mixture()],
        n_draws=6,
        epsilon=1.0,
        tau=0.0,
        min_cell_count=1,
        random_seed=9,
    )
    by_method = result.draws.groupby("method").sample_id.apply(set)
    assert by_method["bayesian_naturalness"] == by_method["bayesian_manski"]
    assert set(result.draws.estimand_type) == {"cate", "ate"}
    assert (result.draws.loc[result.draws.estimand_type.eq("ate"), "estimand_key"] == "site_average").all()
    envelopes = posterior_outer_envelopes(result.draws)
    assert envelopes.feasibility_probability.between(0.0, 1.0).all()
    defined = envelopes.dropna(subset=["outer_lower", "outer_upper"])
    assert (defined.outer_upper >= defined.outer_lower).all()
    infeasible = result.draws.loc[
        result.draws.method.eq("bayesian_naturalness") & ~result.draws.feasible
    ]
    assert not infeasible.empty
    assert not infeasible.used_manski_fallback.any()


def test_no_trigger_is_manski_fallback_not_failure() -> None:
    """A draw without an eligible discovery reproduces its matched Manski set."""

    target = [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=3, random_seed=13)]
    result = draw_random_identified_sets(
        target,
        [_fake_mixture()],
        n_draws=3,
        epsilon=0.0,
        tau=1.0,
        min_cell_count=1,
    )
    naturalness = result.draws.loc[result.draws.method.eq("bayesian_naturalness")].sort_values(
        ["sample_id", "estimand_type", "estimand_key"]
    )
    manski = result.draws.loc[result.draws.method.eq("bayesian_manski")].sort_values(
        ["sample_id", "estimand_type", "estimand_key"]
    )
    assert naturalness.used_manski_fallback.all()
    np.testing.assert_allclose(naturalness.lower, manski.lower)
    np.testing.assert_allclose(naturalness.upper, manski.upper)
    references = draw_rct_reference_posterior(
        _target_rows(),
        target,
        n_draws=3,
        min_arm_count=1,
        random_seed=19,
    )
    _, metrics = score_random_sets_against_rct(result.draws, references)
    assert 0.0 <= metrics["rct_reference_membership"] <= 1.0
    assert 0.0 <= metrics["rct_reference_95_interval_outer_containment_mass"] <= 1.0
    assert isinstance(metrics["ate_reference_95_interval_outer_contained"], bool)


def test_rct_reference_rejects_target_draw_without_mass_on_evaluation_strata() -> None:
    """A target draw sharing no strata with the RCT cannot weight an ATE."""

    target = [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=1, random_seed=13)]
    disjoint: SitePMFDraw = replace(target[0], idle_z={(9, 9): 1.0})
    with pytest.raises(ValueError, match="zero target posterior mass"):
        draw_rct_reference_posterior(_target_rows(), [disjoint], n_draws=1, random_seed=19)


def test_numpyro_model_enforces_hard_support_when_available() -> None:
    """A tiny optional NUTS smoke keeps every latent tolerance above omega."""

    pytest.importorskip("numpyro")

    posterior = _fit_one_surface_numpyro(
        candidate="z_clean",
        tolerance_kind="c",
        features=np.asarray([[0.1], [0.3], [0.6]]),
        omega=np.asarray([0.05, 0.10, 0.15]),
        imputation_id="smoke",
        random_seed=1,
        num_chains=1,
        num_warmup=20,
        num_samples=20,
        target_accept_prob=0.8,
        progress_bar=False,
    )
    assert np.all(posterior.posterior_samples["source_tolerance"] >= posterior.source_omega - 1e-10)


def test_target_prediction_uses_fit_nugget_correlation() -> None:
    """A target perfectly correlated with the sources keeps its nugget variance.

    With an effectively infinite length scale, ``K`` is all ones, so the joint
    latent correlation of two sources and the target is ``(1 - nu) J + nu I``.
    Gaussian conditioning then gives a closed-form target mean and variance.
    """

    surface = _fake_surface("z_clean", "c", latent_value=0.8)
    surface.posterior_samples["length_scale"] = np.full((20, 1), 1e6)
    draws = np.asarray([
        sample_tolerance_surface(surface, (0.5,), random_seed=seed, posterior_index=0)
        for seed in range(4000)
    ])
    latent = ndtri(draws)
    shrink = (1.0 - GP_NUGGET) * 2.0 / (2.0 - GP_NUGGET)
    assert latent.mean() == pytest.approx(shrink * 0.8, abs=0.01)
    assert latent.var() == pytest.approx(1.0 - (1.0 - GP_NUGGET) * shrink, rel=0.1)


def test_learned_marginal_location_is_sampled_kept_and_conditioned_on(monkeypatch) -> None:
    """A learned location samples ``mu``, keeps hard support, and centres prediction on ``mu``.

    Far from every source, the conditional latent mean is the draw's ``mu``
    (zero for the paper's uniform-marginal prior).  Every attempt receives the
    option, and an unknown location is refused.
    """

    pytest.importorskip("numpyro")
    posterior = _fit_one_surface_numpyro(
        candidate="z_clean", tolerance_kind="c", features=np.asarray([[0.1], [0.3], [0.6]]),
        omega=np.asarray([0.05, 0.10, 0.15]), imputation_id="smoke", random_seed=1, num_chains=1, num_warmup=20,
        num_samples=20, target_accept_prob=0.8, progress_bar=False, marginal_location="learned",
    )
    assert posterior.posterior_samples["mu"].shape == (20,)
    assert np.all(posterior.posterior_samples["source_tolerance"] >= posterior.source_omega - 1e-10)

    surface = _fake_surface("z_clean", "c", latent_value=0.8)
    far = np.asarray([1e6])
    assert module._conditional_latent_moments(surface, far, 0)[0] == pytest.approx(0.0, abs=1e-12)
    surface.posterior_samples["mu"] = np.full(20, -1.3)
    mean, variance = module._conditional_latent_moments(surface, far, 0)
    assert mean == pytest.approx(-1.3, abs=1e-12)
    assert variance == pytest.approx(1.0, abs=1e-12)

    seen: list[str] = []

    def fake_fit(**kwargs):
        """Record the location and fail every attempt."""

        seen.append(kwargs["marginal_location"])
        return SimpleNamespace(diagnostics={"passed": False})

    monkeypatch.setattr(module, "_fit_one_surface_numpyro", fake_fit)
    evidence = pd.DataFrame({
        "sample_id": "imputation", "site": ["a", "b"], "candidate": "z_clean", "i_deviation": [0.1, 0.2],
        "d_strength": [0.3, 0.4], "required_two_c": [0.2, 0.3], "required_two_d": [0.1, 0.2],
    })
    second = {"num_warmup": 8, "num_samples": 8, "target_accept_prob": 0.95}
    module.fit_tolerance_surfaces(evidence, on_unconverged="exclude", second_retry=second, marginal_location="learned")
    assert seen == ["learned"] * 6
    monkeypatch.undo()
    with pytest.raises(ValueError, match="marginal_location"):
        _fit_one_surface_numpyro(
            candidate="z_clean", tolerance_kind="c", features=np.asarray([[0.1], [0.3]]),
            omega=np.asarray([0.05, 0.10]), imputation_id="smoke", random_seed=1, num_chains=1, num_warmup=5,
            num_samples=5, target_accept_prob=0.8, progress_bar=False, marginal_location="empirical",
        )


def test_numpyro_fit_mixes_on_near_duplicate_source_statistics() -> None:
    """Regression for a real archive fit in which every chain froze.

    The evidence is the invalid candidate's ``2c`` evidence from the first raw
    PMF imputation of a Causal Chambers archive condition (seed=20260921,
    beta=0.5, target=uniform_l_11_mid), taken from an experiment since removed
    from the repository.  Under float32 and a 1e-6 jitter, NUTS adapted a
    step size of about 1e-7, giving R-hat in the millions and ESS of two.
    """

    pytest.importorskip("numpyro")

    features = np.asarray([
        0.611081, 0.63768, 0.679444, 0.695623, 0.697605, 0.70167,
        0.704991, 0.724154, 0.725562, 0.756489, 0.757163,
    ])[:, None]
    omega = np.asarray([
        0.695511, 0.690651, 0.780305, 0.764188, 0.742821, 0.70167,
        0.750317, 0.859309, 0.810134, 0.756489, 0.793707,
    ])
    posterior = _fit_one_surface_numpyro(
        candidate="z_bad",
        tolerance_kind="c",
        features=features,
        omega=omega,
        imputation_id="archive-chain-freeze-regression",
        random_seed=3,
        num_chains=2,
        num_warmup=500,
        num_samples=500,
        target_accept_prob=0.90,
        progress_bar=False,
    )
    assert posterior.posterior_samples["u"].dtype == np.float64
    assert posterior.diagnostics["n_divergent"] == 0
    assert posterior.diagnostics["max_r_hat"] < 1.05
    assert posterior.diagnostics["min_bulk_ess"] > 100


def test_jax_caches_are_cleared_once_per_interval(monkeypatch) -> None:
    """Per-fit programs are released periodically, not after every fit."""

    clears: list[int] = []
    fake_jax = SimpleNamespace(clear_caches=lambda: clears.append(1))
    monkeypatch.setattr(module, "_JAX_CACHE_CLEARER", module._PeriodicJaxCacheClearer())
    for _ in range(2 * module.JAX_CACHE_CLEAR_INTERVAL + 3):
        module._clear_jax_caches_periodically(fake_jax)
    assert len(clears) == 2
    assert module._JAX_CACHE_CLEARER.fits_since_clear == 3
    module._JAX_CACHE_CLEARER.reset()
    assert module._JAX_CACHE_CLEARER.fits_since_clear == 0


def test_retry_uses_its_own_retained_draw_count(monkeypatch) -> None:
    """A failed first attempt is refitted with the retry warm-up and draw count."""

    calls: list[tuple[int, int]] = []

    def fake_fit(**kwargs):
        """Record the budget and fail every first attempt."""

        calls.append((kwargs["num_warmup"], kwargs["num_samples"]))
        return SimpleNamespace(diagnostics={"passed": kwargs["num_warmup"] == 4000})

    monkeypatch.setattr(module, "_fit_one_surface_numpyro", fake_fit)
    evidence = pd.DataFrame({
        "sample_id": "imputation",
        "site": ["a", "b"],
        "candidate": "z_clean",
        "i_deviation": [0.1, 0.2],
        "d_strength": [0.3, 0.4],
        "required_two_c": [0.2, 0.3],
        "required_two_d": [0.1, 0.2],
    })
    budget = {"num_warmup": 2000, "num_samples": 3000, "retry_warmup": 4000}
    module.fit_tolerance_surfaces(evidence, **budget, retry_num_samples=6000)
    assert calls == [(2000, 3000), (4000, 6000)] * 2
    calls.clear()
    module.fit_tolerance_surfaces(evidence, **budget)
    assert calls == [(2000, 3000), (4000, 3000)] * 2


def _surfaces_far_from(candidates: tuple[str, ...] = ("z_clean", "z_bad")) -> dict[tuple[str, str], ToleranceSurfacePosterior]:
    """Return fake surfaces whose source support excludes low trigger statistics."""

    mixture = {}
    for candidate in candidates:
        for kind in ("c", "d"):
            surface = _fake_surface(candidate, kind)
            if kind == "c":
                surface.source_features = np.asarray([[2.0], [3.0]])
            else:
                surface.source_features = np.asarray([[2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0]])
            mixture[(candidate, kind)] = surface
    return mixture


def test_grid_gate_fires_on_any_grid_pair_and_records_the_tightest() -> None:
    """A candidate that misses a single threshold still triggers on a grid pair."""

    target = [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=4, random_seed=21)]
    single = draw_random_identified_sets(target, [_fake_mixture()], n_draws=4, epsilon=0.0, tau=1.0, min_cell_count=1)
    assert not single.triggers.triggered.any()
    assert "trigger_epsilon" not in single.triggers
    grid = draw_random_identified_sets(
        target, [_fake_mixture()], n_draws=4, epsilon=0.0, tau=1.0, min_cell_count=1,
        epsilons=(0.0, 1.0), taus=(0.0, 1.0),
    )
    triggers = grid.triggers
    assert triggers.triggered.all()
    np.testing.assert_allclose(
        triggers.trigger_epsilon,
        np.where(triggers.i_deviation <= 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        triggers.trigger_tau,
        np.where(triggers.d_strength >= 1.0, 1.0, 0.0),
    )
    with pytest.raises(ValueError, match="together"):
        draw_random_identified_sets(target, [_fake_mixture()], n_draws=1, epsilons=(0.1,))


def test_source_support_can_be_waived_for_extrapolated_reads() -> None:
    """Without the hull rule a triggered candidate outside support stays eligible."""

    surface = _surfaces_far_from()[("z_clean", "c")]
    assert np.isnan(sample_tolerance_surface(surface, (0.1,), random_seed=1))
    assert np.isfinite(sample_tolerance_surface(surface, (0.1,), random_seed=1, allow_extrapolation=True))
    target = [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=3, random_seed=23)]
    common = dict(n_draws=3, epsilon=1.0, tau=0.0, min_cell_count=1, random_seed=5)
    strict = draw_random_identified_sets(target, [_surfaces_far_from()], **common)
    assert strict.triggers.triggered.all() and not strict.triggers.in_source_support.any()
    assert not strict.triggers.eligible.any()
    waived = draw_random_identified_sets(target, [_surfaces_far_from()], require_source_support=False, **common)
    assert waived.triggers.eligible.all()
    assert np.isfinite(waived.triggers[["two_c", "two_d"]].to_numpy()).all()


def test_draw_rows_record_indices_statuses_and_posterior_draws() -> None:
    """Every row traces its draw, and empty sets are told apart from failures."""

    target = [next(iter(item.values())) for item in draw_raw_pmfs(_target_rows(), n_draws=4, random_seed=3)]
    mixtures = [_fake_mixture(latent_value=-8.0), _fake_mixture(latent_value=-8.0)]
    result = draw_random_identified_sets(target, mixtures, n_draws=4, min_cell_count=1, epsilons=(1.0,), taus=(0.0,))
    draws, triggers = result.draws, result.triggers
    assert draws.groupby("draw_index").sample_id.nunique().eq(1).all()
    assert (draws.source_imputation_index == draws.draw_index % 2).all()
    assert (draws.target_pmf_draw_index == draws.draw_index % 4).all()
    naturalness = draws.loc[draws.method.eq("bayesian_naturalness")]
    assert naturalness.lower_status.eq("infeasible").all() and not naturalness.feasible.any()
    assert draws.loc[draws.method.eq("bayesian_manski"), ["lower_status", "upper_status"]].eq("optimal").all().all()
    drawn = triggers.loc[triggers.eligible]
    assert drawn.c_posterior_index.between(0, 19).all() and drawn.d_posterior_index.between(0, 19).all()
    assert set(triggers.columns) >= {"draw_index", "target_pmf_draw_index", "source_imputation_index"}


def test_return_index_keeps_the_tolerance_stream() -> None:
    """Asking for the posterior index never changes the drawn tolerance."""

    surface = _fake_surface("z_clean", "c", latent_value=0.3)
    plain = sample_tolerance_surface(surface, (0.4,), random_seed=17)
    value, index = sample_tolerance_surface(surface, (0.4,), random_seed=17, return_index=True)
    assert value == plain and 0 <= index < 20
    far = _surfaces_far_from()[("z_clean", "c")]
    refused, missing = sample_tolerance_surface(far, (0.1,), random_seed=1, return_index=True)
    assert np.isnan(refused) and missing == -1


def test_grid_trigger_fires_on_any_pair_and_reports_the_tightest() -> None:
    """The shared trigger rule fires on any grid pair and never on NaN."""

    assert module.grid_trigger(0.06, 0.12, (0.05, 0.1, 0.5), (0.01, 0.1, 0.2)) == (True, 0.1, 0.1)
    assert module.grid_trigger(0.6, 0.5, (0.05, 0.5), (0.01,))[0] is False
    triggered, epsilon, tau = module.grid_trigger(float("nan"), 0.5, (0.5,), (0.01,))
    assert not triggered and np.isnan(epsilon) and np.isnan(tau)


def test_held_out_scores_are_exact_for_an_uninformative_surface() -> None:
    """Far from every source the predictive tolerance is Uniform(0, 1).

    With latent sources at zero and a vanishing length scale the query's latent
    is standard normal, so ``P(tolerance >= omega) = 1 - omega`` exactly, and the
    predictive density of omega is the integral of the likelihood over c.
    """

    from scipy.integrate import quad

    surface = _fake_surface("z_clean", "c", latent_value=0.0)
    surface.posterior_samples["length_scale"] = np.full((20, 1), 1e-6)
    surface.posterior_samples = {name: np.repeat(value, 200, axis=0) for name, value in surface.posterior_samples.items()}
    omega = 0.3
    scores = module.score_held_out_tolerance(surface, (0.5,), omega, max_posterior_draws=4000, random_seed=4)
    assert scores["validity_probability"] == pytest.approx(1.0 - omega, abs=1e-9)
    expected = quad(lambda c: np.exp(omega) / np.expm1(c), omega, 1.0)[0]
    assert np.exp(scores["log_predictive_density"]) == pytest.approx(expected, rel=0.08)
    assert 0.0 <= scores["predictive_tolerance_q05"] <= scores["predictive_tolerance_q95"] <= 1.0


def test_leave_one_source_out_refits_without_each_site(monkeypatch) -> None:
    """Each site is scored by a refit that never saw it; a failed refit is kept."""

    fitted_sites: list[set[str]] = []

    def fake_fit(evidence: pd.DataFrame, **kwargs) -> dict:
        """Record which sites a refit saw; fail the refit that omits site c."""

        sites = set(evidence.site)
        fitted_sites.append(sites)
        if "c" not in sites:
            raise RuntimeError("NUTS diagnostics failed after retry")
        return _fake_mixture()

    monkeypatch.setattr(module, "fit_tolerance_surfaces", fake_fit)
    evidence = pd.DataFrame({
        "sample_id": "imputation",
        "site": np.repeat(["a", "b", "c"], 2),
        "candidate": ["z_clean", "z_bad"] * 3,
        "i_deviation": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "d_strength": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "required_two_c": 0.4,
        "required_two_d": 0.3,
    })
    scores = module.leave_one_source_out_scores(
        evidence, target_queries={"z_clean": (0.5, 0.5), "z_bad": (0.5, 0.5)}, max_posterior_draws=10
    )
    assert fitted_sites == [{"b", "c"}, {"a", "c"}, {"a", "b"}]
    assert len(scores) == 3 * 2 * 2
    held_c = scores.loc[scores.held_out_site.eq("c")]
    assert held_c.fit_error.str.contains("diagnostics failed").all() and not held_c.scored.any()
    scored = scores.loc[scores.scored]
    assert scored.validity_probability.between(0.0, 1.0).all()
    assert scored.target_predictive_q50.between(0.0, 1.0).all()


def test_second_retry_then_exclusion_never_returns_an_unconverged_surface(monkeypatch) -> None:
    """A surface failing every attempt is excluded and reported, or raises by default."""

    calls: list[tuple[int, int, str]] = []

    def failing_fit(**kwargs):
        """Fail the gate on every attempt, recording each attempt's budget."""

        calls.append((kwargs["num_warmup"], kwargs["num_samples"], kwargs["candidate"]))
        return SimpleNamespace(diagnostics={"passed": False, "max_r_hat": 1.02, "num_warmup": kwargs["num_warmup"],
                                            "num_samples": kwargs["num_samples"]})

    monkeypatch.setattr(module, "_fit_one_surface_numpyro", failing_fit)
    evidence = pd.DataFrame({
        "sample_id": "imputation", "site": ["a", "b"], "candidate": "z_clean", "i_deviation": [0.1, 0.2],
        "d_strength": [0.3, 0.4], "required_two_c": [0.2, 0.3], "required_two_d": [0.1, 0.2],
    })
    budget = {"num_warmup": 20, "num_samples": 30, "retry_warmup": 40, "retry_num_samples": 60,
              "second_retry": {"num_warmup": 80, "num_samples": 240, "target_accept_prob": 0.95}}
    unconverged: list[dict] = []
    fitted = module.fit_tolerance_surfaces(evidence, on_unconverged="exclude", unconverged=unconverged, **budget)
    assert fitted == {}
    assert [(entry["candidate"], entry["tolerance_kind"], entry["attempt"]) for entry in unconverged] == [
        ("z_clean", "c", "second_retry"), ("z_clean", "d", "second_retry")]
    assert [call[:2] for call in calls] == [(20, 30), (40, 60), (80, 240)] * 2
    with pytest.raises(RuntimeError, match="failed after retry"):
        module.fit_tolerance_surfaces(evidence, **budget)
    with pytest.raises(ValueError, match="on_unconverged"):
        module.fit_tolerance_surfaces(evidence, on_unconverged="ignore")


def test_held_out_subsets_reproduce_the_full_leave_one_out_scores(monkeypatch) -> None:
    """Per-site calls, as process isolation makes them, equal one full call."""

    monkeypatch.setattr(module, "fit_tolerance_surfaces", lambda evidence, **kwargs: _fake_mixture(latent_value=0.3))
    evidence = pd.DataFrame({
        "sample_id": "imputation",
        "site": np.repeat(["a", "b", "c", "d"], 2),
        "candidate": ["z_clean", "z_bad"] * 4,
        "i_deviation": np.linspace(0.2, 0.7, 8),
        "d_strength": np.linspace(0.3, 0.8, 8),
        "required_two_c": 0.4,
        "required_two_d": 0.3,
    })
    options = {"target_queries": {"z_clean": (0.5, 0.5), "z_bad": (0.4, 0.6)}, "max_posterior_draws": 10, "random_seed": 7}
    full = module.leave_one_source_out_scores(evidence, **options)
    parts = pd.concat(
        [module.leave_one_source_out_scores(evidence, held_out_sites=[site], **options) for site in "abcd"],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(full, parts)
    with pytest.raises(ValueError, match="Unknown held-out sites"):
        module.leave_one_source_out_scores(evidence, held_out_sites=["z"], **options)


class _BootstrapClassifier:
    """TabICLv2 stand-in: predicts the fitted rows' target rate, so resamples move it."""

    def __init__(self, **kwargs):
        """Record the options a TabICLv2 classifier would receive."""

        self.kwargs = kwargs
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """Store the target rate of the fitted rows."""

        self.rate_ = float(np.clip(np.mean(y), 0.05, 0.95))
        return self

    def predict_proba(self, X):
        """Return the stored rate for every row."""

        rows = len(np.asarray(X))
        return np.column_stack([np.full(rows, 1.0 - self.rate_), np.full(rows, self.rate_)])


@pytest.fixture
def bootstrap_tabicl(monkeypatch):
    """Route TabICLv2 predictors to the stand-in without loading weights."""

    from causal_grounding import predictors

    monkeypatch.setattr(predictors, "_TABICL_AVAILABLE", True)
    monkeypatch.setattr(predictors, "TabICLClassifier", _BootstrapClassifier)


def test_tabicl_bootstrap_draws_are_seeded_and_keep_the_observed_support(bootstrap_tabicl) -> None:
    """Draws reproduce under a seed, vary across draws, and keep physical support.

    Sources get randomized tables and the nesting flag; the target does not.
    """

    data = _overlap_rows(("a", "b"))
    first = module.draw_tabicl_bootstrap_pmfs(data, n_draws=3, random_seed=4)
    second = module.draw_tabicl_bootstrap_pmfs(data, n_draws=3, random_seed=4)
    assert [draw["a"].idle_xy_given_z for draw in first] == [draw["a"].idle_xy_given_z for draw in second]
    assert first[0]["a"].idle_xy_given_z != first[1]["a"].idle_xy_given_z
    assert first[0]["a"].sample_id == first[0]["b"].sample_id != first[1]["a"].sample_id
    draw = first[0]["a"]
    assert draw.overlap_model and draw.do_y_given_xz and np.isnan(draw.alpha)
    idle = data.loc[data.site.eq("a") & data.F.eq("idle")]
    assert set(draw.z_values) == set(idle[["z_clean", "z_bad"]].itertuples(index=False, name=None))
    assert sum(draw.n_idle_z.values()) == len(idle)
    target = module.draw_tabicl_bootstrap_pmfs(_target_rows(), n_draws=2, random_seed=1)[0]["target"]
    assert not target.overlap_model and not target.do_y_given_xz
    assert sum(target.idle_z.values()) == pytest.approx(1.0)
    evidence = compute_source_evidence(draw)
    assert evidence.required_two_c.between(0.0, 1.0).all() and evidence.required_two_d.between(0.0, 1.0).all()


def test_stored_pmf_draws_refuse_rows_they_were_not_drawn_from(bootstrap_tabicl) -> None:
    """The stored table rebuilds its draws and rejects rows with other strata."""

    data = _overlap_rows()
    draws = module.draw_tabicl_bootstrap_pmfs(data, n_draws=2, random_seed=9)
    frame = raw_pmf_draws_to_frame(draws)
    rebuilt = module.raw_pmf_draws_from_frame(frame, data)
    assert [draw["source"].idle_xy_given_z for draw in rebuilt] == [draw["source"].idle_xy_given_z for draw in draws]
    assert rebuilt[1]["source"].n_idle_xz == draws[1]["source"].n_idle_xz
    fewer = data.loc[~(data.z_clean.eq(1) & data.z_bad.eq(1))]
    with pytest.raises(ValueError, match="differ from its observed idle strata"):
        module.raw_pmf_draws_from_frame(frame, fewer)
