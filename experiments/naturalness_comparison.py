"""Fitting engine for the Exp28 method comparison on Twins.

For every (seed, beta, target site) condition it builds the randomized and confounded regimes,
fits Naturalness and the competitor arms on the target's reporting cells, sends every failed or
unsupported cell to the shared Manski fallback, and scores widths, sign decisions and containment
of the reference effect. Per-cell arms: Manski, Naturalness, the B-Learner at Gamma=exp(beta) and
at the oracle Gamma, Balke-Pearl IV bounds and the CB meta learners.

Naturalness endpoints are means of LP endpoints over GP tolerance draws: plug-in bounds, not
credible intervals. Site-average endpoints are weighted sums of cell endpoints, which outer-bound
the joint identified set, and containment of the reference in one realization is descriptive, not
coverage.

:mod:`experiments.exp28_twins_method_comparison` supplies the grid and is the entry point that
``paper/reproduce_experiment2_twins.sh`` runs. Data preparation lives in
:mod:`experiments.naturalness_comparison_data`; merging and reporting in
:mod:`experiments.naturalness_comparison_results`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from causal_grounding import balke_pearl
from causal_grounding import naturalness_statistics as stats
from causal_grounding.lp_solver_learned_naturalness import (
    estimate_target_observed_binary_quantities,
)
from causal_grounding.train_target_split import create_train_target_split
from experiments import exp16_osrct_calibrated_policy as exp16
from experiments import naturalness_comparison_data as data
from experiments.utils import (
    build_configured_naturalness_grounder,
    build_lp_probability_model_kwargs,
    json_default,
    write_run_manifest,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_DATASETS = ("twins_reduced",)
SUPPORTED_DATASETS = ("twins_reduced",)

@dataclass(frozen=True)
class ComparisonBaseConfig:
    """Settings shared by every comparison experiment.

    ``max_fit_rows`` caps each site/regime's fitting rows without inspecting treatment or outcome
    (zero disables the cap), so every arm fits the same target subset; evaluation uses all target
    idle rows. Twins has a single release, so it is loaded once whatever ``dgp_seeds`` holds;
    repetition comes from ``seeds``.
    """

    data_root: str = "data"
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    dgp_seeds: tuple[int, ...] = (20260918,)
    seeds: tuple[int, ...] = (20260918,)
    betas: tuple[float, ...] = (.5, 1.)
    target_sites: tuple[str, ...] = ("0",)
    twins_covariates: tuple[str, ...] = ("birmon", "gestat10", "mager8")
    twins_bias_covariate: str = "birmon"
    stage1_backend: str = "tabicl_v2"
    tabicl_device: str = "auto"
    epsilons: tuple[float, ...] = exp16.Exp16Config().epsilons
    taus: tuple[float, ...] = exp16.Exp16Config().taus
    min_cell_count: int = 20
    max_fit_rows: int = 5000
    lp_posterior_samples: int = 250
    n_jobs: int = 4
    blearner_repository: str = "/opt/BLearner" if Path("/opt/BLearner").exists() else "/tmp/BLearner"
    output_dir: str = "results/naturalness_method_comparison"
    resume: bool = False
    cluster_manifest_path: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid grids, duplicated settings, or unsupported datasets."""
        if not self.datasets or not set(self.datasets) <= set(SUPPORTED_DATASETS):
            raise ValueError(f"Comparison supports only {SUPPORTED_DATASETS}")
        for name in ("datasets", "dgp_seeds", "seeds", "betas", "target_sites"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and contain no duplicates")
        if not all(np.isfinite(b) and 0 <= b < 700 for b in self.betas):
            raise ValueError("betas must be finite, nonnegative, and below 700")
        if self.stage1_backend not in ("tabicl_v2", "hierarchical_dirichlet", "xgboost"):
            raise ValueError("Unsupported Stage-1 backend")
        if self.min_cell_count < 1 or self.max_fit_rows < 0 or self.lp_posterior_samples < 1 or self.n_jobs < 1:
            raise ValueError("Support, draws, and workers must be positive; fitting cap nonnegative")
        if not self.twins_covariates or len(set(self.twins_covariates)) != len(self.twins_covariates):
            raise ValueError("twins_covariates must be nonempty and unique")
        # The selector need not be in the conditioning set: withheld from every method it acts as
        # a hidden confounder, whereas inside it the target CATE would be point identified.
        if not self.twins_bias_covariate:
            raise ValueError("A Twins selector covariate must be named")
        if self.target_sites != ("all",) and any(not s.isdigit() for s in self.target_sites):
            raise ValueError("target_sites must be numeric site codes or the single value all")
        for name in ("epsilons", "taus"):
            if not getattr(self, name) or not all(np.isfinite(v) and 0 <= v <= 1 for v in getattr(self, name)):
                raise ValueError(f"{name} must contain finite probabilities")

def condition_covariates(config: ComparisonBaseConfig, dataset: str) -> tuple[tuple[str, ...], str]:
    """Resolve the conditioning covariates and the selecting covariate for a dataset.

    Every method sees exactly the returned conditioning covariates, so a selecting covariate
    absent from them is withheld from all methods and acts as an unmeasured confounder.

    Args:
        config: Settings supplying the per-dataset conditioning sets.
        dataset: Name of a dataset in :data:`SUPPORTED_DATASETS`.

    Returns:
        The conditioning covariate names and the selecting covariate name; the caller looks the
        selector up in the dataset's full covariate pool, not in the conditioning set.

    Raises:
        ValueError: If ``dataset`` is not supported.
    """
    if dataset == "twins_reduced":
        return tuple(config.twins_covariates), config.twins_bias_covariate
    raise ValueError(f"Unsupported dataset: {dataset}")

def _sha256(path: Path) -> str:
    """Hash a source or input file at ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def dataset_root(config: ComparisonBaseConfig, dataset: str, dgp_seed: int | None) -> Path:
    """Resolve the supplied input root for a dataset and optional DGP draw.

    Twins has a single release, so ``dgp_seed`` does not select anything.

    Raises:
        ValueError: If ``dataset`` is not supported.
    """
    if dataset == "twins_reduced":
        return Path(config.data_root)
    raise ValueError(f"Unsupported dataset: {dataset}")

def prepare_condition(prepared: Mapping[str, Any], dataset: str, beta: float, seed: int,
                      config: ComparisonBaseConfig) -> dict[str, Any]:
    """Construct disjoint regimes and learner frames with a separate truth store.

    Takes the prepared dataset arrays, dataset name, assignment strength ``beta``, seed and
    settings. Regimes are seeded from (dataset, selector index, ``beta``, seed), and the
    per-site/regime fitting cap ignores outcomes and truth. The evaluation store covers the full
    target idle population before capping; no oracle column reaches the fitting frames.
    """
    names = prepared["covariate_names"]
    covariates, bias_name = condition_covariates(config, dataset)
    absent = set(covariates) - set(names)
    if absent:
        raise ValueError(f"Missing observed covariates: {sorted(absent)}")
    if bias_name not in names:
        raise ValueError(f"Selecting covariate {bias_name!r} absent from the {dataset} covariate pool")
    bias = names.index(bias_name)
    if prepared.get("assignment") is not None:
        raise ValueError(f"{dataset} declares a latent assignment mechanism, which this comparison does not support")
    partitions, treatment, outcome = data.build_partitions(
        prepared["sites"], prepared["treatment"], prepared["covariates"], bias, beta, .4,
        np.random.default_rng(data.stable_seed(dataset, bias, beta, seed)),
        potential=prepared["potential"],
        clusters=prepared.get("clusters"), trial_probability=prepared.get("trial_probability"))
    frame = pd.DataFrame(prepared["covariates"][:, [names.index(c) for c in covariates]], columns=covariates)
    frame["id"] = np.arange(len(frame))
    frame["site"] = prepared["sites"].astype(str)
    frame["iv"], frame["dv_binary"] = treatment, outcome
    fitting, evaluation, row_audits = [], {}, []
    for partition in partitions:
        site = str(partition.site)
        idle_rows = partition.idle
        idle_propensity = data.selection_probability(prepared["covariates"][idle_rows, bias], beta)
        potential = tuple(v[idle_rows] for v in prepared["potential"])
        responses = prepared.get("responses")
        effect = (responses[1][idle_rows] - responses[0][idle_rows]) if responses is not None else potential[1].astype(float) - potential[0]
        evaluation[site] = dict(frame=frame.iloc[idle_rows].copy(), effect=effect,
                                potential=potential, propensity=idle_propensity,
                                reference_kind="oracle_response_model" if responses is not None else "paired_benchmark")
        for regime, rows in (("on", partition.rct), ("idle", idle_rows)):
            fit_rows = rows
            if config.max_fit_rows and len(rows) > config.max_fit_rows:
                fit_rows = np.sort(np.random.default_rng(data.stable_seed("stage2-fit-cap", seed, site, regime)).choice(
                    rows, config.max_fit_rows, replace=False))
            fitting.append(frame.iloc[fit_rows].assign(F=regime))
            row_audits.append(dict(site=site, regime=regime, population_rows=len(rows), fitting_rows=len(fit_rows),
                                   fitting_ids_sha256=hashlib.sha256(np.asarray(fit_rows, dtype=np.int64).tobytes()).hexdigest()))
    fitted = pd.concat(fitting, ignore_index=True)
    assert set(fitted.columns) == set(covariates) | {"id", "site", "iv", "dv_binary", "F"}
    assert not set(fitted.loc[fitted.F == "on", "id"]) & set(fitted.loc[fitted.F == "idle", "id"])
    return dict(on=fitted[fitted.F == "on"].copy(), idle=fitted[fitted.F == "idle"].copy(),
                evaluation=evaluation, covariates=covariates, row_audit=pd.DataFrame(row_audits))

def manski_bounds(target_idle: pd.DataFrame, strata: pd.DataFrame,
                  covariates: Sequence[str], min_count: int) -> pd.DataFrame:
    """Compute unsmoothed binary Manski bounds and a shared fitting-support mask.

    ``target_idle`` is the common learner subset; ``strata`` enumerates all
    evaluation cells. Missing fitting cells receive [-1,1]. A missing treatment
    arm is kept as unobserved mass rather than fabricated or dropped. Both arm
    counts must meet ``min_count`` for the shared support mask.
    """
    records = []
    for z in strata[list(covariates)].itertuples(index=False, name=None):
        mask = np.logical_and.reduce([target_idle[c].to_numpy() == v for c, v in zip(covariates, z)])
        cell = target_idle.loc[mask]
        n = len(cell)
        record = dict(zip(covariates, z))
        n0, n1 = int((cell.iv == 0).sum()), int((cell.iv == 1).sum())
        if n:
            a0 = float(cell.loc[cell.iv == 0, "dv_binary"].sum()) / n
            a1 = float(cell.loc[cell.iv == 1, "dv_binary"].sum()) / n
            lower, upper = a1 - a0 - n1 / n, a1 - a0 + n0 / n
        else:
            lower, upper = -1., 1.
        records.append(dict(**record, lower=lower, upper=upper, n_fit=n, n_fit_arm0=n0, n_fit_arm1=n1,
                            common_supported=n0 >= min_count and n1 >= min_count))
    return pd.DataFrame(records)

def _complete_bounds(raw: pd.DataFrame, baseline: pd.DataFrame, covariates: Sequence[str],
                     failure: str = "missing") -> pd.DataFrame:
    """Retain every evaluation cell, replacing invalid/unsupported bounds by Manski.

    ``raw`` contains a method's predictions; ``baseline`` holds the common
    support and fallback endpoints. Raw endpoints/status are preserved. Every
    per-cell arm goes through this same support and failure rule.
    """
    columns = list(covariates)
    raw = raw.copy()
    if raw.empty:
        raw = pd.DataFrame(columns=columns + ["lower", "upper", "status"])
    if raw.duplicated(columns).any():
        raise ValueError("Duplicate method predictions for a target cell")
    if "status" not in raw:
        raw["status"] = "optimal"
    for column in ("lower", "upper"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.rename(columns={"lower": "raw_lower", "upper": "raw_upper", "status": "raw_status"})
    result = baseline.rename(columns={"lower": "manski_lower", "upper": "manski_upper"}).merge(
        raw, on=columns, how="left", validate="one_to_one")
    valid = (result.raw_lower.notna() & result.raw_upper.notna() & np.isfinite(result.raw_lower)
             & np.isfinite(result.raw_upper) & (result.raw_lower <= result.raw_upper)
             & result.raw_status.isin(["optimal", "rearranged"]) & result.common_supported)
    result["used_manski_fallback"] = ~valid
    result["lower"] = result.raw_lower.clip(-1, 1).where(valid, result.manski_lower)
    result["upper"] = result.raw_upper.clip(-1, 1).where(valid, result.manski_upper)
    result["status"] = result.raw_status.fillna(failure).where(result.common_supported, "common_unsupported")
    result["outside_binary_support"] = (result.raw_lower < -1) | (result.raw_upper > 1)
    return result

#: Arms whose cell endpoints are intersected with the assumption-free Manski interval: the
#: B-Learner's Gamma-weighted regressions are not confined to it and overshoot it at large Gamma.
MANSKI_CAPPED_METHODS = ("blearner_exp_beta", "blearner_oracle")

def intersect_with_manski(raw: pd.DataFrame, baseline: pd.DataFrame,
                          covariates: Sequence[str]) -> pd.DataFrame:
    """Intersect one arm's fitted cell bounds with the Manski bounds.

    The fitted endpoints are kept as ``uncapped_lower`` and ``uncapped_upper``, and
    ``manski_capped`` flags a cell whose interval moved. An interval wholly outside Manski
    contradicts the observed data and is marked ``manski_disjoint``, which
    :func:`_complete_bounds` sends to the Manski fallback. A missing endpoint stays missing, so
    the cap cannot turn a failed cell into a valid one.

    Args:
        raw: One arm's per-cell predictions with ``lower``, ``upper`` and optionally ``status``.
        baseline: Manski endpoints keyed by the cells, from :func:`manski_bounds`.
        covariates: Conditioning covariates keying the cells.

    Returns:
        ``raw`` with ``lower``/``upper`` intersected with the Manski interval
        and the three provenance columns added.
    """
    columns = list(covariates)
    manski = baseline[columns + ["lower", "upper"]].rename(
        columns={"lower": "_manski_lower", "upper": "_manski_upper"})
    capped = raw.merge(manski, on=columns, how="left", validate="one_to_one")
    lower = pd.to_numeric(capped["lower"], errors="coerce")
    upper = pd.to_numeric(capped["upper"], errors="coerce")
    capped["uncapped_lower"], capped["uncapped_upper"] = lower, upper
    # np.maximum/np.minimum propagate NaN; fmax/fmin would pass a failed endpoint off as Manski.
    capped["lower"] = np.maximum(lower, capped["_manski_lower"])
    capped["upper"] = np.minimum(upper, capped["_manski_upper"])
    capped["manski_capped"] = (capped["lower"] > lower) | (capped["upper"] < upper)
    if "status" not in capped:
        capped["status"] = "optimal"
    capped.loc[capped["lower"] > capped["upper"], "status"] = "manski_disjoint"
    return capped.drop(columns=["_manski_lower", "_manski_upper"])

def posterior_bounds(samples: pd.DataFrame, baseline: pd.DataFrame,
                     covariates: Sequence[str], n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average endpoints over every requested draw, retaining failures as Manski.

    ``samples`` is Exp16's sample-level LP table. Missing draws and cells are completed with the
    Manski fallback, so keeping only optimal draws cannot shrink the primary interval. Returns the
    primary (averaged) endpoints and all completed draws.
    """
    completed = []
    for sample_id in range(n_samples):
        raw = samples.loc[samples.lp_posterior_sample_id == sample_id] if not samples.empty else pd.DataFrame()
        current = _complete_bounds(raw, baseline, covariates)
        current["lp_posterior_sample_id"] = sample_id
        completed.append(current)
    draws = pd.concat(completed, ignore_index=True)
    summary = draws.groupby(list(covariates), observed=True, dropna=False).agg(
        lower=("lower", "mean"), upper=("upper", "mean"),
        n_requested_draws=("lp_posterior_sample_id", "size"),
        fallback_draw_fraction=("used_manski_fallback", "mean"),
        outside_binary_support_fraction=("outside_binary_support", "mean")).reset_index()
    optimal = draws.assign(optimal=draws.raw_status.eq("optimal")).groupby(
        list(covariates), observed=True, dropna=False).optimal.mean().rename("optimal_draw_fraction").reset_index()
    summary = summary.merge(optimal, on=list(covariates), validate="one_to_one")
    summary["status"] = "optimal"
    primary = _complete_bounds(summary, baseline, covariates)
    primary["used_manski_fallback"] = primary.fallback_draw_fraction.gt(0)
    primary["procedure"] = "all_requested_draw_endpoint_mean_with_manski_fallback"
    return primary, draws

def naturalness_config(config: ComparisonBaseConfig, covariates: Sequence[str], seed: int) -> exp16.Exp16Config:
    """Adapt the comparison settings to the Exp16 inference APIs."""
    return exp16.Exp16Config(
        covariates=tuple(covariates), naturalness_covariates=tuple(covariates),
        lp_probability_covariates=tuple(covariates), seed=seed,
        epsilons=config.epsilons, taus=config.taus, min_cell_count=config.min_cell_count,
        stage1_backend=config.stage1_backend, validation_stage1_backend=config.stage1_backend,
        target_trigger_probability_backend="stage1", lp_probability_backend="stage1",
        tabicl_device=None if config.tabicl_device == "auto" else config.tabicl_device,
        fit_gp=True, gp_feature_mode="thresholds", target_policy_uncertainty="posterior_mean",
        lp_posterior_samples=config.lp_posterior_samples, lp_posterior_n_jobs=config.n_jobs,
        candidate_selection_mode="all_triggered", skip_calibration=True,
        skip_lp_intervention_diagnostics=True, make_plots=False)

def fit_naturalness(training: Mapping[str, pd.DataFrame], target_idle: pd.DataFrame,
                    baseline: pd.DataFrame, covariates: Sequence[str], seed: int,
                    config: ComparisonBaseConfig, output: Path) -> pd.DataFrame:
    """Fit the source c/d surfaces and GP, then solve the full target LP over GP draws.

    Only source on/idle and target idle rows enter. All non-target sources train the GP, with no
    calibration split, and every triggered choice on the configured epsilon/tau grid is intersected.
    Intermediate tables are written to ``output``; returns the per-cell bounds of
    :func:`posterior_bounds`.
    """
    # Deferred: experiments.benchmarks imports catenets and torch at module level.
    from experiments.benchmarks import atomic_write_parquet
    inference = naturalness_config(config, covariates, seed)
    grounder = build_configured_naturalness_grounder(inference)
    factory = grounder.stage1_backend_factory_

    def logged_site_fit(**kwargs: Any) -> Any:
        """Log each actual source model fit, forwarding unchanged arguments."""
        started = time.monotonic()
        LOGGER.info("Naturalness source=%s fit start rows=%s", kwargs["site_id"], len(kwargs["data"]))
        model = factory(**kwargs)
        LOGGER.info("Naturalness source=%s fit done seconds=%.1f", kwargs["site_id"], time.monotonic() - started)
        return model

    grounder.stage1_backend_factory_ = logged_site_fit
    grounder.fit(dict(training), treatment="iv", outcome="dv_binary", regime_col="F", fit_gp=False)
    atomic_write_parquet(grounder.observations_, output / "source_observations.parquet")
    LOGGER.info("Naturalness source GP fit start")
    grounder.fit_stage3_gp()
    LOGGER.info("Naturalness target probability fit start rows=%s", len(target_idle))
    observed = estimate_target_observed_binary_quantities(
        target_idle, treatment="iv", outcome="dv_binary", covariates=covariates,
        model_covariates=covariates, min_count=config.min_cell_count,
        probability_backend=config.stage1_backend,
        probability_model_kwargs=build_lp_probability_model_kwargs(inference, config.stage1_backend))
    candidates = exp16._policy_candidates(grounder, target_idle, inference, trigger_quantities=observed)
    choices = exp16.select_policy_choices(candidates, len(covariates), mode="all_triggered")
    atomic_write_parquet(candidates, output / "target_candidates.parquet")
    atomic_write_parquet(choices, output / "selected_choices.parquet")
    LOGGER.info("Naturalness full LP choices=%s posterior draws=%s", len(choices), config.lp_posterior_samples)
    direct = exp16.solve_choices_lp(target_idle, choices, inference, observed_quantities=observed)
    atomic_write_parquet(_complete_bounds(direct, baseline, covariates), output / "naturalness_direct.parquet")
    tolerances, samples, conditional = exp16.solve_choices_lp_posterior_samples(
        target_idle, choices, grounder, inference, random_seed=seed, observed_quantities=observed)
    atomic_write_parquet(tolerances, output / "tolerance_draws.parquet")
    atomic_write_parquet(samples, output / "lp_draws_raw.parquet")
    atomic_write_parquet(conditional, output / "posterior_conditional_on_optimal.parquet")
    primary, completed = posterior_bounds(samples, baseline, covariates, config.lp_posterior_samples)
    atomic_write_parquet(completed, output / "lp_draws_completed.parquet")
    return primary

def evaluation_strata(evaluation: Mapping[str, Any], covariates: Sequence[str]) -> pd.DataFrame:
    """Aggregate full target counts and reference effects after fitted inference."""
    frame = evaluation["frame"][list(covariates)].copy()
    frame["reference_effect"] = evaluation["effect"]
    frame["paired_effect"] = evaluation["potential"][1].astype(float) - evaluation["potential"][0]
    result = frame.groupby(list(covariates), observed=True, dropna=False).agg(
        n_evaluation=("reference_effect", "size"), reference_effect=("reference_effect", "mean"),
        paired_effect=("paired_effect", "mean")).reset_index()
    result["evaluation_weight"] = result.n_evaluation / result.n_evaluation.sum()
    return result

def score_bounds(bounds: pd.DataFrame, truth: pd.DataFrame, covariates: Sequence[str],
                 method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score widths, reference containment, and sign decisions over all target mass.

    ``truth`` is evaluation-only and joins predictions after fitting. Site ATE endpoints are
    prevalence-weighted sums of the cell endpoints. Returns the scored cells and a metrics record.
    """
    scored = bounds.merge(truth, on=list(covariates), how="left", validate="one_to_one")
    if scored.n_evaluation.isna().any() or len(scored) != len(truth):
        raise ValueError("Predictions must retain all evaluation cells")
    scored["method"] = method
    scored["width"] = scored.upper - scored.lower
    scored["contains_reference"] = (scored.lower <= scored.reference_effect + 1e-10) & (scored.upper >= scored.reference_effect - 1e-10)
    scored["contains_paired_reference"] = (scored.lower <= scored.paired_effect + 1e-10) & (scored.upper >= scored.paired_effect - 1e-10)
    positive, negative = scored.lower > 0, scored.upper < 0
    scored["sign_decision"] = positive | negative
    scored["wrong_sign_decision"] = (positive & (scored.reference_effect <= 0)) | (negative & (scored.reference_effect >= 0))
    weight = scored.evaluation_weight
    lower, upper, reference = (float(np.dot(weight, scored[c])) for c in ("lower", "upper", "reference_effect"))
    metrics = dict(method=method, n_evaluation=int(scored.n_evaluation.sum()), n_cells=len(scored),
        weighted_width=float(np.dot(weight, scored.width)),
        reference_containment_mass=float(np.dot(weight, scored.contains_reference)),
        paired_reference_containment_mass=float(np.dot(weight, scored.contains_paired_reference)),
        sign_decision_mass=float(np.dot(weight, scored.sign_decision)),
        wrong_sign_decision_mass=float(np.dot(weight, scored.wrong_sign_decision)),
        common_supported_mass=float(np.dot(weight, scored.common_supported)),
        fallback_mass=float(np.dot(weight, scored.used_manski_fallback)),
        ate_lower=lower, ate_upper=upper, ate_reference=reference,
        ate_contains_reference=lower - 1e-10 <= reference <= upper + 1e-10)
    if "fallback_draw_fraction" in scored:
        metrics["fallback_draw_mass"] = float(np.dot(weight, scored.fallback_draw_fraction))
        metrics["optimal_draw_mass"] = float(np.dot(weight, scored.optimal_draw_fraction))
    return scored, metrics

def rescore_with_manski_cap(leaf: Path, covariates: Sequence[str]
                            ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Apply the Manski cap to a condition leaf whose capped arms were scored uncapped.

    Nothing is refitted: each arm's ``<arm>_raw.parquet`` and the Manski rows of
    ``bounds_scored.parquet`` (shared baseline and truth) go through
    :func:`intersect_with_manski`, :func:`_complete_bounds` and :func:`score_bounds`, as in the
    runner. The uncapped predictions must first reproduce the stored rows, which confirms the
    leaf matches this scoring code before anything is replaced.

    Args:
        leaf: Condition directory holding ``bounds_scored.parquet``.
        covariates: Conditioning covariates of the condition.

    Returns:
        The leaf's scored bounds with every capped arm replaced, the new metrics keyed by arm,
        and one record per arm counting its cells, capped cells and disjoint fallbacks. Arms
        the leaf never fit are skipped.

    Raises:
        ValueError: If an arm is scored without raw predictions, was already
            capped, or does not reproduce its stored scores uncapped.
    """
    columns = list(covariates)
    scored = pd.read_parquet(leaf / "bounds_scored.parquet")
    manski = scored.loc[scored.method.eq("manski")].reset_index(drop=True)
    baseline = manski[columns + ["lower", "upper", "n_fit", "n_fit_arm0", "n_fit_arm1",
                                 "common_supported"]]
    truth = manski[columns + ["n_evaluation", "reference_effect", "paired_effect",
                              "evaluation_weight"]]
    kept = [scored.loc[~scored.method.isin(MANSKI_CAPPED_METHODS)]]
    metrics: dict[str, dict[str, Any]] = {}
    counts: list[dict[str, Any]] = []
    for method in MANSKI_CAPPED_METHODS:
        stored = scored.loc[scored.method.eq(method)]
        path = leaf / f"{method}_raw.parquet"
        if not path.exists():
            if not stored.empty:
                raise ValueError(f"{leaf} scores {method} but kept no raw predictions")
            continue
        raw = pd.read_parquet(path)
        if "uncapped_lower" in raw:
            raise ValueError(f"{path} is already capped at Manski")
        uncapped, _ = score_bounds(_complete_bounds(raw, baseline, columns), truth, columns, method)
        check = stored.merge(uncapped, on=columns, suffixes=("", "_rescored"), validate="one_to_one")
        if len(check) != len(stored) or not all(
                np.allclose(check[c], check[f"{c}_rescored"], atol=1e-12, rtol=0, equal_nan=True)
                for c in ("lower", "upper", "width")):
            raise ValueError(f"{leaf}: re-scoring {method} uncapped does not reproduce the stored bounds")
        capped = intersect_with_manski(raw, baseline, columns)
        rescored, metrics[method] = score_bounds(
            _complete_bounds(capped, baseline, columns), truth, columns, method)
        kept.append(rescored)
        counts.append(dict(method=method, cells=len(capped),
                           cells_capped=int(capped.manski_capped.sum()),
                           cells_disjoint=int(capped.status.eq("manski_disjoint").sum())))
    return pd.concat(kept, ignore_index=True), metrics, counts

def _write_json(payload: Any, path: Path) -> None:
    """Atomically publish a JSON payload at the supplied output path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    temporary.replace(path)

def _input_files(config: ComparisonBaseConfig) -> list[Path]:
    """Resolve exactly the first-stage input files used by the requested datasets."""
    files = []
    for dataset in config.datasets:
        root = dataset_root(config, dataset, None)
        for letter in ("X", "T", "Y"):
            matches = sorted((root / "twins").glob(f"twin_pairs_{letter}*.csv"))
            if not matches:
                raise FileNotFoundError(f"Missing Twins {letter} input in {root}")
            files.append(matches[0])
    return files

#: CB meta learners in the default roster, all four paper-faithful catenets fits.
DEFAULT_META_LEARNERS = ("cb_pi", "cb_ra", "cb_ipw", "cb_dr")

METHOD_LABELS = {
    "manski": "Manski",
    "naturalness": "Naturalness",
    "blearner_exp_beta": r"B-Learner $\Gamma=e^{\beta}$",
    "blearner_oracle": r"B-Learner $\Gamma$ oracle",
    "cb_pi": "CB-PI",
    "cb_ra": "CB-RA",
    "cb_ipw": "CB-IPW",
    "cb_dr": "CB-DR",
    "balke_pearl": "Balke-Pearl IV",
}


def blearner_bounds_at_gamma(target_idle: pd.DataFrame, strata: pd.DataFrame,
                             covariates: Sequence[str], gamma: float, seed: int,
                             n_jobs: int, blearner_repository: str) -> pd.DataFrame:
    """Fit the pinned B-Learner on target idle rows at one fixed ``Gamma``.

    Serves both the ``exp(beta)`` and the oracle-``Gamma`` arms. Only the shared conditioning
    covariates are predictors and only target idle rows are fitted, so the information set matches
    Manski's. Crossed endpoints are recorded before the pointwise rearrangement.

    Args:
        target_idle: Target-site observational rows, the common fitting subset.
        strata: Unique evaluation cells at which endpoints are predicted.
        covariates: Shared conditioning covariates, the only predictors.
        gamma: Marginal sensitivity-model parameter, at least 1.0.
        seed: Seed for the eight nuisance models.
        n_jobs: Worker count handed to the underlying forests.
        blearner_repository: Path to the pinned upstream B-Learner checkout.

    Returns:
        One row per evaluation cell carrying ``lower``/``upper`` (rearranged),
        ``unrearranged_lower``/``unrearranged_upper``, ``endpoints_crossed``,
        ``status`` and the ``gamma`` that produced them.

    Raises:
        ValueError: If ``gamma`` is below 1.0, raised by the upstream wiring.
    """
    # Deferred: experiments.benchmarks imports catenets and torch at module level.
    from experiments.benchmarks import BenchmarkFitConfig, make_blearner
    fit_config = BenchmarkFitConfig(
        seed=seed, n_jobs=n_jobs, worker_threads=1, n_layers_out=2, n_units_out=100,
        n_iter=1000, batch_size=128, validation_fraction=.2, n_iter_print=100,
        blearner_repository=blearner_repository)
    learner = make_blearner(fit_config, float(gamma), model_n_jobs=n_jobs)
    learner.fit(target_idle[list(covariates)].to_numpy(dtype=float),
                target_idle.iv.to_numpy(dtype=int),
                target_idle.dv_binary.to_numpy(dtype=float))
    lower, upper = learner.effect(strata[list(covariates)].to_numpy(dtype=float))
    lower, upper = np.asarray(lower).reshape(-1), np.asarray(upper).reshape(-1)
    result = strata[list(covariates)].copy()
    result["unrearranged_lower"], result["unrearranged_upper"] = lower, upper
    result["endpoints_crossed"] = lower > upper
    result["lower"], result["upper"] = np.minimum(lower, upper), np.maximum(lower, upper)
    result["status"] = np.where(result.endpoints_crossed, "rearranged", "optimal")
    result["gamma"] = float(gamma)
    return result


def oracle_required_gamma(evaluation: Mapping[str, Any], covariates: Sequence[str]) -> float:
    """Return the finite-population sensitivity parameter the assignment requires.

    The assignment probability is known by construction, so the smallest marginal-sensitivity-model
    ``Gamma`` covering it within the reporting cells is computed exactly: the largest symmetric
    odds ratio between a unit's latent-conditioned propensity and its cell's mean propensity. It
    sets the oracle-``Gamma`` B-Learner arm and shows whether ``exp(beta)`` covers the
    assignment; it never selects a condition.

    Args:
        evaluation: Target-site evaluation store from ``prepare_condition``, carrying the full idle
            frame and its latent-conditioned propensity.
        covariates: Conditioning covariates defining the reporting cells.

    Returns:
        The required ``Gamma``, exactly 1.0 when assignment is already ignorable within the cells.
    """
    frame = evaluation["frame"][list(covariates)].copy()
    frame["propensity"] = evaluation["propensity"]
    marginal = frame.groupby(list(covariates), observed=True).propensity.transform("mean").to_numpy()
    propensity = frame.propensity.to_numpy()
    ratio = (propensity / (1 - propensity)) / (marginal / (1 - marginal))
    return float(np.maximum(ratio, 1 / ratio).max())



def balke_pearl_bounds(target_idle: pd.DataFrame, strata: pd.DataFrame,
                       covariates: Sequence[str], min_count: int,
                       tau: float) -> pd.DataFrame:
    """Sharp Balke-Pearl bounds over every candidate that clears relevance.

    The hard-constraint reference for Naturalness: it reads only the target site's observational
    rows (no source trials, GP or transported tolerance). An instrument must vary within the
    reporting cell, so the bound is computed on ``E[Y(1) - Y(0) | Z_-j]`` and assigned to every
    full cell sharing that ``Z_-j``; this is valid only under Balke-Pearl's exogeneity assumption,
    the assumption under test. Candidates are those whose relevance statistic ``D_j(tau)`` from
    :mod:`causal_grounding.naturalness_statistics` clears ``tau`` on the target observational
    data, as in Naturalness.

    Args:
        target_idle: Target-site idle rows, the common learner subset.
        strata: All evaluation cells over ``covariates``.
        covariates: Conditioning covariates defining the reporting cells.
        min_count: Support floor applied to each candidate level within a cell.
        tau: Relevance threshold a candidate must clear to be used at all.

    Returns:
        A frame in the shared per-cell schema, with ``bp_candidates`` recording how many
        instruments contributed and ``bp_status`` carrying ``sharp``, ``infeasible`` (the
        instrumental inequality fired) or ``unsupported``.
    """
    names = list(covariates)
    values = {c: target_idle[c].to_numpy() for c in names}
    treat = target_idle["iv"].to_numpy().astype(int)
    outcome = target_idle["dv_binary"].to_numpy().astype(int)

    relevant = _relevant_candidates(target_idle, names, min_count, tau)
    per_stratum: dict[tuple[int, tuple], balke_pearl.BalkePearlInterval] = {}
    for j in relevant:
        rest = [c for c in names if c != names[j]]
        levels = sorted(np.unique(values[names[j]]))
        if len(levels) != 2:
            continue
        rest_codes = _cell_code(values, rest)
        for code in np.unique(rest_codes):
            block_mask = rest_codes == code
            counts = np.zeros((2, 2, 2))
            usable = True
            for zi, level in enumerate(levels):
                arm = block_mask & (values[names[j]] == level)
                if arm.sum() < min_count:
                    usable = False
                    break
                for y in (0, 1):
                    for x in (0, 1):
                        counts[zi, y, x] = int((arm & (outcome == y) & (treat == x)).sum())
            if not usable:
                continue
            table = balke_pearl.joint_table_from_counts(counts)
            if not np.isfinite(table).all():
                continue
            key = (j, tuple(int(v) for v in code) if np.ndim(code) else (int(code),))
            per_stratum[key] = balke_pearl.balke_pearl_interval(table)

    records = []
    for z in strata[names].itertuples(index=False, name=None):
        mask = np.logical_and.reduce([values[c] == v for c, v in zip(names, z)])
        cell = target_idle.loc[mask]
        n0 = int((cell.iv == 0).sum())
        n1 = int((cell.iv == 1).sum())
        intervals = []
        for j in relevant:
            rest = [c for c in names if c != names[j]]
            code = tuple(int(v) for c, v in zip(names, z) if c in rest)
            found = per_stratum.get((j, _pack_code(code)))
            if found is not None:
                intervals.append(found)
        combined = balke_pearl.intersect_intervals(intervals)
        # A non-"optimal" ``status`` sends the cell to the Manski fallback; ``bp_status`` keeps the
        # Balke-Pearl reason. Support columns are not emitted: _complete_bounds takes them from the
        # shared baseline, and duplicates would be suffixed away by the merge.
        records.append(dict(zip(names, z))
                       | dict(lower=combined.lower, upper=combined.upper,
                              status="optimal" if combined.feasible else combined.status,
                              bp_candidates=len(intervals), bp_status=combined.status,
                              bp_n_arm0=n0, bp_n_arm1=n1))
    return pd.DataFrame(records)


def _cell_code(values: Mapping[str, np.ndarray], columns: Sequence[str]) -> np.ndarray:
    """Return an integer code per row for a tuple of binary covariate columns."""
    if not columns:
        return np.zeros(len(next(iter(values.values()))), dtype=int)
    code = np.zeros(len(values[columns[0]]), dtype=int)
    for power, column in enumerate(columns):
        code += values[column].astype(int) << power
    return code


def _pack_code(levels: Sequence[int]) -> tuple:
    """Pack covariate levels into the integer key used by the stratum cache."""
    code = 0
    for power, value in enumerate(levels):
        code += int(value) << power
    return (code,)


def _relevant_candidates(target_idle: pd.DataFrame, names: Sequence[str],
                         min_count: int, tau: float) -> list[int]:
    """Return candidate indices clearing the Naturalness relevance trigger.

    Reuses :func:`causal_grounding.naturalness_statistics.compute_d_strength`,
    the same ``D_j(tau)`` statistic the Naturalness policy applies, fed with
    empirical cell frequencies rather than a fitted model.
    """
    values = {c: target_idle[c].to_numpy() for c in names}
    treat = target_idle["iv"].to_numpy().astype(int)
    kept = []
    for j, name in enumerate(names):
        rest = [c for c in names if c != name]
        z_minus = [tuple(int(b) for b in v)
                   for v in {tuple(row) for row in np.column_stack(
                       [values[c] for c in rest])}] if rest else [()]

        def probability(x: int, z: Sequence[int], _j: int = j) -> float:
            mask = np.ones(len(treat), dtype=bool)
            for c, v in zip(names, z):
                mask &= values[c] == v
            if mask.sum() < min_count:
                return float("nan")
            return float((treat[mask] == x).mean())

        strength = stats.compute_d_strength(names, j, z_minus, probability, target_idle)
        if np.isfinite(strength) and strength >= tau:
            kept.append(j)
    return kept

def meta_learner_bounds(condition: Mapping[str, Any], target: str, covariates: Sequence[str],
                        *, seed: int, n_jobs: int, blearner_repository: str,
                        meta_learners: Sequence[str] = DEFAULT_META_LEARNERS,
                        source_regime: str = "on") -> dict[str, pd.DataFrame]:
    """Fit the paper-faithful CB meta learners on this condition's environments.

    The fit is delegated to
    :func:`~experiments.exp18_cb_meta_learner_comparison.fit_prepared_study_bounds`, the WB/CB
    implementation Exp18 and Exp25 use, which keeps target randomized rows from every learner.
    With ``source_regime="on"`` the source environment pools the source randomized rows, the
    source outcomes Naturalness reads; ``"idle"`` pools source observational rows, as in Exp25.

    Args:
        condition: Output of ``prepare_condition``, supplying the ``on``/``idle`` fitting frames.
        target: Target site label; every other site is pooled as the source.
        covariates: Shared conditioning covariates, used as reporting strata.
        seed: Seed for the torch fits.
        n_jobs: Worker count for the fit pool.
        blearner_repository: Pinned checkout path, required by the config even
            though no B-Learner is fitted here.
        meta_learners: Which CB learners to fit.
        source_regime: ``"on"`` to pool source randomized rows, ``"idle"`` to
            pool source observational rows.

    Returns:
        A mapping from meta-learner name to a frame of the conditioning covariates plus
        ``lower``, ``upper`` and ``status``, ready for the shared support and Manski-fallback
        rule. Learners whose interval collapsed to the full binary range are marked
        ``full_range_fallback``.

    Raises:
        ValueError: If the target site has no rows, raised upstream.
    """
    # Deferred: exp18 imports experiments.benchmarks, which imports catenets and torch.
    from experiments import exp18_cb_meta_learner_comparison as exp18
    sites = sorted(set(condition["idle"]["site"].astype(str)) | set(condition["on"]["site"].astype(str)))
    sources = tuple(s for s in sites if s != str(target))
    config = exp18.Exp18Config(
        data_source="exp16", comparison_covariates=tuple(covariates), target_site=str(target),
        source_site_policy="all_non_target", source_regime=source_regime,
        pair_scope="target_anchored", meta_learners=tuple(meta_learners), seed=seed,
        n_jobs=n_jobs, worker_threads=1, blearner_repository=blearner_repository,
        make_plots=False, verbose=False)
    prepared = exp18.PreparedStudyData(idle=condition["idle"].copy(), rct=condition["on"].copy())
    fitted = exp18.fit_prepared_study_bounds(
        config, study=str(target), study_data=prepared, source_sites=sources, blearner_gammas=())
    final = fitted["cb_final_bounds"]
    result = {}
    for method in meta_learners:
        columns = list(covariates)
        frame = final[columns].copy()
        frame["lower"] = pd.to_numeric(final[f"{method}_lower"], errors="coerce")
        frame["upper"] = pd.to_numeric(final[f"{method}_upper"], errors="coerce")
        fallback = final.get(f"{method}_used_full_range_fallback", pd.Series(False, index=final.index))
        inverted = final.get(f"{method}_interval_inverted", pd.Series(False, index=final.index))
        frame["status"] = np.where(fallback.to_numpy(dtype=bool), "full_range_fallback",
                                   np.where(inverted.to_numpy(dtype=bool), "interval_inverted", "optimal"))
        result[method] = frame.reset_index(drop=True)
    return result


#: Per-cell arms in reporting order; Manski is first because every other arm falls back to it.
PER_CELL_METHODS = ("manski", "naturalness", "blearner_exp_beta", "blearner_oracle",
                    "balke_pearl", "cb_pi", "cb_ra", "cb_ipw", "cb_dr")

#: CB learners in the Exp28 roster (CB-IPW and CB-DR). :data:`PER_CELL_METHODS` and
#: ``DEFAULT_META_LEARNERS`` keep all four, since the config validators check subsets of them.
NATURALNESS_META_LEARNERS = ("cb_ipw", "cb_dr")
NATURALNESS_PER_CELL_METHODS = tuple(
    m for m in PER_CELL_METHODS
    if m in NATURALNESS_META_LEARNERS or not m.startswith("cb_"))


@dataclass(frozen=True)
class ComparisonConfig(ComparisonBaseConfig):
    """Settings for one method-comparison experiment: the shared fields plus the arm roster.

    Attributes:
        experiment_name: Identifier recorded in the run manifest and used as the results namespace.
        methods: Per-cell arms to run; must include ``manski``.
        meta_learners: Which CB learners to fit, a subset of ``DEFAULT_META_LEARNERS``.
        cb_source_regime: ``"on"`` pools source randomized rows into the CB source environment
            (the source data Naturalness reads); ``"idle"`` pools source observational rows.
        withheld_covariates: Pool covariates withheld from every method, recorded in the manifest.
    """

    experiment_name: str = "naturalness_method_comparison"
    methods: tuple[str, ...] = PER_CELL_METHODS
    meta_learners: tuple[str, ...] = DEFAULT_META_LEARNERS
    cb_source_regime: str = "on"
    withheld_covariates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the roster on top of the shared comparison validation."""
        super().__post_init__()
        unknown = set(self.methods) - set(PER_CELL_METHODS)
        if unknown:
            raise ValueError(f"Unknown per-cell methods: {sorted(unknown)}")
        if "manski" not in self.methods:
            raise ValueError("Manski is the shared fallback and must always be reported")
        if set(self.meta_learners) - set(DEFAULT_META_LEARNERS):
            raise ValueError(f"meta_learners must be a subset of {DEFAULT_META_LEARNERS}")
        if self.cb_source_regime not in ("on", "idle"):
            raise ValueError("cb_source_regime must be 'on' or 'idle'")


def _requested_meta_learners(config: ComparisonConfig) -> tuple[str, ...]:
    """Return the CB learners that are both configured and listed in ``methods``."""
    return tuple(m for m in config.meta_learners if m in config.methods)


def _per_cell_bounds(condition: Mapping[str, Any], training: Mapping[str, pd.DataFrame],
                     target_idle: pd.DataFrame, baseline: pd.DataFrame, strata: pd.DataFrame,
                     evaluation: Mapping[str, Any], covariates: Sequence[str], target: str,
                     beta: float, seed: int, config: ComparisonConfig,
                     leaf: Path) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Fit every requested per-cell arm, retaining all target mass on failure.

    A method that raises is recorded and scored as a Manski fallback over the
    whole target population rather than being dropped, so a fragile method
    cannot improve its apparent width by failing on hard cells.

    Args:
        condition: Output of ``prepare_condition`` for this condition.
        training: Source-site frames keyed by site, from the train/target split.
        target_idle: Target observational rows shared by every fitted arm.
        baseline: Manski endpoints and the shared support mask.
        strata: Unique evaluation cells.
        evaluation: Target evaluation store, used only for the oracle ``Gamma``.
        covariates: Shared conditioning covariates.
        target: Target site label.
        beta: Assignment strength for this condition.
        seed: Seed of this condition.
        config: Experiment settings.
        leaf: Output directory for this condition's raw artifacts.

    Returns:
        A tuple of per-method bound frames, recorded errors, per-method fit
        seconds, and the sensitivity-parameter diagnostic.
    """
    # Deferred: experiments.benchmarks imports catenets and torch at module level.
    from experiments.benchmarks import atomic_write_parquet
    required = oracle_required_gamma(evaluation, covariates)
    gamma_exp_beta = float(np.exp(beta))
    diagnostic = dict(
        gamma_exp_beta=gamma_exp_beta,
        oracle_finite_population_gamma_required=required,
        exp_beta_covers_declared_assignment=bool(gamma_exp_beta >= required - 1e-10))
    bounds = {"manski": baseline.assign(used_manski_fallback=False, status="manski")}
    errors: list[dict[str, Any]] = []
    timings = {"manski": 0.}
    cb_requested = _requested_meta_learners(config)

    def record_failure(name: str, error: Exception) -> None:
        """Log a failed arm and give it the full-population Manski fallback."""
        LOGGER.exception("%s fit failed; retaining all target mass", name)
        errors.append(dict(method=name, error_type=type(error).__name__, error=str(error)))
        bounds[name] = _complete_bounds(pd.DataFrame(), baseline, covariates, failure="fit_failed")

    for method in config.methods:
        if method == "manski" or method in cb_requested:
            continue
        started = time.monotonic()
        LOGGER.info("%s fit start target_rows=%s source_sites=%s", method, len(target_idle), len(training))
        try:
            if method == "naturalness":
                bounds[method] = fit_naturalness(
                    training, target_idle, baseline, covariates, seed, config, leaf)
            elif method == "balke_pearl":
                # The weakest tau on purpose: Naturalness intersects every triggered choice, so
                # Balke-Pearl admits every candidate that clears relevance at all.
                raw = balke_pearl_bounds(
                    target_idle, strata, covariates, config.min_cell_count,
                    tau=min(config.taus))
                atomic_write_parquet(raw, leaf / "balke_pearl_raw.parquet")
                diagnostic["balke_pearl_infeasible_cells"] = int(
                    (raw["bp_status"] == "infeasible").sum())
                diagnostic["balke_pearl_mean_candidates"] = float(raw["bp_candidates"].mean())
                bounds[method] = _complete_bounds(raw, baseline, covariates)
            else:
                gamma = gamma_exp_beta if method == "blearner_exp_beta" else max(required, 1.0)
                raw = blearner_bounds_at_gamma(
                    target_idle, strata, covariates, gamma, seed, config.n_jobs,
                    config.blearner_repository)
                if method in MANSKI_CAPPED_METHODS:
                    raw = intersect_with_manski(raw, baseline, covariates)
                atomic_write_parquet(raw, leaf / f"{method}_raw.parquet")
                bounds[method] = _complete_bounds(raw, baseline, covariates)
                diagnostic[f"{method}_gamma"] = float(gamma)
        except Exception as error:  # noqa: BLE001 - every arm failure must be recorded, not raised
            record_failure(method, error)
        timings[method] = time.monotonic() - started
        LOGGER.info("%s fit done seconds=%.1f", method, timings[method])

    if cb_requested:
        started = time.monotonic()
        LOGGER.info("CB meta learners fit start learners=%s source_regime=%s",
                    list(cb_requested), config.cb_source_regime)
        try:
            fitted = meta_learner_bounds(
                condition, target, covariates, seed=seed, n_jobs=config.n_jobs,
                blearner_repository=config.blearner_repository, meta_learners=cb_requested,
                source_regime=config.cb_source_regime)
            for name, raw in fitted.items():
                atomic_write_parquet(raw, leaf / f"{name}_raw.parquet")
                bounds[name] = _complete_bounds(raw, baseline, covariates)
        except Exception as error:  # noqa: BLE001 - one failure must not lose the other arms
            for name in cb_requested:
                record_failure(name, error)
        elapsed = time.monotonic() - started
        for name in cb_requested:
            timings[name] = elapsed / len(cb_requested)
        LOGGER.info("CB meta learners fit done seconds=%.1f", elapsed)
    return bounds, errors, timings, diagnostic


def _fingerprint(config: ComparisonConfig) -> tuple[str, dict[str, Any]]:
    """Hash the settings, code and inputs that a resumed run must match.

    Returns:
        The fingerprint and the payload it was computed from.
    """
    # Deferred: experiments.benchmarks imports catenets and torch at module level.
    from experiments.benchmarks import BLEARNER_COMMIT
    settings = asdict(config)
    for key in ("resume", "output_dir", "cluster_manifest_path"):
        settings.pop(key, None)
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), repo / "experiments/naturalness_comparison_data.py",
               repo / "experiments/benchmarks.py", repo / "experiments/utils.py",
               repo / "experiments/exp16_osrct_calibrated_policy.py",
               repo / "experiments/exp18_cb_meta_learner_comparison.py"]
    sources.extend(sorted((repo / "causal_grounding").rglob("*.py")))
    payload = dict(config=settings, blearner_commit=BLEARNER_COMMIT,
                   inputs={str(p): _sha256(p) for p in _input_files(config)},
                   sources={str(p.relative_to(repo)): _sha256(p) for p in sources if p.exists()})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload



def run_comparison(config: ComparisonConfig, argv: Sequence[str] = ()) -> pd.DataFrame:
    """Run one full method-comparison experiment over its grid.

    Resuming requires an identical configuration, code and input fingerprint. The reference
    effects are joined only after every fitted arm of a condition has finished.

    Args:
        config: Experiment settings.
        argv: Command line recorded in the run manifest.

    Returns:
        One row per (condition, method) with widths, containment and the
        sensitivity diagnostics.

    Raises:
        ValueError: If an existing output directory does not match, or a
            requested target site is absent from a dataset.
    """
    # Deferred: experiments.benchmarks imports catenets and torch at module level.
    from experiments.benchmarks import atomic_write_parquet
    output = Path(config.output_dir)
    fingerprint, payload = _fingerprint(config)
    identity = output / "identity.json"
    if identity.exists():
        if not config.resume or json.loads(identity.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Existing output requires --resume with identical inputs, code, and settings")
    elif output.exists() and any(output.iterdir()) and {p.name for p in output.iterdir()} != {"run.log"}:
        raise ValueError("Refusing to overwrite an existing output directory")
    output.mkdir(parents=True, exist_ok=True)
    if not identity.exists():
        _write_json(dict(fingerprint=fingerprint, **payload), identity)
        write_run_manifest(
            output, experiment_name=config.experiment_name, script_path=__file__, config=config,
            argv=argv, cluster_manifest_path=config.cluster_manifest_path,
            extra=dict(
                fingerprint=fingerprint, per_cell_methods=config.methods,
                calibration="none; raw plug-in identification endpoints",
                conformal="none",
                withheld_covariates=config.withheld_covariates,
                sensitivity="B-Learner at Gamma=exp(beta) and at the oracle-required Gamma",
                information_sets=(
                    "Naturalness: all non-target sites' on+idle rows plus target idle. "
                    "B-Learner and Manski: target idle only. "
                    "CB meta learners: one pooled source environment plus target idle."),
                naturalness_extra_step=(
                    "Naturalness endpoints average over GP tolerance draws with Manski-replaced "
                    "failures; no competitor receives an equivalent averaging step."),
                scientific_scope=(
                    "Reference containment in one realization is descriptive, not repeated-sampling "
                    "coverage. Site averages are weighted sums of per-cell endpoints, which outer-bound "
                    "the joint identified set rather than equalling it.")))

    all_metrics: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    condition_count = 0
    for dataset in config.datasets:
        draws = (None,) if dataset == "twins_reduced" else config.dgp_seeds
        for draw in draws:
            root = dataset_root(config, dataset, draw)
            prepared = data.prepare_dataset(
                data.BUILTIN_SPECS[dataset], root,
                n_sites=6, min_site_rows=150)
            for seed in config.seeds:
                for beta in config.betas:
                    condition = prepare_condition(prepared, dataset, beta, seed, config)
                    targets = (sorted(condition["evaluation"], key=int)
                               if config.target_sites == ("all",) else config.target_sites)
                    for target in targets:
                        condition_count += 1
                        metadata = dict(experiment=config.experiment_name, dataset=dataset,
                                        dgp_seed=draw, seed=seed, beta=beta, target_site=target,
                                        stage1_backend=config.stage1_backend)
                        leaf = (output / dataset / f"dgp{draw}" / f"seed{seed}"
                                / f"beta{beta:g}" / f"target{target}")
                        complete = leaf / "completion.json"
                        if config.resume and complete.exists():
                            prior = json.loads(complete.read_text())
                            all_metrics.extend(prior["metrics"])
                            all_errors.extend(prior["errors"])
                            LOGGER.info("Resume completed condition %s", metadata)
                            continue
                        if target not in condition["evaluation"]:
                            raise ValueError(f"Requested target {target} absent for {dataset}")
                        leaf.mkdir(parents=True, exist_ok=True)
                        LOGGER.info("Condition %s start %s", condition_count, metadata)
                        training, target_idle = create_train_target_split(
                            condition["idle"], condition["on"], target)
                        evaluation = condition["evaluation"][target]
                        covariates = condition["covariates"]
                        strata = (evaluation["frame"][list(covariates)].drop_duplicates()
                                  .sort_values(list(covariates)))
                        baseline = manski_bounds(target_idle, strata, covariates, config.min_cell_count)
                        atomic_write_parquet(condition["row_audit"], leaf / "row_audit.parquet")
                        _write_json(prepared["provenance"], leaf / "dataset_provenance.json")

                        bounds, errors, timings, diagnostic = _per_cell_bounds(
                            condition, training, target_idle, baseline, strata, evaluation,
                            covariates, target, beta, seed, config, leaf)
                        for error in errors:
                            error.update(metadata)

                        # Reference effects enter only here, after every fitted arm has finished.
                        truth = evaluation_strata(evaluation, covariates)
                        _write_json(diagnostic, leaf / "sensitivity_diagnostic.json")
                        metrics, scored_frames = [], []
                        for method, frame in bounds.items():
                            scored, metric = score_bounds(frame, truth, covariates, method)
                            metric.update(metadata, reference_kind=evaluation["reference_kind"],
                                          fit_seconds=timings.get(method, float("nan")), **diagnostic)
                            metrics.append(metric)
                            scored_frames.append(scored)
                        atomic_write_parquet(pd.concat(scored_frames, ignore_index=True),
                                             leaf / "bounds_scored.parquet")
                        _write_json(
                            dict(metrics=metrics, errors=errors,
                                 status="fit_failures" if errors else "complete"), complete)
                        all_metrics.extend(metrics)
                        all_errors.extend(errors)
                        LOGGER.info("Condition %s done fit_failures=%s", condition_count, len(errors))
                        atomic_write_parquet(pd.DataFrame(all_metrics), output / "metrics.parquet")
                        _write_json(all_errors, output / "fit_failures.json")

    result = pd.DataFrame(all_metrics)
    atomic_write_parquet(result, output / "metrics.parquet")
    _write_json(all_errors, output / "fit_failures.json")
    _write_json(dict(status="fit_failures" if all_errors else "complete",
                            n_conditions=condition_count, n_method_results=len(result),
                            n_fit_failures=len(all_errors), fingerprint=fingerprint),
                       output / "completion.json")
    return result


def build_parser(defaults: ComparisonConfig) -> argparse.ArgumentParser:
    """Build the shared command line for a comparison experiment.

    Args:
        defaults: The entry point's settings, which become the CLI defaults.

    Returns:
        A parser whose parsed namespace maps onto :class:`ComparisonConfig`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "output_dir", "blearner_repository",
                 "cluster_manifest_path", "experiment_name", "twins_bias_covariate",
                 "cb_source_regime"):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(defaults, name))
    for name, dtype in (("datasets", str), ("target_sites", str), ("methods", str),
                        ("meta_learners", str), ("twins_covariates", str),
                        ("withheld_covariates", str), ("dgp_seeds", int), ("seeds", int),
                        ("betas", float), ("epsilons", float), ("taus", float)):
        parser.add_argument("--" + name.replace("_", "-"), nargs="+", type=dtype,
                            default=getattr(defaults, name))
    for name in ("min_cell_count", "max_fit_rows", "lp_posterior_samples", "n_jobs"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--stage1-backend", choices=("tabicl_v2", "hierarchical_dirichlet", "xgboost"),
                        default=defaults.stage1_backend)
    parser.add_argument("--tabicl-device", default=defaults.tabicl_device)
    parser.add_argument("--resume", action="store_true")
    return parser


TUPLE_FIELDS = ("datasets", "target_sites", "methods", "meta_learners",
                "twins_covariates", "withheld_covariates",
                "dgp_seeds", "seeds", "betas", "epsilons", "taus")


def main_from_defaults(defaults: ComparisonConfig, argv: Sequence[str] | None = None) -> int:
    """Parse arguments against an entry point's defaults and run the experiment.

    Args:
        defaults: The experiment's default settings.
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``1`` when any arm failed to fit, otherwise ``0``, so a partial run is visible to a
        job scheduler.
    """
    args = vars(build_parser(defaults).parse_args(argv))
    for name in TUPLE_FIELDS:
        args[name] = tuple(args[name])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    run_comparison(ComparisonConfig(**args), tuple(sys.argv[1:] if argv is None else argv))
    completion = json.loads((Path(args["output_dir"]) / "completion.json").read_text())
    LOGGER.info("Finished method_results=%s fit_failures=%s",
                completion["n_method_results"], completion["n_fit_failures"])
    return 1 if completion["n_fit_failures"] else 0
