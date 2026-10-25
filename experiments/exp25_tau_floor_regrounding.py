"""Exp25 sensitivity: re-ground Naturalness without its smallest tau values.

Discovery is kept as Exp25 ran it: its stored Stage-2 source observations are reused and only the grounding is
redone, on the tau grid at or above a floor: the Stage-3 GP for ``c`` and ``d``, the target choices and their
tolerances, and the LP posterior, with Exp25's seeds. Floor 0.01 keeps Exp25's grid and must reproduce the
published interval, which checks the replay. The Exp25 stores are only read.

Passes (``--mode``):
    ``tables`` (GPU): refit and pickle each target's probability tables (not stored by Exp25) and idle rows.
    ``reground`` (CPU): refit the GP, recompute the target choices and rerun the LP posterior for every floor.
    ``summarize``: pair every floor's intervals with the published Exp25 intervals, stratum by stratum.
    ``bounds`` (CPU): recompute Manski and Balke-Pearl with Balke-Pearl's relevance threshold at the floor.
    ``assemble``: copy a merged Exp25 leaf with one floor's Naturalness rows and, optionally, bounds.
    ``ablate``: grounding ablation; solve the target program under fixed, exact or prior-only tolerances.
    ``summarize-arms`` / ``plot-arms``: score the ablation and mechanism arms; write their tables and figures.

``reground`` options (the defaults reproduce Exp25; ``--tables-dir`` reads the tables of an earlier pass):
    ``--gp-training-target envelope``: fit the GP to the largest triggering-site tolerance per grid point.
    ``--gp-target-scale tilde``: fit the GP on ``logit(2c)`` rather than ``logit(c)``.
    ``--target-pairs admissible``: build the target program from the admissible (Pareto) triggered pairs only.
    ``--effect-shift DELTA`` (also ``tables``): effect-shift arm; every target stratum CATE moves by DELTA.
    ``--direct-effect GAMMA`` (also ``tables``): invalid-instrument arm; the selection covariate affects Y directly.
    ``--source-mismatch beta|covariate``: family-mismatch arm; ground targets on another condition's sources.

``paper/reproduce_experiment1_manylabs.sh`` and ``paper/reproduce_ablations.sh`` run these passes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.osrct_target_pipeline as target_pipeline
from causal_grounding.lp_solver_learned_naturalness import (
    estimate_target_observed_binary_quantities,
)
from causal_grounding.naturalness_grounder import (
    GP_TARGET_SCALE_NOTE,
    GP_TRAINING_TARGET_PER_SITE,
    SUPPORTED_GP_TARGET_SCALES,
    SUPPORTED_GP_TRAINING_TARGETS,
    validate_gp_target_scale,
    validate_gp_training_target,
)
from causal_grounding.paper_grids import MIN_RELEVANCE_TAU, PAPER_BETAS, floor_taus
from causal_grounding.train_target_split import create_train_target_split
from experiments.benchmarks import atomic_write_parquet, format_beta
from experiments.osrct_target_pipeline import Exp16Config
from experiments.exp25_osrct_mc_baseline_comparison import (
    Exp25Config,
    _exp16_config_for_replication,
)
from experiments.exp32_manylabs_noise_candidates import (
    _parquet_safe,
    exp25_target_index,
    load_exp25_store_config,
)
from experiments.osrct_data_utils import load_prepared_data
from experiments.utils import (
    build_configured_naturalness_grounder,
    build_lp_probability_model_kwargs,
    json_default,
    render_latex_table,
    resolve_lp_probability_backend,
    resolve_target_trigger_probability_backend,
    uses_hierarchical_dirichlet_draws,
)


#: Tau floors: 0.01 keeps Exp25's grid (the replay check); 0.05 drops tau = 0.01; 0.075 also drops 0.05.
DEFAULT_FLOORS = (0.01, 0.05, MIN_RELEVANCE_TAU)
#: Stage-2 columns the Stage-3 GP and the tolerance lookup read.
STAGE2_COLUMNS = (
    "site_id", "covariate", "covariate_index", "epsilon", "tau", "i_deviation",
    "d_strength", "trigger_fired", "c_hat", "d_hat", "n_z_minus_evaluated",
)
TARGET_IDLE_COLUMNS = ("site", "iv", "dv_binary", "F")
#: Target (epsilon, tau) pairs entering the program: every triggered pair (Exp25) or only the
#: admissible ones (:func:`admissible_choices`).
TARGET_PAIRS = ("all", "admissible")
#: Base seed of the effect-shift arm's outcome flips, combined with study and shift.
EFFECT_SHIFT_SEED = 20261007
#: Where the family-mismatch arm takes each target's source evidence from.
SOURCE_MISMATCHES = ("none", "beta", "covariate")
#: ``beta``: the paper beta farthest from the target's.
MISMATCHED_BETA = {0.5: 2.0, 1.0: 2.0, 1.5: 0.5, 2.0: 0.5}
#: ``covariate``: another selection covariate, in a cycle.
MISMATCHED_PATTERN = {"age": "polideo", "polideo": "gender", "gender": "age"}
#: Level of the transport baseline's random-effects prediction interval.
TRANSPORT_LEVEL = 0.90


@dataclass(frozen=True)
class Condition:
    """One Exp25 cell: store, replication, study, selection pattern and beta."""

    store: str
    replication: int
    study: str
    pattern: str
    beta: float

    @property
    def label(self) -> str:
        """Return a path-safe identifier for the condition."""
        return (
            f"replication={self.replication:03d}/{self.study}/{self.pattern}/"
            f"beta{format_beta(self.beta)}"
        )


def enumerate_conditions(
    stores: Sequence[str | Path],
    *,
    betas: Sequence[float] = PAPER_BETAS,
    replications: Sequence[int] | None = None,
) -> list[Condition]:
    """List every stored Exp25 condition with one of the requested betas.

    Args:
        stores: Exp25 backend leaves holding ``replications/`` and
            ``run_manifest.yml``.
        betas: Selection strengths to keep.
        replications: Replication indices to keep; ``None`` keeps all.

    Returns:
        Conditions in a stable order, so shards can split the list.

    Raises:
        ValueError: If two stores hold the same beta.
    """
    conditions: list[Condition] = []
    seen_betas: dict[float, str] = {}
    for store in stores:
        exp25 = load_exp25_store_config(Path(store))
        store_betas = [float(b) for b in exp25.betas if any(np.isclose(b, x) for x in betas)]
        for beta in store_betas:
            if beta in seen_betas:
                raise ValueError(f"beta={beta} is in both {seen_betas[beta]} and {store}")
            seen_betas[beta] = str(store)
        reps = range(int(exp25.n_replications)) if replications is None else replications
        for replication in reps:
            for study in exp25.studies:
                for pattern in exp25.bias_covariates:
                    for beta in store_betas:
                        conditions.append(Condition(str(store), int(replication), str(study), str(pattern), beta))
    return sorted(conditions, key=lambda c: (c.replication, c.study, c.pattern, c.beta))


def condition_config(condition: Condition) -> tuple[Exp16Config, int, Exp25Config]:
    """Rebuild the exact Exp16 configuration Exp25 used for one condition.

    Returns:
        The Exp16 configuration, the Exp16 target index that seeded source roles
        and GP draws in Exp25, and the store's Exp25 configuration.
    """
    exp25 = load_exp25_store_config(Path(condition.store))
    seed = int(exp25.base_seed + condition.replication)
    data_root = (
        Path(condition.store) / "replications" / f"replication={condition.replication:03d}" / "osrct"
    )
    full = _exp16_config_for_replication(
        exp25, seed=seed, data_root=data_root, output_dir=Path(tempfile.gettempdir())
    )
    config = replace(
        full,
        study=condition.study, studies=(condition.study,),
        pattern=condition.pattern, patterns=(condition.pattern,),
        beta=float(condition.beta), betas=(float(condition.beta),),
        verbose=False,
    )
    target_index = exp25_target_index(
        exp25, study=condition.study, pattern=condition.pattern, beta=condition.beta
    )
    return config, target_index, exp25


def bundle_leaf(condition: Condition, config: Exp16Config) -> Path:
    """Return the persisted Exp25 Naturalness leaf of a condition."""
    return Path(condition.store).joinpath(
        "replications", f"replication={condition.replication:03d}", "naturalness",
        "all_triggered", str(config.target_sites[0]), condition.pattern,
        format_beta(condition.beta),
    )


def read_stage2_observations(leaf: Path, study: str) -> pd.DataFrame:
    """Read one study's stored final-grounder Stage-2 observations."""
    frame = pd.read_parquet(leaf / "final_source_naturalness_observations.parquet")
    frame = frame.loc[frame["study"].astype(str).eq(study), list(STAGE2_COLUMNS)].copy()
    frame["trigger_fired"] = frame["trigger_fired"].astype(bool)
    return frame.reset_index(drop=True)


def _flip_target_outcomes(
    trial: pd.DataFrame, rate_change: Any, *, covariates: Sequence[str], seed: Sequence[int],
    equal_arms: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flip random trial outcomes so that each stratum-arm outcome rate moves as asked.

    ``rate_change(record, arm)`` is the signed rate change of one arm in stratum
    ``record``: ``round(n * |change|)`` random ``Y = 0`` (rise) or ``Y = 1``
    (fall) are flipped, or all eligible ones if fewer. With ``equal_arms`` both
    arms move by one magnitude, capped by what each can reach, so a common change
    keeps the CATE up to rounding. Strata and arms (treated first) are visited in
    a fixed order, so a seed fixes the draw.

    Returns:
        ``flips`` (``id`` and new ``dv_binary``) and a per-stratum ``log`` of arm
        sizes, flips, realized ``rate_change_arm1``/``rate_change_arm0`` and ``cate_shift``.
    """
    rng = np.random.default_rng(list(seed))
    flips: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for key, stratum in trial.sort_values("id").groupby(list(covariates), sort=True):
        record = dict(zip(covariates, key if isinstance(key, tuple) else (key,)))
        changes = {arm: float(rate_change(record, arm)) for arm in (1, 0)}
        if equal_arms:
            reachable = [abs(change) for change in changes.values()]
            for arm in (1, 0):
                members = stratum.loc[stratum["iv"].astype(int).eq(arm)]
                source = 0 if changes[arm] > 0 else 1
                if len(members):
                    reachable.append(float(members["dv_binary"].astype(int).eq(source).mean()))
            magnitude = min(reachable)
            changes = {arm: np.sign(change) * magnitude for arm, change in changes.items()}
        for arm in (1, 0):
            change = changes[arm]
            members = stratum.loc[stratum["iv"].astype(int).eq(arm)]
            source = 0 if change > 0 else 1
            eligible = np.sort(members.loc[members["dv_binary"].astype(int).eq(source), "id"].to_numpy())
            wanted = int(round(len(members) * abs(change)))
            chosen = rng.choice(eligible, size=min(wanted, len(eligible)), replace=False) if wanted else []
            flips.extend({"id": unit, "dv_binary": 1 - source} for unit in chosen)
            realized = (len(chosen) / len(members)) if len(members) else 0.0
            record.update({f"n_arm{arm}": len(members), f"wanted_arm{arm}": wanted, f"flipped_arm{arm}": len(chosen),
                           f"rate_change_arm{arm}": realized if change > 0 else -realized})
        record["cate_shift"] = record["rate_change_arm1"] - record["rate_change_arm0"]
        log.append(record)
    return pd.DataFrame(flips, columns=["id", "dv_binary"]), pd.DataFrame(log)


def _check_trial_ids(trial: pd.DataFrame) -> None:
    """Raise unless every trial participant has its own ``id``."""
    if trial["id"].duplicated().any():
        raise ValueError("Target trial ids must be unique to carry flips to the samples")


def effect_shift_flips(
    trial: pd.DataFrame, delta: float, *, covariates: Sequence[str], study: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the target-trial outcomes to flip so that every stratum CATE moves by ``delta``.

    In each stratum ``round(n * |delta| / 2)`` random participants of each arm
    change outcome: for a positive ``delta`` treated ``Y = 0`` become 1 and
    control ``Y = 1`` become 0 (the reverse for a negative one); the log records
    the realized shift. Rates move additively, so the differences across a
    candidate's levels that the naturalness requirements read are unchanged at
    the stratum level. The draw depends only on the trial, the study and
    ``delta``, so every replication and selection covariate sees the same trial.

    Args:
        trial: Target trial rows with ``id``, ``iv``, ``dv_binary`` and ``covariates``.
        delta: CATE shift, strictly inside (-1, 1).
        covariates: Stratum covariates.
        study: Study name, part of the seed.

    Returns:
        ``flips`` and the per-stratum ``log`` of :func:`_flip_target_outcomes`.

    Raises:
        ValueError: If ``delta`` is outside (-1, 1) or the trial repeats an ``id``.
    """
    if not -1.0 < float(delta) < 1.0:
        raise ValueError(f"delta must lie in (-1, 1); got {delta}")
    _check_trial_ids(trial)
    half = float(delta) / 2.0
    seed = [EFFECT_SHIFT_SEED, zlib.crc32(str(study).encode()), int(round((float(delta) + 1.0) * 1e6))]
    return _flip_target_outcomes(trial, lambda record, arm: half if arm == 1 else -half,
                                 covariates=covariates, seed=seed)


def direct_effect_flips(
    trial: pd.DataFrame, gamma: float, *, covariates: Sequence[str], candidate: str, study: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the target-trial outcomes to flip so that ``candidate`` gets a direct effect ``gamma`` on Y.

    The invalid-instrument arm. Both arms' outcome rates rise by ``gamma / 2``
    in every stratum with ``candidate = 1`` and fall by ``gamma / 2`` where
    ``candidate = 0``, so the exclusion restriction fails by ``gamma`` while
    every stratum CATE is unchanged up to rounding. Where an arm cannot move by
    ``gamma / 2`` both arms of that stratum move by what it can; the log records
    the realized rates.

    Args:
        trial: Target trial rows with ``id``, ``iv``, ``dv_binary`` and ``covariates``.
        gamma: Direct effect, in [0, 1).
        covariates: Stratum covariates; must include ``candidate``.
        candidate: The covariate given the direct effect.
        study: Study name, part of the seed with ``candidate``.

    Returns:
        ``flips`` and the per-stratum ``log`` of :func:`_flip_target_outcomes`.

    Raises:
        ValueError: If ``gamma`` is outside [0, 1), ``candidate`` is not a stratum
            covariate, or the trial repeats an ``id``.
    """
    if not 0.0 <= float(gamma) < 1.0:
        raise ValueError(f"gamma must lie in [0, 1); got {gamma}")
    if candidate not in covariates:
        raise ValueError(f"{candidate} is not one of the stratum covariates {list(covariates)}")
    _check_trial_ids(trial)
    half = float(gamma) / 2.0
    seed = [EFFECT_SHIFT_SEED, zlib.crc32(f"{study}|{candidate}".encode()), int(round((float(gamma) + 2.0) * 1e6))]
    return _flip_target_outcomes(trial, lambda record, arm: half if int(record[candidate]) == 1 else -half,
                                 covariates=covariates, seed=seed, equal_arms=True)


def apply_effect_shift(rows: pd.DataFrame, flips: pd.DataFrame, *, target_site: str) -> pd.DataFrame:
    """Give the flipped target participants their new outcome, matched by ``id``.

    Only rows of ``target_site`` change, since ids are unique within a site's
    trial but not necessarily across sites.
    """
    shifted = rows.copy()
    if flips.empty:
        return shifted
    at_target = shifted["site"].astype(str).eq(str(target_site))
    new = shifted.loc[at_target, "id"].map(flips.set_index("id")["dv_binary"])
    changed = new.notna()
    shifted.loc[new.index[changed], "dv_binary"] = new[changed].astype(int).to_numpy()
    return shifted


def shift_target_data(
    osrct: pd.DataFrame, rct: pd.DataFrame, *, target_site: str, covariates: Sequence[str], study: str,
    delta: float = 0.0, direct_effect: float = 0.0, candidate: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Modify the target's outcomes in its trial and in its observational sample alike.

    ``delta`` shifts every stratum CATE (:func:`effect_shift_flips`);
    ``direct_effect`` gives ``candidate`` a direct effect on the outcome
    (:func:`direct_effect_flips`); exactly one of the two is used. OSRCT
    selection never reads ``Y``, so the observational sample keeps its
    participants and only their outcomes change; other sites are untouched.

    Returns:
        The modified OSRCT rows, the modified trial rows and the per-stratum log.

    Raises:
        ValueError: If both modifications are asked for, or neither.
    """
    if (float(delta) != 0.0) == (float(direct_effect) != 0.0):
        raise ValueError("Give exactly one of delta and direct_effect")
    trial = rct.loc[rct["site"].astype(str).eq(str(target_site))]
    if float(delta) != 0.0:
        flips, log = effect_shift_flips(trial, delta, covariates=covariates, study=study)
    else:
        flips, log = direct_effect_flips(trial, direct_effect, covariates=covariates,
                                         candidate=str(candidate), study=study)
    return (apply_effect_shift(osrct, flips, target_site=target_site),
            apply_effect_shift(rct, flips, target_site=target_site), log)


def instrument_candidate(pattern: str) -> str:
    """Return the binary stratum covariate that the selection on ``pattern`` makes relevant."""
    return f"resp_{pattern}_bin"


def target_data(
    config: Exp16Config, effect_shift: float = 0.0, direct_effect: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild the target rows Exp16 used, optionally with modified outcomes.

    ``effect_shift`` shifts the target's stratum CATEs; ``direct_effect`` gives
    the binary version of the condition's selection covariate a direct effect
    on the outcome (:func:`shift_target_data`).

    Returns:
        The target idle rows (as in ``run_one_target``), the target trial rows,
        every site's trial rows and the modification log (empty without one).
    """
    osrct, rct, _ = load_prepared_data(config)
    target = str(config.target_sites[0])
    log = pd.DataFrame()
    if float(effect_shift) != 0.0 or float(direct_effect) != 0.0:
        osrct, rct, log = shift_target_data(
            osrct, rct, target_site=target, covariates=config.covariates, study=config.study,
            delta=float(effect_shift), direct_effect=float(direct_effect),
            candidate=instrument_candidate(config.pattern),
        )
    _, target_idle = create_train_target_split(
        osrct_data=osrct, rct_data=rct, target_site=target, site_col="site", regime_col="F",
    )
    trial = rct.loc[rct["site"].astype(str).eq(target)]
    return target_idle[[*config.covariates, *TARGET_IDLE_COLUMNS]].copy(), trial, rct, log


def target_idle_rows(config: Exp16Config, effect_shift: float = 0.0, direct_effect: float = 0.0) -> pd.DataFrame:
    """Rebuild the target idle rows Exp16 fitted, as in ``run_one_target``."""
    return target_data(config, effect_shift, direct_effect)[0]


def mismatched_source_condition(
    condition: Condition, mismatch: str, lookup: Mapping[tuple[int, str, str, float], Condition],
) -> Condition:
    """Return the condition whose source evidence grounds ``condition`` in the family-mismatch arm.

    Args:
        condition: Target condition.
        mismatch: One of :data:`SOURCE_MISMATCHES`.
        lookup: Every stored condition by ``(replication, study, pattern, beta)``.

    Raises:
        ValueError: If ``mismatch`` is unknown.
        KeyError: If the mismatched condition is not stored.
    """
    if mismatch not in SOURCE_MISMATCHES:
        raise ValueError(f"mismatch must be one of {SOURCE_MISMATCHES}; got {mismatch!r}")
    if mismatch == "none":
        return condition
    pattern, beta = condition.pattern, float(condition.beta)
    if mismatch == "beta":
        beta = MISMATCHED_BETA[round(beta, 6)]
    else:
        pattern = MISMATCHED_PATTERN[pattern]
    return lookup[(condition.replication, condition.study, pattern, round(beta, 6))]


def _difference_in_means(group: pd.DataFrame) -> tuple[float, float]:
    """Return a site-stratum difference in means and its smoothed Wald variance."""
    treated = group.loc[group["iv"].astype(int).eq(1), "dv_binary"].astype(float)
    control = group.loc[group["iv"].astype(int).eq(0), "dv_binary"].astype(float)
    # Add-one-half smoothing keeps the variance positive in all-0 or all-1 arms.
    p1 = (treated.sum() + 0.5) / (len(treated) + 1.0)
    p0 = (control.sum() + 0.5) / (len(control) + 1.0)
    return float(treated.mean() - control.mean()), float(p1 * (1 - p1) / len(treated) + p0 * (1 - p0) / len(control))


def transport_intervals(
    trials: pd.DataFrame, *, source_sites: Sequence[str], covariates: Sequence[str],
    level: float = TRANSPORT_LEVEL, min_arm_count: int = 2,
) -> pd.DataFrame:
    """Transport baseline: where a new site's stratum CATE should fall, from the source trials alone.

    In each stratum, every source site with at least ``min_arm_count``
    participants per arm contributes its difference in means. DerSimonian-Laird
    pools them into a mean ``mu`` with standard error ``se`` and a between-site
    variance ``tau2``, and the prediction interval ``mu +- t_{k-2} sqrt(tau2 +
    se^2)`` (Higgins, Thompson and Spiegelhalter, 2009) is the interval for a new
    site of the same family, clipped to [-1, 1]. It reads no target data, so it
    is valid exactly when the target's stratum effects are exchangeable with the
    sources'.

    Args:
        trials: Trial rows of every site with ``site``, ``iv``, ``dv_binary`` and ``covariates``.
        source_sites: Sites to pool, the condition's grounding sources.
        covariates: Stratum covariates.
        level: Prediction-interval level.
        min_arm_count: Fewest participants per arm for a site-stratum to enter.

    Returns:
        One row per stratum with ``k_sites``, ``mu``, ``se``, ``tau2``,
        ``lower`` and ``upper``; strata without a source give ``NaN`` endpoints.
    """
    sources = trials.loc[trials["site"].astype(str).isin([str(site) for site in source_sites])]
    rows = []
    for key, stratum in sources.groupby(list(covariates), sort=True):
        estimates = []
        for _, group in stratum.groupby("site", sort=True):
            arms = group["iv"].astype(int)
            if arms.eq(1).sum() >= min_arm_count and arms.eq(0).sum() >= min_arm_count:
                estimates.append(_difference_in_means(group))
        record = dict(zip(covariates, key if isinstance(key, tuple) else (key,)))
        record["k_sites"] = len(estimates)
        if not estimates:
            rows.append({**record, "mu": np.nan, "se": np.nan, "tau2": np.nan, "lower": np.nan, "upper": np.nan})
            continue
        effect, variance = (np.array(values) for values in zip(*estimates))
        weight = 1.0 / variance
        fixed = float(np.sum(weight * effect) / np.sum(weight))
        q = float(np.sum(weight * (effect - fixed) ** 2))
        c = float(np.sum(weight) - np.sum(weight**2) / np.sum(weight))
        tau2 = max(0.0, (q - (len(effect) - 1)) / c) if c > 0 else 0.0
        random_weight = 1.0 / (variance + tau2)
        mu = float(np.sum(random_weight * effect) / np.sum(random_weight))
        se = float(np.sqrt(1.0 / np.sum(random_weight)))
        half = float(stats.t.ppf(0.5 + level / 2.0, df=max(len(effect) - 2, 1))) * np.sqrt(tau2 + se**2)
        rows.append({**record, "mu": mu, "se": se, "tau2": tau2,
                     "lower": max(-1.0, mu - half), "upper": min(1.0, mu + half)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def _published_identification_rows(path: str) -> pd.DataFrame:
    """Read one store's published Manski/Balke-Pearl rows once per process."""
    return pd.read_parquet(path)


def published_identities(identities_root: Path, condition: Condition) -> pd.DataFrame:
    """Return the published primary strata of one condition, with their seed.

    The rows are the Manski rows of the identification bounds pass at the
    paper's floor (``<identities_root>/<store results root>/mc_intervals.parquet``),
    which carry the identity columns ``identification_bound_intervals`` checks.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    path = Path(identities_root) / Path(condition.store).parents[2].name / exp25.EXP25_INTERVALS_FILENAME
    rows = _published_identification_rows(str(path))
    rows = rows.loc[
        rows["method"].eq("manski") & rows["replication"].eq(condition.replication)
        & rows["study"].astype(str).eq(condition.study)
        & rows["bias_covariate"].astype(str).eq(condition.pattern)
        & np.isclose(rows["beta"].astype(float), float(condition.beta))
    ]
    if rows.empty:
        raise ValueError(f"No published identities for {condition.label} under {path}")
    return rows[[*exp25.EXP25_IDENTITY_COLUMNS, "seed"]].reset_index(drop=True)


def compute_target_tables(target_idle: pd.DataFrame, config: Exp16Config) -> dict[str, Any]:
    """Fit the target probability tables Exp16's triggers and LP read.

    Mirrors ``run_one_target``: one reusable TabICLv2 table serves both when
    Exp16 would reuse it; otherwise each is estimated as Exp16 would. A
    posterior-draw LP backend estimates per draw, so its LP table is ``None``.

    Returns:
        ``{"trigger": table or None, "lp": table or None}``.
    """
    reusable = target_pipeline.fit_reusable_tabicl_target_quantities(target_idle, config)
    if reusable is not None:
        return {"trigger": reusable, "lp": reusable}
    trigger = target_pipeline.estimate_target_trigger_quantities(
        target_idle, config, backend=resolve_target_trigger_probability_backend(config)
    )
    backend = resolve_lp_probability_backend(config)
    lp = None
    if not uses_hierarchical_dirichlet_draws(backend):
        lp = estimate_target_observed_binary_quantities(
            target_idle, treatment="iv", outcome="dv_binary",
            covariates=config.covariates, model_covariates=config.lp_probability_covariates,
            probability_backend=backend, min_count=int(config.min_cell_count),
            probability_model_kwargs=build_lp_probability_model_kwargs(config, backend),
        )
    return {"trigger": trigger, "lp": lp}


def admissible_choices(choices: pd.DataFrame) -> pd.DataFrame:
    """Keep, per candidate, the triggered ``(epsilon, tau)`` pairs that are admissible.

    The admissible pairs are the Pareto frontier of small ``epsilon`` and large
    ``tau``: a triggered pair is dropped when another triggered pair of the same
    candidate has an ``epsilon`` no larger and a ``tau`` no smaller, one of them
    strictly. The target triggers on ``{epsilon >= I_j} x {tau <= D_j}``, so the
    frontier is the single pair with the smallest ``epsilon`` and the largest
    ``tau``. With the tightest valid tolerances the dropped pairs only add looser
    constraints; with drawn tolerances that is not implied, so the choice matters.

    Args:
        choices: Selected target rows in policy order, as returned by
            :func:`target_pipeline.select_policy_choices`.

    Returns:
        The admissible rows in their original order, re-finalized so that the
        primary-choice flag and selection-mode metadata are current.
    """
    if choices.empty:
        return choices
    epsilon = pd.to_numeric(choices["epsilon"], errors="raise").to_numpy(float)
    tau = pd.to_numeric(choices["tau"], errors="raise").to_numpy(float)
    covariate = choices["covariate"].astype(str).to_numpy()
    keep = np.ones(len(choices), dtype=bool)
    for row in range(len(choices)):
        same = covariate == covariate[row]
        no_worse = (epsilon <= epsilon[row] + 1e-12) & (tau >= tau[row] - 1e-12)
        strictly_better = (epsilon < epsilon[row] - 1e-12) | (tau > tau[row] + 1e-12)
        keep[row] = not (same & no_worse & strictly_better).any()
    top_k = int(choices["policy_top_k"].iloc[0]) if "policy_top_k" in choices else 1
    return target_pipeline.finalize_policy_choices(
        choices.loc[keep], top_k=top_k, candidate_selection_mode="admissible_triggered",
    )


def reground_condition(
    *,
    config: Exp16Config,
    target_index: int,
    observations: pd.DataFrame,
    target_idle: pd.DataFrame,
    tables: Mapping[str, Any],
    floor: float,
    gp_training_target: str = GP_TRAINING_TARGET_PER_SITE,
    gp_target_scale: str = GP_TARGET_SCALE_NOTE,
    target_pairs: str = "all",
) -> dict[str, pd.DataFrame]:
    """Redo Exp16's grounding for one condition on a raised tau floor.

    Args:
        config: The condition's Exp16 configuration, as Exp25 ran it.
        target_index: Exp16 target index Exp25 used; it seeds the GP draws.
        observations: Stored Stage-2 observations of the final grounder.
        target_idle: Target idle rows.
        tables: Target probability tables from :func:`compute_target_tables`.
        floor: Smallest tau kept, in the source GP data and at the target.
        gp_training_target: ``per_site`` (Exp25) or ``envelope``; see the module docstring.
        gp_target_scale: ``note`` (Exp25) or ``tilde``.
        target_pairs: ``all`` (Exp25) or ``admissible`` (:func:`admissible_choices`).

    Returns:
        ``posterior_summary`` (the reported interval per stratum), ``choices``
        (selected target rows and their tolerances) and ``gp`` (surface diagnostics).

    Raises:
        ValueError: If a grounding setting is unsupported.
    """
    grounder, floored, choices = ground_and_choose(
        config=config, observations=observations, target_idle=target_idle, tables=tables, floor=floor,
        gp_training_target=gp_training_target, gp_target_scale=gp_target_scale, target_pairs=target_pairs,
    )
    _, _, summary = target_pipeline.solve_choices_lp_posterior_samples(
        target_idle, choices, grounder, floored,
        random_seed=int(config.seed + 20_000 + target_index),
        observed_quantities=tables.get("lp"),
    )
    gp = target_pipeline.grounder_gp_diagnostics(grounder, training_stage=f"tau_floor_{floor:g}")
    return {"posterior_summary": summary, "choices": choices, "gp": gp}


def ground_and_choose(
    *,
    config: Exp16Config,
    observations: pd.DataFrame,
    target_idle: pd.DataFrame,
    tables: Mapping[str, Any],
    floor: float,
    gp_training_target: str = GP_TRAINING_TARGET_PER_SITE,
    gp_target_scale: str = GP_TARGET_SCALE_NOTE,
    target_pairs: str = "all",
) -> tuple[Any, Exp16Config, pd.DataFrame]:
    """Fit the Stage-3 surfaces on the stored source rows and pick the target's triggered rows.

    The discovery half of :func:`reground_condition`, shared with the
    grounding ablation (:func:`ablate_condition`).

    Returns:
        The fitted grounder, the floored configuration and the target choices.

    Raises:
        ValueError: If a grounding setting is unsupported.
    """
    if target_pairs not in TARGET_PAIRS:
        raise ValueError(f"target_pairs must be one of {TARGET_PAIRS}; got {target_pairs!r}")
    taus = floor_taus(config.taus, floor)
    floored = replace(config, taus=taus)
    grounder = build_configured_naturalness_grounder(floored)
    grounder.gp_training_target = validate_gp_training_target(gp_training_target)
    grounder.gp_target_scale = validate_gp_target_scale(gp_target_scale)
    keep = np.isclose(observations["tau"].to_numpy(float)[:, None], np.asarray(taus)[None, :]).any(axis=1)
    grounder.observations_ = observations.loc[keep].reset_index(drop=True)
    grounder.treatment_, grounder.outcome_, grounder.regime_col_ = "iv", "dv_binary", "F"
    if floored.fit_gp:
        grounder.fit_stage3_gp()
    grounder.is_fitted_ = True

    candidates = target_pipeline.build_policy_candidates(
        grounder, target_idle, floored, trigger_quantities=tables.get("trigger")
    )
    choices = target_pipeline.select_policy_choices(
        candidates,
        min(int(value) for value in floored.policy_top_k_values),
        mode=floored.candidate_selection_mode,
    )
    if target_pairs == "admissible":
        choices = admissible_choices(choices)
    return grounder, floored, choices


#: Grounding ablation: the fixed tolerances, on the solver's c~ = 2c scale.
ABLATION_TWO_C = (0.1, 0.2, 0.3, 0.5)
ABLATION_TWO_D = (0.02, 0.05, 0.1, 0.2)
#: Offset of the prior-only tolerance draws' seed from the grounded LP seed.
ABLATION_PRIOR_SEED_OFFSET = 41
#: Margin keeping the exact rule's c~1 feasible at the target's own deviation.
EXACT_MARGIN = 1e-9


class _RuleTolerances:
    """Stand-in grounder whose tolerance draws follow an ablation rule, not the source trials.

    Every attribute but :meth:`sample_tolerances` is the wrapped grounder's, so
    the Exp16 LP posterior runs unchanged on the same target choices.

    Args:
        grounder: Fitted grounder of the condition, used only for its row checks.
        rule: ``fixed`` (``two_c``, ``two_d`` everywhere), ``exact`` (``two_c`` at
            each candidate's observed deviation ``i_deviation``, the smallest the
            target's data allow, and ``two_d = 0``), or ``prior`` (independent
            Uniform(0, 1) draws of ``two_c`` and ``two_d`` per candidate, the
            margins of the copula prior).
        two_c: Fixed ``2c`` for ``fixed``.
        two_d: Fixed ``2d`` for ``fixed``.
    """

    def __init__(self, grounder: Any, rule: str, *, two_c: float | None = None, two_d: float | None = None) -> None:
        """Wrap ``grounder`` under ``rule``."""
        if rule not in ("fixed", "exact", "prior"):
            raise ValueError(f"Unknown ablation rule {rule!r}")
        if rule == "fixed" and (two_c is None or two_d is None):
            raise ValueError("The fixed rule needs two_c and two_d")
        self._grounder, self._rule, self._two_c, self._two_d = grounder, rule, two_c, two_d

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the wrapped grounder."""
        return getattr(self._grounder, name)

    def sample_tolerances(
        self, rows: pd.DataFrame, n_samples: int, random_seed: int | None = None, clip: bool = True,
    ) -> pd.DataFrame:
        """Return ``n_samples`` tolerance draws per row in the grounder's long format."""
        prepared = self._grounder._prepare_tolerance_sample_rows(rows)
        rng = np.random.default_rng(random_seed)
        candidates = sorted(prepared["covariate"].astype(str).unique())
        records: list[dict[str, Any]] = []
        for sample_id in range(int(n_samples)):
            drawn = {name: (float(rng.uniform()), float(rng.uniform())) for name in candidates} \
                if self._rule == "prior" else {}
            for record in prepared.to_dict("records"):
                if self._rule == "fixed":
                    two_c, two_d = float(self._two_c), float(self._two_d)
                elif self._rule == "exact":
                    two_c, two_d = min(1.0, float(record["i_deviation"]) + EXACT_MARGIN), 0.0
                else:
                    two_c, two_d = drawn[str(record["covariate"])]
                records.append({**record, "sample_id": sample_id, "c": two_c / 2, "d": two_d / 2,
                                "two_c": two_c, "two_d": two_d, "c_source": self._rule,
                                "d_source": self._rule, "prediction_scale": "probability"})
        return pd.DataFrame(records)


def ablate_condition(
    *,
    config: Exp16Config,
    target_index: int,
    observations: pd.DataFrame,
    target_idle: pd.DataFrame,
    tables: Mapping[str, Any],
    floor: float,
    two_c_grid: Sequence[float] = ABLATION_TWO_C,
    two_d_grid: Sequence[float] = ABLATION_TWO_D,
) -> dict[str, pd.DataFrame]:
    """Solve one condition's target program with tolerances that do not use the source trials.

    The grounding ablation. Discovery is kept (the grounded run's triggered
    candidates, one admissible pair each, since with one tolerance per candidate
    the other pairs add nothing); only the tolerances change: every fixed pair of
    ``two_c_grid`` x ``two_d_grid`` and the ``exact`` and ``prior`` rules of
    :class:`_RuleTolerances`. Fixed and exact rules are one LP each; the prior
    rule uses the grounded run's number of draws and reports, as Exp25 does, the
    mean endpoints of the draws that solve. The Stage-3 surfaces only reproduce
    discovery (the envelope fit is the fast one) and set no tolerance.

    Returns:
        ``ablation_summary`` (one LP posterior summary per rule, with ``rule``,
        ``two_c`` and ``two_d``) and the ``choices`` the rules share.
    """
    grounder, floored, choices = ground_and_choose(
        config=config, observations=observations, target_idle=target_idle, tables=tables, floor=floor,
        gp_training_target="envelope", gp_target_scale="tilde", target_pairs="admissible",
    )
    rules: list[tuple[str, float | None, float | None]] = [("exact", None, None), ("prior", None, None)]
    rules += [("fixed", float(two_c), float(two_d)) for two_c in two_c_grid for two_d in two_d_grid]
    seed = int(config.seed + 20_000 + target_index + ABLATION_PRIOR_SEED_OFFSET)
    summaries = []
    for rule, two_c, two_d in rules:
        draws = int(floored.lp_posterior_samples) if rule == "prior" else 1
        _, _, summary = target_pipeline.solve_choices_lp_posterior_samples(
            target_idle, choices, _RuleTolerances(grounder, rule, two_c=two_c, two_d=two_d),
            replace(floored, lp_posterior_samples=draws), random_seed=seed,
            observed_quantities=tables.get("lp"),
        )
        label = rule if rule != "fixed" else f"fixed_{two_c:g}_{two_d:g}"
        summaries.append(summary.assign(rule=label, two_c=two_c, two_d=two_d))
    return {"ablation_summary": pd.concat(summaries, ignore_index=True), "choices": choices}


def run_ablation(
    conditions: Sequence[Condition], output_dir: Path, *, tables_dir: Path, floor: float,
    lp_posterior_n_jobs: int, two_c_grid: Sequence[float] = ABLATION_TWO_C,
    two_d_grid: Sequence[float] = ABLATION_TWO_D,
) -> None:
    """Run :func:`ablate_condition` on every condition with the published target tables.

    Each leaf ``output_dir/floor=<floor>/<condition>/`` gets ``choices`` and,
    last since it marks completion, ``ablation_summary``. Finished leaves are skipped.
    """
    for condition in conditions:
        leaf = _condition_dir(output_dir / f"floor={floor:g}", condition)
        if (leaf / "ablation_summary.parquet").exists():
            continue
        start = time.time()
        with (_condition_dir(Path(tables_dir), condition) / "target_tables.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        config, target_index, _ = condition_config(condition)
        config = replace(config, lp_posterior_n_jobs=int(lp_posterior_n_jobs))
        observations = read_stage2_observations(bundle_leaf(condition, config), condition.study)
        outputs = ablate_condition(
            config=config, target_index=target_index, observations=observations,
            target_idle=payload["target_idle"], tables=payload["tables"], floor=floor,
            two_c_grid=two_c_grid, two_d_grid=two_d_grid,
        )
        atomic_write_parquet(_parquet_safe(outputs["choices"]), leaf / "choices.parquet")
        atomic_write_parquet(_parquet_safe(outputs["ablation_summary"]), leaf / "ablation_summary.parquet")
        print(f"ablate floor={floor:g} {condition.label}: {time.time() - start:.1f}s", flush=True)


def _condition_dir(root: Path, condition: Condition) -> Path:
    """Return the per-condition directory under a pass root."""
    return root / condition.label


def _shard(items: Sequence[Any], shard: int, n_shards: int) -> list[Any]:
    """Return the items of one shard; shards interleave for balance."""
    if not 0 <= shard < n_shards:
        raise ValueError("shard must lie in [0, n_shards)")
    return list(items[shard::n_shards])


def run_tables(
    conditions: Sequence[Condition], output_dir: Path, *, tabicl_device: str, effect_shift: float = 0.0,
    direct_effect: float = 0.0,
) -> None:
    """Fit and pickle the target tables and idle rows for each condition.

    With ``effect_shift`` or ``direct_effect`` the target rows carry the
    modified outcomes of :func:`shift_target_data`, and the payload records it.
    """
    for condition in conditions:
        path = _condition_dir(output_dir / "tables", condition) / "target_tables.pkl"
        if path.exists():
            continue
        config, _, _ = condition_config(condition)
        config = replace(config, tabicl_device=tabicl_device, tabicl_share_checkpoint=True)
        start = time.time()
        target_idle = target_idle_rows(config, effect_shift=effect_shift, direct_effect=direct_effect)
        payload = {"target_idle": target_idle, "tables": compute_target_tables(target_idle, config),
                   "effect_shift": float(effect_shift), "direct_effect": float(direct_effect)}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle)
        temporary.replace(path)
        print(f"tables {condition.label}: {time.time() - start:.1f}s", flush=True)


def _wait_for(path: Path, wait_seconds: float, poll_seconds: float = 30.0) -> None:
    """Block until ``path`` exists, for at most ``wait_seconds``.

    Raises:
        FileNotFoundError: If the file has not appeared in time.
    """
    deadline = time.time() + float(wait_seconds)
    while not path.exists():
        if time.time() >= deadline:
            raise FileNotFoundError(f"Run the tables pass first: {path}")
        time.sleep(min(poll_seconds, max(deadline - time.time(), 0.0)))


def shifted_target_outputs(
    condition: Condition, config: Exp16Config, *, effect_shift: float, floor: float,
    identities_root: Path, source_sites: Sequence[str], target_idle: pd.DataFrame,
    direct_effect: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Score a modified-target arm's target-side comparators for one condition.

    Recomputes the target rows with the modification (``effect_shift`` or
    ``direct_effect``, :func:`target_data`), checks them against the rows the
    tables pass fitted, and returns the modified reference (the Exp16 RCT
    labels of the modified trial), Manski and Balke-Pearl on the modified target
    rows with the paper's relevance threshold and support floor, the transport
    baseline over the condition's grounding sources, and the modification log.

    Raises:
        ValueError: If the tables' target rows differ from the modified rows.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    idle, trial, trials, log = target_data(config, effect_shift, direct_effect)
    key = [*config.covariates, "iv", "dv_binary"]
    if not idle[key].reset_index(drop=True).astype(int).equals(target_idle[key].reset_index(drop=True).astype(int)):
        raise ValueError(f"{condition.label}: the tables' target rows are not the rows modified by "
                         f"effect_shift={effect_shift}, direct_effect={direct_effect}")
    reference = target_pipeline.build_empirical_rct_cate_labels(
        trial, covariates=config.covariates, min_arm_count=int(config.rct_min_arm_count),
    )
    published = published_identities(identities_root, condition)
    identities = published.drop(columns=["rct_cate", "rct_cate_ci90_lower", "rct_cate_ci90_upper"]).merge(
        reference[[*config.covariates, "rct_cate", "rct_cate_ci90_lower", "rct_cate_ci90_upper"]],
        on=list(config.covariates), how="left", validate="one_to_one",
    )
    identities["rct_cate_available"] = identities["rct_cate"].notna()
    store_config = load_exp25_store_config(Path(condition.store))
    min_count = int(store_config.identification_min_count)
    recorded = Path(condition.store) / "baseline_variants" / exp25.EXP25_IDENTIFICATION_DIRNAME / "run_manifest.yml"
    if recorded.exists():
        manifest = yaml.safe_load(recorded.read_text(encoding="utf-8"))
        min_count = int(manifest["extra"]["identification_config"]["min_count"])
    bounds = exp25.identification_bound_intervals(
        {condition.study: idle}, identities[list(exp25.EXP25_IDENTITY_COLUMNS)],
        min_count=min_count, tau=min(floor_taus(config.taus, floor)),
        replication=condition.replication, seed=int(published["seed"].iloc[0]),
        pattern=condition.pattern, beta=float(condition.beta),
        config_fingerprint_value=f"effect_shift={effect_shift:g},direct_effect={direct_effect:g}",
    )
    transport = transport_intervals(trials, source_sites=source_sites, covariates=config.covariates)
    return {"reference": reference, "identification_bounds": bounds, "transport": transport,
            "effect_shift_log": log}


def run_reground(
    conditions: Sequence[Condition],
    output_dir: Path,
    *,
    floors: Sequence[float],
    validate_replications: Sequence[int],
    lp_posterior_n_jobs: int,
    wait_for_tables: float = 0.0,
    tables_dir: Path | None = None,
    grounding: Mapping[str, str] | None = None,
    effect_shift: float | None = None,
    identities_root: Path | None = None,
    source_mismatch: str = "none",
    lookup: Mapping[tuple[int, str, str, float], Condition] | None = None,
    direct_effect: float | None = None,
) -> None:
    """Reground every condition at every floor; floor 0.01 only where validated.

    Tables are read from ``tables_dir`` (default ``output_dir / "tables"``),
    waiting up to ``wait_for_tables`` seconds for each, so this pass can run
    beside the tables pass. ``grounding`` holds the keyword settings of
    :func:`reground_condition`. With ``effect_shift`` or ``direct_effect`` (the
    tables must carry the same modification) each leaf also gets the outputs of
    :func:`shifted_target_outputs`, which need ``identities_root``; with a
    ``source_mismatch`` the source evidence comes from
    :func:`mismatched_source_condition`, which needs ``lookup``. Every leaf
    records these settings in ``provenance.json``.

    Raises:
        ValueError: If the tables were built with another shift, or a setting
            lacks the input it needs.
    """
    modified = effect_shift is not None or direct_effect is not None
    if modified and identities_root is None:
        raise ValueError("effect_shift and direct_effect need identities_root")
    if source_mismatch != "none" and lookup is None:
        raise ValueError("source_mismatch needs the condition lookup")
    tables_root = output_dir / "tables" if tables_dir is None else Path(tables_dir)
    for condition in conditions:
        table_path = _condition_dir(tables_root, condition) / "target_tables.pkl"
        pending = [
            floor for floor in floors
            if (floor > 0.01 + 1e-12 or condition.replication in validate_replications)
            and not (_condition_dir(output_dir / f"floor={floor:g}", condition) / "posterior_summary.parquet").exists()
        ]
        if not pending:
            continue
        _wait_for(table_path, wait_for_tables)
        config, target_index, _ = condition_config(condition)
        config = replace(config, lp_posterior_n_jobs=int(lp_posterior_n_jobs))
        with table_path.open("rb") as handle:
            payload = pickle.load(handle)
        for name, wanted in (("effect_shift", effect_shift), ("direct_effect", direct_effect)):
            built = float(payload.get(name, 0.0))
            if not np.isclose(built, 0.0 if wanted is None else float(wanted)):
                raise ValueError(f"{table_path} was built with {name}={built}, not {wanted}")
        source = mismatched_source_condition(condition, source_mismatch, lookup or {})
        source_config = config if source == condition else condition_config(source)[0]
        observations = read_stage2_observations(bundle_leaf(source, source_config), condition.study)
        for floor in pending:
            start = time.time()
            outputs = reground_condition(
                config=config, target_index=target_index, observations=observations,
                target_idle=payload["target_idle"], tables=payload["tables"], floor=floor,
                **dict(grounding or {}),
            )
            if modified:
                outputs.update(shifted_target_outputs(
                    condition, config, effect_shift=float(effect_shift or 0.0), floor=floor,
                    identities_root=Path(identities_root),
                    source_sites=sorted(observations["site_id"].astype(str).unique()),
                    target_idle=payload["target_idle"], direct_effect=float(direct_effect or 0.0),
                ))
            leaf = _condition_dir(output_dir / f"floor={floor:g}", condition)
            extra = [name for name in ("reference", "identification_bounds", "transport", "effect_shift_log")
                     if name in outputs]
            for name in ("choices", "gp", *extra):
                atomic_write_parquet(_parquet_safe(outputs[name]), leaf / f"{name}.parquet")
            provenance = {
                "condition": condition.label, "source_condition": source.label,
                "source_store": source.store, "source_mismatch": source_mismatch,
                "effect_shift": effect_shift, "direct_effect": direct_effect,
                "floor": floor, "grounding": dict(grounding or {}),
                "tables": str(table_path),
            }
            (leaf / "provenance.json").write_text(json.dumps(provenance, indent=2, default=json_default),
                                                  encoding="utf-8")
            # The summary goes last: it marks the leaf complete.
            atomic_write_parquet(_parquet_safe(outputs["posterior_summary"]), leaf / "posterior_summary.parquet")
            print(f"reground floor={floor:g} {condition.label}: {time.time() - start:.1f}s", flush=True)


def summarize(conditions: Sequence[Condition], output_dir: Path, *, floors: Sequence[float]) -> dict[str, pd.DataFrame]:
    """Pair every regrounded interval with the published Exp25 interval.

    Writes ``regrounding_strata`` (one row per floor, condition and stratum),
    ``regrounding_candidates`` (binding 2c/2d per candidate, regrounded against
    published) and ``regrounding_summary`` (per floor and beta).

    Returns:
        The three tables by name.
    """
    strata_rows, candidate_rows = [], []
    for condition in conditions:
        config, _, _ = condition_config(condition)
        leaf = bundle_leaf(condition, config)
        published = pd.read_parquet(leaf / "target_gp_posterior_summary.parquet")
        published = published.loc[published["study"].astype(str).eq(condition.study)]
        published_choices = pd.read_parquet(leaf / "target_policy_choices.parquet")
        published_choices = published_choices.loc[
            published_choices["study"].astype(str).eq(condition.study) & published_choices["covariate"].notna()
        ]
        base = published[[*config.covariates, "lower_mean", "upper_mean", "lp_posterior_prob_optimal", "rct_cate"]].rename(
            columns={"lower_mean": "published_lower", "upper_mean": "published_upper",
                     "lp_posterior_prob_optimal": "published_prob_optimal"})
        old_binding = published_choices.groupby("covariate").agg(published_min_2c=("two_c", "min"), published_min_2d=("two_d", "min"))
        keys = {"replication": condition.replication, "study": condition.study, "pattern": condition.pattern, "beta": condition.beta}
        for floor in floors:
            fleaf = _condition_dir(output_dir / f"floor={floor:g}", condition)
            if not (fleaf / "posterior_summary.parquet").exists():
                continue
            summary = pd.read_parquet(fleaf / "posterior_summary.parquet")
            merged = base.merge(
                summary[[*config.covariates, "lower_mean", "upper_mean", "lp_posterior_prob_optimal"]].rename(
                    columns={"lower_mean": "lower", "upper_mean": "upper", "lp_posterior_prob_optimal": "prob_optimal"}),
                on=list(config.covariates), how="left", validate="one_to_one",
            )
            strata_rows.append(merged.assign(floor=floor, **keys))
            choices = pd.read_parquet(fleaf / "choices.parquet")
            new_binding = (
                choices.loc[choices["covariate"].notna()].groupby("covariate").agg(min_2c=("two_c", "min"), min_2d=("two_d", "min"))
                if "covariate" in choices else pd.DataFrame(columns=["min_2c", "min_2d"])
            )
            both = old_binding.join(new_binding, how="outer").reset_index().rename(columns={"index": "covariate"})
            candidate_rows.append(both.assign(floor=floor, **keys))
    if not strata_rows:
        return {}
    strata = pd.concat(strata_rows, ignore_index=True)
    strata["published_width"] = strata.published_upper - strata.published_lower
    strata["width"] = strata.upper - strata.lower
    strata["width_change"] = strata.width - strata.published_width
    truth = pd.to_numeric(strata["rct_cate"], errors="coerce")
    strata["published_covers"] = truth.between(strata.published_lower, strata.published_upper)
    strata["covers"] = truth.between(strata.lower, strata.upper)
    candidates = pd.concat(candidate_rows, ignore_index=True)
    candidates["bias_covariate"] = candidates.covariate.eq("resp_" + candidates.pattern + "_bin")
    candidates["vanished"] = candidates.min_2c.isna() & candidates.published_min_2c.notna()
    scored = strata[truth.notna()]
    summary = strata.groupby(["floor", "beta"]).agg(
        strata=("width", "size"),
        with_interval=("width", lambda s: int(s.notna().sum())),
        published_with_interval=("published_width", lambda s: int(s.notna().sum())),
        median_width=("width", "median"),
        median_published_width=("published_width", "median"),
        median_width_change=("width_change", "median"),
        max_abs_width_change=("width_change", lambda s: float(s.abs().max())),
        share_wider=("width_change", lambda s: float((s > 1e-9).mean())),
        share_narrower=("width_change", lambda s: float((s < -1e-9).mean())),
        prob_optimal=("prob_optimal", "mean"),
        published_prob_optimal=("published_prob_optimal", "mean"),
    ).join(scored.groupby(["floor", "beta"]).agg(
        coverage=("covers", "mean"), published_coverage=("published_covers", "mean"),
    )).reset_index()
    tables = {"regrounding_strata": strata, "regrounding_candidates": candidates, "regrounding_summary": summary}
    for name, table in tables.items():
        atomic_write_parquet(table, output_dir / f"{name}.parquet")
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


#: Groundings of the mechanism arms and their directory names.
ARM_GROUNDINGS = ("per_site", "theory")
#: Effect shifts of the effect-shift arm; 0 reads the published leaves.
ARM_SHIFTS = (0.0, 0.1, 0.2, 0.3)
#: Direct effects of the invalid-instrument arm (``iv_gamma=<gamma>`` directories).
ARM_DIRECT_EFFECTS = (0.05, 0.1, 0.2)


def _score_against_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Add width, hit, decisiveness, confirmed sign and sign reversal per stratum."""
    out = frame.copy()
    out["width"] = out["upper"] - out["lower"]
    out["hit"] = out["lower"].le(out["rct_cate"]) & out["upper"].ge(out["rct_cate"])
    positive, negative = out["lower"].gt(0), out["upper"].lt(0)
    out["decisive"] = positive | negative
    out["confirmed"] = (positive & out["rct_cate_ci90_lower"].gt(0)) | (negative & out["rct_cate_ci90_upper"].lt(0))
    out["reversal"] = (positive & out["rct_cate_ci90_upper"].lt(0)) | (negative & out["rct_cate_ci90_lower"].gt(0))
    return out


def _aggregate_scores(scored: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Average the stratum scores over ``keys``; the correct-sign rate is among decisive strata."""
    table = scored.groupby(list(keys), sort=True, dropna=False).agg(
        strata=("width", "size"), width=("width", "mean"), hit=("hit", "mean"),
        decisive=("decisive", "mean"), decisive_n=("decisive", "sum"), confirmed_n=("confirmed", "sum"),
        reversals=("reversal", "sum"), no_interval=("no_interval", "sum"),
    ).reset_index()
    table["correct_sign_rate"] = table.confirmed_n / table.decisive_n.where(table.decisive_n > 0)
    return table.drop(columns=["decisive_n", "confirmed_n"])


def _arm_rows(
    intervals: pd.DataFrame, manski: pd.DataFrame, reference: pd.DataFrame, keys: Mapping[str, Any],
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Join one method's stratum intervals to Manski and the reference; Manski where it has none."""
    rows = manski[[*covariates, "manski_lower", "manski_upper"]].merge(
        reference[[*covariates, "rct_cate", "rct_cate_ci90_lower", "rct_cate_ci90_upper"]],
        on=list(covariates), validate="one_to_one",
    ).merge(intervals[[*covariates, "lower", "upper"]], on=list(covariates), how="left", validate="one_to_one")
    missing = rows["lower"].isna() | rows["upper"].isna()
    rows["no_interval"] = missing
    rows["lower"] = rows["lower"].where(~missing, rows["manski_lower"])
    rows["upper"] = rows["upper"].where(~missing, rows["manski_upper"])
    return rows.drop(columns=["manski_lower", "manski_upper"]).assign(**keys)


def _naturalness_intervals(leaf: Path) -> pd.DataFrame | None:
    """Return a leaf's Naturalness interval per stratum, NaN where no draw was optimal."""
    path = leaf / "posterior_summary.parquet"
    if not path.exists():
        return None
    return _optimal_endpoints(pd.read_parquet(path))


def _optimal_endpoints(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep an LP posterior summary's mean endpoints only where some draw was optimal."""
    failed = pd.to_numeric(summary["lp_posterior_prob_optimal"], errors="coerce").fillna(0).le(0)
    return summary.assign(lower=summary["lower_mean"].where(~failed), upper=summary["upper_mean"].where(~failed))


def _binding_tolerances(leaf: Path) -> pd.DataFrame | None:
    """Return the smallest 2c and 2d entering the program, per candidate."""
    path = leaf / "choices.parquet"
    if not path.exists():
        return None
    choices = pd.read_parquet(path)
    choices = choices.loc[choices["covariate"].notna()] if "covariate" in choices else choices.iloc[0:0]
    if choices.empty:
        return pd.DataFrame(columns=["covariate", "triplets", "binding_2c", "binding_2d"])
    return choices.groupby("covariate", as_index=False).agg(
        triplets=("tau", "size"), binding_2c=("two_c", "min"), binding_2d=("two_d", "min"))


def family_mismatch_table_latex(strata: pd.DataFrame, *, grounding: str = "per_site") -> str:
    """Set the family-mismatch arm as the paper's table (``tab:ablation-family``).

    One row per target ``beta`` and a pooled row; each cell is the mean width and,
    in parentheses, the hit ratio, scored as in Experiment I, of Naturalness
    grounded on the matched sources, on the farthest-``beta`` condition and on
    another covariate's condition, beside Balke-Pearl, which reads no source.
    Only the tabular block is emitted; the caption and label belong to the paper.

    Args:
        strata: The arms' stratum table (``summary/arms_strata.parquet``).
        grounding: Naturalness grounding to report.

    Returns:
        LaTeX tabular source, or an empty string when the arm has no rows.
    """
    family = strata.loc[strata["arm"].eq("family_mismatch") & strata["grounding"].eq(grounding)]
    if family.empty:
        return ""
    baseline = strata.loc[strata["arm"].eq("effect_shift") & strata["delta"].eq(0.0)]
    betas = sorted(float(beta) for beta in family["beta"].unique())
    columns = {
        "Matched": baseline.loc[baseline["method"].eq("naturalness") & baseline["grounding"].eq(grounding)],
        r"Another $\beta$": family.loc[family["mismatch"].eq("beta")],
        "Another covariate": family.loc[family["mismatch"].eq("covariate")],
        "Balke--Pearl": baseline.loc[baseline["method"].eq("balke_pearl")],
    }
    columns = {label: frame.loc[frame["beta"].astype(float).isin(betas)] for label, frame in columns.items()}

    def cell(frame: pd.DataFrame) -> str:
        """Format one column's rows as ``mean width (hit ratio)``."""
        if frame.empty:
            return "--"
        return f"{frame['width'].mean():.2f} ({frame['hit'].astype(float).mean():.2f})"

    rows = [{r"$\beta$": f"{beta:g}", **{label: cell(frame.loc[np.isclose(frame["beta"].astype(float), beta)])
                                         for label, frame in columns.items()}} for beta in betas]
    rows.append({r"$\beta$": "Pooled", **{label: cell(frame) for label, frame in columns.items()}})
    return render_latex_table(pd.DataFrame(rows), column_format="lcccc", midrule_before=[len(betas)])


def summarize_arms(
    conditions: Sequence[Condition], arms_root: Path, *, published_root: Path, identities_root: Path,
    floor: float = MIN_RELEVANCE_TAU,
) -> dict[str, pd.DataFrame]:
    """Score the four mechanism arms of Exp25 and write their tables.

    Every method is scored on every stratum against the difference-in-means
    reference of the (modified) target trial, as in Experiment I: a stratum a
    method cannot bound gets its Manski interval. The matched baseline reads the
    published grounding (``published_root/floor=<floor>``) and identification
    bounds (``identities_root``) for ``per_site``, and ``arms_root/matched/theory``
    for ``theory``. The arms read, under ``arms_root``:

    * effect shift, ``delta=<D>/<grounding>``: Naturalness, Manski, Balke-Pearl, transport;
    * invalid instrument, ``iv_gamma=<gamma>/<grounding>``: the same methods;
    * family mismatch, ``d3_<beta|covariate>/<grounding>``: Naturalness only;
    * grounding ablation, ``ablation``: the fixed, exact and prior-only rules.

    Missing leaves are skipped and listed, so partial runs can be summarized.
    Writes ``arms_strata.parquet`` and one CSV per table under ``arms_root/summary``.

    Returns:
        The tables by name.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    covariates = list(exp25.EXP25_STRATUM_COLUMNS)
    frames, tolerances, realized, missing = [], [], [], []
    modified_arms = (("effect_shift", "delta", ARM_SHIFTS[1:], "delta={:g}"),
                     ("invalid_instrument", "gamma", ARM_DIRECT_EFFECTS, "iv_gamma={:g}"))
    for condition in conditions:
        leaf_name = Path(f"floor={floor:g}") / condition.label
        published = _published_identification_rows(
            str(Path(identities_root) / Path(condition.store).parents[2].name / exp25.EXP25_INTERVALS_FILENAME))
        published = published.loc[
            published["replication"].eq(condition.replication) & published["study"].astype(str).eq(condition.study)
            & published["bias_covariate"].astype(str).eq(condition.pattern)
            & np.isclose(published["beta"].astype(float), float(condition.beta))]
        base_manski = published.loc[published["method"].eq("manski")].rename(
            columns={"lower": "manski_lower", "upper": "manski_upper"})
        keys = {"replication": condition.replication, "study": condition.study,
                "pattern": condition.pattern, "beta": float(condition.beta)}
        base_keys = {"arm": "effect_shift", "delta": 0.0, "gamma": 0.0, "mismatch": "none", **keys}

        # Matched baseline: published per-site grounding, matched envelope run, published bounds.
        comparators = {"manski": published.loc[published["method"].eq("manski")],
                       "balke_pearl": published.loc[published["method"].eq("balke_pearl")]}
        transport = next((arms_root / sub / grounding / leaf_name / "transport.parquet"
                          for sub in ("delta=0.1", "delta=0.2", "delta=0.3", "iv_gamma=0.05", "iv_gamma=0.1")
                          for grounding in ARM_GROUNDINGS
                          if (arms_root / sub / grounding / leaf_name / "transport.parquet").exists()), None)
        if transport is not None:
            comparators["transport"] = pd.read_parquet(transport)
        for method, intervals in comparators.items():
            frames.append(_arm_rows(intervals, base_manski, base_manski, {
                **base_keys, "grounding": "", "method": method}, covariates))
        for grounding, leaf in (("per_site", Path(published_root) / leaf_name),
                                ("theory", arms_root / "matched" / "theory" / leaf_name)):
            intervals = _naturalness_intervals(leaf)
            if intervals is None:
                missing.append(f"matched/{grounding} {condition.label}")
                continue
            frames.append(_arm_rows(intervals, base_manski, base_manski, {
                **base_keys, "grounding": grounding, "method": "naturalness"}, covariates))
            binding = _binding_tolerances(leaf)
            if binding is not None:
                tolerances.append(binding.assign(mismatch="none", grounding=grounding, **keys))

        # Modified targets: shifted effects, and an instrument with a direct effect on Y.
        for arm, size_key, sizes, pattern in modified_arms:
            for size in sizes:
                leaves = {grounding: arms_root / pattern.format(size) / grounding / leaf_name
                          for grounding in ARM_GROUNDINGS}
                side = next((leaf for leaf in leaves.values() if (leaf / "reference.parquet").exists()), None)
                if side is None:
                    missing.append(f"{pattern.format(size)} {condition.label}")
                    continue
                bounds = pd.read_parquet(side / "identification_bounds.parquet")
                manski = bounds.loc[bounds["method"].eq("manski")].rename(
                    columns={"lower": "manski_lower", "upper": "manski_upper"})
                reference = pd.read_parquet(side / "reference.parquet")
                arm_keys = {**base_keys, "arm": arm, "delta": 0.0, "gamma": 0.0, size_key: float(size)}
                methods = {"manski": bounds.loc[bounds["method"].eq("manski")],
                           "balke_pearl": bounds.loc[bounds["method"].eq("balke_pearl")],
                           "transport": pd.read_parquet(side / "transport.parquet")}
                for method, intervals in methods.items():
                    frames.append(_arm_rows(intervals, manski, reference, {
                        **arm_keys, "grounding": "", "method": method}, covariates))
                for grounding, leaf in leaves.items():
                    intervals = _naturalness_intervals(leaf)
                    if intervals is None:
                        missing.append(f"{pattern.format(size)}/{grounding} {condition.label}")
                        continue
                    frames.append(_arm_rows(intervals, manski, reference, {
                        **arm_keys, "grounding": grounding, "method": "naturalness"}, covariates))
                log = pd.read_parquet(side / "effect_shift_log.parquet")
                realized.append(log.assign(arm=arm, size=float(size), **keys))

        # Family mismatch: another condition's sources.
        for mismatch in ("beta", "covariate"):
            for grounding in ARM_GROUNDINGS:
                leaf = arms_root / f"d3_{mismatch}" / grounding / leaf_name
                intervals = _naturalness_intervals(leaf)
                if intervals is None:
                    missing.append(f"d3_{mismatch}/{grounding} {condition.label}")
                    continue
                frames.append(_arm_rows(intervals, base_manski, base_manski, {
                    **base_keys, "arm": "family_mismatch", "mismatch": mismatch, "grounding": grounding,
                    "method": "naturalness"}, covariates))
                binding = _binding_tolerances(leaf)
                if binding is not None:
                    tolerances.append(binding.assign(mismatch=mismatch, grounding=grounding, **keys))

        # Grounding ablation: tolerances that ignore the source trials.
        path = arms_root / "ablation" / leaf_name / "ablation_summary.parquet"
        if path.exists():
            for rule, summary in pd.read_parquet(path).groupby("rule", sort=True):
                frames.append(_arm_rows(_optimal_endpoints(summary), base_manski, base_manski, {
                    **base_keys, "arm": "grounding_ablation", "grounding": str(rule), "method": "naturalness"},
                    covariates))
        else:
            missing.append(f"ablation {condition.label}")
    if not frames:
        return {}
    strata = _score_against_reference(pd.concat(frames, ignore_index=True))
    baseline = strata.loc[strata["arm"].eq("effect_shift") & strata["delta"].eq(0.0)]
    shift = strata.loc[strata["arm"].eq("effect_shift")]
    invalid = pd.concat([baseline, strata.loc[strata["arm"].eq("invalid_instrument")]])
    matched = baseline.loc[baseline["method"].eq("naturalness")]
    mismatched = pd.concat([matched, strata.loc[strata["arm"].eq("family_mismatch")]])
    pair_keys = ["grounding", "replication", "study", "pattern", "beta", *covariates]
    paired = mismatched.merge(matched[[*pair_keys, "width", "hit"]], on=pair_keys, suffixes=("", "_matched"))
    paired["width_change"] = paired["width"] - paired["width_matched"]
    paired["hit_change"] = paired["hit"].astype(float) - paired["hit_matched"].astype(float)
    ablation = pd.concat([matched, strata.loc[strata["arm"].eq("grounding_ablation")]])
    tables = {
        "effect_shift_by_beta": _aggregate_scores(shift, ["delta", "method", "grounding", "beta"]),
        "effect_shift_by_study": _aggregate_scores(shift, ["delta", "method", "grounding", "study"]),
        "invalid_instrument_by_beta": _aggregate_scores(invalid, ["gamma", "method", "grounding", "beta"]),
        "invalid_instrument_by_study": _aggregate_scores(invalid, ["gamma", "method", "grounding", "study"]),
        "family_mismatch_by_beta": _aggregate_scores(paired, ["grounding", "mismatch", "beta"]).merge(
            paired.groupby(["grounding", "mismatch", "beta"], as_index=False)[["width_change", "hit_change"]].mean()),
        "family_mismatch_by_study": _aggregate_scores(paired, ["grounding", "mismatch", "study"]).merge(
            paired.groupby(["grounding", "mismatch", "study"], as_index=False)[["width_change", "hit_change"]].mean()),
        "grounding_ablation_by_beta": _aggregate_scores(ablation, ["grounding", "beta"]),
        "grounding_ablation_by_study": _aggregate_scores(ablation, ["grounding", "study"]),
    }
    if tolerances:
        tolerance = pd.concat(tolerances, ignore_index=True)
        tolerance["selection_candidate"] = tolerance["covariate"].astype(str).eq(
            tolerance["pattern"].map(instrument_candidate))
        tables["family_mismatch_tolerances"] = tolerance.groupby(
            ["grounding", "mismatch", "selection_candidate", "beta"], as_index=False
        )[["triplets", "binding_2c", "binding_2d"]].mean()
    if realized:
        log = pd.concat(realized, ignore_index=True).drop_duplicates(["arm", "size", "study", "pattern", *covariates])
        group_keys = ["arm", "size", "study", "pattern"]
        out = log.assign(abs_cate_shift=log["cate_shift"].abs()).groupby(group_keys, as_index=False).agg(
            mean_cate_shift=("cate_shift", "mean"), max_abs_cate_shift=("abs_cate_shift", "max"))
        if {"rate_change_arm1", "rate_change_arm0"} <= set(log.columns):
            # The realized direct effect: the gap the flips open between the candidate's levels.
            log["outcome_change"] = (log["rate_change_arm1"] + log["rate_change_arm0"]) / 2.0
            log["candidate_level"] = [int(row[instrument_candidate(row["pattern"])])
                                      for _, row in log.iterrows()]
            levels = log.pivot_table(index=group_keys, columns="candidate_level", values="outcome_change",
                                     aggfunc="mean")
            gap = (levels.get(1) - levels.get(0)).rename("direct_effect").reset_index() \
                if {0, 1} <= set(levels.columns) else None
            if gap is not None:
                out = out.merge(gap, on=group_keys, how="left")
        tables["modified_target_realized"] = out
    summary_dir = arms_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(_parquet_safe(strata), summary_dir / "arms_strata.parquet")
    for name, table in tables.items():
        table.to_csv(summary_dir / f"{name}.csv", index=False)
    family_table = family_mismatch_table_latex(strata)
    if family_table:
        (summary_dir / "family_mismatch_table.tex").write_text(family_table, encoding="utf-8")
    (summary_dir / "missing_leaves.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"summarized {len(strata)} stratum rows; {len(missing)} leaves missing", flush=True)
    plot_arms(summary_dir)
    return {"arms_strata": strata, **tables}


def plot_arms(summary_dir: Path, *, grounding: str = "per_site") -> list[Path]:
    """Draw the mechanism-arm figures from a written ``arms_strata.parquet``.

    The figures go to ``summary_dir/figures`` as PDF and PNG
    (:func:`experiments.exp25_hit_ratio_plots.make_mechanism_arm_plots`), so
    they can be redrawn without summarizing the leaves again.

    Args:
        summary_dir: The ``summary`` directory of an arms root.
        grounding: Naturalness grounding to draw.

    Returns:
        Every path written.
    """
    from experiments.exp25_hit_ratio_plots import make_mechanism_arm_plots

    strata = pd.read_parquet(Path(summary_dir) / "arms_strata.parquet")
    paths = make_mechanism_arm_plots(strata, Path(summary_dir), grounding=grounding)
    print(f"wrote {len(paths)} figure files under {Path(summary_dir) / 'figures'}", flush=True)
    return paths


def run_bounds(tasks: Sequence[tuple[str | Path, float]], output_dir: Path) -> None:
    """Recompute Manski and Balke-Pearl for each ``(store, floor)`` task.

    Balke-Pearl uses as instruments the candidates whose empirical ``D_j(tau)``
    clears Exp25's ``identification_tau`` (by default the smallest tau of the
    Exp16 grid); here it is the smallest tau at or above the floor, so
    Balke-Pearl and the regrounded Naturalness share the candidate threshold.
    The support floor is the one the published bounds used. Manski (which has
    no tau) and floor 0.01 must reproduce the published leaf, which checks the
    replay. Each task writes ``floor=<floor>/<store results root name>/`` with
    ``exp25.run_identification_bounds_replay``; finished leaves are skipped.

    Args:
        tasks: Exp25 backend leaves (``<root>/<stage1>/<trigger>/<lp>``) paired with tau floors.
        output_dir: Root of the bounds pass.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    for store, floor in tasks:
        store = Path(store)
        leaf = output_dir / f"floor={floor:g}" / store.parents[2].name
        if (leaf / exp25.EXP25_INTERVALS_FILENAME).exists():
            continue
        config = load_exp25_store_config(store)
        taus = _exp16_config_for_replication(
            config, seed=int(config.base_seed),
            data_root=store / "replications" / "replication=000" / "osrct",
            output_dir=Path(tempfile.gettempdir()),
        ).taus
        min_count = int(config.identification_min_count)
        published = store / "baseline_variants" / exp25.EXP25_IDENTIFICATION_DIRNAME / "run_manifest.yml"
        if published.exists():
            recorded = yaml.safe_load(published.read_text(encoding="utf-8"))
            min_count = int(recorded["extra"]["identification_config"]["min_count"])
        config = replace(
            config, output_dir=str(store.parents[2]), identification_bounds_only=True,
            identification_tau=min(floor_taus(taus, floor)), identification_min_count=min_count,
            make_plots=False,
        )
        start = time.time()
        exp25.run_identification_bounds_replay(config, output_dir=leaf)
        print(f"bounds floor={floor:g} tau={config.identification_tau:g} {leaf.name}: "
              f"{time.time() - start:.1f}s", flush=True)


def _regrounded_naturalness_rows(
    old_rows: pd.DataFrame,
    regrounding_root: Path,
    floor: float,
) -> pd.DataFrame:
    """Rebuild Exp25 Naturalness interval rows from one floor's posterior summaries.

    Identity and reference columns are taken from the existing rows; the
    endpoints and posterior optimal-draw probability come from the regrounded
    summary, and validity is derived by Exp25's own row builder
    (``exp25.comparison_to_long_intervals``).

    Args:
        old_rows: Published Naturalness rows of one interval table.
        regrounding_root: Root holding ``floor=<floor>/`` regrounding outputs.
        floor: Tau floor whose summaries replace the published endpoints.

    Returns:
        Replacement rows with the index of ``old_rows``.

    Raises:
        FileNotFoundError: If a condition has no regrounded summary.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    strata = list(exp25.EXP25_STRATUM_COLUMNS)
    keys = ["replication", "seed", "bias_covariate", "beta", "study"]
    frames = []
    for (replication, seed, pattern, beta, study), rows in old_rows.groupby(keys, sort=False):
        summary_path = (
            regrounding_root / f"floor={floor:g}" / f"replication={int(replication):03d}"
            / str(study) / str(pattern) / f"beta{format_beta(float(beta))}" / "posterior_summary.parquet"
        )
        summary = pd.read_parquet(summary_path)
        summary = summary[[*strata, "lower_mean", "upper_mean", "lp_posterior_prob_optimal"]].rename(
            columns={"lower_mean": "naturalness_lower", "upper_mean": "naturalness_upper"})
        identity = rows[list(exp25.EXP25_IDENTITY_COLUMNS)].copy()
        for column in strata:
            identity[column] = identity[column].astype(int)
            summary[column] = summary[column].astype(int)
        comparison = identity.merge(summary, on=strata, how="left", validate="one_to_one")
        rebuilt = exp25.comparison_to_long_intervals(
            comparison, replication=int(replication), seed=int(seed), pattern=str(pattern),
            beta=float(beta), config_fingerprint_value=str(rows["config_fingerprint"].iloc[0]),
            methods=("naturalness",),
        )
        rebuilt.index = rows.index
        for column in strata:
            rebuilt[column] = rows[column].to_numpy()
        frames.append(rebuilt[old_rows.columns])
    return pd.concat(frames).loc[old_rows.index]


def _recomputed_identification_tables(merged_dir: Path, floor_dir: Path) -> dict[str, pd.DataFrame]:
    """Return a bounds pass's Manski/Balke-Pearl tables in a merged leaf's row order.

    The per-store leaves under ``floor_dir`` are concatenated, as
    ``exp25.merge_result_roots`` concatenates the stores' leaves, and every row
    of the merged table must be matched by exactly one recomputed row.

    Args:
        merged_dir: The merged leaf's ``baseline_variants/identification_bounds``.
        floor_dir: ``<bounds root>/floor=<floor>`` holding one leaf per store.

    Returns:
        Replacement tables keyed by file name, for each table the merged leaf has.

    Raises:
        FileNotFoundError: If ``floor_dir`` holds no finished leaf.
        ValueError: If the recomputed rows do not match the merged rows one to one.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    keys = {
        exp25.EXP25_INTERVALS_FILENAME: [
            "replication", "bias_covariate", "beta", "study", "target_site",
            *exp25.EXP25_STRATUM_COLUMNS, "method",
        ],
        # The config fingerprint changes with tau, so it cannot match rows.
        exp25.EXP25_DIAGNOSTICS_FILENAME: ["replication", "seed", "bias_covariate", "beta", "study", "method"],
    }
    leaves = sorted(path for path in floor_dir.glob("*") if (path / exp25.EXP25_INTERVALS_FILENAME).exists())
    if not leaves:
        raise FileNotFoundError(f"No finished bounds leaf under {floor_dir}")
    tables: dict[str, pd.DataFrame] = {}
    for name, key in keys.items():
        if not (merged_dir / name).exists():
            continue
        old = pd.read_parquet(merged_dir / name)
        new = pd.concat([pd.read_parquet(leaf / name) for leaf in leaves], ignore_index=True, sort=False)
        if set(new.columns) != set(old.columns) or len(new) != len(old) or new.duplicated(key).any():
            raise ValueError(f"Recomputed {name} under {floor_dir} does not match the merged table")
        order = old[key].merge(new[key].assign(_row=np.arange(len(new))), on=key, how="left", validate="one_to_one")
        if order["_row"].isna().any():
            raise ValueError(f"Recomputed {name} under {floor_dir} misses merged rows")
        tables[name] = new.iloc[order["_row"].to_numpy(dtype=int)][list(old.columns)].set_axis(old.index)
    return tables


def assemble_tree(
    *,
    merged_leaf: Path,
    regrounding_root: Path,
    floor: float,
    destination_leaf: Path,
    bounds_root: Path | None = None,
) -> dict[str, Any]:
    """Write a copy of a merged Exp25 leaf whose Naturalness rows come from one floor.

    Every ``baseline_variants`` table is copied unchanged, except the Naturalness
    rows of each ``mc_intervals.parquet``, rebuilt from the regrounded posterior
    summaries, and their ``mc_run_diagnostics`` rows, recomputed with
    ``exp25.build_run_diagnostics`` (keeping each condition's runtime). With
    ``bounds_root`` the identification-bounds tables are replaced by that pass's
    tables for the same floor (:func:`run_bounds`). Analysis directories are not
    copied; run the analyses (e.g. ``exp25_hit_ratio_analysis``) on the new leaf.

    Args:
        merged_leaf: Source Exp25 backend leaf (read only).
        regrounding_root: Root holding the regrounding outputs.
        floor: Tau floor whose Naturalness intervals are used.
        destination_leaf: New backend leaf; must not exist.
        bounds_root: Optional root of a ``bounds`` pass holding ``floor=<floor>/``.

    Returns:
        A provenance record, also written as ``regrounding_manifest.json``.

    Raises:
        FileExistsError: If ``destination_leaf`` exists.
    """
    import experiments.exp25_osrct_mc_baseline_comparison as exp25

    if destination_leaf.exists():
        raise FileExistsError(f"Refusing to write into an existing tree: {destination_leaf}")
    bounds = None
    if bounds_root is not None:
        # Read and check before anything is written.
        bounds = _recomputed_identification_tables(
            merged_leaf / "baseline_variants" / exp25.EXP25_IDENTIFICATION_DIRNAME,
            bounds_root / f"floor={floor:g}",
        )
    replaced: dict[str, int] = {}
    for source_dir in sorted((merged_leaf / "baseline_variants").iterdir()):
        if not source_dir.is_dir():
            continue
        target_dir = destination_leaf / "baseline_variants" / source_dir.name
        target_dir.mkdir(parents=True)
        if bounds is not None and source_dir.name == exp25.EXP25_IDENTIFICATION_DIRNAME:
            for name, table in bounds.items():
                atomic_write_parquet(table, target_dir / name)
                replaced[f"{source_dir.name}/{name}"] = int(len(table))
        intervals_path = source_dir / "mc_intervals.parquet"
        new_intervals = None
        if intervals_path.exists():
            intervals = pd.read_parquet(intervals_path)
            mask = intervals["method"].astype(str).eq("naturalness")
            if mask.any():
                rebuilt = _regrounded_naturalness_rows(intervals.loc[mask], regrounding_root, floor)
                new_intervals = pd.concat([intervals.loc[~mask], rebuilt]).sort_index()
                atomic_write_parquet(new_intervals, target_dir / "mc_intervals.parquet")
                replaced[f"{source_dir.name}/mc_intervals.parquet"] = int(mask.sum())
        for source_file in sorted(source_dir.iterdir()):
            target_file = target_dir / source_file.name
            if target_file.exists():
                continue
            if source_file.name == "mc_run_diagnostics.parquet" and new_intervals is not None:
                diagnostics = pd.read_parquet(source_file)
                keys = ["config_fingerprint", "replication", "seed", "bias_covariate", "beta", "study", "method"]
                old = diagnostics.loc[diagnostics["method"].astype(str).eq("naturalness")]
                runtimes = old.set_index(keys[:-1])["condition_runtime_seconds"]
                naturalness = new_intervals.loc[new_intervals["method"].astype(str).eq("naturalness")]
                rebuilt = pd.concat([
                    exp25.build_run_diagnostics(rows, runtime_seconds=float(runtimes.loc[key]))
                    for key, rows in naturalness.groupby(keys[:-1], sort=False)
                ], ignore_index=True)
                rebuilt = old[keys].merge(rebuilt, on=keys, how="left", validate="one_to_one")
                rebuilt.index = old.index
                diagnostics = pd.concat([diagnostics.loc[~diagnostics.index.isin(old.index)], rebuilt[diagnostics.columns]]).sort_index()
                atomic_write_parquet(diagnostics, target_file)
                replaced[f"{source_dir.name}/mc_run_diagnostics.parquet"] = int(len(old))
            else:
                shutil.copy2(source_file, target_file)
    if (merged_leaf / "merge_manifest.json").exists():
        shutil.copy2(merged_leaf / "merge_manifest.json", destination_leaf / "merge_manifest.json")
    record = {
        "source_merged_leaf": str(merged_leaf.resolve()),
        "regrounding_root": str(regrounding_root.resolve()),
        "bounds_root": None if bounds_root is None else str(bounds_root.resolve()),
        "tau_floor": float(floor),
        "naturalness": "intervals regrounded on the tau grid at or above tau_floor; discovery (Stage 1-2) is the merged tree's",
        "identification_bounds": (
            "copied unchanged (Balke-Pearl at the merged tree's tau)" if bounds_root is None else
            "Manski and Balke-Pearl recomputed with Balke-Pearl's D_j(tau) threshold at the "
            "smallest tau at or above tau_floor (bounds_root)"
        ),
        "competitors": "every other method row and the Gamma calibration copied unchanged",
        "rows_replaced": replaced,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (destination_leaf / "regrounding_manifest.json").write_text(
        json.dumps(record, indent=2, default=json_default) + "\n", encoding="utf-8")
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("tables", "reground", "ablate", "summarize", "summarize-arms", "plot-arms",
                                           "bounds", "assemble"), required=True)
    parser.add_argument("--stores", nargs="+", default=None, help="Exp25 backend leaves (read only)")
    parser.add_argument("--output-dir", required=True, help="regrounding result root (bounds: bounds-pass root)")
    parser.add_argument("--merged-leaf", default=None, help="assemble: merged Exp25 backend leaf to copy")
    parser.add_argument("--floor", type=float, default=None, help="assemble: tau floor to use")
    parser.add_argument("--destination", default=None, help="assemble: new backend leaf (must not exist)")
    parser.add_argument("--bounds-root", default=None,
                        help="assemble: bounds-pass root whose Manski/Balke-Pearl rows replace the merged leaf's")
    parser.add_argument("--betas", type=float, nargs="+", default=list(PAPER_BETAS))
    parser.add_argument("--replications", type=int, nargs="+", default=None)
    parser.add_argument("--floors", type=float, nargs="+", default=list(DEFAULT_FLOORS))
    parser.add_argument("--validate-replications", type=int, nargs="*", default=[0],
                        help="replications that also run floor 0.01, which must reproduce Exp25")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--tabicl-device", default="cuda")
    parser.add_argument("--lp-posterior-n-jobs", type=int, default=4)
    parser.add_argument("--wait-for-tables", type=float, default=0.0,
                        help="seconds a reground shard waits for each missing table")
    parser.add_argument("--tables-dir", default=None,
                        help="reground: tables-pass directory to read (default OUTPUT_DIR/tables)")
    parser.add_argument("--gp-training-target", choices=SUPPORTED_GP_TRAINING_TARGETS,
                        default=GP_TRAINING_TARGET_PER_SITE,
                        help="reground: fit the GP to every triggering site (Exp25) or to the per-grid-point envelope")
    parser.add_argument("--gp-target-scale", choices=SUPPORTED_GP_TARGET_SCALES, default=GP_TARGET_SCALE_NOTE,
                        help="reground: logit of c (Exp25) or of c~ = 2c")
    parser.add_argument("--target-pairs", choices=TARGET_PAIRS, default="all",
                        help="reground: every triggered (epsilon, tau) (Exp25) or the admissible pairs only")
    parser.add_argument("--effect-shift", type=float, default=None,
                        help="tables/reground: shift every target stratum CATE by this amount (effect-shift arm)")
    parser.add_argument("--direct-effect", type=float, default=None,
                        help="tables/reground: give the selection covariate's binary version a direct effect on Y "
                             "of this size at the target (invalid-instrument arm)")
    parser.add_argument("--identities-root", default=None,
                        help="reground with --effect-shift or --direct-effect: floor directory of the published "
                             "identification bounds")
    parser.add_argument("--source-mismatch", choices=SOURCE_MISMATCHES, default="none",
                        help="reground: ground each target on another condition's sources (family-mismatch arm)")
    parser.add_argument("--published-root", default=None,
                        help="summarize-arms: regrounding root of the published per-site intervals")
    parser.add_argument("--ablation-two-c", type=float, nargs="+", default=list(ABLATION_TWO_C),
                        help="ablate: fixed 2c values (solver scale)")
    parser.add_argument("--ablation-two-d", type=float, nargs="+", default=list(ABLATION_TWO_D),
                        help="ablate: fixed 2d values (solver scale)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one pass of the regrounding sensitivity analysis."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.mode == "assemble":
        if not (args.merged_leaf and args.floor is not None and args.destination):
            raise SystemExit("assemble needs --merged-leaf, --floor and --destination")
        record = assemble_tree(merged_leaf=Path(args.merged_leaf), regrounding_root=output_dir,
                               floor=args.floor, destination_leaf=Path(args.destination),
                               bounds_root=None if args.bounds_root is None else Path(args.bounds_root))
        print(json.dumps(record["rows_replaced"], indent=2))
        return 0
    if args.mode == "plot-arms":
        plot_arms(output_dir / "summary")
        return 0
    if not args.stores:
        raise SystemExit(f"--mode {args.mode} needs --stores")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "bounds":
        tasks = [(store, float(floor)) for store in args.stores for floor in args.floors]
        (output_dir / f"manifest_bounds_shard{args.shard}.json").write_text(
            json.dumps({"argv": sys.argv[1:] if argv is None else list(argv), "n_tasks": len(tasks)},
                       indent=2, default=json_default), encoding="utf-8")
        run_bounds(_shard(tasks, args.shard, args.n_shards), output_dir)
        return 0
    conditions = enumerate_conditions(args.stores, betas=args.betas, replications=args.replications)
    if args.mode == "summarize":
        summarize(conditions, output_dir, floors=args.floors)
        return 0
    if args.mode == "summarize-arms":
        if not (args.published_root and args.identities_root):
            raise SystemExit("summarize-arms needs --published-root and --identities-root")
        summarize_arms(conditions, output_dir, published_root=Path(args.published_root),
                       identities_root=Path(args.identities_root), floor=max(args.floors))
        return 0
    manifest = output_dir / f"manifest_{args.mode}_shard{args.shard}.json"
    manifest.write_text(json.dumps({"argv": sys.argv[1:] if argv is None else list(argv), "n_conditions": len(conditions)},
                                   indent=2, default=json_default), encoding="utf-8")
    mine = _shard(conditions, args.shard, args.n_shards)
    if args.mode == "ablate":
        if not args.tables_dir:
            raise SystemExit("ablate needs --tables-dir")
        run_ablation(mine, output_dir, tables_dir=Path(args.tables_dir), floor=max(args.floors),
                     lp_posterior_n_jobs=args.lp_posterior_n_jobs,
                     two_c_grid=args.ablation_two_c, two_d_grid=args.ablation_two_d)
        return 0
    if args.mode == "tables":
        run_tables(mine, output_dir, tabicl_device=args.tabicl_device,
                   effect_shift=0.0 if args.effect_shift is None else args.effect_shift,
                   direct_effect=0.0 if args.direct_effect is None else args.direct_effect)
    else:
        lookup = None
        if args.source_mismatch != "none":
            every = enumerate_conditions(args.stores, betas=PAPER_BETAS, replications=args.replications)
            lookup = {(c.replication, c.study, c.pattern, round(float(c.beta), 6)): c for c in every}
        run_reground(mine, output_dir, floors=args.floors,
                     validate_replications=args.validate_replications,
                     lp_posterior_n_jobs=args.lp_posterior_n_jobs,
                     wait_for_tables=args.wait_for_tables,
                     tables_dir=None if args.tables_dir is None else Path(args.tables_dir),
                     grounding={"gp_training_target": args.gp_training_target,
                                "gp_target_scale": args.gp_target_scale,
                                "target_pairs": args.target_pairs},
                     effect_shift=args.effect_shift,
                     identities_root=None if args.identities_root is None else Path(args.identities_root),
                     source_mismatch=args.source_mismatch, lookup=lookup, direct_effect=args.direct_effect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
