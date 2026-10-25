"""Cut-Bayesian uncertainty propagation for learned Naturalness bounds.

The module implements three separated stages:

1. posterior draws of the observed source and target PMFs, either conjugate
   Dirichlet draws of the raw cells (:func:`draw_raw_pmfs`) or a stratified
   Bayesian bootstrap of TabICLv2 fits (:func:`draw_tabicl_bootstrap_pmfs`);
2. conditional posterior learning of the solver-scale ``2c``/``2d`` tolerance
   surfaces (:func:`fit_tolerance_surfaces`); and
3. deterministic propagation of paired PMF/tolerance draws through the
   learned-Naturalness LP (:func:`draw_random_identified_sets`).

Tolerance evidence never feeds back into the PMF posterior, and the target LP
defines no posterior over unidentified causal parameters.  The output is a
posterior over random identified sets under a modular (cut) analysis; its
envelopes (:func:`posterior_outer_envelopes`) are not confidence intervals or
credible intervals for the causal effect.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial import Delaunay, QhullError
from scipy.special import logsumexp, ndtr, ndtri

from .loaders.schema import OUTCOME_COLUMN, TREATMENT_COLUMN
from .lp_solver_learned_naturalness import (
    TargetObservedBinaryQuantities,
    estimate_target_observed_binary_quantities,
    solve_target_bounds_from_observed_binary_quantities,
    solve_target_weighted_cate_bounds_from_observed_binary_quantities,
)
from .naturalness_statistics import as_tuple, safe_sorted
from .seeding import sha256_seed


LOGGER = logging.getLogger(__name__)


DEFAULT_COVARIATES = ("z_clean", "z_bad")
from .paper_grids import EPSILON_GRID as DEFAULT_EPSILONS
from .paper_grids import TAU_GRID as DEFAULT_TAUS
NUMERICAL_PROBABILITY_FLOOR = 1e-6
# Nugget nu of the Gaussian-copula latent correlation, R = (1 - nu) K + nu I.
# R keeps a unit diagonal, so every tolerance margin stays Uniform(0, 1), and its
# smallest eigenvalue is at least nu; without it, source sites with near-identical
# trigger statistics make R numerically singular and NUTS freezes or diverges.
GP_NUGGET = 5e-2
# Fits between JAX compilation-cache clears: bounds memory growth without
# recompiling after every fit (see ``_clear_jax_caches_periodically``).
JAX_CACHE_CLEAR_INTERVAL = 20
# Location of the latent GP field.  ``uniform`` (the paper's GP-copula prior) has
# zero mean, so every tolerance margin is Uniform(0, 1); ``learned`` adds a common
# mean mu ~ Normal(0, LEARNED_LOCATION_PRIOR_SD), so the tolerance margin
# Phi(N(mu, 1)) is located by the sites' evidence.
MARGINAL_LOCATIONS = ("uniform", "learned")
LEARNED_LOCATION_PRIOR_SD = 1.5
# Candidate-specific surfaces: tolerance kind, solver-scale evidence column and
# GP inputs.  ``2c`` depends on the exclusion statistic only; ``2d`` also on
# treatment relevance.
_SURFACE_SPECS = (
    ("c", "required_two_c", ("i_deviation",)),
    ("d", "required_two_d", ("i_deviation", "d_strength")),
)


#: The SHA-256 seed scheme of this pipeline; see :mod:`causal_grounding.seeding`.
_stable_seed = sha256_seed


def _stable_sample_id(*parts: Any) -> str:
    """Return a short deterministic identifier for one posterior draw."""

    payload = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _validate_binary(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that the named columns contain only binary nonmissing values."""

    for column in columns:
        if column not in frame:
            raise ValueError(f"Missing required column {column!r}")
        values = set(pd.to_numeric(frame[column], errors="raise").dropna().astype(int))
        if not values <= {0, 1}:
            raise ValueError(f"Column {column!r} is not binary: {sorted(values)}")


@dataclass(frozen=True)
class SitePMFDraw:
    """One posterior draw of the observed probability tables for one site.

    Attributes:
        sample_id: Stable joint-imputation identifier shared across sites.
        site_id: Source or target environment label.
        covariates: Ordered binary variables defining ``Z``.
        z_values: Supported target/source strata retained in the draw.
        idle_xy_given_z: ``P(X=x,Y=y | Z=z,F=idle)`` keyed by ``(z,x,y)``.
        do_y_given_xz: ``P(Y=y | do(X=x),Z=z)`` keyed by ``(z,x,y)``.
            This mapping is empty for target-only idle draws.
        idle_z: ``P(Z=z | F=idle)``.
        n_idle_z: Observed idle counts by stratum.
        n_idle_xz: Observed idle counts by treatment and stratum.
        overlap_model: Whether the on/idle tables were derived from a single
            posterior over ``(X,Y,idle_indicator)`` for overlapping rows.
        alpha: Symmetric Dirichlet concentration used for each raw cell.
    """

    sample_id: str
    site_id: str
    covariates: tuple[str, ...]
    z_values: tuple[tuple[int, ...], ...]
    idle_xy_given_z: Mapping[tuple[tuple[int, ...], int, int], float]
    do_y_given_xz: Mapping[tuple[tuple[int, ...], int, int], float]
    idle_z: Mapping[tuple[int, ...], float]
    n_idle_z: Mapping[tuple[int, ...], int]
    n_idle_xz: Mapping[tuple[int, tuple[int, ...]], int]
    overlap_model: bool
    alpha: float

    def p_idle_x_given_z(self, x: int, z: Sequence[int]) -> float:
        """Return the drawn idle ``P(X=x | Z=z)``."""

        key = as_tuple(z)
        return float(sum(self.idle_xy_given_z[(key, int(x), y)] for y in (0, 1)))

    def p_idle_y_given_xz(self, y: int, x: int, z: Sequence[int]) -> float:
        """Return the drawn idle ``P(Y=y | X=x,Z=z)``."""

        key = as_tuple(z)
        numerator = float(self.idle_xy_given_z[(key, int(x), int(y))])
        denominator = self.p_idle_x_given_z(int(x), key)
        if denominator <= 0.0:
            raise ValueError(f"Zero idle treatment mass at site={self.site_id}, x={x}, z={key}")
        return numerator / denominator

    def p_do_y_given_xz(self, y: int, x: int, z: Sequence[int]) -> float:
        """Return the drawn randomized ``P(Y=y | do(X=x),Z=z)``."""

        key = (as_tuple(z), int(x), int(y))
        if key not in self.do_y_given_xz:
            raise ValueError(f"Site {self.site_id!r} has no randomized PMF for {key}")
        return float(self.do_y_given_xz[key])

    def p_star_y_given_xz(
        self,
        star: int,
        y: int,
        x: int,
        z: Sequence[int],
    ) -> tuple[float, float]:
        """Recover ``P(Y(star)=y | X=x,Z=z,idle)`` and its unclipped value.

        Consistency supplies the factual cell.  The off-treatment cell follows
        from the randomized marginal and idle treatment mixture.  Independent
        finite-sample posterior tables need not satisfy this decomposition
        exactly, so the operational probability is clipped and the raw value
        is returned for diagnostics.
        """

        star = int(star)
        x = int(x)
        if x == star:
            value = self.p_idle_y_given_xz(y, x, z)
            return value, value
        p_do = self.p_do_y_given_xz(y, star, z)
        p_factual = self.p_idle_y_given_xz(y, star, z)
        p_star = self.p_idle_x_given_z(star, z)
        p_x = self.p_idle_x_given_z(x, z)
        raw = (p_do - p_factual * p_star) / max(p_x, NUMERICAL_PROBABILITY_FLOOR)
        return float(np.clip(raw, 0.0, 1.0)), float(raw)

    def observed_quantities(self, min_cell_count: int = 1) -> TargetObservedBinaryQuantities:
        """Convert a target idle draw into the existing binary-LP interface.

        A stratum is retained only when each observed treatment arm has at
        least ``min_cell_count`` physical rows.  Dirichlet prior mass therefore
        cannot manufacture empirical support for the LP.
        """

        supported = [
            z
            for z in self.z_values
            if self.n_idle_z.get(z, 0) >= min_cell_count
            and all(self.n_idle_xz.get((x, z), 0) >= min_cell_count for x in (0, 1))
        ]
        p_x1 = {z: self.p_idle_x_given_z(1, z) for z in supported}
        return TargetObservedBinaryQuantities(
            covariates=list(self.covariates),
            z_values=list(supported),
            p_x1=p_x1,
            p_x0={z: 1.0 - p_x1[z] for z in supported},
            p_y1_given_xz={
                (x, z): self.p_idle_y_given_xz(1, x, z)
                for z in supported
                for x in (0, 1)
            },
            n_z={z: int(self.n_idle_z[z]) for z in supported},
            n_xz={(x, z): int(self.n_idle_xz[(x, z)]) for z in supported for x in (0, 1)},
        )


RawPMFImputation = dict[str, SitePMFDraw]


def _categorical_counts(
    frame: pd.DataFrame,
    columns: Sequence[str],
    cells: Sequence[tuple[int, ...]],
) -> np.ndarray:
    """Count a fixed categorical support in canonical order."""

    if frame.empty:
        return np.zeros(len(cells), dtype=float)
    grouped = frame.groupby(list(columns), observed=True).size()
    return np.asarray([float(grouped.get(cell, 0)) for cell in cells], dtype=float)


def _sample_dirichlet(counts: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a categorical probability vector under a symmetric Dirichlet prior."""

    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and strictly positive")
    return np.asarray(rng.dirichlet(np.asarray(counts, dtype=float) + float(alpha)), dtype=float)


def _site_frames(
    data: pd.DataFrame,
    *,
    site: str,
    site_col: str,
    regime_col: str,
    on_value: str,
    idle_value: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return randomized and idle rows for one site."""

    site_data = data.loc[data[site_col].astype(str).eq(site)].copy()
    if regime_col not in site_data:
        raise ValueError(f"Missing regime column {regime_col!r}")
    on = site_data.loc[site_data[regime_col].eq(on_value)].copy()
    idle = site_data.loc[site_data[regime_col].eq(idle_value)].copy()
    if idle.empty:
        raise ValueError(f"Site {site!r} has no idle rows")
    return on, idle


def _validate_overlap(
    on: pd.DataFrame,
    idle: pd.DataFrame,
    *,
    id_col: str,
    columns: Sequence[str],
) -> bool:
    """Detect and validate the archive's nested on/idle physical rows."""

    if on.empty:
        return False
    if on[id_col].duplicated().any() or idle[id_col].duplicated().any():
        raise ValueError("Physical row IDs must be unique within each regime")
    overlap_ids = set(on[id_col]) & set(idle[id_col])
    if not overlap_ids:
        return False
    if overlap_ids != set(idle[id_col]):
        raise ValueError("Partially overlapping regimes are unsupported; idle must be nested or disjoint")
    left = on.loc[on[id_col].isin(overlap_ids), [id_col, *columns]].sort_values(id_col)
    right = idle[[id_col, *columns]].sort_values(id_col)
    merged = left.merge(right, on=id_col, suffixes=("_on", "_idle"), validate="one_to_one")
    for column in columns:
        if not merged[f"{column}_on"].reset_index(drop=True).equals(
            merged[f"{column}_idle"].reset_index(drop=True)
        ):
            raise ValueError(f"Overlapping physical rows disagree on {column!r}")
    return True


def _idle_support(
    idle: pd.DataFrame,
    covariates: Sequence[str],
    treatment: str,
) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, ...], int], dict[tuple[int, tuple[int, ...]], int]]:
    """Return a site's observed idle strata and their physical row counts.

    Args:
        idle: The site's idle rows.
        covariates: Ordered binary ``Z`` columns.
        treatment: Binary treatment column.

    Returns:
        The sorted observed strata, ``n_idle_z`` by stratum and ``n_idle_xz``
        by ``(x, z)``.  Every PMF draw of the site, whatever produced it, keeps
        this support: prior mass or a smoother never adds a stratum.
    """

    z_values = tuple(
        safe_sorted(
            {
                tuple(int(value) for value in row)
                for row in idle[list(covariates)].itertuples(index=False, name=None)
            }
        )
    )
    n_idle_z: dict[tuple[int, ...], int] = {}
    n_idle_xz: dict[tuple[int, tuple[int, ...]], int] = {}
    for z in z_values:
        mask = np.logical_and.reduce([idle[column].eq(value) for column, value in zip(covariates, z)])
        frame = idle.loc[mask]
        n_idle_z[z] = int(len(frame))
        for x in (0, 1):
            n_idle_xz[(x, z)] = int(frame[treatment].eq(x).sum())
    return z_values, n_idle_z, n_idle_xz


def _draw_site_pmf(
    on: pd.DataFrame,
    idle: pd.DataFrame,
    *,
    sample_id: str,
    site_id: str,
    covariates: Sequence[str],
    treatment: str,
    outcome: str,
    id_col: str,
    alpha: float,
    rng: np.random.Generator,
) -> SitePMFDraw:
    """Draw one site's PMFs, respecting nested archive rows when present."""

    overlap = _validate_overlap(
        on,
        idle,
        id_col=id_col,
        columns=[treatment, outcome, *covariates],
    )
    z_values, n_idle_z, n_idle_xz = _idle_support(idle, covariates, treatment)
    if not z_values:
        raise ValueError(f"Site {site_id!r} has no observed idle strata")

    idle_xy: dict[tuple[tuple[int, ...], int, int], float] = {}
    do_y: dict[tuple[tuple[int, ...], int, int], float] = {}
    xy_cells = tuple((x, y) for x in (0, 1) for y in (0, 1))
    xys_cells = tuple((x, y, selected) for x in (0, 1) for y in (0, 1) for selected in (0, 1))

    if overlap:
        selected_ids = set(idle[id_col])
        physical = on.copy()
        physical["__in_idle__"] = physical[id_col].isin(selected_ids).astype(int)

    for z in z_values:
        idle_mask = np.logical_and.reduce([idle[column].eq(value) for column, value in zip(covariates, z)])
        idle_z_frame = idle.loc[idle_mask]

        if overlap:
            physical_mask = np.logical_and.reduce(
                [physical[column].eq(value) for column, value in zip(covariates, z)]
            )
            z_frame = physical.loc[physical_mask]
            probabilities = _sample_dirichlet(
                _categorical_counts(z_frame, [treatment, outcome, "__in_idle__"], xys_cells),
                alpha,
                rng,
            ).reshape(2, 2, 2)
            selected_mass = float(probabilities[:, :, 1].sum())
            for x, y in xy_cells:
                idle_xy[(z, x, y)] = float(probabilities[x, y, 1] / selected_mass)
            if not on.empty:
                for x in (0, 1):
                    arm = probabilities[x, :, :].sum(axis=1)
                    arm = arm / arm.sum()
                    for y in (0, 1):
                        do_y[(z, x, y)] = float(arm[y])
        else:
            idle_probabilities = _sample_dirichlet(
                _categorical_counts(idle_z_frame, [treatment, outcome], xy_cells),
                alpha,
                rng,
            ).reshape(2, 2)
            for x, y in xy_cells:
                idle_xy[(z, x, y)] = float(idle_probabilities[x, y])
            if not on.empty:
                on_mask = np.logical_and.reduce([on[column].eq(value) for column, value in zip(covariates, z)])
                on_z_frame = on.loc[on_mask]
                on_probabilities = _sample_dirichlet(
                    _categorical_counts(on_z_frame, [treatment, outcome], xy_cells),
                    alpha,
                    rng,
                ).reshape(2, 2)
                for x in (0, 1):
                    arm = on_probabilities[x, :] / on_probabilities[x, :].sum()
                    for y in (0, 1):
                        do_y[(z, x, y)] = float(arm[y])

    z_counts = np.asarray([n_idle_z[z] for z in z_values], dtype=float)
    z_probabilities = _sample_dirichlet(z_counts, alpha, rng)
    return SitePMFDraw(
        sample_id=sample_id,
        site_id=site_id,
        covariates=tuple(covariates),
        z_values=z_values,
        idle_xy_given_z=idle_xy,
        do_y_given_xz=do_y,
        idle_z={z: float(probability) for z, probability in zip(z_values, z_probabilities)},
        n_idle_z=n_idle_z,
        n_idle_xz=n_idle_xz,
        overlap_model=overlap,
        alpha=float(alpha),
    )


def draw_raw_pmfs(
    data: pd.DataFrame,
    *,
    n_draws: int,
    alpha: float = 1.0,
    random_seed: int = 20260922,
    treatment: str = TREATMENT_COLUMN,
    outcome: str = OUTCOME_COLUMN,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    site_col: str = "site",
    regime_col: str = "F",
    on_value: str = "on",
    idle_value: str = "idle",
    id_col: str = "id",
) -> list[RawPMFImputation]:
    """Draw joint raw-PMF imputations without downstream feedback.

    Args:
        data: Learner-facing rows containing source on/idle regimes or target
            idle rows.  Multiple sites may be supplied.
        n_draws: Number of joint imputations.  Draw index ``s`` contains one
            independently drawn PMF for every site.
        alpha: Symmetric Dirichlet concentration per categorical cell.
        random_seed: Root seed used to derive stable site/draw seeds.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        covariates: Ordered binary ``Z`` columns.
        site_col: Environment identifier.
        regime_col: Column distinguishing ``on`` and ``idle`` rows.
        on_value: Randomized regime label.
        idle_value: Idle regime label.
        id_col: Physical-row identifier used to detect archive overlap.

    Returns:
        A list of joint imputations.  Each item maps site labels to
        :class:`SitePMFDraw` objects and shares one stable ``sample_id``.
    """

    if n_draws < 1:
        raise ValueError("n_draws must be at least one")
    required = {site_col, regime_col, id_col, treatment, outcome, *covariates}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Raw-PMF data lacks columns {sorted(missing)}")
    _validate_binary(data, [treatment, outcome, *covariates])
    sites = safe_sorted(data[site_col].astype(str).unique())
    frames = {
        site: _site_frames(
            data,
            site=site,
            site_col=site_col,
            regime_col=regime_col,
            on_value=on_value,
            idle_value=idle_value,
        )
        for site in sites
    }
    imputations: list[RawPMFImputation] = []
    for draw_index in range(int(n_draws)):
        sample_id = _stable_sample_id("raw-pmf", random_seed, alpha, draw_index)
        imputation: RawPMFImputation = {}
        for site in sites:
            on, idle = frames[site]
            rng = np.random.default_rng(_stable_seed(random_seed, alpha, draw_index, site))
            imputation[site] = _draw_site_pmf(
                on,
                idle,
                sample_id=sample_id,
                site_id=site,
                covariates=covariates,
                treatment=treatment,
                outcome=outcome,
                id_col=id_col,
                alpha=alpha,
                rng=rng,
            )
        imputations.append(imputation)
    return imputations


def _resample_within_groups(frame: pd.DataFrame, columns: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Return row positions drawn with replacement within every group of ``columns``.

    Each group keeps its size, and groups are visited in sorted order, so a seed
    fixes the resample.
    """

    if frame.empty:
        return np.zeros(0, dtype=int)
    codes = frame.groupby(list(columns), sort=True, observed=True).ngroup().to_numpy()
    return np.concatenate([
        rng.choice(members, size=len(members), replace=True)
        for members in (np.flatnonzero(codes == code) for code in range(int(codes.max()) + 1))
    ])


def _bootstrap_site_rows(
    on: pd.DataFrame,
    idle: pd.DataFrame,
    *,
    covariates: Sequence[str],
    id_col: str,
    overlap: bool,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample one site's rows with replacement, stratum by stratum.

    When the idle rows are nested in the randomized rows, the physical units are
    resampled within each ``Z`` stratum, separately for units in and out of the
    idle regime, and the idle resample is the selected part; the nesting and
    every stratum's idle and randomized counts are preserved.  Otherwise each
    regime is resampled within its own strata.

    Args:
        on: The site's randomized rows (empty for a target).
        idle: The site's idle rows.
        covariates: Ordered binary ``Z`` columns.
        id_col: Physical-row identifier linking nested regimes.
        overlap: Whether the idle rows are nested in the randomized rows.
        rng: The draw's random stream.

    Returns:
        The resampled randomized and idle rows.
    """

    if overlap:
        physical = on.reset_index(drop=True).copy()
        physical["__in_idle__"] = physical[id_col].isin(set(idle[id_col])).astype(int)
        picked = physical.iloc[_resample_within_groups(physical, [*covariates, "__in_idle__"], rng)]
        idle_by_id = idle.drop_duplicates(id_col).set_index(id_col, drop=False)
        selected_ids = picked.loc[picked["__in_idle__"].eq(1), id_col].to_numpy()
        return (
            picked.drop(columns="__in_idle__").reset_index(drop=True),
            idle_by_id.loc[selected_ids].reset_index(drop=True),
        )
    idle = idle.reset_index(drop=True)
    resampled_idle = idle.iloc[_resample_within_groups(idle, covariates, rng)].reset_index(drop=True)
    if on.empty:
        return on, resampled_idle
    on = on.reset_index(drop=True)
    return on.iloc[_resample_within_groups(on, covariates, rng)].reset_index(drop=True), resampled_idle


def _tabicl_site_probabilities(
    on: pd.DataFrame,
    idle: pd.DataFrame,
    *,
    site_id: str,
    z_values: Sequence[tuple[int, ...]],
    covariates: Sequence[str],
    treatment: str,
    outcome: str,
    regime_col: str,
    on_value: str,
    idle_value: str,
    predictor_kwargs: Mapping[str, Any],
    probability_backend: str = "tabicl_v2",
) -> tuple[dict, dict, dict]:
    """Fit the TabICLv2 Stage-1 models to one site's rows and read them on ``z_values``.

    ``probability_backend`` names the registered backend
    (:mod:`causal_grounding.probability_backends`) whose Stage-1 adapter and
    target-LP path are fitted. The default is TabICLv2.

    A source site uses the grounder's
    :class:`~causal_grounding.naturalness_grounder.TabICLv2SitePMFModels`; a
    target, which has idle rows only, uses the TabICLv2 path of
    :func:`~causal_grounding.lp_solver_learned_naturalness.estimate_target_observed_binary_quantities`.
    Both fit ``P(X=1 | Z)`` and ``P(Y=1 | X, Z)`` on the idle rows, and a source
    also ``P(Y=1 | X, Z)`` on its randomized rows.

    Returns:
        ``P(X=1 | z)`` by ``z``; idle ``P(Y=1 | X=x, z)`` by ``(x, z)``; and
        randomized ``P(Y=1 | do(X=x), z)`` by ``(x, z)``, empty for a target.
    """

    covariates = list(covariates)
    if on.empty:
        quantities = estimate_target_observed_binary_quantities(
            idle, treatment, outcome, covariates, regime_col=None, min_count=1,
            probability_backend=probability_backend, probability_model_kwargs=dict(predictor_kwargs),
        )
        return (
            {z: float(quantities.p_x1[z]) for z in z_values},
            {(x, z): float(quantities.p_y1_given_xz[(x, z)]) for z in z_values for x in (0, 1)},
            {},
        )
    # Resolved here, through the backend registry, so the CPU sampling stages
    # never load the predictor stack.
    from .probability_backends import get_probability_backend

    site_pmf_models = get_probability_backend(probability_backend).stage1_factory

    model = site_pmf_models(
        site_id=site_id,
        data=pd.concat([on.assign(**{regime_col: on_value}), idle.assign(**{regime_col: idle_value})], ignore_index=True),
        covariates=covariates,
        treatment=treatment,
        outcome=outcome,
        regime_col=regime_col,
        idle_value=idle_value,
        on_value=on_value,
        predictor_kwargs=dict(predictor_kwargs),
    )
    return (
        {z: float(model.p_idle_x_given_z(1, z)) for z in z_values},
        {(x, z): float(model.p_idle_y_given_xz(1, x, z)) for z in z_values for x in (0, 1)},
        {(x, z): float(model.p_do_y_given_z(x, 1, z)) for z in z_values for x in (0, 1)},
    )


def draw_tabicl_bootstrap_pmfs(
    data: pd.DataFrame,
    *,
    n_draws: int,
    random_seed: int,
    predictor_kwargs: Mapping[str, Any] | None = None,
    treatment: str = TREATMENT_COLUMN,
    outcome: str = OUTCOME_COLUMN,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    site_col: str = "site",
    regime_col: str = "F",
    on_value: str = "on",
    idle_value: str = "idle",
    id_col: str = "id",
    resample: bool = True,
) -> list[RawPMFImputation]:
    """Draw site PMFs by a stratified Bayesian bootstrap of TabICLv2 fits.

    The TabICLv2 alternative to :func:`draw_raw_pmfs`.  Draw ``s`` of a site
    resamples its rows with replacement within every ``Z`` stratum
    (:func:`_bootstrap_site_rows`), refits the TabICLv2 Stage-1 models on the
    resample (:func:`_tabicl_site_probabilities`) and reads them on the site's
    observed idle strata; ``P(Z | idle)`` is drawn from ``Dirichlet(n_z)``.
    Stratifying is exact because Bayesian-bootstrap stratum masses and
    within-stratum weights are independent, and the resample approximates those
    weights for a learner that takes none.  Each draw thus carries the sampling
    uncertainty of TabICLv2's smoothed estimate, around it rather than around
    raw cell frequencies.

    The support is the site's observed idle strata with physical counts
    ``n_idle_z``/``n_idle_xz``, so TabICLv2 never adds LP support.
    ``P(X=1 | z)`` is kept at least ``NUMERICAL_PROBABILITY_FLOOR`` from 0 and 1,
    so ``P(Y | X, z)`` stays defined, and exact, when a resample has one
    treatment class.  ``alpha`` is ``NaN`` (no Dirichlet prior).  With
    ``resample=False`` the single draw is the plug-in fit to every site's full
    rows, with ``P(Z | idle)`` the observed idle share.

    Args:
        data: Rows of one or more sites, as for :func:`draw_raw_pmfs`.
        n_draws: Number of joint imputations.
        random_seed: Root seed; each site and draw has its own stream.
        predictor_kwargs: Keyword arguments of the TabICLv2 predictor, such as
            ``device``, ``random_state``, ``kv_cache`` and ``share_checkpoint``.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        covariates: Ordered binary ``Z`` columns.
        site_col: Environment identifier.
        regime_col: Column distinguishing ``on`` and ``idle`` rows.
        on_value: Randomized regime label.
        idle_value: Idle regime label.
        id_col: Physical-row identifier used to detect nesting.
        resample: Whether to bootstrap; ``False`` returns the one plug-in fit
            and needs ``n_draws == 1``.

    Returns:
        A list of joint imputations mapping each site to its
        :class:`SitePMFDraw`; draw ``s`` shares one ``sample_id`` across sites.
    """

    if n_draws < 1:
        raise ValueError("n_draws must be at least one")
    if not resample and n_draws != 1:
        raise ValueError("A plug-in fit without resampling is a single draw")
    required = {site_col, regime_col, id_col, treatment, outcome, *covariates}
    missing = required - set(data)
    if missing:
        raise ValueError(f"PMF data lacks columns {sorted(missing)}")
    _validate_binary(data, [treatment, outcome, *covariates])
    sites = safe_sorted(data[site_col].astype(str).unique())
    frames = {
        site: _site_frames(data, site=site, site_col=site_col, regime_col=regime_col, on_value=on_value,
                           idle_value=idle_value)
        for site in sites
    }
    overlaps = {
        site: _validate_overlap(on, idle, id_col=id_col, columns=[treatment, outcome, *covariates])
        for site, (on, idle) in frames.items()
    }
    supports = {site: _idle_support(idle, covariates, treatment) for site, (_, idle) in frames.items()}
    kwargs = dict(predictor_kwargs or {})
    floor = NUMERICAL_PROBABILITY_FLOOR
    imputations: list[RawPMFImputation] = []
    for draw_index in range(int(n_draws)):
        sample_id = _stable_sample_id("tabicl-bootstrap", random_seed, draw_index)
        imputation: RawPMFImputation = {}
        for site in sites:
            on, idle = frames[site]
            z_values, n_idle_z, n_idle_xz = supports[site]
            rng = np.random.default_rng(_stable_seed("tabicl-bootstrap", random_seed, draw_index, site))
            boot_on, boot_idle = _bootstrap_site_rows(on, idle, covariates=covariates, id_col=id_col,
                                                      overlap=overlaps[site], rng=rng) if resample else (on, idle)
            p_x1, p_y1, p_do = _tabicl_site_probabilities(
                boot_on, boot_idle, site_id=site, z_values=z_values, covariates=covariates, treatment=treatment,
                outcome=outcome, regime_col=regime_col, on_value=on_value, idle_value=idle_value,
                predictor_kwargs=kwargs,
            )
            idle_xy: dict[tuple[tuple[int, ...], int, int], float] = {}
            for z in z_values:
                treated = float(np.clip(p_x1[z], floor, 1.0 - floor))
                for x in (0, 1):
                    p_x = treated if x else 1.0 - treated
                    for y in (0, 1):
                        idle_xy[(z, x, y)] = p_x * (p_y1[(x, z)] if y else 1.0 - p_y1[(x, z)])
            do_y = {
                (z, x, y): p_do[(x, z)] if y else 1.0 - p_do[(x, z)]
                for z in z_values for x in (0, 1) for y in (0, 1)
            } if p_do else {}
            counts = np.asarray([n_idle_z[z] for z in z_values], dtype=float)
            masses = rng.dirichlet(counts) if resample else counts / counts.sum()
            imputation[site] = SitePMFDraw(
                sample_id=sample_id,
                site_id=site,
                covariates=tuple(covariates),
                z_values=z_values,
                idle_xy_given_z=idle_xy,
                do_y_given_xz=do_y,
                idle_z={z: float(mass) for z, mass in zip(z_values, masses)},
                n_idle_z=n_idle_z,
                n_idle_xz=n_idle_xz,
                overlap_model=overlaps[site],
                alpha=float("nan"),
            )
        imputations.append(imputation)
        if (draw_index + 1) % 10 == 0 or draw_index + 1 == n_draws:
            LOGGER.info("TabICLv2 bootstrap: %d of %d draws over %d sites", draw_index + 1, n_draws, len(sites))
    return imputations


def raw_pmf_draws_to_frame(imputations: Sequence[RawPMFImputation]) -> pd.DataFrame:
    """Serialize raw PMF draws into a long, audit-friendly table."""

    rows: list[dict[str, Any]] = []
    for imputation_index, imputation in enumerate(imputations):
        for draw in imputation.values():
            for z in draw.z_values:
                common = {
                    "imputation_index": int(imputation_index),
                    "sample_id": draw.sample_id,
                    "site": draw.site_id,
                    "alpha": draw.alpha,
                    "overlap_model": draw.overlap_model,
                    **{column: int(value) for column, value in zip(draw.covariates, z)},
                }
                rows.append({**common, "table": "idle_z", "x": pd.NA, "y": pd.NA, "probability": draw.idle_z[z]})
                for x in (0, 1):
                    for y in (0, 1):
                        rows.append({
                            **common,
                            "table": "idle_xy_given_z",
                            "x": x,
                            "y": y,
                            "probability": draw.idle_xy_given_z[(z, x, y)],
                        })
                        if (z, x, y) in draw.do_y_given_xz:
                            rows.append({
                                **common,
                                "table": "do_y_given_xz",
                                "x": x,
                                "y": y,
                                "probability": draw.do_y_given_xz[(z, x, y)],
                            })
    return pd.DataFrame(rows)


def raw_pmf_draws_from_frame(
    frame: pd.DataFrame,
    data: pd.DataFrame,
    *,
    treatment: str = TREATMENT_COLUMN,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    site_col: str = "site",
    regime_col: str = "F",
    idle_value: str = "idle",
) -> list[RawPMFImputation]:
    """Rebuild the imputations that :func:`raw_pmf_draws_to_frame` serialized.

    The probabilities come from ``frame``.  The physical support counts are
    recomputed from ``data``, the rows the draws were made from, with
    :func:`_idle_support`, and every site's strata must equal the frame's, so a
    table cannot be paired with rows it was not drawn from.

    Args:
        frame: A table written by :func:`raw_pmf_draws_to_frame`.
        data: The rows the draws were made from.
        treatment: Binary treatment column.
        covariates: Ordered binary ``Z`` columns.
        site_col: Environment identifier.
        regime_col: Column distinguishing ``on`` and ``idle`` rows.
        idle_value: Idle regime label.

    Returns:
        The imputations, in ``imputation_index`` order.

    Raises:
        ValueError: If a site's strata in ``frame`` differ from its observed
            idle strata in ``data``.
    """

    covariates = list(covariates)
    idle_rows = data.loc[data[regime_col].eq(idle_value)]
    sites = {str(site) for site in frame["site"].unique()}
    supports = {
        site: _idle_support(idle_rows.loc[idle_rows[site_col].astype(str).eq(site)], covariates, treatment)
        for site in sites
    }
    imputations: list[RawPMFImputation] = []
    for _, block in frame.groupby("imputation_index", sort=True):
        imputation: RawPMFImputation = {}
        for site, rows in block.groupby("site", sort=False):
            site = str(site)
            z_values, n_idle_z, n_idle_xz = supports[site]
            idle_z: dict[tuple[int, ...], float] = {}
            idle_xy: dict[tuple[tuple[int, ...], int, int], float] = {}
            do_y: dict[tuple[tuple[int, ...], int, int], float] = {}
            for row in rows.itertuples(index=False):
                z = tuple(int(getattr(row, column)) for column in covariates)
                if row.table == "idle_z":
                    idle_z[z] = float(row.probability)
                elif row.table == "idle_xy_given_z":
                    idle_xy[(z, int(row.x), int(row.y))] = float(row.probability)
                else:
                    do_y[(z, int(row.x), int(row.y))] = float(row.probability)
            if set(idle_z) != set(z_values):
                raise ValueError(f"Stored PMF strata of site {site!r} differ from its observed idle strata")
            first = rows.iloc[0]
            imputation[site] = SitePMFDraw(
                sample_id=str(first["sample_id"]),
                site_id=site,
                covariates=tuple(covariates),
                z_values=z_values,
                idle_xy_given_z=idle_xy,
                do_y_given_xz=do_y,
                idle_z={z: idle_z[z] for z in z_values},
                n_idle_z=n_idle_z,
                n_idle_xz=n_idle_xz,
                overlap_model=bool(first["overlap_model"]),
                alpha=float(first["alpha"]),
            )
        imputations.append(imputation)
    return imputations


def raw_pmf_posterior_parameters(
    data: pd.DataFrame,
    *,
    n_draws: int,
    alpha: float,
    random_seed: int,
    treatment: str = TREATMENT_COLUMN,
    outcome: str = OUTCOME_COLUMN,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    site_col: str = "site",
    regime_col: str = "F",
    on_value: str = "on",
    idle_value: str = "idle",
    id_col: str = "id",
) -> pd.DataFrame:
    """Return the Dirichlet parameters that :func:`draw_raw_pmfs` samples.

    The table is a compact, lossless representation of the draws: they depend on
    the data only through these cell counts and on the seed only through each
    site's own stream, so :func:`draw_raw_pmfs` applied to
    :func:`rows_from_raw_pmf_posterior_parameters` of this table, with the
    recorded ``n_draws``, ``alpha`` and ``random_seed``, reproduces every draw
    and sample ID exactly.

    Blocks mirror the sampler: a site whose idle rows are nested in its
    randomized rows has one ``xy_in_idle_given_z`` table per idle stratum over
    ``(x, y, in_idle)``; otherwise each idle stratum has an ``idle_xy_given_z``
    table and, when randomized rows exist, an ``on_xy_given_z`` table.  Every
    site ends with one ``idle_z`` table over its observed idle strata (strata
    without idle rows are structural zeros with no cell).

    Args:
        data: The rows passed to :func:`draw_raw_pmfs`.
        n_draws: Number of joint imputations drawn from these parameters.
        alpha: Symmetric Dirichlet concentration per cell.
        random_seed: Root seed passed to :func:`draw_raw_pmfs`.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        covariates: Ordered binary ``Z`` columns.
        site_col: Environment identifier.
        regime_col: Column distinguishing ``on`` and ``idle`` rows.
        on_value: Randomized regime label.
        idle_value: Idle regime label.
        id_col: Physical-row identifier used to detect nesting.

    Returns:
        One row per Dirichlet cell: ``site``, ``block``, ``block_order`` (that
        call's position in the site's random stream), the stratum, ``x``, ``y``
        and ``in_idle`` where they apply, ``count``, ``concentration`` and the
        constant columns needed to regenerate the draws.
    """

    _validate_binary(data, [treatment, outcome, *covariates])
    covariates = list(covariates)
    xy_cells = tuple((x, y) for x in (0, 1) for y in (0, 1))
    xys_cells = tuple((x, y, selected) for x in (0, 1) for y in (0, 1) for selected in (0, 1))
    rows: list[dict[str, Any]] = []
    for site in safe_sorted(data[site_col].astype(str).unique()):
        on, idle = _site_frames(
            data,
            site=site,
            site_col=site_col,
            regime_col=regime_col,
            on_value=on_value,
            idle_value=idle_value,
        )
        overlap = _validate_overlap(on, idle, id_col=id_col, columns=[treatment, outcome, *covariates])
        z_values = tuple(
            safe_sorted({tuple(int(value) for value in row) for row in idle[covariates].itertuples(index=False, name=None)})
        )
        if overlap:
            physical = on.copy()
            physical["__in_idle__"] = physical[id_col].isin(set(idle[id_col])).astype(int)
        # One entry per Dirichlet call, in the order the site's stream makes them.
        blocks: list[tuple[str, tuple[int, ...], Sequence[tuple[int, ...]], np.ndarray, tuple[str, ...]]] = []
        for z in z_values:
            if overlap:
                mask = np.logical_and.reduce([physical[column].eq(value) for column, value in zip(covariates, z)])
                counts = _categorical_counts(physical.loc[mask], [treatment, outcome, "__in_idle__"], xys_cells)
                blocks.append(("xy_in_idle_given_z", z, xys_cells, counts, ("x", "y", "in_idle")))
                continue
            mask = np.logical_and.reduce([idle[column].eq(value) for column, value in zip(covariates, z)])
            counts = _categorical_counts(idle.loc[mask], [treatment, outcome], xy_cells)
            blocks.append(("idle_xy_given_z", z, xy_cells, counts, ("x", "y")))
            if not on.empty:
                on_mask = np.logical_and.reduce([on[column].eq(value) for column, value in zip(covariates, z)])
                counts = _categorical_counts(on.loc[on_mask], [treatment, outcome], xy_cells)
                blocks.append(("on_xy_given_z", z, xy_cells, counts, ("x", "y")))
        idle_z_order = len(blocks)
        for order, (block, z, cells, counts, names) in enumerate(blocks):
            for cell, count in zip(cells, counts):
                rows.append({
                    "site": site,
                    "block": block,
                    "block_order": order,
                    **{column: int(value) for column, value in zip(covariates, z)},
                    "x": pd.NA,
                    "y": pd.NA,
                    "in_idle": pd.NA,
                    **{name: int(value) for name, value in zip(names, cell)},
                    "count": int(count),
                    "concentration": float(count) + float(alpha),
                    "overlap_model": bool(overlap),
                })
        for z in z_values:
            count = int(np.logical_and.reduce([idle[column].eq(value) for column, value in zip(covariates, z)]).sum())
            rows.append({
                "site": site,
                "block": "idle_z",
                "block_order": idle_z_order,
                **{column: int(value) for column, value in zip(covariates, z)},
                "x": pd.NA,
                "y": pd.NA,
                "in_idle": pd.NA,
                "count": count,
                "concentration": float(count) + float(alpha),
                "overlap_model": bool(overlap),
            })
    frame = pd.DataFrame(rows)
    for column in ("x", "y", "in_idle"):
        frame[column] = frame[column].astype("Int8")
    frame["alpha"] = float(alpha)
    frame["random_seed"] = int(random_seed)
    frame["n_draws"] = int(n_draws)
    frame["site_stream_seed_rule"] = "_stable_seed(random_seed, alpha, draw_index, site)"
    return frame


def rows_from_raw_pmf_posterior_parameters(
    parameters: pd.DataFrame,
    *,
    treatment: str = TREATMENT_COLUMN,
    outcome: str = OUTCOME_COLUMN,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """Rebuild rows whose raw-PMF posterior equals a stored parameter table.

    This inverts :func:`raw_pmf_posterior_parameters` up to rows that never
    reach the sampler (randomized rows in strata without idle rows), which is
    all that is needed to regenerate the draws.  Nested sites get one
    randomized row per ``xy_in_idle_given_z`` count and an idle twin with the
    same ID for every ``in_idle = 1`` count.

    Args:
        parameters: A table written by :func:`raw_pmf_posterior_parameters`.
        treatment: Binary treatment column to create.
        outcome: Binary outcome column to create.
        covariates: Ordered binary ``Z`` columns to create.

    Returns:
        Rows with ``id``, ``site``, ``F``, the treatment, the outcome and the
        covariates, ready for :func:`draw_raw_pmfs`.
    """

    rows: list[dict[str, Any]] = []
    covariates = list(covariates)
    cells = parameters.loc[parameters.block.ne("idle_z") & parameters["count"].gt(0)]
    for cell in cells.itertuples(index=False):
        base = {
            "site": str(cell.site),
            treatment: int(cell.x),
            outcome: int(cell.y),
            **{column: int(getattr(cell, column)) for column in covariates},
        }
        prefix = f"{cell.site}|{cell.block}|{'|'.join(str(int(getattr(cell, column))) for column in covariates)}|{int(cell.x)}{int(cell.y)}"
        for repeat in range(int(cell.count)):
            if cell.block == "xy_in_idle_given_z":
                row_id = f"{prefix}|{int(cell.in_idle)}|{repeat}"
                rows.append({"id": row_id, "F": "on", **base})
                if int(cell.in_idle) == 1:
                    rows.append({"id": row_id, "F": "idle", **base})
            else:
                regime = "idle" if cell.block == "idle_xy_given_z" else "on"
                rows.append({"id": f"{prefix}|{repeat}", "F": regime, **base})
    return pd.DataFrame(rows, columns=["id", "site", "F", treatment, outcome, *covariates])


def _candidate_pairs(
    z_values: Sequence[tuple[int, ...]],
    candidate_index: int,
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Yield unordered supported stratum pairs differing only in a candidate."""

    groups: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for z in z_values:
        z_minus = tuple(value for index, value in enumerate(z) if index != candidate_index)
        groups.setdefault(z_minus, []).append(z)
    for group in groups.values():
        yield from combinations(safe_sorted(group), 2)


def compute_trigger_statistics(draw: SitePMFDraw) -> pd.DataFrame:
    """Compute exact-draw ``I`` and ``D`` statistics once per candidate."""

    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(draw.covariates):
        i_differences: list[float] = []
        d_differences: list[float] = []
        pairs = list(_candidate_pairs(draw.z_values, candidate_index))
        for z, z_prime in pairs:
            for x in (0, 1):
                d_differences.append(abs(draw.p_idle_x_given_z(x, z) - draw.p_idle_x_given_z(x, z_prime)))
                for y in (0, 1):
                    i_differences.append(
                        abs(draw.p_idle_y_given_xz(y, x, z) - draw.p_idle_y_given_xz(y, x, z_prime))
                    )
        rows.append({
            "sample_id": draw.sample_id,
            "site": draw.site_id,
            "candidate": candidate,
            "candidate_index": candidate_index,
            "i_deviation": max(i_differences, default=np.nan),
            "d_strength": min(d_differences, default=np.nan),
            "n_comparison_pairs": len(pairs),
        })
    return pd.DataFrame(rows)


def compute_source_evidence(
    draw: SitePMFDraw,
    pair_filter: Callable[[tuple[int, ...], tuple[int, ...]], bool] | None = None,
) -> pd.DataFrame:
    """Compute one tolerance evidence unit per source-site candidate.

    ``required_two_c`` and ``required_two_d`` are on the solver scale: pass them
    directly as the LP's ``two_c`` and ``two_d`` right-hand sides, with no extra
    factor of two.

    ``pair_filter``, given two strata that differ only in the candidate, selects
    the stratum pairs the requirements range over (the selected subset of
    ``Z_{-j}`` in Stage 2 of the grounding algorithm); ``None`` keeps every
    supported pair, and a candidate without a selected pair gets ``NaN``
    requirements.  The trigger statistics ``I`` and ``D`` always use every pair,
    as the target's do.
    """

    if not draw.do_y_given_xz:
        raise ValueError(f"Source evidence requires randomized PMFs for site {draw.site_id!r}")
    trigger = compute_trigger_statistics(draw).set_index("candidate")
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(draw.covariates):
        c_differences: list[float] = []
        d_differences: list[float] = []
        raw_violations: list[float] = []
        pairs = [pair for pair in _candidate_pairs(draw.z_values, candidate_index)
                 if pair_filter is None or pair_filter(*pair)]
        for z, z_prime in pairs:
            for star in (0, 1):
                for y in (0, 1):
                    d_differences.append(
                        abs(draw.p_do_y_given_xz(y, star, z) - draw.p_do_y_given_xz(y, star, z_prime))
                    )
                    for x in (0, 1):
                        left, raw_left = draw.p_star_y_given_xz(star, y, x, z)
                        right, raw_right = draw.p_star_y_given_xz(star, y, x, z_prime)
                        c_differences.append(abs(left - right))
                        raw_violations.extend((max(0.0, -raw_left, raw_left - 1.0), max(0.0, -raw_right, raw_right - 1.0)))
        stat = trigger.loc[candidate]
        rows.append({
            "sample_id": draw.sample_id,
            "site": draw.site_id,
            "candidate": candidate,
            "candidate_index": candidate_index,
            "i_deviation": float(stat.i_deviation),
            "d_strength": float(stat.d_strength),
            "required_two_c": max(c_differences, default=np.nan),
            "required_two_d": max(d_differences, default=np.nan),
            "n_comparison_pairs": len(pairs),
            "cross_pmf_max_clip": max(raw_violations, default=0.0),
            "cross_pmf_was_clipped": bool(max(raw_violations, default=0.0) > 0.0),
        })
    return pd.DataFrame(rows)


@dataclass
class ToleranceSurfacePosterior:
    """Posterior draws and support metadata for one candidate/tolerance surface."""

    candidate: str
    tolerance_kind: str
    feature_names: tuple[str, ...]
    source_features: np.ndarray
    source_omega: np.ndarray
    posterior_samples: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    imputation_id: str

    def save(self, path: str | Path) -> None:
        """Persist posterior arrays and JSON metadata in one compressed archive."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "candidate": self.candidate,
            "tolerance_kind": self.tolerance_kind,
            "feature_names": list(self.feature_names),
            "diagnostics": self.diagnostics,
            "imputation_id": self.imputation_id,
        }
        arrays = {f"posterior__{name}": value for name, value in self.posterior_samples.items()}
        np.savez_compressed(
            output,
            source_features=self.source_features,
            source_omega=self.source_omega,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ToleranceSurfacePosterior":
        """Load a posterior written by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as bundle:
            metadata = json.loads(str(bundle["metadata_json"]))
            posterior = {
                name.removeprefix("posterior__"): np.asarray(bundle[name])
                for name in bundle.files
                if name.startswith("posterior__")
            }
            return cls(
                candidate=metadata["candidate"],
                tolerance_kind=metadata["tolerance_kind"],
                feature_names=tuple(metadata["feature_names"]),
                source_features=np.asarray(bundle["source_features"], dtype=float),
                source_omega=np.asarray(bundle["source_omega"], dtype=float),
                posterior_samples=posterior,
                diagnostics=dict(metadata["diagnostics"]),
                imputation_id=metadata["imputation_id"],
            )


def _squared_exponential(
    left: np.ndarray,
    right: np.ndarray,
    length_scale: np.ndarray,
) -> np.ndarray:
    """Evaluate the unit-variance squared-exponential correlation kernel."""

    scaled = (left[:, None, :] - right[None, :, :]) / length_scale[None, None, :]
    return np.exp(-0.5 * np.sum(scaled * scaled, axis=-1))


def _surface_diagnostics(
    samples_by_chain: Mapping[str, np.ndarray],
    diverging: np.ndarray,
    source_omega: np.ndarray,
) -> dict[str, Any]:
    """Summarize convergence, tail mixing, divergences, and hard support."""

    try:
        from numpyro.diagnostics import effective_sample_size, summary
    except ImportError as error:  # pragma: no cover - guarded by fitter import.
        raise RuntimeError("NumPyro is required to diagnose tolerance surfaces") from error

    report = summary(dict(samples_by_chain), prob=0.90, group_by_chain=True)
    r_hats: list[float] = []
    bulk_ess: list[float] = []
    tail_ess: list[float] = []
    for name, values in samples_by_chain.items():
        if name not in report:
            continue
        r_hats.extend(np.asarray(report[name]["r_hat"], dtype=float).ravel().tolist())
        bulk_ess.extend(np.asarray(report[name]["n_eff"], dtype=float).ravel().tolist())
        array = np.asarray(values, dtype=float)
        flat_components = array.reshape(array.shape[0], array.shape[1], -1)
        for component in range(flat_components.shape[-1]):
            component_values = flat_components[:, :, component]
            quantiles = np.quantile(component_values, [0.05, 0.95])
            tail_indicator = (
                (component_values <= quantiles[0]) | (component_values >= quantiles[1])
            ).astype(float)
            tail_ess.append(float(np.asarray(effective_sample_size(tail_indicator))))
    source = np.asarray(samples_by_chain["source_tolerance"], dtype=float)
    support_violations = int(np.sum(source < source_omega.reshape((1, 1, -1)) - 1e-10))
    max_rhat = max(r_hats, default=np.inf)
    min_bulk = min(bulk_ess, default=0.0)
    min_tail = min(tail_ess, default=0.0)
    n_divergent = int(np.asarray(diverging, dtype=bool).sum())
    passed = bool(
        np.isfinite(max_rhat)
        and max_rhat <= 1.01
        and min_bulk >= 400.0
        and min_tail >= 400.0
        and n_divergent == 0
        and support_violations == 0
    )
    return {
        "max_r_hat": float(max_rhat),
        "min_bulk_ess": float(min_bulk),
        "min_tail_ess": float(min_tail),
        "n_divergent": n_divergent,
        "support_violations": support_violations,
        "passed": passed,
        "tail_ess_definition": "ESS of pooled 5-percent tail indicators",
    }


class _PeriodicJaxCacheClearer:
    """Count tolerance-surface fits and clear JAX's caches once per interval.

    Attributes:
        fits_since_clear: Fits since the last clear.
    """

    def __init__(self) -> None:
        """Start a fresh count."""
        self.fits_since_clear = 0

    def reset(self) -> None:
        """Restart the count without clearing JAX's caches."""
        self.fits_since_clear = 0

    def record_fit(self, jax: Any) -> None:
        """Count one fit, and clear the caches when the count reaches the interval.

        Args:
            jax: The imported ``jax`` module.
        """
        self.fits_since_clear += 1
        if self.fits_since_clear >= JAX_CACHE_CLEAR_INTERVAL:
            jax.clear_caches()
            self.fits_since_clear = 0
            _malloc_trim()


def _malloc_trim() -> None:
    """Return freed heap to the operating system; a no-op outside glibc."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        LOGGER.debug("malloc_trim is unavailable; freed heap stays with the process")


_JAX_CACHE_CLEARER = _PeriodicJaxCacheClearer()


def _clear_jax_caches_periodically(jax: Any) -> None:
    """Drop JAX's compilation caches every ``JAX_CACHE_CLEAR_INTERVAL`` fits.

    Each tolerance-surface fit closes over its own data and compiles a program
    that is never reused, and JAX retains every such executable, so memory grows
    with the number of fits; clearing after every fit would also discard JAX's
    reusable programs and recompile them.  Clearing never changes numerical
    results.  The call also runs ``malloc_trim`` (a no-op outside glibc), which
    returns freed heap that glibc keeps in XLA's per-thread arenas.

    Args:
        jax: The imported ``jax`` module (the fitter imports JAX lazily).
    """

    _JAX_CACHE_CLEARER.record_fit(jax)


def _fit_one_surface_numpyro(
    *,
    candidate: str,
    tolerance_kind: str,
    features: np.ndarray,
    omega: np.ndarray,
    imputation_id: str,
    random_seed: int,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    target_accept_prob: float,
    progress_bar: bool,
    dense_mass: bool = False,
    marginal_location: str = "uniform",
) -> ToleranceSurfacePosterior:
    """Fit one exact-support probit-GP tolerance posterior with NUTS.

    ``dense_mass`` adapts a full rather than diagonal mass matrix during warm-up;
    the posterior is unchanged, but sampling improves when near-identical source
    trigger statistics make the latents strongly correlated.

    ``marginal_location`` is one of :data:`MARGINAL_LOCATIONS`.  Under
    ``learned`` the latent field has a common mean
    ``mu ~ Normal(0, LEARNED_LOCATION_PRIOR_SD)``, sampled jointly with the
    field and kept in the posterior samples for prediction.
    """

    if marginal_location not in MARGINAL_LOCATIONS:
        raise ValueError(f"marginal_location must be one of {MARGINAL_LOCATIONS}")

    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as error:
        raise RuntimeError(
            "Bayesian tolerance fitting requires NumPyro and JAX. "
            "Install the pinned cluster requirements before running Exp30."
        ) from error

    # JAX's default float32 cannot resolve the latent correlation matrix when
    # source statistics nearly coincide, and NUTS step-size adaptation collapses.
    numpyro.enable_x64()
    x = np.asarray(features, dtype=float)
    observed = np.asarray(omega, dtype=float)
    clipped = np.clip(observed, NUMERICAL_PROBABILITY_FLOOR, 1.0 - NUMERICAL_PROBABILITY_FLOOR)
    x_jax = jnp.asarray(x)
    omega_jax = jnp.asarray(clipped)
    n_observations, n_features = x.shape

    def model() -> None:
        """NumPyro model for a hard-truncated Gaussian-copula field."""

        length_scale = numpyro.sample(
            "length_scale",
            dist.LogNormal(math.log(0.25), 0.75).expand([n_features]).to_event(1),
        )
        eta = numpyro.sample("eta", dist.LogNormal(0.0, 1.0))
        location = (numpyro.sample("mu", dist.Normal(0.0, LEARNED_LOCATION_PRIOR_SD))
                    if marginal_location == "learned" else 0.0)
        raw_u = numpyro.sample(
            "raw_u",
            dist.Normal(0.0, 1.0).expand([n_observations]).to_event(1),
        )
        lower_u = jsp.special.ndtri(omega_jax)
        u = lower_u + jax.nn.softplus(raw_u)
        scaled = (x_jax[:, None, :] - x_jax[None, :, :]) / length_scale
        covariance = jnp.exp(-0.5 * jnp.sum(scaled * scaled, axis=-1))
        covariance = (1.0 - GP_NUGGET) * covariance + GP_NUGGET * jnp.eye(n_observations)
        desired_prior = dist.MultivariateNormal(
            loc=location * jnp.ones(n_observations), covariance_matrix=covariance
        ).log_prob(u)
        proposal_prior = dist.Normal(0.0, 1.0).log_prob(raw_u).sum()
        log_jacobian = jax.nn.log_sigmoid(raw_u).sum()
        numpyro.factor("truncated_gp_prior", desired_prior + log_jacobian - proposal_prior)

        tolerance = jsp.special.ndtr(u)
        denominator_argument = eta * tolerance
        log_denominator = denominator_argument + jnp.log(-jnp.expm1(-denominator_argument))
        log_likelihood = jnp.log(eta) + eta * omega_jax - log_denominator
        numpyro.factor("reverse_exponential_likelihood", log_likelihood.sum())
        numpyro.deterministic("u", u)
        numpyro.deterministic("source_tolerance", tolerance)

    kernel = NUTS(model, target_accept_prob=float(target_accept_prob), dense_mass=bool(dense_mass))
    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        chain_method="parallel",
        progress_bar=bool(progress_bar),
    )
    mcmc.run(jax.random.PRNGKey(int(random_seed)))
    samples_by_chain = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    del mcmc, kernel
    _clear_jax_caches_periodically(jax)
    diagnostics = _surface_diagnostics(samples_by_chain, diverging, clipped)
    diagnostics.update({
        "num_chains": int(num_chains),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "target_accept_prob": float(target_accept_prob),
        "dense_mass": bool(dense_mass),
        "n_evidence_units": int(n_observations),
        "omega_clipped_count": int(np.sum(observed != clipped)),
    })
    flattened = {
        name: value.reshape((-1, *value.shape[2:]))
        for name, value in samples_by_chain.items()
        if name in {"length_scale", "eta", "mu", "u", "source_tolerance"}
    }
    feature_names = ("i_deviation",) if tolerance_kind == "c" else ("i_deviation", "d_strength")
    return ToleranceSurfacePosterior(
        candidate=candidate,
        tolerance_kind=tolerance_kind,
        feature_names=feature_names,
        source_features=x,
        source_omega=clipped,
        posterior_samples=flattened,
        diagnostics=diagnostics,
        imputation_id=imputation_id,
    )


def fit_tolerance_surfaces(
    evidence: pd.DataFrame,
    *,
    random_seed: int = 20260922,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    target_accept_prob: float = 0.90,
    retry_warmup: int = 2000,
    retry_target_accept_prob: float = 0.95,
    retry_num_samples: int | None = None,
    progress_bar: bool = False,
    dense_mass: bool = False,
    second_retry: Mapping[str, Any] | None = None,
    on_unconverged: str = "raise",
    unconverged: list[dict[str, Any]] | None = None,
    marginal_location: str = "uniform",
) -> dict[tuple[str, str], ToleranceSurfacePosterior]:
    """Fit all candidate-specific ``c`` and ``d`` surfaces for one imputation.

    A surface whose NUTS diagnostics fail is refitted with ``retry_warmup``,
    ``retry_target_accept_prob`` and ``retry_num_samples`` retained draws per
    chain (``num_samples`` when ``None``), then, if ``second_retry`` (a mapping
    with ``num_warmup``, ``num_samples`` and ``target_accept_prob``) is given,
    with those settings.  Every attempt uses the same diagnostic gate,
    ``dense_mass`` and ``marginal_location`` (see
    :func:`_fit_one_surface_numpyro`).

    A surface that fails its last attempt is never replaced by a plug-in
    surface: ``on_unconverged="raise"`` (the default) raises, and ``"exclude"``
    leaves it out of the returned mapping, so the candidate cannot be grounded
    from this imputation, and appends its final diagnostics to ``unconverged``
    when a list is given.

    Raises:
        RuntimeError: If a surface fails its last attempt under ``"raise"``.
        ValueError: For an unknown ``on_unconverged`` or too little evidence.
    """

    if on_unconverged not in {"raise", "exclude"}:
        raise ValueError("on_unconverged must be 'raise' or 'exclude'")

    required = {
        "sample_id",
        "site",
        "candidate",
        "i_deviation",
        "d_strength",
        "required_two_c",
        "required_two_d",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"Tolerance evidence lacks columns {sorted(missing)}")
    if evidence.empty:
        raise ValueError("Tolerance evidence is empty")
    sample_ids = evidence.sample_id.astype(str).unique()
    if len(sample_ids) != 1:
        raise ValueError("fit_tolerance_surfaces expects exactly one raw-PMF imputation")
    imputation_id = str(sample_ids[0])
    fitted: dict[tuple[str, str], ToleranceSurfacePosterior] = {}
    for candidate, candidate_rows in evidence.groupby("candidate", observed=True, sort=True):
        for kind, omega_column, feature_names in _SURFACE_SPECS:
            rows = candidate_rows.dropna(subset=[*feature_names, omega_column]).copy()
            if len(rows) < 2:
                raise ValueError(f"Candidate {candidate!r} has fewer than two {kind} evidence units")
            features = rows[list(feature_names)].to_numpy(dtype=float)
            omega = rows[omega_column].to_numpy(dtype=float)
            seed = _stable_seed(random_seed, imputation_id, candidate, kind, "primary")
            posterior = _fit_one_surface_numpyro(
                candidate=str(candidate),
                tolerance_kind=kind,
                features=features,
                omega=omega,
                imputation_id=imputation_id,
                random_seed=seed,
                num_chains=num_chains,
                num_warmup=num_warmup,
                num_samples=num_samples,
                target_accept_prob=target_accept_prob,
                progress_bar=progress_bar,
                dense_mass=dense_mass,
                marginal_location=marginal_location,
            )
            posterior.diagnostics["attempt"] = "primary"
            if not posterior.diagnostics["passed"]:
                posterior = _fit_one_surface_numpyro(
                    candidate=str(candidate),
                    tolerance_kind=kind,
                    features=features,
                    omega=omega,
                    imputation_id=imputation_id,
                    random_seed=_stable_seed(random_seed, imputation_id, candidate, kind, "retry"),
                    num_chains=num_chains,
                    num_warmup=retry_warmup,
                    num_samples=num_samples if retry_num_samples is None else retry_num_samples,
                    target_accept_prob=retry_target_accept_prob,
                    progress_bar=progress_bar,
                    dense_mass=dense_mass,
                    marginal_location=marginal_location,
                )
                posterior.diagnostics["attempt"] = "retry"
            if not posterior.diagnostics["passed"] and second_retry is not None:
                posterior = _fit_one_surface_numpyro(
                    candidate=str(candidate),
                    tolerance_kind=kind,
                    features=features,
                    omega=omega,
                    imputation_id=imputation_id,
                    random_seed=_stable_seed(random_seed, imputation_id, candidate, kind, "second_retry"),
                    num_chains=num_chains,
                    num_warmup=int(second_retry["num_warmup"]),
                    num_samples=int(second_retry["num_samples"]),
                    target_accept_prob=float(second_retry["target_accept_prob"]),
                    progress_bar=progress_bar,
                    dense_mass=dense_mass,
                    marginal_location=marginal_location,
                )
                posterior.diagnostics["attempt"] = "second_retry"
            if not posterior.diagnostics["passed"]:
                if on_unconverged == "raise":
                    raise RuntimeError(
                        f"NUTS diagnostics failed after retry for candidate={candidate!r}, kind={kind!r}: "
                        f"{posterior.diagnostics}"
                    )
                if unconverged is not None:
                    unconverged.append({
                        "candidate": str(candidate),
                        "tolerance_kind": kind,
                        "imputation_id": imputation_id,
                        **posterior.diagnostics,
                    })
                continue
            fitted[(str(candidate), kind)] = posterior
    return fitted


def is_within_source_support(
    source_features: np.ndarray,
    query: Sequence[float],
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Check whether a query lies in the affine convex hull of source inputs."""

    source = np.asarray(source_features, dtype=float)
    point = np.asarray(query, dtype=float)
    if source.ndim != 2 or point.shape != (source.shape[1],):
        raise ValueError("Source/query feature dimensions do not agree")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(source)):
        return False
    centered = source - source.mean(axis=0)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        return bool(np.linalg.norm(point - source[0]) <= tolerance)
    basis = right[:rank].T
    residual = point - source.mean(axis=0) - basis @ (basis.T @ (point - source.mean(axis=0)))
    if np.linalg.norm(residual) > tolerance:
        return False
    projected_source = centered @ basis
    projected_point = (point - source.mean(axis=0)) @ basis
    if rank == 1:
        return bool(
            projected_source[:, 0].min() - tolerance
            <= projected_point[0]
            <= projected_source[:, 0].max() + tolerance
        )
    try:
        hull = Delaunay(projected_source)
        return bool(hull.find_simplex(projected_point) >= 0)
    except QhullError:
        return bool(
            np.all(projected_point >= projected_source.min(axis=0) - tolerance)
            and np.all(projected_point <= projected_source.max(axis=0) + tolerance)
        )


def _conditional_latent_moments(
    posterior: ToleranceSurfacePosterior,
    query: np.ndarray,
    index: int,
) -> tuple[float, float]:
    """Return the latent GP mean and variance at a query for one posterior draw.

    Conditions the unit-variance probit field on retained draw ``index`` of the
    source latents and length scales, under the fit's nugget correlation
    ``(1 - GP_NUGGET) K + GP_NUGGET I``; the query is a new site, so its nugget
    component is independent of every source site.  The field's mean is that
    draw's ``mu`` under a learned location, otherwise zero.

    Args:
        posterior: Fitted tolerance surface.
        query: Feature vector in the surface's input coordinates.
        index: Retained posterior draw to condition on.

    Returns:
        The conditional latent mean and variance.
    """

    length_scale = np.asarray(posterior.posterior_samples["length_scale"][index], dtype=float)
    source_u = np.asarray(posterior.posterior_samples["u"][index], dtype=float)
    k_ss = _squared_exponential(posterior.source_features, posterior.source_features, length_scale)
    k_ss = (1.0 - GP_NUGGET) * k_ss + GP_NUGGET * np.eye(len(k_ss))
    k_qs = (1.0 - GP_NUGGET) * _squared_exponential(query[None, :], posterior.source_features, length_scale)[0]
    location = float(posterior.posterior_samples["mu"][index]) if "mu" in posterior.posterior_samples else 0.0
    factor = linalg.cho_factor(k_ss, lower=True, check_finite=False)
    solved_u = linalg.cho_solve(factor, source_u - location, check_finite=False)
    solved_k = linalg.cho_solve(factor, k_qs, check_finite=False)
    mean = float(location + k_qs @ solved_u)
    variance = max(0.0, float(1.0 - k_qs @ solved_k))
    return mean, variance


def sample_tolerance_surface(
    posterior: ToleranceSurfacePosterior,
    query_features: Sequence[float],
    *,
    random_seed: int,
    posterior_index: int | None = None,
    allow_extrapolation: bool = False,
    return_index: bool = False,
) -> float | tuple[float, int]:
    """Draw one target tolerance from a fitted conditional GP posterior.

    The latent at the query is drawn on the probit scale, conditioned on one
    retained draw (``posterior_index`` modulo the draw count, else a random one)
    by :func:`_conditional_latent_moments`.  A query outside the source convex
    hull returns ``NaN`` unless ``allow_extrapolation``, under which distance
    from the sources shows up as predictive variance growing toward the prior's.
    With ``return_index`` the result is ``(tolerance, index)``, with index ``-1``
    when the query is refused; the random stream is the same either way.
    """

    query = np.asarray(query_features, dtype=float)
    if not allow_extrapolation and not is_within_source_support(posterior.source_features, query):
        return (float("nan"), -1) if return_index else float("nan")
    n_posterior = len(posterior.posterior_samples["eta"])
    rng = np.random.default_rng(int(random_seed))
    index = int(rng.integers(n_posterior)) if posterior_index is None else int(posterior_index) % n_posterior
    mean, variance = _conditional_latent_moments(posterior, query, index)
    latent = float(rng.normal(mean, math.sqrt(variance)))
    tolerance = float(ndtr(latent))
    return (tolerance, index) if return_index else tolerance


def posterior_predictive_tolerances(
    posterior: ToleranceSurfacePosterior,
    query_features: Sequence[float],
    *,
    max_posterior_draws: int = 2000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Draw the posterior predictive tolerance at one query.

    An evenly spaced subset of at most ``max_posterior_draws`` retained draws is
    used.  For each, the latent at the query is Gaussian given the source
    latents (:func:`_conditional_latent_moments`), and one tolerance is drawn
    from it.  The query is never refused, so a caller that needs the source
    support rule must apply :func:`is_within_source_support` itself.

    Args:
        posterior: Fitted tolerance surface.
        query_features: Feature vector in the surface's input coordinates.
        max_posterior_draws: Largest number of retained draws to use.
        random_seed: Seed of the predictive draws.

    Returns:
        One row per used draw with ``posterior_index``, the latent mean and
        standard deviation, the drawn ``tolerance`` and that draw's ``eta``.
    """

    if max_posterior_draws < 1:
        raise ValueError("max_posterior_draws must be positive")
    query = np.asarray(query_features, dtype=float)
    n_posterior = len(posterior.posterior_samples["eta"])
    count = min(n_posterior, int(max_posterior_draws))
    indices = np.unique(np.linspace(0, n_posterior - 1, count).round().astype(int))
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, float | int]] = []
    for index in indices:
        mean, variance = _conditional_latent_moments(posterior, query, int(index))
        sd = math.sqrt(variance)
        rows.append({
            "posterior_index": int(index),
            "latent_mean": mean,
            "latent_sd": sd,
            "tolerance": float(ndtr(rng.normal(mean, sd))),
            "eta": float(posterior.posterior_samples["eta"][int(index)]),
        })
    return pd.DataFrame(rows)


def score_held_out_tolerance(
    posterior: ToleranceSurfacePosterior,
    query_features: Sequence[float],
    omega: float,
    *,
    max_posterior_draws: int = 2000,
    random_seed: int = 0,
) -> dict[str, float]:
    """Score one held-out evidence unit under a tolerance posterior.

    ``validity_probability`` is ``P(tolerance >= omega)``, exact given each
    draw's Gaussian latent: the posterior probability that a tolerance grounded
    without this unit is still admissible for it.  A low value means grounding
    transports too tight a tolerance to a new site, which invalidates bounds.
    ``log_predictive_density`` is the log posterior predictive density of
    ``omega`` under the fit's reverse-exponential likelihood
    ``eta exp(eta omega) / (exp(eta c) - 1)`` on ``omega <= c`` (the unit's
    leave-one-out expected-log-predictive-density contribution), ``-inf`` when
    no predictive draw admits ``omega``.

    Args:
        posterior: Tolerance surface fitted without the held-out unit.
        query_features: The held-out unit's own trigger statistics.
        omega: The held-out unit's solver-scale required tolerance.
        max_posterior_draws: Largest number of retained draws to use.
        random_seed: Seed of the predictive draws.

    Returns:
        The two scores, predictive tolerance quantiles and the draw count.
    """

    value = float(np.clip(float(omega), NUMERICAL_PROBABILITY_FLOOR, 1.0 - NUMERICAL_PROBABILITY_FLOOR))
    draws = posterior_predictive_tolerances(
        posterior,
        query_features,
        max_posterior_draws=max_posterior_draws,
        random_seed=random_seed,
    )
    mean = draws.latent_mean.to_numpy(dtype=float)
    sd = draws.latent_sd.to_numpy(dtype=float)
    lower_latent = float(ndtri(value))
    # A zero-variance latent is a point mass: admissible iff its mean clears omega.
    validity = np.where(
        sd > 0.0,
        1.0 - ndtr((lower_latent - mean) / np.where(sd > 0.0, sd, 1.0)),
        (mean >= lower_latent).astype(float),
    )
    tolerance = draws.tolerance.to_numpy(dtype=float)
    eta = draws.eta.to_numpy(dtype=float)
    admissible = tolerance >= value
    log_density = np.full(len(draws), -np.inf)
    log_density[admissible] = (
        np.log(eta[admissible])
        + eta[admissible] * value
        - np.log(np.expm1(eta[admissible] * tolerance[admissible]))
    )
    return {
        "validity_probability": float(validity.mean()),
        "log_predictive_density": (
            float(logsumexp(log_density) - math.log(len(draws))) if admissible.any() else float("-inf")
        ),
        "predictive_tolerance_q05": float(np.quantile(tolerance, 0.05)),
        "predictive_tolerance_q50": float(np.quantile(tolerance, 0.50)),
        "predictive_tolerance_q95": float(np.quantile(tolerance, 0.95)),
        "n_predictive_draws": int(len(draws)),
    }


def leave_one_source_out_scores(
    evidence: pd.DataFrame,
    *,
    target_queries: Mapping[str, Sequence[float]] | None = None,
    random_seed: int = 20260922,
    max_posterior_draws: int = 2000,
    held_out_sites: Sequence[str] | None = None,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Refit every surface without each source site and score that site.

    For each source site ``k`` of one imputation, all surfaces are refitted by
    :func:`fit_tolerance_surfaces` on the other sites' evidence, and each of site
    ``k``'s evidence units is scored by :func:`score_held_out_tolerance` at its
    own trigger statistics.  This diagnoses how grounding transports to a new
    site; nothing feeds back into a fit.  A refit whose sampler diagnostics still
    fail after the retries is recorded with its error rather than raised, so it
    cannot hide the other sites' scores.

    Args:
        evidence: One imputation's :func:`compute_source_evidence` rows for every
            source site.
        target_queries: Optional ``candidate -> (i_deviation, d_strength)``; when
            given, each refit's predictive median tolerance at the target's
            statistics is also reported, showing how much the target's grounded
            tolerance rests on any single source site.
        random_seed: Root seed; each omission derives its own fitting seed.
        max_posterior_draws: Retained draws used per predictive score.
        held_out_sites: Optional subset of sites to hold out, so refits can be
            spread over processes; every refit still uses all other sites and
            seeds do not depend on the subset, so the scores equal a full call's.
        **fit_kwargs: Sampler settings passed to :func:`fit_tolerance_surfaces`.

    Returns:
        One row per held-out site, candidate and tolerance kind: the held-out
        statistics and evidence, whether they lie in the remaining sources'
        support, the scores, the refit's diagnostics and, when requested, the
        refit's predictive median at the target query.
    """

    sites = safe_sorted(evidence.site.astype(str).unique())
    if len(sites) < 3:
        raise ValueError("Leave-one-source-out scoring needs at least three source sites")
    selected = sites if held_out_sites is None else [str(site) for site in held_out_sites]
    unknown = sorted(set(selected) - set(sites))
    if unknown:
        raise ValueError(f"Unknown held-out sites {unknown}")
    rows: list[dict[str, Any]] = []
    for held_out in selected:
        LOGGER.info("Leave-one-source-out refit %s/%s: without %s", sites.index(held_out) + 1, len(sites), held_out)
        remaining = evidence.loc[evidence.site.astype(str).ne(held_out)]
        held = evidence.loc[evidence.site.astype(str).eq(held_out)]
        fit_error = ""
        unconverged: list[dict[str, Any]] = []
        try:
            fitted = fit_tolerance_surfaces(
                remaining,
                random_seed=_stable_seed(random_seed, "leave-one-source-out", held_out),
                unconverged=unconverged,
                **fit_kwargs,
            )
        except (RuntimeError, ValueError) as error:
            fitted = {}
            fit_error = repr(error)
        excluded = {(str(entry["candidate"]), str(entry["tolerance_kind"])) for entry in unconverged}
        for unit in held.itertuples(index=False):
            candidate = str(unit.candidate)
            for kind, omega_column, feature_names in _SURFACE_SPECS:
                query = [float(getattr(unit, name)) for name in feature_names]
                omega = float(getattr(unit, omega_column))
                record: dict[str, Any] = {
                    "held_out_site": held_out,
                    "candidate": candidate,
                    "tolerance_kind": kind,
                    "i_deviation": float(unit.i_deviation),
                    "d_strength": float(unit.d_strength),
                    "omega": omega,
                    "fit_error": "surface_not_converged" if (candidate, kind) in excluded else fit_error,
                }
                posterior = fitted.get((candidate, kind))
                scoreable = posterior is not None and bool(np.all(np.isfinite([*query, omega])))
                record["scored"] = scoreable
                if scoreable:
                    record["n_remaining_units"] = int(len(posterior.source_features))
                    record["in_source_support"] = bool(
                        is_within_source_support(posterior.source_features, query)
                    )
                    record.update(score_held_out_tolerance(
                        posterior,
                        query,
                        omega,
                        max_posterior_draws=max_posterior_draws,
                        random_seed=_stable_seed(random_seed, "held-out-score", held_out, candidate, kind),
                    ))
                    for key in ("passed", "attempt", "max_r_hat", "min_bulk_ess", "min_tail_ess", "n_divergent"):
                        record[f"refit_{key}"] = posterior.diagnostics.get(key)
                    if target_queries is not None and candidate in target_queries:
                        target = [float(value) for value in target_queries[candidate]][: len(feature_names)]
                        if np.all(np.isfinite(target)):
                            predictive = posterior_predictive_tolerances(
                                posterior,
                                target,
                                max_posterior_draws=max_posterior_draws,
                                random_seed=_stable_seed(random_seed, "target-query", held_out, candidate, kind),
                            )
                            record["target_predictive_q50"] = float(predictive.tolerance.median())
                            record["target_in_source_support"] = bool(
                                is_within_source_support(posterior.source_features, target)
                            )
                rows.append(record)
    return pd.DataFrame(rows)


def grid_trigger(
    i_deviation: float,
    d_strength: float,
    epsilons: Sequence[float],
    taus: Sequence[float],
) -> tuple[bool, float, float]:
    """Apply the any-grid-pair trigger rule to one candidate's statistics.

    The candidate triggers when some pair of the grid fires, that is when
    ``i_deviation <= epsilon`` for some ``epsilon`` and ``d_strength >= tau`` for
    some ``tau``.  Nonfinite statistics never trigger.

    Args:
        i_deviation: The exclusion statistic ``I_j``.
        d_strength: The treatment-relevance statistic ``D_j``.
        epsilons: The ``epsilon`` grid.
        taus: The ``tau`` grid.

    Returns:
        Whether the candidate triggers, and the tightest firing pair: the
        smallest firing ``epsilon`` and the largest firing ``tau`` (both ``NaN``
        when it does not trigger).
    """

    finite = bool(np.isfinite(i_deviation) and np.isfinite(d_strength))
    firing_epsilons = [float(value) for value in epsilons if finite and i_deviation <= float(value)]
    firing_taus = [float(value) for value in taus if finite and d_strength >= float(value)]
    triggered = bool(firing_epsilons and firing_taus)
    return (
        triggered,
        min(firing_epsilons) if triggered else float("nan"),
        max(firing_taus) if triggered else float("nan"),
    )


@dataclass(frozen=True)
class RandomIdentifiedSetResult:
    """Sample-level random identified sets and target discovery diagnostics."""

    draws: pd.DataFrame
    triggers: pd.DataFrame


def _nonbinding_manski_choices(covariates: Sequence[str]) -> pd.DataFrame:
    """Create solver-compatible constraints that do not tighten binary bounds."""

    return pd.DataFrame([
        {
            "covariate": candidate,
            "covariate_index": index,
            "two_c": 1.0,
            "two_d": 1.0,
            "epsilon": np.nan,
            "tau": np.nan,
        }
        for index, candidate in enumerate(covariates)
    ])


def _site_average_weights(
    draw: SitePMFDraw,
    quantities: TargetObservedBinaryQuantities,
) -> dict[tuple[int, ...], float]:
    """Normalize a target ``P(Z)`` draw over LP-supported strata."""

    masses = {z: float(draw.idle_z[z]) for z in quantities.z_values}
    total = float(sum(masses.values()))
    if total <= 0.0:
        raise ValueError("Supported target strata have zero posterior mass")
    return {z: mass / total for z, mass in masses.items()}


def _lp_rows(
    *,
    sample_id: str,
    imputation_id: str,
    target_draw: SitePMFDraw,
    quantities: TargetObservedBinaryQuantities,
    choices: pd.DataFrame,
    method: str,
    used_manski_fallback: bool,
    trigger_payload: str,
) -> list[dict[str, Any]]:
    """Solve cellwise and jointly weighted LP objectives for one posterior draw.

    ``status`` is ``optimal`` only when both endpoint solves are.  Endpoint
    statuses are kept because an ``infeasible`` endpoint means an empty random
    identified set, whereas ``failed`` or ``unbounded`` means the solver gave no
    answer; the two must not be pooled.
    """

    cate = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        choices,
        return_details=True,
        include_intervention_diagnostics=False,
    )
    weights = _site_average_weights(target_draw, quantities)
    ate = solve_target_weighted_cate_bounds_from_observed_binary_quantities(
        quantities,
        choices,
        {("site_average",): weights},
        query_covariates=("estimand",),
        return_details=True,
        include_intervention_diagnostics=False,
    )
    rows: list[dict[str, Any]] = []
    for result in cate.itertuples(index=False):
        z = as_tuple(result.z_value)
        rows.append({
            "sample_id": sample_id,
            "imputation_id": imputation_id,
            "target_pmf_sample_id": target_draw.sample_id,
            "method": method,
            "estimand_type": "cate",
            "estimand_key": json.dumps(list(z)),
            "estimand_weight": weights[z],
            "lower": float(result.lower),
            "upper": float(result.upper),
            "status": str(result.status),
            "lower_status": str(result.lower_status),
            "upper_status": str(result.upper_status),
            "feasible": bool(result.status == "optimal"),
            "used_manski_fallback": bool(used_manski_fallback),
            "trigger_payload": trigger_payload,
        })
    for result in ate.itertuples(index=False):
        rows.append({
            "sample_id": sample_id,
            "imputation_id": imputation_id,
            "target_pmf_sample_id": target_draw.sample_id,
            "method": method,
            "estimand_type": "ate",
            "estimand_key": "site_average",
            "estimand_weight": 1.0,
            "lower": float(result.lower),
            "upper": float(result.upper),
            "status": str(result.status),
            "lower_status": str(result.lower_status),
            "upper_status": str(result.upper_status),
            "feasible": bool(result.status == "optimal"),
            "used_manski_fallback": bool(used_manski_fallback),
            "trigger_payload": trigger_payload,
        })
    return rows


def draw_random_identified_sets(
    target_pmf_draws: Sequence[SitePMFDraw],
    tolerance_posteriors: Sequence[Mapping[tuple[str, str], ToleranceSurfacePosterior]],
    *,
    n_draws: int = 2000,
    epsilon: float = 0.10,
    tau: float = 0.10,
    min_cell_count: int = 1,
    random_seed: int = 20260922,
    epsilons: Sequence[float] | None = None,
    taus: Sequence[float] | None = None,
    require_source_support: bool = True,
    saved_tolerances: Mapping[tuple[str, str], tuple[float, int, float, int]] | None = None,
) -> RandomIdentifiedSetResult:
    """Propagate paired PMF/tolerance draws through Naturalness and Manski LPs.

    Target draws are paired cyclically with the equal-weight source-imputation
    mixture, and the target's discovery statistics are recomputed from the PMF
    draw that builds the LP coefficients.  With no eligible candidate the
    Naturalness draw equals its matched Manski draw; LP infeasibility is kept as
    missing endpoints.

    A candidate triggers at ``(epsilon, tau)`` or, when the grids ``epsilons``
    and ``taus`` are given, when any grid pair fires (:func:`grid_trigger`;
    trigger rows then record the tightest firing pair).  The threshold only
    decides eligibility: the tolerance is read at the target's own statistics,
    which must lie in the source support when ``require_source_support`` is set.

    Draw and trigger rows carry ``draw_index``, ``target_pmf_draw_index`` and
    ``source_imputation_index``; trigger rows also carry the retained surface
    draws the tolerances were conditioned on (``c_posterior_index`` and
    ``d_posterior_index``, ``-1`` when none was drawn), and LP rows both
    endpoint statuses.

    ``saved_tolerances`` maps ``(sample_id, candidate)`` to an earlier call's
    ``(two_c, c_posterior_index, two_d, d_posterior_index)`` and replays them
    instead of sampling.  Tolerance seeds depend on the random seed, sample and
    candidate but not on the grid, so a replay with the same target draws,
    surfaces and seed on a coarser grid reproduces every candidate that still
    triggers; the surfaces then supply only source features and imputation IDs.
    A needed tolerance missing from the mapping raises ``KeyError``.
    """

    if n_draws < 1:
        raise ValueError("n_draws must be at least one")
    if not target_pmf_draws:
        raise ValueError("At least one target PMF draw is required")
    if not tolerance_posteriors:
        raise ValueError("At least one source-imputation posterior is required")
    if (epsilons is None) != (taus is None):
        raise ValueError("epsilons and taus must be given together")
    grid = None if epsilons is None else (
        tuple(sorted(float(value) for value in epsilons)),
        tuple(sorted(float(value) for value in taus)),
    )
    if grid is not None and (not grid[0] or not grid[1]):
        raise ValueError("epsilons and taus must be nonempty")
    draw_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    for draw_index in range(int(n_draws)):
        target_index = draw_index % len(target_pmf_draws)
        target_draw = target_pmf_draws[target_index]
        posterior_index = draw_index % len(tolerance_posteriors)
        posteriors = tolerance_posteriors[posterior_index]
        # An imputation whose every surface was excluded as unconverged has an
        # empty mapping; its draws are Manski fallbacks under a stable label.
        imputation_id = (
            next(iter(posteriors.values())).imputation_id if posteriors else f"no-converged-surface-{posterior_index}"
        )
        sample_id = _stable_sample_id("random-set", random_seed, draw_index, target_draw.sample_id, imputation_id)
        # Input positions, so every row traces back to its target draw and imputation.
        draw_indices = {
            "draw_index": int(draw_index),
            "target_pmf_draw_index": int(target_index),
            "source_imputation_index": int(posterior_index),
        }
        statistics = compute_trigger_statistics(target_draw)
        choices: list[dict[str, Any]] = []
        payload: list[dict[str, Any]] = []
        for statistic in statistics.itertuples(index=False):
            c_surface = posteriors.get((str(statistic.candidate), "c"))
            d_surface = posteriors.get((str(statistic.candidate), "d"))
            finite = bool(np.isfinite(statistic.i_deviation) and np.isfinite(statistic.d_strength))
            if grid is None:
                triggered = bool(
                    finite
                    and float(statistic.i_deviation) <= float(epsilon)
                    and float(statistic.d_strength) >= float(tau)
                )
            else:
                triggered, trigger_epsilon, trigger_tau = grid_trigger(
                    float(statistic.i_deviation), float(statistic.d_strength), grid[0], grid[1]
                )
            two_c = float("nan")
            two_d = float("nan")
            c_posterior_index = -1
            d_posterior_index = -1
            in_support = False
            if (
                c_surface is not None
                and d_surface is not None
                and np.isfinite(statistic.i_deviation)
                and np.isfinite(statistic.d_strength)
            ):
                c_query = (float(statistic.i_deviation),)
                d_query = (float(statistic.i_deviation), float(statistic.d_strength))
                in_support = is_within_source_support(c_surface.source_features, c_query) and is_within_source_support(
                    d_surface.source_features, d_query
                )
                if triggered and (in_support or not require_source_support) and saved_tolerances is not None:
                    saved = saved_tolerances.get((sample_id, str(statistic.candidate)))
                    if saved is None:
                        raise KeyError(f"No saved tolerance for sample {sample_id} and candidate {statistic.candidate!r}")
                    two_c, c_posterior_index, two_d, d_posterior_index = (
                        float(saved[0]), int(saved[1]), float(saved[2]), int(saved[3])
                    )
                elif triggered and (in_support or not require_source_support):
                    two_c, c_posterior_index = sample_tolerance_surface(
                        c_surface,
                        c_query,
                        random_seed=_stable_seed(random_seed, sample_id, statistic.candidate, "c"),
                        allow_extrapolation=not require_source_support,
                        return_index=True,
                    )
                    two_d, d_posterior_index = sample_tolerance_surface(
                        d_surface,
                        d_query,
                        random_seed=_stable_seed(random_seed, sample_id, statistic.candidate, "d"),
                        allow_extrapolation=not require_source_support,
                        return_index=True,
                    )
            eligible = bool(
                triggered
                and (in_support or not require_source_support)
                and np.isfinite(two_c)
                and np.isfinite(two_d)
            )
            if eligible:
                choices.append({
                    "covariate": statistic.candidate,
                    "covariate_index": int(statistic.candidate_index),
                    "two_c": two_c,
                    "two_d": two_d,
                    "epsilon": float(statistic.i_deviation),
                    "tau": float(statistic.d_strength),
                })
            record = {
                "sample_id": sample_id,
                **draw_indices,
                "imputation_id": imputation_id,
                "target_pmf_sample_id": target_draw.sample_id,
                "candidate": statistic.candidate,
                "candidate_index": int(statistic.candidate_index),
                "i_deviation": float(statistic.i_deviation),
                "d_strength": float(statistic.d_strength),
                "triggered": triggered,
                "in_source_support": in_support,
                "eligible": eligible,
                "two_c": two_c,
                "two_d": two_d,
                "c_posterior_index": int(c_posterior_index),
                "d_posterior_index": int(d_posterior_index),
            }
            if grid is not None:
                record["trigger_epsilon"] = trigger_epsilon
                record["trigger_tau"] = trigger_tau
            trigger_rows.append(record)
            payload.append({key: record[key] for key in ("candidate", "i_deviation", "d_strength", "triggered", "in_source_support", "eligible", "two_c", "two_d")})

        quantities = target_draw.observed_quantities(min_cell_count=min_cell_count)
        if not quantities.z_values:
            raise ValueError(f"Target draw {target_draw.sample_id} has no empirically supported LP strata")
        manski_choices = _nonbinding_manski_choices(target_draw.covariates)
        trigger_payload = json.dumps(payload, sort_keys=True)
        manski_rows = [
            {**draw_indices, **row}
            for row in _lp_rows(
                sample_id=sample_id,
                imputation_id=imputation_id,
                target_draw=target_draw,
                quantities=quantities,
                choices=manski_choices,
                method="bayesian_manski",
                used_manski_fallback=False,
                trigger_payload=trigger_payload,
            )
        ]
        draw_rows.extend(manski_rows)
        if choices:
            draw_rows.extend(
                {**draw_indices, **row}
                for row in _lp_rows(
                    sample_id=sample_id,
                    imputation_id=imputation_id,
                    target_draw=target_draw,
                    quantities=quantities,
                    choices=pd.DataFrame(choices),
                    method="bayesian_naturalness",
                    used_manski_fallback=False,
                    trigger_payload=trigger_payload,
                )
            )
        else:
            for row in manski_rows:
                draw_rows.append({
                    **row,
                    "method": "bayesian_naturalness",
                    "used_manski_fallback": True,
                })
    return RandomIdentifiedSetResult(
        draws=pd.DataFrame(draw_rows),
        triggers=pd.DataFrame(trigger_rows),
    )


def posterior_outer_envelopes(draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize random-set draws with 95% posterior outer envelopes.

    The envelope is ``[Q_.025(L), Q_.975(U)]`` among feasible draws, reported
    with the unconditional feasibility probability; it summarizes the random
    identified set (see the module docstring for how to read it).
    """

    required = {"sample_id", "method", "estimand_type", "estimand_key", "lower", "upper", "feasible"}
    missing = required - set(draws)
    if missing:
        raise ValueError(f"Random-set table lacks columns {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for keys, group in draws.groupby(["method", "estimand_type", "estimand_key"], observed=True, sort=True):
        method, estimand_type, estimand_key = keys
        feasible = group.loc[group.feasible.astype(bool)].copy()
        rows.append({
            "method": method,
            "estimand_type": estimand_type,
            "estimand_key": estimand_key,
            "n_draws": int(group.sample_id.nunique()),
            "n_feasible": int(feasible.sample_id.nunique()),
            "feasibility_probability": float(feasible.sample_id.nunique() / group.sample_id.nunique()),
            "outer_lower": float(feasible.lower.quantile(0.025)) if not feasible.empty else np.nan,
            "outer_upper": float(feasible.upper.quantile(0.975)) if not feasible.empty else np.nan,
            "mean_random_set_width": float((feasible.upper - feasible.lower).mean()) if not feasible.empty else np.nan,
            "mean_estimand_weight": float(feasible.estimand_weight.mean()) if "estimand_weight" in feasible and not feasible.empty else np.nan,
        })
    output = pd.DataFrame(rows)
    output["outer_width"] = output.outer_upper - output.outer_lower
    return output


def draw_rct_reference_posterior(
    evaluation: pd.DataFrame,
    target_pmf_draws: Sequence[SitePMFDraw],
    *,
    n_draws: int,
    alpha: float = 1.0,
    min_arm_count: int = 1,
    random_seed: int = 20260922,
    treatment: str = TREATMENT_COLUMN,
    outcome: str = OUTCOME_COLUMN,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """Draw held-out randomized CATE and target-weighted ATE references.

    Evaluation outcomes enter only this function.  They never enter raw-PMF
    fitting, tolerance learning, target discovery, or LP construction.  Every
    treatment-by-stratum cell must have at least ``min_arm_count`` physical
    rows; prior-only reference cells are forbidden.
    """

    if not target_pmf_draws:
        raise ValueError("Target PMF draws are required for ATE reference weights")
    if min_arm_count < 1:
        raise ValueError("min_arm_count must be at least one")
    _validate_binary(evaluation, [treatment, outcome, *covariates])
    z_values = tuple(safe_sorted({tuple(int(v) for v in row) for row in evaluation[list(covariates)].itertuples(index=False, name=None)}))
    counts: dict[tuple[int, tuple[int, ...]], tuple[int, int]] = {}
    for z in z_values:
        mask_z = np.logical_and.reduce([evaluation[column].eq(value) for column, value in zip(covariates, z)])
        for x in (0, 1):
            cell = evaluation.loc[mask_z & evaluation[treatment].eq(x), outcome]
            counts[(x, z)] = (int(cell.sum()), int(len(cell) - cell.sum()))
            if len(cell) < min_arm_count:
                raise ValueError(
                    f"Held-out RCT cell x={x}, z={z} has {len(cell)} rows; "
                    f"requires {min_arm_count}"
                )
    rows: list[dict[str, Any]] = []
    for draw_index in range(int(n_draws)):
        target_draw = target_pmf_draws[draw_index % len(target_pmf_draws)]
        rng = np.random.default_rng(_stable_seed(random_seed, "rct-reference", draw_index))
        cate: dict[tuple[int, ...], float] = {}
        for z in z_values:
            arm = {}
            for x in (0, 1):
                successes, failures = counts[(x, z)]
                arm[x] = float(rng.beta(successes + alpha, failures + alpha))
            cate[z] = arm[1] - arm[0]
            rows.append({
                "reference_draw_index": draw_index,
                "estimand_type": "cate",
                "estimand_key": json.dumps(list(z)),
                "reference_value": cate[z],
            })
        common = [z for z in z_values if z in target_draw.idle_z]
        total = float(sum(target_draw.idle_z[z] for z in common))
        if total <= 0.0:
            raise ValueError("Held-out RCT strata have zero target posterior mass")
        ate = float(sum(target_draw.idle_z[z] * cate[z] for z in common) / total)
        rows.append({
            "reference_draw_index": draw_index,
            "estimand_type": "ate",
            "estimand_key": "site_average",
            "reference_value": ate,
        })
    return pd.DataFrame(rows)


def score_random_sets_against_rct(
    draws: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair random sets with independent held-out-RCT posterior draws."""

    sample_order = draws[["sample_id"]].drop_duplicates().reset_index(drop=True)
    sample_order.insert(0, "reference_draw_index", np.arange(len(sample_order), dtype=int))
    paired = draws.merge(sample_order, on="sample_id", validate="many_to_one")
    paired = paired.merge(
        references,
        on=["reference_draw_index", "estimand_type", "estimand_key"],
        how="left",
        validate="many_to_one",
    )
    paired["reference_member"] = paired.feasible.astype(bool) & paired.reference_value.between(
        paired.lower, paired.upper, inclusive="both"
    )
    paired["wrong_sign"] = paired.feasible.astype(bool) & (
        ((paired.lower > 0.0) & (paired.reference_value <= 0.0))
        | ((paired.upper < 0.0) & (paired.reference_value >= 0.0))
    )
    naturalness = paired.loc[
        paired.method.eq("bayesian_naturalness") & paired.estimand_type.eq("cate")
    ].copy()
    if naturalness.empty:
        metrics: dict[str, Any] = {
            "rct_reference_membership": np.nan,
            "wrong_sign_mass": np.nan,
        }
    else:
        per_sample_rows: list[dict[str, float | str]] = []
        for sample_id, group in naturalness.groupby("sample_id", observed=True):
            feasible = group.loc[group.feasible.astype(bool)]
            per_sample_rows.append({
                "sample_id": str(sample_id),
                "membership": float(np.average(group.reference_member, weights=group.estimand_weight)),
                "wrong_sign": (
                    float(np.average(feasible.wrong_sign, weights=feasible.estimand_weight))
                    if not feasible.empty
                    else np.nan
                ),
            })
        per_sample = pd.DataFrame(per_sample_rows)
        metrics = {
            "rct_reference_membership": float(per_sample.membership.mean()),
            "wrong_sign_mass": float(per_sample.wrong_sign.mean()),
        }
    reference_intervals = (
        references.groupby(["estimand_type", "estimand_key"], observed=True).reference_value
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "reference_q025", 0.975: "reference_q975"})
    )
    naturalness_envelopes = posterior_outer_envelopes(draws).loc[
        lambda frame: frame.method.eq("bayesian_naturalness")
    ]
    outer_checks = naturalness_envelopes.merge(
        reference_intervals,
        on=["estimand_type", "estimand_key"],
        how="left",
        validate="one_to_one",
    )
    outer_checks["contains_reference_95_interval"] = (
        outer_checks.outer_lower.le(outer_checks.reference_q025)
        & outer_checks.outer_upper.ge(outer_checks.reference_q975)
    )
    cate_checks = outer_checks.loc[outer_checks.estimand_type.eq("cate")]
    if cate_checks.empty:
        metrics["rct_reference_95_interval_outer_containment_mass"] = np.nan
    else:
        weights = cate_checks.mean_estimand_weight.to_numpy(dtype=float)
        weights = weights / weights.sum()
        metrics["rct_reference_95_interval_outer_containment_mass"] = float(
            np.dot(weights, cate_checks.contains_reference_95_interval.astype(float))
        )
    ate_check = outer_checks.loc[outer_checks.estimand_type.eq("ate")]
    metrics["ate_reference_95_interval_outer_contained"] = bool(
        len(ate_check) == 1 and ate_check.contains_reference_95_interval.iloc[0]
    )
    return paired, metrics


__all__ = [
    "DEFAULT_COVARIATES",
    "DEFAULT_EPSILONS",
    "DEFAULT_TAUS",
    "RandomIdentifiedSetResult",
    "RawPMFImputation",
    "SitePMFDraw",
    "ToleranceSurfacePosterior",
    "compute_source_evidence",
    "compute_trigger_statistics",
    "draw_random_identified_sets",
    "draw_raw_pmfs",
    "draw_rct_reference_posterior",
    "fit_tolerance_surfaces",
    "grid_trigger",
    "is_within_source_support",
    "leave_one_source_out_scores",
    "posterior_outer_envelopes",
    "posterior_predictive_tolerances",
    "raw_pmf_draws_to_frame",
    "raw_pmf_posterior_parameters",
    "rows_from_raw_pmf_posterior_parameters",
    "sample_tolerance_surface",
    "score_held_out_tolerance",
    "score_random_sets_against_rct",
]
