"""A short public entry point to Naturalness bounds.

:class:`Naturalness` wraps the package's pipeline in a fit/predict-style
estimator: Stages 1-3 on the source sites (:class:`NaturalnessGrounder`), the
target conditions (:class:`TargetNaturalnessPolicy`), and tolerance draws
propagated through the target linear program. It only calls the existing
functions, in the order and with the arguments of ``examples/toy_naturalness.py``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from .loaders.contract import SiteBundle
from .lp_solver_learned_naturalness import (
    estimate_target_observed_binary_quantities,
    solve_target_bounds_from_observed_binary_quantities,
)
from .naturalness_grounder import NaturalnessGrid, NaturalnessGrounder
from .naturalness_target_policy import (
    TargetNaturalnessPolicy,
    TargetNaturalnessPolicyResult,
)

from .paper_grids import MIN_RELEVANCE_TAU, PAPER_EPSILONS, PAPER_TAUS  # noqa: F401  re-exported

SourceData = Mapping[str, pd.DataFrame] | pd.DataFrame | SiteBundle


class Naturalness:
    """Naturalness bounds on the target CATE of a binary treatment.

    Parameters
    ----------
    covariates : sequence of str
        Discrete covariate columns that define the strata ``Z``.
    epsilons : sequence of float, default=PAPER_EPSILONS
        Naturalness thresholds ``epsilon`` of the trigger ``I_j <= epsilon``.
    taus : sequence of float, default=PAPER_TAUS
        Relevance thresholds ``tau`` of the trigger ``D_j >= tau``.
    stage1_backend : str, default="empirical"
        Probability model of Stage 1 on the sources. The paper uses
        ``"tabicl_v2"``.
    target_backend : str, default="empirical"
        Probability model of the target's ``P(X | Z)`` and ``P(Y | X, Z)``,
        passed as ``probability_backend`` to
        :func:`estimate_target_observed_binary_quantities`.
    n_draws : int, default=1000
        Tolerance draws propagated through the target linear program.
    random_state : int or None, default=None
        Seed of the Gaussian-process fits and of the tolerance draws.
    treatment, outcome, regime_col : str, default="X", "Y", "F"
        Binary treatment, binary outcome and regime columns. The regime column
        holds ``"on"`` for randomized rows and ``"idle"`` for observational rows.
    grid_options : dict or None, default=None
        Extra keyword arguments of :class:`NaturalnessGrid`, such as
        ``candidate_covariates`` or ``min_cell_count``.
    grounder_options : dict or None, default=None
        Extra keyword arguments of :class:`NaturalnessGrounder`.
    policy_options : dict or None, default=None
        Keyword arguments of :class:`TargetNaturalnessPolicy`.

    Attributes
    ----------
    grid_ : NaturalnessGrid
        The grid of the fitted grounder.
    grounder_ : NaturalnessGrounder
        The grounder fitted on the source sites.
    n_sources_ : int
        Number of source sites.
    policy_result_ : TargetNaturalnessPolicyResult
        The target policy of the last :meth:`get_bounds` call.
    conditions_ : pandas.DataFrame
        The target conditions that fired in the last :meth:`get_bounds` call.

    Examples
    --------
    >>> model = Naturalness(covariates=["Z1", "Z2"], random_state=0)  # doctest: +SKIP
    >>> bounds = model.fit(sources).get_bounds(target)  # doctest: +SKIP
    """

    def __init__(
        self,
        covariates: Sequence[str],
        epsilons: Sequence[float] = PAPER_EPSILONS,
        taus: Sequence[float] = PAPER_TAUS,
        *,
        stage1_backend: str = "empirical",
        target_backend: str = "empirical",
        n_draws: int = 1000,
        random_state: int | None = None,
        treatment: str = "X",
        outcome: str = "Y",
        regime_col: str = "F",
        grid_options: dict[str, Any] | None = None,
        grounder_options: dict[str, Any] | None = None,
        policy_options: dict[str, Any] | None = None,
    ) -> None:
        self.covariates = covariates
        self.epsilons = epsilons
        self.taus = taus
        self.stage1_backend = stage1_backend
        self.target_backend = target_backend
        self.n_draws = n_draws
        self.random_state = random_state
        self.treatment = treatment
        self.outcome = outcome
        self.regime_col = regime_col
        self.grid_options = grid_options
        self.grounder_options = grounder_options
        self.policy_options = policy_options

    def fit(self, sources: SourceData, site_col: str | None = None) -> Naturalness:
        """Fit Stages 1-3 on the source sites.

        Parameters
        ----------
        sources : mapping, DataFrame or SiteBundle
            Source-site rows with both regimes: a mapping from site name to
            rows, one long frame with a ``site_col`` column, or a
            :class:`~causal_grounding.loaders.contract.SiteBundle` whose frames
            carry the regime column.
        site_col : str or None, default=None
            Site column of a long frame. Required for a frame, unused otherwise.

        Returns
        -------
        Naturalness
            The fitted estimator.
        """
        grid = NaturalnessGrid(
            epsilons=list(self.epsilons),
            taus=list(self.taus),
            covariates=list(self.covariates),
            **(self.grid_options or {}),
        )
        if grid.conditioning_covariates is not None:
            raise NotImplementedError(
                "Naturalness does not support grids with conditioning_covariates: "
                "their target linear program is solved per candidate state. Use "
                "NaturalnessGrounder and TargetNaturalnessPolicy directly."
            )
        site_frames = _site_frames(sources, site_col)
        grounder = NaturalnessGrounder(
            grid,
            stage1_backend=self.stage1_backend,
            random_seed=self.random_state,
            **(self.grounder_options or {}),
        )
        with warnings.catch_warnings():
            # The GP hyperparameter search reports bounds it touches; harmless.
            warnings.simplefilter("ignore", ConvergenceWarning)
            grounder.fit(
                site_frames,
                treatment=self.treatment,
                outcome=self.outcome,
                regime_col=self.regime_col,
            )
        self.grid_ = grid
        self.grounder_ = grounder
        self.n_sources_ = len(site_frames)
        return self

    def get_bounds(
        self,
        target: pd.DataFrame,
        n_draws: int | None = None,
        return_conditions: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Bound the CATE of every target stratum.

        Parameters
        ----------
        target : pandas.DataFrame
            Target-site observational rows.
        n_draws : int or None, default=None
            Tolerance draws; ``None`` uses the constructor's ``n_draws``.
        return_conditions : bool, default=False
            Also return the target conditions that fired.

        Returns
        -------
        bounds : pandas.DataFrame
            One row per target stratum with the covariates, ``lower``,
            ``upper`` and ``p_optimal``, the share of draws whose program is
            optimal. The endpoints are the means over the optimal draws, and
            ``NaN`` where no draw is optimal. When no condition fires at the
            target, Naturalness abstains: every endpoint is ``NaN`` and
            ``p_optimal`` is 0.
        conditions : pandas.DataFrame
            The target conditions that fired, empty when Naturalness abstains.
            Returned only when ``return_conditions`` is true.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "grounder_"):
            raise NotFittedError(
                "This Naturalness instance is not fitted yet. Call 'fit' first."
            )
        covariates = list(self.grid_.covariates)
        grounder = self.grounder_
        policy = TargetNaturalnessPolicy(**(self.policy_options or {})).choose(
            target, grounder
        )
        quantities = estimate_target_observed_binary_quantities(
            target,
            treatment=self.treatment,
            outcome=self.outcome,
            covariates=covariates,
            regime_col=self.regime_col,
            probability_backend=self.target_backend,
        )
        strata = pd.DataFrame(quantities.z_values, columns=covariates)
        self.policy_result_: TargetNaturalnessPolicyResult = policy
        if policy.status != "selected":
            # Nothing fires at the target: Naturalness abstains.
            bounds = strata.assign(lower=np.nan, upper=np.nan, p_optimal=0.0)
            self.conditions_ = pd.DataFrame()
            return (bounds, self.conditions_) if return_conditions else bounds

        conditions = policy.to_frame(selected_only=True)
        draws = grounder.sample_tolerances(
            conditions,
            n_samples=self.n_draws if n_draws is None else n_draws,
            random_seed=self.random_state,
        )
        solved = pd.concat(
            [
                solve_target_bounds_from_observed_binary_quantities(
                    quantities,
                    draw,
                    return_details=True,
                    include_intervention_diagnostics=False,
                )
                for _, draw in draws.groupby("sample_id")
            ],
            ignore_index=True,
        )
        solved["optimal"] = solved["status"].eq("optimal")
        summary = (
            solved.groupby(covariates).agg(p_optimal=("optimal", "mean")).reset_index()
        )
        endpoints = (
            solved[solved["optimal"]]
            .groupby(covariates)[["lower", "upper"]]
            .mean()
            .reset_index()
        )
        bounds = strata.merge(summary, on=covariates, how="left").merge(
            endpoints, on=covariates, how="left"
        )
        bounds = bounds[[*covariates, "lower", "upper", "p_optimal"]]
        self.conditions_ = conditions
        return (bounds, conditions) if return_conditions else bounds


def _site_frames(
    sources: SourceData, site_col: str | None
) -> dict[str, pd.DataFrame]:
    """Return the source rows as a mapping from site name to frame."""
    if isinstance(sources, SiteBundle):
        return {site: table.frame for site, table in sources.sites.items()}
    if isinstance(sources, pd.DataFrame):
        if site_col is None:
            raise ValueError("site_col is required when sources is one DataFrame")
        if site_col not in sources.columns:
            raise ValueError(f"site_col {site_col!r} is not a column of sources")
        return {
            str(site): frame.drop(columns=site_col).reset_index(drop=True)
            for site, frame in sources.groupby(site_col, sort=False)
        }
    if isinstance(sources, Mapping):
        return dict(sources)
    raise TypeError(
        "sources must be a mapping of site frames, a DataFrame with site_col, "
        f"or a SiteBundle; got {type(sources).__name__}"
    )
