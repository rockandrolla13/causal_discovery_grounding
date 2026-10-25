"""A short public entry point to Naturalness bounds.

:class:`Naturalness` wraps the package's pipeline in a fit/predict-style
estimator: Stages 1-3 on the source sites (:class:`NaturalnessGrounder`), the
target conditions (:class:`TargetNaturalnessPolicy`), and tolerance draws
propagated through the target linear program by :func:`posterior_target_bounds`.
With the default empirical backends it calls the package functions in the order
and with the arguments of ``examples/toy_naturalness.py``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from .loaders.contract import SiteBundle
from .lp_solver_learned_naturalness import (
    TargetObservedBinaryQuantities,
    estimate_target_observed_binary_quantities,
)
from .naturalness_grounder import (
    NATURALNESS_STAGE1_BACKENDS,
    NaturalnessGrid,
    NaturalnessGrounder,
)
from .naturalness_target_policy import (
    TargetNaturalnessPolicy,
    TargetNaturalnessPolicyResult,
)
from .target_bounds import (
    TARGET_PROBABILITY_BACKENDS,
    TARGET_PROBABILITY_DRAW_BACKENDS,
    posterior_target_bounds,
    target_probability_draw_seeds,
)

from .paper_grids import MIN_RELEVANCE_TAU, PAPER_EPSILONS, PAPER_TAUS  # noqa: F401  re-exported

SourceData = Mapping[str, pd.DataFrame] | pd.DataFrame | SiteBundle

# Keyword arguments the facade passes itself, by options parameter.
_RESERVED_OPTIONS: dict[str, frozenset[str]] = {
    "grid_options": frozenset({"epsilons", "taus", "covariates"}),
    "grounder_options": frozenset({"grid", "stage1_backend", "random_seed"}),
    "target_options": frozenset({"posterior_sample_seed"}),
}


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
        Probability model of Stage 1 on the sources, a key of
        ``NATURALNESS_STAGE1_BACKENDS``. The paper uses ``"tabicl_v2"``.
    target_backend : str, default="empirical"
        Probability model of the target's ``P(X | Z)`` and ``P(Y | X, Z)`` in
        the target linear program, passed as ``probability_backend`` to
        :func:`estimate_target_observed_binary_quantities`. A
        ``"hierarchical_dirichlet_draws"`` backend draws one target PMF per
        tolerance draw, as in the paper's Experiment I.
    trigger_backend : str or None, default=None
        Probability model of the target trigger screen. ``None`` uses
        ``target_backend``. ``"empirical"`` keeps the policy's smoothed cell
        proportions. A posterior-draw backend screens with its posterior mean.
    n_draws : int, default=1000
        Tolerance draws propagated through the target linear program.
    random_state : int or None, default=0
        Seed of the Gaussian-process fits, the tolerance draws and the target
        PMF draws. The default makes every run reproducible. ``None`` leaves
        them unseeded.
    n_jobs : int, default=1
        Threads that solve the per-draw target programs. The bounds do not
        depend on it.
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
    target_options : dict or None, default=None
        ``probability_model_kwargs`` of the target probability models, such as
        ``{"random_state": 0}`` for ``"tabicl_v2"``.

    Attributes
    ----------
    grid_ : NaturalnessGrid
        The grid of the fitted grounder.
    grounder_ : NaturalnessGrounder
        The grounder fitted on the source sites.
    n_sources_ : int
        Number of source sites.

    Examples
    --------
    >>> model = Naturalness(covariates=["Z1", "Z2"])  # doctest: +SKIP
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
        trigger_backend: str | None = None,
        n_draws: int = 1000,
        random_state: int | None = 0,
        n_jobs: int = 1,
        treatment: str = "X",
        outcome: str = "Y",
        regime_col: str = "F",
        grid_options: dict[str, Any] | None = None,
        grounder_options: dict[str, Any] | None = None,
        policy_options: dict[str, Any] | None = None,
        target_options: dict[str, Any] | None = None,
    ) -> None:
        self.covariates = covariates
        self.epsilons = epsilons
        self.taus = taus
        self.stage1_backend = stage1_backend
        self.target_backend = target_backend
        self.trigger_backend = trigger_backend
        self.n_draws = n_draws
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.treatment = treatment
        self.outcome = outcome
        self.regime_col = regime_col
        self.grid_options = grid_options
        self.grounder_options = grounder_options
        self.policy_options = policy_options
        self.target_options = target_options

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

        Raises
        ------
        TypeError
            If ``covariates`` is one string or ``sources`` has another type.
        ValueError
            If a parameter is invalid, an options dict sets a keyword the
            facade sets itself, or the sources do not match the model.
        """
        self._validate_params()
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
        site_frames = _as_site_mapping(
            sources,
            site_col,
            treatment=self.treatment,
            outcome=self.outcome,
            regime_col=self.regime_col,
            covariates=list(self.covariates),
        )
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
        return_policy: bool = False,
    ) -> pd.DataFrame | tuple[Any, ...]:
        """Bound the CATE of every target stratum.

        The call leaves the fitted model unchanged.

        Parameters
        ----------
        target : pandas.DataFrame
            Target-site observational rows.
        n_draws : int or None, default=None
            Tolerance draws; ``None`` uses the constructor's ``n_draws``.
        return_conditions : bool, default=False
            Also return the target conditions that fired.
        return_policy : bool, default=False
            Also return the full target policy result.

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
        policy : TargetNaturalnessPolicyResult
            The target policy, whose ``status`` says why Naturalness abstains.
            Returned only when ``return_policy`` is true, after ``conditions``
            when both are requested.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        ValueError
            If ``n_draws`` is below 1.
        """
        if not hasattr(self, "grounder_"):
            raise NotFittedError(
                "This Naturalness instance is not fitted yet. Call 'fit' first."
            )
        n_draws = _check_n_draws(self.n_draws if n_draws is None else n_draws)
        covariates = list(self.grid_.covariates)
        grounder = self.grounder_
        target_backend = self.target_backend.lower()
        trigger_backend = (self.trigger_backend or target_backend).lower()

        # For a posterior-draw backend this is the posterior mean. It fixes
        # the strata and, when the backends agree, feeds the trigger screen.
        quantities = self._target_quantities(target, target_backend)
        if trigger_backend == "empirical":
            trigger_quantities = None  # the policy's own smoothed proportions
        elif trigger_backend == target_backend:
            trigger_quantities = quantities
        else:
            trigger_quantities = self._target_quantities(target, trigger_backend)
        policy: TargetNaturalnessPolicyResult = TargetNaturalnessPolicy(
            **(self.policy_options or {})
        ).choose(target, grounder, trigger_quantities=trigger_quantities)

        if policy.status != "selected":
            # Nothing fires at the target: Naturalness abstains.
            strata = pd.DataFrame(quantities.z_values, columns=covariates)
            bounds = strata.assign(lower=np.nan, upper=np.nan, p_optimal=0.0)
            conditions = pd.DataFrame()
        else:
            conditions = policy.to_frame(selected_only=True)
            draws = grounder.sample_tolerances(
                conditions, n_samples=n_draws, random_seed=self.random_state
            )
            bounds = posterior_target_bounds(
                quantities,
                draws,
                covariates,
                draw_quantities=self._draw_quantities(target, target_backend, n_draws),
                n_jobs=self.n_jobs,
            )
        extras: list[Any] = []
        if return_conditions:
            extras.append(conditions)
        if return_policy:
            extras.append(policy)
        return (bounds, *extras) if extras else bounds

    def _target_quantities(
        self, target: pd.DataFrame, backend: str, **extra_options: Any
    ) -> TargetObservedBinaryQuantities:
        """Estimate the target probabilities with one backend."""
        options = {**(self.target_options or {}), **extra_options}
        return estimate_target_observed_binary_quantities(
            target,
            treatment=self.treatment,
            outcome=self.outcome,
            covariates=list(self.grid_.covariates),
            regime_col=self.regime_col,
            probability_backend=backend,
            probability_model_kwargs=options or None,
        )

    def _draw_quantities(
        self, target: pd.DataFrame, backend: str, n_draws: int
    ) -> Callable[[int], TargetObservedBinaryQuantities] | None:
        """Return the per-draw target PMFs of a posterior-draw backend, else None."""
        if backend not in TARGET_PROBABILITY_DRAW_BACKENDS:
            return None
        seed = self.random_state
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        seeds = target_probability_draw_seeds(seed, n_draws)

        def draw(sample_id: int) -> TargetObservedBinaryQuantities:
            """Estimate the target PMF of one tolerance draw."""
            return self._target_quantities(
                target, backend, posterior_sample_seed=seeds[sample_id]
            )

        return draw

    def _validate_params(self) -> None:
        """Reject invalid constructor parameters before any fitting."""
        if isinstance(self.covariates, str):
            raise TypeError(
                "covariates must be a sequence of column names, not the string "
                f"{self.covariates!r}; use [{self.covariates!r}]"
            )
        if len(self.covariates) == 0:
            raise ValueError("covariates must name at least one column")
        _check_n_draws(self.n_draws)
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, Integral):
            raise TypeError(f"n_jobs must be an int, got {self.n_jobs!r}")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        if (
            isinstance(self.stage1_backend, str)
            and self.stage1_backend not in NATURALNESS_STAGE1_BACKENDS
        ):
            raise ValueError(
                f"Unknown stage1_backend {self.stage1_backend!r}. Available: "
                f"{', '.join(sorted(NATURALNESS_STAGE1_BACKENDS))}"
            )
        for name in ("target_backend", "trigger_backend"):
            backend = getattr(self, name)
            if backend is None and name == "trigger_backend":
                continue
            if not isinstance(backend, str) or (
                backend.lower() not in TARGET_PROBABILITY_BACKENDS
            ):
                raise ValueError(
                    f"Unknown {name} {backend!r}. Available: "
                    f"{', '.join(sorted(TARGET_PROBABILITY_BACKENDS))}"
                )
        for name, reserved in _RESERVED_OPTIONS.items():
            clashes = sorted(reserved.intersection(getattr(self, name) or {}))
            if clashes:
                raise ValueError(
                    f"{name} must not set {clashes}: Naturalness sets them "
                    "itself; pass them as Naturalness parameters instead"
                )


def _check_n_draws(n_draws: Any) -> int:
    """Return ``n_draws`` as an int, or raise if it is not a positive integer."""
    if isinstance(n_draws, bool) or not isinstance(n_draws, Integral):
        raise TypeError(f"n_draws must be an int, got {n_draws!r}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    return int(n_draws)


def _as_site_mapping(
    sources: SourceData,
    site_col: str | None,
    *,
    treatment: str = "X",
    outcome: str = "Y",
    regime_col: str = "F",
    covariates: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    """Return the source rows as a mapping from site name to frame.

    A ``SiteBundle`` must match the model's treatment, outcome and covariates
    and carry the regime column, which loader bundles lack.
    """
    if isinstance(sources, SiteBundle):
        _check_bundle(sources, treatment, outcome, regime_col, covariates)
        site_frames = {site: table.frame for site, table in sources.sites.items()}
    elif isinstance(sources, pd.DataFrame):
        if site_col is None:
            raise ValueError("site_col is required when sources is one DataFrame")
        if site_col not in sources.columns:
            raise ValueError(f"site_col {site_col!r} is not a column of sources")
        site_frames = {
            str(site): frame.drop(columns=site_col).reset_index(drop=True)
            for site, frame in sources.groupby(site_col, sort=False, observed=True)
        }
    elif isinstance(sources, Mapping):
        site_frames = dict(sources)
    else:
        raise TypeError(
            "sources must be a mapping of site frames, a DataFrame with site_col, "
            f"or a SiteBundle; got {type(sources).__name__}"
        )
    if not site_frames:
        raise ValueError("sources holds no site")
    return site_frames


def _check_bundle(
    bundle: SiteBundle,
    treatment: str,
    outcome: str,
    regime_col: str,
    covariates: Sequence[str],
) -> None:
    """Raise if a bundle does not match the model or lacks the regime column."""
    missing = [name for name in covariates if name not in bundle.covariates]
    if missing:
        raise ValueError(
            f"covariates {missing} are not covariates of the SiteBundle "
            f"{bundle.dataset!r}, which has {list(bundle.covariates)}"
        )
    for site, table in bundle.sites.items():
        if (table.treatment, table.outcome) != (treatment, outcome):
            raise ValueError(
                f"SiteBundle site {site!r} has treatment {table.treatment!r} and "
                f"outcome {table.outcome!r}; the model expects {treatment!r} and "
                f"{outcome!r}. Set Naturalness(treatment=..., outcome=...)"
            )
        if regime_col not in table.frame.columns:
            raise ValueError(
                f"SiteBundle site {site!r} has no regime column {regime_col!r}. "
                "Loader bundles carry no regime column (see "
                "causal_grounding/loaders/contract.py): add one per site with "
                "'on' for randomized and 'idle' for observational rows, or set "
                "Naturalness(regime_col=...)"
            )
