"""
Naturalness grounding workflow.

This module learns empirical, site-level estimates of

    c_k(j, epsilon, tau), d_k(j, epsilon, tau)

from training sites that contain both observational (``F='idle'``) and RCT (``F='on'``) data.

The implementation is intentionally conservative:
- variables must be discrete;
- treatment and outcome must be binary;
- the Stage-1 PMF model is empirical, hierarchical-Dirichlet, XGBoost-based,
  or TabICLv2-based;
- the Stage-3 smoother is a Gaussian process with a custom kernel and simple optimizer.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from itertools import combinations, product
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

try:
    from .predictors import create_predictor
except ImportError:
    from predictors import create_predictor

try:
    from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, ConstantKernel, WhiteKernel
    from sklearn.gaussian_process import GaussianProcessRegressor
except ImportError:  # pragma: no cover - exercised only without sklearn installed
    Hyperparameter = None

    class Kernel:  # type: ignore[no-redef]
        """Fallback base class used when scikit-learn is unavailable."""

        pass

from scipy.optimize import minimize

from . import naturalness_statistics


def _logit(p: np.ndarray, clip: float = 1e-4) -> np.ndarray:
    """Apply a numerically stable logit transform to probabilities."""
    clipped = np.clip(p, clip, 1.0 - clip)
    return np.log(clipped / (1.0 - clipped))


def _inv_logit(x: np.ndarray) -> np.ndarray:
    """Map logit-scale values back to probabilities."""
    return 1.0 / (1.0 + np.exp(-x))


GP_FEATURE_MODE_THRESHOLDS = "thresholds"
GP_FEATURE_MODE_OBSERVED_STATS = "observed_stats"
SUPPORTED_GP_FEATURE_MODES = (
    GP_FEATURE_MODE_THRESHOLDS,
    GP_FEATURE_MODE_OBSERVED_STATS,
)
_GP_FEATURE_COLUMNS = {
    GP_FEATURE_MODE_THRESHOLDS: ("epsilon", "tau", "covariate_index"),
    GP_FEATURE_MODE_OBSERVED_STATS: ("i_deviation", "d_strength", "covariate_index"),
}


def _validate_gp_feature_mode(mode: str) -> str:
    """Return a normalized GP feature mode or raise for unsupported values."""
    normalized = str(mode)
    if normalized not in SUPPORTED_GP_FEATURE_MODES:
        raise ValueError(
            "gp_feature_mode must be one of "
            f"{SUPPORTED_GP_FEATURE_MODES}; got {mode!r}"
        )
    return normalized


def _gp_feature_columns(mode: str) -> Tuple[str, str, str]:
    """Return the three GP feature columns for a validated feature mode."""
    return _GP_FEATURE_COLUMNS[_validate_gp_feature_mode(mode)]


#: What the Stage-3 GP is fitted to. ``per_site`` regresses every triggering
#: source site's own tolerance at each grid point; ``envelope`` fits one value per
#: grid point, the maximum over triggering sites, approximating the tightest valid tolerance.
GP_TRAINING_TARGET_PER_SITE = "per_site"
GP_TRAINING_TARGET_ENVELOPE = "envelope"
SUPPORTED_GP_TRAINING_TARGETS = (
    GP_TRAINING_TARGET_PER_SITE,
    GP_TRAINING_TARGET_ENVELOPE,
)

#: Scale of the Stage-3 logit transform: ``note`` models ``logit(c)`` (the LP uses ``2c``),
#: ``tilde`` models ``logit(2c)``. Predictions and draws are always on the note scale.
GP_TARGET_SCALE_NOTE = "note"
GP_TARGET_SCALE_TILDE = "tilde"
SUPPORTED_GP_TARGET_SCALES = (GP_TARGET_SCALE_NOTE, GP_TARGET_SCALE_TILDE)
_GP_TARGET_SCALE_FACTORS = {GP_TARGET_SCALE_NOTE: 1.0, GP_TARGET_SCALE_TILDE: 2.0}

#: Upper end of the note-scale c/d tolerance range ``[0, 0.5]`` that clipped
#: GP predictions and draws are restricted to before they reach the LP.
GP_NOTE_SCALE_TOLERANCE_CAP = 0.5


def validate_gp_training_target(target: str) -> str:
    """Return a supported Stage-3 GP training target or raise ``ValueError``."""
    normalized = str(target)
    if normalized not in SUPPORTED_GP_TRAINING_TARGETS:
        raise ValueError(
            "gp_training_target must be one of "
            f"{SUPPORTED_GP_TRAINING_TARGETS}; got {target!r}"
        )
    return normalized


def validate_gp_target_scale(scale: str) -> str:
    """Return a supported Stage-3 logit scale or raise ``ValueError``."""
    normalized = str(scale)
    if normalized not in SUPPORTED_GP_TARGET_SCALES:
        raise ValueError(
            "gp_target_scale must be one of "
            f"{SUPPORTED_GP_TARGET_SCALES}; got {scale!r}"
        )
    return normalized


def make_lbfgs_optimizer(maxiter: int = 5000):
    def optimizer(obj_func, initial_theta, bounds):
        result = minimize(
            obj_func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": 1e-9,
                "gtol": 1e-6,
            },
        )
        return result.x, result.fun
    return optimizer


@dataclass
class NaturalnessGrid:
    """
    Grid configuration for the naturalness sweep.

    Attributes:
        epsilons: Values for the I_j(epsilon) naturalness trigger.
        taus: Values for the D_j(tau) treatment-strength trigger.
        covariates: Discrete covariate names defining Z.
        candidate_covariates: Optional subset of ``covariates`` eligible for
            naturalness testing (default: all). Other covariates stay in ``Z``,
            and hence in every conditioning set, but are never candidates.
        min_cell_count: Minimum support for observed Z_minus strata.
        max_z_minus_samples: Optional cap on Z_minus support values per site and
            covariate, for when the support of Z_minus is large.
        conditioning_covariates: Optional subset of ``covariates`` that every
            candidate is conditioned on, replacing ``Z_-j = covariates minus j``.
            Candidate ``j`` then lives in its own *candidate state* (the
            conditioning covariates plus ``j``, in ``covariates`` order), where
            its triggers, tolerances and target LP are computed. ``None`` keeps
            one full state shared by every candidate.
    """

    epsilons: List[float]
    taus: List[float]
    covariates: List[str]
    candidate_covariates: Optional[List[str]] = None
    min_cell_count: int = 5
    max_z_minus_samples: Optional[int] = None
    conditioning_covariates: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate that the naturalness grid is usable."""
        if not self.epsilons:
            raise ValueError("epsilons must contain at least one value")
        if not self.taus:
            raise ValueError("taus must contain at least one value")
        if not self.covariates:
            raise ValueError("covariates must contain at least one column")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("covariates must not contain duplicates")
        if self.candidate_covariates is None:
            self.candidate_covariates = list(self.covariates)
        if not self.candidate_covariates:
            raise ValueError("candidate_covariates must contain at least one column")
        if len(set(self.candidate_covariates)) != len(self.candidate_covariates):
            raise ValueError("candidate_covariates must not contain duplicates")
        unknown_candidates = sorted(
            set(self.candidate_covariates) - set(self.covariates)
        )
        if unknown_candidates:
            raise ValueError(
                "candidate_covariates must be a subset of covariates; "
                f"unknown values: {unknown_candidates}"
            )
        if self.min_cell_count < 1:
            raise ValueError("min_cell_count must be >= 1")
        if self.conditioning_covariates is not None:
            self.conditioning_covariates = list(self.conditioning_covariates)
            if len(set(self.conditioning_covariates)) != len(
                self.conditioning_covariates
            ):
                raise ValueError("conditioning_covariates must not contain duplicates")
            unknown_conditioning = sorted(
                set(self.conditioning_covariates) - set(self.covariates)
            )
            if unknown_conditioning:
                raise ValueError(
                    "conditioning_covariates must be a subset of covariates; "
                    f"unknown values: {unknown_conditioning}"
                )

    @property
    def candidate_items(self) -> List[Tuple[int, str]]:
        """Return full-``Z`` indices and names for eligible candidates."""

        candidate_set = set(self.candidate_covariates or [])
        return [
            (index, covariate)
            for index, covariate in enumerate(self.covariates)
            if covariate in candidate_set
        ]

    @property
    def uses_candidate_states(self) -> bool:
        """Return whether candidates are conditioned on a declared subset."""

        return self.conditioning_covariates is not None

    def candidate_state(self, j: int) -> Tuple[List[str], int]:
        """
        Return the covariate state in which candidate ``j`` is evaluated.

        Without ``conditioning_covariates`` every candidate shares the full
        state ``covariates``, so ``Z_-j`` is every other covariate. With it,
        the state is the conditioning covariates plus candidate ``j``, kept in
        ``covariates`` order, and ``Z_-j`` is the conditioning set without
        ``j``.

        Args:
            j: Index of the candidate in ``covariates``.

        Returns:
            The ordered state covariates and the position of candidate ``j``
            inside that state.

        Raises:
            IndexError: If ``j`` does not index ``covariates``.
        """

        if not 0 <= int(j) < len(self.covariates):
            raise IndexError(f"Candidate index {j} is outside the covariate grid")
        if self.conditioning_covariates is None:
            return list(self.covariates), int(j)
        candidate = self.covariates[int(j)]
        conditioning = set(self.conditioning_covariates)
        state = [
            covariate
            for covariate in self.covariates
            if covariate in conditioning or covariate == candidate
        ]
        return state, state.index(candidate)

    @property
    def candidate_state_keys(self) -> List[Tuple[str, ...]]:
        """Return the distinct candidate states in candidate order."""

        keys: List[Tuple[str, ...]] = []
        for j, _covariate in self.candidate_items:
            key = tuple(self.candidate_state(j)[0])
            if key not in keys:
                keys.append(key)
        return keys


@dataclass
class NaturalnessObservation:
    """Single row of the stage-2 naturalness sweep."""

    site_id: str
    covariate: str
    covariate_index: int
    epsilon: float
    tau: float
    i_deviation: float
    d_strength: float
    trigger_fired: bool
    c_hat: Optional[float]
    d_hat: Optional[float]
    n_z_minus_evaluated: int
    warnings: str = ""


@dataclass
class NaturalnessTolerance:
    """
    Learned naturalness tolerances for one covariate/grid point.

    The values are note-scale quantities ``c(j, epsilon, tau)`` and
    ``d(j, epsilon, tau)``. They summarize learned response-type stability for
    a triggered naturalness/relevance row. They are not expanded into
    downstream LP right-hand sides here; solver-facing code converts them to
    ``2c`` and ``2d``.
    """

    covariate: str
    epsilon: float
    tau: float
    c: float
    d: float
    c_source: str
    d_source: str
    c_std_logit: Optional[float] = None
    d_std_logit: Optional[float] = None
    upper_confidence_level: Optional[float] = None
    upper_critical_value: Optional[float] = None
    upper_family_size: Optional[int] = None
    upper_n_samples: Optional[int] = None


class BaseNaturalnessPMFModel(ABC):
    """
    Interface for Stage-1 probability models used by ``NaturalnessGrounder``.

    A backend can be empirical, XGBoost-based, TabICLv2-based, or any other model
    as long as it exposes these probability queries. Stage 2 only talks to this
    interface, which keeps the naturalness sweep independent of how the PMFs are
    estimated.
    """

    diagnostics: List[str]

    @abstractmethod
    def p_idle_x_given_z(self, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(X=x | z; idle)."""

    @abstractmethod
    def p_idle_y_given_xz(self, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(Y=y | X=x, z; idle)."""

    @abstractmethod
    def p_star_y_given_z(self, star: int, y: int, z: Tuple[Any, ...]) -> float:
        """Return P(Y(star)=y | z)."""

    @abstractmethod
    def p_star_y_given_xz(self, star: int, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(Y(star)=y | X=x, z; idle)."""


class EmpiricalSitePMFModels(BaseNaturalnessPMFModel):
    """
    Empirical PMF query layer for one site.

    The object exposes the probability queries required by the naturalness
    algorithm. It uses Laplace smoothing and records diagnostics when the
    off-treatment formula is unstable.
    """

    def __init__(
        self,
        site_id: str,
        data: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        outcome: str,
        regime_col: str = "F",
        alpha: float = 0.5,
        min_overlap: float = 0.02,
        idle_value: str = "idle",
        on_value: str = "on",
    ) -> None:
        """Initialize one empirical PMF backend for a single site."""
        self.site_id = site_id
        self.data = data.copy()
        self.covariates = list(covariates)
        self.treatment = treatment
        self.outcome = outcome
        self.regime_col = regime_col
        self.alpha = float(alpha)
        self.min_overlap = float(min_overlap)
        self.idle_value = idle_value
        self.on_value = on_value
        self.diagnostics: List[str] = []
        self._diagnostic_set: set = set()

        missing = [
            col for col in self.covariates + [treatment, outcome, regime_col]
            if col not in self.data.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns for site {site_id}: {missing}")

        self.idle = self.data[self.data[regime_col] == idle_value].copy()
        self.on = self.data[self.data[regime_col] == on_value].copy()

        if self.idle.empty:
            raise ValueError(f"Site {site_id} has no {regime_col}={idle_value!r} rows")
        if self.on.empty:
            raise ValueError(f"Site {site_id} has no {regime_col}={on_value!r} rows")

        if not naturalness_statistics.is_binary_series(self.data[treatment]):
            raise ValueError(f"Treatment column {treatment!r} must be binary")
        if not naturalness_statistics.is_binary_series(self.data[outcome]):
            raise ValueError(f"Outcome column {outcome!r} must be binary")

    def _record_diagnostic(self, message: str) -> None:
        """Store a diagnostic message once per site model."""
        if message not in self._diagnostic_set:
            self._diagnostic_set.add(message)
            self.diagnostics.append(message)

    def _z_mask(self, data: pd.DataFrame, z: Tuple[Any, ...]) -> pd.Series:
        """Return rows whose covariate tuple exactly matches ``z``."""
        z = naturalness_statistics.as_tuple(z)
        if len(z) != len(self.covariates):
            raise ValueError(
                f"Expected z with {len(self.covariates)} values, got {len(z)}"
            )

        mask = pd.Series(True, index=data.index)
        for col, value in zip(self.covariates, z):
            mask &= data[col].eq(value)
        return mask

    def _smoothed_binary_prob(self, n_success: int, n_total: int) -> float:
        """Estimate a binary probability with symmetric Laplace smoothing."""
        return float((n_success + self.alpha) / (n_total + 2.0 * self.alpha))

    def p_idle_x_given_z(self, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(X=x | z; idle)."""
        mask_z = self._z_mask(self.idle, z)
        mask_x = self.idle[self.treatment].eq(x)
        return self._smoothed_binary_prob(int((mask_z & mask_x).sum()), int(mask_z.sum()))

    def p_idle_y_given_xz(self, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(Y=y | X=x, z; idle)."""
        mask_z = self._z_mask(self.idle, z)
        mask_xz = mask_z & self.idle[self.treatment].eq(x)
        mask_y = self.idle[self.outcome].eq(y)
        return self._smoothed_binary_prob(int((mask_xz & mask_y).sum()), int(mask_xz.sum()))

    def p_do_y_given_z(self, star: int, y: int, z: Tuple[Any, ...]) -> float:
        """
        Return ``P(Y(star)=y | Z=z)`` from randomized ``F='on'`` rows.

        In the controlled regime, rows with treatment ``X=star`` identify the
        interventional distribution:

            P(Y(star)=y | Z=z) = P(Y=y | X=star, Z=z; F='on').
        """
        mask_z = self._z_mask(self.on, z)
        mask_xz = mask_z & self.on[self.treatment].eq(star)
        mask_y = self.on[self.outcome].eq(y)
        return self._smoothed_binary_prob(int((mask_xz & mask_y).sum()), int(mask_xz.sum()))

    def p_star_y_given_z(self, star: int, y: int, z: Tuple[Any, ...]) -> float:
        """
        Return the marginal potential-outcome probability ``P(Y(star)=y | Z=z)``.

        This is the quantity compared by the learned ``d`` naturalness
        tolerance after the target row ``(j, epsilon, tau)`` fires.
        """
        return self.p_do_y_given_z(star, y, z)

    def p_star_y_given_xz(self, star: int, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """
        Return P(Y(star)=y | X=x, z; idle).

        If ``x == star``, consistency identifies the probability from idle data:

            P(Y(star)=y | X=star,Z=z; idle) = P(Y=y | X=star,Z=z; idle).

        If ``x != star``, the off-treatment cell is recovered from the
        decomposition:

            P(Y(star)=y | Z=z)
              = sum_x P(Y(star)=y | X=x,Z=z; idle) P(X=x | Z=z; idle).

        Solving for the unknown off-treatment term gives:

            [P(Y(star)=y | z)
             - P(Y=y | X=star,z; idle) P(X=star | z; idle)]
            / P(X=x | z; idle)
        """
        if int(x) == int(star):
            return self.p_idle_y_given_xz(y, x, z)

        p_star_y_z = self.p_star_y_given_z(star, y, z)
        p_y_given_star_z = self.p_idle_y_given_xz(y, star, z)
        p_star_given_z = self.p_idle_x_given_z(star, z)
        p_x_given_z = self.p_idle_x_given_z(x, z)

        if p_x_given_z < self.min_overlap:
            self._record_diagnostic(
                f"poor overlap for star={star}, x={x}, z={z}; "
                f"P(X=x|z)={p_x_given_z:.4f}"
            )
            print(f"WARNING: Site {self.site_id} has poor overlap for star={star}, x={x}, z={z}; "
                  f"P(X=x|z)={p_x_given_z:.4f}")
        raw = (
            p_star_y_z - p_y_given_star_z * p_star_given_z
        ) / max(p_x_given_z, self.min_overlap)
        clipped = naturalness_statistics.clip_probability(raw)

        if not np.isclose(raw, clipped):
            self._record_diagnostic(
                f"clipped p_star_y_given_xz from {raw:.3f} to {clipped:.3f} "
                f"for star={star}, y={y}, x={x}, z={z}"
            )

        return clipped


class PredictorBackedSitePMFModels(BaseNaturalnessPMFModel):
    """
    Predictor-backed Stage-1 PMF query layer for one site.

    This adapter fits any ``causal_grounding.predictors`` backend that exposes
    the package predictor API and turns it into the PMF queries required by the
    naturalness sweep:

    - an idle-regime predictor for ``P(X | Z; idle)`` and
      ``P(Y | X,Z; idle)``;
    - an on-regime outcome predictor for ``P(Y | Z, do(X=star))``.

    The off-treatment query ``p_star_y_given_xz`` is still computed here by
    solving the two-treatment decomposition of ``P(Y(star)=y | Z=z)``, so
    overlap diagnostics stay local to the naturalness backend. Concrete
    subclasses choose the nuisance learner by setting ``predictor_method``.
    """

    predictor_method = ""
    predictor_defaults: Dict[str, Any] = {}

    def __init__(
        self,
        site_id: str,
        data: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        outcome: str,
        regime_col: str = "F",
        alpha: float = 0.5,
        min_overlap: float = 0.02,
        idle_value: str = "idle",
        on_value: str = "on",
        predictor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit predictor-backed models for one site's Stage-1 PMF queries."""
        self.site_id = site_id
        self.data = data.copy()
        self.covariates = list(covariates)
        self.treatment = treatment
        self.outcome = outcome
        self.regime_col = regime_col
        self.alpha = float(alpha)
        self.min_overlap = float(min_overlap)
        self.idle_value = idle_value
        self.on_value = on_value
        self.predictor_kwargs = dict(predictor_kwargs or {})
        for key, value in self.predictor_defaults.items():
            self.predictor_kwargs.setdefault(key, value)
        self.diagnostics: List[str] = []
        self._diagnostic_set: set = set()

        missing = [
            col for col in self.covariates + [treatment, outcome, regime_col]
            if col not in self.data.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns for site {site_id}: {missing}")

        self.idle = self.data[self.data[regime_col] == idle_value].copy()
        self.on = self.data[self.data[regime_col] == on_value].copy()

        if self.idle.empty:
            raise ValueError(f"Site {site_id} has no {regime_col}={idle_value!r} rows")
        if self.on.empty:
            raise ValueError(f"Site {site_id} has no {regime_col}={on_value!r} rows")

        if not naturalness_statistics.is_binary_series(self.data[treatment]):
            raise ValueError(f"Treatment column {treatment!r} must be binary")
        if not naturalness_statistics.is_binary_series(self.data[outcome]):
            raise ValueError(f"Outcome column {outcome!r} must be binary")

        self.feature_columns_ = self._fit_feature_columns()
        self.idle_predictor_ = self._fit_predictor(self.idle, fit_propensity=True)
        self.on_predictor_ = self._fit_predictor(self.on, fit_propensity=False)
        # Aliases of the fitted predictors under their ``*_model_`` names.
        self.idle_model_ = self.idle_predictor_
        self.on_model_ = self.on_predictor_

    def _record_diagnostic(self, message: str) -> None:
        """Store a diagnostic message once per site model."""
        if message not in self._diagnostic_set:
            self._diagnostic_set.add(message)
            self.diagnostics.append(message)

    def _fit_feature_columns(self) -> List[str]:
        """Create a stable one-hot feature schema for the site's covariates."""
        encoded = pd.get_dummies(self.data[self.covariates], dummy_na=False)
        if encoded.empty:
            raise ValueError(f"Site {self.site_id} produced no encoded covariate columns")
        return encoded.columns.tolist()

    def _encode_covariates(self, data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode covariates and align to the fitted feature schema."""
        encoded = pd.get_dummies(data[self.covariates], dummy_na=False)
        return encoded.reindex(columns=self.feature_columns_, fill_value=0).astype(float)

    def _make_z_frame(self, z: Tuple[Any, ...]) -> pd.DataFrame:
        """Convert a covariate tuple into one encoded feature row."""
        z = naturalness_statistics.as_tuple(z)
        if len(z) != len(self.covariates):
            raise ValueError(
                f"Expected z with {len(self.covariates)} values, got {len(z)}"
            )
        raw = pd.DataFrame([{col: value for col, value in zip(self.covariates, z)}])
        return self._encode_covariates(raw)

    def _fit_predictor(self, data: pd.DataFrame, fit_propensity: bool) -> Any:
        """Fit a package predictor on encoded site data."""

        encoded = self._encode_covariates(data)
        fit_data = encoded.copy()
        fit_data[self.treatment] = data[self.treatment].to_numpy()
        fit_data[self.outcome] = data[self.outcome].to_numpy()

        predictor = create_predictor(self.predictor_method, **self.predictor_kwargs)
        if fit_propensity:
            predictor.fit_propensity(fit_data, self.treatment, self.feature_columns_)
        predictor.fit_outcome(
            fit_data,
            outcome=self.outcome,
            treatment=self.treatment,
            covariates=self.feature_columns_,
        )
        return predictor

    def _prob_y1(self, predictor: Any, z: Tuple[Any, ...], x: int) -> float:
        """Predict P(Y=1 | X=x,Z=z) from a fitted package predictor."""
        encoded = self._make_z_frame(z)
        return naturalness_statistics.clip_probability(
            float(predictor.predict_outcome(encoded, int(x))[0])
        )

    def p_idle_x_given_z(self, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(X=x | z; idle)."""
        encoded = self._make_z_frame(z)
        p_x1 = naturalness_statistics.clip_probability(
            float(self.idle_predictor_.predict_propensity(encoded)[0])
        )
        return p_x1 if int(x) == 1 else 1.0 - p_x1

    def p_idle_y_given_xz(self, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(Y=y | X=x, z; idle)."""
        p_y1 = self._prob_y1(self.idle_predictor_, z, x)
        return p_y1 if int(y) == 1 else 1.0 - p_y1

    def p_do_y_given_z(self, star: int, y: int, z: Tuple[Any, ...]) -> float:
        """
        Return ``P(Y(star)=y | Z=z)`` estimated from ``F='on'`` rows.

        The on-regime outcome model supplies the controlled-regime proxy:

            P(Y(star)=y | Z=z) = P(Y=y | X=star,Z=z; F='on').
        """
        p_y1 = self._prob_y1(self.on_predictor_, z, star)
        return p_y1 if int(y) == 1 else 1.0 - p_y1

    def p_star_y_given_z(self, star: int, y: int, z: Tuple[Any, ...]) -> float:
        """
        Return the marginal potential-outcome probability ``P(Y(star)=y | Z=z)``.

        This is the model-backed quantity used in learned ``d`` tolerance
        estimates.
        """
        return self.p_do_y_given_z(star, y, z)

    def p_star_y_given_xz(self, star: int, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """Return P(Y(star)=y | X=x, z; idle)."""
        if int(x) == int(star):
            return self.p_idle_y_given_xz(y, x, z)

        p_star_y_z = self.p_star_y_given_z(star, y, z)
        p_y_given_star_z = self.p_idle_y_given_xz(y, star, z)
        p_star_given_z = self.p_idle_x_given_z(star, z)
        p_x_given_z = self.p_idle_x_given_z(x, z)

        if p_x_given_z < self.min_overlap:
            self._record_diagnostic(
                f"poor overlap for star={star}, x={x}, z={z}; "
                f"P(X=x|z)={p_x_given_z:.4f}"
            )

        raw = (
            p_star_y_z - p_y_given_star_z * p_star_given_z
        ) / max(p_x_given_z, self.min_overlap)
        clipped = naturalness_statistics.clip_probability(raw)

        if not np.isclose(raw, clipped):
            self._record_diagnostic(
                f"clipped p_star_y_given_xz from {raw:.3f} to {clipped:.3f} "
                f"for star={star}, y={y}, x={x}, z={z}"
            )

        return clipped


class XGBoostSitePMFModels(PredictorBackedSitePMFModels):
    """XGBoost-backed Stage-1 PMF query layer for one site."""

    predictor_method = "xgboost"
    predictor_defaults = {"use_grid_search": False}


class TabICLv2SitePMFModels(PredictorBackedSitePMFModels):
    """
    TabICLv2-backed Stage-1 PMF query layer for one site.

    Representation caching is enabled while the three nuisance contexts are
    queried over every supported site stratum. Their finite probability tables
    are then retained and the heavyweight classifiers are released before the
    next source site is fitted. This prevents checkpoint copies from
    accumulating across all Stage-1 sites while preserving the exact PMF query
    interface. Callers can override the temporary cache mode through
    ``predictor_kwargs``.
    """

    predictor_method = "tabicl_v2"
    predictor_defaults = {"kv_cache": "repr"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Fit TabICLv2 contexts and materialize finite site probability tables.

        Args:
            *args: Positional arguments forwarded to
                ``PredictorBackedSitePMFModels``.
            **kwargs: Keyword arguments forwarded to
                ``PredictorBackedSitePMFModels``.
        """
        super().__init__(*args, **kwargs)
        self._materialize_probability_tables()
        self._release_tabicl_predictors()

    def _materialize_probability_tables(self) -> None:
        """Predict every supported site stratum before releasing TabICLv2."""
        raw_z = self._full_idle_covariate_grid()
        encoded_z = self._encode_covariates(raw_z)
        z_values = [
            naturalness_statistics.as_tuple(values)
            for values in raw_z.itertuples(index=False, name=None)
        ]

        propensity = np.asarray(
            self.idle_predictor_.predict_propensity(encoded_z),
            dtype=float,
        )
        idle_outcome = {
            x: np.asarray(
                self.idle_predictor_.predict_outcome(encoded_z, x),
                dtype=float,
            )
            for x in [0, 1]
        }
        randomized_outcome = {
            x: np.asarray(
                self.on_predictor_.predict_outcome(encoded_z, x),
                dtype=float,
            )
            for x in [0, 1]
        }

        self._p_idle_x1_by_z = {
            z: naturalness_statistics.clip_probability(value)
            for z, value in zip(z_values, propensity)
        }
        self._p_idle_y1_by_xz = {
            (x, z): naturalness_statistics.clip_probability(idle_outcome[x][index])
            for index, z in enumerate(z_values)
            for x in [0, 1]
        }
        self._p_on_y1_by_xz = {
            (x, z): naturalness_statistics.clip_probability(
                randomized_outcome[x][index]
            )
            for index, z in enumerate(z_values)
            for x in [0, 1]
        }

    def _full_idle_covariate_grid(self) -> pd.DataFrame:
        """
        Return the Cartesian grid of values supported by each idle covariate.

        Stage 2 holds ``Z_-j`` fixed while changing ``Z_j``. The resulting
        joint tuple need not occur as a row in the site data, but a model-based
        backend must still extrapolate to it. Materializing the product before
        releasing TabICLv2 preserves that behavior.

        Returns:
            DataFrame containing every combination of per-covariate values
            observed in the site's idle regime.
        """
        supports = [
            naturalness_statistics.safe_sorted(
                self.idle[covariate].dropna().unique().tolist()
            )
            for covariate in self.covariates
        ]
        empty_covariates = [
            covariate
            for covariate, support in zip(self.covariates, supports)
            if not support
        ]
        if empty_covariates:
            raise ValueError(
                f"Site {self.site_id} has no idle support for {empty_covariates}"
            )
        return pd.DataFrame(
            product(*supports),
            columns=self.covariates,
        )

    def _release_tabicl_predictors(self) -> None:
        """Release fitted TabICLv2 contexts and reclaim cached CUDA memory."""
        self.idle_predictor_ = None
        self.on_predictor_ = None
        self.idle_model_ = None
        self.on_model_ = None
        self.predictors_released_ = True
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cached_probability(
        self,
        table: Dict[Any, float],
        key: Any,
    ) -> float:
        """
        Read one materialized probability with a useful support error.

        Args:
            table: Cached probability mapping.
            key: Stratum key requested by a PMF method.

        Returns:
            Cached probability.

        Raises:
            ValueError: If the requested stratum was absent during fitting.
        """
        try:
            return table[key]
        except KeyError as exc:
            raise ValueError(
                f"Site {self.site_id} has no fitted TabICLv2 stratum for {key!r}"
            ) from exc

    def p_idle_x_given_z(self, x: int, z: Tuple[Any, ...]) -> float:
        """Return cached ``P(X=x | z; idle)``."""
        z = naturalness_statistics.as_tuple(z)
        p_x1 = self._cached_probability(self._p_idle_x1_by_z, z)
        return p_x1 if int(x) == 1 else 1.0 - p_x1

    def p_idle_y_given_xz(self, y: int, x: int, z: Tuple[Any, ...]) -> float:
        """Return cached ``P(Y=y | X=x,z; idle)``."""
        z = naturalness_statistics.as_tuple(z)
        p_y1 = self._cached_probability(
            self._p_idle_y1_by_xz,
            (int(x), z),
        )
        return p_y1 if int(y) == 1 else 1.0 - p_y1

    def p_do_y_given_z(self, star: int, y: int, z: Tuple[Any, ...]) -> float:
        """Return cached randomized ``P(Y(star)=y | z)``."""
        z = naturalness_statistics.as_tuple(z)
        p_y1 = self._cached_probability(
            self._p_on_y1_by_xz,
            (int(star), z),
        )
        return p_y1 if int(y) == 1 else 1.0 - p_y1


class HierarchicalDirichletSitePMFModels(PredictorBackedSitePMFModels):
    """
    Hierarchical-Dirichlet Stage-1 PMF query layer for one site.

    The adapter shares the encoded site-data path used by the model-based
    Stage-1 backends, but its underlying predictor estimates finite covariate
    cells with empirical-Bayes Dirichlet shrinkage instead of fitting a
    parametric smoother.
    """

    predictor_method = "hierarchical_dirichlet"


Stage1Backend = Union[
    str,
    Type[BaseNaturalnessPMFModel],
    Callable[..., BaseNaturalnessPMFModel],
]


NATURALNESS_STAGE1_BACKENDS: Dict[str, Callable[..., BaseNaturalnessPMFModel]] = {
    "empirical": EmpiricalSitePMFModels,
    "xgboost": XGBoostSitePMFModels,
    "hierarchical_dirichlet": HierarchicalDirichletSitePMFModels,
    "tabicl_v2": TabICLv2SitePMFModels,
}


def register_naturalness_stage1_backend(
    name: str,
    backend: Callable[..., BaseNaturalnessPMFModel],
) -> None:
    """Register a Stage-1 PMF backend for use by name."""
    if not name:
        raise ValueError("backend name must be non-empty")
    if not callable(backend):
        raise TypeError("backend must be callable")
    NATURALNESS_STAGE1_BACKENDS[name] = backend


def _resolve_stage1_backend(
    backend: Stage1Backend,
) -> Tuple[str, Callable[..., BaseNaturalnessPMFModel]]:
    """Resolve a registered backend name or callable into a backend factory."""
    if isinstance(backend, str):
        try:
            return backend, NATURALNESS_STAGE1_BACKENDS[backend]
        except KeyError as exc:
            available = ", ".join(sorted(NATURALNESS_STAGE1_BACKENDS))
            raise ValueError(
                f"Unknown stage1_backend {backend!r}. Available backends: {available}"
            ) from exc

    if not callable(backend):
        raise TypeError("stage1_backend must be a registered name or callable")

    return getattr(backend, "__name__", backend.__class__.__name__), backend


def _validate_stage1_model(model: Any) -> BaseNaturalnessPMFModel:
    """Check that a Stage-1 backend object exposes the required PMF queries."""
    required_methods = [
        "p_idle_x_given_z",
        "p_idle_y_given_xz",
        "p_star_y_given_z",
        "p_star_y_given_xz",
    ]
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            "Stage-1 backend must return an object with methods "
            f"{required_methods}; missing {missing}"
        )
    if not hasattr(model, "diagnostics"):
        model.diagnostics = []
    return model


class NaturalnessGPKernel(Kernel):
    """
    Gaussian-process kernel over the naturalness grid coordinates.

    The kernel acts on two continuous naturalness coordinates plus
    ``covariate_index``. By default those coordinates are ``epsilon`` and
    ``tau``; the optional observed-statistics GP mode instead supplies
    ``i_deviation`` and ``d_strength``. The kernel implements the non-amplitude
    part of

    ``lambda_0 * exp(-(epsilon-epsilon')^2/lambda_1
                    -(tau-tau')^2/lambda_2
                    -1[j != j']/lambda_3)``.

    ``lambda_0`` is intentionally handled by scikit-learn's
    ``ConstantKernel`` so the amplitude can be optimized using the standard
    kernel composition API.
    """

    def __init__(
        self,
        epsilon_lambda: float = 0.01,
        tau_lambda: float = 0.01,
        covariate_lambda: float = 1.0,
        epsilon_lambda_bounds: Tuple[float, float] = (1e-5, 1e2),
        tau_lambda_bounds: Tuple[float, float] = (1e-5, 1e2),
        covariate_lambda_bounds: Tuple[float, float] = (1e-3, 1e3),
    ) -> None:
        """Configure the positive kernel denominators and optimizer bounds."""
        self.epsilon_lambda = epsilon_lambda
        self.tau_lambda = tau_lambda
        self.covariate_lambda = covariate_lambda
        self.epsilon_lambda_bounds = epsilon_lambda_bounds
        self.tau_lambda_bounds = tau_lambda_bounds
        self.covariate_lambda_bounds = covariate_lambda_bounds

    @property
    def hyperparameter_epsilon_lambda(self) -> Any:
        """Return optimizer metadata for the epsilon denominator."""
        if Hyperparameter is None:
            raise RuntimeError("scikit-learn is required to use NaturalnessGPKernel")
        return Hyperparameter("epsilon_lambda", "numeric", self.epsilon_lambda_bounds)

    @property
    def hyperparameter_tau_lambda(self) -> Any:
        """Return optimizer metadata for the tau denominator."""
        if Hyperparameter is None:
            raise RuntimeError("scikit-learn is required to use NaturalnessGPKernel")
        return Hyperparameter("tau_lambda", "numeric", self.tau_lambda_bounds)

    @property
    def hyperparameter_covariate_lambda(self) -> Any:
        """Return optimizer metadata for the categorical covariate penalty."""
        if Hyperparameter is None:
            raise RuntimeError("scikit-learn is required to use NaturalnessGPKernel")
        return Hyperparameter("covariate_lambda", "numeric", self.covariate_lambda_bounds)

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Evaluate the covariance matrix, optionally with log-parameter gradients.

        Args:
            X: Left feature matrix with columns ``epsilon``, ``tau``, and
                ``covariate_index``.
            Y: Optional right feature matrix. Gradients are only supported for
                the symmetric ``Y is None`` case required by scikit-learn.
            eval_gradient: Whether to return derivatives with respect to the
                log-transformed kernel hyperparameters.

        Returns:
            Kernel matrix, or ``(kernel_matrix, gradient)`` when
            ``eval_gradient=True``.
        """
        X = np.atleast_2d(X).astype(float)
        if Y is None:
            Y = X
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y = np.atleast_2d(Y).astype(float)

        epsilon_sq = (X[:, 0, None] - Y[None, :, 0]) ** 2
        tau_sq = (X[:, 1, None] - Y[None, :, 1]) ** 2
        covariate_mismatch = (X[:, 2, None] != Y[None, :, 2]).astype(float)

        epsilon_lambda = float(self.epsilon_lambda)
        tau_lambda = float(self.tau_lambda)
        covariate_lambda = float(self.covariate_lambda)

        K = np.exp(
            -epsilon_sq / epsilon_lambda
            - tau_sq / tau_lambda
            - covariate_mismatch / covariate_lambda
        )

        if not eval_gradient:
            return K

        gradients = []
        if not self.hyperparameter_epsilon_lambda.fixed:
            gradients.append(K * epsilon_sq / epsilon_lambda)
        if not self.hyperparameter_tau_lambda.fixed:
            gradients.append(K * tau_sq / tau_lambda)
        if not self.hyperparameter_covariate_lambda.fixed:
            gradients.append(K * covariate_mismatch / covariate_lambda)

        if gradients:
            return K, np.dstack(gradients)
        return K, np.empty((*K.shape, 0))

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Return the diagonal of the kernel matrix for ``X``."""
        return np.ones(np.atleast_2d(X).shape[0])

    def is_stationary(self) -> bool:
        """Return whether the kernel is stationary in the strict numeric sense."""
        return False

    def __repr__(self) -> str:
        """Return a concise representation of the optimized kernel."""
        return (
            "NaturalnessGPKernel("
            f"epsilon_lambda={self.epsilon_lambda:.3g}, "
            f"tau_lambda={self.tau_lambda:.3g}, "
            f"covariate_lambda={self.covariate_lambda:.3g})"
        )


class NaturalnessGPSurface:
    """
    Stage-3 GP smoother for ``c_hat`` or ``d_hat`` observations.

    The model is fitted on the plug-in logit transform of the tolerance and
    uses ``NaturalnessGPKernel`` for either threshold features
    ``(epsilon, tau, covariate_index)`` or observed trigger-statistic features
    ``(i_deviation, d_strength, covariate_index)``. ``training_target`` selects
    the training rows (every triggering site, or the per-grid-point envelope)
    and ``target_scale`` the logit transform (of ``c`` or of ``c~ = 2c``); see
    :data:`SUPPORTED_GP_TRAINING_TARGETS` and :data:`SUPPORTED_GP_TARGET_SCALES`.
    Probability-scale outputs are always note-scale ``c``. This wrapper keeps
    scikit-learn optional: if scikit-learn is unavailable or too few
    observations are available, ``is_fitted`` remains false and callers can
    fall back to empirical averages.
    """

    def __init__(
        self,
        target_col: str,
        min_observations: int = 10,
        random_seed: Optional[int] = None,
        feature_mode: str = GP_FEATURE_MODE_THRESHOLDS,
        training_target: str = GP_TRAINING_TARGET_PER_SITE,
        target_scale: str = GP_TARGET_SCALE_NOTE,
    ) -> None:
        """Initialize a GP wrapper for one learned naturalness surface.

        Args:
            target_col: Stage-2 column fitted, ``c_hat`` or ``d_hat``.
            min_observations: Fewest training rows for which a fit is attempted.
            random_seed: Seed of the optimizer restarts and default draws.
            feature_mode: GP input coordinates, thresholds or observed statistics.
            training_target: ``per_site`` or ``envelope``; the envelope needs
                threshold features, since it is a maximum per grid point.
            target_scale: ``note`` (logit of ``c``) or ``tilde`` (logit of ``2c``).

        Raises:
            ValueError: If a setting is unsupported, or the envelope is combined
                with observed-statistic features.
        """
        self.target_col = target_col
        self.min_observations = min_observations
        self.random_seed = random_seed
        self.feature_mode = _validate_gp_feature_mode(feature_mode)
        self.feature_columns = _gp_feature_columns(self.feature_mode)
        self.training_target = validate_gp_training_target(training_target)
        self.target_scale = validate_gp_target_scale(target_scale)
        if (
            self.training_target == GP_TRAINING_TARGET_ENVELOPE
            and self.feature_mode != GP_FEATURE_MODE_THRESHOLDS
        ):
            raise ValueError(
                "gp_training_target='envelope' is a maximum per (epsilon, tau) "
                "grid point and needs gp_feature_mode='thresholds'"
            )
        self.model = None
        self.is_fitted = False
        self.status = "not_fitted"
        self.message = ""
        self.training_data_: Optional[pd.DataFrame] = None
        self.n_raw_training_observations_ = 0
        self.training_features_: Optional[np.ndarray] = None
        self.training_target_logit_: Optional[np.ndarray] = None
        self.optimized_kernel_: Optional[str] = None
        self.log_marginal_likelihood_: Optional[float] = None
        self.learned_hyperparameters_: Dict[str, Any] = {}

    def fit(self, observations: pd.DataFrame) -> "NaturalnessGPSurface":
        """Fit the GP surface from triggered Stage-2 tolerance observations."""
        triggered = observations[
            observations["trigger_fired"] & observations[self.target_col].notna()
        ].copy()
        self.n_raw_training_observations_ = int(len(triggered))
        triggered = self._prepare_training_rows(triggered)
        if self.training_target == GP_TRAINING_TARGET_ENVELOPE:
            triggered = self._envelope_training_rows(triggered)
        self.training_data_ = triggered

        if len(triggered) < self.min_observations:
            self.status = "insufficient_observations"
            self.message = (
                f"Need at least {self.min_observations} triggered observations "
                f"for {self.target_col}; got {len(triggered)}."
            )
            return self

        features = triggered[list(self.feature_columns)].to_numpy(dtype=float)
        target = self.to_logit(triggered[self.target_col].to_numpy(dtype=float))
        self.training_features_ = features
        self.training_target_logit_ = target

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * NaturalnessGPKernel()
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
        )

        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=20,
            random_state=self.random_seed,
            optimizer=make_lbfgs_optimizer(maxiter=2000)
        )
        model.fit(features, target)

        self.model = model
        self.is_fitted = True
        self.status = "fitted"
        self.optimized_kernel_ = str(model.kernel_)
        self.message = self.optimized_kernel_
        self.log_marginal_likelihood_ = float(model.log_marginal_likelihood_value_)
        self.learned_hyperparameters_ = self._extract_kernel_parameters(model.kernel_)
        return self

    def predict(
        self,
        epsilon: float,
        tau: float,
        covariate_index: int,
        scale: str = "probability",
        quantile: Optional[float] = None,
        i_deviation: Optional[float] = None,
        d_strength: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Predict a Stage-3 tolerance value.

        Args:
            epsilon: Naturalness trigger threshold.
            tau: Treatment-strength trigger threshold.
            covariate_index: Integer index of the covariate in the grid.
            scale: ``"probability"`` for c/d scale or ``"logit"`` for c^0/d^0.
            quantile: Optional posterior quantile on the logit scale. ``None``
                returns the posterior mean.
            i_deviation: Observed ``I_j`` statistic used only when
                ``feature_mode='observed_stats'``.
            d_strength: Observed ``D_j`` statistic used only when
                ``feature_mode='observed_stats'``.

        Returns:
            Predicted value on the requested scale and posterior standard
            deviation on the logit scale.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"GP surface for {self.target_col} is not fitted: {self.message}")
        if scale not in {"probability", "logit"}:
            raise ValueError("scale must be 'probability' or 'logit'")
        if quantile is not None and not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be strictly between 0 and 1")

        features = self._prediction_feature_array(
            epsilon=epsilon,
            tau=tau,
            covariate_index=covariate_index,
            i_deviation=i_deviation,
            d_strength=d_strength,
        )
        mean_logit, covariance_logit = self._posterior(
            features,
            include_observation_noise=False,
        )
        value_logit = float(mean_logit[0])
        std = float(np.sqrt(max(float(covariance_logit[0, 0]), 0.0)))

        if quantile is not None:
            value_logit = value_logit + NormalDist().inv_cdf(quantile) * std

        if scale == "logit":
            return value_logit, std
        return float(self.to_probability(np.array([value_logit]))[0]), std

    def sample(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        n_samples: int,
        scale: str = "probability",
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw joint posterior samples for several naturalness grid rows.

        Args:
            features: Rows with this surface's ``feature_columns``. In default
                threshold mode these are ``epsilon``, ``tau``, and
                ``covariate_index``; in observed-statistics mode these are
                ``i_deviation``, ``d_strength``, and ``covariate_index``. A
                NumPy array with those three columns is also accepted.
            n_samples: Number of joint posterior draws to return.
            scale: ``"probability"`` for note-scale probabilities or
                ``"logit"`` for raw GP draws, on this surface's
                ``target_scale``.
            random_seed: Optional seed for reproducible sampling. If omitted,
                the surface-level ``random_seed`` is reused.

        Returns:
            Array with shape ``(n_samples, n_rows)``. Each row is one joint
            posterior draw across all supplied grid rows.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"GP surface for {self.target_col} is not fitted: {self.message}")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if scale not in {"probability", "logit"}:
            raise ValueError("scale must be 'probability' or 'logit'")

        feature_array = self._coerce_sample_features(features)
        if feature_array.shape[0] == 0:
            return np.empty((n_samples, 0), dtype=float)

        seed = self.random_seed if random_seed is None else random_seed
        mean_logit, covariance_logit = self._posterior(
            feature_array,
            include_observation_noise=False,
        )
        draws = np.asarray(
            np.random.default_rng(seed).multivariate_normal(
                mean_logit,
                covariance_logit,
                size=n_samples,
                check_valid="raise",
            ),
            dtype=float,
        ).reshape(n_samples, feature_array.shape[0])
        if scale == "logit":
            return draws
        return self.to_probability(draws)

    @property
    def _scale_factor(self) -> float:
        """Return the factor taking note-scale ``c`` to the modelled quantity."""
        return _GP_TARGET_SCALE_FACTORS[self.target_scale]

    def to_probability(self, values_logit: np.ndarray) -> np.ndarray:
        """Map GP logit-scale values to note-scale tolerances ``c``.

        Args:
            values_logit: Values on this surface's logit scale, ``logit(c)`` or
                ``logit(2c)`` depending on ``target_scale``.

        Returns:
            Note-scale tolerances, in ``(0, 1)`` for ``note`` and ``(0, 0.5)``
            for ``tilde``.
        """
        return _inv_logit(np.asarray(values_logit, dtype=float)) / self._scale_factor

    def to_logit(self, values: np.ndarray) -> np.ndarray:
        """Map note-scale tolerances ``c`` to this surface's logit scale."""
        return _logit(self._scale_factor * np.asarray(values, dtype=float))

    def _envelope_training_rows(self, triggered: pd.DataFrame) -> pd.DataFrame:
        """Collapse triggered site rows to one maximum per grid point.

        Each ``(covariate_index, epsilon, tau)`` keeps the largest tolerance over
        the source sites that trigger there, which approximates the tightest
        valid tolerance ``max_k omega^k`` over source sites ``k``.

        Args:
            triggered: Finite triggered rows from :meth:`_prepare_training_rows`.

        Returns:
            One row per grid point with the feature columns, ``covariate`` when
            present, the maximum of ``target_col`` and ``n_triggering_sites``.
        """
        if triggered.empty:
            return triggered
        keys = [column for column in ("covariate_index", "epsilon", "tau") if column in triggered]
        aggregations: Dict[str, Any] = {self.target_col: (self.target_col, "max")}
        if "covariate" in triggered:
            aggregations["covariate"] = ("covariate", "first")
        site_column = "site_id" if "site_id" in triggered else self.target_col
        aggregations["n_triggering_sites"] = (site_column, "nunique" if site_column == "site_id" else "size")
        envelope = triggered.groupby(keys, as_index=False, sort=True).agg(**aggregations)
        envelope["trigger_fired"] = True
        return envelope

    def diagnostics(self) -> Dict[str, Any]:
        """Return Stage-3 fit diagnostics and learned kernel metadata."""
        return {
            "target_col": self.target_col,
            "status": self.status,
            "message": self.message,
            "n_training_observations": (
                0 if self.training_data_ is None else int(len(self.training_data_))
            ),
            "optimized_kernel": self.optimized_kernel_,
            "log_marginal_likelihood": self.log_marginal_likelihood_,
            "learned_hyperparameters": self.learned_hyperparameters_,
            "target_scale": "logit",
            "gp_target_scale": self.target_scale,
            "gp_training_target": self.training_target,
            "feature_mode": self.feature_mode,
            "feature_columns": list(self.feature_columns),
            "n_raw_training_observations": int(self.n_raw_training_observations_),
            "posterior_prediction_target": "latent_surface",
        }

    def _posterior(
        self,
        features: np.ndarray,
        *,
        include_observation_noise: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return joint GP posterior moments on the fitted logit scale.

        Training covariance always includes the fitted ``WhiteKernel`` noise.
        Latent-surface prediction removes that white-noise component from the
        test covariance while retaining it in the conditioning covariance.
        This distinction prevents new independent measurement noise from
        dominating posterior tolerance draws.

        Args:
            features: Numeric feature matrix at which to evaluate the posterior.
            include_observation_noise: Whether test covariance should include
                fitted white observation noise.

        Returns:
            Posterior mean vector and covariance matrix on the original,
            normalized-back logit target scale.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                f"GP surface for {self.target_col} is not fitted: {self.message}"
            )

        feature_array = np.atleast_2d(features).astype(float)
        if include_observation_noise:
            mean, covariance = self.model.predict(feature_array, return_cov=True)
            return np.asarray(mean, dtype=float), self._stabilize_covariance(
                np.asarray(covariance, dtype=float)
            )

        signal_kernel = self._signal_kernel()
        cross_covariance = signal_kernel(feature_array, self.model.X_train_)
        normalized_mean = cross_covariance @ self.model.alpha_
        triangular_solution = solve_triangular(
            self.model.L_,
            cross_covariance.T,
            lower=True,
            check_finite=False,
        )
        normalized_covariance = (
            signal_kernel(feature_array)
            - triangular_solution.T @ triangular_solution
        )

        target_mean = float(np.asarray(self.model._y_train_mean).reshape(-1)[0])
        target_std = float(np.asarray(self.model._y_train_std).reshape(-1)[0])
        mean = np.asarray(normalized_mean, dtype=float) * target_std + target_mean
        covariance = np.asarray(normalized_covariance, dtype=float) * target_std**2
        return mean, self._stabilize_covariance(covariance)

    def _signal_kernel(self) -> Any:
        """Return optimized covariance kernel without white observation noise."""
        if self.model is None:
            raise RuntimeError("GP surface is not fitted")
        kernel = self.model.kernel_
        if isinstance(getattr(kernel, "k2", None), WhiteKernel):
            return kernel.k1
        if isinstance(getattr(kernel, "k1", None), WhiteKernel):
            return kernel.k2
        raise RuntimeError(
            "Expected optimized Stage-3 kernel to contain one WhiteKernel "
            "component."
        )

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Symmetrize covariance and remove tiny negative numerical eigenvalues."""
        stabilized = 0.5 * (covariance + covariance.T)
        if stabilized.size == 0:
            return stabilized
        minimum_eigenvalue = float(np.linalg.eigvalsh(stabilized).min())
        if minimum_eigenvalue < 0.0:
            stabilized = stabilized + np.eye(stabilized.shape[0]) * (
                -minimum_eigenvalue + 1e-12
            )
        return stabilized

    def _prepare_training_rows(self, triggered: pd.DataFrame) -> pd.DataFrame:
        """Return finite, mode-specific GP training rows for one tolerance surface."""
        missing = [
            column
            for column in (*self.feature_columns, self.target_col)
            if column not in triggered.columns
        ]
        if missing:
            raise ValueError(f"Missing required GP training columns: {missing}")

        prepared = triggered.copy()
        numeric_columns = list(dict.fromkeys([*self.feature_columns, self.target_col]))
        numeric = prepared[numeric_columns].apply(pd.to_numeric, errors="coerce")
        finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        prepared = prepared.loc[finite_mask].copy()
        if prepared.empty:
            return prepared
        for column in numeric_columns:
            prepared[column] = numeric.loc[prepared.index, column].astype(float)

        if self.feature_mode != GP_FEATURE_MODE_OBSERVED_STATS:
            return prepared

        dedupe_columns = [
            column
            for column in (
                "site_id",
                "covariate",
                *self.feature_columns,
            )
            if column in prepared.columns
        ]
        if not dedupe_columns:
            return prepared
        aggregations = {
            column: "first"
            for column in prepared.columns
            if column not in set(dedupe_columns) | {self.target_col}
        }
        aggregations[self.target_col] = "mean"
        return prepared.groupby(
            dedupe_columns,
            as_index=False,
            dropna=False,
        ).agg(aggregations)

    def _prediction_feature_array(
        self,
        *,
        epsilon: float,
        tau: float,
        covariate_index: int,
        i_deviation: Optional[float],
        d_strength: Optional[float],
    ) -> np.ndarray:
        """Build the single-row feature matrix for one GP prediction."""
        if self.feature_mode == GP_FEATURE_MODE_THRESHOLDS:
            values = (epsilon, tau, covariate_index)
        else:
            if i_deviation is None or d_strength is None:
                raise ValueError(
                    "i_deviation and d_strength are required when "
                    "gp_feature_mode='observed_stats'"
                )
            values = (i_deviation, d_strength, covariate_index)
        features = np.array([values], dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("GP prediction features must be finite")
        return features

    def _coerce_sample_features(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Return a numeric ``(n_rows, 3)`` feature matrix for GP sampling."""
        if isinstance(features, pd.DataFrame):
            required = list(self.feature_columns)
            missing = [column for column in required if column not in features.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            feature_array = features[required].to_numpy(dtype=float)
        else:
            feature_array = np.asarray(features, dtype=float)
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)

        if feature_array.ndim != 2 or feature_array.shape[1] != 3:
            raise ValueError(
                "features must have shape (n_rows, 3) with columns "
                f"{', '.join(self.feature_columns)}"
            )
        if not np.isfinite(feature_array).all():
            raise ValueError("GP sample features must be finite")
        return feature_array

    @staticmethod
    def _extract_kernel_parameters(kernel: Any) -> Dict[str, Any]:
        """Extract numeric learned kernel parameters from a fitted sklearn kernel."""
        params: Dict[str, Any] = {}
        for name, value in kernel.get_params(deep=True).items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                params[name] = float(value)
            elif isinstance(value, np.ndarray):
                params[name] = value.astype(float).tolist()
        return params


class NaturalnessGrounder:
    """
    Learn empirical naturalness estimates across sites and grid values.

    Stage 1 is intentionally pluggable: the default ``"empirical"`` backend
    uses discrete cell counts, while custom backends can expose the same PMF
    query interface with learned predictors.

    Example:
        grid = NaturalnessGrid(
            epsilons=[0.01, 0.025, 0.05],
            taus=[0.05, 0.10],
            covariates=["Z1", "Z2"],
        )
        grounder = NaturalnessGrounder(grid, stage1_backend="empirical")
        grounder.fit(training_data, treatment="X", outcome="Y")
        observations = grounder.observations_
        c = grounder.predict_c("Z1", epsilon=0.05, tau=0.10)
    """

    def __init__(
        self,
        grid: NaturalnessGrid,
        alpha: float = 0.5,
        min_overlap: float = 0.02,
        random_seed: Optional[int] = None,
        fit_gp: bool = True,
        idle_value: str = "idle",
        on_value: str = "on",
        stage1_backend: Stage1Backend = "empirical",
        stage1_kwargs: Optional[Dict[str, Any]] = None,
        gp_feature_mode: str = GP_FEATURE_MODE_THRESHOLDS,
        gp_training_target: str = GP_TRAINING_TARGET_PER_SITE,
        gp_target_scale: str = GP_TARGET_SCALE_NOTE,
    ) -> None:
        """Configure the naturalness learner and its Stage-1 backend.

        ``gp_training_target`` and ``gp_target_scale`` set what the Stage-3
        surfaces are fitted to and on which logit scale
        (:class:`NaturalnessGPSurface`); the defaults fit every triggering
        site's tolerance on the logit scale of ``c``.
        """
        self.grid = grid
        self.alpha = alpha
        self.min_overlap = min_overlap
        self.random_seed = random_seed
        self.fit_gp = fit_gp
        self.idle_value = idle_value
        self.on_value = on_value
        self.stage1_backend = stage1_backend
        self.stage1_kwargs = dict(stage1_kwargs or {})
        self.gp_feature_mode = _validate_gp_feature_mode(gp_feature_mode)
        self.gp_training_target = validate_gp_training_target(gp_training_target)
        self.gp_target_scale = validate_gp_target_scale(gp_target_scale)
        self.stage1_backend_name_, self.stage1_backend_factory_ = _resolve_stage1_backend(
            stage1_backend
        )

        self.site_models_: Dict[str, BaseNaturalnessPMFModel] = {}
        self.site_state_models_: Dict[
            str,
            Dict[Tuple[str, ...], BaseNaturalnessPMFModel],
        ] = {}
        self.observations_: Optional[pd.DataFrame] = None
        self.c_surface_: Optional[NaturalnessGPSurface] = None
        self.d_surface_: Optional[NaturalnessGPSurface] = None
        self.training_data_: Optional[Dict[str, pd.DataFrame]] = None
        self.treatment_: Optional[str] = None
        self.outcome_: Optional[str] = None
        self.regime_col_: Optional[str] = None
        self.is_fitted_ = False

    def fit(
        self,
        training_data: Dict[str, pd.DataFrame],
        treatment: str,
        outcome: str,
        regime_col: str = "F",
        fit_gp: Optional[bool] = None,
    ) -> "NaturalnessGrounder":
        """Run stages 1, 2, and optionally 3."""
        self.fit_stage1_pmfs(training_data, treatment, outcome, regime_col)
        self.compute_stage2_observations()
        should_fit_gp = self.fit_gp if fit_gp is None else fit_gp
        if should_fit_gp:
            self.fit_stage3_gp()
        self.is_fitted_ = True
        return self

    def fit_stage1_pmfs(
        self,
        training_data: Dict[str, pd.DataFrame],
        treatment: str,
        outcome: str,
        regime_col: str = "F",
    ) -> Dict[str, BaseNaturalnessPMFModel]:
        """
        Fit the configured Stage-1 PMF model for every site.

        One model is fitted per site over the full covariate state. When the
        grid declares ``conditioning_covariates``, one model is fitted per site
        and distinct candidate state instead, on exactly that state's
        covariates, and ``site_state_models_`` holds all of them.
        ``site_models_`` keeps one model per site in both cases: the full-state
        model, or the first candidate state's model as an alias for per-site
        bookkeeping.

        Args:
            training_data: Site-keyed rows containing both regimes.
            treatment: Binary treatment column.
            outcome: Binary outcome column.
            regime_col: Column separating ``idle`` and randomized rows.

        Returns:
            The per-site model mapping ``site_models_``.
        """
        if not training_data:
            raise ValueError("training_data must contain at least one site")

        self.training_data_ = {site: df.copy() for site, df in training_data.items()}
        self.treatment_ = treatment
        self.outcome_ = outcome
        self.regime_col_ = regime_col
        self.site_models_ = {}
        self.site_state_models_ = {}
        state_keys = (
            self.grid.candidate_state_keys
            if self.grid.uses_candidate_states
            else [tuple(self.grid.covariates)]
        )

        for site_id, site_data in self.training_data_.items():
            state_models: Dict[Tuple[str, ...], BaseNaturalnessPMFModel] = {}
            for state in state_keys:
                backend_kwargs = {
                    "site_id": site_id,
                    "data": site_data,
                    "covariates": list(state),
                    "treatment": treatment,
                    "outcome": outcome,
                    "regime_col": regime_col,
                    "alpha": self.alpha,
                    "min_overlap": self.min_overlap,
                    "idle_value": self.idle_value,
                    "on_value": self.on_value,
                }
                backend_kwargs.update(self.stage1_kwargs)
                model = self.stage1_backend_factory_(**backend_kwargs)
                state_models[state] = _validate_stage1_model(model)
            self.site_state_models_[site_id] = state_models
            self.site_models_[site_id] = state_models[state_keys[0]]

        return self.site_models_

    def _site_state_model(
        self,
        site_id: str,
        state_covariates: List[str],
    ) -> BaseNaturalnessPMFModel:
        """Return one site's Stage-1 model for a candidate state.

        Args:
            site_id: Source-site label.
            state_covariates: Ordered covariates of the candidate state.

        Returns:
            The Stage-1 model fitted on that state. Models assigned directly
            to ``site_models_`` without state models are returned as-is.
        """
        state_models = self.site_state_models_.get(site_id)
        if state_models is None:
            return self.site_models_[site_id]
        return state_models[tuple(state_covariates)]

    def _iter_site_state_models(
        self,
    ) -> List[Tuple[str, Tuple[str, ...], BaseNaturalnessPMFModel]]:
        """Return ``(site, state, model)`` for every fitted Stage-1 model."""
        if not self.site_state_models_:
            return [
                (site, tuple(self.grid.covariates), model)
                for site, model in self.site_models_.items()
            ]
        return [
            (site, state, model)
            for site, state_models in self.site_state_models_.items()
            for state, model in state_models.items()
        ]

    def compute_stage2_observations(self) -> pd.DataFrame:
        """
        Run the source-site ``(site, covariate, epsilon, tau)`` sweep.

        For each site ``k`` and covariate index ``j``, the observed trigger
        row fires when both conditions hold:

            I_kj <= epsilon
            D_kj >= tau

        where ``I_kj`` is the maximum observed idle outcome shift
        ``|P(Y=y | X=x,Z=z; idle) - P(Y=y | X=x,Z=z'; idle)|`` over matching
        ``Z_-j`` strata, and ``D_kj`` is the minimum idle treatment shift
        ``|P(X=x | Z=z; idle) - P(X=x | Z=z'; idle)|``.  Fired rows receive
        estimated note-scale tolerances ``c_hat`` and ``d_hat``; unfired rows
        retain diagnostics but have missing tolerances. With grid
        ``conditioning_covariates``, ``Z`` is candidate ``j``'s own state
        (``NaturalnessGrid.candidate_state``) and the statistics use that
        state's Stage-1 model.
        """
        if self.training_data_ is None:
            raise RuntimeError("Call fit_stage1_pmfs before compute_stage2_observations")

        rows: List[NaturalnessObservation] = []

        for site_id, site_data in self.training_data_.items():
            assert self.regime_col_ is not None
            idle = site_data[site_data[self.regime_col_] == self.idle_value]

            for j, covariate in self.grid.candidate_items:
                state_covariates, state_index = self.grid.candidate_state(j)
                model = self._site_state_model(site_id, state_covariates)
                z_minus_values = self._get_z_minus_values(site_data, j)
                z_minus_values = [
                    tuple(int(v) if isinstance(v, bool) else v for v in row)
                    for row in z_minus_values
                ]
                i_deviation = naturalness_statistics.compute_i_deviation(
                    state_covariates,
                    state_index,
                    z_minus_values,
                    model.p_idle_y_given_xz,
                    idle,
                )
                d_strength = naturalness_statistics.compute_d_strength(
                    state_covariates,
                    state_index,
                    z_minus_values,
                    model.p_idle_x_given_z,
                    idle,
                )

                c_hat_cache: Optional[float] = None
                d_hat_cache: Optional[float] = None

                for epsilon in self.grid.epsilons:
                    for tau in self.grid.taus:
                        trigger_fired = bool(
                            np.isfinite(i_deviation)
                            and np.isfinite(d_strength)
                            and i_deviation <= epsilon
                            and d_strength >= tau
                        )

                        c_hat: Optional[float] = None
                        d_hat: Optional[float] = None

                        if trigger_fired:
                            if c_hat_cache is None or d_hat_cache is None:
                                c_hat_cache, d_hat_cache = self._estimate_c_d_hat(
                                    model, site_data, j, z_minus_values
                                )
                            c_hat = c_hat_cache
                            d_hat = d_hat_cache

                        rows.append(
                            NaturalnessObservation(
                                site_id=site_id,
                                covariate=covariate,
                                covariate_index=j,
                                epsilon=float(epsilon),
                                tau=float(tau),
                                i_deviation=float(i_deviation),
                                d_strength=float(d_strength),
                                trigger_fired=trigger_fired,
                                c_hat=c_hat,
                                d_hat=d_hat,
                                n_z_minus_evaluated=len(z_minus_values),
                                warnings=" | ".join(model.diagnostics[-3:]),
                            )
                        )

        self.observations_ = pd.DataFrame([asdict(row) for row in rows])
        return self.observations_

    def fit_stage3_gp(self, min_observations: int = 4) -> "NaturalnessGrounder":
        """Fit optional GP smoothers for c_hat and d_hat."""
        if self.observations_ is None:
            raise RuntimeError("Call compute_stage2_observations before fit_stage3_gp")

        self.c_surface_ = NaturalnessGPSurface(
            "c_hat",
            min_observations=min_observations,
            random_seed=self.random_seed,
            feature_mode=self.gp_feature_mode,
            training_target=self.gp_training_target,
            target_scale=self.gp_target_scale,
        ).fit(self.observations_)
        self.d_surface_ = NaturalnessGPSurface(
            "d_hat",
            min_observations=min_observations,
            random_seed=self.random_seed,
            feature_mode=self.gp_feature_mode,
            training_target=self.gp_training_target,
            target_scale=self.gp_target_scale,
        ).fit(self.observations_)
        return self

    def predict_grid(
        self,
        scale: str = "probability",
        quantile: Optional[float] = None,
    ) -> pd.DataFrame:
        """Predict c/d or c^0/d^0 over the configured grid."""
        rows = []
        for j, covariate in self.grid.candidate_items:
            for epsilon in self.grid.epsilons:
                for tau in self.grid.taus:
                    c_mean, c_std, c_source = self._predict_surface_details(
                        "c_hat", self.c_surface_, covariate, epsilon, tau,
                        scale=scale, quantile=quantile
                    )
                    d_mean, d_std, d_source = self._predict_surface_details(
                        "d_hat", self.d_surface_, covariate, epsilon, tau,
                        scale=scale, quantile=quantile
                    )
                    rows.append({
                        "covariate": covariate,
                        "covariate_index": j,
                        "epsilon": float(epsilon),
                        "tau": float(tau),
                        "c_mean": c_mean,
                        "c_std_logit": c_std,
                        "c_source": c_source,
                        "d_mean": d_mean,
                        "d_std_logit": d_std,
                        "d_source": d_source,
                        "prediction_scale": scale,
                        "prediction_quantile": quantile,
                    })
        return pd.DataFrame(rows)

    def predict_tolerance(
        self,
        covariate: str,
        epsilon: float,
        tau: float,
        quantile: Optional[float] = None,
        clip: bool = True,
        i_deviation: Optional[float] = None,
        d_strength: Optional[float] = None,
    ) -> NaturalnessTolerance:
        """
        Predict learned ``c`` and ``d`` for one covariate/grid point.

        Args:
            covariate: Candidate naturalness covariate.
            epsilon: Naturalness trigger threshold.
            tau: Treatment-strength trigger threshold.
            quantile: Optional posterior quantile on the GP logit scale.
                ``None`` returns the posterior mean or empirical fallback.
            clip: Whether to keep finite predictions in the valid
                note-scale interval ``[0, 0.5]``.
            i_deviation: Observed target/source ``I_j`` statistic used when the
                fitted GP uses ``gp_feature_mode='observed_stats'``.
            d_strength: Observed target/source ``D_j`` statistic used when the
                fitted GP uses ``gp_feature_mode='observed_stats'``.

        Returns:
            NaturalnessTolerance containing only the learned discovery
            quantities and their provenance.
        """
        c, c_std, c_source = self._predict_surface_details(
            "c_hat",
            self.c_surface_,
            covariate,
            epsilon,
            tau,
            quantile=quantile,
            i_deviation=i_deviation,
            d_strength=d_strength,
        )
        d, d_std, d_source = self._predict_surface_details(
            "d_hat",
            self.d_surface_,
            covariate,
            epsilon,
            tau,
            quantile=quantile,
            i_deviation=i_deviation,
            d_strength=d_strength,
        )

        if clip:
            c = self._clip_note_scale_tolerance(c)
            d = self._clip_note_scale_tolerance(d)

        return NaturalnessTolerance(
            covariate=covariate,
            epsilon=float(epsilon),
            tau=float(tau),
            c=c,
            d=d,
            c_source=c_source,
            d_source=d_source,
            c_std_logit=c_std,
            d_std_logit=d_std,
        )

    def sample_tolerances(
        self,
        rows: pd.DataFrame,
        n_samples: int,
        random_seed: Optional[int] = None,
        clip: bool = True,
    ) -> pd.DataFrame:
        """
        Draw joint posterior samples of learned ``c`` and ``d`` tolerances.

        Args:
            rows: Naturalness grid rows, usually target-triggered candidates,
                with ``epsilon`` and ``tau`` plus either ``covariate_index`` or
                ``covariate``.
            n_samples: Number of posterior draws to return.
            random_seed: Optional seed for reproducible sampling. If omitted,
                the grounder's ``random_seed`` is reused.
            clip: Whether to keep sampled note-scale tolerances in
                ``[0, 0.5]``.

        Returns:
            Long-format DataFrame with one row per input row and posterior
            sample. Sampled values are exposed as ``c``, ``d``, ``two_c``, and
            ``two_d``.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.c_surface_ is None or self.d_surface_ is None:
            raise RuntimeError("Call fit_stage3_gp before sample_tolerances")
        if not self.c_surface_.is_fitted or not self.d_surface_.is_fitted:
            raise RuntimeError("Both c and d GP surfaces must be fitted before sampling")

        prepared = self._prepare_tolerance_sample_rows(rows)
        sample_columns = [
            "sample_id",
            "c",
            "d",
            "two_c",
            "two_d",
            "c_source",
            "d_source",
            "prediction_scale",
        ]
        base_columns = [
            column for column in prepared.columns if column not in sample_columns
        ]
        output_columns = base_columns + sample_columns
        if prepared.empty:
            return pd.DataFrame(columns=output_columns)

        features = prepared[list(self.c_surface_.feature_columns)]
        c_seed, d_seed = self._split_sample_seeds(random_seed)
        c_samples = self.c_surface_.sample(
            features,
            n_samples=n_samples,
            scale="probability",
            random_seed=c_seed,
        )
        d_samples = self.d_surface_.sample(
            features,
            n_samples=n_samples,
            scale="probability",
            random_seed=d_seed,
        )

        if clip:
            c_samples = np.clip(c_samples, 0.0, GP_NOTE_SCALE_TOLERANCE_CAP)
            d_samples = np.clip(d_samples, 0.0, GP_NOTE_SCALE_TOLERANCE_CAP)

        records: List[Dict[str, Any]] = []
        base_records = prepared[base_columns].to_dict("records")
        for sample_id in range(n_samples):
            for row_index, base_record in enumerate(base_records):
                c_value = float(c_samples[sample_id, row_index])
                d_value = float(d_samples[sample_id, row_index])
                record = dict(base_record)
                record.update({
                    "sample_id": int(sample_id),
                    "c": c_value,
                    "d": d_value,
                    "two_c": 2.0 * c_value,
                    "two_d": 2.0 * d_value,
                    "c_source": "gp_sample",
                    "d_source": "gp_sample",
                    "prediction_scale": "probability",
                })
                records.append(record)
        return pd.DataFrame(records, columns=output_columns)

    def predict_simultaneous_upper_tolerances(
        self,
        rows: pd.DataFrame,
        *,
        confidence_level: float = 0.95,
        n_samples: int = 4096,
        random_seed: Optional[int] = None,
        clip: bool = True,
    ) -> List[NaturalnessTolerance]:
        """
        Construct one simultaneous posterior upper vector for target LP use.

        Joint latent-surface draws are standardized coordinate-wise across all
        requested ``c`` and ``d`` values; the empirical ``confidence_level``
        quantile of each draw's largest standardized deviation is one shared
        critical value. Adding it, times the posterior standard deviation, to
        every logit-scale posterior mean gives a one-sided band whose joint
        posterior coverage targets the requested level under the fitted GP.

        Args:
            rows: Triggered target rows with ``epsilon``, ``tau`` and ``covariate``
                or ``covariate_index``; observed-stat GP mode also needs
                ``i_deviation`` and ``d_strength``.
            confidence_level: Joint posterior probability of the one-sided upper vector.
            n_samples: Monte Carlo draws used to estimate the maximum-statistic critical value.
            random_seed: Optional reproducibility seed. Defaults to the grounder's seed.
            clip: Whether to restrict note-scale results to ``[0, 0.5]`` before the LP.

        Returns:
            Tolerances in the same row order, all carrying the common joint-band
            confidence level, critical value, family size, and sample count.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if self.c_surface_ is None or self.d_surface_ is None:
            raise RuntimeError("Call fit_stage3_gp before upper-tolerance prediction")
        if not self.c_surface_.is_fitted or not self.d_surface_.is_fitted:
            raise RuntimeError(
                "Both c and d GP surfaces must be fitted before upper-tolerance "
                "prediction"
            )

        prepared = self._prepare_tolerance_sample_rows(rows)
        if prepared.empty:
            return []

        c_features = prepared[list(self.c_surface_.feature_columns)].to_numpy(
            dtype=float
        )
        d_features = prepared[list(self.d_surface_.feature_columns)].to_numpy(
            dtype=float
        )
        c_mean, c_covariance = self.c_surface_._posterior(
            c_features,
            include_observation_noise=False,
        )
        d_mean, d_covariance = self.d_surface_._posterior(
            d_features,
            include_observation_noise=False,
        )
        c_std = np.sqrt(np.maximum(np.diag(c_covariance), 0.0))
        d_std = np.sqrt(np.maximum(np.diag(d_covariance), 0.0))

        c_seed, d_seed = self._split_sample_seeds(random_seed)
        c_draws = self.c_surface_.sample(
            c_features,
            n_samples=n_samples,
            scale="logit",
            random_seed=c_seed,
        )
        d_draws = self.d_surface_.sample(
            d_features,
            n_samples=n_samples,
            scale="logit",
            random_seed=d_seed,
        )
        combined_mean = np.concatenate([c_mean, d_mean])
        combined_std = np.concatenate([c_std, d_std])
        combined_draws = np.concatenate([c_draws, d_draws], axis=1)
        stochastic = combined_std > np.sqrt(np.finfo(float).eps)
        if stochastic.any():
            standardized = (
                combined_draws[:, stochastic] - combined_mean[stochastic]
            ) / combined_std[stochastic]
            maximum_scores = standardized.max(axis=1)
            critical_value = float(
                np.quantile(maximum_scores, confidence_level, method="higher")
            )
        else:
            critical_value = 0.0

        c_upper = self.c_surface_.to_probability(c_mean + critical_value * c_std)
        d_upper = self.d_surface_.to_probability(d_mean + critical_value * d_std)
        if clip:
            c_upper = np.clip(c_upper, 0.0, GP_NOTE_SCALE_TOLERANCE_CAP)
            d_upper = np.clip(d_upper, 0.0, GP_NOTE_SCALE_TOLERANCE_CAP)

        source = f"gp_simultaneous_upper_{confidence_level:.2f}"
        family_size = int(2 * len(prepared))
        tolerances: List[NaturalnessTolerance] = []
        for row_index, row in enumerate(prepared.itertuples(index=False)):
            covariate_index = int(row.covariate_index)
            covariate = (
                str(row.covariate)
                if hasattr(row, "covariate")
                else str(self.grid.covariates[covariate_index])
            )
            tolerances.append(NaturalnessTolerance(
                covariate=covariate,
                epsilon=float(row.epsilon),
                tau=float(row.tau),
                c=float(c_upper[row_index]),
                d=float(d_upper[row_index]),
                c_source=source,
                d_source=source,
                c_std_logit=float(c_std[row_index]),
                d_std_logit=float(d_std[row_index]),
                upper_confidence_level=float(confidence_level),
                upper_critical_value=critical_value,
                upper_family_size=family_size,
                upper_n_samples=int(n_samples),
            ))
        return tolerances

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return fitting diagnostics."""
        observations = self.observations_
        return {
            "n_sites": len(self.site_models_),
            "n_observations": 0 if observations is None else len(observations),
            "n_triggered": 0 if observations is None else int(observations["trigger_fired"].sum()),
            "stage1_backend": self.stage1_backend_name_,
            "gp_feature_mode": self.gp_feature_mode,
            "c_gp_status": None if self.c_surface_ is None else self.c_surface_.status,
            "c_gp_message": None if self.c_surface_ is None else self.c_surface_.message,
            "c_gp_diagnostics": (
                None if self.c_surface_ is None else self.c_surface_.diagnostics()
            ),
            "d_gp_status": None if self.d_surface_ is None else self.d_surface_.status,
            "d_gp_message": None if self.d_surface_ is None else self.d_surface_.message,
            "d_gp_diagnostics": (
                None if self.d_surface_ is None else self.d_surface_.diagnostics()
            ),
            "site_warnings": self._site_warnings(),
        }

    def _site_warnings(self) -> Dict[str, List[str]]:
        """Return nonempty Stage-1 warnings by site.

        With candidate states, a site's warnings from every state model are
        combined and prefixed with the state they came from.
        """
        if not self.grid.uses_candidate_states or not self.site_state_models_:
            return {
                site: getattr(model, "diagnostics", [])
                for site, model in self.site_models_.items()
                if getattr(model, "diagnostics", [])
            }
        warnings: Dict[str, List[str]] = {}
        for site, state, model in self._iter_site_state_models():
            messages = getattr(model, "diagnostics", [])
            if messages:
                label = "|".join(state)
                warnings.setdefault(site, []).extend(
                    f"[{label}] {message}" for message in messages
                )
        return warnings

    def get_stage1_diagnostics_frame(self) -> pd.DataFrame:
        """
        Return row-wise diagnostics exposed by Stage-1 site models.

        Returns:
            DataFrame containing backend-specific Stage-1 diagnostics. Backends
            that do not implement ``stage1_diagnostics`` simply contribute no
            rows. With candidate states, every state model contributes and a
            ``candidate_state`` column names its covariates.
        """
        rows: List[pd.DataFrame] = []
        with_states = self.grid.uses_candidate_states and bool(self.site_state_models_)
        for site, state, model in self._iter_site_state_models():
            getter = getattr(model, "stage1_diagnostics", None)
            if not callable(getter):
                continue
            frame = getter()
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            if "site" not in frame.columns:
                frame["site"] = site
            if with_states:
                frame["candidate_state"] = "|".join(state)
            rows.append(frame)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True, sort=False)

    def _predict_surface(
        self,
        target_col: str,
        surface: Optional[NaturalnessGPSurface],
        covariate: str,
        epsilon: float,
        tau: float,
        return_std: bool,
        scale: str = "probability",
        quantile: Optional[float] = None,
        i_deviation: Optional[float] = None,
        d_strength: Optional[float] = None,
    ) -> Any:
        """Return a learned or empirical fallback prediction for one surface."""
        mean, std, _source = self._predict_surface_details(
            target_col,
            surface,
            covariate,
            epsilon,
            tau,
            scale=scale,
            quantile=quantile,
            i_deviation=i_deviation,
            d_strength=d_strength,
        )
        if return_std:
            return mean, std
        return mean

    def _predict_surface_details(
        self,
        target_col: str,
        surface: Optional[NaturalnessGPSurface],
        covariate: str,
        epsilon: float,
        tau: float,
        scale: str = "probability",
        quantile: Optional[float] = None,
        i_deviation: Optional[float] = None,
        d_strength: Optional[float] = None,
    ) -> Tuple[float, Optional[float], str]:
        """Return the prediction mean, uncertainty, and source label."""
        if self.observations_ is None:
            raise RuntimeError("NaturalnessGrounder is not fitted")
        if covariate not in self.grid.covariates:
            raise ValueError(f"Unknown covariate {covariate!r}")
        if scale not in {"probability", "logit"}:
            raise ValueError("scale must be 'probability' or 'logit'")

        covariate_index = self.grid.covariates.index(covariate)

        can_use_gp = (
            surface is not None
            and surface.is_fitted
            and (
                surface.feature_mode != GP_FEATURE_MODE_OBSERVED_STATS
                or (i_deviation is not None and d_strength is not None)
            )
        )
        if can_use_gp:
            mean, std = surface.predict(
                epsilon,
                tau,
                covariate_index,
                scale=scale,
                quantile=quantile,
                i_deviation=i_deviation,
                d_strength=d_strength,
            )
            source = "gp" if quantile is None else f"gp_quantile_{quantile:.2f}"
            return mean, std, source

        exact = self.observations_[
            (self.observations_["covariate"] == covariate)
            & np.isclose(self.observations_["epsilon"], epsilon)
            & np.isclose(self.observations_["tau"], tau)
            & self.observations_["trigger_fired"]
            & self.observations_[target_col].notna()
        ]
        if not exact.empty:
            value = self._empirical_tolerance(exact, target_col)
            if scale == "logit":
                value = self._empirical_logit(value)
            return value, None, "empirical_exact"

        nearby = self.observations_[
            (self.observations_["covariate"] == covariate)
            & self.observations_["trigger_fired"]
            & self.observations_[target_col].notna()
        ]
        if not nearby.empty:
            value = self._empirical_tolerance(nearby, target_col)
            if scale == "logit":
                value = self._empirical_logit(value)
            return value, None, "empirical_covariate_mean"

        return float("nan"), None, "unavailable"

    def _empirical_tolerance(self, rows: pd.DataFrame, target_col: str) -> float:
        """Return the empirical fallback tolerance under the GP training target.

        ``per_site`` averages the triggered site rows. ``envelope`` averages the
        per-grid-point maxima over sites, which is the maximum itself when
        ``rows`` hold a single grid point.
        """
        if self.gp_training_target == GP_TRAINING_TARGET_ENVELOPE:
            return float(rows.groupby(["epsilon", "tau"])[target_col].max().mean())
        return float(rows[target_col].mean())

    def _empirical_logit(self, value: float) -> float:
        """Map a note-scale fallback tolerance to the GP's logit scale."""
        factor = _GP_TARGET_SCALE_FACTORS[self.gp_target_scale]
        return float(_logit(np.array([factor * value]))[0])

    def _prepare_tolerance_sample_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Validate and enrich grid rows used for posterior tolerance samples."""
        required = ["epsilon", "tau"]
        missing = [column for column in required if column not in rows.columns]
        if missing:
            raise ValueError(f"Missing required sampling columns: {missing}")

        prepared = rows.copy()
        if "covariate_index" not in prepared.columns:
            if "covariate" not in prepared.columns:
                raise ValueError("rows must include covariate_index or covariate")
            prepared["covariate_index"] = prepared["covariate"].map(
                {name: index for index, name in enumerate(self.grid.covariates)}
            )

        if prepared["covariate_index"].isna().any():
            raise ValueError("rows contain unknown covariate values")
        prepared["epsilon"] = pd.to_numeric(prepared["epsilon"], errors="raise")
        prepared["tau"] = pd.to_numeric(prepared["tau"], errors="raise")
        prepared["covariate_index"] = pd.to_numeric(
            prepared["covariate_index"], errors="raise"
        ).astype(int)
        for column in _gp_feature_columns(self.gp_feature_mode):
            if column in {"epsilon", "tau", "covariate_index"}:
                continue
            if column not in prepared.columns:
                raise ValueError(
                    "rows must include "
                    f"{column} when gp_feature_mode={self.gp_feature_mode!r}"
                )
            prepared[column] = pd.to_numeric(prepared[column], errors="raise")

        if "covariate" not in prepared.columns:
            index_to_covariate = {
                index: name for index, name in enumerate(self.grid.covariates)
            }
            prepared["covariate"] = prepared["covariate_index"].map(index_to_covariate)
        if prepared["covariate"].isna().any():
            raise ValueError("rows contain covariate_index values outside the grid")
        return prepared

    def _split_sample_seeds(self, random_seed: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        """Return separate reproducible seeds for c and d posterior sampling."""
        seed = self.random_seed if random_seed is None else random_seed
        if seed is None:
            return None, None
        seed_sequence = np.random.SeedSequence(int(seed))
        child_sequences = seed_sequence.spawn(2)
        return (
            int(child_sequences[0].generate_state(1)[0]),
            int(child_sequences[1].generate_state(1)[0]),
        )

    @staticmethod
    def _clip_note_scale_tolerance(value: float) -> float:
        """Keep finite c/d tolerance predictions in their note-scale range."""
        if np.isnan(value):
            return float(value)
        return float(np.clip(value, 0.0, GP_NOTE_SCALE_TOLERANCE_CAP))

    def _get_z_minus_values(self, site_data: pd.DataFrame, j: int) -> List[Tuple[Any, ...]]:
        """Enumerate supported observed values of ``Z_-j`` in candidate j's state."""
        assert self.regime_col_ is not None
        idle = site_data[site_data[self.regime_col_] == self.idle_value]
        state_covariates, state_index = self.grid.candidate_state(j)
        z_minus_cols = [
            c for idx, c in enumerate(state_covariates) if idx != state_index
        ]

        if not z_minus_cols:
            return [()]

        grouped = idle.groupby(z_minus_cols, observed=True).size()
        grouped = grouped[grouped >= self.grid.min_cell_count]

        values = [
            naturalness_statistics.as_tuple(index)
            for index in grouped.index.tolist()
        ]
        values = naturalness_statistics.safe_sorted(values)

        max_values = self.grid.max_z_minus_samples
        if max_values is not None and len(values) > max_values:
            rng = np.random.default_rng(self.random_seed)
            keep = sorted(rng.choice(len(values), size=max_values, replace=False).tolist())
            values = [values[i] for i in keep]

        return values

    def _estimate_c_d_hat(
        self,
        model: BaseNaturalnessPMFModel,
        site_data: pd.DataFrame,
        j: int,
        z_minus_values: List[Tuple[Any, ...]],
    ) -> Tuple[float, float]:
        """
        Estimate note-scale ``c_hat`` and ``d_hat`` for a fired source row.

        For the selected covariate ``j`` and matching ``Z_-j`` strata, this
        computes the maximum operational potential-outcome gaps:

            max |P(Y(star)=y | X=x,Z=(z_j,z_-j); idle)
                 - P(Y(star)=y | X=x,Z=(z'_j,z_-j); idle)|

        and:

            max |P(Y(star)=y | Z=(z_j,z_-j))
                 - P(Y(star)=y | Z=(z'_j,z_-j))|.

        The naturalness condition defines hidden response-type tolerances
        ``c`` and ``d`` on the half scale.  Binary potential-outcome events
        aggregate two response-type masses, so the LP uses ``2c`` and ``2d``.
        The stored estimates are therefore:

            c_hat = max conditional potential-outcome gap / 2
            d_hat = max marginal potential-outcome gap / 2.
        """
        assert self.regime_col_ is not None
        idle = site_data[site_data[self.regime_col_] == self.idle_value]
        # ``model`` is fitted on candidate j's state, so strata are rebuilt there.
        state_covariates, state_index = self.grid.candidate_state(j)
        z_j_values = naturalness_statistics.observed_covariate_values(
            idle,
            state_covariates[state_index],
        )
        c_diffs: List[float] = []
        d_diffs: List[float] = []

        for z_minus in z_minus_values:
            for z_j, z_j_prime in combinations(z_j_values, 2):
                z = naturalness_statistics.assemble_covariate_state(
                    state_index,
                    z_j,
                    z_minus,
                    len(state_covariates),
                )
                z_prime = naturalness_statistics.assemble_covariate_state(
                    state_index,
                    z_j_prime,
                    z_minus,
                    len(state_covariates),
                )

                for star in [0, 1]:
                    for y in [0, 1]:
                        for x in [0, 1]:
                            c_diffs.append(abs(
                                model.p_star_y_given_xz(star, y, x, z)
                                - model.p_star_y_given_xz(star, y, x, z_prime)
                            ))
                        d_diffs.append(abs(
                            model.p_star_y_given_z(star, y, z)
                            - model.p_star_y_given_z(star, y, z_prime)
                        ))

        c_hat = float(np.max(c_diffs) / 2.0) if c_diffs else float("nan")
        d_hat = float(np.max(d_diffs) / 2.0) if d_diffs else float("nan")
        return c_hat, d_hat


__all__ = [
    "BaseNaturalnessPMFModel",
    "EmpiricalSitePMFModels",
    "NaturalnessGrid",
    "NaturalnessObservation",
    "NaturalnessTolerance",
    "NaturalnessGPKernel",
    "NaturalnessGPSurface",
    "NATURALNESS_STAGE1_BACKENDS",
    "SUPPORTED_GP_FEATURE_MODES",
    "NaturalnessGrounder",
    "HierarchicalDirichletSitePMFModels",
    "PredictorBackedSitePMFModels",
    "TabICLv2SitePMFModels",
    "XGBoostSitePMFModels",
    "register_naturalness_stage1_backend",
]
