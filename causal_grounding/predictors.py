"""
Prediction Models for CATE Estimation

This module provides pluggable prediction models for estimating
P(Y|X,A) and P(A|X) probabilities used in LP bounds computation:

- EmpiricalPredictor: Empirical counts (the default)
- XGBoostPredictor: XGBoost-based models
- HierarchicalDirichletPredictor: Empirical-Bayes smoothed cell tables
- TabICLv2Predictor: TabICLv2 in-context binary classification models

:func:`create_predictor` builds a predictor from its backend name. The XGBoost
models are based on Ricardo's learn_xgb and XGBoostBackdoorModel implementations
(``ricardo_code/method.ipynb``).
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple, Union
import warnings

# XGBoost is optional
_XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import log_loss, make_scorer
    _XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore
    GridSearchCV = None  # type: ignore
    log_loss = None  # type: ignore
    make_scorer = None  # type: ignore

_TABICL_AVAILABLE: Optional[bool] = None
TabICLClassifier: Any = None


TABICL_V2_CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def _load_tabicl_classifier() -> Any:
    """Load the optional TabICLv2 classifier only when its backend is selected.

    Returns:
        The imported ``tabicl.TabICLClassifier`` class.

    Raises:
        ImportError: If TabICL is unavailable in the active environment.
        Exception: Propagates non-import initialization failures from TabICL
            so backend-specific runtime problems retain their original cause.
    """

    global _TABICL_AVAILABLE, TabICLClassifier

    if _TABICL_AVAILABLE is False:
        raise ImportError(
            "TabICLv2 is not available. Install with: "
            "pip install tabicl==2.1.1"
        )
    if TabICLClassifier is not None:
        _TABICL_AVAILABLE = True
        return TabICLClassifier

    try:
        from tabicl import TabICLClassifier as classifier
    except ImportError as exc:
        _TABICL_AVAILABLE = False
        raise ImportError(
            "TabICLv2 is not available. Install with: "
            "pip install tabicl==2.1.1"
        ) from exc

    TabICLClassifier = classifier
    _TABICL_AVAILABLE = True
    return classifier


def _make_binary_log_loss_scorer() -> Any:
    """
    Build a binary log-loss scorer with explicit class labels.

    GridSearchCV can produce validation folds containing only one binary class
    when a site/regime has imbalanced outcomes. Passing labels explicitly keeps
    scikit-learn from inferring a one-class label set for that fold.
    """
    return make_scorer(
        log_loss,
        response_method="predict_proba",
        greater_is_better=False,
        labels=[0, 1],
    )


#: ``(category, message regex)`` of the warnings an XGBoost fit or its grid search
#: raises on small or imbalanced sites. scikit-learn's cross-validation emits them
#: when a fold misses a class. ``FitFailedWarning`` is added at call time because
#: scikit-learn is optional at import.
_EXPECTED_XGBOOST_FIT_WARNINGS: Tuple[Tuple[type, str], ...] = (
    (UserWarning, r"The least populated class in y has only \d+ members"),
    (UserWarning, r"One or more of the (test|train) scores are non-finite"),
)


@contextmanager
def _expected_xgboost_fit_warnings_ignored() -> Iterator[None]:
    """Silence the known cross-validation warnings of an XGBoost fit.

    Other warnings still reach the caller.

    Yields:
        None, inside a ``warnings.catch_warnings`` block.
    """
    with warnings.catch_warnings():
        for category, message in _EXPECTED_XGBOOST_FIT_WARNINGS:
            warnings.filterwarnings("ignore", message=message, category=category)
        try:
            from sklearn.exceptions import FitFailedWarning
        except ImportError:  # pragma: no cover - scikit-learn is a core dependency
            pass
        else:
            warnings.filterwarnings("ignore", category=FitFailedWarning)
        yield


class BasePredictorModel(ABC):
    """
    Abstract base class for prediction models.

    Defines interface for propensity and outcome prediction.
    """

    @abstractmethod
    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str]
    ) -> 'BasePredictorModel':
        """
        Fit propensity model P(A|X).

        Args:
            data: DataFrame with all variables
            treatment: Treatment column name
            covariates: List of covariate column names

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str]
    ) -> 'BasePredictorModel':
        """
        Fit outcome model P(Y|X,A).

        Args:
            data: DataFrame with all variables
            outcome: Outcome column name
            treatment: Treatment column name
            covariates: List of covariate column names

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict P(A=1|X).

        Args:
            X: Covariate values

        Returns:
            Array of propensity scores
        """
        pass

    @abstractmethod
    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int]
    ) -> np.ndarray:
        """
        Predict P(Y=1|X,A) or E[Y|X,A].

        Args:
            X: Covariate values
            A: Treatment values (can be scalar for potential outcome prediction)

        Returns:
            Array of outcome predictions
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> 'BasePredictorModel':
        """
        Fit both propensity and outcome models.

        Args:
            data: DataFrame with all variables
            treatment: Treatment column name
            outcome: Outcome column name
            covariates: List of covariate column names

        Returns:
            self for chaining
        """
        self.fit_propensity(data, treatment, covariates)
        self.fit_outcome(data, outcome, treatment, covariates)
        return self

    def predict_potential_outcomes(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict potential outcomes Y(0) and Y(1).

        Args:
            X: Covariate values

        Returns:
            (Y0_pred, Y1_pred) arrays
        """
        Y0 = self.predict_outcome(X, 0)
        Y1 = self.predict_outcome(X, 1)
        return Y0, Y1

    def predict_cate(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict CATE = E[Y(1) - Y(0) | X].

        Args:
            X: Covariate values

        Returns:
            Array of CATE predictions
        """
        Y0, Y1 = self.predict_potential_outcomes(X)
        return Y1 - Y0


class EmpiricalPredictor(BasePredictorModel):
    """
    Empirical counts predictor.

    Uses empirical frequencies P(Y|X,A) and P(A|X) from the data.
    This is the default approach for the LP bounds computation.

    For discrete covariates, computes exact conditional probabilities.
    For continuous covariates, uses binning.
    """

    def __init__(
        self,
        smoothing: float = 0.01,
        n_bins: int = 5
    ):
        """
        Initialize empirical predictor.

        Args:
            smoothing: Laplace smoothing parameter for counts
            n_bins: Number of bins for continuous covariates
        """
        self.smoothing = smoothing
        self.n_bins = n_bins

        self._propensity_table = None
        self._outcome_tables = {}  # {A: table}
        self._covariates = None
        self._is_fitted_propensity = False
        self._is_fitted_outcome = False

    def _discretize_continuous(
        self,
        values: np.ndarray
    ) -> np.ndarray:
        """Discretize continuous values into bins."""
        if len(np.unique(values)) > self.n_bins:
            return pd.qcut(values, self.n_bins, labels=False, duplicates='drop')
        return values

    def _create_key(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Create string key from covariate values."""
        if isinstance(X, pd.DataFrame):
            X = X[self._covariates].values

        keys = []
        for row in X:
            key = '-'.join(str(int(v)) for v in row)
            keys.append(key)
        return np.array(keys)

    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str]
    ) -> 'EmpiricalPredictor':
        """Fit propensity model using empirical counts."""
        self._covariates = covariates

        # Create key for each row
        keys = self._create_key(data)

        # Count A=1 and total for each key
        self._propensity_table = {}

        for key in np.unique(keys):
            mask = keys == key
            n_treated = data.loc[mask, treatment].sum()
            n_total = mask.sum()

            # Laplace smoothing
            prob = (n_treated + self.smoothing) / (n_total + 2 * self.smoothing)
            self._propensity_table[key] = prob

        self._is_fitted_propensity = True
        return self

    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str]
    ) -> 'EmpiricalPredictor':
        """Fit outcome model using empirical counts."""
        self._covariates = covariates

        # Create key for each row
        keys = self._create_key(data)

        # Count Y=1 for each (key, A) combination
        for a in [0, 1]:
            self._outcome_tables[a] = {}
            a_mask = data[treatment] == a

            for key in np.unique(keys):
                key_mask = keys == key
                mask = key_mask & a_mask

                if mask.sum() > 0:
                    n_positive = data.loc[mask, outcome].sum()
                    n_total = mask.sum()
                    prob = (n_positive + self.smoothing) / (n_total + 2 * self.smoothing)
                else:
                    # No data for this combination, use marginal
                    prob = 0.5

                self._outcome_tables[a][key] = prob

        self._is_fitted_outcome = True
        return self

    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict P(A=1|X) using empirical frequencies."""
        if not self._is_fitted_propensity:
            raise RuntimeError("Must fit propensity model first")

        keys = self._create_key(X)
        probs = np.array([
            self._propensity_table.get(key, 0.5)
            for key in keys
        ])
        return probs

    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int]
    ) -> np.ndarray:
        """Predict P(Y=1|X,A) using empirical frequencies."""
        if not self._is_fitted_outcome:
            raise RuntimeError("Must fit outcome model first")

        keys = self._create_key(X)

        if isinstance(A, (int, np.integer)):
            A = np.full(len(keys), A)

        probs = np.array([
            self._outcome_tables[int(a)].get(key, 0.5)
            for key, a in zip(keys, A)
        ])
        return probs

    def get_probability_table(self) -> pd.DataFrame:
        """
        Get full probability table for debugging.

        Returns:
            DataFrame with columns: key, P(A=1|X), P(Y=1|X,A=0), P(Y=1|X,A=1)
        """
        if not self._is_fitted_outcome:
            raise RuntimeError("Must fit models first")

        all_keys = set(self._propensity_table.keys())
        for table in self._outcome_tables.values():
            all_keys.update(table.keys())

        rows = []
        for key in sorted(all_keys):
            rows.append({
                'key': key,
                'P(A=1|X)': self._propensity_table.get(key, 0.5),
                'P(Y=1|X,A=0)': self._outcome_tables[0].get(key, 0.5),
                'P(Y=1|X,A=1)': self._outcome_tables[1].get(key, 0.5)
            })

        return pd.DataFrame(rows)


class HierarchicalDirichletPredictor(BasePredictorModel):
    """
    Empirical-Bayes Dirichlet-smoothed predictor for finite covariate cells.

    This backend keeps the same discrete-cell structure as
    ``EmpiricalPredictor`` but replaces raw cell proportions with posterior
    predictive means. For propensity, each ``P(A=1 | X=x)`` cell is shrunk
    toward the global treatment rate. For outcomes, each
    ``P(Y=1 | X=x,A=a)`` cell is shrunk toward the treatment-arm-specific
    global outcome rate. The global rates themselves use a weak symmetric
    prior so empty or one-class samples remain bounded.
    """

    def __init__(
        self,
        prior_strength: float = 2.0,
        propensity_prior_strength: Optional[float] = None,
        outcome_prior_strength: Optional[float] = None,
        base_prior_strength: float = 2.0,
        global_prior: float = 0.5,
    ) -> None:
        """
        Initialize finite hierarchical Dirichlet smoothing settings.

        Args:
            prior_strength: Default pseudo-count strength used for both
                propensity and outcome cells.
            propensity_prior_strength: Optional pseudo-count strength for
                ``P(A=1 | X)`` cells. Defaults to ``prior_strength``.
            outcome_prior_strength: Optional pseudo-count strength for
                ``P(Y=1 | X,A)`` cells. Defaults to ``prior_strength``.
            base_prior_strength: Pseudo-count strength used when estimating
                global shrinkage targets.
            global_prior: Symmetric prior mean used for global binary rates.

        Raises:
            ValueError: If any strength is negative or ``global_prior`` is not
                a valid probability.
        """
        self.prior_strength = float(prior_strength)
        self.propensity_prior_strength = float(
            prior_strength
            if propensity_prior_strength is None
            else propensity_prior_strength
        )
        self.outcome_prior_strength = float(
            prior_strength if outcome_prior_strength is None else outcome_prior_strength
        )
        self.base_prior_strength = float(base_prior_strength)
        self.global_prior = float(global_prior)
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        if self.propensity_prior_strength < 0:
            raise ValueError("propensity_prior_strength must be non-negative")
        if self.outcome_prior_strength < 0:
            raise ValueError("outcome_prior_strength must be non-negative")
        if self.base_prior_strength < 0:
            raise ValueError("base_prior_strength must be non-negative")
        if not 0.0 <= self.global_prior <= 1.0:
            raise ValueError("global_prior must lie in [0, 1]")

        self._covariates: Optional[List[str]] = None
        self._treatment: Optional[str] = None
        self._propensity_table: Dict[Tuple[Any, ...], float] = {}
        self._outcome_tables: Dict[int, Dict[Tuple[Any, ...], float]] = {
            0: {},
            1: {},
        }
        self._propensity_base_rate: Optional[float] = None
        self._propensity_base_params: Optional[Tuple[float, float]] = None
        self._propensity_cell_params: Dict[Tuple[Any, ...], Tuple[float, float]] = {}
        self._propensity_raw_rate: Optional[float] = None
        self._propensity_parent_weight: Optional[float] = None
        self._propensity_cell_parent_weights: Dict[Tuple[Any, ...], float] = {}
        self._outcome_base_rates: Dict[int, float] = {}
        self._outcome_base_rate: Optional[float] = None
        self._outcome_base_params: Optional[Tuple[float, float]] = None
        self._outcome_arm_params: Dict[int, Tuple[float, float]] = {}
        self._outcome_cell_params: Dict[int, Dict[Tuple[Any, ...], Tuple[float, float]]] = {
            0: {},
            1: {},
        }
        self._outcome_raw_rate: Optional[float] = None
        self._outcome_parent_weight: Optional[float] = None
        self._outcome_arm_raw_rates: Dict[int, float] = {}
        self._outcome_arm_parent_weights: Dict[int, float] = {}
        self._outcome_cell_parent_weights: Dict[int, Dict[Tuple[Any, ...], float]] = {
            0: {},
            1: {},
        }
        self._is_fitted_propensity = False
        self._is_fitted_outcome = False

    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str],
    ) -> "HierarchicalDirichletPredictor":
        """
        Fit smoothed propensity cells ``P(A=1 | X)``.

        Args:
            data: DataFrame containing treatment and covariates.
            treatment: Binary treatment column.
            covariates: Covariate columns defining finite cells.

        Returns:
            ``self`` for chaining.
        """
        self._covariates = list(covariates)
        self._treatment = treatment
        y = self._binary_target(data[treatment], treatment)
        keys = self._keys(data[self._covariates])
        parent_rate = self.global_prior
        self._propensity_base_params = self._beta_params(
            n_success=int(y.sum()),
            n_total=int(len(y)),
            prior_mean=parent_rate,
            prior_strength=self.base_prior_strength,
        )
        self._propensity_base_rate = self._beta_mean_from_params(
            self._propensity_base_params,
            parent_rate,
        )
        self._propensity_raw_rate = (
            float(y.mean()) if len(y) > 0 else float(parent_rate)
        )
        self._propensity_parent_weight = self._prior_weight(
            n_total=int(len(y)),
            prior_strength=self.base_prior_strength,
        )
        self._propensity_table = {}
        self._propensity_cell_params = {}
        self._propensity_cell_parent_weights = {}
        for key in sorted(set(keys)):
            mask = np.asarray([candidate == key for candidate in keys], dtype=bool)
            params = self._beta_params(
                n_success=int(y[mask].sum()),
                n_total=int(mask.sum()),
                prior_mean=self._propensity_base_rate,
                prior_strength=self.propensity_prior_strength,
            )
            self._propensity_cell_params[key] = params
            self._propensity_cell_parent_weights[key] = self._prior_weight(
                n_total=int(mask.sum()),
                prior_strength=self.propensity_prior_strength,
            )
            self._propensity_table[key] = self._beta_mean_from_params(
                params,
                self._propensity_base_rate,
            )
        self._is_fitted_propensity = True
        return self

    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str],
    ) -> "HierarchicalDirichletPredictor":
        """
        Fit smoothed outcome cells ``P(Y=1 | X,A)``.

        Args:
            data: DataFrame containing outcome, treatment, and covariates.
            outcome: Binary outcome column.
            treatment: Binary treatment column.
            covariates: Covariate columns defining finite cells.

        Returns:
            ``self`` for chaining.
        """
        self._covariates = list(covariates)
        self._treatment = treatment
        y = self._binary_target(data[outcome], outcome)
        a_values = self._binary_target(data[treatment], treatment)
        keys = self._keys(data[self._covariates])
        parent_rate = self.global_prior
        self._outcome_base_params = self._beta_params(
            n_success=int(y.sum()),
            n_total=int(len(y)),
            prior_mean=parent_rate,
            prior_strength=self.base_prior_strength,
        )
        self._outcome_base_rate = self._beta_mean_from_params(
            self._outcome_base_params,
            parent_rate,
        )
        self._outcome_raw_rate = float(y.mean()) if len(y) > 0 else float(parent_rate)
        self._outcome_parent_weight = self._prior_weight(
            n_total=int(len(y)),
            prior_strength=self.base_prior_strength,
        )
        self._outcome_base_rates = {}
        self._outcome_arm_params = {}
        self._outcome_tables = {0: {}, 1: {}}
        self._outcome_cell_params = {0: {}, 1: {}}
        self._outcome_arm_raw_rates = {}
        self._outcome_arm_parent_weights = {}
        self._outcome_cell_parent_weights = {0: {}, 1: {}}
        for treatment_value in [0, 1]:
            treatment_mask = a_values == treatment_value
            arm_parent_rate = self._outcome_base_rate
            arm_params = self._beta_params(
                n_success=int(y[treatment_mask].sum()),
                n_total=int(treatment_mask.sum()),
                prior_mean=arm_parent_rate,
                prior_strength=self.base_prior_strength,
            )
            self._outcome_arm_params[treatment_value] = arm_params
            self._outcome_arm_raw_rates[treatment_value] = (
                float(y[treatment_mask].mean())
                if int(treatment_mask.sum()) > 0
                else float(arm_parent_rate)
            )
            self._outcome_arm_parent_weights[treatment_value] = self._prior_weight(
                n_total=int(treatment_mask.sum()),
                prior_strength=self.base_prior_strength,
            )
            self._outcome_base_rates[treatment_value] = self._beta_mean_from_params(
                arm_params,
                arm_parent_rate,
            )
            for key in sorted(set(keys)):
                key_mask = np.asarray(
                    [candidate == key for candidate in keys],
                    dtype=bool,
                )
                cell_mask = key_mask & treatment_mask
                params = self._beta_params(
                    n_success=int(y[cell_mask].sum()),
                    n_total=int(cell_mask.sum()),
                    prior_mean=self._outcome_base_rates[treatment_value],
                    prior_strength=self.outcome_prior_strength,
                )
                self._outcome_cell_params[treatment_value][key] = params
                self._outcome_cell_parent_weights[treatment_value][key] = (
                    self._prior_weight(
                        n_total=int(cell_mask.sum()),
                        prior_strength=self.outcome_prior_strength,
                    )
                )
                self._outcome_tables[treatment_value][key] = (
                    self._beta_mean_from_params(
                        params,
                        self._outcome_base_rates[treatment_value],
                    )
                )
        self._is_fitted_outcome = True
        return self

    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict ``P(A=1 | X)`` with smoothed finite-cell probabilities.

        Args:
            X: Covariate feature matrix.

        Returns:
            Propensity probabilities for each row.
        """
        if not self._is_fitted_propensity or self._propensity_base_rate is None:
            raise RuntimeError("Must fit propensity model first")
        keys = self._prediction_keys(X)
        return np.asarray([
            self._propensity_table.get(key, self._propensity_base_rate)
            for key in keys
        ])

    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int],
    ) -> np.ndarray:
        """
        Predict ``P(Y=1 | X,A)`` with smoothed finite-cell probabilities.

        Args:
            X: Covariate feature matrix.
            A: Treatment values or scalar treatment intervention.

        Returns:
            Outcome probabilities for each row.
        """
        if not self._is_fitted_outcome or self._outcome_base_rate is None:
            raise RuntimeError("Must fit outcome model first")
        keys = self._prediction_keys(X)
        if isinstance(A, (int, np.integer)):
            treatment_values = np.full(len(keys), int(A), dtype=int)
        else:
            treatment_values = np.asarray(A, dtype=int)
            if treatment_values.shape[0] != len(keys):
                raise ValueError("A must be a scalar or match the number of X rows")

        return np.asarray([
            self._outcome_tables[int(a)].get(
                key,
                self._outcome_base_rates.get(int(a), self._outcome_base_rate),
            )
            for key, a in zip(keys, treatment_values)
        ])

    def get_probability_table(self) -> pd.DataFrame:
        """
        Return fitted smoothed cell probabilities for diagnostics.

        Returns:
            DataFrame with one row per observed finite covariate cell.
        """
        if not self._is_fitted_outcome:
            raise RuntimeError("Must fit models first")
        keys = set(self._propensity_table)
        for table in self._outcome_tables.values():
            keys.update(table)
        rows = []
        for key in sorted(keys):
            rows.append({
                "key": key,
                "P(A=1|X)": self._propensity_table.get(
                    key,
                    self._propensity_base_rate,
                ),
                "P(Y=1|X,A=0)": self._outcome_tables[0].get(
                    key,
                    self._outcome_base_rates.get(0, self._outcome_base_rate),
                ),
                "P(Y=1|X,A=1)": self._outcome_tables[1].get(
                    key,
                    self._outcome_base_rates.get(1, self._outcome_base_rate),
                ),
            })
        return pd.DataFrame(rows)

    def get_shrinkage_diagnostics(self) -> Dict[str, Any]:
        """
        Return finite-cell shrinkage diagnostics for the fitted tables.

        Returns:
            Dictionary summarizing parent rates, raw site rates, posterior site
            rates, and average parent-weight contributions for propensity and
            outcome cells. Missing model components are omitted.
        """
        diagnostics: Dict[str, Any] = {
            "prior_strength": float(self.prior_strength),
            "propensity_prior_strength": float(self.propensity_prior_strength),
            "outcome_prior_strength": float(self.outcome_prior_strength),
            "base_prior_strength": float(self.base_prior_strength),
            "global_prior": float(self.global_prior),
        }
        if self._is_fitted_propensity:
            cell_weights = list(self._propensity_cell_parent_weights.values())
            diagnostics.update({
                "propensity_parent_rate": (
                    float(self.global_prior)
                ),
                "propensity_raw_rate": self._none_or_float(
                    self._propensity_raw_rate
                ),
                "propensity_site_base_rate": self._none_or_float(
                    self._propensity_base_rate
                ),
                "propensity_site_parent_weight": self._none_or_float(
                    self._propensity_parent_weight
                ),
                "propensity_n_cells": int(len(self._propensity_table)),
                "propensity_mean_cell_parent_weight": self._mean_or_nan(
                    cell_weights
                ),
                "propensity_max_cell_parent_weight": self._max_or_nan(
                    cell_weights
                ),
            })
        if self._is_fitted_outcome:
            diagnostics.update({
                "outcome_parent_rate": (
                    float(self.global_prior)
                ),
                "outcome_raw_rate": self._none_or_float(self._outcome_raw_rate),
                "outcome_site_base_rate": self._none_or_float(
                    self._outcome_base_rate
                ),
                "outcome_site_parent_weight": self._none_or_float(
                    self._outcome_parent_weight
                ),
            })
            for treatment_value in [0, 1]:
                cell_weights = list(
                    self._outcome_cell_parent_weights.get(
                        treatment_value,
                        {},
                    ).values()
                )
                prefix = f"outcome_arm{treatment_value}"
                diagnostics.update({
                    f"{prefix}_parent_rate": float(
                        self._outcome_base_rate
                        if self._outcome_base_rate is not None
                        else self.global_prior
                    ),
                    f"{prefix}_raw_rate": self._none_or_float(
                        self._outcome_arm_raw_rates.get(treatment_value)
                    ),
                    f"{prefix}_site_base_rate": self._none_or_float(
                        self._outcome_base_rates.get(treatment_value)
                    ),
                    f"{prefix}_site_parent_weight": self._none_or_float(
                        self._outcome_arm_parent_weights.get(treatment_value)
                    ),
                    f"{prefix}_n_cells": int(
                        len(self._outcome_tables.get(treatment_value, {}))
                    ),
                    f"{prefix}_mean_cell_parent_weight": self._mean_or_nan(
                        cell_weights
                    ),
                    f"{prefix}_max_cell_parent_weight": self._max_or_nan(
                        cell_weights
                    ),
                })
        return diagnostics

    def sample_posterior(
        self,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> "HierarchicalDirichletPredictor":
        """
        Draw one fitted predictor from the empirical-Bayes Beta posteriors.

        The deterministic ``hierarchical_dirichlet`` backend stores posterior
        predictive means. This method keeps the fitted finite-cell support and
        draws each global, arm-level, and cell-level binary probability from
        the Beta posterior parameters that produced those means. It is useful
        for LP sensitivity analyses that need coherent sampled probability
        tables without changing the standard prediction API.

        Args:
            random_state: Optional integer seed or numpy random generator.

        Returns:
            A fitted ``HierarchicalDirichletPredictor`` whose prediction tables
            contain posterior draws instead of posterior means.

        Raises:
            RuntimeError: If neither propensity nor outcome tables have been
                fitted before sampling.
        """
        if not self._is_fitted_propensity and not self._is_fitted_outcome:
            raise RuntimeError("Must fit propensity or outcome model before sampling")

        rng = self._coerce_rng(random_state)
        sampled = self._empty_like()
        sampled._covariates = None if self._covariates is None else list(self._covariates)
        sampled._treatment = self._treatment

        if self._is_fitted_propensity:
            sampled._is_fitted_propensity = True
            sampled._propensity_base_params = self._propensity_base_params
            sampled._propensity_cell_params = dict(self._propensity_cell_params)
            sampled._propensity_base_rate = self._sample_beta_from_params(
                self._propensity_base_params,
                self._propensity_base_rate,
                rng,
            )
            sampled._propensity_table = {
                key: self._sample_beta_from_params(
                    params,
                    self._propensity_table.get(key, sampled._propensity_base_rate),
                    rng,
                )
                for key, params in self._propensity_cell_params.items()
            }

        if self._is_fitted_outcome:
            sampled._is_fitted_outcome = True
            sampled._outcome_base_params = self._outcome_base_params
            sampled._outcome_arm_params = dict(self._outcome_arm_params)
            sampled._outcome_cell_params = {
                treatment_value: dict(params)
                for treatment_value, params in self._outcome_cell_params.items()
            }
            sampled._outcome_base_rate = self._sample_beta_from_params(
                self._outcome_base_params,
                self._outcome_base_rate,
                rng,
            )
            sampled._outcome_base_rates = {
                treatment_value: self._sample_beta_from_params(
                    params,
                    self._outcome_base_rates.get(
                        treatment_value,
                        sampled._outcome_base_rate,
                    ),
                    rng,
                )
                for treatment_value, params in self._outcome_arm_params.items()
            }
            sampled._outcome_tables = {0: {}, 1: {}}
            for treatment_value, table in self._outcome_cell_params.items():
                fallback = sampled._outcome_base_rates.get(
                    treatment_value,
                    sampled._outcome_base_rate,
                )
                sampled._outcome_tables[treatment_value] = {
                    key: self._sample_beta_from_params(
                        params,
                        self._outcome_tables[treatment_value].get(key, fallback),
                        rng,
                    )
                    for key, params in table.items()
                }

        return sampled

    def _empty_like(self) -> "HierarchicalDirichletPredictor":
        """
        Create an unfitted predictor with the same smoothing hyperparameters.

        Returns:
            New predictor instance ready to receive copied sampled tables.
        """
        return self.__class__(
            prior_strength=self.prior_strength,
            propensity_prior_strength=self.propensity_prior_strength,
            outcome_prior_strength=self.outcome_prior_strength,
            base_prior_strength=self.base_prior_strength,
            global_prior=self.global_prior,
        )

    def _prediction_keys(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> List[Tuple[Any, ...]]:
        """
        Convert prediction input into canonical finite-cell keys.

        Args:
            X: DataFrame or array supplied to a predict method.

        Returns:
            One tuple key per row.
        """
        if isinstance(X, pd.DataFrame):
            if self._covariates is None:
                raise RuntimeError("Covariates are unknown before fitting")
            return self._keys(X[self._covariates])
        return self._keys(np.asarray(X, dtype=object))

    @classmethod
    def _keys(
        cls,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> List[Tuple[Any, ...]]:
        """
        Create canonical tuple keys from finite covariate rows.

        Args:
            X: DataFrame or two-dimensional array of covariates.

        Returns:
            List of hashable tuple keys.
        """
        values = X.to_numpy(dtype=object) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=object)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return [tuple(cls._canonical_value(value) for value in row) for row in values]

    @staticmethod
    def _canonical_value(value: Any) -> Any:
        """
        Convert numpy and missing-value scalars into stable key components.

        Args:
            value: Scalar covariate value.

        Returns:
            Hashable value suitable for tuple keys.
        """
        if pd.isna(value):
            return "<NA>"
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _binary_target(values: pd.Series, name: str) -> np.ndarray:
        """
        Validate and return a binary target vector.

        Args:
            values: Target values.
            name: Column name used in validation errors.

        Returns:
            Integer binary target vector.
        """
        unique_values = set(pd.Series(values).dropna().unique().tolist())
        if not unique_values.issubset({0, 1, 0.0, 1.0, False, True}):
            raise ValueError(f"{name!r} must be binary for HierarchicalDirichletPredictor")
        return np.asarray(values, dtype=int)

    @staticmethod
    def _posterior_mean(
        n_success: int,
        n_total: int,
        prior_mean: float,
        prior_strength: float,
    ) -> float:
        """
        Compute a binary Dirichlet/Beta posterior predictive mean.

        Args:
            n_success: Number of observed positive labels.
            n_total: Number of observed labels.
            prior_mean: Prior mean for the positive class.
            prior_strength: Pseudo-count strength of the prior.

        Returns:
            Posterior predictive probability for the positive class.
        """
        params = HierarchicalDirichletPredictor._beta_params(
            n_success=n_success,
            n_total=n_total,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
        )
        return HierarchicalDirichletPredictor._beta_mean_from_params(
            params,
            prior_mean,
        )

    @staticmethod
    def _beta_params(
        n_success: int,
        n_total: int,
        prior_mean: float,
        prior_strength: float,
    ) -> Tuple[float, float]:
        """
        Return Beta posterior shape parameters for a binary cell.

        Args:
            n_success: Number of observed positive labels.
            n_total: Number of observed labels.
            prior_mean: Prior mean for the positive class.
            prior_strength: Pseudo-count strength of the prior.

        Returns:
            ``(alpha, beta)`` posterior shape parameters.
        """
        clipped_mean = float(np.clip(prior_mean, 0.0, 1.0))
        alpha = float(n_success) + float(prior_strength) * clipped_mean
        beta = (
            float(n_total)
            - float(n_success)
            + float(prior_strength) * (1.0 - clipped_mean)
        )
        return alpha, beta

    @staticmethod
    def _beta_mean_from_params(
        params: Tuple[float, float],
        fallback: Optional[float],
    ) -> float:
        """
        Compute a bounded Beta mean with fallback for zero-strength cells.

        Args:
            params: ``(alpha, beta)`` posterior shape parameters.
            fallback: Value used when both shape parameters sum to zero.

        Returns:
            Mean probability in ``[0, 1]``.
        """
        alpha, beta = params
        denominator = float(alpha) + float(beta)
        if denominator <= 0:
            return float(np.clip(0.5 if fallback is None else fallback, 0.0, 1.0))
        return float(np.clip(float(alpha) / denominator, 0.0, 1.0))

    @staticmethod
    def _prior_weight(
        n_total: int,
        prior_strength: float,
    ) -> float:
        """
        Return the posterior-mean weight assigned to the parent probability.

        Args:
            n_total: Number of observations in the fitted cell.
            prior_strength: Parent pseudo-count strength.

        Returns:
            Weight in ``[0, 1]`` multiplying the parent rate.
        """
        denominator = float(n_total) + float(prior_strength)
        if denominator <= 0.0:
            return 1.0
        return float(np.clip(float(prior_strength) / denominator, 0.0, 1.0))

    @staticmethod
    def _none_or_float(value: Optional[float]) -> float:
        """
        Convert optional numeric diagnostics to a serializable float.

        Args:
            value: Optional scalar.

        Returns:
            ``float(value)`` or ``nan`` when ``value`` is missing.
        """
        return float("nan") if value is None else float(value)

    @staticmethod
    def _mean_or_nan(values: List[float]) -> float:
        """
        Compute a finite mean for diagnostic weights.

        Args:
            values: Numeric values to summarize.

        Returns:
            Mean value, or ``nan`` for an empty list.
        """
        return float(np.mean(values)) if values else float("nan")

    @staticmethod
    def _max_or_nan(values: List[float]) -> float:
        """
        Compute a finite maximum for diagnostic weights.

        Args:
            values: Numeric values to summarize.

        Returns:
            Maximum value, or ``nan`` for an empty list.
        """
        return float(np.max(values)) if values else float("nan")

    @classmethod
    def _sample_beta_from_params(
        cls,
        params: Optional[Tuple[float, float]],
        fallback: Optional[float],
        rng: np.random.Generator,
    ) -> float:
        """
        Draw a bounded scalar from Beta parameters with deterministic fallback.

        Args:
            params: Optional ``(alpha, beta)`` shape parameters.
            fallback: Mean value used when sampling is undefined.
            rng: Random number generator used for the draw.

        Returns:
            Drawn probability in ``[0, 1]``.
        """
        if params is None:
            return float(np.clip(0.5 if fallback is None else fallback, 0.0, 1.0))
        alpha, beta = params
        if alpha <= 0.0 or beta <= 0.0:
            return cls._beta_mean_from_params(params, fallback)
        return float(np.clip(rng.beta(alpha, beta), 0.0, 1.0))

    @staticmethod
    def _coerce_rng(
        random_state: Optional[Union[int, np.random.Generator]],
    ) -> np.random.Generator:
        """
        Normalize accepted random-state inputs to a numpy generator.

        Args:
            random_state: Optional integer seed or existing generator.

        Returns:
            Numpy random generator.
        """
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)


class XGBoostPredictor(BasePredictorModel):
    """
    XGBoost-based predictor.

    Uses XGBoost gradient boosting for propensity and outcome models.
    Based on Ricardo's learn_xgb implementation from method.ipynb.

    Advantages:
    - Handles high-dimensional covariates
    - Captures non-linear relationships
    - Provides feature importance
    - Better generalization to new covariate values
    """

    def __init__(
        self,
        use_grid_search: bool = True,
        n_estimators_range: List[int] = None,
        max_depth_range: List[int] = None,
        learning_rate_range: List[float] = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize XGBoost predictor.

        Args:
            use_grid_search: Use GridSearchCV for hyperparameter tuning
            n_estimators_range: Range of n_estimators to try
            max_depth_range: Range of max_depth to try
            learning_rate_range: Range of learning_rate to try
            n_estimators: Fixed estimator count when grid search is disabled.
            max_depth: Fixed tree depth when grid search is disabled.
            learning_rate: Fixed learning rate when grid search is disabled.
            cv_folds: Number of CV folds for grid search
            random_state: Random seed for reproducibility
            n_jobs: Worker count for grid search and XGBoost. ``None`` runs
                the grid search on all cores.
        """
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not available. Install with: pip install xgboost"
            )

        self.use_grid_search = use_grid_search
        self.n_estimators_range = n_estimators_range or [100, 200, 400]
        self.max_depth_range = max_depth_range or [3, 5, 7]
        self.learning_rate_range = learning_rate_range or [0.05, 0.1, 0.2]
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._propensity_model = None
        self._outcome_model = None  # Single model that takes A as feature
        self._covariates = None
        self._treatment = None
        self._is_binary_outcome = True

    def _learn_xgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        objective: str = 'binary:logistic'
    ) -> Any:
        """
        Learn XGBoost model with optional grid search.

        Based on Ricardo's learn_xgb implementation.

        Args:
            X: Feature matrix
            y: Target vector
            objective: XGBoost objective function

        Returns:
            Fitted XGBoost model (XGBClassifier)
        """
        base_params = {
            'objective': objective,
            'eval_metric': 'logloss',
            'random_state': self.random_state,
        }
        if self.n_jobs is not None:
            base_params["n_jobs"] = int(self.n_jobs)

        if self.use_grid_search:
            param_grid = {
                'n_estimators': self.n_estimators_range,
                'max_depth': self.max_depth_range,
                'learning_rate': self.learning_rate_range
            }

            base_model = xgb.XGBClassifier(**base_params)

            with _expected_xgboost_fit_warnings_ignored():
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring=_make_binary_log_loss_scorer(),
                    n_jobs=-1 if self.n_jobs is None else int(self.n_jobs),
                )
                grid_search.fit(X, y)

            return grid_search.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                **base_params
            )

            with _expected_xgboost_fit_warnings_ignored():
                model.fit(X, y)

            return model

    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str]
    ) -> 'XGBoostPredictor':
        """Fit propensity model P(A|X) using XGBoost."""
        self._covariates = covariates
        self._treatment = treatment

        X = data[covariates].values
        y = data[treatment].values

        self._propensity_model = self._learn_xgb(X, y)
        return self

    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str]
    ) -> 'XGBoostPredictor':
        """Fit outcome model P(Y|X,A) using XGBoost."""
        self._covariates = covariates
        self._treatment = treatment

        # Check if outcome is binary
        unique_y = np.unique(data[outcome])
        self._is_binary_outcome = len(unique_y) == 2

        # Add treatment as feature
        X = np.column_stack([data[covariates].values, data[treatment].values])
        y = data[outcome].values

        if self._is_binary_outcome:
            self._outcome_model = self._learn_xgb(X, y)
        else:
            # For continuous outcome, use regression
            self._outcome_model = self._learn_xgb_regressor(X, y)

        return self

    def _learn_xgb_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """Learn XGBoost regressor for continuous outcome."""
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': self.random_state
        }
        if self.n_jobs is not None:
            base_params["n_jobs"] = int(self.n_jobs)

        if self.use_grid_search:
            param_grid = {
                'n_estimators': self.n_estimators_range,
                'max_depth': self.max_depth_range,
                'learning_rate': self.learning_rate_range
            }

            base_model = xgb.XGBRegressor(**base_params)

            with _expected_xgboost_fit_warnings_ignored():
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1 if self.n_jobs is None else int(self.n_jobs),
                )
                grid_search.fit(X, y)

            return grid_search.best_estimator_
        else:
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                **base_params
            )

            with _expected_xgboost_fit_warnings_ignored():
                model.fit(X, y)

            return model

    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict P(A=1|X) using XGBoost."""
        if self._propensity_model is None:
            raise RuntimeError("Must fit propensity model first")

        if isinstance(X, pd.DataFrame):
            X = X[self._covariates].values

        return self._propensity_model.predict_proba(X)[:, 1]

    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int]
    ) -> np.ndarray:
        """Predict P(Y=1|X,A) or E[Y|X,A] using XGBoost."""
        if self._outcome_model is None:
            raise RuntimeError("Must fit outcome model first")

        if isinstance(X, pd.DataFrame):
            X = X[self._covariates].values

        n = X.shape[0]

        if isinstance(A, (int, np.integer)):
            A = np.full(n, A)

        X_with_A = np.column_stack([X, A])

        if self._is_binary_outcome:
            return self._outcome_model.predict_proba(X_with_A)[:, 1]
        else:
            return self._outcome_model.predict(X_with_A)

    def get_feature_importance(
        self,
        model_type: str = 'outcome'
    ) -> pd.DataFrame:
        """
        Get feature importance from XGBoost model.

        Args:
            model_type: 'propensity' or 'outcome'

        Returns:
            DataFrame with feature importances
        """
        if model_type == 'propensity':
            if self._propensity_model is None:
                raise RuntimeError("Propensity model not fitted")
            model = self._propensity_model
            features = self._covariates
        else:
            if self._outcome_model is None:
                raise RuntimeError("Outcome model not fitted")
            model = self._outcome_model
            features = list(self._covariates) + [self._treatment]

        importance = model.feature_importances_

        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)


class _ConstantBinaryClassifier:
    """Minimal probability model used when a binary target has one class."""

    def __init__(self, probability_one: float) -> None:
        """Store the constant probability returned by ``predict_proba``."""
        self.probability_one = float(np.clip(probability_one, 0.0, 1.0))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return constant binary probabilities for every row in ``X``.

        Args:
            X: Feature matrix whose row count determines output length.

        Returns:
            Array with columns ``P(Y=0)`` and ``P(Y=1)``.
        """
        n_rows = len(X)
        p1 = np.full(n_rows, self.probability_one, dtype=float)
        return np.column_stack([1.0 - p1, p1])


class _SharedTabICLCheckpointCache:
    """Process-wide cache behind ``share_checkpoint`` TabICL classifiers.

    Attributes:
        models: Loaded ``(model_path_, model_, model_config_)`` triples, keyed by
            ``(model_path, checkpoint_version)``.
        classifier_classes: Shared-checkpoint subclass of each TabICL classifier
            class.
    """

    def __init__(self) -> None:
        """Start with empty caches."""
        self.models: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}
        self.classifier_classes: Dict[Any, Any] = {}

    def clear(self) -> None:
        """Drop every loaded checkpoint and subclass, in place.

        Classifiers fitted later load their checkpoint again.
        """
        self.models.clear()
        self.classifier_classes.clear()


_SHARED_TABICL_CHECKPOINTS = _SharedTabICLCheckpointCache()
# Names kept for existing callers; they are the cache's own dictionaries.
_SHARED_TABICL_MODELS = _SHARED_TABICL_CHECKPOINTS.models
_SHARED_CHECKPOINT_CLASSES = _SHARED_TABICL_CHECKPOINTS.classifier_classes


def _shared_checkpoint_classifier(classifier_class: Any) -> Any:
    """Return a subclass of a TabICL classifier that loads each checkpoint once.

    ``TabICLClassifier.fit`` rebuilds the model from its checkpoint on every
    call.  The weights never change during in-context inference, so a caller
    that makes thousands of small fits in one process (bootstrap draws, for
    instance) can share one loaded model instead.  The KV cache a fit builds is
    stored on the classifier, not on the model, and the model's own transient
    cache is cleared after every batch, so sequential fits sharing a model give
    the same predictions as separately loaded ones.  Sharing is not safe for
    concurrent fits in several threads.

    Args:
        classifier_class: ``tabicl.TabICLClassifier`` or a compatible class.

    Returns:
        The cached subclass whose ``_load_model`` reuses a process-wide model.
    """
    cached = _SHARED_TABICL_CHECKPOINTS.classifier_classes.get(classifier_class)
    if cached is not None:
        return cached

    class SharedCheckpointTabICLClassifier(classifier_class):  # type: ignore[misc, valid-type]
        """TabICL classifier whose checkpoint is loaded once per process."""

        def _load_model(self) -> None:
            """Reuse the process-wide model for this checkpoint, loading it once."""
            key = (str(self.model_path), str(self.checkpoint_version))
            shared = _SHARED_TABICL_CHECKPOINTS.models.get(key)
            if shared is None:
                super()._load_model()
                _SHARED_TABICL_CHECKPOINTS.models[key] = (self.model_path_, self.model_, self.model_config_)
            else:
                self.model_path_, self.model_, self.model_config_ = shared

    SharedCheckpointTabICLClassifier.__name__ = f"SharedCheckpoint{getattr(classifier_class, '__name__', 'TabICL')}"
    _SHARED_TABICL_CHECKPOINTS.classifier_classes[classifier_class] = SharedCheckpointTabICLClassifier
    return SharedCheckpointTabICLClassifier


class TabICLv2Predictor(BasePredictorModel):
    """
    TabICLv2 classifier backend for binary propensity and outcome probabilities.

    The predictor fits separate in-context classifiers for ``P(A=1 | X)`` and
    ``P(Y=1 | X,A)``. It pins the released TabICLv2 classifier checkpoint by
    default while exposing the main inference and memory controls from
    ``tabicl.TabICLClassifier``. One-class treatment or outcome samples use a
    constant classifier, avoiding an unnecessary checkpoint load and defining
    valid probabilities for sparse naturalness sites.

    TabICL performs most computation during ``predict_proba`` rather than
    conventional parameter fitting. Stage-1 callers that repeatedly query the
    same site can therefore enable ``kv_cache`` (including the lower-memory
    ``"repr"`` mode) through this constructor.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[Union[str, List[str]]] = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: Optional[int] = 8,
        kv_cache: Union[bool, str] = False,
        model_path: Optional[Union[str, Path]] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = TABICL_V2_CLASSIFIER_CHECKPOINT,
        device: Optional[str] = None,
        use_amp: Union[bool, str] = "auto",
        use_fa3: Union[bool, str] = "auto",
        offload_mode: Union[bool, str] = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[Any] = None,
        share_checkpoint: bool = False,
    ) -> None:
        """Initialize TabICLv2 inference settings.

        Args:
            n_estimators: Number of transformed dataset views in each model.
            norm_methods: TabICL normalization method or methods.
            feat_shuffle_method: Feature-permutation strategy.
            class_shuffle_method: Class-label permutation strategy.
            outlier_threshold: Z-score clipping threshold.
            softmax_temperature: Classification softmax temperature.
            average_logits: Average logits rather than probabilities.
            support_many_classes: Enable TabICL many-class support.
            batch_size: Number of ensemble members processed together.
            kv_cache: Training-context cache: ``False``, ``True``, ``"kv"`` or ``"repr"``.
            model_path: Optional local checkpoint path.
            allow_auto_download: Permit checkpoint download from Hugging Face.
            checkpoint_version: Checkpoint filename. Defaults to TabICLv2.
            device: Inference device such as ``"cpu"`` or ``"cuda"``.
            use_amp: Automatic mixed-precision setting.
            use_fa3: Flash Attention 3 setting.
            offload_mode: Column-embedding offload strategy.
            disk_offload_dir: Directory used by disk offloading.
            random_state: Ensemble-generation seed.
            n_jobs: CPU inference thread count.
            verbose: Enable TabICL inference logging.
            inference_config: Optional advanced TabICL inference configuration.
            share_checkpoint: Load each checkpoint once per process and share it between
                fits (:func:`_shared_checkpoint_classifier`); predictions are unchanged.

        Raises:
            ImportError: If ``tabicl`` is not installed.
        """
        self._classifier_class = _load_tabicl_classifier()
        self.share_checkpoint = bool(share_checkpoint)
        if self.share_checkpoint:
            self._classifier_class = _shared_checkpoint_classifier(self._classifier_class)

        self.n_estimators = int(n_estimators)
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = float(outlier_threshold)
        self.softmax_temperature = float(softmax_temperature)
        self.average_logits = bool(average_logits)
        self.support_many_classes = bool(support_many_classes)
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = bool(allow_auto_download)
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = bool(verbose)
        self.inference_config = inference_config

        self._propensity_model = None
        self._outcome_model = None
        self._covariates: Optional[List[str]] = None
        self._treatment: Optional[str] = None

    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str],
    ) -> "TabICLv2Predictor":
        """
        Fit TabICLv2 for ``P(A=1 | X)``.

        Args:
            data: DataFrame containing treatment and covariates.
            treatment: Binary treatment column name.
            covariates: Covariate feature columns.

        Returns:
            ``self`` for chaining.
        """
        self._covariates = list(covariates)
        self._treatment = treatment
        features = self._feature_matrix(data, covariates)
        target = self._binary_target(data[treatment], treatment)
        self._propensity_model = self._fit_binary_model(features, target)
        return self

    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str],
    ) -> "TabICLv2Predictor":
        """
        Fit TabICLv2 for ``P(Y=1 | X,A)``.

        Args:
            data: DataFrame containing outcome, treatment, and covariates.
            outcome: Binary outcome column name.
            treatment: Binary treatment column name.
            covariates: Covariate feature columns.

        Returns:
            ``self`` for chaining.
        """
        self._covariates = list(covariates)
        self._treatment = treatment
        features = self._feature_matrix(data, covariates)
        treatment_values = self._binary_target(data[treatment], treatment)
        outcome_values = self._binary_target(data[outcome], outcome)
        features_with_treatment = np.column_stack([features, treatment_values])
        self._outcome_model = self._fit_binary_model(
            features_with_treatment,
            outcome_values,
        )
        return self

    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict ``P(A=1 | X)`` with the fitted TabICLv2 classifier.

        Args:
            X: Covariate feature matrix.

        Returns:
            Propensity probabilities.
        """
        if self._propensity_model is None:
            raise RuntimeError("Must fit propensity model first")
        features = self._prediction_matrix(X)
        return self._predict_positive_probability(
            self._propensity_model,
            features,
        )

    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int],
    ) -> np.ndarray:
        """
        Predict ``P(Y=1 | X,A)`` with the fitted TabICLv2 classifier.

        Args:
            X: Covariate feature matrix.
            A: Binary treatment values or a scalar intervention.

        Returns:
            Binary outcome probabilities.
        """
        if self._outcome_model is None:
            raise RuntimeError("Must fit outcome model first")
        features = self._prediction_matrix(X)
        n_rows = features.shape[0]
        if np.isscalar(A):
            raw_treatment = np.full(n_rows, A)
        else:
            raw_treatment = np.asarray(A).reshape(-1)
            if raw_treatment.shape[0] != n_rows:
                raise ValueError("A must be a scalar or match the number of X rows")
        if pd.isna(raw_treatment).any() or not set(
            np.unique(raw_treatment).tolist()
        ).issubset({0, 1, 0.0, 1.0, False, True}):
            raise ValueError("A must contain only binary treatment values")
        treatment_values = np.asarray(raw_treatment, dtype=int)
        features_with_treatment = np.column_stack([features, treatment_values])
        return self._predict_positive_probability(
            self._outcome_model,
            features_with_treatment,
        )

    def _fit_binary_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Fit TabICLv2 or return a constant model for a one-class target.

        Args:
            X: Numeric feature matrix.
            y: Binary target vector.

        Returns:
            Fitted classifier exposing ``predict_proba`` and ``classes_``.
        """
        classes = np.unique(y)
        if classes.size == 1:
            return _ConstantBinaryClassifier(float(classes[0]))
        model = self._classifier_class(**self._classifier_kwargs())
        model.fit(X, y)
        return model

    def _classifier_kwargs(self) -> Dict[str, Any]:
        """Return constructor arguments for ``TabICLClassifier``."""
        return {
            "n_estimators": self.n_estimators,
            "norm_methods": self.norm_methods,
            "feat_shuffle_method": self.feat_shuffle_method,
            "class_shuffle_method": self.class_shuffle_method,
            "outlier_threshold": self.outlier_threshold,
            "softmax_temperature": self.softmax_temperature,
            "average_logits": self.average_logits,
            "support_many_classes": self.support_many_classes,
            "batch_size": self.batch_size,
            "kv_cache": self.kv_cache,
            "model_path": self.model_path,
            "allow_auto_download": self.allow_auto_download,
            "checkpoint_version": self.checkpoint_version,
            "device": self.device,
            "use_amp": self.use_amp,
            "use_fa3": self.use_fa3,
            "offload_mode": self.offload_mode,
            "disk_offload_dir": self.disk_offload_dir,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "inference_config": self.inference_config,
        }

    @staticmethod
    def _feature_matrix(
        data: pd.DataFrame,
        covariates: List[str],
    ) -> np.ndarray:
        """
        Extract numeric fitting features.

        Args:
            data: DataFrame containing requested covariates.
            covariates: Covariate columns to extract.

        Returns:
            Two-dimensional numeric feature matrix.
        """
        return data[covariates].to_numpy(dtype=float)

    def _prediction_matrix(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Convert prediction input to the fitted covariate schema.

        Args:
            X: DataFrame or array supplied to a prediction method.

        Returns:
            Two-dimensional numeric feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            if self._covariates is None:
                raise RuntimeError("Covariates are unknown before fitting")
            return X[self._covariates].to_numpy(dtype=float)
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features

    @staticmethod
    def _binary_target(values: pd.Series, name: str) -> np.ndarray:
        """
        Validate and return a binary target vector.

        Args:
            values: Target values.
            name: Column name used in validation errors.

        Returns:
            Integer binary target vector.
        """
        series = pd.Series(values)
        if series.isna().any():
            raise ValueError(f"{name!r} must not contain missing values")
        unique_values = set(series.unique().tolist())
        if not unique_values.issubset({0, 1, 0.0, 1.0, False, True}):
            raise ValueError(f"{name!r} must be binary for TabICLv2Predictor")
        return series.to_numpy(dtype=int)

    @staticmethod
    def _predict_positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
        """
        Return probability assigned to class ``1``.

        Args:
            model: Fitted classifier with ``predict_proba``.
            X: Numeric feature matrix.

        Returns:
            Positive-class probability for each row.
        """
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return np.zeros(probabilities.shape[0], dtype=float)


@dataclass
class XGBoostBackdoorModel:
    """
    Container for fitted XGBoost propensity and outcome models.

    Based on Ricardo's XGBoostBackdoorModel dataclass.

    Attributes:
        bst_propensity: Fitted propensity model
        bst_outcome: Fitted outcome model
        covariates: List of covariate names
        treatment: Treatment column name
        outcome: Outcome column name
    """
    bst_propensity: Any  # XGBClassifier
    bst_outcome: Any  # XGBClassifier or XGBRegressor
    covariates: List[str]
    treatment: str
    outcome: str
    is_binary_outcome: bool = True


def create_predictor(
    method: str = 'empirical',
    **kwargs
) -> BasePredictorModel:
    """
    Factory function to create predictor.

    The backend names and their factories come from
    :mod:`causal_grounding.probability_backends`.

    Args:
        method: Registered predictor backend name.
        **kwargs: Arguments passed to predictor constructor

    Returns:
        BasePredictorModel instance

    Example:
        predictor = create_predictor('xgboost', use_grid_search=True)
        predictor = create_predictor('empirical', smoothing=0.1)
        predictor = create_predictor('hierarchical_dirichlet')
        predictor = create_predictor('hierarchical_dirichlet_draws')
        predictor = create_predictor('tabicl_v2', device='cuda')
    """
    from .probability_backends import (
        get_probability_backend,
        probability_backend_names,
    )

    try:
        factory = get_probability_backend(method).predictor_factory
    except KeyError:
        factory = None
    if factory is None:
        available = ", ".join(repr(name) for name in probability_backend_names(predictor=True))
        raise ValueError(f"Unknown method: {method}. Use one of {available}.")
    return factory(**kwargs)


def _create_xgboost_predictor(**kwargs: Any) -> 'XGBoostPredictor':
    """Create an :class:`XGBoostPredictor`, failing early when XGBoost is missing.

    Args:
        **kwargs: Arguments passed to the ``XGBoostPredictor`` constructor.

    Returns:
        The new predictor.

    Raises:
        ImportError: If XGBoost is not installed.
    """
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost not available. Install with: pip install xgboost"
        )
    return XGBoostPredictor(**kwargs)

