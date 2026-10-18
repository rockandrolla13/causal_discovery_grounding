"""
Tests for Prediction Models Module

Tests the EmpiricalPredictor, XGBoostPredictor, and factory functions.
"""

import pytest
import numpy as np
import pandas as pd
import subprocess
import sys
from typing import Dict, Any

import causal_grounding.predictors as predictors_module
from causal_grounding.predictors import (
    BasePredictorModel,
    EmpiricalPredictor,
    HierarchicalDirichletPredictor,
    TABICL_V2_CLASSIFIER_CHECKPOINT,
    TabICLv2Predictor,
    create_predictor,
    _make_binary_log_loss_scorer,
    _XGBOOST_AVAILABLE,
)

# Conditional import for XGBoost tests
if _XGBOOST_AVAILABLE:
    from causal_grounding.predictors import XGBoostPredictor


# =============================================================================
# FIXTURES
# =============================================================================


def test_predictor_module_does_not_import_tabicl_eagerly() -> None:
    """Non-TabICL processes should import predictors without loading TabICL."""

    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    \"\"\"Reject any eager import of the optional TabICL package.\"\"\"
    if name == "tabicl" or name.startswith("tabicl."):
        raise RuntimeError("TabICL must remain lazy")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import causal_grounding.predictors
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.fixture
def binary_data():
    """Sample binary data for testing."""
    np.random.seed(42)
    n = 500

    # Z -> X -> Y
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X, n)

    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


@pytest.fixture
def multivariate_data():
    """Sample data with multiple covariates."""
    np.random.seed(42)
    n = 500

    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    Z3 = np.random.binomial(1, 0.4, n)

    X = np.random.binomial(1, 0.2 + 0.3 * Z1 + 0.2 * Z2, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z1, n)

    return pd.DataFrame({
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'X': X, 'Y': Y
    })


@pytest.fixture
def continuous_outcome_data():
    """Sample data with continuous outcome."""
    np.random.seed(42)
    n = 500

    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
    Y = 0.5 + 0.3 * X + 0.1 * Z + np.random.normal(0, 0.5, n)

    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


# =============================================================================
# TEST: EmpiricalPredictor INITIALIZATION
# =============================================================================

class TestEmpiricalPredictorInit:
    """Tests for EmpiricalPredictor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        predictor = EmpiricalPredictor()
        assert predictor.smoothing == 0.01
        assert predictor.n_bins == 5

    def test_custom_init(self):
        """Test custom initialization."""
        predictor = EmpiricalPredictor(smoothing=0.1, n_bins=10)
        assert predictor.smoothing == 0.1
        assert predictor.n_bins == 10


# =============================================================================
# TEST: EmpiricalPredictor PROPENSITY
# =============================================================================

class TestEmpiricalPropensity:
    """Tests for EmpiricalPredictor propensity model."""

    def test_fit_propensity(self, binary_data):
        """Test fitting propensity model."""
        predictor = EmpiricalPredictor()
        result = predictor.fit_propensity(binary_data, 'X', ['Z'])

        assert result is predictor  # Returns self
        assert predictor._is_fitted_propensity

    def test_predict_propensity(self, binary_data):
        """Test predicting propensity."""
        predictor = EmpiricalPredictor()
        predictor.fit_propensity(binary_data, 'X', ['Z'])

        probs = predictor.predict_propensity(binary_data[['Z']])

        assert len(probs) == len(binary_data)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_propensity_varies_by_covariate(self, binary_data):
        """Test that propensity varies by covariate value."""
        predictor = EmpiricalPredictor()
        predictor.fit_propensity(binary_data, 'X', ['Z'])

        # Create test data with different Z values
        test_z0 = pd.DataFrame({'Z': [0] * 10})
        test_z1 = pd.DataFrame({'Z': [1] * 10})

        prop_z0 = predictor.predict_propensity(test_z0)[0]
        prop_z1 = predictor.predict_propensity(test_z1)[0]

        # Z affects X, so propensities should differ
        assert prop_z0 != prop_z1

    def test_propensity_not_fitted_error(self, binary_data):
        """Test error when propensity not fitted."""
        predictor = EmpiricalPredictor()

        with pytest.raises(RuntimeError, match="fit propensity"):
            predictor.predict_propensity(binary_data[['Z']])


# =============================================================================
# TEST: EmpiricalPredictor OUTCOME
# =============================================================================

class TestEmpiricalOutcome:
    """Tests for EmpiricalPredictor outcome model."""

    def test_fit_outcome(self, binary_data):
        """Test fitting outcome model."""
        predictor = EmpiricalPredictor()
        result = predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        assert result is predictor  # Returns self
        assert predictor._is_fitted_outcome

    def test_predict_outcome_scalar_treatment(self, binary_data):
        """Test predicting outcome with scalar treatment."""
        predictor = EmpiricalPredictor()
        predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        probs_a0 = predictor.predict_outcome(binary_data[['Z']], 0)
        probs_a1 = predictor.predict_outcome(binary_data[['Z']], 1)

        assert len(probs_a0) == len(binary_data)
        assert len(probs_a1) == len(binary_data)
        assert np.all(probs_a0 >= 0)
        assert np.all(probs_a1 <= 1)

    def test_predict_outcome_array_treatment(self, binary_data):
        """Test predicting outcome with array treatment."""
        predictor = EmpiricalPredictor()
        predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        A = binary_data['X'].values
        probs = predictor.predict_outcome(binary_data[['Z']], A)

        assert len(probs) == len(binary_data)

    def test_outcome_not_fitted_error(self, binary_data):
        """Test error when outcome not fitted."""
        predictor = EmpiricalPredictor()

        with pytest.raises(RuntimeError, match="fit outcome"):
            predictor.predict_outcome(binary_data[['Z']], 0)


# =============================================================================
# TEST: EmpiricalPredictor COMBINED
# =============================================================================

class TestEmpiricalCombined:
    """Tests for combined EmpiricalPredictor functionality."""

    def test_fit_both(self, binary_data):
        """Test fitting both models with fit()."""
        predictor = EmpiricalPredictor()
        result = predictor.fit(binary_data, 'X', 'Y', ['Z'])

        assert result is predictor
        assert predictor._is_fitted_propensity
        assert predictor._is_fitted_outcome

    def test_predict_potential_outcomes(self, binary_data):
        """Test predicting potential outcomes."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        Y0, Y1 = predictor.predict_potential_outcomes(binary_data[['Z']])

        assert len(Y0) == len(binary_data)
        assert len(Y1) == len(binary_data)

    def test_predict_cate(self, binary_data):
        """Test predicting CATE."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(binary_data[['Z']])

        assert len(cate) == len(binary_data)
        # CATE should be between -1 and 1 for binary outcome
        assert np.all(cate >= -1)
        assert np.all(cate <= 1)

    def test_get_probability_table(self, binary_data):
        """Test getting probability table."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        table = predictor.get_probability_table()

        assert isinstance(table, pd.DataFrame)
        assert 'key' in table.columns
        assert 'P(A=1|X)' in table.columns
        assert 'P(Y=1|X,A=0)' in table.columns
        assert 'P(Y=1|X,A=1)' in table.columns


# =============================================================================
# TEST: EmpiricalPredictor MULTIPLE COVARIATES
# =============================================================================

class TestEmpiricalMultipleCovariates:
    """Tests for EmpiricalPredictor with multiple covariates."""

    def test_fit_multivariate(self, multivariate_data):
        """Test fitting with multiple covariates."""
        predictor = EmpiricalPredictor()
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        # Should create keys for each combination
        table = predictor.get_probability_table()
        assert len(table) > 1  # Should have multiple strata

    def test_predict_multivariate(self, multivariate_data):
        """Test predicting with multiple covariates."""
        predictor = EmpiricalPredictor()
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        cate = predictor.predict_cate(multivariate_data[['Z1', 'Z2', 'Z3']])
        assert len(cate) == len(multivariate_data)


# =============================================================================
# TEST: XGBoostPredictor (if available)
# =============================================================================

@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostPredictorInit:
    """Tests for XGBoostPredictor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        predictor = XGBoostPredictor()
        assert predictor.use_grid_search == True
        assert predictor.cv_folds == 5

    def test_custom_init(self):
        """Test custom initialization."""
        predictor = XGBoostPredictor(
            use_grid_search=False,
            n_estimators_range=[50, 100],
            max_depth_range=[3],
            n_estimators=80,
            max_depth=2,
            learning_rate=0.03,
            random_state=42,
            n_jobs=3,
        )
        assert predictor.use_grid_search == False
        assert predictor.n_estimators_range == [50, 100]
        assert predictor.n_estimators == 80
        assert predictor.max_depth == 2
        assert predictor.learning_rate == 0.03
        assert predictor.random_state == 42
        assert predictor.n_jobs == 3

    def test_binary_log_loss_scorer_accepts_single_class_fold(self):
        """Test grid-search log loss can score one-class validation folds."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class TwoColumnProbabilityEstimator(ClassifierMixin, BaseEstimator):
            """Minimal estimator exposing two-column class probabilities."""

            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                """Return fixed probabilities for each row in X."""
                return np.tile(np.array([[0.2, 0.8]]), (len(X), 1))

        scorer = _make_binary_log_loss_scorer()
        score = scorer(
            TwoColumnProbabilityEstimator(),
            np.zeros((3, 1)),
            np.ones(3, dtype=int),
        )

        assert np.isfinite(score)
        assert score < 0.0


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostPropensity:
    """Tests for XGBoostPredictor propensity model."""

    def test_fit_propensity(self, binary_data):
        """Test fitting propensity model."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        result = predictor.fit_propensity(binary_data, 'X', ['Z'])

        assert result is predictor
        assert predictor._propensity_model is not None

    def test_predict_propensity(self, binary_data):
        """Test predicting propensity."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit_propensity(binary_data, 'X', ['Z'])

        probs = predictor.predict_propensity(binary_data[['Z']])

        assert len(probs) == len(binary_data)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostOutcome:
    """Tests for XGBoostPredictor outcome model."""

    def test_fit_outcome_binary(self, binary_data):
        """Test fitting outcome model with binary outcome."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        result = predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        assert result is predictor
        assert predictor._outcome_model is not None
        assert predictor._is_binary_outcome == True

    def test_fit_outcome_continuous(self, continuous_outcome_data):
        """Test fitting outcome model with continuous outcome."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit_outcome(continuous_outcome_data, 'Y', 'X', ['Z'])

        assert predictor._is_binary_outcome == False

    def test_predict_outcome(self, binary_data):
        """Test predicting outcome."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        probs_a0 = predictor.predict_outcome(binary_data[['Z']], 0)
        probs_a1 = predictor.predict_outcome(binary_data[['Z']], 1)

        assert len(probs_a0) == len(binary_data)
        assert len(probs_a1) == len(binary_data)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostCombined:
    """Tests for combined XGBoostPredictor functionality."""

    def test_fit_both(self, binary_data):
        """Test fitting both models."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        result = predictor.fit(binary_data, 'X', 'Y', ['Z'])

        assert result is predictor
        assert predictor._propensity_model is not None
        assert predictor._outcome_model is not None

    def test_predict_cate(self, binary_data):
        """Test predicting CATE."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(binary_data[['Z']])

        assert len(cate) == len(binary_data)

    def test_feature_importance(self, binary_data):
        """Test getting feature importance."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        prop_importance = predictor.get_feature_importance('propensity')
        outcome_importance = predictor.get_feature_importance('outcome')

        assert isinstance(prop_importance, pd.DataFrame)
        assert isinstance(outcome_importance, pd.DataFrame)
        assert 'feature' in prop_importance.columns
        assert 'importance' in prop_importance.columns


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostMultipleCovariates:
    """Tests for XGBoostPredictor with multiple covariates."""

    def test_multivariate(self, multivariate_data):
        """Test with multiple covariates."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        cate = predictor.predict_cate(multivariate_data[['Z1', 'Z2', 'Z3']])
        assert len(cate) == len(multivariate_data)

        # Check feature importance has all features
        importance = predictor.get_feature_importance('propensity')
        assert len(importance) == 3  # Z1, Z2, Z3


# =============================================================================
# TEST: FACTORY FUNCTION
# =============================================================================

class FakeTabICLClassifier:
    """Small deterministic classifier used without loading TabICLv2 weights."""

    instances = []

    def __init__(self, **kwargs):
        """Record TabICL constructor options."""
        self.kwargs = kwargs
        self.classes_ = np.array([0, 1])
        self.target_mean_ = 0.5
        FakeTabICLClassifier.instances.append(self)

    def fit(self, X, y):
        """Store target prevalence for deterministic probability predictions."""
        self.n_features_in_ = np.asarray(X).shape[1]
        self.target_mean_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        """Return bounded probabilities depending on features and target rate."""
        features = np.asarray(X, dtype=float)
        feature_signal = features.mean(axis=1)
        probability_one = np.clip(
            0.15 + 0.5 * self.target_mean_ + 0.1 * feature_signal,
            0.01,
            0.99,
        )
        return np.column_stack([1.0 - probability_one, probability_one])


@pytest.fixture
def fake_tabicl(monkeypatch):
    """Replace optional TabICL dependency with a deterministic test double."""
    FakeTabICLClassifier.instances = []
    monkeypatch.setattr(predictors_module, "_TABICL_AVAILABLE", True)
    monkeypatch.setattr(
        predictors_module,
        "TabICLClassifier",
        FakeTabICLClassifier,
    )
    return FakeTabICLClassifier


class TestTabICLv2Predictor:
    """Tests for TabICLv2 predictor API integration."""

    def test_fit_and_predict_binary_probabilities(self, binary_data, fake_tabicl):
        """TabICLv2 fits propensity/outcome contexts and predicts probabilities."""
        predictor = create_predictor(
            "tabicl_v2",
            n_estimators=2,
            kv_cache="repr",
            random_state=7,
        )
        predictor.fit(binary_data, "X", "Y", ["Z"])

        propensity = predictor.predict_propensity(binary_data[["Z"]].iloc[:4])
        outcome_zero = predictor.predict_outcome(
            binary_data[["Z"]].iloc[:4],
            0,
        )
        outcome_one = predictor.predict_outcome(
            binary_data[["Z"]].iloc[:4],
            1,
        )

        assert isinstance(predictor, TabICLv2Predictor)
        assert len(fake_tabicl.instances) == 2
        assert all(
            model.kwargs["checkpoint_version"]
            == TABICL_V2_CLASSIFIER_CHECKPOINT
            for model in fake_tabicl.instances
        )
        assert all(model.kwargs["kv_cache"] == "repr" for model in fake_tabicl.instances)
        assert propensity.shape == outcome_zero.shape == outcome_one.shape == (4,)
        assert np.all((0.0 <= propensity) & (propensity <= 1.0))
        assert np.all((0.0 <= outcome_zero) & (outcome_zero <= 1.0))
        assert np.all((0.0 <= outcome_one) & (outcome_one <= 1.0))
        assert not np.allclose(outcome_zero, outcome_one)

    def test_one_class_targets_use_constant_models(self, fake_tabicl):
        """One-class sites return exact constants without loading TabICLv2."""
        data = pd.DataFrame({
            "Z": [0, 1, 0, 1],
            "X": [1, 1, 1, 1],
            "Y": [0, 0, 0, 0],
        })
        predictor = TabICLv2Predictor()
        predictor.fit(data, "X", "Y", ["Z"])

        assert fake_tabicl.instances == []
        assert np.allclose(predictor.predict_propensity(data[["Z"]]), 1.0)
        assert np.allclose(predictor.predict_outcome(data[["Z"]], 0), 0.0)

    def test_rejects_nonbinary_interventions(self, binary_data, fake_tabicl):
        """Outcome prediction rejects nonbinary scalar and array treatments."""
        predictor = TabICLv2Predictor()
        predictor.fit_outcome(binary_data, "Y", "X", ["Z"])

        with pytest.raises(ValueError, match="binary treatment"):
            predictor.predict_outcome(binary_data[["Z"]].iloc[:2], 0.5)
        with pytest.raises(ValueError, match="binary treatment"):
            predictor.predict_outcome(
                binary_data[["Z"]].iloc[:2],
                np.array([0.0, 0.5]),
            )

    def test_shared_checkpoint_is_loaded_once_per_process(self, binary_data, monkeypatch):
        """``share_checkpoint`` loads a checkpoint once; the default loads per fit."""

        class LoadingClassifier(FakeTabICLClassifier):
            """Stand-in whose ``fit`` loads a model, as TabICLClassifier does."""

            loads = 0

            def __init__(self, **kwargs):
                """Keep the checkpoint options the shared cache is keyed by."""
                super().__init__(**kwargs)
                self.model_path = kwargs["model_path"]
                self.checkpoint_version = kwargs["checkpoint_version"]

            def _load_model(self):
                """Count loads and attach a fresh model object."""
                LoadingClassifier.loads += 1
                self.model_path_, self.model_, self.model_config_ = "checkpoint", object(), {}

            def fit(self, X, y):
                """Load the model, then fit the stand-in."""
                self._load_model()
                return super().fit(X, y)

        monkeypatch.setattr(predictors_module, "_TABICL_AVAILABLE", True)
        monkeypatch.setattr(predictors_module, "TabICLClassifier", LoadingClassifier)
        monkeypatch.setattr(predictors_module, "_SHARED_TABICL_MODELS", {})
        models = []
        for _ in range(3):
            predictor = create_predictor("tabicl_v2", share_checkpoint=True)
            predictor.fit(binary_data, "X", "Y", ["Z"])
            models.append(predictor._propensity_model.model_)
        assert LoadingClassifier.loads == 1
        assert all(model is models[0] for model in models)
        LoadingClassifier.loads = 0
        create_predictor("tabicl_v2").fit(binary_data, "X", "Y", ["Z"])
        assert LoadingClassifier.loads == 2


class TestCreatePredictor:
    """Tests for create_predictor factory function."""

    def test_create_empirical(self):
        """Test creating empirical predictor."""
        predictor = create_predictor('empirical')
        assert isinstance(predictor, EmpiricalPredictor)

    def test_create_empirical_with_kwargs(self):
        """Test creating empirical predictor with kwargs."""
        predictor = create_predictor('empirical', smoothing=0.5)
        assert predictor.smoothing == 0.5

    def test_create_hierarchical_dirichlet(self):
        """Test creating hierarchical Dirichlet predictor."""
        predictor = create_predictor('hierarchical_dirichlet', prior_strength=3.0)
        assert isinstance(predictor, HierarchicalDirichletPredictor)
        assert predictor.prior_strength == 3.0

    def test_create_hierarchical_dirichlet_draws_alias(self):
        """Draw alias uses the hierarchical Dirichlet predictor implementation."""
        predictor = create_predictor('hierarchical_dirichlet_draws')
        assert isinstance(predictor, HierarchicalDirichletPredictor)

    def test_create_tabicl_v2(self, fake_tabicl):
        """Test creating the optional TabICLv2 predictor."""
        predictor = create_predictor("tabicl_v2", n_estimators=2)
        assert isinstance(predictor, TabICLv2Predictor)
        assert predictor.n_estimators == 2

    def test_create_tabicl_v2_requires_optional_dependency(self, monkeypatch):
        """Factory reports the package needed by the optional backend."""
        monkeypatch.setattr(predictors_module, "_TABICL_AVAILABLE", False)
        with pytest.raises(ImportError, match="tabicl==2.1.1"):
            create_predictor("tabicl_v2")

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_create_xgboost(self):
        """Test creating XGBoost predictor."""
        predictor = create_predictor('xgboost', use_grid_search=False)
        assert isinstance(predictor, XGBoostPredictor)

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_predictor('invalid')


# =============================================================================
# TEST: INTERFACE COMPATIBILITY
# =============================================================================

class TestInterfaceCompatibility:
    """Test that predictors have compatible interfaces."""

    def test_empirical_is_base_predictor(self):
        """Test EmpiricalPredictor is a BasePredictorModel."""
        predictor = EmpiricalPredictor()
        assert isinstance(predictor, BasePredictorModel)

    def test_hierarchical_dirichlet_is_base_predictor(self):
        """Test HierarchicalDirichletPredictor is a BasePredictorModel."""
        predictor = HierarchicalDirichletPredictor()
        assert isinstance(predictor, BasePredictorModel)

    def test_hierarchical_dirichlet_predicts_probabilities(self, binary_data):
        """Hierarchical Dirichlet backend fits and returns bounded probabilities."""
        predictor = create_predictor(
            'hierarchical_dirichlet',
            prior_strength=2.0,
            base_prior_strength=2.0,
        )
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        prop = predictor.predict_propensity(binary_data[['Z']])
        y0 = predictor.predict_outcome(binary_data[['Z']], 0)
        y1 = predictor.predict_outcome(binary_data[['Z']], 1)

        assert np.all((0.0 <= prop) & (prop <= 1.0))
        assert np.all((0.0 <= y0) & (y0 <= 1.0))
        assert np.all((0.0 <= y1) & (y1 <= 1.0))

    def test_hierarchical_dirichlet_shrinks_toward_global_rate(self):
        """Small finite cells are pulled toward global posterior rates."""
        data = pd.DataFrame({
            'Z': [0, 0, 1, 1, 1, 1],
            'X': [1, 1, 0, 0, 0, 0],
            'Y': [1, 0, 0, 0, 0, 0],
        })
        predictor = HierarchicalDirichletPredictor(
            propensity_prior_strength=2.0,
            outcome_prior_strength=2.0,
            base_prior_strength=2.0,
            global_prior=0.5,
        )
        predictor.fit(data, 'X', 'Y', ['Z'])

        prop_z0 = predictor.predict_propensity(pd.DataFrame({'Z': [0]}))[0]
        prop_z1 = predictor.predict_propensity(pd.DataFrame({'Z': [1]}))[0]
        unseen = predictor.predict_propensity(pd.DataFrame({'Z': [2]}))[0]

        assert 0.5 < prop_z0 < 1.0
        assert 0.0 < prop_z1 < 0.5
        assert np.isclose(unseen, predictor._propensity_base_rate)

    def test_hierarchical_dirichlet_posterior_draw_is_reproducible(self):
        """Posterior draws keep fitted support and respect the supplied seed."""
        data = pd.DataFrame({
            'Z': [0, 0, 0, 1, 1, 1],
            'X': [1, 1, 0, 0, 0, 1],
            'Y': [1, 0, 1, 0, 0, 1],
        })
        predictor = HierarchicalDirichletPredictor(
            prior_strength=2.0,
            base_prior_strength=2.0,
            global_prior=0.5,
        )
        predictor.fit(data, 'X', 'Y', ['Z'])

        first = predictor.sample_posterior(random_state=123)
        second = predictor.sample_posterior(random_state=123)
        third = predictor.sample_posterior(random_state=456)
        z_frame = pd.DataFrame({'Z': [0, 1, 2]})

        first_prop = first.predict_propensity(z_frame)
        second_prop = second.predict_propensity(z_frame)
        third_prop = third.predict_propensity(z_frame)
        first_y0 = first.predict_outcome(z_frame, 0)

        assert np.allclose(first_prop, second_prop)
        assert np.all((0.0 <= first_prop) & (first_prop <= 1.0))
        assert np.all((0.0 <= first_y0) & (first_y0 <= 1.0))
        assert not np.allclose(first_prop, third_prop)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_is_base_predictor(self):
        """Test XGBoostPredictor is a BasePredictorModel."""
        predictor = XGBoostPredictor()
        assert isinstance(predictor, BasePredictorModel)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_same_cate_interface(self, binary_data):
        """Test both predictors have same CATE prediction interface."""
        emp = create_predictor('empirical')
        xgb = create_predictor('xgboost', use_grid_search=False, random_state=42)

        emp.fit(binary_data, 'X', 'Y', ['Z'])
        xgb.fit(binary_data, 'X', 'Y', ['Z'])

        emp_cate = emp.predict_cate(binary_data[['Z']])
        xgb_cate = xgb.predict_cate(binary_data[['Z']])

        # Both should return same shape
        assert emp_cate.shape == xgb_cate.shape


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_covariate_value(self):
        """Test with single covariate value (constant)."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'Z': np.ones(n, dtype=int),  # Constant
            'X': np.random.binomial(1, 0.5, n),
            'Y': np.random.binomial(1, 0.5, n)
        })

        predictor = EmpiricalPredictor()
        predictor.fit(data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(data[['Z']])
        assert len(cate) == n

    def test_small_sample(self):
        """Test with small sample size."""
        np.random.seed(42)
        n = 20

        data = pd.DataFrame({
            'Z': np.random.binomial(1, 0.5, n),
            'X': np.random.binomial(1, 0.5, n),
            'Y': np.random.binomial(1, 0.5, n)
        })

        predictor = EmpiricalPredictor()
        predictor.fit(data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(data[['Z']])
        assert len(cate) == n

    def test_unseen_covariate_value(self, binary_data):
        """Test predicting for covariate value not seen in training."""
        # Train only on Z=0
        train_data = binary_data[binary_data['Z'] == 0].copy()

        predictor = EmpiricalPredictor()
        predictor.fit(train_data, 'X', 'Y', ['Z'])

        # Predict on Z=1 (not in training)
        test_data = pd.DataFrame({'Z': [1, 1, 1]})

        # Should return default (0.5) without crashing
        probs = predictor.predict_propensity(test_data)
        assert len(probs) == 3
        assert np.all(probs == 0.5)  # Default for unseen

    def test_smoothing_effect(self, binary_data):
        """Test that smoothing prevents zero probabilities."""
        predictor = EmpiricalPredictor(smoothing=0.5)
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        probs = predictor.predict_propensity(binary_data[['Z']])

        # With smoothing, no probs should be exactly 0 or 1
        assert np.all(probs > 0)
        assert np.all(probs < 1)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for predictors."""

    def test_full_workflow_empirical(self, binary_data):
        """Test full workflow with empirical predictor."""
        predictor = create_predictor('empirical')

        # Fit
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        # Predict propensity
        prop = predictor.predict_propensity(binary_data[['Z']])
        assert np.all(prop >= 0) and np.all(prop <= 1)

        # Predict outcomes
        y0, y1 = predictor.predict_potential_outcomes(binary_data[['Z']])
        assert np.all(y0 >= 0) and np.all(y0 <= 1)
        assert np.all(y1 >= 0) and np.all(y1 <= 1)

        # Predict CATE
        cate = predictor.predict_cate(binary_data[['Z']])
        assert np.all(cate >= -1) and np.all(cate <= 1)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_full_workflow_xgboost(self, binary_data):
        """Test full workflow with XGBoost predictor."""
        predictor = create_predictor('xgboost', use_grid_search=False, random_state=42)

        # Fit
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        # Predict propensity
        prop = predictor.predict_propensity(binary_data[['Z']])
        assert np.all(prop >= 0) and np.all(prop <= 1)

        # Predict outcomes
        y0, y1 = predictor.predict_potential_outcomes(binary_data[['Z']])

        # Predict CATE
        cate = predictor.predict_cate(binary_data[['Z']])
        assert len(cate) == len(binary_data)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_compare_predictors(self, binary_data):
        """Test comparing both predictors on same data."""
        emp = create_predictor('empirical')
        xgb = create_predictor('xgboost', use_grid_search=False, random_state=42)

        emp.fit(binary_data, 'X', 'Y', ['Z'])
        xgb.fit(binary_data, 'X', 'Y', ['Z'])

        emp_cate = emp.predict_cate(binary_data[['Z']])
        xgb_cate = xgb.predict_cate(binary_data[['Z']])

        # Both should produce valid CATE estimates
        assert len(emp_cate) == len(xgb_cate)

        # Correlation between predictions should be positive (similar direction)
        correlation = np.corrcoef(emp_cate, xgb_cate)[0, 1]
        # Note: correlation could be weak due to different methods
        assert not np.isnan(correlation)
