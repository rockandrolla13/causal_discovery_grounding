"""
Test discretize.py module - covariate discretization functions.
"""

import pytest
import numpy as np
import pandas as pd
from causal_grounding import (
    discretize_age,
    discretize_polideo,
    discretize_covariates,
    get_discretized_covariate_names
)


# =============================================================================
# TEST CLASS: TestDiscretizeAge
# =============================================================================

class TestDiscretizeAge:
    """Test discretize_age function."""

    def test_default_five_bins(self):
        """Test that default discretization creates 5 bins."""
        ages = pd.Series(np.linspace(18, 80, 100))
        
        result = discretize_age(ages)
        
        assert len(result.unique()) == 5, f"Expected 5 bins, got {len(result.unique())}"

    def test_custom_bins(self):
        """Test discretization with custom number of bins."""
        ages = pd.Series(np.linspace(18, 80, 100))
        
        result = discretize_age(ages, n_bins=3)
        
        assert len(result.unique()) == 3, f"Expected 3 bins, got {len(result.unique())}"

    def test_handles_missing_values(self):
        """Test that NaN values are handled properly."""
        ages = pd.Series([25, 35, np.nan, 45, 55, np.nan, 65])
        
        result = discretize_age(ages)
        
        # NaN values should remain NaN
        assert result.isna().sum() == 2, "NaN values should be preserved"

    def test_quantile_distribution(self):
        """Test that bins have approximately equal sizes."""
        np.random.seed(42)
        ages = pd.Series(np.random.uniform(18, 80, 1000))
        
        result = discretize_age(ages, n_bins=5)
        
        # Each bin should have approximately 200 values (within 15% tolerance)
        value_counts = result.value_counts()
        for count in value_counts:
            assert 170 < count < 230, f"Bin count {count} outside expected range"

    def test_result_is_categorical_or_integer(self):
        """Test that result has appropriate dtype."""
        ages = pd.Series(np.linspace(18, 80, 50))
        
        result = discretize_age(ages)
        
        # Should be categorical or numeric
        assert result.dtype.name in ['category', 'int64', 'int32', 'float64'] or \
               pd.api.types.is_categorical_dtype(result) or \
               pd.api.types.is_numeric_dtype(result)


# =============================================================================
# TEST CLASS: TestDiscretizePolideo
# =============================================================================

class TestDiscretizePolideo:
    """Test discretize_polideo function."""

    def test_three_categories(self):
        """Test that discretization creates exactly 3 categories."""
        polideo = pd.Series(list(range(7)) * 10)  # 0-6 repeated
        
        result = discretize_polideo(polideo)
        
        assert len(result.unique()) == 3, f"Expected 3 categories, got {len(result.unique())}"

    def test_conservative_mapping(self):
        """Test that 0, 1, 2 map to conservative (0)."""
        polideo = pd.Series([0, 1, 2])
        
        result = discretize_polideo(polideo)
        
        assert all(result == 0), "Values 0, 1, 2 should all map to 0 (conservative)"

    def test_moderate_mapping(self):
        """Test that 3 maps to moderate (1)."""
        polideo = pd.Series([3])
        
        result = discretize_polideo(polideo)
        
        assert result.iloc[0] == 1, "Value 3 should map to 1 (moderate)"

    def test_liberal_mapping(self):
        """Test that 4, 5, 6 map to liberal (2)."""
        polideo = pd.Series([4, 5, 6])
        
        result = discretize_polideo(polideo)
        
        assert all(result == 2), "Values 4, 5, 6 should all map to 2 (liberal)"

    def test_handles_edge_values(self):
        """Test exact boundary values 0 and 6."""
        polideo = pd.Series([0, 6])
        
        result = discretize_polideo(polideo)
        
        assert result.iloc[0] == 0, "Value 0 should map to 0"
        assert result.iloc[1] == 2, "Value 6 should map to 2"


# =============================================================================
# TEST CLASS: TestDiscretizeCovariates
# =============================================================================

class TestDiscretizeCovariates:
    """Test discretize_covariates function."""

    def test_adds_cat_columns(self):
        """Test that _cat columns are added."""
        df = pd.DataFrame({
            'resp_age': np.random.uniform(18, 70, 100),
            'resp_gender': np.random.choice([0, 1], 100),
            'resp_polideo': np.random.choice(range(7), 100)
        })
        
        result = discretize_covariates(df)
        
        assert 'resp_age_cat' in result.columns
        assert 'resp_polideo_cat' in result.columns
        # Original columns should still exist
        assert 'resp_age' in result.columns
        assert 'resp_polideo' in result.columns

    def test_preserves_other_columns(self):
        """Test that non-covariate columns are preserved."""
        df = pd.DataFrame({
            'iv': np.random.choice([0, 1], 50),
            'dv': np.random.randn(50),
            'site': ['A'] * 50,
            'resp_age': np.random.uniform(18, 70, 50),
            'resp_gender': np.random.choice([0, 1], 50),
            'resp_polideo': np.random.choice(range(7), 50)
        })
        
        result = discretize_covariates(df)
        
        assert 'iv' in result.columns
        assert 'dv' in result.columns
        assert 'site' in result.columns
        # Values should be unchanged
        pd.testing.assert_series_equal(result['iv'], df['iv'])
        pd.testing.assert_series_equal(result['dv'], df['dv'])

    def test_returns_copy(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'resp_age': [25, 35, 45, 55, 65],
            'resp_gender': [0, 1, 0, 1, 0],
            'resp_polideo': [0, 2, 3, 4, 6]
        })
        original_columns = list(df.columns)
        
        result = discretize_covariates(df)
        result['new_col'] = 1  # Modify the result
        
        # Original should be unchanged
        assert list(df.columns) == original_columns
        assert 'new_col' not in df.columns

    def test_gender_unchanged(self):
        """Test that resp_gender is not modified (already discrete)."""
        df = pd.DataFrame({
            'resp_age': np.random.uniform(18, 70, 50),
            'resp_gender': np.random.choice([0, 1], 50),
            'resp_polideo': np.random.choice(range(7), 50)
        })
        
        result = discretize_covariates(df)
        
        # Gender should remain unchanged
        pd.testing.assert_series_equal(
            result['resp_gender'], 
            df['resp_gender'],
            check_names=True
        )


# =============================================================================
# TEST CLASS: TestGetDiscretizedCovariateNames
# =============================================================================

class TestGetDiscretizedCovariateNames:
    """Test get_discretized_covariate_names function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_discretized_covariate_names()
        
        assert isinstance(result, list)

    def test_contains_expected_names(self):
        """Test that default names are correct."""
        result = get_discretized_covariate_names()
        
        assert 'resp_age_cat' in result
        assert 'resp_gender' in result
        assert 'resp_polideo_cat' in result

    def test_length(self):
        """Test that exactly 3 covariate names are returned."""
        result = get_discretized_covariate_names()
        
        assert len(result) == 3

    def test_custom_column_names(self):
        """Test with custom column name arguments."""
        result = get_discretized_covariate_names(
            age_col='age',
            gender_col='gender',
            polideo_col='ideology'
        )
        
        assert 'age_cat' in result
        assert 'gender' in result
        assert 'ideology_cat' in result
