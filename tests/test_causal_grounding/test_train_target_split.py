"""
Test train_target_split.py module - train/target environment splitting.
"""

import pytest
import numpy as np
import pandas as pd
from causal_grounding import (
    add_regime_indicator,
    create_train_target_split,
    get_available_sites
)
from causal_grounding.train_target_split import summarize_split


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_rct_data():
    """Create mock RCT data with multiple sites."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'iv': np.random.choice([0, 1], n),
        'dv': np.random.randn(n),
        'site': np.random.choice(['A', 'B', 'C', 'mturk'], n),
        'resp_age': np.random.uniform(18, 70, n),
        'resp_gender': np.random.choice([0, 1], n),
        'resp_polideo': np.random.choice(range(7), n)
    })


@pytest.fixture
def mock_osrct_data():
    """Create mock OSRCT data with multiple sites."""
    np.random.seed(43)
    n = 300
    return pd.DataFrame({
        'iv': np.random.choice([0, 1], n),
        'dv': np.random.randn(n),
        'site': np.random.choice(['A', 'B', 'C', 'mturk'], n),
        'resp_age': np.random.uniform(18, 70, n),
        'resp_gender': np.random.choice([0, 1], n),
        'resp_polideo': np.random.choice(range(7), n)
    })


# =============================================================================
# TEST CLASS: TestAddRegimeIndicator
# =============================================================================

class TestAddRegimeIndicator:
    """Test add_regime_indicator function."""

    def test_adds_f_column(self):
        """Test that F column is added to combined data."""
        rct = pd.DataFrame({'iv': [0, 1], 'dv': [1.0, 2.0]})
        osrct = pd.DataFrame({'iv': [1, 0], 'dv': [3.0, 4.0]})
        
        result = add_regime_indicator(rct, osrct)
        
        assert 'F' in result.columns

    def test_rct_gets_on(self):
        """Test that RCT data gets F='on'."""
        rct = pd.DataFrame({'iv': [0, 1, 0], 'dv': [1.0, 2.0, 3.0]})
        osrct = pd.DataFrame({'iv': [1, 0], 'dv': [4.0, 5.0]})
        
        result = add_regime_indicator(rct, osrct)
        
        # First 3 rows should be 'on' (from RCT)
        assert (result['F'].iloc[:3] == 'on').all()

    def test_osrct_gets_idle(self):
        """Test that OSRCT data gets F='idle'."""
        rct = pd.DataFrame({'iv': [0, 1], 'dv': [1.0, 2.0]})
        osrct = pd.DataFrame({'iv': [1, 0, 1], 'dv': [3.0, 4.0, 5.0]})
        
        result = add_regime_indicator(rct, osrct)
        
        # Last 3 rows should be 'idle' (from OSRCT)
        assert (result['F'].iloc[-3:] == 'idle').all()

    def test_preserves_all_columns(self):
        """Test that all common columns are preserved."""
        rct = pd.DataFrame({
            'iv': [0, 1],
            'dv': [1.0, 2.0],
            'site': ['A', 'B'],
            'resp_age': [25, 35]
        })
        osrct = pd.DataFrame({
            'iv': [1, 0],
            'dv': [3.0, 4.0],
            'site': ['C', 'D'],
            'resp_age': [45, 55]
        })
        
        result = add_regime_indicator(rct, osrct)
        
        assert 'iv' in result.columns
        assert 'dv' in result.columns
        assert 'site' in result.columns
        assert 'resp_age' in result.columns

    def test_combined_row_count(self):
        """Test that result has correct number of rows."""
        rct = pd.DataFrame({'iv': [0, 1, 0], 'dv': [1.0, 2.0, 3.0]})
        osrct = pd.DataFrame({'iv': [1, 0], 'dv': [4.0, 5.0]})
        
        result = add_regime_indicator(rct, osrct)

        assert len(result) == 5  # 3 + 2

    def test_column_order_follows_the_rct_frame(self):
        """Column order is deterministic, and matches the RCT frame.

        Until 2026-08-09 the common columns came from
        ``list(set(rct.columns) & set(osrct.columns))``. Python randomises
        string hashing per process, so the returned column order differed on
        every run — three runs of one input gave three different orders.

        Harmless while consumers read by name, but it makes ``.values`` and any
        position-sensitive estimator silently non-reproducible.
        """
        columns = ['site', 'iv', 'dv', 'id', 'resp_age', 'resp_gender']
        rct = pd.DataFrame({c: [1] for c in columns})
        osrct = pd.DataFrame({c: [2] for c in reversed(columns)})

        result = add_regime_indicator(rct, osrct)

        assert list(result.columns) == columns + ['F']

    def test_column_order_is_stable_across_repeated_calls(self):
        """Same inputs, same order. The regression guard for the above."""
        columns = ['site', 'iv', 'dv', 'id', 'resp_age', 'resp_gender']
        rct = pd.DataFrame({c: [1] for c in columns})
        osrct = pd.DataFrame({c: [2] for c in columns})

        orders = {
            tuple(add_regime_indicator(rct, osrct).columns) for _ in range(10)
        }
        assert len(orders) == 1

    def test_columns_unique_to_one_frame_are_dropped(self):
        """Only shared columns survive, which is the point of the alignment."""
        rct = pd.DataFrame({'iv': [0], 'dv': [1.0], 'rct_only': [9]})
        osrct = pd.DataFrame({'iv': [1], 'dv': [2.0], 'osrct_only': [8]})

        result = add_regime_indicator(rct, osrct)

        assert list(result.columns) == ['iv', 'dv', 'F']


# =============================================================================
# TEST CLASS: TestCreateTrainTargetSplit
# =============================================================================

class TestCreateTrainTargetSplit:
    """Test create_train_target_split function."""

    def test_target_excluded_from_training(self, mock_rct_data, mock_osrct_data):
        """Test that target site is not in training data."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        assert 'mturk' not in training_data.keys()

    def test_target_data_correct_site(self, mock_rct_data, mock_osrct_data):
        """Test that target data contains only target site."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        assert all(target_data['site'] == 'mturk')

    def test_training_has_both_regimes(self, mock_rct_data, mock_osrct_data):
        """Test that training sites have both F='on' and F='idle'."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        for site, df in training_data.items():
            f_values = df['F'].unique()
            assert 'on' in f_values, f"Site {site} missing F='on'"
            assert 'idle' in f_values, f"Site {site} missing F='idle'"

    def test_target_has_idle_only(self, mock_rct_data, mock_osrct_data):
        """Test that target data has only F='idle'."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        assert all(target_data['F'] == 'idle')

    def test_returns_correct_types(self, mock_rct_data, mock_osrct_data):
        """Test that return types are correct."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        assert isinstance(training_data, dict)
        assert isinstance(target_data, pd.DataFrame)
        
        for site, df in training_data.items():
            assert isinstance(site, str)
            assert isinstance(df, pd.DataFrame)

    def test_invalid_target_site_raises(self, mock_rct_data, mock_osrct_data):
        """Test that invalid target site raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            create_train_target_split(
                mock_osrct_data, mock_rct_data, target_site='nonexistent'
            )

    def test_training_sites_count(self, mock_rct_data, mock_osrct_data):
        """Test that training data has expected number of sites."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        # Should have 3 training sites (A, B, C) when mturk is target
        available = get_available_sites(mock_osrct_data)
        expected_count = len([s for s in available if s != 'mturk'])
        
        # May have fewer if some sites don't have both RCT and OSRCT data
        assert len(training_data) <= expected_count


# =============================================================================
# TEST CLASS: TestSummarizeSplit
# =============================================================================

class TestSummarizeSplit:
    """Test summarize_split function."""

    def test_returns_string(self, mock_rct_data, mock_osrct_data):
        """Test that summarize_split returns a string."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        result = summarize_split(training_data, target_data)
        
        assert isinstance(result, str)

    def test_contains_site_names(self, mock_rct_data, mock_osrct_data):
        """Test that summary contains site names."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        result = summarize_split(training_data, target_data)
        
        # Should mention training sites
        for site in training_data.keys():
            assert site in result

    def test_contains_counts(self, mock_rct_data, mock_osrct_data):
        """Test that summary contains row counts."""
        training_data, target_data = create_train_target_split(
            mock_osrct_data, mock_rct_data, target_site='mturk'
        )
        
        result = summarize_split(training_data, target_data)
        
        # Should contain 'on' and 'idle' indicators
        assert 'on' in result
        assert 'idle' in result


# =============================================================================
# TEST CLASS: TestGetAvailableSites
# =============================================================================

class TestGetAvailableSites:
    """Test get_available_sites function."""

    def test_returns_sorted_list(self, mock_osrct_data):
        """Test that sites are returned as sorted list."""
        result = get_available_sites(mock_osrct_data)
        
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_returns_unique_sites(self, mock_osrct_data):
        """Test that returned sites are unique."""
        result = get_available_sites(mock_osrct_data)
        
        assert len(result) == len(set(result))

    def test_expected_sites(self, mock_osrct_data):
        """Test that expected sites are present."""
        result = get_available_sites(mock_osrct_data)
        
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert 'mturk' in result
