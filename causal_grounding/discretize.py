"""
Covariate Discretization for LP Solver and CI Tests

This module discretizes continuous covariates for the LP solver and CI tests.
The LP solver requires discrete covariate spaces to enumerate all possible
(Z_a, Z_b) combinations.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def discretize_age(
    series: pd.Series,
    n_bins: int = 5,
    labels: Optional[List] = None
) -> pd.Series:
    """
    Discretize age into quantile-based bins.

    Args:
        series: Continuous age values
        n_bins: Number of bins (default 5)
        labels: Optional bin labels (default: 0, 1, ..., n_bins-1)

    Returns:
        Categorical series with bin assignments
    """
    if labels is None:
        labels = list(range(n_bins))

    try:
        return pd.qcut(series, n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # Fallback if too few unique values
        return pd.cut(series, n_bins, labels=labels[:n_bins], duplicates='drop')


def discretize_polideo(series: pd.Series) -> pd.Series:
    """
    Discretize political ideology (0-6 scale) into 3 categories.

    Mapping:
        0-2: 0 (conservative)
        3:   1 (moderate)
        4-6: 2 (liberal)

    Returns:
        Categorical series with values 0, 1, 2
    """
    bins = [-0.1, 2.5, 3.5, 6.1]
    labels = [0, 1, 2]
    return pd.cut(series, bins=bins, labels=labels).astype(int)


def discretize_covariates(
    data: pd.DataFrame,
    age_col: str = 'resp_age',
    gender_col: str = 'resp_gender',
    polideo_col: str = 'resp_polideo',
    age_bins: int = 5,
    copy: bool = True
) -> pd.DataFrame:
    """
    Discretize all continuous covariates for LP solver.

    Creates new columns:
        - {age_col}_cat: discretized age
        - {polideo_col}_cat: discretized political ideology
        - {gender_col} remains unchanged (already binary)

    Args:
        data: DataFrame with covariate columns
        age_col: Name of age column
        gender_col: Name of gender column
        polideo_col: Name of political ideology column
        age_bins: Number of age bins
        copy: Whether to copy DataFrame

    Returns:
        DataFrame with additional discretized columns
    """
    if copy:
        data = data.copy()

    if age_col in data.columns:
        data[f'{age_col}_cat'] = discretize_age(data[age_col], n_bins=age_bins)

    if polideo_col in data.columns:
        data[f'{polideo_col}_cat'] = discretize_polideo(data[polideo_col])

    return data


def get_discretized_covariate_names(
    age_col: str = 'resp_age',
    gender_col: str = 'resp_gender',
    polideo_col: str = 'resp_polideo'
) -> List[str]:
    """Return list of discretized covariate column names."""
    return [f'{age_col}_cat', gender_col, f'{polideo_col}_cat']


# Test
if __name__ == "__main__":
    # Load sample data
    from pathlib import Path

    # Try CSV first (the actual format)
    data_path = Path('confounded_datasets/anchoring1/age_beta0.5_seed42.csv')
    if data_path.exists():
        data = pd.read_csv(data_path)
        result = discretize_covariates(data)
        print("Original columns:", list(data.columns)[:10], "...")
        print("New columns:", [c for c in result.columns if c not in data.columns])
        print("\nAge discretization:")
        print(result[['resp_age', 'resp_age_cat']].head(10))
        print("\nPolideo discretization:")
        print(result[['resp_polideo', 'resp_polideo_cat']].head(10))
        print("\nDiscretized covariate names:")
        print(get_discretized_covariate_names())
    else:
        print(f"Data file not found: {data_path}")
