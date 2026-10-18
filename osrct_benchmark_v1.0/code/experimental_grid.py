"""
Experimental Grid Configuration for OSRCT Benchmark Dataset Generation

This module defines the complete experimental design for generating confounded
observational datasets from ManyLabs1 RCT data. It specifies all combinations
of studies, confounding strengths (beta values), and covariate patterns.

Reference:
Gentzel, M., Garant, D., & Jensen, D. (2021). The Case for Evaluating Causal
Models Using Controlled Experiments. NeurIPS 2021.
"""

from typing import Dict, List, Callable, Any

# =============================================================================
# EXPERIMENTAL GRID DEFINITION
# =============================================================================

# All 15 ManyLabs1 studies (as available in preprocessed data)
STUDIES = [
    'anchoring1',    # Anchoring effect (NYC population question)
    'anchoring2',    # Anchoring effect (Mt. Everest height)
    'anchoring3',    # Anchoring effect (US babies born question)
    'anchoring4',    # Anchoring effect (Chicago distance question)
    'gainloss',      # Gain/loss framing (Asian disease problem)
    'sunk',          # Sunk costs effect
    'allowedforbidden',  # Allowed/forbidden asymmetry
    'reciprocity',   # Reciprocity in generosity
    'flag',          # Flag priming effect
    'quote',         # Quote attribution effect
    'gambfal',       # Gambler's fallacy
    'scales',        # Scale anchoring
    'money',         # Money priming
    'contact',       # Contact hypothesis
    'iat',           # Implicit association test (math attitudes)
]

# Confounding strength values (beta coefficients)
# Higher beta = stronger confounding = more bias in naive estimates
BETA_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Description of beta values for documentation
BETA_DESCRIPTIONS = {
    0.1: 'Very weak confounding - minimal bias',
    0.25: 'Weak confounding - small bias',
    0.5: 'Moderate confounding - realistic observational scenario',
    0.75: 'Moderate-strong confounding - challenging scenario',
    1.0: 'Strong confounding - significant bias',
    1.5: 'Very strong confounding - severe bias',
    2.0: 'Extreme confounding - method stress test'
}

# =============================================================================
# COVARIATE PATTERNS
# =============================================================================

# Single-covariate patterns
SINGLE_COVARIATE_PATTERNS = {
    'age': {
        'covariates': ['resp_age'],
        'coefficients': lambda beta: {'resp_age': beta},
        'description': 'Confounding by age only (continuous)'
    },
    'gender': {
        'covariates': ['resp_gender'],
        'coefficients': lambda beta: {'resp_gender': beta},
        'description': 'Confounding by gender only (binary)'
    },
    'polideo': {
        'covariates': ['resp_polideo'],
        'coefficients': lambda beta: {'resp_polideo': beta},
        'description': 'Confounding by political ideology (ordinal)'
    }
}

# Multi-covariate patterns
MULTI_COVARIATE_PATTERNS = {
    'demo_basic': {
        'covariates': ['resp_age', 'resp_gender'],
        'coefficients': lambda beta: {
            'resp_age': beta,
            'resp_gender': beta
        },
        'description': 'Basic demographics (age + gender, equal weights)'
    },
    'demo_full': {
        'covariates': ['resp_age', 'resp_gender', 'resp_polideo'],
        'coefficients': lambda beta: {
            'resp_age': beta,
            'resp_gender': 0.8 * beta,
            'resp_polideo': 0.6 * beta
        },
        'description': 'Full demographics (age dominant, polideo weakest)'
    },
    'political': {
        'covariates': ['resp_polideo', 'resp_pid'],
        'coefficients': lambda beta: {
            'resp_polideo': beta,
            'resp_pid': beta
        },
        'description': 'Political variables (ideology + party ID)'
    }
}

# Combined patterns dictionary
COVARIATE_PATTERNS = {
    **SINGLE_COVARIATE_PATTERNS,
    **MULTI_COVARIATE_PATTERNS
}

# =============================================================================
# SITE-STRATIFIED CONFIGURATION
# =============================================================================

# Studies to use for site-stratified analysis (higher sample sizes)
SITE_STRATIFIED_STUDIES = [
    'anchoring1',
    'gainloss',
    'sunk',
    'flag',
    'iat'
]

# Beta values for site-stratified (subset due to computational cost)
SITE_STRATIFIED_BETA_VALUES = [0.5, 1.0]

# Pattern for site-stratified (consistent across sites)
SITE_STRATIFIED_PATTERN = 'demo_basic'

# Minimum sample size per site to include
MIN_SITE_SAMPLE_SIZE = 50

# =============================================================================
# FULL EXPERIMENTAL GRID
# =============================================================================

EXPERIMENTAL_GRID = {
    'studies': STUDIES,
    'beta_values': BETA_VALUES,
    'beta_descriptions': BETA_DESCRIPTIONS,
    'covariate_patterns': COVARIATE_PATTERNS,

    # Site-stratified subset
    'site_stratified': {
        'studies': SITE_STRATIFIED_STUDIES,
        'beta_values': SITE_STRATIFIED_BETA_VALUES,
        'pattern': SITE_STRATIFIED_PATTERN,
        'min_site_n': MIN_SITE_SAMPLE_SIZE
    },

    # Default random seeds for reproducibility
    'random_seeds': [42],

    # Multiple seeds for variance estimation
    'variance_seeds': [42, 123, 456, 789, 1011]
}


def get_total_configurations(
    studies: List[str] = None,
    beta_values: List[float] = None,
    patterns: List[str] = None,
    seeds: List[int] = None
) -> int:
    """
    Calculate total number of configurations in experimental grid.

    Parameters
    ----------
    studies : List[str], optional
        List of studies (default: all 15)
    beta_values : List[float], optional
        List of beta values (default: 7 values)
    patterns : List[str], optional
        List of patterns (default: 6 patterns)
    seeds : List[int], optional
        List of seeds (default: 1 seed)

    Returns
    -------
    total : int
        Total number of configurations
    """
    if studies is None:
        studies = STUDIES
    if beta_values is None:
        beta_values = BETA_VALUES
    if patterns is None:
        patterns = list(COVARIATE_PATTERNS.keys())
    if seeds is None:
        seeds = [42]

    return len(studies) * len(beta_values) * len(patterns) * len(seeds)


def get_grid_summary() -> Dict[str, Any]:
    """
    Get summary of experimental grid for documentation.

    Returns
    -------
    summary : dict
        Summary statistics and configuration details
    """
    n_pooled = get_total_configurations()

    # Estimate site-stratified (assuming ~30 usable sites per study)
    n_site_stratified = (
        len(SITE_STRATIFIED_STUDIES) *
        len(SITE_STRATIFIED_BETA_VALUES) *
        30  # approximate usable sites
    )

    summary = {
        'n_studies': len(STUDIES),
        'n_beta_values': len(BETA_VALUES),
        'n_patterns': len(COVARIATE_PATTERNS),
        'n_single_covariate_patterns': len(SINGLE_COVARIATE_PATTERNS),
        'n_multi_covariate_patterns': len(MULTI_COVARIATE_PATTERNS),
        'n_pooled_datasets': n_pooled,
        'n_site_stratified_datasets_estimate': n_site_stratified,
        'n_total_estimate': n_pooled + n_site_stratified,
        'studies': STUDIES,
        'beta_values': BETA_VALUES,
        'pattern_names': list(COVARIATE_PATTERNS.keys())
    }

    return summary


def validate_grid_config(
    data_columns: List[str]
) -> Dict[str, bool]:
    """
    Validate that all covariates in grid exist in data.

    Parameters
    ----------
    data_columns : List[str]
        Column names from the data

    Returns
    -------
    validation : dict
        Validation results for each pattern
    """
    validation = {}

    for pattern_name, pattern_config in COVARIATE_PATTERNS.items():
        covariates = pattern_config['covariates']
        missing = set(covariates) - set(data_columns)

        validation[pattern_name] = {
            'valid': len(missing) == 0,
            'covariates': covariates,
            'missing': list(missing)
        }

    return validation


if __name__ == "__main__":
    # Print grid summary
    print("OSRCT Experimental Grid Configuration")
    print("=" * 60)

    summary = get_grid_summary()

    print(f"\nStudies: {summary['n_studies']}")
    for study in STUDIES:
        print(f"  - {study}")

    print(f"\nBeta Values: {summary['n_beta_values']}")
    for beta, desc in BETA_DESCRIPTIONS.items():
        print(f"  - {beta}: {desc}")

    print(f"\nCovariate Patterns: {summary['n_patterns']}")
    print(f"  Single-covariate: {summary['n_single_covariate_patterns']}")
    for name, config in SINGLE_COVARIATE_PATTERNS.items():
        print(f"    - {name}: {config['description']}")
    print(f"  Multi-covariate: {summary['n_multi_covariate_patterns']}")
    for name, config in MULTI_COVARIATE_PATTERNS.items():
        print(f"    - {name}: {config['description']}")

    print(f"\nTotal Configurations:")
    print(f"  Pooled datasets: {summary['n_pooled_datasets']}")
    print(f"  Site-stratified (estimate): {summary['n_site_stratified_datasets_estimate']}")
    print(f"  Total (estimate): {summary['n_total_estimate']}")
