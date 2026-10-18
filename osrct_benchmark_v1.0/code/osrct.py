"""
OSRCT (Observational Sampling from Randomized Controlled Trials) Implementation

This module implements the OSRCT procedure from Gentzel et al. (2021) to generate
semi-synthetic confounded observational datasets from RCT data while preserving
known ground-truth treatment effects.

Reference:
Gentzel, M., Garant, D., & Steeg, D. (2021). The Case for Evaluating Causal
Models Using Controlled Experiments. NeurIPS 2021.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from scipy.special import expit  # logistic function
from sklearn.preprocessing import StandardScaler
import warnings


class OSRCTSampler:
    """
    Observational Sampling from RCTs (OSRCT) sampler.

    Generates confounded observational datasets from RCT data by:
    1. Computing selection probabilities based on biasing covariates
    2. Sampling preferred treatment for each unit
    3. Keeping only units where actual treatment matches preferred treatment

    This creates realistic observational data with known ground-truth effects.
    """

    def __init__(
        self,
        biasing_covariates: List[str],
        biasing_coefficients: Optional[Dict[str, float]] = None,
        intercept: float = 0.0,
        standardize: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize OSRCT sampler.

        Parameters
        ----------
        biasing_covariates : list of str
            Names of pre-treatment covariates to use for biasing
        biasing_coefficients : dict, optional
            Dictionary mapping covariate names to coefficients.
            If None, random coefficients will be generated.
        intercept : float, default=0.0
            Intercept term for biasing function
        standardize : bool, default=True
            Whether to standardize covariates before applying biasing function
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.biasing_covariates = biasing_covariates
        self.biasing_coefficients = biasing_coefficients
        self.intercept = intercept
        self.standardize = standardize
        self.random_seed = random_seed

        # Use isolated random generator (does not affect global state)
        self.rng = np.random.default_rng(random_seed)

        self.scaler = StandardScaler() if standardize else None
        self.is_fitted = False

    def _compute_selection_probability(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute selection probability P(T=1|C) using logistic function.

        P(T=1|C) = 1 / (1 + exp(-β₀ - Σ(βⱼ*Cⱼ)))

        Parameters
        ----------
        X : DataFrame
            Covariate data

        Returns
        -------
        probabilities : ndarray
            Selection probabilities for each unit
        """
        # Extract biasing covariates
        X_bias = X[self.biasing_covariates].values

        # Handle missing values by imputing with mean
        if np.any(np.isnan(X_bias)):
            warnings.warn("Missing values detected in covariates. Imputing with mean.")
            col_means = np.nanmean(X_bias, axis=0)
            inds = np.where(np.isnan(X_bias))
            X_bias[inds] = np.take(col_means, inds[1])

        # Standardize if requested
        if self.standardize:
            if not self.is_fitted:
                X_bias = self.scaler.fit_transform(X_bias)
                self.is_fitted = True
            else:
                X_bias = self.scaler.transform(X_bias)

        # Compute linear predictor: β₀ + Σ(βⱼ*Cⱼ)
        if self.biasing_coefficients is None:
            # Generate random coefficients if not provided
            self.biasing_coefficients = {
                cov: self.rng.standard_normal()
                for cov in self.biasing_covariates
            }

        # Extract coefficients in correct order
        beta = np.array([
            self.biasing_coefficients[cov]
            for cov in self.biasing_covariates
        ])

        # Linear predictor
        linear_pred = self.intercept + X_bias @ beta

        # Apply logistic function
        probabilities = expit(linear_pred)

        return probabilities

    def selection_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Return preferred-treatment probabilities for the supplied rows.

        The first call fits the same study-level standardization used by
        :meth:`sample`; later calls reuse that fitted transform.

        Parameters
        ----------
        X : DataFrame
            Rows containing every configured biasing covariate.

        Returns
        -------
        ndarray
            Probability that OSRCT assigns preferred treatment one.
        """

        return self._compute_selection_probability(X)

    def sample(
        self,
        rct_data: pd.DataFrame,
        treatment_col: str = 'iv',
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate confounded observational sample from RCT data.

        Algorithm (Gentzel et al. 2021, Algorithm 2):
        1. For each unit i in the RCT:
           - Compute p_i = f(C^b_i) using biasing function
           - Sample preferred treatment t_s ~ Bernoulli(p_i)
           - If unit i actually received t_s in the RCT, include in output
           - Otherwise, discard the unit

        Parameters
        ----------
        rct_data : DataFrame
            Original RCT dataset with treatment, outcome, and covariates
        treatment_col : str, default='iv'
            Name of treatment column (should be binary 0/1)
        verbose : bool, default=True
            Whether to print sampling statistics

        Returns
        -------
        observational_sample : DataFrame
            Biased observational dataset
        selection_probs : ndarray
            Selection probability for each unit in original RCT
        """
        # Validate treatment column
        if treatment_col not in rct_data.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in data")

        treatment = rct_data[treatment_col].values
        if not np.all(np.isin(treatment, [0, 1])):
            raise ValueError("Treatment must be binary (0/1)")

        # Check that biasing covariates exist in data
        missing_covs = set(self.biasing_covariates) - set(rct_data.columns)
        if missing_covs:
            raise ValueError(f"Biasing covariates not found in data: {missing_covs}")

        # Compute selection probabilities
        selection_probs = self._compute_selection_probability(rct_data)

        # Sample preferred treatment for each unit
        # t_s ~ Bernoulli(p_i)
        preferred_treatment = self.rng.binomial(1, selection_probs)

        # Keep only units where actual treatment matches preferred treatment
        keep_mask = (treatment == preferred_treatment)
        observational_sample = rct_data[keep_mask].copy()

        # Add selection probability to output for analysis
        observational_sample['_selection_prob'] = selection_probs[keep_mask]
        observational_sample['_preferred_treatment'] = preferred_treatment[keep_mask]

        if verbose:
            n_original = len(rct_data)
            n_sampled = len(observational_sample)
            pct_sampled = 100 * n_sampled / n_original

            print(f"OSRCT Sampling Results:")
            print(f"  Original RCT size: {n_original:,}")
            print(f"  Sampled size: {n_sampled:,} ({pct_sampled:.1f}%)")
            print(f"  Selection prob range: [{selection_probs.min():.3f}, {selection_probs.max():.3f}]")
            print(f"  Selection prob mean: {selection_probs.mean():.3f}")

            # Check treatment balance
            if treatment_col in observational_sample.columns:
                treated = observational_sample[treatment_col].sum()
                control = len(observational_sample) - treated
                print(f"  Treatment group: {treated:,} ({100*treated/n_sampled:.1f}%)")
                print(f"  Control group: {control:,} ({100*control/n_sampled:.1f}%)")

        return observational_sample, selection_probs

    def get_confounding_strength(
        self,
        rct_data: pd.DataFrame,
        outcome_col: str = 'dv'
    ) -> Dict[str, float]:
        """
        Measure confounding strength by computing correlations between
        biasing covariates and outcome.

        Parameters
        ----------
        rct_data : DataFrame
            RCT dataset
        outcome_col : str, default='dv'
            Name of outcome column

        Returns
        -------
        correlations : dict
            Correlation between each biasing covariate and outcome
        """
        if outcome_col not in rct_data.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")

        correlations = {}
        outcome = rct_data[outcome_col].values

        for cov in self.biasing_covariates:
            if cov in rct_data.columns:
                cov_values = rct_data[cov].values
                # Compute correlation, handling NaNs
                valid_mask = ~(np.isnan(outcome) | np.isnan(cov_values))
                if np.sum(valid_mask) > 0:
                    corr = np.corrcoef(
                        outcome[valid_mask],
                        cov_values[valid_mask]
                    )[0, 1]
                    correlations[cov] = corr
                else:
                    correlations[cov] = np.nan

        return correlations


def select_biasing_covariates(
    rct_data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    candidate_covariates: Optional[List[str]] = None,
    min_correlation: float = 0.1,
    max_covariates: Optional[int] = None
) -> List[str]:
    """
    Select biasing covariates based on correlation with outcome.

    Good biasing covariates should be:
    1. Pre-treatment (measured before intervention)
    2. Correlated with outcome (to induce confounding)

    Parameters
    ----------
    rct_data : DataFrame
        RCT dataset
    treatment_col : str, default='iv'
        Name of treatment column
    outcome_col : str, default='dv'
        Name of outcome column
    candidate_covariates : list of str, optional
        List of candidate covariates. If None, use all numeric columns
        except treatment and outcome.
    min_correlation : float, default=0.1
        Minimum absolute correlation with outcome to be selected
    max_covariates : int, optional
        Maximum number of covariates to select

    Returns
    -------
    selected_covariates : list of str
        Selected biasing covariates, ordered by correlation strength
    """
    if outcome_col not in rct_data.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in data")

    # Get candidate covariates
    if candidate_covariates is None:
        # Use all numeric columns except treatment and outcome
        exclude_cols = [treatment_col, outcome_col]
        candidate_covariates = [
            col for col in rct_data.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols and not col.startswith('_')
        ]

    # Compute correlations
    outcome = rct_data[outcome_col].values
    correlations = {}

    for cov in candidate_covariates:
        if cov in rct_data.columns:
            cov_values = rct_data[cov].values
            valid_mask = ~(np.isnan(outcome) | np.isnan(cov_values))

            if np.sum(valid_mask) > 10:  # Require at least 10 valid observations
                corr = np.corrcoef(
                    outcome[valid_mask],
                    cov_values[valid_mask]
                )[0, 1]
                correlations[cov] = abs(corr)

    # Filter by minimum correlation
    selected = [
        cov for cov, corr in correlations.items()
        if corr >= min_correlation
    ]

    # Sort by correlation strength
    selected = sorted(selected, key=lambda x: correlations[x], reverse=True)

    # Limit number of covariates if requested
    if max_covariates is not None:
        selected = selected[:max_covariates]

    return selected


def evaluate_osrct_sample(
    rct_data: pd.DataFrame,
    obs_data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate OSRCT sample by comparing to original RCT.

    Computes:
    - True treatment effect (from RCT)
    - Naive observational effect (from biased sample)
    - Covariate balance (standardized mean differences)

    Parameters
    ----------
    rct_data : DataFrame
        Original RCT dataset
    obs_data : DataFrame
        OSRCT observational sample
    treatment_col : str, default='iv'
        Name of treatment column
    outcome_col : str, default='dv'
        Name of outcome column
    covariates : list of str, optional
        Covariates to check balance for

    Returns
    -------
    metrics : dict
        Evaluation metrics
    """
    metrics = {}

    # True treatment effect from RCT
    rct_treated = rct_data[rct_data[treatment_col] == 1][outcome_col]
    rct_control = rct_data[rct_data[treatment_col] == 0][outcome_col]
    metrics['rct_ate'] = rct_treated.mean() - rct_control.mean()
    metrics['rct_ate_se'] = np.sqrt(
        rct_treated.var() / len(rct_treated) +
        rct_control.var() / len(rct_control)
    )

    # Naive observational effect (biased)
    obs_treated = obs_data[obs_data[treatment_col] == 1][outcome_col]
    obs_control = obs_data[obs_data[treatment_col] == 0][outcome_col]
    metrics['obs_ate_naive'] = obs_treated.mean() - obs_control.mean()
    metrics['obs_ate_se'] = np.sqrt(
        obs_treated.var() / len(obs_treated) +
        obs_control.var() / len(obs_control)
    )

    # Confounding bias
    metrics['confounding_bias'] = metrics['obs_ate_naive'] - metrics['rct_ate']

    # Sample size reduction
    metrics['sample_size_rct'] = len(rct_data)
    metrics['sample_size_obs'] = len(obs_data)
    metrics['sample_retention_rate'] = len(obs_data) / len(rct_data)

    # Treatment balance
    metrics['rct_treatment_rate'] = rct_data[treatment_col].mean()
    metrics['obs_treatment_rate'] = obs_data[treatment_col].mean()

    # Covariate balance (standardized mean differences)
    if covariates is not None:
        smd_values = {}
        for cov in covariates:
            if cov in rct_data.columns and cov in obs_data.columns:
                # RCT balance
                rct_t1 = rct_data[rct_data[treatment_col] == 1][cov]
                rct_t0 = rct_data[rct_data[treatment_col] == 0][cov]
                rct_smd = (rct_t1.mean() - rct_t0.mean()) / np.sqrt(
                    (rct_t1.var() + rct_t0.var()) / 2
                )

                # Observational balance
                obs_t1 = obs_data[obs_data[treatment_col] == 1][cov]
                obs_t0 = obs_data[obs_data[treatment_col] == 0][cov]
                obs_smd = (obs_t1.mean() - obs_t0.mean()) / np.sqrt(
                    (obs_t1.var() + obs_t0.var()) / 2
                )

                smd_values[cov] = {
                    'rct_smd': rct_smd,
                    'obs_smd': obs_smd,
                    'smd_change': obs_smd - rct_smd
                }

        metrics['covariate_balance'] = smd_values

    return metrics


def load_manylabs1_data(
    data_path: str,
    study_filter: Optional[Union[str, List[str]]] = None,
    site_filter: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Load ManyLabs1 data from pickle, CSV, or RData file.

    Parameters
    ----------
    data_path : str
        Path to ManyLabs1 data file. Supports:
        - .pkl (pickle format from Python preprocessing)
        - .csv (CSV format from Python preprocessing)
        - .RData (R format, requires pyreadr)
    study_filter : str or list of str, optional
        Filter to specific studies
    site_filter : str or list of str, optional
        Filter to specific sites

    Returns
    -------
    data : DataFrame
        Loaded and filtered ManyLabs1 data
    """
    # Determine file format and load accordingly
    if data_path.endswith('.pkl'):
        data = pd.read_pickle(data_path)
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.RData') or data_path.endswith('.rdata'):
        try:
            import pyreadr
        except ImportError:
            raise ImportError(
                "pyreadr is required to load RData files. "
                "Install with: pip install pyreadr"
            )
        # Load RData file
        result = pyreadr.read_r(data_path)
        # Extract pdata
        if 'pdata' not in result:
            raise ValueError("'pdata' not found in RData file")
        data = result['pdata']
    else:
        raise ValueError(
            f"Unsupported file format: {data_path}. "
            "Supported formats: .pkl, .csv, .RData"
        )

    # Apply filters
    if study_filter is not None:
        if isinstance(study_filter, str):
            study_filter = [study_filter]
        data = data[data['original_study'].isin(study_filter)]

    if site_filter is not None:
        if isinstance(site_filter, str):
            site_filter = [site_filter]
        data = data[data['site'].isin(site_filter)]

    return data


def load_pipeline_data(
    data_dir: str,
    study_id: Optional[Union[int, List[int]]] = None
) -> pd.DataFrame:
    """
    Load Pipeline data from processed CSV files.

    Parameters
    ----------
    data_dir : str
        Directory containing processed Pipeline CSV files
    study_id : int or list of int, optional
        Filter to specific study IDs (e.g., 5, 7, 8)

    Returns
    -------
    data : DataFrame
        Loaded and filtered Pipeline data
    """
    import os

    # Map study IDs to file names
    study_files = {
        5: '5_moral_inversion.csv',
        7: '7_intuitive_economics.csv',
        8: '8_burn_in_hell.csv',
        # Add more as needed
    }

    if study_id is None:
        study_id = list(study_files.keys())
    elif isinstance(study_id, int):
        study_id = [study_id]

    # Load and concatenate data
    data_list = []
    for sid in study_id:
        if sid not in study_files:
            warnings.warn(f"Study ID {sid} not recognized. Skipping.")
            continue

        file_path = os.path.join(data_dir, 'clean_for_analysis', study_files[sid])
        if not os.path.exists(file_path):
            warnings.warn(f"File not found: {file_path}. Skipping.")
            continue

        df = pd.read_csv(file_path, index_col=0)
        df['study_id'] = sid
        data_list.append(df)

    if not data_list:
        raise ValueError("No data loaded. Check data_dir and study_id.")

    data = pd.concat(data_list, ignore_index=True)

    return data


if __name__ == "__main__":
    # Example usage
    print("OSRCT Module - Observational Sampling from RCTs")
    print("=" * 60)
    print("\nThis module implements the OSRCT procedure for generating")
    print("confounded observational datasets from RCT data.")
    print("\nKey classes:")
    print("  - OSRCTSampler: Main sampling class")
    print("\nKey functions:")
    print("  - select_biasing_covariates: Select covariates for confounding")
    print("  - evaluate_osrct_sample: Evaluate generated samples")
    print("  - load_manylabs1_data: Load ManyLabs1 data")
    print("  - load_pipeline_data: Load Pipeline data")
