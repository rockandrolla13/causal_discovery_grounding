"""
Regression Tests for ManyLabs1 Preprocessing

This module validates that the Python preprocessing produces output that
matches the expected structure and statistics from the R preprocessing.

Run with: pytest test_ml1_preprocess.py -v
Or standalone: python test_ml1_preprocess.py
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ml1_preprocess import (
    preprocess_manylabs1,
    categorize_race,
    categorize_major,
    STUDIES,
    POLIDEO_NUMERIC
)


# ============================================================================
# Test Configuration
# ============================================================================

# Path to test data (update this to your local path)
TEST_DATA_PATH = Path('osrct_benchmark_v1.0/data/CleanedDataset.sav')


# Expected values based on ManyLabs1 documentation and R output
EXPECTED = {
    # Total observations in raw data
    'raw_n_rows': 6344,

    # Number of studies
    'n_studies': 15,

    # Study names
    'studies': [
        'allowedforbidden', 'anchoring1', 'anchoring2', 'anchoring3', 'anchoring4',
        'contact', 'flag', 'gainloss', 'gambfal', 'iat', 'money', 'quote',
        'reciprocity', 'scales', 'sunk'
    ],

    # Number of unique sites
    'n_sites': 36,

    # Sample sites
    'sample_sites': ['abington', 'brasilia', 'mturk', 'tilburg', 'ufl'],

    # Treatment should be binary (0, 1)
    'iv_values': [0, 1],

    # Key covariates that should exist
    'required_covariates': [
        'resp_gender', 'resp_age', 'resp_ethnicity_hisp', 'resp_polideo',
        'resp_american', 'resp_american_pid', 'resp_american_ideo'
    ],

    # Race dummy variables
    'race_dummies': [
        'RACE_white', 'RACE_black_american', 'RACE_east_asian',
        'RACE_south_asian', 'RACE_others'
    ],

    # Major dummy variables
    'major_dummies': [
        'MAJOR_Social', 'MAJOR_Engineer', 'MAJOR_Science', 'MAJOR_others'
    ],

    # Political ideology range (0-6)
    'polideo_range': (0, 6),

    # Minimum observations per study (should be > 100 after filtering)
    'min_obs_per_study': 100,

    # Total observations should be roughly 15 * 6344 - missing values
    # (each subject appears once per study in long format)
    'total_obs_range': (70000, 100000)  # Approximate range
}


# ============================================================================
# Unit Tests
# ============================================================================

class TestCategorization:
    """Test categorization functions."""

    def test_categorize_race_white(self):
        assert categorize_race('white') == 'white'

    def test_categorize_race_black(self):
        assert categorize_race('black or african american') == 'black_american'

    def test_categorize_race_east_asian(self):
        assert categorize_race('east asian') == 'east_asian'
        assert categorize_race('chinese') == 'east_asian'

    def test_categorize_race_south_asian(self):
        assert categorize_race('south asian') == 'south_asian'
        assert categorize_race('indian') == 'south_asian'

    def test_categorize_race_brazil(self):
        assert categorize_race('brazilwhite') == 'brazil'
        assert categorize_race('brazilblack') == 'brazil'

    def test_categorize_race_others(self):
        assert categorize_race('unknown') == 'others'
        assert categorize_race(np.nan) == 'others'

    def test_categorize_major_social(self):
        assert categorize_major('Psychology') == 'Social'
        assert categorize_major('Education') == 'Social'

    def test_categorize_major_engineer(self):
        assert categorize_major('Computer and information sciences') == 'Engineer'

    def test_categorize_major_science(self):
        assert categorize_major('Biological sciences/life sciences') == 'Science'

    def test_categorize_major_others(self):
        assert categorize_major('Business') == 'others'
        assert categorize_major(np.nan) == 'others'


class TestPoliticalIdeoMapping:
    """Test political ideology mapping."""

    def test_polideo_conservative(self):
        assert POLIDEO_NUMERIC['Strongly Conservative'] == 0
        assert POLIDEO_NUMERIC['Moderately Conservative'] == 1

    def test_polideo_liberal(self):
        assert POLIDEO_NUMERIC['Strongly Liberal'] == 6
        assert POLIDEO_NUMERIC['Moderately Liberal'] == 5

    def test_polideo_neutral(self):
        assert POLIDEO_NUMERIC['Neutral (Moderate)'] == 3


class TestStudyDefinitions:
    """Test study definitions."""

    def test_all_studies_defined(self):
        assert len(STUDIES) == EXPECTED['n_studies']

    def test_studies_have_required_fields(self):
        for study_name, study_info in STUDIES.items():
            assert 'name' in study_info, f"{study_name} missing 'name'"
            assert 'iv_col' in study_info, f"{study_name} missing 'iv_col'"
            assert 'dv_col' in study_info, f"{study_name} missing 'dv_col'"

    def test_reverse_studies(self):
        """Studies that should have reversed IV."""
        reverse_studies = ['sunk', 'gainloss', 'iat', 'reciprocity', 'allowedforbidden']
        for study in reverse_studies:
            assert STUDIES[study].get('reverse_iv', False), \
                f"{study} should have reverse_iv=True"


# ============================================================================
# Integration Tests (require data file)
# ============================================================================

@pytest.mark.skipif(not TEST_DATA_PATH.exists(), reason="Data file not found")
class TestPreprocessing:
    """Integration tests for full preprocessing pipeline."""

    @pytest.fixture(scope='class')
    def processed_data(self, tmp_path_factory):
        """Run preprocessing once for all tests in this class."""
        output_dir = tmp_path_factory.mktemp('output')
        pdata, covariates_list = preprocess_manylabs1(
            input_path=str(TEST_DATA_PATH),
            output_dir=str(output_dir),
            verbose=False
        )
        return pdata, covariates_list, output_dir

    def test_output_files_created(self, processed_data):
        """Check that output files are created."""
        _, _, output_dir = processed_data
        assert (output_dir / 'Manylabs1_data.pkl').exists()
        assert (output_dir / 'Manylabs1_data.csv').exists()
        assert (output_dir / 'covariates_list.txt').exists()

    def test_total_observations(self, processed_data):
        """Check total number of observations is reasonable."""
        pdata, _, _ = processed_data
        min_obs, max_obs = EXPECTED['total_obs_range']
        assert min_obs <= len(pdata) <= max_obs, \
            f"Total observations {len(pdata)} outside expected range [{min_obs}, {max_obs}]"

    def test_all_studies_present(self, processed_data):
        """Check all studies are in output."""
        pdata, _, _ = processed_data
        studies_in_data = set(pdata['original_study'].unique())
        expected_studies = set(EXPECTED['studies'])
        assert studies_in_data == expected_studies, \
            f"Missing studies: {expected_studies - studies_in_data}"

    def test_treatment_is_binary(self, processed_data):
        """Check treatment variable is binary."""
        pdata, _, _ = processed_data
        iv_values = set(pdata['iv'].dropna().unique())
        assert iv_values == {0, 1}, \
            f"Treatment values should be {{0, 1}}, got {iv_values}"

    def test_sites_present(self, processed_data):
        """Check expected sites are present."""
        pdata, _, _ = processed_data
        sites = set(pdata['site'].unique())
        for site in EXPECTED['sample_sites']:
            assert site in sites, f"Site '{site}' not found in data"

    def test_required_covariates_exist(self, processed_data):
        """Check required covariates are present."""
        pdata, covariates_list, _ = processed_data
        for cov in EXPECTED['required_covariates']:
            assert cov in pdata.columns, f"Missing covariate: {cov}"

    def test_race_dummies_exist(self, processed_data):
        """Check race dummy variables exist."""
        pdata, _, _ = processed_data
        race_cols = [c for c in pdata.columns if c.startswith('RACE_')]
        assert len(race_cols) > 0, "No race dummy variables found"

    def test_major_dummies_exist(self, processed_data):
        """Check major dummy variables exist."""
        pdata, _, _ = processed_data
        major_cols = [c for c in pdata.columns if c.startswith('MAJOR_')]
        assert len(major_cols) > 0, "No major dummy variables found"

    def test_polideo_range(self, processed_data):
        """Check political ideology is in expected range."""
        pdata, _, _ = processed_data
        polideo = pdata['resp_polideo'].dropna()
        min_val, max_val = EXPECTED['polideo_range']
        assert polideo.min() >= min_val, \
            f"Polideo min {polideo.min()} < expected {min_val}"
        assert polideo.max() <= max_val, \
            f"Polideo max {polideo.max()} > expected {max_val}"

    def test_no_missing_iv_dv(self, processed_data):
        """Check no missing values in IV and DV after filtering."""
        pdata, _, _ = processed_data
        assert pdata['iv'].isna().sum() == 0, "Found missing IV values"
        assert pdata['dv'].isna().sum() == 0, "Found missing DV values"

    def test_min_observations_per_study(self, processed_data):
        """Check each study has minimum observations."""
        pdata, _, _ = processed_data
        for study in EXPECTED['studies']:
            n = len(pdata[pdata['original_study'] == study])
            assert n >= EXPECTED['min_obs_per_study'], \
                f"Study '{study}' has {n} obs, expected >= {EXPECTED['min_obs_per_study']}"

    def test_treatment_balance(self, processed_data):
        """Check treatment is reasonably balanced (not all 0 or all 1)."""
        pdata, _, _ = processed_data
        for study in EXPECTED['studies']:
            study_data = pdata[pdata['original_study'] == study]
            treatment_rate = study_data['iv'].mean()
            assert 0.1 < treatment_rate < 0.9, \
                f"Study '{study}' has extreme treatment rate: {treatment_rate:.3f}"

    def test_covariates_imputed(self, processed_data):
        """Check key covariates have been imputed (low missing rate)."""
        pdata, _, _ = processed_data
        imputed_vars = ['resp_age', 'resp_gender', 'resp_polideo']
        for var in imputed_vars:
            missing_rate = pdata[var].isna().sum() / len(pdata)
            assert missing_rate < 0.1, \
                f"Variable '{var}' has {missing_rate:.1%} missing after imputation"

    def test_csv_loadable(self, processed_data):
        """Check CSV output is loadable."""
        _, _, output_dir = processed_data
        csv_path = output_dir / 'Manylabs1_data.csv'
        df = pd.read_csv(csv_path)
        assert len(df) > 0, "CSV file is empty"

    def test_pickle_loadable(self, processed_data):
        """Check pickle output is loadable."""
        _, _, output_dir = processed_data
        pkl_path = output_dir / 'Manylabs1_data.pkl'
        df = pd.read_pickle(pkl_path)
        assert len(df) > 0, "Pickle file is empty"


# ============================================================================
# Statistical Validation Tests
# ============================================================================

@pytest.mark.skipif(not TEST_DATA_PATH.exists(), reason="Data file not found")
class TestStatisticalProperties:
    """Statistical validation tests."""

    @pytest.fixture(scope='class')
    def processed_data(self, tmp_path_factory):
        """Run preprocessing once for all tests in this class."""
        output_dir = tmp_path_factory.mktemp('output')
        pdata, covariates_list = preprocess_manylabs1(
            input_path=str(TEST_DATA_PATH),
            output_dir=str(output_dir),
            verbose=False
        )
        return pdata, covariates_list

    def test_age_distribution(self, processed_data):
        """Check age distribution is reasonable."""
        pdata, _ = processed_data
        age = pdata['resp_age'].dropna()
        assert 15 < age.mean() < 50, f"Mean age {age.mean():.1f} outside expected range"
        assert age.min() >= 10, f"Min age {age.min()} seems too low"
        assert age.max() <= 100, f"Max age {age.max()} seems too high"

    def test_gender_distribution(self, processed_data):
        """Check gender distribution is reasonable."""
        pdata, _ = processed_data
        gender = pdata['resp_gender'].dropna()
        female_rate = gender.mean()
        assert 0.3 < female_rate < 0.7, \
            f"Female rate {female_rate:.1%} seems extreme"

    def test_dv_has_variance(self, processed_data):
        """Check DV has variance within each study."""
        pdata, _ = processed_data
        for study in pdata['original_study'].unique():
            study_data = pdata[pdata['original_study'] == study]
            dv_var = study_data['dv'].var()
            assert dv_var > 0, f"Study '{study}' has zero DV variance"


# ============================================================================
# Standalone Test Runner
# ============================================================================

def run_quick_validation():
    """Run a quick validation without pytest."""
    print("=" * 60)
    print("Quick Validation of ManyLabs1 Preprocessing")
    print("=" * 60)

    if not TEST_DATA_PATH.exists():
        print(f"\nERROR: Data file not found at {TEST_DATA_PATH}")
        print("Please update TEST_DATA_PATH in this file.")
        return False

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\nRunning preprocessing...")
        pdata, covariates_list = preprocess_manylabs1(
            input_path=str(TEST_DATA_PATH),
            output_dir=tmp_dir,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("Validation Checks")
        print("=" * 60)

        checks_passed = 0
        checks_failed = 0

        # Check 1: Total observations
        min_obs, max_obs = EXPECTED['total_obs_range']
        if min_obs <= len(pdata) <= max_obs:
            print(f"[PASS] Total observations: {len(pdata):,}")
            checks_passed += 1
        else:
            print(f"[FAIL] Total observations: {len(pdata):,} (expected {min_obs}-{max_obs})")
            checks_failed += 1

        # Check 2: All studies present
        studies_in_data = set(pdata['original_study'].unique())
        expected_studies = set(EXPECTED['studies'])
        if studies_in_data == expected_studies:
            print(f"[PASS] All {len(expected_studies)} studies present")
            checks_passed += 1
        else:
            print(f"[FAIL] Missing studies: {expected_studies - studies_in_data}")
            checks_failed += 1

        # Check 3: Treatment is binary
        iv_values = set(pdata['iv'].dropna().unique())
        if iv_values == {0, 1}:
            print(f"[PASS] Treatment is binary (0, 1)")
            checks_passed += 1
        else:
            print(f"[FAIL] Treatment values: {iv_values}")
            checks_failed += 1

        # Check 4: No missing IV/DV
        if pdata['iv'].isna().sum() == 0 and pdata['dv'].isna().sum() == 0:
            print(f"[PASS] No missing IV/DV values")
            checks_passed += 1
        else:
            print(f"[FAIL] Missing IV: {pdata['iv'].isna().sum()}, DV: {pdata['dv'].isna().sum()}")
            checks_failed += 1

        # Check 5: Sites count
        n_sites = pdata['site'].nunique()
        if n_sites == EXPECTED['n_sites']:
            print(f"[PASS] Site count: {n_sites}")
            checks_passed += 1
        else:
            print(f"[FAIL] Site count: {n_sites} (expected {EXPECTED['n_sites']})")
            checks_failed += 1

        # Check 6: Covariates exist
        missing_covs = [c for c in EXPECTED['required_covariates'] if c not in pdata.columns]
        if not missing_covs:
            print(f"[PASS] All required covariates present")
            checks_passed += 1
        else:
            print(f"[FAIL] Missing covariates: {missing_covs}")
            checks_failed += 1

        # Check 7: Political ideology range
        polideo = pdata['resp_polideo'].dropna()
        if 0 <= polideo.min() and polideo.max() <= 6:
            print(f"[PASS] Political ideology range: [{polideo.min():.0f}, {polideo.max():.0f}]")
            checks_passed += 1
        else:
            print(f"[FAIL] Political ideology range: [{polideo.min()}, {polideo.max()}]")
            checks_failed += 1

        # Summary
        print("\n" + "=" * 60)
        print(f"Results: {checks_passed} passed, {checks_failed} failed")
        print("=" * 60)

        return checks_failed == 0


if __name__ == '__main__':
    # Run quick validation when executed directly
    success = run_quick_validation()
    sys.exit(0 if success else 1)
