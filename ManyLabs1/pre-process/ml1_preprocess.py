"""
ManyLabs1 Data Preprocessing Script (Python)

This script converts the R preprocessing pipeline (ML1_data_process.R and
ML1_data_process_2.R) to Python. It processes the raw ManyLabs1 SPSS data
and outputs a clean dataset ready for OSRCT analysis.

Input: CleanedDataset.sav (SPSS file from OSF)
Output: Manylabs1_data.pkl (Python pickle) and Manylabs1_data.csv

Usage:
    python ml1_preprocess.py --input /path/to/CleanedDataset.sav --output ./
"""

import argparse
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


# ============================================================================
# Constants and Mappings
# ============================================================================

# Race value mappings (from SPSS labels)
RACE_MAP = {
    '1': 'american indian/alaskan native',
    '2': 'east asian',
    '3': 'south asian',
    '4': 'native hawaiian or other pacific islander',
    '5': 'black or african american',
    '6': 'white',
    '7': 'more than one race - black/white',
    '8': 'more than one race - other',
    '9': 'other or unknown'
}

# Political ID mappings (from SPSS labels)
POLITICAL_ID_MAP = {
    -3.0: 'Strongly Conservative',
    -2.0: 'Moderately Conservative',
    -1.0: 'Slightly Conservative',
    0.0: 'Neutral (Moderate)',
    1.0: 'Slightly Liberal',
    2.0: 'Moderately Liberal',
    3.0: 'Strongly Liberal'
}

# Political ID to numeric (0-6 scale, conservative to liberal)
POLIDEO_NUMERIC = {
    'Strongly Conservative': 0,
    'Moderately Conservative': 1,
    'Slightly Conservative': 2,
    'Neutral (Moderate)': 3,
    'Slightly Liberal': 4,
    'Moderately Liberal': 5,
    'Strongly Liberal': 6
}

# Major field mappings (from codebook)
MAJOR_MAP = {
    1: 'Agriculture and related sciences',
    2: 'Biological sciences/life sciences',
    3: 'Business',
    4: 'Communications',
    5: 'Computer and information sciences',
    6: 'Education',
    7: 'Engineering, mathematics, physical sciences/technologies',
    8: 'Health professions or related sciences',
    9: 'Humanities (English, history, languages, philosophy, religion, etc.)',
    10: 'Law or legal studies',
    11: 'Psychology',
    12: 'Social sciences or history',
    13: 'Visual and performing arts'
}

# Major categories for dummy variables
MAJOR_CATEGORIES = {
    'Social': ['Communications', 'Education', 'Law or legal studies',
               'Psychology', 'Social sciences or history'],
    'Engineer': ['Computer and information sciences',
                 'Engineering, mathematics, physical sciences/technologies'],
    'Science': ['Biological sciences/life sciences',
                'Health professions or related sciences']
}

# Site to country mapping (derived from ManyLabs1 documentation)
SITE_COUNTRY_MAP = {
    'brasilia': 'Brazil',
    'swps': 'Poland',
    'swpson': 'Poland',
    'tilburg': 'Netherlands',
    'lse': 'UK',
    'unipd': 'Italy',
    'pi': 'Turkey',
    'msvu': 'Canada',
    'laurier': 'Canada',
    'wl': 'Canada'
}

# US sites (all others default to USA if in US)
US_SITES = ['abington', 'charles', 'conncoll', 'csun', 'help', 'ithaca',
            'jmu', 'ku', 'luc', 'mcdaniel', 'mturk', 'osu', 'oxy',
            'psu', 'qccuny', 'qccuny2', 'sdsu', 'tamu', 'tamuc', 'tamuon',
            'ufl', 'uva', 'vcu', 'wisc', 'wku', 'wpi']

# Study definitions
STUDIES = {
    'allowedforbidden': {
        'name': 'Allowed/Forbidden (Rugg, 1941)',
        'iv_col': 'allowedforbiddenGroup',
        'dv_col': 'allowedforbidden',
        'reverse_iv': True,
        'flip_dv_for_iv0': True
    },
    'anchoring1': {
        'name': 'Anchoring (Jacowitz & Kahneman, 1995) - NYC',
        'iv_col': 'anch1group',
        'dv_col': 'Ranch1',
        'reverse_iv': False
    },
    'anchoring2': {
        'name': 'Anchoring (Jacowitz & Kahneman, 1995) - Chicago',
        'iv_col': 'anch2group',
        'dv_col': 'Ranch2',
        'reverse_iv': False
    },
    'anchoring3': {
        'name': 'Anchoring (Jacowitz & Kahneman, 1995) - Everest',
        'iv_col': 'anch3group',
        'dv_col': 'Ranch3',
        'reverse_iv': False
    },
    'anchoring4': {
        'name': 'Anchoring (Jacowitz & Kahneman, 1995) - Babies',
        'iv_col': 'anch4group',
        'dv_col': 'Ranch4',
        'reverse_iv': False
    },
    'contact': {
        'name': 'Imagined contact (Husnu & Crisp, 2010)',
        'iv_col': 'ContactGroup',
        'dv_col': 'Imagineddv',
        'reverse_iv': False
    },
    'flag': {
        'name': 'Flag Priming (Carter et al., 2011)',
        'iv_col': 'flagGroup',
        'dv_col': 'flagdv',
        'reverse_iv': False
    },
    'gainloss': {
        'name': 'Gain vs loss framing (Tversky & Kahneman, 1981)',
        'iv_col': 'gainlossgroup',
        'dv_col': 'gainlossDV',
        'reverse_iv': True
    },
    'gambfal': {
        'name': "Retro. gambler's fallacy (Oppenheimer & Monin, 2009)",
        'iv_col': 'gambfalgroup',
        'dv_col': 'gambfalDV',
        'reverse_iv': False
    },
    'iat': {
        'name': 'Sex diff. in implicit math attitudes (Nosek et al., 2002)',
        'iv_col': 'partgender',
        'dv_col': 'd_art',
        'reverse_iv': True,
        # partgender: 1=no response, 2=female, 3=male
        # We want: female=1, male=0 (after reverse)
        # So: subtract 2 to get female=0, male=1, then reverse to get female=1, male=0
        # But first filter out value 1 (no response) by setting to NaN
        'iv_transform': lambda x: np.where(x == 1, np.nan, x - 2)
    },
    'money': {
        'name': 'Currency priming (Caruso et al., 2012)',
        'iv_col': 'MoneyGroup',
        'dv_col': 'Sysjust',
        'reverse_iv': False
    },
    'quote': {
        'name': 'Quote Attribution (Lorge & Curtis, 1936)',
        'iv_col': 'quoteGroup',
        'dv_col': 'quote',
        'reverse_iv': False
    },
    'reciprocity': {
        'name': 'Norm of reciprocity (Hyman and Sheatsley, 1950)',
        'iv_col': 'reciprocitygroup',
        'dv_col': 'reciprocityus',
        'reverse_iv': True
    },
    'scales': {
        'name': 'Low-vs.-high category scales (Schwarz et al., 1985)',
        'iv_col': 'scalesgroup',
        'dv_col': 'scales',
        'reverse_iv': False
    },
    'sunk': {
        'name': 'Sunk costs (Oppenheimer et al., 2009)',
        'iv_col': 'sunkgroup',
        'dv_col': 'sunkDV',
        'reverse_iv': True
    }
}


# ============================================================================
# Data Loading
# ============================================================================

def load_spss_data(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """Load SPSS .sav file and return dataframe with metadata."""
    df, meta = pyreadstat.read_sav(filepath)

    # Clean string columns - replace empty/missing markers with NaN
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: np.nan if pd.isna(x) or str(x).strip() in ['', '.', '-'] else x
        )

    return df, meta


# ============================================================================
# Covariate Processing (Phase 1 - from ML1_data_process.R)
# ============================================================================

def process_respondent_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Process respondent demographic covariates."""

    # Gender
    df['resp_sex'] = df['sex'].apply(
        lambda x: 'female' if x == 'f' else ('male' if x == 'm' else np.nan)
    )

    # Age
    df['resp_age'] = pd.to_numeric(df['age'], errors='coerce')

    # Race - map numeric codes to labels
    df['resp_race'] = df['race'].astype(str).str.strip()
    for code, label in RACE_MAP.items():
        df.loc[df['resp_race'] == code, 'resp_race'] = label
    df['resp_race'] = df['resp_race'].str.lower()

    # Ethnicity
    df['resp_ethnicity'] = df['ethnicity'].apply(
        lambda x: 'hispanic or latino' if str(x) == '1'
                  else ('not hispanic or latino' if str(x) == '2' else np.nan)
    )

    # Citizenship - coalesce citizenship and citizenship2
    df['resp_citizenship'] = df['citizenship'].combine_first(df['citizenship2'])

    # Major - map numeric codes
    df['resp_major'] = df['major'].apply(
        lambda x: MAJOR_MAP.get(int(x), np.nan) if pd.notna(x) and str(x).replace('.','').isdigit() else np.nan
    )

    # Political ID - map numeric codes to labels
    df['resp_pid'] = df['politicalid'].map(POLITICAL_ID_MAP)

    # Native language
    df['resp_nativelang'] = df['nativelang'].str.lower().str.strip()

    # Religion
    df['resp_religion'] = df['religion']

    # American identity questions
    df['resp_american'] = pd.to_numeric(df['flagsupplement1'], errors='coerce')
    df['resp_american_pid'] = pd.to_numeric(df['flagsupplement2'], errors='coerce')
    df['resp_american_ideo'] = pd.to_numeric(df['flagsupplement3'], errors='coerce')

    return df


def process_experimenter_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Process experimenter covariates."""

    # Experimenter sex
    df['exp_sex'] = df['expgender'].apply(
        lambda x: x if x in ['female', 'male'] else np.nan
    )

    # Experimenter race
    df['exp_race'] = df['exprace'].astype(str).str.strip()
    for code, label in RACE_MAP.items():
        df.loc[df['exp_race'] == code, 'exp_race'] = label
    df.loc[df['exp_race'] == '10', 'exp_race'] = 'hispanic or latino'
    df['exp_race'] = df['exp_race'].str.lower()

    return df


def process_study_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Process study environment covariates."""

    df['study_numparticipants'] = df['numparticipants']
    df['study_exprunafter'] = df['exprunafter']
    df['study_separated'] = df['separatedornot']
    df['study_recruit'] = df['recruitment']
    df['study_compensation'] = df['compensation']
    df['study_online'] = pd.to_numeric(df['lab_or_online'], errors='coerce')

    # Add country based on site
    df['study_country'] = df['referrer'].str.lower().apply(
        lambda x: SITE_COUNTRY_MAP.get(x, 'USA')
    )
    df['study_usa'] = (df['study_country'] == 'USA').astype(int)

    return df


def process_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Process filter variables."""

    # IAT filter - also exclude if no gender response
    df['filter_iat'] = df['IATfilter'].apply(
        lambda x: 0 if pd.isna(x) else x
    )
    df.loc[df['resp_sex'].isna(), 'filter_iat'] = 0

    return df


# ============================================================================
# Treatment and Outcome Processing (Phase 1 continued)
# ============================================================================

def create_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format to long format with one row per (subject, study)."""

    # Base columns to keep for each observation
    base_cols = [
        'session_id', 'session_date', 'referrer',
        'resp_sex', 'resp_age', 'resp_race', 'resp_ethnicity',
        'resp_citizenship', 'resp_major', 'resp_pid', 'resp_nativelang',
        'resp_religion', 'resp_american', 'resp_american_pid', 'resp_american_ideo',
        'exp_sex', 'exp_race',
        'study_numparticipants', 'study_exprunafter', 'study_separated',
        'study_recruit', 'study_compensation', 'study_online',
        'study_country', 'study_usa',
        'filter_iat'
    ]

    long_data = []

    for study_name, study_info in STUDIES.items():
        # Get IV and DV columns
        iv_col = study_info['iv_col']
        dv_col = study_info['dv_col']

        # Skip if columns don't exist
        if iv_col not in df.columns or dv_col not in df.columns:
            warnings.warn(f"Skipping study {study_name}: missing columns")
            continue

        # Select base columns that exist
        existing_base = [c for c in base_cols if c in df.columns]

        # Create study-specific dataframe
        study_df = df[existing_base + [iv_col, dv_col]].copy()

        # Process IV
        study_df['iv'] = pd.to_numeric(study_df[iv_col], errors='coerce')

        # Apply IV transform if specified
        if 'iv_transform' in study_info:
            study_df['iv'] = study_info['iv_transform'](study_df['iv'])

        # Reverse IV if needed
        if study_info.get('reverse_iv', False):
            study_df['iv'] = 1 - study_df['iv']

        # Process DV
        study_df['dv'] = pd.to_numeric(study_df[dv_col], errors='coerce')

        # Flip DV for IV=0 if needed (allowedforbidden special case)
        if study_info.get('flip_dv_for_iv0', False):
            mask = study_df['iv'] == 0
            study_df.loc[mask, 'dv'] = 1 - study_df.loc[mask, 'dv']

        # Add study identifiers
        study_df['original_study'] = study_name
        study_df['original_study_name'] = study_info['name']

        # Drop the original IV/DV columns
        study_df = study_df.drop(columns=[iv_col, dv_col])

        long_data.append(study_df)

    # Concatenate all studies
    result = pd.concat(long_data, ignore_index=True)

    # Rename columns
    result = result.rename(columns={
        'session_id': 'id',
        'session_date': 'date',
        'referrer': 'site'
    })

    # Convert date
    result['date'] = pd.to_datetime(result['date'], errors='coerce')

    return result


# ============================================================================
# Phase 2 Processing (from ML1_data_process_2.R)
# ============================================================================

def process_phase2(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase 2 processing: dummy variables and NA imputation."""

    # Create binary gender variable
    df['resp_gender'] = (df['resp_sex'] == 'female').astype(float)
    df.loc[df['resp_sex'].isna(), 'resp_gender'] = np.nan

    # Create Hispanic ethnicity indicator
    df['resp_ethnicity_hisp'] = (df['resp_ethnicity'] == 'hispanic or latino').astype(float)
    df.loc[df['resp_ethnicity'].isna(), 'resp_ethnicity_hisp'] = np.nan

    # Process race into categories
    df['RACE'] = df['resp_race'].apply(categorize_race)

    # Create race dummy variables
    race_dummies = pd.get_dummies(df['RACE'], prefix='RACE')
    df = pd.concat([df, race_dummies], axis=1)

    # Process major into categories
    df['MAJOR'] = df['resp_major'].apply(categorize_major)

    # Create major dummy variables
    major_dummies = pd.get_dummies(df['MAJOR'], prefix='MAJOR')
    df = pd.concat([df, major_dummies], axis=1)

    # Process political ideology to numeric scale
    df['resp_polideo'] = df['resp_pid'].map(POLIDEO_NUMERIC)

    # Filter out rows with missing IV or DV
    df = df[df['iv'].notna() & df['dv'].notna()].copy()

    # Impute missing values with site-level means
    df = impute_site_means(df)

    return df


def categorize_race(race: str) -> str:
    """Categorize race into broader groups."""
    if pd.isna(race):
        return 'others'

    race = str(race).lower()

    if race == 'white':
        return 'white'
    elif race in ['belgisch nederlands', 'nederlands', 'nederlandse',
                  'marokkaans nederlands', 'italiaans nederlands']:
        return 'nederland'
    elif race == 'black or african american':
        return 'black_american'
    elif race in ['chinese', 'east asian']:
        return 'east_asian'
    elif race in ['indian', 'south asian']:
        return 'south_asian'
    elif race in ['more than one race - other', 'more than one race - black/white']:
        return 'more_than_one'
    elif race == 'american indian/alaskan native':
        return 'american_indian'
    elif race in ['brazilbrown', 'brazilblack', 'brazilwhite', 'brazilyellow']:
        return 'brazil'
    else:
        return 'others'


def categorize_major(major: str) -> str:
    """Categorize major into broader groups."""
    if pd.isna(major):
        return 'others'

    for category, majors in MAJOR_CATEGORIES.items():
        if major in majors:
            return category

    return 'others'


def impute_site_means(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with site-level means."""

    # Variables to impute
    impute_vars = [
        'resp_age', 'resp_gender', 'resp_ethnicity_hisp', 'resp_polideo',
        'resp_american', 'resp_american_pid', 'resp_american_ideo'
    ]

    for var in impute_vars:
        if var not in df.columns:
            continue

        # Compute site-level means
        site_means = df.groupby('site')[var].transform('mean')

        # Fill NaN with site means
        df[var] = df[var].fillna(site_means)

        # If still NaN (entire site missing), fill with global mean
        global_mean = df[var].mean()
        df[var] = df[var].fillna(global_mean)

    return df


# ============================================================================
# Main Processing Function
# ============================================================================

def preprocess_manylabs1(
    input_path: str,
    output_dir: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Main preprocessing function.

    Parameters
    ----------
    input_path : str
        Path to CleanedDataset.sav
    output_dir : str
        Directory to save output files
    verbose : bool
        Print progress messages

    Returns
    -------
    pdata : pd.DataFrame
        Processed data in long format
    """

    if verbose:
        print("=" * 60)
        print("ManyLabs1 Data Preprocessing")
        print("=" * 60)

    # Load data
    if verbose:
        print(f"\n[1/6] Loading SPSS data from {input_path}...")
    df, meta = load_spss_data(input_path)
    if verbose:
        print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Process respondent covariates
    if verbose:
        print("\n[2/6] Processing respondent covariates...")
    df = process_respondent_covariates(df)

    # Process experimenter covariates
    if verbose:
        print("\n[3/6] Processing experimenter covariates...")
    df = process_experimenter_covariates(df)

    # Process study covariates
    if verbose:
        print("\n[4/6] Processing study covariates...")
    df = process_study_covariates(df)
    df = process_filters(df)

    # Create long format
    if verbose:
        print("\n[5/6] Creating long format (one row per subject-study)...")
    df = create_long_format(df)
    if verbose:
        print(f"      Created {len(df):,} observations across {df['original_study'].nunique()} studies")

    # Phase 2 processing
    if verbose:
        print("\n[6/6] Phase 2: Dummy variables and imputation...")
    pdata = process_phase2(df)
    if verbose:
        print(f"      Final dataset: {len(pdata):,} observations")

    # Define covariates list (matching R output)
    race_cols = [c for c in pdata.columns if c.startswith('RACE_')]
    major_cols = [c for c in pdata.columns if c.startswith('MAJOR_')]

    covariates_list = [
        'resp_gender', 'resp_age', 'resp_ethnicity_hisp',
        'resp_polideo',
        'resp_american', 'resp_american_pid', 'resp_american_ideo'
    ] + race_cols + major_cols

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle (similar to RData)
    pkl_path = output_dir / 'Manylabs1_data.pkl'
    pdata.to_pickle(pkl_path)
    if verbose:
        print(f"\n      Saved: {pkl_path}")

    # Save as CSV
    csv_path = output_dir / 'Manylabs1_data.csv'
    pdata.to_csv(csv_path, index=False)
    if verbose:
        print(f"      Saved: {csv_path}")

    # Save covariates list
    cov_path = output_dir / 'covariates_list.txt'
    with open(cov_path, 'w') as f:
        f.write('\n'.join(covariates_list))
    if verbose:
        print(f"      Saved: {cov_path}")

    # Print summary statistics
    if verbose:
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"\nTotal observations: {len(pdata):,}")
        print(f"Unique subjects: {pdata['id'].nunique():,}")
        print(f"Unique sites: {pdata['site'].nunique()}")
        print(f"Studies: {pdata['original_study'].nunique()}")

        print("\nObservations per study:")
        for study in sorted(pdata['original_study'].unique()):
            n = len(pdata[pdata['original_study'] == study])
            print(f"  {study}: {n:,}")

        print("\nTreatment distribution:")
        print(f"  IV=0: {(pdata['iv'] == 0).sum():,}")
        print(f"  IV=1: {(pdata['iv'] == 1).sum():,}")

        print("\nCovariates missing values:")
        for cov in covariates_list[:7]:  # Just main covariates
            if cov in pdata.columns:
                n_missing = pdata[cov].isna().sum()
                print(f"  {cov}: {n_missing}")

    return pdata, covariates_list


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ManyLabs1 data for OSRCT analysis'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='CleanedDataset.sav',
        help='Path to CleanedDataset.sav'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='.',
        help='Output directory'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    preprocess_manylabs1(
        input_path=args.input,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
