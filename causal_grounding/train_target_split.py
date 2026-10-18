"""
Train/Target Environment Splitting for Causal Grounding

This module splits multi-site data into training and target environments.

Key concept:
- Training sites have BOTH F=on (RCT) and F=idle (OSRCT) data
- Target site has ONLY F=idle data
- F=on comes from original RCT, F=idle from OSRCT-confounded data
"""

import pandas as pd
from typing import Dict, List, Tuple


def add_regime_indicator(
    rct_data: pd.DataFrame,
    osrct_data: pd.DataFrame,
    regime_col: str = 'F'
) -> pd.DataFrame:
    """
    Combine RCT and OSRCT data with regime indicator.

    Args:
        rct_data: Original RCT data (unconfounded)
        osrct_data: OSRCT-sampled data (confounded)
        regime_col: Name for regime column

    Returns:
        Combined DataFrame with regime_col ('on' or 'idle')
    """
    rct = rct_data.copy()
    rct[regime_col] = 'on'

    osrct = osrct_data.copy()
    osrct[regime_col] = 'idle'

    # Align columns, preserving the RCT frame's order. Do not use a set
    # intersection here: string hashing is randomised per process, so the
    # column order would differ between runs.
    osrct_columns = set(osrct.columns)
    common_cols = [col for col in rct.columns if col in osrct_columns]

    return pd.concat([rct[common_cols], osrct[common_cols]], ignore_index=True)


def get_available_sites(
    data: pd.DataFrame,
    site_col: str = 'site'
) -> List[str]:
    """Return sorted list of unique site identifiers."""
    return sorted(data[site_col].unique().tolist())


def create_train_target_split(
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    target_site: str,
    site_col: str = 'site',
    regime_col: str = 'F'
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Split data into training environments and target environment.

    Training environments: All sites except target, with both F=on and F=idle
    Target environment: Target site only, with F=idle only

    Args:
        osrct_data: OSRCT-confounded data (all sites)
        rct_data: Original RCT data (all sites)
        target_site: Site identifier to hold out as target
        site_col: Column name for site identifier
        regime_col: Column name for regime indicator

    Returns:
        training_data: Dict[site_id -> DataFrame with 'F' column]
        target_data: DataFrame (F=idle only, target site only)

    Raises:
        ValueError: If target_site not found in data
    """
    available_sites = get_available_sites(osrct_data, site_col)

    if target_site not in available_sites:
        raise ValueError(
            f"Target site '{target_site}' not found. "
            f"Available sites: {available_sites}"
        )

    training_sites = [s for s in available_sites if s != target_site]
    training_data = {}

    for site in training_sites:
        site_rct = rct_data[rct_data[site_col] == site].copy()
        site_osrct = osrct_data[osrct_data[site_col] == site].copy()

        if len(site_rct) > 0 and len(site_osrct) > 0:
            training_data[site] = add_regime_indicator(
                site_rct, site_osrct, regime_col
            )

    # Target: only F=idle data
    target_data = osrct_data[osrct_data[site_col] == target_site].copy()
    target_data[regime_col] = 'idle'

    return training_data, target_data


def summarize_split(
    training_data: Dict[str, pd.DataFrame],
    target_data: pd.DataFrame,
    regime_col: str = 'F'
) -> str:
    """Generate summary string of train/target split."""
    lines = ["Train/Target Split Summary", "=" * 40]

    lines.append(f"\nTraining sites: {len(training_data)}")
    for site, df in sorted(training_data.items()):
        n_on = (df[regime_col] == 'on').sum()
        n_idle = (df[regime_col] == 'idle').sum()
        lines.append(f"  {site}: {n_on} on, {n_idle} idle")

    lines.append(f"\nTarget site: {len(target_data)} samples (idle only)")

    return "\n".join(lines)
