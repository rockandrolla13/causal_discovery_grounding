"""OSRCT data-file discovery for the ManyLabs experiment studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_MANYLABS_STUDIES = (
    "gainloss",
    "reciprocity",
    "allowedforbidden",
)


def find_osrct_file(config: Any) -> Path:
    """
    Find an OSRCT file for a study, pattern, beta, and seed configuration.

    Args:
        config: Object exposing ``data_root``, ``study``, ``pattern``, ``beta``,
            and ``seed`` attributes.

    Returns:
        Path to the first matching Parquet, CSV, or pickle file.

    Raises:
        FileNotFoundError: If none of the supported naming conventions exist.
    """
    data_root = Path(config.data_root)
    study_dir = data_root / config.study
    beta = config.beta
    stems = [
        f"{config.pattern}_beta{beta}_seed{config.seed}",
        f"{config.pattern}_beta{beta:g}_seed{config.seed}",
        f"{config.pattern}_beta{beta:.1f}_seed{config.seed}",
        f"{config.pattern}_beta{beta:.2f}_seed{config.seed}",
        f"{config.study}_{config.pattern}_beta_{beta}",
    ]
    candidates = [
        study_dir / f"{stem}{suffix}"
        for stem in stems
        for suffix in (".parquet", ".csv", ".pkl")
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "OSRCT file not found. Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )

