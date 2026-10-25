"""Column schema of the ManyLabs / OSRCT site tables.

The one home for the column names that the ManyLabs1 preprocessing writes and
that every OSRCT experiment reads. Import these instead of spelling the
literals. See AR-DRY-102.

The generic package API (``causal_grounding.api``) defaults to ``X`` / ``Y`` /
``F``. These names are specific to the ManyLabs data family.
"""

from typing import Tuple

TREATMENT_COLUMN = "iv"
"""Randomised treatment indicator (0/1)."""

SOURCE_OUTCOME_COLUMN = "dv"
"""Raw study outcome before binarisation."""

OUTCOME_COLUMN = "dv_binary"
"""Binarised outcome (0/1) derived from :data:`SOURCE_OUTCOME_COLUMN`."""

UNIT_ID_COLUMN = "id"
"""Participant identifier, derived from ``session_id``."""

SITE_COLUMN = "site"
"""ManyLabs collection site."""

REGIME_COLUMN = "F"
"""Regime indicator: ``"on"`` for RCT rows, ``"idle"`` for OSRCT rows."""

MANYLABS_COVARIATES: Tuple[str, ...] = (
    "resp_age_bin",
    "resp_gender_bin",
    "resp_polideo_bin",
)
"""Binary participant covariates, in the canonical order."""
