"""Shared cohort restrictions for ManyLabs OSRCT experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd


OBSERVED_GENDER_SOURCE_COLUMN = "resp_sex"
OBSERVED_GENDER_COHORT_RULE = "resp_sex_not_missing"


def restrict_to_observed_gender(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows whose original, pre-imputation gender response is missing.

    ``resp_gender`` cannot identify these rows because the ManyLabs
    preprocessing pipeline replaces its missing values with site-level means.
    The original ``resp_sex`` column retains missingness and therefore defines
    the complete-case cohort unambiguously.

    Args:
        data: ManyLabs rows containing ``resp_sex`` and, when available,
            ``original_study``, ``site``, and ``id``.

    Returns:
        A filtered copy and study/site-level diagnostics. Diagnostics include
        aggregate rows whose ``study`` or ``site`` value is ``"__all__"``.

    Raises:
        ValueError: If the original gender-response column is unavailable or
            the restriction removes every row.
    """

    if OBSERVED_GENDER_SOURCE_COLUMN not in data:
        raise ValueError(
            "Observed-gender cohort restriction requires original column "
            f"{OBSERVED_GENDER_SOURCE_COLUMN!r}"
        )
    excluded = data[OBSERVED_GENDER_SOURCE_COLUMN].isna()
    filtered = data.loc[~excluded].copy()
    if filtered.empty:
        raise ValueError("Observed-gender cohort restriction removed every row")

    working = data.copy()
    working["_cohort_excluded"] = excluded.to_numpy(dtype=bool)
    working["_cohort_study"] = (
        working["original_study"].astype(str)
        if "original_study" in working
        else "__all__"
    )
    working["_cohort_site"] = (
        working["site"].astype(str) if "site" in working else "__all__"
    )

    def summarize(group: pd.DataFrame, study: str, site: str) -> dict[str, Any]:
        """Summarize one cohort-filter reporting group."""

        group_excluded = group["_cohort_excluded"]
        return {
            "cohort_filter": OBSERVED_GENDER_COHORT_RULE,
            "study": study,
            "site": site,
            "n_rows_before": int(len(group)),
            "n_rows_excluded": int(group_excluded.sum()),
            "n_rows_after": int((~group_excluded).sum()),
            "n_unique_ids_excluded": (
                int(group.loc[group_excluded, "id"].nunique())
                if "id" in group
                else pd.NA
            ),
        }

    rows = [summarize(working, "__all__", "__all__")]
    for study, study_data in working.groupby("_cohort_study", sort=True):
        rows.append(summarize(study_data, str(study), "__all__"))
        for site, site_data in study_data.groupby("_cohort_site", sort=True):
            rows.append(summarize(site_data, str(study), str(site)))
    diagnostics = pd.DataFrame(rows)
    return filtered, diagnostics


__all__ = [
    "OBSERVED_GENDER_COHORT_RULE",
    "OBSERVED_GENDER_SOURCE_COLUMN",
    "restrict_to_observed_gender",
]
