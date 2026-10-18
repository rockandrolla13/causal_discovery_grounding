"""
The eleven invariants, enforced centrally, all failing loudly.

Ten of the eleven raise :class:`LoaderValidationError` naming the site and the
offending columns or values. Nothing skips, coerces or repairs — a skipped site
is a silent failure with paperwork.

Invariant 9 is the exception. It records rather than raises, because real data
violates it; see :func:`check_covariates_vary`. Pass
``strict_covariate_variation=True`` to make it fatal too.

Ten invariants come from ``docs/loader_contract_spec.md`` §6. The eleventh is
new; see :func:`check_row_order_preserved` for why it exists.

Three of the spec's assumptions were wrong and are corrected here:

- Spec §2 says seven grounder requirements go unenforced. **Five are enforced**
  at ``naturalness_grounder.py:283-288`` and ``:298-301``. The invariants below
  still check them, cheaply and earlier, but they are belt-and-braces rather
  than the sole guard.
- Spec §6 offers "raise, *or* skip site with a recorded reason" for invariant 8.
  Skipping is not available. ``make_osrct_regimes`` standardises globally before
  splitting by site, so dropping one site perturbs every other site's numbers.
- Spec §6 says invariants 9 and 10 "are not currently violated anywhere".
  **Invariant 9 is violated**, at 11 of 108 site-study combinations on the
  OSRCT data. Measured, not inferred.
"""

from typing import Dict, Set, Tuple
from dataclasses import replace
import logging

import pandas as pd

from .contract import SiteBundle, SiteTable

logger = logging.getLogger(__name__)


class LoaderValidationError(ValueError):
    """An invariant was violated. Always names the site and the cause."""


# Invariant 10 is the only cross-bundle check, so it needs process state.
_LOADED_DATASET_NAMES: Set[str] = set()


def reset_loaded_datasets() -> None:
    """Forget every dataset name seen so far.

    Only invariant 10 consults this. Tests that load the same bundle twice must
    call it, otherwise the second load fails on a name collision that is an
    artefact of the test rather than of the data.
    """
    _LOADED_DATASET_NAMES.clear()


# =============================================================================
# PER-SITE INVARIANTS
# =============================================================================

def check_required_columns(table: SiteTable) -> None:
    """Invariant 1 and 4: treatment, outcome and every covariate exist."""
    required = (table.treatment, table.outcome, table.unit_id, *table.covariates)
    missing = [column for column in required if column not in table.frame.columns]
    if missing:
        raise LoaderValidationError(
            f"site {table.site_id!r}: missing column(s) {missing}. "
            f"Frame has {list(table.frame.columns)}"
        )


def check_binary(table: SiteTable, column: str, invariant: int) -> None:
    """Invariant 2 and 3: values are a subset of {0, 1}.

    NaN is *not* tolerated here even though ``is_binary_series`` in
    ``naturalness_statistics`` drops it first. That ``.dropna()`` is precisely
    why an all-NaN treatment column passes the grounder's own binary check.
    """
    observed = pd.unique(table.frame[column].dropna())
    n_null = int(table.frame[column].isna().sum())
    offenders = sorted(value for value in observed if value not in (0, 1))
    if offenders or n_null:
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant {invariant}, column {column!r} "
            f"is not binary. Observed non-binary values {offenders}, "
            f"{n_null} null(s)."
        )


def check_no_nulls(table: SiteTable) -> None:
    """Invariant 6: no NaN in treatment, outcome or any declared covariate.

    Reports per-column counts rather than a bare boolean, because the useful
    question is always *which* column and *how many*.
    """
    columns = (table.treatment, table.outcome, *table.covariates)
    counts = {
        column: int(table.frame[column].isna().sum())
        for column in columns
        if int(table.frame[column].isna().sum()) > 0
    }
    if counts:
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant 6, null values present {counts}"
        )


def check_unit_id_unique(table: SiteTable) -> None:
    """Invariant 7: unit_id is unique within the site.

    Note:
        This can be false after regime construction because idle and on
        regimes are concatenated. Do not read a pass here as coverage.
    """
    ids = table.frame[table.unit_id]
    n_duplicate = int(len(ids) - ids.nunique(dropna=False))
    if n_duplicate:
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant 7, {n_duplicate} duplicate "
            f"{table.unit_id!r} value(s)"
        )


def check_min_rows(table: SiteTable, min_rows: int) -> None:
    """Invariant 8: the site has at least ``min_rows`` rows. Raises, never skips.

    The default of 1 is deliberate and reproduces current behaviour:
    ``train_target_split.py:96`` already drops a site missing either regime,
    silently. This makes that drop loud without changing which sites survive.

    Do not raise the default to 50 on the strength of ``MIN_SITE_SAMPLE_SIZE``
    in the benchmark generator. That constant sits on a dead path, and adopting
    it would drop eight of gainloss's 36 sites.
    """
    if len(table) < min_rows:
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant 8, {len(table)} row(s) "
            f"below min_rows={min_rows}"
        )


def check_covariates_vary(table: SiteTable, *, strict: bool) -> Tuple[str, ...]:
    """Invariant 9: every covariate has at least two observed levels.

    A constant covariate cannot be a naturalness candidate — there is no
    contrast to measure.

    Returns:
        The constant covariate names, empty if all vary.

    Raises:
        LoaderValidationError: Only when ``strict`` is set.

    Note:
        **Not fatal by default, and the spec is wrong about this.**
        ``docs/loader_contract_spec.md`` §6 says invariants 9 and 10 "are not
        currently violated anywhere". Invariant 9 is violated on real data, at
        11 of 108 site-study combinations: ``resp_age_bin`` is constant at 2 of
        36 gainloss sites, 4 of 36 reciprocity, and 5 of 36 allowedforbidden.
        ``resp_gender_bin`` and ``resp_polideo_bin`` never go constant.

        The cause is structural rather than a data error. The age threshold is
        21 and these are university samples, so some sites sit entirely on one
        side of it. Median age across gainloss is exactly 21.

        Raising would therefore reject data that loads fine today and break the
        §8 equivalence proof. So the default records instead, and the names go
        into ``provenance["constant_covariates"]`` where a caller can act on
        them. Pass ``strict_covariate_variation=True`` to make it fatal.

        This is the same reasoning that puts ``min_rows`` at 1: reproduce
        current behaviour, make the silent thing visible, and let the change in
        behaviour be a separate, deliberate decision.
    """
    constant = tuple(
        column for column in table.covariates
        if table.frame[column].nunique(dropna=True) < 2
    )
    if constant and strict:
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant 9, covariate(s) "
            f"{list(constant)} have fewer than 2 observed levels"
        )
    if constant:
        logger.warning(
            "site %r: covariate(s) %s are constant; they cannot be naturalness "
            "candidates at this site", table.site_id, list(constant),
        )
    return constant


def check_row_order_preserved(table: SiteTable) -> None:
    """Invariant 11: rows are in source order, and positions are real.

    Not in the spec. A reordered frame has the same shape, columns and values,
    so no other invariant can see the difference, while positional downstream
    operations can silently change.

    Note:
        This is *within-frame* order. It is not a licence to align two frames
        positionally: OSRCT and RCT have different lengths and different ids at
        the same offsets, so cross-regime alignment must go through the id
        column.
    """
    positions = table.source_positions
    if len(positions) != len(table.frame):
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant 11, {len(positions)} recorded "
            f"source position(s) for {len(table.frame)} row(s)"
        )
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise LoaderValidationError(
            f"site {table.site_id!r}: invariant 11, source positions are not "
            f"strictly increasing — rows were reordered"
        )


# =============================================================================
# CROSS-SITE INVARIANTS
# =============================================================================

def check_covariates_identical(bundle: SiteBundle) -> None:
    """Invariant 5: covariate set *and order* match across every site.

    This is the invariant that matters most. It is the reason cross-site GP
    training is meaningful, and nothing checked it before this module existed.
    Order matters as well as membership, because downstream code indexes
    covariates positionally.
    """
    for site_id, table in bundle.sites.items():
        if table.covariates != bundle.covariates:
            missing = set(bundle.covariates) - set(table.covariates)
            extra = set(table.covariates) - set(bundle.covariates)
            detail = (
                f"missing {sorted(missing)}, extra {sorted(extra)}"
                if (missing or extra)
                else f"same set, different order: {table.covariates}"
            )
            raise LoaderValidationError(
                f"site {site_id!r}: invariant 5, covariates disagree with the "
                f"bundle. Bundle has {bundle.covariates}; {detail}"
            )


def check_membership_complete(bundle: SiteBundle) -> None:
    """Invariant 11, bundle half: the site split loses and duplicates nothing.

    Every parent row must land in exactly one site. A row in two sites, or in
    none, changes the global standardisation that regime construction depends on.
    """
    seen: list = []
    for table in bundle.sites.values():
        seen.extend(table.source_positions)
    duplicated = len(seen) - len(set(seen))
    if duplicated:
        raise LoaderValidationError(
            f"dataset {bundle.dataset!r}: invariant 11, {duplicated} row(s) "
            f"appear in more than one site"
        )
    expected = set(range(len(seen)))
    if set(seen) != expected:
        lost = sorted(expected - set(seen))[:10]
        raise LoaderValidationError(
            f"dataset {bundle.dataset!r}: invariant 11, the site split does not "
            f"cover the parent frame. First missing position(s): {lost}"
        )


def check_dataset_name_unique(bundle: SiteBundle, *, strict: bool) -> None:
    """Invariant 10: no two loaded bundles share a name.

    Guards against silent site-id collision when two bundles are merged.

    Note:
        **Off by default, because the spec put it in the wrong place.**
        §6 frames this as a load-time check, but loading the same dataset twice
        in one process is completely normal, including repeated simulation or
        sensitivity runs. A load-time check would raise on the second load.

        The collision it actually guards against happens at *merge* time, not
        load time. Until something merges bundles, there is nothing to protect,
        so this is opt-in via ``strict_dataset_names=True``. When a merge helper
        is written, the check belongs inside it and this can go.
    """
    if not strict:
        return
    if bundle.dataset in _LOADED_DATASET_NAMES:
        raise LoaderValidationError(
            f"dataset name {bundle.dataset!r} already loaded in this process. "
            f"Call reset_loaded_datasets() if this is intentional."
        )
    _LOADED_DATASET_NAMES.add(bundle.dataset)


# =============================================================================
# THE ENTRY POINT
# =============================================================================

def validate_bundle(
    bundle: SiteBundle,
    *,
    min_rows: int = 1,
    strict_covariate_variation: bool = False,
    strict_dataset_names: bool = False,
) -> SiteBundle:
    """Apply every invariant. Returns a validated bundle, or raises.

    Args:
        bundle: The bundle a loader's ``read()`` produced.
        min_rows: Invariant 8's threshold. See :func:`check_min_rows` before
            changing it.
        strict_covariate_variation: Make invariant 9 fatal. Off by default
            because it is violated by real data; see
            :func:`check_covariates_vary`.
        strict_dataset_names: Enforce invariant 10. Off by default because
            reloading a dataset in one process is normal; see
            :func:`check_dataset_name_unique`.

    Returns:
        The bundle, with ``provenance["constant_covariates"]`` added recording
        any invariant-9 findings. This is a new object, not the input — the
        dataclass is frozen.

    Raises:
        LoaderValidationError: On the first violation, naming site and cause.
    """
    if not bundle.sites:
        raise LoaderValidationError(
            f"dataset {bundle.dataset!r}: no sites. At least one is required."
        )

    check_dataset_name_unique(bundle, strict=strict_dataset_names)
    check_covariates_identical(bundle)
    check_membership_complete(bundle)

    constant: Dict[str, Tuple[str, ...]] = {}
    for table in bundle.sites.values():
        check_required_columns(table)
        check_binary(table, table.treatment, invariant=2)
        check_binary(table, table.outcome, invariant=3)
        check_no_nulls(table)
        check_unit_id_unique(table)
        check_min_rows(table, min_rows)
        check_row_order_preserved(table)
        found = check_covariates_vary(
            table, strict=strict_covariate_variation
        )
        if found:
            constant[table.site_id] = found

    logger.info(
        "validated %s: %d sites, %d rows, covariates %s, %d site(s) with a "
        "constant covariate",
        bundle.dataset, len(bundle.sites), bundle.n_rows, bundle.covariates,
        len(constant),
    )
    return replace(
        bundle,
        provenance={**dict(bundle.provenance), "constant_covariates": constant},
    )
