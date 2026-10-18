"""Characterisation tests pinning the real-data OSRCT reader behaviour.

These tests exist to freeze the observable behaviour of
:func:`experiments.osrct_data_utils.load_prepared_data` before it is replaced
by a dedicated Loader. They deliberately use the real
``osrct_benchmark_v1.0`` extract and the real ``Manylabs1_data.pkl`` rather
than a synthetic frame: the facts pinned here (row counts, the ``resp_gender``
truncation, the site inventory, the id subset relation) are properties of the
real data that no synthetic frame can witness.

Scope is reader-only. Regime construction
(:func:`causal_grounding.train_target_split.create_train_target_split`) is out
of scope and is not exercised here.

:func:`test_resp_gender_astype_int_truncates_imputed_values` pins behaviour
that is wrong rather than merely surprising. It exists so that a tidy-up
changing it fails loudly instead of silently altering covariate values.

Updated 2026-08-09: this module used to pin an index asymmetry between the two
returned frames as a second such wart. ``load_rct_data`` now resets its index,
so both frames carry a ``RangeIndex``. The change was made only after auditing
all 11 call sites and re-running the live pipeline; every concat on that path
already used ``ignore_index=True``, so no result moved. The invariant that
survives, and the one that actually matters, is that the two frames cannot be
aligned positionally at all.

The whole module skips when either the benchmark CSVs or the 23 MB ManyLabs
pickle is absent, so a checkout without the data files stays green.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.osrct_data_utils import OSRCTConfig, load_prepared_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSRCT_ROOT = (
    PROJECT_ROOT / "osrct_benchmark_v1.0" / "confounded_datasets" / "by_study"
)
GAINLOSS_PATH = OSRCT_ROOT / "gainloss" / "age_beta0.5_seed42.csv"
ANCHORING1_PATH = OSRCT_ROOT / "anchoring1" / "age_beta0.5_seed42.csv"
RCT_PATH = PROJECT_ROOT / "ManyLabs1" / "pre-process" / "Manylabs1_data.pkl"

pytestmark = pytest.mark.skipif(
    not GAINLOSS_PATH.is_file() or not RCT_PATH.is_file(),
    reason=(
        "real-data characterisation requires "
        "osrct_benchmark_v1.0/confounded_datasets/by_study and "
        "ManyLabs1/pre-process/Manylabs1_data.pkl"
    ),
)


#: The pinned configuration. Every expectation below is measured against it.
STUDY = "gainloss"
PATTERN = "age"
BETA = 0.5
SEED = 42

EXPECTED_N_OSRCT_ROWS = 3105
EXPECTED_N_RCT_ROWS = 6271

#: The 48 columns both frames share, in file order.
SHARED_SOURCE_COLUMNS = (
    "id",
    "date",
    "site",
    "resp_sex",
    "resp_age",
    "resp_race",
    "resp_ethnicity",
    "resp_citizenship",
    "resp_major",
    "resp_pid",
    "resp_nativelang",
    "resp_religion",
    "resp_american",
    "resp_american_pid",
    "resp_american_ideo",
    "exp_sex",
    "exp_race",
    "study_numparticipants",
    "study_exprunafter",
    "study_separated",
    "study_recruit",
    "study_compensation",
    "study_online",
    "study_country",
    "study_usa",
    "filter_iat",
    "iv",
    "dv",
    "original_study",
    "original_study_name",
    "resp_gender",
    "resp_ethnicity_hisp",
    "RACE",
    "RACE_american_indian",
    "RACE_black_american",
    "RACE_brazil",
    "RACE_east_asian",
    "RACE_more_than_one",
    "RACE_nederland",
    "RACE_others",
    "RACE_south_asian",
    "RACE_white",
    "MAJOR",
    "MAJOR_Engineer",
    "MAJOR_Science",
    "MAJOR_Social",
    "MAJOR_others",
    "resp_polideo",
)

#: Written by the OSRCT sampler, so present in the OSRCT frame only.
OSRCT_ONLY_COLUMNS = ("_selection_prob", "_preferred_treatment")

#: The seven columns the loader appends, in creation order. The first five
#: come from ``add_naturalness_columns``, the last two from
#: ``add_binary_covariates``. Order is part of the contract being pinned.
EXPECTED_APPENDED_COLUMNS = (
    "dv_binary",
    "dv_positive_value",
    "resp_age_cat",
    "resp_gender_bin",
    "resp_polideo_cat",
    "resp_age_bin",
    "resp_polideo_bin",
)

#: Every appended column is int64 except the constant ``dv_positive_value``,
#: which carries the source outcome's positive value and stays float64.
EXPECTED_APPENDED_DTYPES = {
    "dv_binary": "int64",
    "dv_positive_value": "float64",
    "resp_age_cat": "int64",
    "resp_gender_bin": "int64",
    "resp_polideo_cat": "int64",
    "resp_age_bin": "int64",
    "resp_polideo_bin": "int64",
}

EXPECTED_SITES = (
    "abington",
    "brasilia",
    "charles",
    "conncoll",
    "csun",
    "help",
    "ithaca",
    "jmu",
    "ku",
    "laurier",
    "lse",
    "luc",
    "mcdaniel",
    "msvu",
    "mturk",
    "osu",
    "oxy",
    "pi",
    "psu",
    "qccuny",
    "qccuny2",
    "sdsu",
    "swps",
    "swpson",
    "tamu",
    "tamuc",
    "tamuon",
    "tilburg",
    "ufl",
    "unipd",
    "uva",
    "vcu",
    "wisc",
    "wku",
    "wl",
    "wpi",
)

#: Positional offsets of the five OSRCT rows whose ``resp_gender`` is a
#: site-mean imputation rather than a clean 0/1, with the value each carries.
#: 0.6479 appears three times; the other two values appear once each.
EXPECTED_IMPUTED_GENDER_ROWS = {
    203: 0.8187820983125459,
    263: 0.7236180904522613,
    1508: 0.6479149026248942,
    1750: 0.6479149026248942,
    2040: 0.6479149026248942,
}


@pytest.fixture(scope="module")
def gainloss_config() -> OSRCTConfig:
    """The pinned gainloss/age/beta=0.5/seed=42 configuration."""

    return OSRCTConfig(study=STUDY, pattern=PATTERN, beta=BETA, seed=SEED)


@pytest.fixture(scope="module")
def prepared(
    gainloss_config: OSRCTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the real pair once for the module.

    Module-scoped because the ManyLabs pickle is 23 MB and every test below
    needs the same result.
    """

    return load_prepared_data(gainloss_config)


@pytest.fixture(scope="module")
def raw_osrct() -> pd.DataFrame:
    """Read the OSRCT CSV directly, for before/after comparisons."""

    return pd.read_csv(GAINLOSS_PATH)


def test_returns_a_three_tuple_of_two_frames_and_a_metadata_dict(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """The return is ``(osrct_data, rct_data, metadata)`` and nothing else."""

    assert isinstance(prepared, tuple)
    assert len(prepared) == 3

    osrct, rct, metadata = prepared
    assert isinstance(osrct, pd.DataFrame)
    assert isinstance(rct, pd.DataFrame)
    assert isinstance(metadata, dict)


def test_shapes_and_appended_columns_in_order(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Both frames gain the same seven columns, appended at the end."""

    osrct, rct, _ = prepared

    assert osrct.shape == (EXPECTED_N_OSRCT_ROWS, 57)
    assert rct.shape == (EXPECTED_N_RCT_ROWS, 55)

    # Appended at the end, in creation order, on both frames.
    assert tuple(osrct.columns[-7:]) == EXPECTED_APPENDED_COLUMNS
    assert tuple(rct.columns[-7:]) == EXPECTED_APPENDED_COLUMNS

    # Source columns keep their original order ahead of the appended block.
    assert tuple(osrct.columns[:48]) == SHARED_SOURCE_COLUMNS
    assert tuple(osrct.columns[48:50]) == OSRCT_ONLY_COLUMNS
    assert tuple(rct.columns[:48]) == SHARED_SOURCE_COLUMNS


def test_appended_column_dtypes(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Six int64 columns and one float64 constant, on both frames."""

    osrct, rct, _ = prepared

    for frame in (osrct, rct):
        actual = {
            column: str(frame[column].dtype)
            for column in EXPECTED_APPENDED_COLUMNS
        }
        assert actual == EXPECTED_APPENDED_DTYPES

    assert osrct["dv_positive_value"].dtype == np.float64
    assert set(osrct["dv_positive_value"].unique()) == {2.0}


def test_rct_frame_lacks_the_two_sampler_columns(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """``_selection_prob`` and ``_preferred_treatment`` are OSRCT-only."""

    osrct, rct, _ = prepared

    assert set(osrct.columns) - set(rct.columns) == set(OSRCT_ONLY_COLUMNS)
    assert set(rct.columns) - set(osrct.columns) == set()

    for column in OSRCT_ONLY_COLUMNS:
        assert column in osrct.columns
        assert column not in rct.columns


def test_returned_frames_have_asymmetric_indexes(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
    raw_osrct: pd.DataFrame,
) -> None:
    """Both frames now carry a fresh ``RangeIndex``. Updated 2026-08-09.

    This test used to pin an asymmetry: the OSRCT frame got a ``RangeIndex``
    from ``read_csv`` while the RCT frame kept the ManyLabs pickle's own labels,
    starting at 44408 with gaps, because ``load_rct_data`` masked without
    reindexing.

    Its docstring said a tidy-up should fail here loudly rather than change
    downstream alignment silently. That tidy-up has now happened, deliberately,
    and the check it asked for was carried out first: all 11 ``load_rct_data``
    call sites were audited, and the live pipeline was re-run against real
    gainloss data with the change applied. Every concat on that path already
    used ``ignore_index=True``, so no result moved.

    What has **not** changed is the thing that actually matters: the two frames
    still cannot be aligned positionally. Different lengths, different ids at
    the same offsets, because the observational sample is a biased subsample.
    Join on ``id``. That is asserted below and is the real invariant.
    """

    osrct, rct, _ = prepared

    assert isinstance(osrct.index, pd.RangeIndex)
    assert osrct.index.equals(pd.RangeIndex(EXPECTED_N_OSRCT_ROWS))
    assert osrct.index.equals(raw_osrct.index)

    assert isinstance(rct.index, pd.RangeIndex)
    assert rct.index.equals(pd.RangeIndex(EXPECTED_N_RCT_ROWS))

    # The real invariant, unchanged by the reindex: no positional alignment
    # between the two frames. Different lengths, and different ids at the same
    # offsets. Cross-regime work goes through `id`.
    assert len(osrct) != len(rct)
    assert osrct["id"].head(5).tolist() != rct["id"].head(5).tolist()
    assert set(osrct["id"]).issubset(set(rct["id"]))


def test_treatment_column_is_iv_and_passes_through_untouched(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
    raw_osrct: pd.DataFrame,
) -> None:
    """There is no ``treatment`` column; ``iv`` is float64 and unmodified.

    The loader never reads, coerces, renames or validates the treatment. It
    stays exactly as the CSV supplied it, including its float dtype.
    """

    osrct, rct, _ = prepared

    assert "treatment" not in osrct.columns
    assert "treatment" not in rct.columns

    for frame in (osrct, rct):
        assert frame["iv"].dtype == np.float64
        assert set(frame["iv"].unique()) == {0.0, 1.0}
        assert int(frame["iv"].isna().sum()) == 0

    # Byte-for-byte identical to the file, values and order.
    pd.testing.assert_series_equal(
        osrct["iv"],
        raw_osrct["iv"],
        check_names=False,
    )


def test_outcome_binarisation_for_gainloss(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """``dv`` is {1.0, 2.0} here, so the positive value is 2.0, not 1.0.

    The source outcome is not already 0/1. The larger of the two observed
    values is taken as positive, which is why ``positive_value`` is 2.0.
    """

    osrct, rct, metadata = prepared

    assert set(osrct["dv"].unique()) == {1.0, 2.0}
    assert set(rct["dv"].unique()) == {1.0, 2.0}

    assert metadata["source_outcome"] == "dv"
    assert metadata["binary_outcome"] == "dv_binary"
    assert metadata["positive_value"] == 2.0

    for frame in (osrct, rct):
        assert set(frame["dv_binary"].unique()) == {0, 1}
        assert frame["dv_binary"].dtype == np.int64
        # Equality against the positive value, not a threshold.
        assert frame["dv_binary"].equals(frame["dv"].eq(2.0).astype(int))


@pytest.mark.skipif(
    not ANCHORING1_PATH.is_file(),
    reason="requires the anchoring1 benchmark extract",
)
def test_continuous_outcome_study_raises_value_error() -> None:
    """anchoring1 has 278 distinct ``dv`` values, so the loader refuses it.

    This is what limits the loader to the three binary-outcome ManyLabs
    studies; the other twelve study directories cannot be read through it.

    Before the Loader migration this failed inside ``add_naturalness_columns``,
    after a full read of the file, with "binary source outcome". Since
    ``load_prepared_data`` reads through ``OSRCTLoader`` now,
    ``OSRCTLoader.__init__`` rejects the study immediately, before any read —
    earlier and with a message that also names the study and the supported
    set. A deliberate improvement, not an accident; pinned so a future change
    to either message is a conscious decision.
    """

    config = OSRCTConfig(
        study="anchoring1",
        pattern=PATTERN,
        beta=BETA,
        seed=SEED,
    )

    with pytest.raises(ValueError, match="is not loadable: its source outcome is not binary"):
        load_prepared_data(config)


def test_resp_gender_astype_int_truncates_imputed_values(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
    raw_osrct: pd.DataFrame,
) -> None:
    """Imputed ``resp_gender`` values truncate to 0 instead of rounding.

    This is a live silent bug, not a latent one. ManyLabs preprocessing
    replaces missing ``resp_gender`` with a site mean, so five OSRCT rows
    carry fractional values. ``add_binary_covariates`` casts with
    ``.astype(int)``, which truncates: 0.8188 becomes 0, not 1. All five
    values exceed 0.5, so rounding instead of truncating would flip every one
    of them.

    Pinned exactly so that anyone who "fixes" the cast sees this fail and has
    to decide deliberately. The affected rows are precisely the rows whose
    original ``resp_sex`` is missing, which is what
    ``require_observed_gender`` exists to exclude.
    """

    osrct, _, _ = prepared

    imputed = raw_osrct.index[~raw_osrct["resp_gender"].isin([0.0, 1.0])]
    assert list(imputed) == sorted(EXPECTED_IMPUTED_GENDER_ROWS)

    actual_values = {
        position: raw_osrct.at[position, "resp_gender"]
        for position in imputed
    }
    assert actual_values == pytest.approx(EXPECTED_IMPUTED_GENDER_ROWS)

    # Every one truncates to 0, including the three above 0.5.
    assert osrct.loc[imputed, "resp_gender_bin"].tolist() == [0, 0, 0, 0, 0]

    above_half = [
        position
        for position, value in EXPECTED_IMPUTED_GENDER_ROWS.items()
        if value > 0.5
    ]
    assert len(above_half) == 5
    assert osrct.loc[above_half, "resp_gender_bin"].eq(0).all()

    # These are exactly the rows with a missing original gender response.
    assert raw_osrct.loc[imputed, "resp_sex"].isna().all()

    # Elsewhere the cast is lossless.
    clean = raw_osrct.index.difference(imputed)
    pd.testing.assert_series_equal(
        osrct.loc[clean, "resp_gender_bin"],
        raw_osrct.loc[clean, "resp_gender"].astype(int),
        check_names=False,
    )


def test_no_rows_dropped_and_cohort_filter_off_by_default(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
    gainloss_config: OSRCTConfig,
    raw_osrct: pd.DataFrame,
) -> None:
    """With the default config the loader drops nothing."""

    osrct, rct, metadata = prepared

    assert gainloss_config.require_observed_gender is False
    assert metadata["cohort_filter"] == "none"
    assert metadata["require_observed_gender"] is False

    assert len(osrct) == EXPECTED_N_OSRCT_ROWS == len(raw_osrct)
    assert len(rct) == EXPECTED_N_RCT_ROWS

    # No cohort-filter bookkeeping keys are emitted when the filter is off.
    assert not [key for key in metadata if "cohort_filter" in key and key != "cohort_filter"]


def test_site_inventory_is_thirty_six_shared_sites(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Both regimes carry the same 36 string-valued sites."""

    osrct, rct, _ = prepared

    osrct_sites = tuple(sorted(osrct["site"].unique()))
    rct_sites = tuple(sorted(rct["site"].unique()))

    assert osrct_sites == EXPECTED_SITES
    assert rct_sites == EXPECTED_SITES
    assert len(osrct_sites) == 36

    assert {type(value) for value in osrct["site"]} == {str}
    # The default target site is among them.
    assert "lse" in osrct_sites


def test_id_is_unique_and_osrct_ids_are_a_strict_subset_of_rct_ids(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """``id`` is a usable join key; position is not.

    OSRCT is a biased subsample of the RCT, so the two frames can only be
    aligned through ``id``. Pinned because the id relation, not row position,
    is what a replacement Loader must preserve.
    """

    osrct, rct, _ = prepared

    assert osrct["id"].is_unique
    assert rct["id"].is_unique
    assert osrct["id"].nunique() == EXPECTED_N_OSRCT_ROWS
    assert rct["id"].nunique() == EXPECTED_N_RCT_ROWS

    osrct_ids = set(osrct["id"])
    rct_ids = set(rct["id"])
    assert osrct_ids < rct_ids

    # Positional alignment across the two frames is meaningless.
    assert osrct["id"].head(5).tolist() != rct["id"].head(5).tolist()


def test_row_order_is_preserved(
    prepared: tuple[pd.DataFrame, pd.DataFrame, dict],
    raw_osrct: pd.DataFrame,
) -> None:
    """Nothing sorts, merges or reindexes the OSRCT frame.

    Order within a frame is preserved even though order across frames means
    nothing. A Loader that sorted by site or id would break any caller that
    masks a positionally-computed array against the frame.
    """

    osrct, _, _ = prepared

    assert osrct["id"].tolist() == raw_osrct["id"].tolist()
    np.testing.assert_array_equal(
        osrct["dv"].to_numpy(),
        raw_osrct["dv"].to_numpy(),
    )

    for column in ("site", "resp_age", "resp_polideo", "_selection_prob"):
        pd.testing.assert_series_equal(
            osrct[column],
            raw_osrct[column],
            check_names=False,
        )
