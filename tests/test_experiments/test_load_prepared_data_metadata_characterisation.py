"""Characterisation tests pinning ``load_prepared_data``'s metadata contract
and its observed-gender cohort branch.

Companion to :mod:`tests.test_experiments.test_loader_characterisation_osrct`,
which pins the returned *frames*. This module pins the two things that one
leaves open, and both of them are load-bearing for the exp16/exp18 migration
onto the Loader:

1. **The metadata dict.** ``exp18:1015`` serialises it whole into a
   ``loader_metadata`` provenance column. It is therefore an output format, not
   an internal detail, and a migration that changes its shape changes a file on
   disk. The exact key set and the values are pinned here so that any such
   change has to be made deliberately.

2. **The ``require_observed_gender=True`` branch.** The sibling module pins only
   the default, filter-off path. That is *not* the path the live experiments
   run: ``exp18:176`` and ``exp25:158`` both default the flag to ``True``. So
   the branch that actually produces published numbers was, until this module,
   entirely uncharacterised.

The filter-on branch used to carry a wart of its own: ``restrict_to_observed_
gender`` masks with ``.loc[~excluded].copy()`` and never reindexes, so both
frames came back with gapped labels — the same defect class fixed in
``load_rct_data`` on 2026-08-09, whose audit covered the filter-*off* path only.
Closed on 2026-08-10; :func:`test_both_cohort_branches_return_a_contiguous_index`
now pins the agreement between the branches instead.

Values below were measured, not derived. Both branches were run against the real
``gainloss/age_beta0.5_seed42`` extract and the real 23 MB ManyLabs pickle; the
module skips when either is absent, so a checkout without the data files stays
green.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from experiments.osrct_data_utils import OSRCTConfig, load_prepared_data
from experiments.run_manifest import json_default


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSRCT_PATH = (
    PROJECT_ROOT
    / "osrct_benchmark_v1.0"
    / "confounded_datasets"
    / "by_study"
    / "gainloss"
    / "age_beta0.5_seed42.csv"
)
RCT_PATH = PROJECT_ROOT / "ManyLabs1" / "pre-process" / "Manylabs1_data.pkl"

pytestmark = pytest.mark.skipif(
    not OSRCT_PATH.is_file() or not RCT_PATH.is_file(),
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

#: After the observed-gender cohort filter.
EXPECTED_N_OSRCT_ROWS_FILTERED = 3100
EXPECTED_N_RCT_ROWS_FILTERED = 6263

#: Metadata keys with the cohort filter off, in insertion order. Order is
#: pinned as well as membership: the dict is serialised to JSON, so key order
#: is visible in the output file even though it carries no meaning.
EXPECTED_METADATA_KEYS_FILTER_OFF = (
    # add_naturalness_columns
    "source_outcome",
    "binary_outcome",
    "positive_value",
    "age_edges",
    "age_bins",
    # load_prepared_data's own block
    "osrct_path",
    "rct_path",
    "study",
    "pattern",
    "beta",
    "seed",
    "cohort_filter",
    "require_observed_gender",
    # add_binary_covariates
    "age_binary_threshold",
    "polideo_binary_threshold",
    "age_binary_rule",
    "polideo_binary_rule",
    "age_binary_share_one",
    "polideo_binary_share_one",
)

#: The six bookkeeping keys the cohort filter adds, inserted after
#: ``require_observed_gender`` and before the binary-covariate block.
EXPECTED_COHORT_KEYS = (
    "n_osrct_rows_before_cohort_filter",
    "n_osrct_rows_excluded_by_cohort_filter",
    "n_osrct_rows_after_cohort_filter",
    "n_rct_rows_before_cohort_filter",
    "n_rct_rows_excluded_by_cohort_filter",
    "n_rct_rows_after_cohort_filter",
)

#: Values that do not depend on the machine the test runs on. The two path
#: keys are excluded deliberately — see
#: :func:`test_metadata_path_keys_are_absolute_machine_specific_strings`.
EXPECTED_METADATA_VALUES_FILTER_OFF = {
    "source_outcome": "dv",
    "binary_outcome": "dv_binary",
    "positive_value": 2.0,
    "age_edges": [float("-inf"), 19.0, 24.0, float("inf")],
    "age_bins": 3,
    "study": "gainloss",
    "pattern": "age",
    "beta": 0.5,
    "seed": 42,
    "cohort_filter": "none",
    "require_observed_gender": False,
    "age_binary_threshold": 21.0,
    "polideo_binary_threshold": 3.0,
    "age_binary_rule": "resp_age_bin = 1[resp_age > 21]",
    "polideo_binary_rule": "resp_polideo_bin = 1[resp_polideo > 3]",
    "age_binary_share_one": 0.4522184300341297,
    "polideo_binary_share_one": 0.4443259385665529,
}

#: The shares are computed over the pooled frames, so the cohort filter moves
#: them. Both are pinned because they are the only metadata values that are a
#: function of the data rather than of the config.
EXPECTED_SHARES_FILTER_ON = {
    "age_binary_share_one": 0.45188507956851437,
    "polideo_binary_share_one": 0.44408843319448893,
}

#: Index labels missing from each frame after the cohort filter. These are
#: positions in the pre-filter frame, and for OSRCT they are exactly the five
#: rows pinned as imputed-gender rows in the sibling characterisation module.
EXPECTED_MISSING_OSRCT_LABELS = [203, 263, 1508, 1750, 2040]
EXPECTED_MISSING_RCT_LABELS = [421, 557, 3013, 3541, 3869, 4091, 4347, 5823]


@pytest.fixture(scope="module")
def config() -> OSRCTConfig:
    """The pinned gainloss/age/beta=0.5/seed=42 configuration, filter off."""

    return OSRCTConfig(study=STUDY, pattern=PATTERN, beta=BETA, seed=SEED)


@pytest.fixture(scope="module")
def prepared_filter_off(
    config: OSRCTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load once with the cohort filter off. The pickle is 23 MB."""

    return load_prepared_data(config)


@pytest.fixture(scope="module")
def prepared_filter_on(
    config: OSRCTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load once with the cohort filter on — the live exp18/exp25 path."""

    return load_prepared_data(replace(config, require_observed_gender=True))


# -- the metadata contract, filter off ------------------------------------


def test_metadata_key_set_and_order_with_filter_off(
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Nineteen keys, in the order three separate blocks contribute them.

    Pinned because ``exp18:1015`` writes this dict to disk. A Loader migration
    that assembles metadata from bundle provenance instead will not reproduce
    this ordering by accident.
    """

    _, _, metadata = prepared_filter_off

    assert tuple(metadata) == EXPECTED_METADATA_KEYS_FILTER_OFF
    assert len(metadata) == 19


def test_metadata_values_with_filter_off(
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Every machine-independent value, pinned exactly.

    ``age_edges`` is the one to watch. It comes from ``pd.qcut`` over the
    *concatenated* OSRCT and RCT ages, so it cannot be reproduced by a
    single-file loader at all. It is pinned here to make that dependency
    visible rather than implicit.
    """

    _, _, metadata = prepared_filter_off

    actual = {
        key: value
        for key, value in metadata.items()
        if key in EXPECTED_METADATA_VALUES_FILTER_OFF
    }
    assert actual == EXPECTED_METADATA_VALUES_FILTER_OFF

    # The infinite outer edges are real, not a formatting artefact: qcut's
    # outermost bins are widened so that unseen ages cannot fall outside.
    assert metadata["age_edges"][0] == float("-inf")
    assert metadata["age_edges"][-1] == float("inf")
    assert len(metadata["age_edges"]) == metadata["age_bins"] + 1


def test_metadata_path_keys_are_absolute_machine_specific_strings(
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """``osrct_path`` and ``rct_path`` are absolute, so they leak into output.

    They are serialised into exp18's ``loader_metadata`` column, which means
    provenance files already carry the absolute path of whichever machine
    produced them. Pinned as a fact, not endorsed: a Loader migration is a
    reasonable moment to make these repo-relative, but doing so changes the
    output file and should be a deliberate decision.
    """

    _, _, metadata = prepared_filter_off

    osrct_path = Path(metadata["osrct_path"])
    rct_path = Path(metadata["rct_path"])

    assert osrct_path.is_absolute()
    assert rct_path.is_absolute()
    assert osrct_path.name == "age_beta0.5_seed42.csv"
    assert osrct_path.parent.name == "gainloss"
    assert rct_path.name == "Manylabs1_data.pkl"


def test_metadata_survives_exp18s_json_serialisation(
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """The dict round-trips through ``json.dumps(..., default=json_default)``.

    This is exactly what ``exp18:1015`` does. The round-trip is lossy in one
    place and that is pinned: the infinite ``age_edges`` become ``Infinity``
    and ``-Infinity``, which are valid for Python's ``json`` but not for strict
    JSON, so anything reading these files with a strict parser will fail.
    """

    _, _, metadata = prepared_filter_off

    encoded = json.dumps(metadata, default=json_default)
    decoded = json.loads(encoded)

    assert decoded["study"] == "gainloss"
    assert decoded["age_edges"][0] == float("-inf")
    assert decoded["age_edges"][-1] == float("inf")

    # Non-strict JSON, emitted by default. Pinned so a switch to a strict
    # encoder fails here rather than in whatever reads the file months later.
    assert "-Infinity" in encoded


# -- the cohort-filter branch, which the live experiments run --------------


def test_live_experiments_default_the_cohort_filter_on() -> None:
    """exp18 and exp25 default ``require_observed_gender`` to ``True``.

    ``OSRCTConfig`` defaults it to ``False``, so the sibling characterisation
    module — which uses the bare default — pins a branch the live experiments
    never take. This test exists to make that divergence fail loudly if either
    default moves.

    Skips without the Exp16-Exp25 cluster ML stack, which is deliberately not
    in ``requirements.txt`` (see ``requirements-grounding.txt:22``); importing
    either experiment module pulls in ``experiments.benchmarks``, which needs
    it. Same convention as
    ``tests/test_experiments/test_exp18_cb_meta_learner_comparison.py:13-15``.
    """

    pytest.importorskip("torch")
    pytest.importorskip("catenets")
    pytest.importorskip("sklearn_quantile")

    from experiments.exp18_cb_meta_learner_comparison import Exp18Config
    from experiments.exp25_osrct_mc_baseline_comparison import Exp25Config

    assert OSRCTConfig().require_observed_gender is False
    assert Exp18Config().require_observed_gender is True
    assert Exp25Config().require_observed_gender is True


def test_cohort_filter_adds_six_bookkeeping_keys_in_place(
    prepared_filter_on: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Twenty-five keys, with the six counts inserted mid-dict."""

    _, _, metadata = prepared_filter_on

    assert len(metadata) == 25
    assert set(metadata) - set(EXPECTED_METADATA_KEYS_FILTER_OFF) == set(
        EXPECTED_COHORT_KEYS
    )

    # Inserted after the flag they describe, ahead of the binary block.
    keys = tuple(metadata)
    flag_at = keys.index("require_observed_gender")
    assert keys[flag_at + 1 : flag_at + 7] == EXPECTED_COHORT_KEYS
    assert keys[flag_at + 7] == "age_binary_threshold"


def test_cohort_filter_counts_and_rule(
    prepared_filter_on: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Five OSRCT rows and eight RCT rows are excluded.

    The rule name is pinned too. It is written to provenance and read back by
    anything comparing a filtered run against an unfiltered one.
    """

    osrct, rct, metadata = prepared_filter_on

    assert metadata["cohort_filter"] == "resp_sex_not_missing"
    assert metadata["require_observed_gender"] is True

    assert metadata["n_osrct_rows_before_cohort_filter"] == EXPECTED_N_OSRCT_ROWS
    assert metadata["n_osrct_rows_excluded_by_cohort_filter"] == 5
    assert (
        metadata["n_osrct_rows_after_cohort_filter"]
        == EXPECTED_N_OSRCT_ROWS_FILTERED
    )

    assert metadata["n_rct_rows_before_cohort_filter"] == EXPECTED_N_RCT_ROWS
    assert metadata["n_rct_rows_excluded_by_cohort_filter"] == 8
    assert (
        metadata["n_rct_rows_after_cohort_filter"] == EXPECTED_N_RCT_ROWS_FILTERED
    )

    # The counts describe the frames actually returned.
    assert len(osrct) == EXPECTED_N_OSRCT_ROWS_FILTERED
    assert len(rct) == EXPECTED_N_RCT_ROWS_FILTERED


def test_cohort_filter_excludes_exactly_the_missing_resp_sex_rows(
    prepared_filter_on: tuple[pd.DataFrame, pd.DataFrame, dict],
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """The filter keys off ``resp_sex``, not the imputed ``resp_gender``.

    ``resp_gender`` cannot identify these rows: ManyLabs preprocessing replaces
    its missing values with site means, so they arrive as plausible fractions.
    ``resp_sex`` retains its missingness and is what defines the cohort.
    """

    osrct_on, rct_on, _ = prepared_filter_on
    osrct_off, rct_off, _ = prepared_filter_off

    for kept, full in ((osrct_on, osrct_off), (rct_on, rct_off)):
        dropped_ids = set(full["id"]) - set(kept["id"])
        assert full.loc[full["id"].isin(dropped_ids), "resp_sex"].isna().all()
        assert kept["resp_sex"].notna().all()

    # For OSRCT these are the same five rows the sibling module pins as
    # carrying truncated, imputed `resp_gender` values.
    dropped_osrct_ids = set(osrct_off["id"]) - set(osrct_on["id"])
    assert len(dropped_osrct_ids) == 5
    assert (
        sorted(
            osrct_off.index[osrct_off["id"].isin(dropped_osrct_ids)].tolist()
        )
        == EXPECTED_MISSING_OSRCT_LABELS
    )


def test_both_cohort_branches_return_a_contiguous_index(
    prepared_filter_on: tuple[pd.DataFrame, pd.DataFrame, dict],
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Both branches hand back a clean ``RangeIndex``, filtered or not.

    This used to pin the opposite. ``restrict_to_observed_gender`` masks with
    ``.loc[~excluded].copy()`` and never reindexes, and it runs *after*
    ``load_rct_data``'s reset, so the 2026-08-09 index fix never reached the
    one path exp18 and exp25 actually run.

    Closed on 2026-08-10 by resetting at the call site in
    ``load_prepared_data``, not inside ``restrict_to_observed_gender``: the
    benchmark generator calls that function too and feeds the result into
    OSRCT sampling, where a reindex could change which rows were drawn.

    The change moves no results. Three full ``run_one_target`` A/B runs —
    gainloss/lse with and without the GP, allowedforbidden/pi with the GP and
    posterior sampling — returned all 22 output frames identical in values,
    dtypes and index. The ``tabicl_v2`` backend and exp18/exp25 were not
    covered; their stack is cluster-only.
    """

    osrct_off, rct_off, _ = prepared_filter_off
    osrct_on, rct_on, _ = prepared_filter_on

    for frame in (osrct_off, rct_off, osrct_on, rct_on):
        assert isinstance(frame.index, pd.RangeIndex)
        assert frame.index[0] == 0
        assert frame.index[-1] == len(frame) - 1

    # The filter still drops rows; it just no longer leaves holes behind. The
    # dropped rows are pinned by their *pre-filter* positions, which is the
    # only place those labels still exist.
    dropped_osrct_ids = set(osrct_off["id"]) - set(osrct_on["id"])
    dropped_rct_ids = set(rct_off["id"]) - set(rct_on["id"])
    assert (
        sorted(osrct_off.index[osrct_off["id"].isin(dropped_osrct_ids)])
        == EXPECTED_MISSING_OSRCT_LABELS
    )
    assert (
        sorted(rct_off.index[rct_off["id"].isin(dropped_rct_ids)])
        == EXPECTED_MISSING_RCT_LABELS
    )

    assert len(osrct_off) == EXPECTED_N_OSRCT_ROWS
    assert len(osrct_on) == EXPECTED_N_OSRCT_ROWS_FILTERED


def test_pooled_age_edges_are_unchanged_by_the_cohort_filter(
    prepared_filter_on: tuple[pd.DataFrame, pd.DataFrame, dict],
    prepared_filter_off: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """Dropping thirteen rows does not move the age quantile boundaries.

    Worth pinning because the two are computed in a fixed order — filter first,
    then quantiles — so a reordering would be invisible in the frames and
    visible only here.
    """

    _, _, metadata_on = prepared_filter_on
    _, _, metadata_off = prepared_filter_off

    assert metadata_on["age_edges"] == metadata_off["age_edges"]
    assert metadata_on["age_bins"] == metadata_off["age_bins"] == 3


def test_binary_shares_do_move_with_the_cohort_filter(
    prepared_filter_on: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    """The two pooled shares are the metadata values the filter does change."""

    _, _, metadata = prepared_filter_on

    actual = {
        key: metadata[key] for key in EXPECTED_SHARES_FILTER_ON
    }
    assert actual == pytest.approx(EXPECTED_SHARES_FILTER_ON)

    # Different from the unfiltered values, which is the point.
    assert (
        metadata["age_binary_share_one"]
        != EXPECTED_METADATA_VALUES_FILTER_OFF["age_binary_share_one"]
    )
