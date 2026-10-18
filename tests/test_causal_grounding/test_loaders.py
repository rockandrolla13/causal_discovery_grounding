"""
Tests for the Loader contract, registry and invariants.

The readers themselves are scaffolded and raise ``NotImplementedError``; their
behaviour is pinned separately by the characterisation tests in
``tests/test_experiments/test_loader_characterisation_*.py``, which run against
real data.

What is tested here is the part that is implemented: the eleven invariants. Each
gets a test that proves it *fires*, not merely that a clean bundle passes — an
invariant that cannot fail is decoration.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from causal_grounding.loaders import (
    RawDataset,
    SiteTable,
    SiteBundle,
    SiteLoader,
    load_sites,
    LoaderValidationError,
    register_site_loader,
    create_site_loader,
    available_site_loaders,
    reset_loaded_datasets,
)
from causal_grounding.loaders.validation import validate_bundle


COVARIATES: Tuple[str, ...] = ("age_group", "married")


@pytest.fixture(autouse=True)
def _clear_registry_state():
    """Invariant 10 is process-global; stop tests leaking into each other."""
    reset_loaded_datasets()
    yield
    reset_loaded_datasets()


def make_table(
    site_id: str,
    *,
    n: int = 4,
    start: int = 0,
    covariates: Tuple[str, ...] = COVARIATES,
    **overrides,
) -> SiteTable:
    """A minimal valid site table. Overrides let each test break one thing."""
    frame = pd.DataFrame({
        "unit_id": np.arange(start, start + n),
        "iv": [0, 1] * (n // 2),
        "dv_binary": [1, 0] * (n // 2),
        "age_group": ["22–25", "26–29"] * (n // 2),
        "married": [0, 1] * (n // 2),
    })
    frame = frame.assign(**{k: v for k, v in overrides.items() if k in frame})
    return SiteTable(
        site_id=site_id,
        frame=overrides.get("frame", frame),
        treatment="iv",
        outcome="dv_binary",
        covariates=overrides.get("covariates", covariates),
        unit_id="unit_id",
        source_positions=overrides.get(
            "source_positions", tuple(range(start, start + n))
        ),
    )


def make_bundle(*tables: SiteTable, name: str = "toy", **overrides) -> SiteBundle:
    """A bundle over the given tables."""
    return SiteBundle(
        dataset=name,
        sites={t.site_id: t for t in tables},
        covariates=overrides.get("covariates", COVARIATES),
        provenance=overrides.get("provenance", {}),
    )


# =============================================================================
# CONTRACT AND REGISTRY
# =============================================================================

def test_a_clean_bundle_validates() -> None:
    """The baseline. If this fails, every other test below is meaningless."""
    bundle = make_bundle(make_table("A"), make_table("B", start=4))
    validated = validate_bundle(bundle)
    assert validated.site_ids == ("A", "B")
    assert validated.n_rows == 8
    assert validated.provenance["constant_covariates"] == {}


def test_shipped_loaders_are_registered() -> None:
    """Two implementations, three names — ``manylabs`` aliases ``osrct``.

    ``manylabsrct`` is the trial-frame counterpart, a separate single-file
    loader rather than a widening of the OSRCT one. See
    ``reviews/2026_08_09_trial_frame_ideation.md``.
    """
    assert available_site_loaders() == ("manylabs", "manylabsrct", "osrct")


def test_registered_loaders_satisfy_the_protocol() -> None:
    """Structural check — the protocol is runtime_checkable for this reason."""
    dataset = RawDataset(name="osrct_gainloss", path="x.csv", format="csv")
    assert isinstance(
        create_site_loader("osrct", dataset, study="gainloss"), SiteLoader
    )


def test_registering_a_duplicate_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_site_loader("osrct")(object)


def test_unknown_family_names_the_available_ones() -> None:
    dataset = RawDataset(name="nope_x", path="x.csv", format="csv")
    with pytest.raises(KeyError, match="manylabs"):
        load_sites(dataset)


def test_family_is_inferred_from_the_dataset_name() -> None:
    """``osrct_gainloss`` resolves to the ``osrct`` family without a key.

    Reaching the reader's own missing-file error proves dispatch worked — the
    alternative would be a KeyError from the registry.
    """
    dataset = RawDataset(name="osrct_gainloss", path="missing.csv", format="csv")
    with pytest.raises(FileNotFoundError):
        load_sites(dataset, study="gainloss")


def test_osrct_rejects_a_non_csv_format() -> None:
    """Every dataset file is csv. A different format means something changed."""
    dataset = RawDataset(name="osrct_gainloss", path="x.parquet", format="parquet")
    loader = create_site_loader("osrct", dataset, study="gainloss")
    with pytest.raises(FileNotFoundError):
        loader.read()  # missing file is checked before format


def test_osrct_rejects_a_study_whose_outcome_is_not_binary() -> None:
    """anchoring1 has 278 distinct dv values; 12 of 15 studies are unloadable."""
    dataset = RawDataset(name="osrct_anchoring1", path="x.csv", format="csv")
    with pytest.raises(ValueError, match="not loadable"):
        create_site_loader("osrct", dataset, study="anchoring1")


# =============================================================================
# INVARIANTS — each test proves the check fires
# =============================================================================

def test_invariant_1_missing_column_names_site_and_column() -> None:
    frame = make_table("A").frame.drop(columns=["dv_binary"])
    bundle = make_bundle(make_table("A", frame=frame))
    with pytest.raises(LoaderValidationError, match="dv_binary"):
        validate_bundle(bundle)


def test_invariant_2_non_binary_treatment_reports_the_values() -> None:
    """The offending value must appear in the message, whatever numpy reprs it as.

    The original assertion was ``match=r"invariant 2.*\[2\]"``. That passes on
    numpy 1.26 where the list reprs as ``[2]`` and fails on numpy 2.x where it
    reprs as ``[np.int64(2)]``. Caught by the benchmark-stack CI job on its
    first run, which is the numpy-2 environment.
    """
    frame = make_table("A").frame.assign(iv=[0, 1, 2, 1])
    bundle = make_bundle(make_table("A", frame=frame))

    with pytest.raises(LoaderValidationError) as caught:
        validate_bundle(bundle)

    message = str(caught.value)
    assert "invariant 2" in message
    assert "'iv'" in message
    assert "is not binary" in message
    assert "2" in message.split("non-binary values")[1]


def test_invariant_3_non_binary_outcome_reports_the_values() -> None:
    """As above: assert the value is reported, not how numpy spells it."""
    frame = make_table("A").frame.assign(dv_binary=[0, 1, 0, 7])
    bundle = make_bundle(make_table("A", frame=frame))

    with pytest.raises(LoaderValidationError) as caught:
        validate_bundle(bundle)

    message = str(caught.value)
    assert "invariant 3" in message
    assert "'dv_binary'" in message
    assert "7" in message.split("non-binary values")[1]


def test_an_all_nan_treatment_is_caught() -> None:
    """The grounder's own binary check drops NaN first, so this slips past it.

    ``is_binary_series`` calls ``.dropna()`` at naturalness_statistics.py:65,
    which means an all-NaN treatment column passes as binary there. This is one
    of the three invariants the Loader genuinely adds.
    """
    frame = make_table("A").frame.assign(iv=[np.nan] * 4)
    bundle = make_bundle(make_table("A", frame=frame))
    with pytest.raises(LoaderValidationError, match="invariant 2"):
        validate_bundle(bundle)


def test_invariant_5_covariate_order_difference_is_caught() -> None:
    """Same set, different order. The subtle half of the most important check."""
    bundle = make_bundle(
        make_table("A"),
        make_table("B", start=4, covariates=("married", "age_group")),
    )
    with pytest.raises(LoaderValidationError, match="different order"):
        validate_bundle(bundle)


def test_invariant_6_nulls_are_reported_per_column() -> None:
    frame = make_table("A").frame.assign(married=[0, 1, np.nan, 1])
    bundle = make_bundle(make_table("A", frame=frame))
    with pytest.raises(LoaderValidationError, match=r"invariant 6.*married"):
        validate_bundle(bundle)


def test_invariant_7_duplicate_unit_ids_are_counted() -> None:
    frame = make_table("A").frame.assign(unit_id=[0, 0, 1, 1])
    bundle = make_bundle(make_table("A", frame=frame))
    with pytest.raises(LoaderValidationError, match=r"invariant 7, 2 duplicate"):
        validate_bundle(bundle)


def test_invariant_8_raises_rather_than_skipping() -> None:
    """A skipped site perturbs every other site, so skipping is not offered."""
    bundle = make_bundle(make_table("A"), make_table("B", start=4))
    with pytest.raises(LoaderValidationError, match="below min_rows=5"):
        validate_bundle(bundle, min_rows=5)


def test_invariant_8_defaults_to_one_row() -> None:
    """Default 1 reproduces train_target_split.py:96, loudly instead of silently."""
    bundle = make_bundle(make_table("A", n=2))
    assert validate_bundle(bundle).n_rows == 2


def test_invariant_9_records_a_constant_covariate_without_raising() -> None:
    """Not fatal by default — real data violates it.

    ``resp_age_bin`` is constant at 11 of 108 OSRCT site-study combinations,
    because the threshold is 21 and these are university samples. Raising would
    reject data that loads fine today and break the equivalence proof.
    """
    frame = make_table("A").frame.assign(married=[1, 1, 1, 1])
    bundle = make_bundle(make_table("A", frame=frame))
    validated = validate_bundle(bundle)
    assert validated.provenance["constant_covariates"] == {"A": ("married",)}


def test_invariant_9_is_fatal_when_strict() -> None:
    """The opt-in. Making it the default is a behaviour change, not a tidy-up."""
    frame = make_table("A").frame.assign(married=[1, 1, 1, 1])
    bundle = make_bundle(make_table("A", frame=frame))
    with pytest.raises(LoaderValidationError, match=r"invariant 9.*married"):
        validate_bundle(bundle, strict_covariate_variation=True)


def test_invariant_10_allows_reloading_by_default() -> None:
    """Reloading a dataset in one process is normal, not an error.

    Repeated simulation or sensitivity runs can reload one dataset in the same
    process. A load-time uniqueness check would raise on the second load.
    The collision this invariant guards against happens at merge time.
    """
    for _ in range(3):
        validate_bundle(make_bundle(make_table("A"), name="dup"))


def test_invariant_10_is_enforced_when_strict() -> None:
    validate_bundle(make_bundle(make_table("A"), name="dup"),
                    strict_dataset_names=True)
    with pytest.raises(LoaderValidationError, match="already loaded"):
        validate_bundle(make_bundle(make_table("A"), name="dup"),
                        strict_dataset_names=True)


def test_invariant_10_can_be_reset() -> None:
    kw = {"strict_dataset_names": True}
    validate_bundle(make_bundle(make_table("A"), name="dup"), **kw)
    reset_loaded_datasets()
    validate_bundle(make_bundle(make_table("A"), name="dup"), **kw)


def test_invariant_11_reordered_rows_are_caught() -> None:
    """The finding that motivated this invariant.

    A reordered frame has the same shape, columns and values. No other check
    can see it — but ``make_osrct_regimes`` masks a positional array, so every
    site's selection probabilities would change silently.
    """
    bundle = make_bundle(make_table("A", source_positions=(0, 2, 1, 3)))
    with pytest.raises(LoaderValidationError, match="not strictly increasing"):
        validate_bundle(bundle)


def test_invariant_11_a_dropped_row_is_caught() -> None:
    """Global standardisation means one lost row changes every site."""
    bundle = make_bundle(make_table("A"), make_table("B", n=4, start=5))
    with pytest.raises(LoaderValidationError, match="does not cover the parent"):
        validate_bundle(bundle)


def test_invariant_11_a_row_in_two_sites_is_caught() -> None:
    bundle = make_bundle(make_table("A"), make_table("B", n=4, start=0))
    with pytest.raises(LoaderValidationError, match="more than one site"):
        validate_bundle(bundle)


def test_invariant_11_position_count_must_match_row_count() -> None:
    bundle = make_bundle(make_table("A", source_positions=(0, 1, 2)))
    with pytest.raises(LoaderValidationError, match="recorded source position"):
        validate_bundle(bundle)


def test_an_empty_bundle_is_rejected() -> None:
    with pytest.raises(LoaderValidationError, match="no sites"):
        validate_bundle(make_bundle())
