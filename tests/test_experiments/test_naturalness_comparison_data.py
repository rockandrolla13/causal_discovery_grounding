"""Binarisation and regime construction for the Exp28 comparison data.

Covers :mod:`experiments.naturalness_comparison_data`, which declares how each
dataset's columns are binarised and split into the on/idle/reference regimes
every arm reads.
"""

import numpy as np
import pandas as pd
import pytest

from causal_grounding.seeding import crc32_seed
from experiments import naturalness_comparison_data as data


# ---------------------------------------------------------------------------
# Binarisation, seeding and regime construction
# ---------------------------------------------------------------------------


def test_declared_binary_rules_preserve_provenance() -> None:
    """Fixed thresholds and sets produce binary codes plus auditable metadata."""
    rule = data.BinaryRule(
        "at_least",
        35,
        label="advanced maternal age",
        justification="Prespecified clinical threshold.",
    )
    codes, record = data.apply_binary_rule(pd.Series([20, 34, 35, 40, None]), rule)
    assert codes.tolist() == [0, 0, 1, 1, data.MISSING_CODE]
    assert record["label"] == "advanced maternal age"
    assert record["prespecified"]

    in_set = data.BinaryRule("in_set", (12, 1, 2), label="winter birth")
    codes, _ = data.apply_binary_rule(pd.Series([12, 1, 2, 7]), in_set)
    assert codes.tolist() == [1, 1, 1, 0]


def test_passthrough_rejects_nonbinary_values() -> None:
    """A declared native binary cannot silently coerce a third level."""
    data.apply_binary_rule(pd.Series([0, 1, 1, 0]), data.BinaryRule("passthrough"))
    with pytest.raises(ValueError, match="binary"):
        data.apply_binary_rule(pd.Series([0, 1, 2]), data.BinaryRule("passthrough"))


def test_seed_and_selection_are_reproducible() -> None:
    """Stable seeds reproduce regime selection across interpreter processes."""
    assert crc32_seed("twins", 1, 0.5) == crc32_seed("twins", 1, 0.5)
    assert crc32_seed("twins", 1, 0.5) != crc32_seed("twins", 1, 2.0)
    candidate = np.tile([0, 1], 1000)
    treatment = np.tile([0, 1, 1, 0], 500)
    first = data.osrct_select(candidate, treatment, 2.0, np.random.default_rng(19))
    second = data.osrct_select(candidate, treatment, 2.0, np.random.default_rng(19))
    np.testing.assert_array_equal(first, second)


def test_builtin_specs_have_declared_rules() -> None:
    """Every active benchmark column has an explicit binarisation rule."""
    for spec in data.BUILTIN_SPECS.values():
        required = {spec.treatment, spec.outcome, *spec.covariates}
        assert required <= set(spec.rules)


def test_prepare_dataset_uses_only_declared_columns(tmp_path) -> None:
    """Prepared arrays retain declared covariates and complete binary rows."""
    n = 600
    frame = pd.DataFrame(
        {
            "site": np.repeat(["a", "b", "c"], n // 3),
            "arm": np.tile([0, 1], n // 2),
            "outcome": np.tile([0, 0, 1, 1], n // 4),
            "z1": np.tile([0, 1, 1, 0], n // 4),
            "z2": np.tile([1, 0], n // 2),
        }
    )
    path = tmp_path / "toy.csv"
    frame.to_csv(path, index=False)
    rules = {
        column: data.BinaryRule(
            "passthrough", label=column, justification="Native toy binary."
        )
        for column in ("arm", "outcome", "z1", "z2")
    }
    spec = data.DatasetSpec(
        name="toy",
        path=path.name,
        treatment="arm",
        outcome="outcome",
        covariates=("z1", "z2"),
        site_column="site",
        rules=rules,
    )
    prepared = data.prepare_dataset(spec, tmp_path, n_sites=3, min_site_rows=100)
    assert prepared["covariate_names"] == ["z1", "z2"]
    assert prepared["covariates"].shape == (n, 2)
    assert set(np.unique(prepared["sites"])) == {0, 1, 2}


def test_direct_reveal_partitions_preserve_all_rows() -> None:
    """Known potential outcomes split each site into randomized and idle halves."""
    n = 600
    rng = np.random.default_rng(7)
    sites = np.repeat(np.arange(3), n // 3)
    covariates = rng.integers(0, 2, (n, 2), dtype=np.uint8)
    y0 = np.zeros(n, dtype=np.uint8)
    y1 = np.ones(n, dtype=np.uint8)
    partitions, treatment, outcome = data.build_partitions(
        sites,
        np.zeros(n, dtype=np.uint8),
        covariates,
        bias_index=0,
        beta=1.0,
        reference_share=0.4,
        rng=np.random.default_rng(11),
        potential=(y0, y1),
    )
    assert len(partitions) == 3
    used = np.concatenate([np.concatenate((part.rct, part.idle)) for part in partitions])
    assert len(np.unique(used)) == n
    np.testing.assert_array_equal(outcome, treatment)
