"""Shared fixtures for the golden-snapshot tests."""

import json
import math
from collections.abc import Callable

import pandas as pd
import pytest

GOLDEN_FLOAT_FORMAT = "%.10g"


@pytest.fixture
def golden_round_trip(tmp_path) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a function that passes a frame through the snapshots' CSV format.

    The frame is written with the snapshots' float format into pytest's
    ``tmp_path`` and read back, so it compares like a frame read from a
    snapshot file. Nothing is written under ``tests/golden``.
    """

    counter = iter(range(1_000_000))

    def round_trip(frame: pd.DataFrame) -> pd.DataFrame:
        path = tmp_path / f"round_trip_{next(counter)}.csv"
        frame.to_csv(path, index=False, float_format=GOLDEN_FLOAT_FORMAT)
        return pd.read_csv(path)

    return round_trip


# Cross-machine tolerance. The Gaussian-process fits land on hyperparameters that
# differ in the last bits between CPUs and BLAS builds, and the tolerance draws
# amplify that: on GitHub's runners (2026-10-25) LP bounds moved by up to 5.2e-5
# against snapshots made locally, while learned hyperparameters agreed to 1e-15.
# 1e-3 leaves about 20x headroom over that noise. A change smaller than 1e-3 in a
# bound would pass unnoticed; logic changes in the pipeline move bounds by more.
GOLDEN_RTOL = 1e-6
GOLDEN_ATOL = 1e-3


def _as_json_value(value):
    """Return ``value`` parsed as JSON when it is a JSON object or array string."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _assert_json_close(actual, expected, where: str) -> None:
    """Assert two parsed JSON values are equal, numbers to the golden tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_json_close(left, right, f"{where}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(actual, expected, rel_tol=GOLDEN_RTOL, abs_tol=GOLDEN_ATOL), (
            f"{where}: {actual} != {expected}"
        )
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def assert_frame_matches_golden(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Assert a round-tripped frame equals its snapshot, to the cross-machine tolerance.

    Numeric columns compare with ``GOLDEN_RTOL`` and ``GOLDEN_ATOL``. Text columns
    that hold JSON (for example learned GP hyperparameters) are parsed and their
    numbers compared the same way; other text compares exactly.

    Args:
        actual: Frame produced by the code under test, after ``golden_round_trip``.
        expected: Frame read from the snapshot CSV.
    """
    json_columns = [
        column for column in expected.columns
        if expected[column].dtype == object
        and expected[column].map(lambda value: _as_json_value(value) is not value).any()
    ]
    pd.testing.assert_frame_equal(
        actual.drop(columns=json_columns), expected.drop(columns=json_columns),
        check_exact=False, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL,
    )
    for column in json_columns:
        for row, (left, right) in enumerate(zip(actual[column], expected[column])):
            _assert_json_close(
                _as_json_value(left), _as_json_value(right), f"{column}[{row}]"
            )
