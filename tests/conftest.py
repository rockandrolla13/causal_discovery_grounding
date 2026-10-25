"""Shared fixtures for the golden-snapshot tests."""

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
