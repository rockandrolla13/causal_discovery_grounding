"""
ManyLabs trial-frame loader — the unconfounded counterpart to :mod:`.osrct`.

Supersedes ``causal_grounding/train_target_split.py:109`` ``load_rct_data``,
which has three defects this fixes: a working-directory-relative default path,
returning a slice rather than a copy, and never resetting the index. That last
one is the source of the OSRCT/RCT index asymmetry.

Deliberately a **second single-file loader** rather than a widening of
:class:`~causal_grounding.loaders.osrct.OSRCTLoader`. The contract stays "one
file in, validated sites out"; pairing the observational and trial frames is a
caller concern, and ``exp18:1007`` already models it as ``PreparedStudyData``.
See ``reviews/2026_08_09_trial_frame_ideation.md``.

One thing to know before using it: the trial frame and its observational
counterpart **cannot be aligned positionally**. Different lengths, different ids
at the same offsets, because the observational sample is a biased subsample.
Join on ``id``.
"""

from typing import Tuple, Dict, Any, FrozenSet
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .contract import RawDataset, SiteBundle, SiteTable
from .covariates import (
    DEFAULT_AGE_BINARY_THRESHOLD,
    DEFAULT_POLIDEO_BINARY_THRESHOLD,
    derive_age_bin,
    derive_polideo_bin,
)
from .osrct import (
    COVARIATES,
    OUTCOME,
    SITE,
    SOURCE_OUTCOME,
    SUPPORTED_STUDIES,
    TREATMENT,
    UNIT_ID,
)

logger = logging.getLogger(__name__)

#: Repo-relative default, resolved absolutely. ``load_rct_data`` used a bare
#: relative string, so it only worked when called from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PICKLE = PROJECT_ROOT / "ManyLabs1" / "pre-process" / "Manylabs1_data.pkl"

STUDY_COLUMN = "original_study"


class ManyLabsRCTLoader:
    """Reads one study's randomised rows out of the ManyLabs pickle.

    Satisfies the :class:`~causal_grounding.loaders.contract.SiteLoader`
    protocol. Derives the same outcome and covariates as
    :class:`~causal_grounding.loaders.osrct.OSRCTLoader`, so the two frames are
    directly comparable, and splits by site.

    Does not sample and does not build regimes. Labelling one frame ``on`` and
    the other ``idle`` is ``add_regime_indicator``'s job and stays there.
    """

    def __init__(
        self,
        dataset: RawDataset,
        *,
        study: str,
        age_threshold: float = DEFAULT_AGE_BINARY_THRESHOLD,
        polideo_threshold: float = DEFAULT_POLIDEO_BINARY_THRESHOLD,
    ) -> None:
        if study not in SUPPORTED_STUDIES:
            raise ValueError(
                f"study {study!r} is not loadable: its source outcome is not "
                f"binary. Supported: {sorted(SUPPORTED_STUDIES)}"
            )
        self.dataset = dataset
        self.study = study
        self.age_threshold = age_threshold
        self.polideo_threshold = polideo_threshold

    # -- protocol ---------------------------------------------------------

    def read(self) -> SiteBundle:
        """Read, filter to the study, canonicalise, split by site."""
        table = self._read_pickle(self.dataset.path)
        study_rows, n_before = self._filter_to_study(table)
        prepared = self._canonicalise(study_rows)
        return self._split_by_site(prepared, source_rows=n_before)

    # -- stages -----------------------------------------------------------

    def _read_pickle(self, path: Path) -> pd.DataFrame:
        """Read the 23MB preprocessed ManyLabs table."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ManyLabs pickle not found at {path}")
        return pd.read_pickle(path)

    def _filter_to_study(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Take one study's rows, as a copy, with the index reset.

        Three deliberate differences from ``load_rct_data``:

        - **Raises on an unknown study.** The old function returned an empty
          frame and nothing downstream checked, so a typo was silent.
        - **Returns a copy**, not a slice.
        - **Resets the index.** The old labels came from the study's offset
          within an 89,979-row table, and served no purpose — every consumer
          either concatenates with ``ignore_index=True`` or masks by boolean.
        """
        if STUDY_COLUMN not in table.columns:
            raise ValueError(
                f"pickle has no {STUDY_COLUMN!r} column; cannot filter to "
                f"{self.study!r}"
            )
        mask = (table[STUDY_COLUMN] == self.study).to_numpy()
        n_matched = int(mask.sum())
        if n_matched == 0:
            available = sorted(table[STUDY_COLUMN].dropna().unique())
            raise ValueError(
                f"study {self.study!r} has no rows in {self.dataset.path}. "
                f"Available: {available}"
            )
        rows = table.iloc[np.flatnonzero(mask)].copy().reset_index(drop=True)
        return rows, len(table)

    def _canonicalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Binarise the outcome and derive the three binary covariates.

        Mirrors :meth:`OSRCTLoader._canonicalise` exactly, including the
        ``resp_gender.astype(int)`` truncation. Reproducing that wart here is
        deliberate: the two frames are compared and combined downstream, so they
        must be derived the same way. Fixing it is a joint decision about both
        loaders, not a tidy-up in one of them.
        """
        result = frame.copy()

        observed = sorted(result[SOURCE_OUTCOME].dropna().unique().tolist())
        if len(observed) != 2:
            raise ValueError(
                f"study {self.study!r} requires a binary source outcome; "
                f"got {observed}"
            )
        positive_value = observed[-1]

        result[OUTCOME] = (result[SOURCE_OUTCOME] == positive_value).astype(int)
        result["dv_positive_value"] = positive_value

        # Truncates rather than rounds. Same as the OSRCT side, on purpose.
        result["resp_gender_bin"] = result["resp_gender"].astype(int)

        result["resp_age_bin"] = derive_age_bin(
            result["resp_age"], self.age_threshold
        )
        result["resp_polideo_bin"] = derive_polideo_bin(
            result["resp_polideo"], self.polideo_threshold
        )

        self._positive_value = float(positive_value)
        return result

    def _split_by_site(
        self, prepared: pd.DataFrame, *, source_rows: int
    ) -> SiteBundle:
        """Partition into per-site tables, recording source positions."""
        site_values = prepared[SITE].astype(str)
        tables = {}
        for site_id in sorted(site_values.unique()):
            positions = np.flatnonzero((site_values == site_id).to_numpy())
            tables[site_id] = SiteTable(
                site_id=site_id,
                frame=prepared.iloc[positions].copy(),
                treatment=TREATMENT,
                outcome=OUTCOME,
                covariates=COVARIATES,
                unit_id=UNIT_ID,
                source_positions=tuple(int(p) for p in positions),
            )

        kept = sum(len(table) for table in tables.values())
        return SiteBundle(
            dataset=self.dataset.name,
            sites=tables,
            covariates=COVARIATES,
            provenance=self._provenance(source_rows, kept),
        )

    def _provenance(self, source_rows: int, kept_rows: int) -> Dict[str, Any]:
        """Record what was read and what the study filter removed."""
        return {
            "source_path": str(self.dataset.path),
            "format": self.dataset.format,
            "study": self.study,
            "regime": "randomised",
            "n_rows_before": source_rows,
            "n_rows_excluded": source_rows - kept_rows,
            "n_rows_after": kept_rows,
            "age_threshold": self.age_threshold,
            "polideo_threshold": self.polideo_threshold,
            "positive_value": getattr(self, "_positive_value", None),
            "loader": type(self).__name__,
        }
