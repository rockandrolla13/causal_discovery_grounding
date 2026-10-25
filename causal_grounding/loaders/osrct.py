"""
OSRCT / ManyLabs loader — reads a confounded-dataset CSV and splits it by site.

Supersedes ``experiments.osrct_data_utils.load_prepared_data``. The
acceptance test is
``tests/test_experiments/test_loader_characterisation_osrct.py``.

Two facts about this family that the contract spec does not state:

- **It is three studies, not fifteen.** Outcome binarisation requires exactly
  two distinct source values. gainloss is ``{1.0, 2.0}``, reciprocity and
  allowedforbidden are ``{0.0, 1.0}``, and anchoring1 has 278 distinct values.
  Twelve of the fifteen study directories cannot be loaded at all.
- **All 525 dataset files are CSV.** No parquet, no pickle. The non-CSV
  branches of the old dispatch were unreachable.

This family carries a real row identifier: ``id``, derived from ``session_id``
in ``ManyLabs1/pre-process/ml1_preprocess.py:400``. It means
"same participant" and ``disjoint_validation_ids`` depends on that meaning, so
it is inherited rather than synthesised.
"""

from typing import Tuple, Dict, Any, FrozenSet
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .contract import RawDataset, SiteBundle, SiteTable
from .covariates import (  # noqa: F401  DEFAULT_POLIDEO_BINARY_THRESHOLD re-exported
    DEFAULT_AGE_BINARY_THRESHOLD,
    DEFAULT_POLIDEO_BINARY_THRESHOLD,
    derive_age_bin,
    derive_polideo_bin,
)

logger = logging.getLogger(__name__)


SUPPORTED_STUDIES: FrozenSet[str] = frozenset(
    {"gainloss", "reciprocity", "allowedforbidden"}
)

TREATMENT = "iv"
SOURCE_OUTCOME = "dv"
OUTCOME = "dv_binary"
UNIT_ID = "id"
SITE = "site"

# Single source of truth. Previously written three times — as
# OSRCTConfig.covariates (never read), as USEFUL_COVARIATES in
# experiments/utils.py (now experiments/covariate_preparation.py), and implicitly in the derivation. See AR-DRY-012.
COVARIATES: Tuple[str, ...] = (
    "resp_age_bin",
    "resp_gender_bin",
    "resp_polideo_bin",
)

# Fixed, not configurable, from causal_grounding.discretize.discretize_polideo.
POLIDEO_BIN_EDGES: Tuple[float, ...] = (-0.1, 2.5, 3.5, 6.1)


class OSRCTLoader:
    """Reads one confounded-dataset CSV into a :class:`SiteBundle`.

    Satisfies the :class:`~causal_grounding.loaders.contract.SiteLoader`
    protocol. Reads the observational file only — pairing it with its RCT
    counterpart is regime construction and stays in the experiments.
    """

    def __init__(
        self,
        dataset: RawDataset,
        *,
        study: str,
        age_threshold: float = DEFAULT_AGE_BINARY_THRESHOLD,
        polideo_threshold: float = DEFAULT_POLIDEO_BINARY_THRESHOLD,
        require_observed_gender: bool = False,
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
        self.require_observed_gender = require_observed_gender

    # -- protocol ---------------------------------------------------------

    def read(self) -> SiteBundle:
        """Read, canonicalise, derive covariates, split by site."""
        raw = self._read_csv(self.dataset.path)
        filtered, n_excluded = self._apply_cohort_filter(raw)
        prepared = self._canonicalise(filtered)
        return self._split_by_site(
            prepared, source_rows=len(raw), n_excluded=n_excluded
        )

    # -- stages -----------------------------------------------------------

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read the confounded dataset.

        The old dispatch fell through to ``read_csv`` for *any* unrecognised
        suffix (``experiments.osrct_data_utils.load_prepared_data``). ``RawDataset.format`` is a
        closed ``Literal`` precisely so that an unexpected extension is an error
        rather than a guess.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"OSRCT dataset not found at {path}")
        if self.dataset.format != "csv":
            raise ValueError(
                f"OSRCTLoader reads csv; got format={self.dataset.format!r}. "
                f"All 525 dataset files are csv — a different format means "
                f"something changed and dtype preservation needs checking."
            )
        return pd.read_csv(path)

    def _apply_cohort_filter(self, raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Optionally drop rows whose pre-imputation gender is missing.

        Off by default, matching ``OSRCTConfig.require_observed_gender``. It keys on
        ``resp_sex`` rather than ``resp_gender`` because ManyLabs preprocessing
        overwrites missing ``resp_gender`` with site means, so only ``resp_sex``
        still carries the missingness.

        Warning:
            Filtering here changes row membership, which invariant 11's bundle
            half checks. Positions must be recomputed against the *filtered*
            frame, not the raw one, or every downstream position is off by the
            number of excluded rows.
        """
        if not self.require_observed_gender:
            return raw, 0
        if "resp_sex" not in raw.columns:
            raise ValueError(
                "require_observed_gender needs a 'resp_sex' column; it is the "
                "only one that still carries the missingness, because "
                "preprocessing overwrites missing resp_gender with site means."
            )
        keep = raw["resp_sex"].notna().to_numpy()
        filtered = raw.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        if filtered.empty:
            raise ValueError(
                "the observed-gender cohort filter removed every row"
            )
        return filtered, int(len(raw) - len(filtered))

    def _canonicalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Binarise the outcome and derive the three binary covariates.

        Binarisation pools values, requires exactly two distinct, and takes the
        larger as positive (``experiments.covariate_preparation.add_naturalness_columns``). For gainloss that
        makes ``2.0`` the positive value.

        Warning:
            ``experiments.covariate_preparation.add_naturalness_columns`` casts ``resp_gender`` with
            ``.astype(int)``, which **truncates**. Five gainloss rows carry
            site-mean imputed values (0.6479 x3, 0.7236, 0.8188) and all five
            become 0 — every one of them is above 0.5, so rounding would flip
            all five. This is a live silent bug, pinned by the characterisation
            test. Reproduce it here; do not quietly fix it. Changing it is a
            deliberate decision with a numbers impact, not a tidy-up.

            The other two covariates fail differently: ``pd.to_numeric(...,
            errors="coerce")`` then a threshold compare, so a non-numeric lands
            in bin 0 via ``NaN > t`` being False. Three covariates, two silent
            failure modes.

        Note:
            **``resp_age_cat`` is deliberately not produced here.** The old code
            builds it with ``pd.qcut`` over the *concatenated* OSRCT and RCT
            ages (``experiments.covariate_preparation.add_naturalness_columns``), so it cannot be computed from one file.
            It is not one of the three declared covariates, so nothing in the
            grounder's covariate path needs it.

            All three declared covariates use fixed thresholds rather than
            pooled quantiles, which is what makes single-file loading possible
            at all. If ``resp_age_cat`` is ever promoted to a covariate, the
            scope decision has to be revisited — that would make the Loader a
            two-file operation.

            ``positive_value`` is derived from this frame alone; the old code
            unions it with the RCT's values. For the three loadable studies both
            frames carry both values, so the answer is the same. A study whose
            observational subsample happened to lose a value would raise here
            and pass there.
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

        # Truncates, does not round. See the warning above — deliberate.
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
        self, prepared: pd.DataFrame, *, source_rows: int, n_excluded: int
    ) -> SiteBundle:
        """Partition into per-site tables, recording source positions.

        ``site`` is a pre-existing string column; it is not synthesised. Expect
        36 sites for every ManyLabs study.
        """
        site_values = prepared[SITE].astype(str)
        tables = {}
        for site_id in sorted(site_values.unique()):
            positions = np.flatnonzero((site_values == site_id).to_numpy())
            tables[site_id] = SiteTable(
                site_id=site_id,
                # .iloc, not .loc — this frame has a RangeIndex but its RCT
                # counterpart does not, and the two must be handled the same way.
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
            provenance=self._provenance(source_rows, kept, n_excluded),
        )

    def _provenance(
        self, source_rows: int, kept_rows: int, n_excluded: int
    ) -> Dict[str, Any]:
        """Record what was read and what was dropped, and why."""
        # TODO: add a file hash.
        return {
            "source_path": str(self.dataset.path),
            "format": self.dataset.format,
            "study": self.study,
            "cohort_filter": (
                "observed_gender" if self.require_observed_gender else "none"
            ),
            "n_rows_before": source_rows,
            "n_rows_excluded": n_excluded,
            "n_rows_after": kept_rows,
            "age_threshold": self.age_threshold,
            "polideo_threshold": self.polideo_threshold,
            "positive_value": getattr(self, "_positive_value", None),
            "loader": type(self).__name__,
        }
