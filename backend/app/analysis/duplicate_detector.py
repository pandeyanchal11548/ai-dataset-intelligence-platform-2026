"""
Duplicate Detection Module
============================

Detects two kinds of duplication in a DataFrame:

  * Full row duplicates   - every column matches another row exactly
  * Partial duplicates    - a chosen subset of columns matches another row,
                             even if other columns differ

If no subset is given for partial detection, this auto-picks one by
excluding any column that looks like a unique identifier.
"""

from __future__ import annotations

import pandas as pd

from .schema import DuplicateReport


class DuplicateDetector:
    """Detects full-row and partial (subset-of-columns) duplicate rows."""

    def detect(
        self, df: pd.DataFrame, partial_subset: list[str] | None = None
    ) -> DuplicateReport:
        full_mask = df.duplicated(keep="first")
        full_indices = df.index[full_mask].tolist()

        subset = partial_subset or self._auto_select_subset(df)

        if subset:
            partial_mask = df.duplicated(subset=subset, keep="first")
            partial_indices = df.index[partial_mask].tolist()
        else:
            partial_indices = []

        return DuplicateReport(
            full_duplicate_count=len(full_indices),
            full_duplicate_indices=full_indices,
            partial_duplicate_subset=subset,
            partial_duplicate_count=len(partial_indices),
            partial_duplicate_indices=partial_indices,
        )

    @staticmethod
    def _auto_select_subset(df: pd.DataFrame) -> list[str]:
        """Use every column except ones that look like unique identifiers."""
        n_rows = len(df)
        if n_rows == 0:
            return []

        candidate_columns = [
            col for col in df.columns if df[col].nunique(dropna=True) < n_rows
        ]
        return candidate_columns