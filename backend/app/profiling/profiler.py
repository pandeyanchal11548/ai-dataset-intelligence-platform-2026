"""
Dataset Profiling Engine
=========================

Takes an already-loaded pandas DataFrame (e.g. from DatasetLoader) and
computes summary statistics about it:

  * Row count
  * Column count
  * Data types (per column)
  * Missing values (count + percentage, per column)
  * Duplicate rows (count + which row numbers are duplicates)
"""

from __future__ import annotations

import pandas as pd

from .schema import ColumnStats, ProfileReport


class DatasetProfiler:
    """Computes row/column/type/missing/duplicate statistics for a DataFrame."""

    def profile(self, df: pd.DataFrame, source_file: str = "unknown") -> ProfileReport:
        row_count = int(df.shape[0])
        column_count = int(df.shape[1])

        columns = [self._profile_column(df[col]) for col in df.columns]

        duplicate_mask = df.duplicated(keep="first")
        duplicate_indices = df.index[duplicate_mask].tolist()

        return ProfileReport(
            source_file=source_file,
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=len(duplicate_indices),
            duplicate_row_indices=duplicate_indices,
            columns=columns,
        )

    @staticmethod
    def _profile_column(series: pd.Series) -> ColumnStats:
        total = len(series)
        missing_count = int(series.isna().sum())
        missing_percentage = (
            round(missing_count / total * 100, 2) if total else 0.0
        )

        return ColumnStats(
            name=str(series.name),
            dtype=str(series.dtype),
            missing_count=missing_count,
            missing_percentage=missing_percentage,
        )