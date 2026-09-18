"""
Outlier Detection Engine
=========================

Detects outliers in the numeric columns of a DataFrame using two
configurable, industry-standard methods:

  * IQR method     - flags values outside [Q1 - k*IQR, Q3 + k*IQR]
                      (k defaults to 1.5, the classic "Tukey's fences")
  * Z-score method - flags values whose absolute z-score exceeds a
                      threshold (defaults to 3.0 standard deviations)

Usage
-----
    from app.analysis.outlier_detector import OutlierDetector

    detector = OutlierDetector()
    report = detector.detect(df, method="iqr", source_file="sales.csv")

    print(report.to_dict())
    print(report.total_outlier_rows)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import pandas as pd

OutlierMethod = Literal["iqr", "zscore"]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnOutlierStats:
    """Outlier statistics for a single numeric column."""

    name: str
    method: OutlierMethod
    outlier_count: int
    outlier_percentage: float
    outlier_indices: list[int] = field(default_factory=list)
    lower_bound: Optional[float] = None   # IQR method only
    upper_bound: Optional[float] = None   # IQR method only
    threshold: Optional[float] = None     # Z-score method only

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutlierReport:
    """Full outlier detection result for one dataset."""

    source_file: str
    method: OutlierMethod
    row_count: int
    columns: list[ColumnOutlierStats] = field(default_factory=list)

    @property
    def total_outlier_rows(self) -> int:
        """Count of distinct rows flagged as an outlier in ANY column checked."""
        return len(self.all_outlier_indices)

    @property
    def all_outlier_indices(self) -> list[int]:
        """Union of outlier row indices across all columns checked."""
        combined: set[int] = set()
        for col in self.columns:
            combined.update(col.outlier_indices)
        return sorted(combined)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "method": self.method,
            "row_count": self.row_count,
            "total_outlier_rows": self.total_outlier_rows,
            "columns": [c.to_dict() for c in self.columns],
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OutlierDetector:
    """
    Detects outliers in numeric columns using IQR or Z-score methods.

    Parameters
    ----------
    iqr_multiplier : the "k" in [Q1 - k*IQR, Q3 + k*IQR]. Default 1.5.
    zscore_threshold : absolute z-score above which a value is flagged.
                        Default 3.0 standard deviations.
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> None:
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
        method: OutlierMethod = "iqr",
        columns: Optional[list[str]] = None,
        source_file: str = "unknown",
    ) -> OutlierReport:
        """
        Run outlier detection across the numeric columns of `df`.

        Parameters
        ----------
        df : DataFrame to analyze
        method : "iqr" or "zscore"
        columns : optional subset of columns to check; defaults to every
                  column that is at least 95% numeric-parseable
        source_file : label carried through into the report

        Raises
        ------
        ValueError : if `method` is not "iqr" or "zscore"
        """
        if method not in ("iqr", "zscore"):
            raise ValueError(
                f"Unknown outlier method '{method}'. Use 'iqr' or 'zscore'."
            )

        target_columns = columns if columns is not None else self._numeric_columns(df)

        column_stats: list[ColumnOutlierStats] = []
        for col in target_columns:
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            if series.dropna().empty:
                continue

            if method == "iqr":
                column_stats.append(self._detect_iqr(series, col))
            else:
                column_stats.append(self._detect_zscore(series, col))

        return OutlierReport(
            source_file=source_file,
            method=method,
            row_count=len(df),
            columns=column_stats,
        )

    # ------------------------------------------------------------------
    # IQR method
    # ------------------------------------------------------------------

    def _detect_iqr(self, series: pd.Series, name: str) -> ColumnOutlierStats:
        non_null = series.dropna()
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        mask = (series < lower) | (series > upper)
        indices = series.index[mask.fillna(False)].tolist()

        total = len(non_null)
        return ColumnOutlierStats(
            name=name,
            method="iqr",
            outlier_count=len(indices),
            outlier_percentage=round(len(indices) / total * 100, 2) if total else 0.0,
            outlier_indices=indices,
            lower_bound=round(float(lower), 4),
            upper_bound=round(float(upper), 4),
        )

    # ------------------------------------------------------------------
    # Z-score method
    # ------------------------------------------------------------------

    def _detect_zscore(self, series: pd.Series, name: str) -> ColumnOutlierStats:
        non_null = series.dropna()
        mean = non_null.mean()
        std = non_null.std()

        if not std or pd.isna(std):
            # Zero variance -> nothing can be an outlier by definition.
            return ColumnOutlierStats(
                name=name,
                method="zscore",
                outlier_count=0,
                outlier_percentage=0.0,
                outlier_indices=[],
                threshold=self.zscore_threshold,
            )

        z_scores = (series - mean) / std
        mask = z_scores.abs() > self.zscore_threshold
        indices = series.index[mask.fillna(False)].tolist()

        total = len(non_null)
        return ColumnOutlierStats(
            name=name,
            method="zscore",
            outlier_count=len(indices),
            outlier_percentage=round(len(indices) / total * 100, 2) if total else 0.0,
            outlier_indices=indices,
            threshold=self.zscore_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> list[str]:
        """Columns that are at least 95% numeric-parseable, non-null."""
        numeric_cols = []
        for col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_null = coerced.dropna()
            if non_null.empty:
                continue
            if coerced.notna().mean() >= 0.95:
                numeric_cols.append(col)
        return numeric_cols