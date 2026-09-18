"""
Data Quality Scoring System
=============================

Computes a single 0-100 data quality score for a dataset, built from four
weighted dimensions:

  * Completeness  - how much of the data is missing
  * Uniqueness    - how many rows are exact duplicates
  * Validity      - how many rows contain a statistical outlier value
                     (IQR method, via OutlierDetector)
  * Consistency   - how well each column's values match its own dominant
                     type (e.g. a "numeric" column that's 90% numbers and
                     10% stray text is not fully consistent)

Usage
-----
    from app.analysis.quality_scorer import DataQualityScorer

    scorer = DataQualityScorer()
    report = scorer.score(df, source_file="sales.csv")

    print(report.overall_score, report.grade)
    print(report.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

from .outlier_detector import OutlierDetector

# Missing data hurts usability the most, so it's weighted highest by default.
# Override via DataQualityScorer(weights={...}) - must sum to 1.0.
DEFAULT_WEIGHTS = {
    "completeness": 0.35,
    "uniqueness": 0.25,
    "validity": 0.20,
    "consistency": 0.20,
}


@dataclass
class DimensionScore:
    """Score (0-100) and explanation for a single quality dimension."""

    name: str
    score: float
    weight: float
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataQualityReport:
    """Full data quality scoring result for one dataset."""

    source_file: str
    overall_score: float
    grade: str
    row_count: int
    column_count: int
    dimensions: list[DimensionScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "warnings": self.warnings,
        }


class DataQualityScorer:
    """Computes a weighted 0-100 data quality score for a DataFrame."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        outlier_detector: Optional[OutlierDetector] = None,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ValueError("Quality dimension weights must sum to 1.0")
        self.outlier_detector = outlier_detector or OutlierDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self, df: pd.DataFrame, source_file: str = "unknown"
    ) -> DataQualityReport:
        if df.empty:
            raise ValueError("Cannot score an empty DataFrame.")

        completeness, completeness_detail = self._completeness(df)
        uniqueness, uniqueness_detail = self._uniqueness(df)
        validity, validity_detail = self._validity(df)
        consistency, consistency_detail = self._consistency(df)

        dimensions = [
            DimensionScore(
                "completeness", completeness, self.weights["completeness"],
                completeness_detail,
            ),
            DimensionScore(
                "uniqueness", uniqueness, self.weights["uniqueness"],
                uniqueness_detail,
            ),
            DimensionScore(
                "validity", validity, self.weights["validity"], validity_detail,
            ),
            DimensionScore(
                "consistency", consistency, self.weights["consistency"],
                consistency_detail,
            ),
        ]

        overall = round(sum(d.score * d.weight for d in dimensions), 2)

        return DataQualityReport(
            source_file=source_file,
            overall_score=overall,
            grade=self._grade(overall),
            row_count=len(df),
            column_count=df.shape[1],
            dimensions=dimensions,
            warnings=self._build_warnings(dimensions),
        )

    # ------------------------------------------------------------------
    # Dimension calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _completeness(df: pd.DataFrame) -> tuple[float, str]:
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isna().sum().sum())
        ratio = 1 - (missing_cells / total_cells) if total_cells else 1.0
        score = round(ratio * 100, 2)
        detail = f"{missing_cells} missing cell(s) out of {total_cells}."
        return score, detail

    @staticmethod
    def _uniqueness(df: pd.DataFrame) -> tuple[float, str]:
        total_rows = len(df)
        duplicate_rows = int(df.duplicated(keep="first").sum())
        ratio = 1 - (duplicate_rows / total_rows) if total_rows else 1.0
        score = round(ratio * 100, 2)
        detail = f"{duplicate_rows} duplicate row(s) out of {total_rows}."
        return score, detail

    def _validity(self, df: pd.DataFrame) -> tuple[float, str]:
        total_rows = len(df)
        outlier_report = self.outlier_detector.detect(df, method="iqr")
        outlier_rows = outlier_report.total_outlier_rows
        ratio = 1 - (outlier_rows / total_rows) if total_rows else 1.0
        score = round(ratio * 100, 2)
        detail = f"{outlier_rows} row(s) contain an outlier value (IQR method)."
        return score, detail

    @staticmethod
    def _consistency(df: pd.DataFrame) -> tuple[float, str]:
        if df.shape[1] == 0:
            return 100.0, "No columns to check."

        column_ratios = []
        inconsistent_cols = []
        for col in df.columns:
            ratio = DataQualityScorer._column_consistency(df[col])
            column_ratios.append(ratio)
            if ratio < 0.95:
                inconsistent_cols.append(col)

        score = round((sum(column_ratios) / len(column_ratios)) * 100, 2)
        detail = (
            f"Mixed/inconsistent values found in: {inconsistent_cols}"
            if inconsistent_cols
            else "All columns consistently match their dominant type."
        )
        return score, detail

    @staticmethod
    def _column_consistency(series: pd.Series) -> float:
        """
        Fraction of a column's values that match its own dominant type.
        Numeric/datetime columns are checked strictly; everything else
        (categorical/text) is assumed consistent since there's no single
        canonical "shape" to compare free text against.
        """
        non_null = series.dropna()
        if non_null.empty:
            return 1.0

        numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
        if numeric_ratio >= 0.5:
            return float(numeric_ratio)

        parsed = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
        datetime_ratio = parsed.notna().mean()
        if datetime_ratio >= 0.5:
            return float(datetime_ratio)

        return 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    @staticmethod
    def _build_warnings(dimensions: list[DimensionScore]) -> list[str]:
        return [
            f"Low {d.name} score ({d.score}/100): {d.detail}"
            for d in dimensions
            if d.score < 70
        ]