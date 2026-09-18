"""
Missing Value Analyzer
========================

For every column, computes:

  * Null count
  * Null percentage
  * Column-wise missing severity, bucketed as:
        none      ->   0%
        low       ->   0%  < x <=  5%
        moderate  ->   5%  < x <= 20%
        high      ->  20%  < x <= 50%
        critical  ->  50%  < x <= 100%
"""

from __future__ import annotations

import pandas as pd

from .schema import ColumnMissingStats, MissingValueReport, Severity


class MissingValueAnalyzer:
    """Analyzes null counts, percentages, and severity per column."""

    def analyze(self, df: pd.DataFrame) -> MissingValueReport:
        columns = [self._analyze_column(df[col]) for col in df.columns]
        return MissingValueReport(columns=columns)

    def _analyze_column(self, series: pd.Series) -> ColumnMissingStats:
        total = len(series)
        null_count = int(series.isna().sum())
        null_percentage = round(null_count / total * 100, 2) if total else 0.0

        return ColumnMissingStats(
            name=str(series.name),
            null_count=null_count,
            null_percentage=null_percentage,
            severity=self._severity(null_percentage),
        )

    @staticmethod
    def _severity(null_percentage: float) -> Severity:
        if null_percentage <= 0:
            return "none"
        if null_percentage <= 5:
            return "low"
        if null_percentage <= 20:
            return "moderate"
        if null_percentage <= 50:
            return "high"
        return "critical"