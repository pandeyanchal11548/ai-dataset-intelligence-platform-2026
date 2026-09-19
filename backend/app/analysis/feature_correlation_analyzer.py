"""
Feature Correlation Analyzer
===============================

Builds on top of CorrelationAnalyzer to specifically flag:

  * Highly correlated feature pairs - numeric columns whose correlation
    magnitude exceeds a "highly correlated" threshold (default 0.8).
    Useful context even when nothing needs to be dropped (e.g. for
    multicollinearity awareness in linear models).

  * Redundant columns                - columns that are exact duplicates
    of another column (any dtype), or numeric columns so strongly
    correlated with another (default >= 0.95) that they carry
    essentially the same information. For each redundant pair, one
    column is recommended to keep and the other to drop.

Usage
-----
    from app.analysis.feature_correlation_analyzer import FeatureCorrelationAnalyzer

    analyzer = FeatureCorrelationAnalyzer()
    report = analyzer.analyze(df, source_file="sales.csv")

    print(report.recommended_drops)
    print(report.to_dict())
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from .correlation_analyzer import CorrelationAnalyzer, CorrelationMethod

HIGH_CORRELATION_THRESHOLD = 0.8
REDUNDANT_CORRELATION_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class HighCorrelationPair:
    """A numeric column pair whose correlation exceeds the "high" threshold."""

    column_a: str
    column_b: str
    correlation: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RedundancyFinding:
    """One column recommended to drop because another column covers the same signal."""

    kept_column: str
    dropped_column: str
    reason: str  # "exact_duplicate" | "near_duplicate_correlation"
    correlation: Optional[float] = None  # set for near_duplicate_correlation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureCorrelationReport:
    """Full feature-correlation result for one dataset."""

    source_file: str
    method: CorrelationMethod
    high_correlation_threshold: float
    redundant_correlation_threshold: float
    highly_correlated_pairs: list[HighCorrelationPair] = field(default_factory=list)
    redundant_columns: list[RedundancyFinding] = field(default_factory=list)

    @property
    def recommended_drops(self) -> list[str]:
        return sorted({f.dropped_column for f in self.redundant_columns})

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "method": self.method,
            "high_correlation_threshold": self.high_correlation_threshold,
            "redundant_correlation_threshold": self.redundant_correlation_threshold,
            "highly_correlated_pairs": [p.to_dict() for p in self.highly_correlated_pairs],
            "redundant_columns": [r.to_dict() for r in self.redundant_columns],
            "recommended_drops": self.recommended_drops,
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class FeatureCorrelationAnalyzer:
    """Flags highly correlated numeric feature pairs and redundant columns."""

    def __init__(
        self,
        high_threshold: float = HIGH_CORRELATION_THRESHOLD,
        redundant_threshold: float = REDUNDANT_CORRELATION_THRESHOLD,
        correlation_analyzer: Optional[CorrelationAnalyzer] = None,
    ) -> None:
        if not 0 < high_threshold <= 1:
            raise ValueError("high_threshold must be in (0, 1].")
        if not 0 < redundant_threshold <= 1:
            raise ValueError("redundant_threshold must be in (0, 1].")
        if redundant_threshold < high_threshold:
            raise ValueError("redundant_threshold must be >= high_threshold.")

        self.high_threshold = high_threshold
        self.redundant_threshold = redundant_threshold
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        method: CorrelationMethod = "pearson",
        columns: Optional[list[str]] = None,
        source_file: str = "unknown",
    ) -> FeatureCorrelationReport:
        if df.shape[1] < 2:
            raise ValueError("Need at least 2 columns to analyze feature correlation.")

        exact_duplicates, dropped = self._exact_duplicate_columns(df, columns)

        highly_correlated_pairs: list[HighCorrelationPair] = []
        near_duplicates: list[RedundancyFinding] = []

        try:
            corr_report = self.correlation_analyzer.analyze(
                df, method=method, columns=columns, source_file=source_file
            )
        except ValueError:
            # Fewer than 2 numeric columns - correlation-based checks simply
            # find nothing; exact-duplicate findings (any dtype) still apply.
            corr_report = None

        if corr_report is not None:
            corr_columns = corr_report.columns
            for i in range(len(corr_columns)):
                for j in range(i + 1, len(corr_columns)):
                    col_a, col_b = corr_columns[i], corr_columns[j]
                    corr_value = corr_report.matrix[col_a][col_b]
                    magnitude = abs(corr_value)

                    if magnitude >= self.high_threshold:
                        highly_correlated_pairs.append(
                            HighCorrelationPair(
                                column_a=col_a, column_b=col_b, correlation=corr_value
                            )
                        )

                    if magnitude >= self.redundant_threshold and col_b not in dropped:
                        near_duplicates.append(
                            RedundancyFinding(
                                kept_column=col_a,
                                dropped_column=col_b,
                                reason="near_duplicate_correlation",
                                correlation=corr_value,
                            )
                        )
                        dropped.add(col_b)

        highly_correlated_pairs.sort(key=lambda p: abs(p.correlation), reverse=True)

        return FeatureCorrelationReport(
            source_file=source_file,
            method=method,
            high_correlation_threshold=self.high_threshold,
            redundant_correlation_threshold=self.redundant_threshold,
            highly_correlated_pairs=highly_correlated_pairs,
            redundant_columns=exact_duplicates + near_duplicates,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_duplicate_columns(
        df: pd.DataFrame, columns: Optional[list[str]]
    ) -> tuple[list[RedundancyFinding], set[str]]:
        """Columns whose values (any dtype) exactly match an earlier column."""
        target_columns = columns if columns is not None else list(df.columns)
        findings: list[RedundancyFinding] = []
        seen: dict[tuple, str] = {}
        dropped: set[str] = set()

        for col in target_columns:
            if col not in df.columns:
                continue
            key = tuple(df[col].astype(str).tolist())
            if key in seen:
                findings.append(
                    RedundancyFinding(
                        kept_column=seen[key], dropped_column=col, reason="exact_duplicate",
                    )
                )
                dropped.add(col)
            else:
                seen[key] = col

        return findings, dropped