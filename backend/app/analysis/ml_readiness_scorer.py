"""
ML Readiness Score Generator
===============================

Produces a single 0-100 "ready for machine learning" score for a
dataset, built from three weighted dimensions:

  * completeness      - how much of the data is missing
  * feature_quality    - the proportion of columns that are usable as
                          features (not constant/duplicate/ID-like/
                          high-missing, via FeatureRecommender) plus
                          whether there are enough rows per feature for
                          stable model fitting
  * noise_level        - outlier prevalence, duplicate rows, and
                          inconsistent per-column value types (all
                          things that add noise a model has to fight
                          through)

Usage
-----
    from app.analysis.ml_readiness_scorer import MLReadinessScorer

    scorer = MLReadinessScorer()
    report = scorer.score(df, source_file="sales.csv")

    print(report.overall_score, report.readiness_level)
    print(report.recommendations)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from .duplicate_detector import DuplicateDetector
from .feature_recommender import FeatureRecommender
from .outlier_detector import OutlierDetector
from .quality_scorer import DataQualityScorer

DEFAULT_WEIGHTS = {
    "completeness": 0.40,
    "feature_quality": 0.35,
    "noise_level": 0.25,
}

# Rule-of-thumb minimum rows per feature for reasonably stable model fitting.
ROWS_PER_FEATURE_TARGET = 10


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ReadinessDimension:
    """Score (0-100) and explanation for a single readiness dimension."""

    name: str
    score: float
    weight: float
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MLReadinessReport:
    """Full ML-readiness scoring result for one dataset."""

    source_file: str
    overall_score: float
    readiness_level: str
    row_count: int
    column_count: int
    dimensions: list[ReadinessDimension] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "overall_score": self.overall_score,
            "readiness_level": self.readiness_level,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class MLReadinessScorer:
    """Computes a weighted 0-100 ML-readiness score for a DataFrame."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        outlier_detector: Optional[OutlierDetector] = None,
        feature_recommender: Optional[FeatureRecommender] = None,
        rows_per_feature_target: int = ROWS_PER_FEATURE_TARGET,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ValueError("Readiness dimension weights must sum to 1.0")

        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.outlier_detector = outlier_detector or OutlierDetector()
        self.feature_recommender = feature_recommender or FeatureRecommender()
        self.rows_per_feature_target = rows_per_feature_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        source_file: str = "unknown",
    ) -> MLReadinessReport:
        if df.empty:
            raise ValueError("Cannot score an empty DataFrame.")

        completeness, completeness_detail = self._completeness(df)
        feature_quality, feature_quality_detail, drop_candidates = self._feature_quality(
            df, target_column
        )
        noise_level, noise_detail, noise_facts = self._noise_level(df)

        dimensions = [
            ReadinessDimension(
                "completeness", completeness, self.weights["completeness"], completeness_detail,
            ),
            ReadinessDimension(
                "feature_quality", feature_quality, self.weights["feature_quality"],
                feature_quality_detail,
            ),
            ReadinessDimension(
                "noise_level", noise_level, self.weights["noise_level"], noise_detail,
            ),
        ]

        overall = round(sum(d.score * d.weight for d in dimensions), 2)

        return MLReadinessReport(
            source_file=source_file,
            overall_score=overall,
            readiness_level=self._readiness_level(overall),
            row_count=int(df.shape[0]),
            column_count=int(df.shape[1]),
            dimensions=dimensions,
            recommendations=self._build_recommendations(
                df, dimensions, drop_candidates, noise_facts, target_column
            ),
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

    def _feature_quality(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> tuple[float, str, list[str]]:
        recs = self.feature_recommender.recommend(df, target_column=target_column)
        drop_candidates = [d.column for d in recs.drop_columns]

        feature_cols = [c for c in df.columns if c != target_column]
        n_features = len(feature_cols) or 1
        drop_ratio = len(drop_candidates) / n_features
        usable_ratio = 1 - drop_ratio

        n_rows = len(df)
        target_rows = self.rows_per_feature_target * n_features
        sample_adequacy = min(1.0, n_rows / target_rows) if target_rows else 1.0

        score = round((0.7 * usable_ratio + 0.3 * sample_adequacy) * 100, 2)

        rows_per_feature = round(n_rows / n_features, 1) if n_features else 0
        detail = (
            f"{len(drop_candidates)} of {n_features} feature column(s) flagged as low-value "
            f"(constant/duplicate/ID-like/high-missing); {n_rows} rows for {n_features} "
            f"feature(s) ({rows_per_feature} rows/feature)."
        )
        return score, detail, drop_candidates

    def _noise_level(self, df: pd.DataFrame) -> tuple[float, str, dict]:
        n_rows = len(df)

        duplicates = self.duplicate_detector.detect(df)
        duplicate_ratio = duplicates.full_duplicate_count / n_rows if n_rows else 0.0

        outlier_report = self.outlier_detector.detect(df, method="iqr")
        outlier_ratio = outlier_report.total_outlier_rows / n_rows if n_rows else 0.0

        # Reuses DataQualityScorer's per-column type-consistency check so
        # "noise" also captures columns that mix types (e.g. numbers stored
        # as text) rather than only row-level issues.
        consistency_ratios = [
            DataQualityScorer._column_consistency(df[col]) for col in df.columns
        ]
        avg_consistency = (
            sum(consistency_ratios) / len(consistency_ratios) if consistency_ratios else 1.0
        )

        cleanliness = (
            0.4 * (1 - duplicate_ratio) + 0.4 * (1 - outlier_ratio) + 0.2 * avg_consistency
        )
        score = round(max(0.0, cleanliness) * 100, 2)

        detail = (
            f"{duplicates.full_duplicate_count} duplicate row(s) "
            f"({round(duplicate_ratio * 100, 1)}%); "
            f"{outlier_report.total_outlier_rows} row(s) with an outlier value "
            f"({round(outlier_ratio * 100, 1)}%); "
            f"average column type-consistency {round(avg_consistency * 100, 1)}%."
        )

        facts = {
            "duplicate_count": duplicates.full_duplicate_count,
            "outlier_row_count": outlier_report.total_outlier_rows,
            "avg_consistency": avg_consistency,
        }
        return score, detail, facts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _readiness_level(score: float) -> str:
        if score >= 85:
            return "Ready for ML"
        if score >= 70:
            return "Needs Minor Cleanup"
        if score >= 50:
            return "Needs Significant Cleanup"
        return "Not Ready"

    def _build_recommendations(
        self,
        df: pd.DataFrame,
        dimensions: list[ReadinessDimension],
        drop_candidates: list[str],
        noise_facts: dict,
        target_column: Optional[str],
    ) -> list[str]:
        recs: list[str] = []

        completeness = next(d for d in dimensions if d.name == "completeness")
        if completeness.score < 90:
            recs.append(
                "Impute or drop columns with heavy missing data before training - "
                f"{completeness.detail}"
            )

        if drop_candidates:
            recs.append(
                f"Remove {len(drop_candidates)} low-value column(s) before training: "
                f"{drop_candidates}."
            )

        n_features = len([c for c in df.columns if c != target_column]) or 1
        if len(df) < self.rows_per_feature_target * n_features:
            recs.append(
                f"Low rows-per-feature ratio ({len(df)} rows, {n_features} features) - "
                "consider gathering more data or reducing dimensionality."
            )

        if noise_facts["duplicate_count"] > 0:
            recs.append(
                f"Deduplicate the {noise_facts['duplicate_count']} exact duplicate row(s) "
                "before training."
            )

        if noise_facts["outlier_row_count"] > 0:
            recs.append(
                f"{noise_facts['outlier_row_count']} row(s) contain outlier values - "
                "consider robust scaling, winsorizing, or reviewing them for data entry errors."
            )

        if noise_facts["avg_consistency"] < 0.95:
            recs.append(
                "Some columns mix types inconsistently (e.g. numbers stored as text) - "
                "clean these before encoding/scaling."
            )

        if not recs:
            recs.append("Dataset looks ready for ML - no major issues detected.")

        return recs