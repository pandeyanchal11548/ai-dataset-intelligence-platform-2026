"""
Feature Recommendation Engine
================================

Given a DataFrame, suggests concrete feature-engineering actions:

  * drop_columns       - columns that are safe/likely candidates to drop
                          (near-empty, constant, duplicate-of-another,
                          identifier-like, or high-cardinality free text)
  * encode_columns      - categorical columns, with a suggested encoding
                          strategy (binary / one-hot / ordinal / target)
                          based on cardinality
  * scale_columns       - numeric columns, with a suggested scaler
                          (standard / min-max / robust) based on outlier
                          presence and skew
  * important_features  - numeric columns ranked by relevance to an
                          optional target column: absolute Pearson
                          correlation for a numeric target, or a
                          between-group variance ratio for a categorical
                          target. Without a target column, falls back to
                          a coefficient-of-variation heuristic and the
                          result is explicitly flagged as a rough proxy
                          rather than true feature importance.

Usage
-----
    from app.analysis.feature_recommender import FeatureRecommender

    engine = FeatureRecommender()
    recs = engine.recommend(df, target_column="churned")

    print(recs.to_dict())
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from .column_classifier import ColumnClassifier
from .duplicate_detector import DuplicateDetector
from .missing_value_analyzer import MissingValueAnalyzer
from .outlier_detector import OutlierDetector

HIGH_MISSING_THRESHOLD = 0.5        # >50% missing -> drop candidate
LOW_CARDINALITY_ONE_HOT_MAX = 10    # <=10 categories -> one-hot; else ordinal/target encode
HIGH_CARDINALITY_TEXT_RATIO = 0.9   # >90% unique strings -> likely free text, drop candidate
NOTABLE_OUTLIER_PERCENTAGE = 5.0
NOTABLE_SKEW = 1.0

# Column-name patterns that suggest an identifier even when the dtype is numeric
# (e.g. "customer_id", "order_id", "uuid") - a plain continuous numeric column
# that merely happens to be all-unique in a given sample won't match this.
_ID_NAME_PATTERN = re.compile(r"(^|_)(id|uuid|guid|key)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class DropRecommendation:
    column: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EncodeRecommendation:
    column: str
    strategy: str  # "binary" | "one_hot" | "ordinal" | "target_encode"
    unique_count: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScaleRecommendation:
    column: str
    strategy: str  # "standard" | "min_max" | "robust"
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureImportance:
    column: str
    score: float
    method: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureRecommendations:
    """Full recommendation result for one dataset."""

    source_file: str
    drop_columns: list[DropRecommendation] = field(default_factory=list)
    encode_columns: list[EncodeRecommendation] = field(default_factory=list)
    scale_columns: list[ScaleRecommendation] = field(default_factory=list)
    important_features: list[FeatureImportance] = field(default_factory=list)
    importance_is_heuristic: bool = True

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "drop_columns": [d.to_dict() for d in self.drop_columns],
            "encode_columns": [e.to_dict() for e in self.encode_columns],
            "scale_columns": [s.to_dict() for s in self.scale_columns],
            "important_features": [i.to_dict() for i in self.important_features],
            "importance_is_heuristic": self.importance_is_heuristic,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FeatureRecommender:
    """Suggests drop/encode/scale actions and ranks feature importance."""

    def __init__(
        self,
        classifier: Optional[ColumnClassifier] = None,
        missing_analyzer: Optional[MissingValueAnalyzer] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        outlier_detector: Optional[OutlierDetector] = None,
        high_missing_threshold: float = HIGH_MISSING_THRESHOLD,
    ) -> None:
        self.classifier = classifier or ColumnClassifier()
        self.missing_analyzer = missing_analyzer or MissingValueAnalyzer()
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.outlier_detector = outlier_detector or OutlierDetector()
        self.high_missing_threshold = high_missing_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        source_file: str = "unknown",
    ) -> FeatureRecommendations:
        if df.empty:
            raise ValueError("Cannot generate recommendations for an empty DataFrame.")

        if target_column is not None and target_column not in df.columns:
            raise KeyError(f"target_column '{target_column}' not found in DataFrame.")

        classification = self.classifier.classify(df)
        category_by_col = {c.name: c.category for c in classification.columns}
        missing = self.missing_analyzer.analyze(df)
        missing_by_col = {c.name: c for c in missing.columns}

        drop_columns, dropped_names = self._recommend_drops(
            df, category_by_col, missing_by_col, target_column
        )
        remaining_cols = [c for c in df.columns if c not in dropped_names]

        encode_columns = self._recommend_encodings(
            df, remaining_cols, category_by_col, target_column
        )
        scale_columns = self._recommend_scaling(df, remaining_cols, category_by_col)
        important_features, is_heuristic = self._rank_importance(
            df, remaining_cols, category_by_col, target_column
        )

        return FeatureRecommendations(
            source_file=source_file,
            drop_columns=drop_columns,
            encode_columns=encode_columns,
            scale_columns=scale_columns,
            important_features=important_features,
            importance_is_heuristic=is_heuristic,
        )

    # ------------------------------------------------------------------
    # Drop columns
    # ------------------------------------------------------------------

    def _recommend_drops(
        self,
        df: pd.DataFrame,
        category_by_col: dict[str, str],
        missing_by_col: dict,
        target_column: Optional[str],
    ) -> tuple[list[DropRecommendation], set[str]]:
        recs: list[DropRecommendation] = []
        n_rows = len(df)

        for col in df.columns:
            if col == target_column:
                continue

            missing_pct = missing_by_col[col].null_percentage / 100
            if missing_pct > self.high_missing_threshold:
                recs.append(
                    DropRecommendation(
                        column=col,
                        reason=f"{missing_by_col[col].null_percentage}% missing values.",
                    )
                )
                continue

            unique_count = int(df[col].nunique(dropna=True))
            if unique_count <= 1:
                recs.append(
                    DropRecommendation(column=col, reason="Constant column (no variance).")
                )
                continue

            is_id_like_name = bool(_ID_NAME_PATTERN.search(str(col)))
            if (
                unique_count == n_rows
                and n_rows > 1
                and (category_by_col.get(col) != "numerical" or is_id_like_name)
            ):
                recs.append(
                    DropRecommendation(
                        column=col,
                        reason="All values unique - looks like an identifier column.",
                    )
                )
                continue

            if category_by_col.get(col) == "text":
                unique_ratio = unique_count / n_rows if n_rows else 0
                if unique_ratio > HIGH_CARDINALITY_TEXT_RATIO:
                    recs.append(
                        DropRecommendation(
                            column=col,
                            reason="High-cardinality free text - not directly usable as a feature.",
                        )
                    )
                    continue

        # Exact duplicate columns (identical values to an earlier column)
        seen: dict[tuple, str] = {}
        for col in df.columns:
            if col == target_column:
                continue
            key = tuple(df[col].astype(str).tolist())
            if key in seen:
                recs.append(
                    DropRecommendation(column=col, reason=f"Duplicate of column '{seen[key]}'.")
                )
            else:
                seen[key] = col

        dropped_names = {r.column for r in recs}
        return recs, dropped_names

    # ------------------------------------------------------------------
    # Encode columns
    # ------------------------------------------------------------------

    def _recommend_encodings(
        self,
        df: pd.DataFrame,
        columns: list[str],
        category_by_col: dict[str, str],
        target_column: Optional[str],
    ) -> list[EncodeRecommendation]:
        recs: list[EncodeRecommendation] = []

        for col in columns:
            if col == target_column or category_by_col.get(col) != "categorical":
                continue

            unique_count = int(df[col].nunique(dropna=True))

            if unique_count <= 2:
                strategy = "binary"
                reason = "Only 2 distinct values - map to 0/1."
            elif unique_count <= LOW_CARDINALITY_ONE_HOT_MAX:
                strategy = "one_hot"
                reason = (
                    f"{unique_count} distinct values - low cardinality, "
                    "safe for one-hot encoding."
                )
            else:
                strategy = "target_encode" if target_column else "ordinal"
                reason = f"{unique_count} distinct values - one-hot would add too many columns; "
                reason += (
                    "target/mean encoding recommended."
                    if target_column
                    else "ordinal encoding recommended (no target available for target encoding)."
                )

            recs.append(
                EncodeRecommendation(
                    column=col, strategy=strategy, unique_count=unique_count, reason=reason,
                )
            )

        return recs

    # ------------------------------------------------------------------
    # Scale columns
    # ------------------------------------------------------------------

    def _recommend_scaling(
        self, df: pd.DataFrame, columns: list[str], category_by_col: dict[str, str]
    ) -> list[ScaleRecommendation]:
        recs: list[ScaleRecommendation] = []
        numeric_cols = [c for c in columns if category_by_col.get(c) == "numerical"]

        if not numeric_cols:
            return recs

        outlier_report = self.outlier_detector.detect(df, method="iqr", columns=numeric_cols)
        outlier_pct_by_col = {c.name: c.outlier_percentage for c in outlier_report.columns}

        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            outlier_pct = outlier_pct_by_col.get(col, 0.0)
            skew = float(series.skew()) if len(series) > 2 else 0.0

            if outlier_pct >= NOTABLE_OUTLIER_PERCENTAGE:
                strategy = "robust"
                reason = (
                    f"{outlier_pct}% outlier values (IQR method) - robust scaling "
                    "(median/IQR) is less sensitive to them than standardization."
                )
            elif abs(skew) >= NOTABLE_SKEW:
                strategy = "min_max"
                reason = (
                    f"Skewed distribution (skew = {round(skew, 2)}) - "
                    "min-max scaling preserves shape."
                )
            else:
                strategy = "standard"
                reason = "Roughly symmetric, low-outlier distribution - standard (z-score) scaling."

            recs.append(ScaleRecommendation(column=col, strategy=strategy, reason=reason))

        return recs

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _rank_importance(
        self,
        df: pd.DataFrame,
        columns: list[str],
        category_by_col: dict[str, str],
        target_column: Optional[str],
    ) -> tuple[list[FeatureImportance], bool]:
        numeric_cols = [
            c for c in columns if category_by_col.get(c) == "numerical" and c != target_column
        ]

        if not numeric_cols:
            return [], True

        if target_column is not None:
            target = df[target_column]

            if category_by_col.get(target_column) == "numerical":
                return self._importance_vs_numeric_target(df, numeric_cols, target, target_column)

            return self._importance_vs_categorical_target(
                df, numeric_cols, target, target_column
            )

        return self._importance_heuristic(df, numeric_cols)

    @staticmethod
    def _importance_vs_numeric_target(
        df: pd.DataFrame, numeric_cols: list[str], target: pd.Series, target_column: str
    ) -> tuple[list[FeatureImportance], bool]:
        target_numeric = pd.to_numeric(target, errors="coerce")
        scores = []
        for col in numeric_cols:
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            corr = col_numeric.corr(target_numeric)
            if pd.isna(corr):
                continue
            scores.append(
                FeatureImportance(
                    column=col,
                    score=round(abs(float(corr)), 4),
                    method="abs_pearson_correlation_with_target",
                    reason=f"Correlation with '{target_column}': {round(float(corr), 4)}.",
                )
            )
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores, False

    @staticmethod
    def _importance_vs_categorical_target(
        df: pd.DataFrame, numeric_cols: list[str], target: pd.Series, target_column: str
    ) -> tuple[list[FeatureImportance], bool]:
        scores = []
        for col in numeric_cols:
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            valid = col_numeric.notna() & target.notna()
            if valid.sum() < 2:
                continue

            grouped = col_numeric[valid].groupby(target[valid])
            if grouped.ngroups < 2:
                continue

            overall_var = col_numeric[valid].var()
            if not overall_var:
                continue

            between_group_var = grouped.mean().var()
            ratio = float(between_group_var / overall_var) if pd.notna(between_group_var) else 0.0

            scores.append(
                FeatureImportance(
                    column=col,
                    score=round(ratio, 4),
                    method="between_group_variance_ratio",
                    reason=(
                        f"Group means across '{target_column}' vary "
                        f"{round(ratio, 2)}x relative to overall variance."
                    ),
                )
            )
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores, False

    @staticmethod
    def _importance_heuristic(
        df: pd.DataFrame, numeric_cols: list[str]
    ) -> tuple[list[FeatureImportance], bool]:
        scores = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty or series.mean() == 0:
                continue
            cv = float(series.std() / abs(series.mean()))  # coefficient of variation
            scores.append(
                FeatureImportance(
                    column=col,
                    score=round(cv, 4),
                    method="coefficient_of_variation",
                    reason=(
                        "No target column provided - ranked by relative variance as a rough "
                        "proxy for information content, not true feature importance."
                    ),
                )
            )
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores, True