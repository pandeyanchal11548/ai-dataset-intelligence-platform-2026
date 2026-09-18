"""
Column Classification Engine
==============================

Classifies every column in a DataFrame into one of four categories:

  * numerical    - continuous or discrete numbers
  * categorical   - a small, fixed set of repeating labels (includes booleans)
  * text          - free-form, high-cardinality strings
  * datetime      - dates / timestamps
"""

from __future__ import annotations

import pandas as pd

from .schema import ClassificationReport, ColumnClassification

CATEGORICAL_MAX_UNIQUE = 50
CATEGORICAL_UNIQUE_RATIO = 0.20


class ColumnClassifier:
    """Classifies each column of a DataFrame as numerical/categorical/text/datetime."""

    def classify(self, df: pd.DataFrame) -> ClassificationReport:
        columns = [self._classify_column(df[col]) for col in df.columns]
        return ClassificationReport(columns=columns)

    def _classify_column(self, series: pd.Series) -> ColumnClassification:
        name = str(series.name)
        non_null = series.dropna()

        if non_null.empty:
            return ColumnClassification(
                name=name, category="text", reason="Column is entirely empty."
            )

        if pd.api.types.is_bool_dtype(non_null):
            return ColumnClassification(
                name=name,
                category="categorical",
                reason="Boolean values form a fixed 2-value category.",
            )

        if pd.api.types.is_datetime64_any_dtype(non_null):
            return ColumnClassification(
                name=name, category="datetime", reason="Already a datetime dtype."
            )

        if pd.api.types.is_numeric_dtype(non_null):
            return ColumnClassification(
                name=name,
                category="numerical",
                reason="Numeric dtype (int/float).",
            )

        # String-typed column: try datetime, then categorical vs. text
        parsed_dates = pd.to_datetime(
            non_null.astype(str).head(200), errors="coerce", format="mixed"
        )
        if parsed_dates.notna().mean() >= 0.90:
            return ColumnClassification(
                name=name,
                category="datetime",
                reason="Over 90% of sampled values parse as dates.",
            )

        # Numeric-looking strings (e.g. "1,200" or "42")
        numeric_attempt = pd.to_numeric(
            non_null.astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        if numeric_attempt.notna().mean() >= 0.95:
            return ColumnClassification(
                name=name,
                category="numerical",
                reason="Over 95% of values parse as numbers.",
            )

        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)

        if unique_count <= CATEGORICAL_MAX_UNIQUE and unique_ratio <= CATEGORICAL_UNIQUE_RATIO:
            return ColumnClassification(
                name=name,
                category="categorical",
                reason=(
                    f"Only {unique_count} distinct values "
                    f"({unique_ratio:.1%} of rows) - a repeating label set."
                ),
            )

        return ColumnClassification(
            name=name,
            category="text",
            reason="High-cardinality free-form string values.",
        )