"""Data quality analysis: completeness, uniqueness, consistency + issue list."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .common import DATETIME, NUMERIC, coerce_series, iqr_outliers, resolve_kinds

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _grade(score: float) -> str:
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


class QualityAnalyzer:
    """Scores a DataFrame 0-100 and lists concrete, per-column issues.

    Score = 50% completeness + 30% row uniqueness + 20% consistency, where
    consistency is the share of columns free of structural problems
    (constant, mixed types, stray whitespace, unparseable values, inf).
    """

    def __init__(
        self,
        missing_medium_pct: float = 30.0,
        missing_high_pct: float = 50.0,
        outlier_pct_threshold: float = 5.0,
        mixed_type_sample: int = 10_000,
    ) -> None:
        self.missing_medium_pct = missing_medium_pct
        self.missing_high_pct = missing_high_pct
        self.outlier_pct_threshold = outlier_pct_threshold
        self.mixed_type_sample = mixed_type_sample

    def analyze(
        self, df: pd.DataFrame, kinds: Optional[Mapping[str, str]] = None
    ) -> dict[str, Any]:
        kinds = kinds or resolve_kinds(df)
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        missing_cells = int(df.isna().sum().sum())
        completeness = 1 - missing_cells / total_cells if total_cells else 1.0
        dup_count = int(df.duplicated().sum())
        uniqueness = 1 - dup_count / n_rows if n_rows else 1.0

        issues: list[dict[str, Any]] = []
        flagged: set[str] = set()

        def add(severity, column, issue_type, message, count=None, structural=True):
            issues.append(
                {
                    "severity": severity,
                    "column": column,
                    "type": issue_type,
                    "message": message,
                    "count": count,
                }
            )
            if structural and column is not None:
                flagged.add(column)

        if dup_count:
            pct = dup_count / n_rows * 100
            add(
                "high" if pct > 5 else "medium",
                None,
                "duplicate_rows",
                f"{dup_count} duplicate row(s) ({pct:.1f}% of rows).",
                dup_count,
                structural=False,
            )

        for col in df.columns:
            self._column_issues(col, df[col], kinds[col], n_rows, add)

        issues.sort(key=lambda i: (_SEVERITY_RANK[i["severity"]], str(i["column"])))
        consistency = 1 - len(flagged) / n_cols if n_cols else 1.0
        score = round(100 * (0.5 * completeness + 0.3 * uniqueness + 0.2 * consistency), 1)

        counts = {"high": 0, "medium": 0, "low": 0}
        for i in issues:
            counts[i["severity"]] += 1

        return {
            "score": score,
            "grade": _grade(score),
            "dimensions": {
                "completeness": round(completeness * 100, 2),
                "uniqueness": round(uniqueness * 100, 2),
                "consistency": round(consistency * 100, 2),
            },
            "missing_cells": missing_cells,
            "duplicate_rows": dup_count,
            "issue_counts": counts,
            "issues": issues,
        }

    # ------------------------------------------------------------------

    def _column_issues(self, col, s, kind, n_rows, add) -> None:
        non_null = s.dropna()
        miss_pct = (1 - len(non_null) / n_rows) * 100 if n_rows else 0.0
        n_missing = n_rows - len(non_null)

        if len(non_null) == 0:
            add("high", col, "all_missing", "Column is entirely empty.", n_rows)
            return
        if miss_pct >= self.missing_high_pct:
            add("high", col, "high_missing", f"{miss_pct:.1f}% of values are missing.",
                n_missing, structural=False)
        elif miss_pct >= self.missing_medium_pct:
            add("medium", col, "high_missing", f"{miss_pct:.1f}% of values are missing.",
                n_missing, structural=False)
        elif miss_pct >= 5:
            add("low", col, "some_missing", f"{miss_pct:.1f}% of values are missing.",
                n_missing, structural=False)

        if n_rows > 1 and non_null.nunique() <= 1:
            add("medium", col, "constant_column", "Column holds a single constant value.")

        if s.dtype == object or pd.api.types.is_string_dtype(s):
            strs = non_null[non_null.map(lambda v: isinstance(v, str))]
            ws = int((strs != strs.str.strip()).sum()) if len(strs) else 0
            if ws:
                add("low", col, "whitespace",
                    f"{ws} value(s) have leading/trailing whitespace.", ws)
            if non_null.head(self.mixed_type_sample).map(type).nunique() > 1:
                add("medium", col, "mixed_types", "Column mixes Python value types.")

        if kind == NUMERIC:
            coerced = coerce_series(s, NUMERIC)
            bad = int((s.notna() & coerced.isna()).sum())
            if bad:
                add("medium", col, "non_numeric_values",
                    f"{bad} value(s) could not be read as numbers.", bad)
            finite = coerced.dropna()
            inf = int(np.isinf(finite).sum())
            if inf:
                add("medium", col, "infinite_values", f"{inf} infinite value(s).", inf)
            finite = finite[np.isfinite(finite)]
            n_out, _, _ = iqr_outliers(finite)
            if len(finite) and n_out / len(finite) * 100 > self.outlier_pct_threshold:
                add("low", col, "outliers",
                    f"{n_out} outlier(s) ({n_out / len(finite) * 100:.1f}%) by the 1.5xIQR rule.",
                    n_out, structural=False)
        elif kind == DATETIME:
            parsed = coerce_series(s, DATETIME)
            bad = int((s.notna() & parsed.isna()).sum())
            if bad:
                add("medium", col, "unparseable_dates",
                    f"{bad} value(s) could not be parsed as dates.", bad)