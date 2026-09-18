"""
Categorical Analysis Engine
=============================

Computes value counts and category distributions for categorical /
low-cardinality columns, plus a bar-chart visualization of the
distribution.

Usage
-----
    from app.analysis.categorical_analyzer import CategoricalAnalyzer

    analyzer = CategoricalAnalyzer()
    summary = analyzer.value_counts(df, "region")
    plot = analyzer.distribution_plot(df, "region")
    all_summaries = analyzer.analyze_all(df)   # auto-detects categorical columns

    print(summary.to_dict())
    print(plot.as_data_uri())
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # headless/server-safe backend - must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

OTHER_LABEL = "Other"


@dataclass
class CategoryCount:
    """Count and share of a single category value."""

    category: str
    count: int
    percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CategoricalSummary:
    """Value-count breakdown for one column."""

    column: str
    total_count: int
    unique_count: int
    missing_count: int
    categories: list[CategoryCount] = field(default_factory=list)

    @property
    def top_category(self) -> Optional[CategoryCount]:
        return self.categories[0] if self.categories else None

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "total_count": self.total_count,
            "unique_count": self.unique_count,
            "missing_count": self.missing_count,
            "categories": [c.to_dict() for c in self.categories],
        }


@dataclass
class CategoricalPlot:
    """A rendered bar-chart image of a column's category distribution."""

    column: str
    image_base64: str  # PNG, base64-encoded (no "data:" prefix)
    saved_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "image_base64": self.image_base64,
            "saved_path": self.saved_path,
        }

    def as_data_uri(self) -> str:
        return f"data:image/png;base64,{self.image_base64}"


class CategoricalAnalyzer:
    """Computes value counts and category distributions for a column."""

    def __init__(self, default_top_n: int = 10) -> None:
        self.default_top_n = default_top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value_counts(
        self,
        df: pd.DataFrame,
        column: str,
        top_n: Optional[int] = None,
    ) -> CategoricalSummary:
        """
        Count occurrences of each category in `column`. Beyond the top
        `top_n` categories, the rest are bucketed together as "Other" so
        the result stays readable for high-cardinality columns.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        series = df[column]
        non_null = series.dropna()
        total = len(series)
        missing = int(series.isna().sum())

        if non_null.empty:
            return CategoricalSummary(
                column=column, total_count=total, unique_count=0,
                missing_count=missing, categories=[],
            )

        n = top_n if top_n is not None else self.default_top_n
        counts = non_null.astype(str).value_counts()

        top = counts.head(n)
        categories = [
            CategoryCount(
                category=str(idx),
                count=int(count),
                percentage=round(count / total * 100, 2) if total else 0.0,
            )
            for idx, count in top.items()
        ]

        remainder = counts.iloc[n:]
        if not remainder.empty:
            other_count = int(remainder.sum())
            categories.append(
                CategoryCount(
                    category=OTHER_LABEL,
                    count=other_count,
                    percentage=round(other_count / total * 100, 2) if total else 0.0,
                )
            )

        return CategoricalSummary(
            column=column,
            total_count=total,
            unique_count=int(non_null.nunique()),
            missing_count=missing,
            categories=categories,
        )

    def distribution_plot(
        self,
        df: pd.DataFrame,
        column: str,
        top_n: Optional[int] = None,
        output_path: Union[str, Path, None] = None,
        figsize: tuple[float, float] = (7, 4.5),
        dpi: int = 110,
    ) -> CategoricalPlot:
        """Render a bar chart of the category distribution for `column`."""
        summary = self.value_counts(df, column, top_n=top_n)

        if not summary.categories:
            raise ValueError(f"Column '{column}' has no non-null values to plot.")

        labels = [c.category for c in summary.categories]
        counts = [c.count for c in summary.categories]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(labels, counts, color="#55A868", edgecolor="white")
        ax.set_title(f"Category Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()

        saved_path = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, format="png")
            saved_path = str(path)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")

        return CategoricalPlot(column=column, image_base64=encoded, saved_path=saved_path)

    def analyze_all(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        max_unique_ratio: float = 0.5,
        max_unique_count: int = 50,
    ) -> list[CategoricalSummary]:
        """Run value_counts across every categorical-looking column, or a given list."""
        target_columns = (
            columns
            if columns is not None
            else self._categorical_columns(df, max_unique_ratio, max_unique_count)
        )
        return [self.value_counts(df, col) for col in target_columns]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorical_columns(
        df: pd.DataFrame, max_unique_ratio: float, max_unique_count: int
    ) -> list[str]:
        categorical_cols = []
        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                continue

            numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if numeric_ratio >= 0.95:
                continue  # numeric column, not categorical

            unique_ratio = non_null.nunique() / len(non_null)
            if non_null.nunique() <= max_unique_count and unique_ratio <= max_unique_ratio:
                categorical_cols.append(col)

        return categorical_cols