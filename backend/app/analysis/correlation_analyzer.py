"""
Correlation Analyzer
======================

Computes a correlation matrix across a dataset's numeric columns, flags
notably correlated pairs, and renders the matrix as an annotated heatmap
image.

Usage
-----
    from app.analysis.correlation_analyzer import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    report = analyzer.analyze(df, method="pearson")
    heatmap = analyzer.heatmap(df, method="pearson")

    print(report.to_dict())
    print(heatmap.as_data_uri())
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib
matplotlib.use("Agg")  # headless/server-safe backend - must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

CorrelationMethod = Literal["pearson", "spearman", "kendall"]


@dataclass
class CorrelationPair:
    """A single pair of columns and how strongly they correlate."""

    column_a: str
    column_b: str
    correlation: float
    strength: str  # "moderate" | "strong"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorrelationReport:
    """Full correlation matrix result for one dataset."""

    source_file: str
    method: CorrelationMethod
    columns: list[str]
    matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    notable_pairs: list[CorrelationPair] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "method": self.method,
            "columns": self.columns,
            "matrix": self.matrix,
            "notable_pairs": [p.to_dict() for p in self.notable_pairs],
        }


@dataclass
class CorrelationHeatmap:
    """A rendered heatmap image of the correlation matrix."""

    method: CorrelationMethod
    columns: list[str]
    image_base64: str  # PNG, base64-encoded (no "data:" prefix)
    saved_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "columns": self.columns,
            "image_base64": self.image_base64,
            "saved_path": self.saved_path,
        }

    def as_data_uri(self) -> str:
        return f"data:image/png;base64,{self.image_base64}"


class CorrelationAnalyzer:
    """Computes correlation matrices and renders heatmaps for numeric columns."""

    def __init__(
        self,
        strong_threshold: float = 0.7,
        moderate_threshold: float = 0.4,
    ) -> None:
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        method: CorrelationMethod = "pearson",
        columns: Optional[list[str]] = None,
        source_file: str = "unknown",
    ) -> CorrelationReport:
        """Compute the correlation matrix and flag notably-correlated pairs."""
        numeric_df = self._numeric_frame(df, columns)

        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns to compute correlations.")

        corr = numeric_df.corr(method=method)
        matrix = {
            row: {col: round(float(corr.loc[row, col]), 4) for col in corr.columns}
            for row in corr.index
        }

        return CorrelationReport(
            source_file=source_file,
            method=method,
            columns=list(corr.columns),
            matrix=matrix,
            notable_pairs=self._extract_pairs(corr),
        )

    def heatmap(
        self,
        df: pd.DataFrame,
        method: CorrelationMethod = "pearson",
        columns: Optional[list[str]] = None,
        output_path: Union[str, Path, None] = None,
        figsize: tuple[float, float] = (7, 6),
        dpi: int = 110,
    ) -> CorrelationHeatmap:
        """Render the correlation matrix as an annotated heatmap image."""
        numeric_df = self._numeric_frame(df, columns)

        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns to compute correlations.")

        corr = numeric_df.corr(method=method)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                value = corr.values[i, j]
                text_color = "white" if abs(value) > 0.6 else "black"
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8,
                )

        ax.set_title(f"Correlation Heatmap ({method})")
        fig.colorbar(im, ax=ax, shrink=0.8)
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

        return CorrelationHeatmap(
            method=method,
            columns=list(corr.columns),
            image_base64=encoded,
            saved_path=saved_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_pairs(self, corr: pd.DataFrame) -> list[CorrelationPair]:
        pairs: list[CorrelationPair] = []
        columns = list(corr.columns)

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                value = float(corr.iloc[i, j])
                if pd.isna(value):
                    continue
                strength = self._classify(value)
                if strength in ("moderate", "strong"):
                    pairs.append(
                        CorrelationPair(
                            column_a=columns[i],
                            column_b=columns[j],
                            correlation=round(value, 4),
                            strength=strength,
                        )
                    )

        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
        return pairs

    def _classify(self, value: float) -> str:
        magnitude = abs(value)
        if magnitude >= self.strong_threshold:
            return "strong"
        if magnitude >= self.moderate_threshold:
            return "moderate"
        return "weak"

    @staticmethod
    def _numeric_frame(
        df: pd.DataFrame, columns: Optional[list[str]]
    ) -> pd.DataFrame:
        candidate_cols = columns if columns is not None else list(df.columns)
        numeric_cols = []
        for col in candidate_cols:
            if col not in df.columns:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any() and coerced.notna().mean() >= 0.95:
                numeric_cols.append(col)

        return df[numeric_cols].apply(pd.to_numeric, errors="coerce")