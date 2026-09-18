"""
Distribution Plot Generator
=============================

Generates histograms and kernel-density (KDE) plots for numeric columns.

Each plot method returns a DistributionPlot containing a base64-encoded
PNG (ready to embed in an API JSON response or an <img> tag) plus the
summary statistics used to build it.

Requires: matplotlib, scipy (scipy powers pandas' KDE/density plotting).
Add both to backend/requirements.txt if not already present.

Usage
-----
    from app.analysis.distribution_plotter import DistributionPlotter

    plotter = DistributionPlotter()
    hist = plotter.histogram(df, "amount")
    density = plotter.density(df, "amount")
    both = plotter.combined(df, "amount")   # histogram + KDE overlay

    print(hist.as_data_uri())   # "data:image/png;base64,...."
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib
matplotlib.use("Agg")  # headless/server-safe backend - must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

PlotKind = Literal["histogram", "density", "combined"]


@dataclass
class DistributionStats:
    """Summary statistics for the column being plotted."""

    column: str
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DistributionPlot:
    """A rendered plot plus the stats used to generate it."""

    column: str
    kind: PlotKind
    image_base64: str  # PNG, base64-encoded (no "data:" prefix)
    stats: DistributionStats
    saved_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "kind": self.kind,
            "image_base64": self.image_base64,
            "stats": self.stats.to_dict(),
            "saved_path": self.saved_path,
        }

    def as_data_uri(self) -> str:
        """Convenience for embedding directly in HTML/<img src=...>."""
        return f"data:image/png;base64,{self.image_base64}"


class DistributionPlotter:
    """Generates histograms and density (KDE) plots for numeric columns."""

    def __init__(self, figsize: tuple[float, float] = (7, 4.5), dpi: int = 110) -> None:
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def histogram(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 30,
        output_path: Union[str, Path, None] = None,
    ) -> DistributionPlot:
        """Plot a histogram of `column`."""
        series = self._numeric_series(df, column)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.hist(series, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.9)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        return self._finalize(fig, series, column, "histogram", output_path)

    def density(
        self,
        df: pd.DataFrame,
        column: str,
        output_path: Union[str, Path, None] = None,
    ) -> DistributionPlot:
        """Plot a kernel density estimate (KDE) of `column`."""
        series = self._numeric_series(df, column)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        series.plot(kind="density", ax=ax, color="#DD8452", linewidth=2)
        ax.set_title(f"Density Plot of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        fig.tight_layout()

        return self._finalize(fig, series, column, "density", output_path)

    def combined(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 30,
        output_path: Union[str, Path, None] = None,
    ) -> DistributionPlot:
        """Overlay a normalized histogram with a KDE curve for `column`."""
        series = self._numeric_series(df, column)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.hist(
            series, bins=bins, density=True, color="#4C72B0",
            edgecolor="white", alpha=0.6, label="Histogram",
        )
        series.plot(kind="density", ax=ax, color="#DD8452", linewidth=2, label="KDE")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        return self._finalize(fig, series, column, "combined", output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Column '{column}' has no numeric values to plot.")
        return series

    def _finalize(
        self,
        fig,
        series: pd.Series,
        column: str,
        kind: PlotKind,
        output_path: Union[str, Path, None],
    ) -> DistributionPlot:
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

        stats = DistributionStats(
            column=column,
            count=int(series.count()),
            mean=round(float(series.mean()), 4),
            median=round(float(series.median()), 4),
            std=round(float(series.std()), 4) if series.count() > 1 else 0.0,
            min=round(float(series.min()), 4),
            max=round(float(series.max()), 4),
        )

        return DistributionPlot(
            column=column,
            kind=kind,
            image_base64=encoded,
            stats=stats,
            saved_path=saved_path,
        )