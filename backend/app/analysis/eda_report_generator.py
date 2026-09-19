"""
Automated EDA Report Generator
=================================

Runs the existing analysis suite (classification, missing values,
duplicates, outliers, correlation, distributions, categorical
breakdowns, and the data quality score) over a DataFrame and assembles
the results into one structured report containing:

  * summary   - top-line dataset stats (rows, columns, quality score/grade,
                duplicate count, total missing cells)
  * columns   - per-column classification + missing-value stats
  * charts    - base64-encoded PNGs: numeric distributions, categorical
                bar charts, and a correlation heatmap (when applicable)
  * insights  - a plain-English bullet list auto-generated from the above

Usage
-----
    from app.analysis.eda_report_generator import EDAReportGenerator

    generator = EDAReportGenerator()
    report = generator.generate(df, source_file="sales.csv")

    report.to_json("out/report.json")
    report.to_html("out/report.html")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .categorical_analyzer import CategoricalAnalyzer
from .column_classifier import ColumnClassifier
from .correlation_analyzer import CorrelationAnalyzer
from .distribution_plotter import DistributionPlotter
from .duplicate_detector import DuplicateDetector
from .missing_value_analyzer import MissingValueAnalyzer
from .outlier_detector import OutlierDetector
from .quality_scorer import DataQualityScorer
from .schema import DuplicateReport

MAX_DISTRIBUTION_CHARTS = 8
MAX_CATEGORICAL_CHARTS = 8
NOTABLE_OUTLIER_PERCENTAGE = 5.0


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ChartArtifact:
    """A single rendered chart embedded in the report."""

    title: str
    kind: str  # "distribution" | "categorical" | "correlation"
    column: Optional[str]
    image_base64: str  # PNG, base64-encoded (no "data:" prefix)

    def to_dict(self) -> dict:
        return asdict(self)

    def as_data_uri(self) -> str:
        return f"data:image/png;base64,{self.image_base64}"


@dataclass
class ColumnSummary:
    """Per-column snapshot shown in the report's column table."""

    name: str
    category: str
    dtype: str
    missing_count: int
    missing_percentage: float
    missing_severity: str
    unique_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EDASummary:
    """Top-line dataset stats."""

    row_count: int
    column_count: int
    quality_score: float
    quality_grade: str
    duplicate_row_count: int
    total_missing_cells: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EDAReport:
    """Full automated EDA report for one dataset."""

    source_file: str
    summary: EDASummary
    columns: list[ColumnSummary] = field(default_factory=list)
    charts: list[ChartArtifact] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "summary": self.summary.to_dict(),
            "columns": [c.to_dict() for c in self.columns],
            "charts": [c.to_dict() for c in self.charts],
            "insights": self.insights,
        }

    def to_json(self, output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def to_html(self, output_path: Union[str, Path]) -> Path:
        """Render a single self-contained HTML file (images inlined as data URIs)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_html = "".join(
            f"<tr><td>{c.name}</td><td>{c.category}</td><td>{c.dtype}</td>"
            f"<td>{c.missing_count}</td><td>{c.missing_percentage}%</td>"
            f"<td>{c.missing_severity}</td><td>{c.unique_count}</td></tr>"
            for c in self.columns
        )
        insights_html = (
            "".join(f"<li>{i}</li>" for i in self.insights)
            or "<li>No notable issues found.</li>"
        )
        charts_html = "".join(
            f'<figure><img src="{c.as_data_uri()}" alt="{c.title}">'
            f"<figcaption>{c.title}</figcaption></figure>"
            for c in self.charts
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EDA Report - {self.source_file}</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1, h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; }}
  .summary-grid {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .summary-card {{ background: #f8f9fb; border: 1px solid #e2e2e2; border-radius: 8px;
                    padding: 1rem 1.5rem; min-width: 140px; }}
  .summary-card .value {{ font-size: 1.6rem; font-weight: 700; }}
  .summary-card .label {{ font-size: 0.8rem; color: #666; text-transform: uppercase; }}
  figure {{ display: inline-block; margin: 0 1rem 1.5rem 0; }}
  figure img {{ max-width: 480px; border: 1px solid #eee; border-radius: 4px; }}
  figcaption {{ font-size: 0.85rem; color: #555; margin-top: 0.3rem; }}
</style>
</head>
<body>
  <h1>EDA Report: {self.source_file}</h1>

  <div class="summary-grid">
    <div class="summary-card"><div class="value">{self.summary.row_count}</div><div class="label">Rows</div></div>
    <div class="summary-card"><div class="value">{self.summary.column_count}</div><div class="label">Columns</div></div>
    <div class="summary-card"><div class="value">{self.summary.quality_score} ({self.summary.quality_grade})</div><div class="label">Quality Score</div></div>
    <div class="summary-card"><div class="value">{self.summary.duplicate_row_count}</div><div class="label">Duplicate Rows</div></div>
    <div class="summary-card"><div class="value">{self.summary.total_missing_cells}</div><div class="label">Missing Cells</div></div>
  </div>

  <h2>Insights</h2>
  <ul>{insights_html}</ul>

  <h2>Columns</h2>
  <table>
    <tr><th>Name</th><th>Category</th><th>Dtype</th><th>Missing</th><th>Missing %</th><th>Severity</th><th>Unique</th></tr>
    {rows_html}
  </table>

  <h2>Charts</h2>
  {charts_html}
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class EDAReportGenerator:
    """Runs the full analysis suite and assembles a structured EDA report."""

    def __init__(
        self,
        classifier: Optional[ColumnClassifier] = None,
        missing_analyzer: Optional[MissingValueAnalyzer] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        outlier_detector: Optional[OutlierDetector] = None,
        correlation_analyzer: Optional[CorrelationAnalyzer] = None,
        distribution_plotter: Optional[DistributionPlotter] = None,
        categorical_analyzer: Optional[CategoricalAnalyzer] = None,
        quality_scorer: Optional[DataQualityScorer] = None,
        max_distribution_charts: int = MAX_DISTRIBUTION_CHARTS,
        max_categorical_charts: int = MAX_CATEGORICAL_CHARTS,
    ) -> None:
        self.classifier = classifier or ColumnClassifier()
        self.missing_analyzer = missing_analyzer or MissingValueAnalyzer()
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.outlier_detector = outlier_detector or OutlierDetector()
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        self.distribution_plotter = distribution_plotter or DistributionPlotter()
        self.categorical_analyzer = categorical_analyzer or CategoricalAnalyzer()
        self.quality_scorer = quality_scorer or DataQualityScorer()
        self.max_distribution_charts = max_distribution_charts
        self.max_categorical_charts = max_categorical_charts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df: pd.DataFrame, source_file: str = "unknown") -> EDAReport:
        if df.empty:
            raise ValueError("Cannot generate an EDA report for an empty DataFrame.")

        classification = self.classifier.classify(df)
        missing = self.missing_analyzer.analyze(df)
        duplicates = self.duplicate_detector.detect(df)
        quality = self.quality_scorer.score(df, source_file=source_file)

        category_by_col = {c.name: c.category for c in classification.columns}
        missing_by_col = {c.name: c for c in missing.columns}

        columns = [
            ColumnSummary(
                name=col,
                category=category_by_col.get(col, "text"),
                dtype=str(df[col].dtype),
                missing_count=missing_by_col[col].null_count,
                missing_percentage=missing_by_col[col].null_percentage,
                missing_severity=missing_by_col[col].severity,
                unique_count=int(df[col].nunique(dropna=True)),
            )
            for col in df.columns
        ]

        summary = EDASummary(
            row_count=int(df.shape[0]),
            column_count=int(df.shape[1]),
            quality_score=quality.overall_score,
            quality_grade=quality.grade,
            duplicate_row_count=duplicates.full_duplicate_count,
            total_missing_cells=int(df.isna().sum().sum()),
        )

        charts = self._build_charts(df, category_by_col)
        insights = self._build_insights(df, summary, columns, duplicates, category_by_col)

        return EDAReport(
            source_file=source_file,
            summary=summary,
            columns=columns,
            charts=charts,
            insights=insights,
        )

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _build_charts(
        self, df: pd.DataFrame, category_by_col: dict[str, str]
    ) -> list[ChartArtifact]:
        charts: list[ChartArtifact] = []

        numeric_cols = [c for c, cat in category_by_col.items() if cat == "numerical"]
        for col in numeric_cols[: self.max_distribution_charts]:
            try:
                plot = self.distribution_plotter.combined(df, col)
            except (KeyError, ValueError):
                continue
            charts.append(
                ChartArtifact(
                    title=f"Distribution of {col}",
                    kind="distribution",
                    column=col,
                    image_base64=plot.image_base64,
                )
            )

        categorical_cols = [c for c, cat in category_by_col.items() if cat == "categorical"]
        for col in categorical_cols[: self.max_categorical_charts]:
            try:
                plot = self.categorical_analyzer.distribution_plot(df, col)
            except (KeyError, ValueError):
                continue
            charts.append(
                ChartArtifact(
                    title=f"Category Distribution of {col}",
                    kind="categorical",
                    column=col,
                    image_base64=plot.image_base64,
                )
            )

        if len(numeric_cols) >= 2:
            try:
                heatmap = self.correlation_analyzer.heatmap(df, columns=numeric_cols)
                charts.append(
                    ChartArtifact(
                        title="Correlation Heatmap",
                        kind="correlation",
                        column=None,
                        image_base64=heatmap.image_base64,
                    )
                )
            except ValueError:
                pass

        return charts

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def _build_insights(
        self,
        df: pd.DataFrame,
        summary: EDASummary,
        columns: list[ColumnSummary],
        duplicates: DuplicateReport,
        category_by_col: dict[str, str],
    ) -> list[str]:
        insights: list[str] = []

        insights.append(
            f"Dataset has {summary.row_count} rows and {summary.column_count} columns, "
            f"with an overall quality score of {summary.quality_score}/100 "
            f"(grade {summary.quality_grade})."
        )

        if duplicates.full_duplicate_count > 0 and summary.row_count:
            pct = round(duplicates.full_duplicate_count / summary.row_count * 100, 1)
            insights.append(
                f"{duplicates.full_duplicate_count} duplicate row(s) found ({pct}% of the dataset)."
            )

        for col in sorted(columns, key=lambda c: c.missing_percentage, reverse=True):
            if col.missing_severity in ("high", "critical"):
                insights.append(
                    f"Column '{col.name}' has {col.missing_percentage}% missing values "
                    f"({col.missing_severity} severity) - consider imputing or dropping it."
                )

        numeric_cols = [c for c, cat in category_by_col.items() if cat == "numerical"]
        if numeric_cols:
            outlier_report = self.outlier_detector.detect(df, method="iqr", columns=numeric_cols)
            for col_stats in outlier_report.columns:
                if col_stats.outlier_percentage >= NOTABLE_OUTLIER_PERCENTAGE:
                    insights.append(
                        f"Column '{col_stats.name}' has {col_stats.outlier_count} outlier "
                        f"value(s) ({col_stats.outlier_percentage}%) per the IQR method."
                    )

        if len(numeric_cols) >= 2:
            try:
                corr_report = self.correlation_analyzer.analyze(df, columns=numeric_cols)
                for pair in corr_report.notable_pairs[:5]:
                    direction = "positively" if pair.correlation > 0 else "negatively"
                    insights.append(
                        f"'{pair.column_a}' and '{pair.column_b}' are {pair.strength}ly "
                        f"{direction} correlated (r = {pair.correlation})."
                    )
            except ValueError:
                pass

        for col in columns:
            if col.unique_count == summary.row_count and summary.row_count > 1:
                insights.append(
                    f"Column '{col.name}' has all-unique values - likely an identifier column."
                )

        constant_cols = [c.name for c in columns if c.unique_count <= 1]
        if constant_cols:
            insights.append(
                f"Column(s) {constant_cols} have a single constant value and carry no information."
            )

        return insights