"""
Report Structuring Engine
==========================

Runs the four analysis stages over one DataFrame and combines them into a
single StructuredReport:

    profiling         -> DatasetProfiler
    quality           -> QualityAnalyzer
    eda               -> EDAAnalyzer
    feature_insights  -> FeatureInsightAnalyzer

Each stage is isolated: if one raises, its section is marked "error" and the
rest of the report is still produced. A top-level summary with headline
numbers and plain-English key findings is derived from the sections.

Usage
-----
    df, loader_profile = DatasetLoader().load("data/raw/sales.csv")
    report = ReportStructuringEngine().build(
        df, source_file="sales.csv", loader_profile=loader_profile, target="amount"
    )
    ReportExporter().export_all(report, "data/processed")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from ..analysis.common import resolve_kinds
from ..analysis.eda import EDAAnalyzer
from ..analysis.feature_insights import FeatureInsightAnalyzer
from ..analysis.quality import QualityAnalyzer
from ..profiling.profiler import DatasetProfiler
from .schema import (
    ENGINE_VERSION,
    SECTION_ORDER,
    SECTION_TITLES,
    ReportSection,
    StructuredReport,
)

MAX_DUPLICATE_INDICES = 100
MAX_FINDINGS = 12


class ReportStructuringEngine:
    def __init__(
        self,
        profiler: Optional[DatasetProfiler] = None,
        quality: Optional[QualityAnalyzer] = None,
        eda: Optional[EDAAnalyzer] = None,
        features: Optional[FeatureInsightAnalyzer] = None,
    ) -> None:
        self.profiler = profiler or DatasetProfiler()
        self.quality = quality or QualityAnalyzer()
        self.eda = eda or EDAAnalyzer()
        self.features = features or FeatureInsightAnalyzer()

    # ------------------------------------------------------------------

    def build(
        self,
        df: pd.DataFrame,
        source_file: str = "unknown",
        loader_profile: Any = None,
        target: Optional[str] = None,
        title: Optional[str] = None,
    ) -> StructuredReport:
        """
        Parameters
        ----------
        loader_profile : DatasetProfile from DatasetLoader (optional). Supplies
            inferred column types and loader warnings.
        target : optional column name; enables target-association analysis.
        """
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("Cannot build a report from an empty DataFrame.")

        if not all(isinstance(c, str) for c in df.columns):
            df = df.copy()
            df.columns = [str(c) for c in df.columns]
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset.")

        inferred = {
            c.name: c.inferred_type for c in getattr(loader_profile, "columns", []) or []
        }
        kinds = resolve_kinds(df, inferred)

        sections = {
            "profiling": self._run(
                "profiling",
                lambda: self._profiling(df, source_file, inferred, kinds, loader_profile),
            ),
            "quality": self._run("quality", lambda: self.quality.analyze(df, kinds)),
            "eda": self._run("eda", lambda: self.eda.analyze(df, kinds)),
            "feature_insights": self._run(
                "feature_insights", lambda: self.features.analyze(df, kinds, target)
            ),
        }

        metadata = {
            "report_title": title or f"Dataset Report: {source_file}",
            "source_file": source_file,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "target": target,
            "engine_version": ENGINE_VERSION,
        }
        summary = self._summarize(sections, df.shape, target)
        return StructuredReport(metadata=metadata, summary=summary, sections=sections)

    # ------------------------------------------------------------------

    @staticmethod
    def _run(key: str, fn: Callable[[], dict]) -> ReportSection:
        section = ReportSection(key=key, title=SECTION_TITLES[key])
        try:
            section.data = fn()
        except Exception as exc:  # isolate: one bad stage must not kill the report
            section.status = "error"
            section.error = f"{type(exc).__name__}: {exc}"
        return section

    def _profiling(self, df, source_file, inferred, kinds, loader_profile) -> dict:
        report = self.profiler.profile(df, source_file=source_file)
        data = report.to_dict()
        idx = data["duplicate_row_indices"]
        data["duplicate_row_indices_truncated"] = len(idx) > MAX_DUPLICATE_INDICES
        data["duplicate_row_indices"] = idx[:MAX_DUPLICATE_INDICES]
        data["columns"] = [
            {
                "name": c.name,
                "dtype": c.dtype,
                "inferred_type": inferred.get(c.name, kinds.get(c.name, "")),
                "missing_count": c.missing_count,
                "missing_percentage": c.missing_percentage,
            }
            for c in report.columns
        ]
        data["loader_warnings"] = list(getattr(loader_profile, "warnings", []) or [])
        return data

    # ------------------------------------------------------------------

    def _summarize(self, sections, shape, target) -> dict[str, Any]:
        n_rows, n_cols = shape
        ok = {k: s.data for k, s in sections.items() if s.status == "ok"}
        summary: dict[str, Any] = {
            "rows": int(n_rows),
            "columns": int(n_cols),
            "section_status": {k: sections[k].status for k in SECTION_ORDER},
            "quality_score": None,
            "quality_grade": None,
            "issue_counts": None,
            "duplicate_rows": None,
            "missing_cells_percentage": None,
        }
        findings: list[str] = []

        prof = ok.get("profiling")
        if prof:
            summary["duplicate_rows"] = prof["duplicate_row_count"]
            missing = sum(c["missing_count"] for c in prof["columns"])
            summary["missing_cells_percentage"] = round(missing / (n_rows * n_cols) * 100, 2)
            for w in prof.get("loader_warnings", [])[:3]:
                findings.append(f"Loader: {w}")

        q = ok.get("quality")
        if q:
            summary.update(
                quality_score=q["score"], quality_grade=q["grade"], issue_counts=q["issue_counts"]
            )
            c = q["issue_counts"]
            findings.insert(
                0,
                f"Data quality score {q['score']}/100 (grade {q['grade']}): "
                f"{c['high']} high, {c['medium']} medium, {c['low']} low severity issue(s).",
            )
            findings += [i["message"] if i["column"] is None else f"'{i['column']}': {i['message']}"
                         for i in q["issues"] if i["severity"] == "high"][:4]

        eda = ok.get("eda")
        if eda and eda["correlations"]["strong_pairs"]:
            p = eda["correlations"]["strong_pairs"][0]
            findings.append(
                f"Strongest correlation: '{p['feature_a']}' vs '{p['feature_b']}' (r={p['correlation']:+.2f})."
            )

        fi = ok.get("feature_insights")
        if fi:
            if fi["exclude_candidates"]:
                names = ", ".join(f"'{n}'" for n in fi["exclude_candidates"][:5])
                findings.append(f"Exclude from modelling (constant/empty/identifier): {names}.")
            if fi["redundant_pairs"]:
                findings.append(f"{len(fi['redundant_pairs'])} near-redundant numeric feature pair(s) found.")
            ta = fi.get("target_analysis")
            if ta and ta["rankings"]:
                top = ta["rankings"][0]
                findings.append(
                    f"Feature most associated with target '{target}': '{top['feature']}' "
                    f"({top['strength']}, {top['association']:.2f})."
                )
                if top["possible_leakage"]:
                    findings.append(f"'{top['feature']}' is almost perfectly associated with the target; check for leakage.")

        for k in SECTION_ORDER:
            if sections[k].status == "error":
                findings.append(f"Section '{sections[k].title}' failed: {sections[k].error}")

        summary["key_findings"] = findings[:MAX_FINDINGS]
        return summary