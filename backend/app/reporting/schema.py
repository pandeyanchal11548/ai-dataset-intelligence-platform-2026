"""Data structures for the unified, structured report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..utils.serialization import to_jsonable

ENGINE_VERSION = "1.0"

# Canonical section order used by the engine and every exporter.
SECTION_ORDER = ("profiling", "quality", "eda", "feature_insights")
SECTION_TITLES = {
    "profiling": "Dataset Profiling",
    "quality": "Data Quality",
    "eda": "Exploratory Data Analysis",
    "feature_insights": "Feature Insights",
}


@dataclass
class ReportSection:
    """One analysis section. status is 'ok', 'error' or 'skipped'."""

    key: str
    title: str
    status: str = "ok"
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "status": self.status,
            "error": self.error,
            "data": to_jsonable(self.data),
        }


@dataclass
class StructuredReport:
    metadata: dict[str, Any]
    summary: dict[str, Any]
    sections: dict[str, ReportSection]

    def to_dict(self) -> dict:
        return {
            "metadata": to_jsonable(self.metadata),
            "summary": to_jsonable(self.summary),
            "sections": {
                k: self.sections[k].to_dict() for k in SECTION_ORDER if k in self.sections
            },
        }