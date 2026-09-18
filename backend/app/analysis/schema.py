"""
Data structures shared by the Column Classification Engine,
Missing Value Analyzer, and Duplicate Detection Module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

ColumnCategory = Literal["numerical", "categorical", "text", "datetime"]
Severity = Literal["none", "low", "moderate", "high", "critical"]


# ---------------------------------------------------------------------------
# Column Classification Engine
# ---------------------------------------------------------------------------


@dataclass
class ColumnClassification:
    name: str
    category: ColumnCategory
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationReport:
    columns: list[ColumnClassification] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {c.name: c.category for c in self.columns}

    def to_detailed_dict(self) -> dict:
        return {c.name: c.to_dict() for c in self.columns}


# ---------------------------------------------------------------------------
# Missing Value Analyzer
# ---------------------------------------------------------------------------


@dataclass
class ColumnMissingStats:
    name: str
    null_count: int
    null_percentage: float
    severity: Severity

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissingValueReport:
    columns: list[ColumnMissingStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {c.name: c.to_dict() for c in self.columns}

    @property
    def worst_columns(self) -> list[str]:
        """Columns with 'high' or 'critical' severity, worst first."""
        rank = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "none": 4}
        flagged = [c for c in self.columns if c.severity in ("high", "critical")]
        return [c.name for c in sorted(flagged, key=lambda c: rank[c.severity])]


# ---------------------------------------------------------------------------
# Duplicate Detection Module
# ---------------------------------------------------------------------------


@dataclass
class DuplicateReport:
    full_duplicate_count: int
    full_duplicate_indices: list[int]
    partial_duplicate_subset: list[str]
    partial_duplicate_count: int
    partial_duplicate_indices: list[int]

    def to_dict(self) -> dict:
        return asdict(self)