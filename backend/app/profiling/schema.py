"""
Data structures returned by the Dataset Profiling Engine.

Kept as plain dataclasses so they can be dumped straight to JSON
(via to_dict) or flattened into a CSV summary report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ColumnStats:
    """Per-column statistics used by the profiling engine."""

    name: str
    dtype: str
    missing_count: int
    missing_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProfileReport:
    """Full profiling result for one dataset."""

    source_file: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_indices: list[int] = field(default_factory=list)
    columns: list[ColumnStats] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def data_types(self) -> dict[str, str]:
        return {c.name: c.dtype for c in self.columns}

    @property
    def missing_values(self) -> dict[str, dict[str, Any]]:
        return {
            c.name: {"count": c.missing_count, "percentage": c.missing_percentage}
            for c in self.columns
        }

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "generated_at": self.generated_at,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_indices": self.duplicate_row_indices,
            "data_types": self.data_types,
            "missing_values": self.missing_values,
        }