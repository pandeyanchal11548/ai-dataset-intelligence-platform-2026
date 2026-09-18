"""
Schema data structures returned by the dataset loader.

These are plain, serializable dataclasses so they can be turned into
JSON directly by the future FastAPI layer / consumed by the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

InferredType = Literal[
    "integer",
    "float",
    "boolean",
    "datetime",
    "categorical",
    "text",
    "unknown",
]


@dataclass
class ColumnProfile:
    """Everything the loader figured out about a single column."""

    name: str
    original_dtype: str
    inferred_type: InferredType
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetProfile:
    """Summary returned alongside the loaded DataFrame."""

    source_file: str
    file_type: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "file_type": self.file_type,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "warnings": self.warnings,
            "columns": [c.to_dict() for c in self.columns],
        }
