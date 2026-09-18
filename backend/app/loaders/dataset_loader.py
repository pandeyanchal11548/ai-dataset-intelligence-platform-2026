"""
Dataset Loader Module
======================

Loads CSV / Excel files into a unified pandas DataFrame, while running:

  * File validation      - extension, existence, size, corruption checks
  * Column detection      - header sanitation, duplicate/blank handling
  * Type inference         - integer / float / boolean / datetime /
                              categorical / text, independent of whatever
                              dtype pandas guessed on read

Usage
-----
    from app.loaders.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    df, profile = loader.load("data/raw/sales.xlsx")

    print(profile.to_dict())
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd

from .exceptions import (
    EmptyDatasetError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from .schema import ColumnProfile, DatasetProfile, InferredType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS

MAX_FILE_SIZE_MB = 500          # guard rail; tune per deployment
SAMPLE_VALUES_PER_COLUMN = 5
CATEGORICAL_UNIQUE_RATIO = 0.05   # <=5% unique values -> treat as categorical
CATEGORICAL_MAX_UNIQUE = 50       # ...but never above this absolute count


class DatasetLoader:
    """Loads tabular files (CSV/Excel) into a validated, profiled DataFrame."""

    def __init__(
        self,
        max_file_size_mb: int = MAX_FILE_SIZE_MB,
        sample_size: int = SAMPLE_VALUES_PER_COLUMN,
    ) -> None:
        self.max_file_size_mb = max_file_size_mb
        self.sample_size = sample_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self, file_path: Union[str, Path], sheet_name: Union[str, int, None] = 0
    ) -> tuple[pd.DataFrame, DatasetProfile]:
        """
        Load a CSV/Excel file and return (DataFrame, DatasetProfile).

        Raises
        ------
        UnsupportedFileTypeError, FileValidationError, EmptyDatasetError
        """
        path = Path(file_path)
        file_type = self._validate_file(path)

        df = self._read_file(path, file_type, sheet_name=sheet_name)

        if df.shape[0] == 0 or df.shape[1] == 0:
            raise EmptyDatasetError(
                f"'{path.name}' loaded but contains no usable rows/columns."
            )

        df, warnings = self._detect_and_clean_columns(df)
        profile = self._build_profile(df, path, file_type, warnings)

        return df, profile

    # ------------------------------------------------------------------
    # 1. File validation
    # ------------------------------------------------------------------

    def _validate_file(self, path: Path) -> str:
        if not path.exists():
            raise FileValidationError(f"File not found: {path}")

        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb == 0:
            raise FileValidationError(f"File is empty (0 bytes): {path}")
        if size_mb > self.max_file_size_mb:
            raise FileValidationError(
                f"File is {size_mb:.1f} MB, exceeds the "
                f"{self.max_file_size_mb} MB limit."
            )

        return "csv" if suffix in CSV_EXTENSIONS else "excel"

    def _read_file(
        self, path: Path, file_type: str, sheet_name: Union[str, int, None]
    ) -> pd.DataFrame:
        try:
            if file_type == "csv":
                # sep=None + python engine => sniff delimiter (comma/tab/semicolon)
                return pd.read_csv(path, sep=None, engine="python")
            else:
                return pd.read_excel(path, sheet_name=sheet_name, engine=None)
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError(f"'{path.name}' has no parsable data.") from exc
        except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as exc:
            raise FileValidationError(
                f"Failed to parse '{path.name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # 2. Column detection
    # ------------------------------------------------------------------

    def _detect_and_clean_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        warnings: list[str] = []
        new_columns = []
        seen: dict[str, int] = {}

        for i, col in enumerate(df.columns):
            name = str(col).strip()

            if name == "" or name.lower().startswith("unnamed:"):
                name = f"column_{i + 1}"
                warnings.append(
                    f"Column at position {i + 1} had no header; "
                    f"auto-named '{name}'."
                )

            if name in seen:
                seen[name] += 1
                deduped = f"{name}_{seen[name]}"
                warnings.append(
                    f"Duplicate column name '{name}' renamed to '{deduped}'."
                )
                name = deduped
            else:
                seen[name] = 0

            new_columns.append(name)

        df = df.copy()
        df.columns = new_columns

        # Drop fully-empty columns/rows that often appear from stray
        # spreadsheet formatting, but only if it doesn't wipe the dataset.
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols and len(empty_cols) < len(df.columns):
            warnings.append(f"Dropped fully-empty column(s): {empty_cols}")
            df = df.drop(columns=empty_cols)

        empty_rows = df.index[df.isna().all(axis=1)]
        if len(empty_rows) > 0:
            warnings.append(f"Dropped {len(empty_rows)} fully-empty row(s).")
            df = df.drop(index=empty_rows).reset_index(drop=True)

        return df, warnings

    # ------------------------------------------------------------------
    # 3. Type inference
    # ------------------------------------------------------------------

    def _infer_column_type(self, series: pd.Series) -> InferredType:
        non_null = series.dropna()
        if non_null.empty:
            return "unknown"

        # Boolean check (handles True/False, 0/1, yes/no, true/false strings)
        if self._is_boolean(non_null):
            return "boolean"

        # Numeric checks
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = numeric.notna().mean()
        if numeric_ratio >= 0.95:
            if (numeric.dropna() % 1 == 0).all():
                return "integer"
            return "float"

        # Datetime check
        if self._is_datetime(non_null):
            return "datetime"

        # Categorical vs free text, based on cardinality
        unique_ratio = non_null.nunique() / len(non_null)
        if (
            non_null.nunique() <= CATEGORICAL_MAX_UNIQUE
            and unique_ratio <= CATEGORICAL_UNIQUE_RATIO + 0.20
        ) or non_null.nunique() <= 10:
            return "categorical"

        return "text"

    @staticmethod
    def _is_boolean(series: pd.Series) -> bool:
        bool_like = {
            "true", "false", "yes", "no", "y", "n", "0", "1", "t", "f",
        }
        as_str = series.astype(str).str.strip().str.lower()
        return as_str.isin(bool_like).mean() >= 0.98

    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        sample = series.astype(str).head(200)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() >= 0.90

    # ------------------------------------------------------------------
    # Profile assembly
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        df: pd.DataFrame,
        path: Path,
        file_type: str,
        warnings: list[str],
    ) -> DatasetProfile:
        columns: list[ColumnProfile] = []

        for col in df.columns:
            series = df[col]
            inferred = self._infer_column_type(series)
            null_count = int(series.isna().sum())

            columns.append(
                ColumnProfile(
                    name=col,
                    original_dtype=str(series.dtype),
                    inferred_type=inferred,
                    null_count=null_count,
                    null_percentage=round(null_count / len(series) * 100, 2)
                    if len(series) else 0.0,
                    unique_count=int(series.nunique(dropna=True)),
                    sample_values=series.dropna()
                    .head(self.sample_size)
                    .tolist(),
                )
            )

        return DatasetProfile(
            source_file=path.name,
            file_type=file_type,
            n_rows=df.shape[0],
            n_columns=df.shape[1],
            columns=columns,
            warnings=warnings,
        )
