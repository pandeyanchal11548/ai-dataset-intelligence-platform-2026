"""Shared helpers for the analysis modules (quality, EDA, feature insights)."""

from __future__ import annotations

import re
from typing import Mapping, Optional

import numpy as np
import pandas as pd

NUMERIC = "numeric"
BOOLEAN = "boolean"
DATETIME = "datetime"
CATEGORICAL = "categorical"
TEXT = "text"

# DatasetLoader's InferredType -> the coarser kinds used for analysis
_LOADER_MAP = {
    "integer": NUMERIC,
    "float": NUMERIC,
    "boolean": BOOLEAN,
    "datetime": DATETIME,
    "categorical": CATEGORICAL,
    "text": TEXT,
}
_FALLBACK_CATEGORICAL_MAX_UNIQUE = 50
_ID_NAME = re.compile(r"(^|[_\s-])id$|^id($|[_\s-])|uuid|guid", re.IGNORECASE)


def resolve_kinds(
    df: pd.DataFrame, inferred_types: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Decide how each column should be analysed.

    Prefers the loader's inferred types (which see through string-typed dates,
    numbers, etc.) and falls back to the pandas dtype when unavailable.
    """
    inferred_types = inferred_types or {}
    return {
        col: _LOADER_MAP.get(inferred_types.get(col, ""))
        or _kind_from_dtype(df[col])
        for col in df.columns
    }


def _kind_from_dtype(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return BOOLEAN
    if pd.api.types.is_numeric_dtype(s):
        return NUMERIC
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME
    if s.nunique(dropna=True) <= _FALLBACK_CATEGORICAL_MAX_UNIQUE:
        return CATEGORICAL
    return TEXT


def coerce_series(series: pd.Series, kind: str) -> pd.Series:
    """Best-effort conversion to the analysed kind; failures become NaN/NaT."""
    if kind == NUMERIC:
        try:
            return pd.to_numeric(series, errors="coerce").astype("float64")
        except (TypeError, ValueError):
            return pd.Series(np.nan, index=series.index, dtype="float64")
    if kind == DATETIME:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.tz_localize(None) if series.dt.tz else series
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
            return parsed.dt.tz_localize(None)
        except (TypeError, ValueError):
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return series


def numeric_frame(df: pd.DataFrame, kinds: Mapping[str, str]) -> pd.DataFrame:
    """DataFrame of all numeric columns, coerced, with +-inf replaced by NaN."""
    cols = {c: coerce_series(df[c], NUMERIC) for c in df.columns if kinds[c] == NUMERIC}
    return pd.DataFrame(cols, index=df.index).replace([np.inf, -np.inf], np.nan)


def iqr_outliers(values: pd.Series) -> tuple[int, Optional[float], Optional[float]]:
    """Count of Tukey-fence (1.5 * IQR) outliers plus the fences themselves."""
    values = values.dropna()
    if len(values) < 4:
        return 0, None, None
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0, float(q1), float(q3)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < lo) | (values > hi)).sum()), float(lo), float(hi)


def id_like_name(name: str) -> bool:
    return bool(_ID_NAME.search(str(name)))