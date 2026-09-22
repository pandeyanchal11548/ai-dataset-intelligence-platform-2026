"""Exploratory data analysis: distributions, categories, dates, correlations."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .common import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    TEXT,
    coerce_series,
    iqr_outliers,
    numeric_frame,
    resolve_kinds,
)


class EDAAnalyzer:
    def __init__(
        self,
        histogram_bins: int = 10,
        top_n: int = 10,
        min_correlation: float = 0.5,
        max_pairs: int = 20,
        max_matrix_columns: int = 15,
    ) -> None:
        self.histogram_bins = histogram_bins
        self.top_n = top_n
        self.min_correlation = min_correlation
        self.max_pairs = max_pairs
        self.max_matrix_columns = max_matrix_columns

    def analyze(
        self, df: pd.DataFrame, kinds: Optional[Mapping[str, str]] = None
    ) -> dict[str, Any]:
        kinds = kinds or resolve_kinds(df)
        numeric, categorical, datetimes = [], [], []

        for col in df.columns:
            kind = kinds[col]
            if kind == NUMERIC:
                entry = self._numeric(col, coerce_series(df[col], NUMERIC), len(df))
                if entry:
                    numeric.append(entry)
            elif kind in (CATEGORICAL, BOOLEAN, TEXT):
                entry = self._categorical(col, df[col], kind, len(df))
                if entry:
                    categorical.append(entry)
            elif kind == DATETIME:
                entry = self._datetime(col, coerce_series(df[col], DATETIME), len(df))
                if entry:
                    datetimes.append(entry)

        pairs, matrix = self._correlations(df, kinds)
        return {
            "kind_counts": {
                k: sum(1 for v in kinds.values() if v == k)
                for k in (NUMERIC, CATEGORICAL, BOOLEAN, DATETIME, TEXT)
            },
            "numeric_summary": numeric,
            "categorical_summary": categorical,
            "datetime_summary": datetimes,
            "correlations": {
                "min_abs_correlation": self.min_correlation,
                "strong_pairs": pairs,
                "matrix": matrix,
            },
        }

    # ------------------------------------------------------------------

    def _numeric(self, name, s: pd.Series, n_rows: int):
        s = s.replace([np.inf, -np.inf], np.nan)
        v = s.dropna()
        if v.empty:
            return None
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        n_out, lo, hi = iqr_outliers(v)
        bins = max(1, min(self.histogram_bins, int(v.nunique())))
        counts, edges = np.histogram(v, bins=bins)
        return {
            "name": name,
            "count": int(len(v)),
            "missing": int(n_rows - len(v)),
            "mean": float(v.mean()),
            "std": float(v.std()) if len(v) > 1 else None,
            "min": float(v.min()),
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "max": float(v.max()),
            "skew": float(v.skew()) if len(v) >= 3 else None,
            "zeros": int((v == 0).sum()),
            "outlier_count": n_out,
            "outlier_percentage": round(n_out / len(v) * 100, 2),
            "histogram": {"counts": counts.tolist(), "bin_edges": edges.tolist()},
        }

    def _categorical(self, name, s: pd.Series, kind: str, n_rows: int):
        v = s.dropna()
        if v.empty:
            return None
        as_str = v.astype(str)
        vc = as_str.value_counts()
        entry = {
            "name": name,
            "kind": kind,
            "count": int(len(v)),
            "missing": int(n_rows - len(v)),
            "unique": int(vc.size),
            "mode": str(vc.index[0]),
            "top_values": [
                {"value": str(val), "count": int(c), "percentage": round(c / len(v) * 100, 2)}
                for val, c in vc.head(self.top_n).items()
            ],
        }
        if kind == TEXT:
            entry["avg_length"] = round(float(as_str.str.len().mean()), 2)
        return entry

    def _datetime(self, name, s: pd.Series, n_rows: int):
        v = s.dropna()
        if v.empty:
            return None
        return {
            "name": name,
            "count": int(len(v)),
            "missing": int(n_rows - len(v)),
            "min": v.min(),
            "max": v.max(),
            "span_days": int((v.max() - v.min()).days),
        }

    def _correlations(self, df, kinds):
        num = numeric_frame(df, kinds)
        num = num.loc[:, num.nunique() > 1]  # constants have undefined correlation
        if num.shape[1] < 2:
            return [], None
        corr = num.corr()
        cols = list(corr.columns)
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iat[i, j]
                if pd.notna(r) and abs(r) >= self.min_correlation:
                    pairs.append(
                        {"feature_a": cols[i], "feature_b": cols[j],
                         "correlation": round(float(r), 4)}
                    )
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        matrix = None
        if len(cols) <= self.max_matrix_columns:
            matrix = {
                "columns": cols,
                "values": [[round(float(x), 4) for x in row] for row in corr.values],
            }
        return pairs[: self.max_pairs], matrix