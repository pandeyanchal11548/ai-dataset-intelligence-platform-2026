"""Feature insights: per-feature roles/recommendations, redundancy, target links."""

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
    id_like_name,
    iqr_outliers,
    numeric_frame,
    resolve_kinds,
)

_EXCLUDED_ROLES = {"empty", "constant", "identifier"}


def _strength(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "weak"
    if v < 0.5:
        return "moderate"
    return "strong"


def _correlation_ratio(num: pd.Series, cat: pd.Series) -> float:
    grand = num.mean()
    ss_total = float(((num - grand) ** 2).sum())
    if ss_total == 0:
        return 0.0
    g = num.groupby(cat)
    ss_between = float((g.count() * (g.mean() - grand) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total))


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    obs = ct.values.astype(float)
    n = obs.sum()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    chi2 = float(((obs - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * (min(obs.shape) - 1))))


class FeatureInsightAnalyzer:
    def __init__(
        self,
        redundancy_threshold: float = 0.9,
        min_rows_for_id: int = 20,
        max_cat_levels_for_target: int = 50,
        top_n_target_features: int = 15,
    ) -> None:
        self.redundancy_threshold = redundancy_threshold
        self.min_rows_for_id = min_rows_for_id
        self.max_cat_levels = max_cat_levels_for_target
        self.top_n = top_n_target_features

    def analyze(
        self,
        df: pd.DataFrame,
        kinds: Optional[Mapping[str, str]] = None,
        target: Optional[str] = None,
    ) -> dict[str, Any]:
        kinds = kinds or resolve_kinds(df)
        n_rows = len(df)
        features = [
            self._feature(c, coerce_series(df[c], kinds[c]), kinds[c], n_rows)
            for c in df.columns
            if c != target
        ]
        roles: dict[str, int] = {}
        for f in features:
            roles[f["role"]] = roles.get(f["role"], 0) + 1

        excluded = {f["name"] for f in features if f["exclude_from_modeling"]}
        redundant = self._redundant_pairs(df, kinds, excluded | ({target} if target else set()))
        return {
            "features": features,
            "role_counts": roles,
            "exclude_candidates": sorted(excluded),
            "redundant_pairs": redundant,
            "target_analysis": self._target(df, kinds, target, excluded) if target else None,
        }

    # ------------------------------------------------------------------

    def _feature(self, name, s: pd.Series, kind: str, n_rows: int) -> dict[str, Any]:
        v = s.dropna()
        n = len(v)
        miss_pct = round((1 - n / n_rows) * 100, 2) if n_rows else 0.0
        nunique = int(v.nunique())
        ratio = nunique / n if n else 0.0
        recs: list[str] = []
        role = kind

        if n == 0:
            role = "empty"
            recs.append("Drop: the column is entirely missing.")
        elif nunique <= 1:
            role = "constant"
            recs.append("Drop: a constant carries no information.")
        else:
            int_valued = kind == NUMERIC and bool(np.isfinite(v).all() and (v % 1 == 0).all())
            can_be_id = kind in (TEXT, CATEGORICAL) or int_valued
            if can_be_id and (
                (id_like_name(name) and ratio >= 0.9)
                or (n >= self.min_rows_for_id and ratio >= 0.95)
            ):
                role = "identifier"
                recs.append("Exclude from model inputs; keep only as a key / row label.")
            elif kind == NUMERIC:
                v = v.replace([np.inf, -np.inf], np.nan).dropna()
                skew = float(v.skew()) if len(v) >= 3 else 0.0
                if abs(skew) >= 1:
                    fix = "log1p transform" if v.min() >= 0 else "Yeo-Johnson transform"
                    recs.append(f"Skewed (skew={skew:.2f}): consider a {fix}.")
                n_out, _, _ = iqr_outliers(v)
                if len(v) and n_out / len(v) > 0.05:
                    recs.append("Many outliers: prefer robust scaling or winsorizing.")
                if int_valued and nunique <= 10:
                    recs.append(f"Only {nunique} distinct integers: may work better as categorical.")
                if not recs:
                    recs.append("Standardize for linear / distance-based models.")
            elif kind == BOOLEAN:
                recs.append("Encode as 0/1.")
            elif kind == DATETIME:
                recs.append("Derive year, month, day-of-week and elapsed-time features.")
            elif kind == CATEGORICAL:
                if nunique <= 10:
                    recs.append("Low cardinality: one-hot encode.")
                else:
                    recs.append(f"{nunique} levels: group rare levels, then target/frequency encode.")
            else:  # text
                recs.append("Free text: use TF-IDF or embeddings, or drop.")

        if role not in ("empty",) and miss_pct >= 50:
            recs.append(f"{miss_pct:.0f}% missing: consider dropping or adding a missing-indicator.")
        elif role not in ("empty", "constant", "identifier") and miss_pct > 0:
            recs.append(f"{miss_pct:.1f}% missing: impute (median for numeric, mode for categorical).")

        return {
            "name": name,
            "kind": kind,
            "role": role,
            "missing_percentage": miss_pct,
            "unique_count": nunique,
            "exclude_from_modeling": role in _EXCLUDED_ROLES,
            "recommendations": recs,
        }

    def _redundant_pairs(self, df, kinds, skip: set) -> list[dict[str, Any]]:
        num = numeric_frame(df, kinds)
        num = num[[c for c in num.columns if c not in skip and num[c].nunique() > 1]]
        if num.shape[1] < 2:
            return []
        corr = num.corr()
        cols, out = list(corr.columns), []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iat[i, j]
                if pd.notna(r) and abs(r) >= self.redundancy_threshold:
                    out.append({
                        "feature_a": cols[i], "feature_b": cols[j],
                        "correlation": round(float(r), 4),
                        "suggestion": f"Nearly redundant: consider dropping one of '{cols[i]}' / '{cols[j]}'.",
                    })
        return sorted(out, key=lambda p: abs(p["correlation"]), reverse=True)

    def _target(self, df, kinds, target, excluded: set) -> dict[str, Any]:
        t_kind = kinds[target]
        result: dict[str, Any] = {"target": target, "target_kind": t_kind, "rankings": []}
        if t_kind not in (NUMERIC, BOOLEAN, CATEGORICAL):
            result["note"] = f"Target kind '{t_kind}' is not supported for association analysis."
            return result
        if t_kind == CATEGORICAL and df[target].nunique() > self.max_cat_levels:
            result["note"] = "Target has too many levels for association analysis."
            return result
        if len(df) < 30:
            result["note"] = "Small sample (<30 rows): treat these associations as unreliable."

        y = coerce_series(df[target], t_kind)
        rows = []
        for col in df.columns:
            k = kinds[col]
            if col == target or col in excluded or k not in (NUMERIC, BOOLEAN, CATEGORICAL):
                continue
            if k in (CATEGORICAL, BOOLEAN):
                nn = df[col].notna().sum()
                # too many levels, or ~one row per level => association is meaningless
                if df[col].nunique() > self.max_cat_levels or df[col].nunique() > 0.5 * nn:
                    continue
            pair = pd.concat([coerce_series(df[col], k), y], axis=1, keys=["x", "y"]).dropna()
            if len(pair) < 3:
                continue
            x, yy = pair["x"], pair["y"]
            xn, yn = k == NUMERIC, t_kind == NUMERIC
            if xn and yn:
                val = abs(float(x.corr(yy))) if x.nunique() > 1 and yy.nunique() > 1 else 0.0
                method = "|Pearson r|"
            elif xn != yn:
                num, cat = (x, yy) if xn else (yy, x)
                val, method = _correlation_ratio(num, cat.astype(str)), "correlation ratio"
            else:
                val, method = _cramers_v(x.astype(str), yy.astype(str)), "Cramér's V"
            val = 0.0 if np.isnan(val) else val
            rows.append({
                "feature": col, "association": round(val, 4), "method": method,
                "strength": _strength(val),
                "possible_leakage": val >= 0.95,
            })
        rows.sort(key=lambda r: r["association"], reverse=True)
        result["rankings"] = rows[: self.top_n]
        return result