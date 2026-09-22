"""
Report Export Module
=====================

Writes a StructuredReport to disk as:

  * JSON - the full structured report (machine-readable, for the API/frontend)
  * CSV  - flat, sectioned tables (opens cleanly in Excel / Sheets)
  * HTML - one self-contained, printable page (inline CSS, no external assets)
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .schema import SECTION_ORDER, StructuredReport

SUPPORTED_FORMATS = ("html", "json", "csv")
_EXT = {"html": ".html", "json": ".json", "csv": ".csv"}


class ReportExporter:
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_json(self, report: StructuredReport, output_path: Union[str, Path]) -> Path:
        path = self._prepare(output_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, allow_nan=False)
        return path

    def to_csv(self, report: StructuredReport, output_path: Union[str, Path]) -> Path:
        path = self._prepare(output_path)
        with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel-friendly
            _write_csv(csv.writer(f), report.to_dict())
        return path

    def to_html(self, report: StructuredReport, output_path: Union[str, Path]) -> Path:
        path = self._prepare(output_path)
        path.write_text(_render_html(report.to_dict()), encoding="utf-8")
        return path

    def export_all(
        self,
        report: StructuredReport,
        output_dir: Union[str, Path],
        formats: Iterable[str] = SUPPORTED_FORMATS,
        basename: Optional[str] = None,
    ) -> dict[str, Path]:
        """Export to several formats at once; returns {format: path}."""
        formats = [f.lower() for f in formats]
        unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
        if unknown:
            raise ValueError(
                f"Unsupported export format(s): {unknown}. Supported: {list(SUPPORTED_FORMATS)}"
            )
        stem = basename or f"{Path(report.metadata['source_file']).stem}_report"
        out_dir = Path(output_dir)
        writers = {"html": self.to_html, "json": self.to_json, "csv": self.to_csv}
        return {f: writers[f](report, out_dir / f"{stem}{_EXT[f]}") for f in formats}

    @staticmethod
    def _prepare(output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# ======================================================================
# CSV
# ======================================================================


def _write_csv(w, d: dict) -> None:
    meta, summ, secs = d["metadata"], d["summary"], d["sections"]

    def table(title, headers, rows):
        w.writerow([title])
        w.writerow(headers)
        w.writerows(rows)
        w.writerow([])

    w.writerow([meta["report_title"]])
    w.writerow(["Source File", meta["source_file"]])
    w.writerow(["Generated At", meta["generated_at"]])
    w.writerow(["Row Count", meta["row_count"]])
    w.writerow(["Column Count", meta["column_count"]])
    w.writerow(["Target", meta["target"] or ""])
    w.writerow(["Quality Score", _blank(summ["quality_score"])])
    w.writerow(["Quality Grade", _blank(summ["quality_grade"])])
    w.writerow([])
    table("Key Findings", ["#", "Finding"], [[i + 1, f] for i, f in enumerate(summ["key_findings"])])
    table("Section Status", ["Section", "Status", "Error"],
          [[k, secs[k]["status"], secs[k]["error"] or ""] for k in SECTION_ORDER if k in secs])

    def data(key):
        s = secs.get(key)
        return s["data"] if s and s["status"] == "ok" else None

    if (p := data("profiling")) is not None:
        table("Profiling: Columns",
              ["Column Name", "Data Type", "Inferred Type", "Missing Count", "Missing Percentage"],
              [[c["name"], c["dtype"], c["inferred_type"], c["missing_count"], c["missing_percentage"]]
               for c in p["columns"]])
        table("Profiling: Duplicates", ["Duplicate Row Count", "Row Indices (first 100)"],
              [[p["duplicate_row_count"], " ".join(map(str, p["duplicate_row_indices"]))]])

    if (q := data("quality")) is not None:
        table("Quality: Dimensions", ["Dimension", "Percentage"],
              [[k.title(), v] for k, v in q["dimensions"].items()])
        table("Quality: Issues", ["Severity", "Column", "Type", "Count", "Message"],
              [[i["severity"], i["column"] or "(dataset)", i["type"], _blank(i["count"]), i["message"]]
               for i in q["issues"]])

    if (e := data("eda")) is not None:
        table("EDA: Numeric Columns",
              ["Column", "Count", "Missing", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max",
               "Skew", "Outliers", "Outlier %"],
              [[n["name"], n["count"], n["missing"], n["mean"], n["std"], n["min"], n["q1"],
                n["median"], n["q3"], n["max"], n["skew"], n["outlier_count"],
                n["outlier_percentage"]] for n in e["numeric_summary"]])
        table("EDA: Categorical Columns",
              ["Column", "Kind", "Count", "Missing", "Unique", "Top Values (value: count)"],
              [[c["name"], c["kind"], c["count"], c["missing"], c["unique"],
                "; ".join(f"{t['value']}: {t['count']}" for t in c["top_values"])]
               for c in e["categorical_summary"]])
        table("EDA: Datetime Columns", ["Column", "Count", "Missing", "Min", "Max", "Span (days)"],
              [[c["name"], c["count"], c["missing"], c["min"], c["max"], c["span_days"]]
               for c in e["datetime_summary"]])
        table("EDA: Strong Correlations", ["Feature A", "Feature B", "Pearson r"],
              [[c["feature_a"], c["feature_b"], c["correlation"]]
               for c in e["correlations"]["strong_pairs"]])

    if (fi := data("feature_insights")) is not None:
        table("Features: Recommendations",
              ["Feature", "Kind", "Role", "Missing %", "Unique", "Exclude", "Recommendations"],
              [[f["name"], f["kind"], f["role"], f["missing_percentage"], f["unique_count"],
                "yes" if f["exclude_from_modeling"] else "no", " | ".join(f["recommendations"])]
               for f in fi["features"]])
        table("Features: Redundant Pairs", ["Feature A", "Feature B", "Pearson r", "Suggestion"],
              [[r["feature_a"], r["feature_b"], r["correlation"], r["suggestion"]]
               for r in fi["redundant_pairs"]])
        ta = fi.get("target_analysis")
        if ta:
            table(f"Features: Association with Target '{ta['target']}'",
                  ["Feature", "Association", "Method", "Strength", "Possible Leakage"],
                  [[r["feature"], r["association"], r["method"], r["strength"],
                    "yes" if r["possible_leakage"] else "no"] for r in ta["rankings"]])


def _blank(v: Any) -> Any:
    return "" if v is None else v


# ======================================================================
# HTML
# ======================================================================

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--muted:#667085;--line:#e4e7ec;--accent:#2563eb;
--good:#16a34a;--warn:#d97706;--bad:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0f131a;--card:#181e29;--fg:#e6e9ef;--muted:#98a2b3;
--line:#2a3242;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:20px 0 8px}
.sub{color:var(--muted);margin-bottom:20px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:18px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px}
.card{flex:1 1 140px;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:22px}.card span{color:var(--muted);font-size:12px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;white-space:nowrap}
.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}
.high,.F,.error{background:var(--bad)}.medium,.D{background:var(--warn)}.low,.C{background:#ca8a04}
.B{background:#65a30d}.A,.ok{background:var(--good)}
.notice{padding:10px 12px;border-radius:8px;border:1px solid var(--bad);color:var(--bad)}
.spark{display:flex;align-items:flex-end;gap:1px;height:26px;min-width:70px}
.spark i{flex:1;background:var(--accent);min-height:1px;opacity:.85}
ul{margin:0;padding-left:20px}li{margin:3px 0}.muted{color:var(--muted)}
@media print{body{background:#fff}section{break-inside:avoid}}
"""


class _Raw(str):
    """Marks a string as already-safe HTML."""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-4 or abs(v) >= 1e15):
            return f"{v:.3g}"
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def _cell(v: Any) -> str:
    return str(v) if isinstance(v, _Raw) else _esc(_fmt(v))


def _table(headers: list[str], rows: list[list[Any]], numeric_cols: Iterable[int] = ()) -> str:
    if not rows:
        return '<p class="muted">Nothing to report.</p>'
    num = set(numeric_cols)
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{_cell(c)}</td>' if i in num else f"<td>{_cell(c)}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _badge(text: str, cls: str) -> _Raw:
    return _Raw(f'<span class="badge {_esc(cls)}">{_esc(text)}</span>')


def _spark(counts: list[int]) -> _Raw:
    top = max(counts) if counts else 0
    bars = "".join(
        f'<i style="height:{(c / top * 100) if top else 0:.0f}%" title="{c}"></i>' for c in counts
    )
    return _Raw(f'<div class="spark">{bars}</div>')


def _cards(pairs: list[tuple[str, Any]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><b>{_esc(_fmt(v))}</b><span>{_esc(k)}</span></div>' for k, v in pairs
    ) + "</div>"


def _render_html(d: dict) -> str:
    meta, summ, secs = d["metadata"], d["summary"], d["sections"]
    parts = [
        f"<h1>{_esc(meta['report_title'])}</h1>",
        f'<div class="sub">Source: {_esc(meta["source_file"])} · Generated {_esc(meta["generated_at"])}'
        + (f" · Target: {_esc(meta['target'])}" if meta["target"] else "")
        + "</div>",
        _html_summary(summ),
    ]
    renderers = {
        "profiling": _html_profiling,
        "quality": _html_quality,
        "eda": _html_eda,
        "feature_insights": _html_features,
    }
    for key in SECTION_ORDER:
        s = secs.get(key)
        if not s:
            continue
        if s["status"] == "ok":
            body = renderers[key](s["data"])
        else:
            body = f'<div class="notice">This section is unavailable ({_esc(s["status"])}): {_esc(s["error"])}</div>'
        parts.append(f'<section id="{key}"><h2>{_esc(s["title"])}</h2>{body}</section>')

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(meta['report_title'])}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )


def _html_summary(s: dict) -> str:
    cards = [("Rows", f"{s['rows']:,}"), ("Columns", s["columns"])]
    if s["quality_score"] is not None:
        cards += [("Quality score", f"{s['quality_score']} ({s['quality_grade']})")]
    if s["duplicate_rows"] is not None:
        cards += [("Duplicate rows", s["duplicate_rows"]),
                  ("Missing cells %", s["missing_cells_percentage"])]
    findings = "".join(f"<li>{_esc(f)}</li>" for f in s["key_findings"])
    return (
        '<section id="summary"><h2>Summary</h2>'
        + _cards(cards)
        + ("<h3>Key findings</h3><ul>" + findings + "</ul>" if findings else "")
        + "</section>"
    )


def _html_profiling(p: dict) -> str:
    out = _cards([("Rows", p["row_count"]), ("Columns", p["column_count"]),
                  ("Duplicate rows", p["duplicate_row_count"])])
    out += _table(
        ["Column", "Data type", "Inferred type", "Missing", "Missing %"],
        [[c["name"], c["dtype"], c["inferred_type"], c["missing_count"], c["missing_percentage"]]
         for c in p["columns"]],
        numeric_cols=(3, 4),
    )
    if p["duplicate_row_indices"]:
        more = " (first 100 shown)" if p["duplicate_row_indices_truncated"] else ""
        out += f"<h3>Duplicate row indices{more}</h3><p>{_esc(', '.join(map(str, p['duplicate_row_indices'])))}</p>"
    if p["loader_warnings"]:
        out += "<h3>Loader warnings</h3><ul>" + "".join(f"<li>{_esc(w)}</li>" for w in p["loader_warnings"]) + "</ul>"
    return out


def _html_quality(q: dict) -> str:
    dims = q["dimensions"]
    out = _cards([("Score", q["score"]), ("Completeness %", dims["completeness"]),
                  ("Uniqueness %", dims["uniqueness"]), ("Consistency %", dims["consistency"])])
    out += f"<p>Grade: {_badge(q['grade'], q['grade'])}</p><h3>Issues</h3>"
    out += _table(
        ["Severity", "Column", "Type", "Count", "Message"],
        [[_badge(i["severity"], i["severity"]), i["column"] or "(dataset)", i["type"], i["count"], i["message"]]
         for i in q["issues"]],
        numeric_cols=(3,),
    ) if q["issues"] else '<p class="muted">No issues detected.</p>'
    return out


def _heat(v: Any) -> str:
    if v is None:
        return '<td class="num">—</td>'
    rgb = "37,99,235" if v >= 0 else "220,38,38"
    color = "#fff" if abs(v) > 0.6 else "inherit"
    return f'<td class="num" style="background:rgba({rgb},{abs(v):.2f});color:{color}">{v:.2f}</td>'


def _html_eda(e: dict) -> str:
    out = ""
    if e["numeric_summary"]:
        out += "<h3>Numeric columns</h3>" + _table(
            ["Column", "Distribution", "Mean", "Std", "Min", "Median", "Max", "Skew", "Outliers"],
            [[n["name"], _spark(n["histogram"]["counts"]), n["mean"], n["std"], n["min"],
              n["median"], n["max"], n["skew"], f"{n['outlier_count']} ({n['outlier_percentage']}%)"]
             for n in e["numeric_summary"]],
            numeric_cols=range(2, 9),
        )
    if e["categorical_summary"]:
        out += "<h3>Categorical / text columns</h3>" + _table(
            ["Column", "Kind", "Unique", "Missing", "Top values"],
            [[c["name"], c["kind"], c["unique"], c["missing"],
              ", ".join(f"{t['value']} ({t['percentage']}%)" for t in c["top_values"][:5])]
             for c in e["categorical_summary"]],
            numeric_cols=(2, 3),
        )
    if e["datetime_summary"]:
        out += "<h3>Date columns</h3>" + _table(
            ["Column", "Earliest", "Latest", "Span (days)", "Missing"],
            [[c["name"], c["min"], c["max"], c["span_days"], c["missing"]] for c in e["datetime_summary"]],
            numeric_cols=(3, 4),
        )
    corr = e["correlations"]
    if corr["strong_pairs"]:
        out += f"<h3>Strong correlations (|r| ≥ {_esc(corr['min_abs_correlation'])})</h3>" + _table(
            ["Feature A", "Feature B", "Pearson r"],
            [[p["feature_a"], p["feature_b"], p["correlation"]] for p in corr["strong_pairs"]],
            numeric_cols=(2,),
        )
    m = corr["matrix"]
    if m:
        head = "".join(f"<th>{_esc(c)}</th>" for c in m["columns"])
        rows = "".join(
            f"<tr><th>{_esc(c)}</th>{''.join(_heat(v) for v in row)}</tr>"
            for c, row in zip(m["columns"], m["values"])
        )
        out += ('<h3>Correlation matrix</h3><div class="scroll"><table><thead><tr><th></th>'
                f"{head}</tr></thead><tbody>{rows}</tbody></table></div>")
    return out or '<p class="muted">No analysable columns.</p>'


def _html_features(f: dict) -> str:
    out = _table(
        ["Feature", "Kind", "Role", "Missing %", "Unique", "Recommendations"],
        [[x["name"], x["kind"], x["role"], x["missing_percentage"], x["unique_count"],
          _Raw("<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in x["recommendations"]) + "</ul>")]
         for x in f["features"]],
        numeric_cols=(3, 4),
    )
    if f["redundant_pairs"]:
        out += "<h3>Redundant feature pairs</h3>" + _table(
            ["Feature A", "Feature B", "Pearson r", "Suggestion"],
            [[r["feature_a"], r["feature_b"], r["correlation"], r["suggestion"]] for r in f["redundant_pairs"]],
            numeric_cols=(2,),
        )
    ta = f.get("target_analysis")
    if ta:
        out += f"<h3>Association with target '{_esc(ta['target'])}'</h3>"
        if ta.get("note"):
            out += f'<p class="muted">{_esc(ta["note"])}</p>'
        out += _table(
            ["Feature", "Association", "Method", "Strength", "Possible leakage"],
            [[r["feature"], r["association"], r["method"], r["strength"], r["possible_leakage"]]
             for r in ta["rankings"]],
            numeric_cols=(1,),
        )
    return out