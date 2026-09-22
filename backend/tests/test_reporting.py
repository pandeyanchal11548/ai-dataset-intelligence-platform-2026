import csv
import io
import json

import numpy as np
import pandas as pd
import pytest

from app.loaders.dataset_loader import DatasetLoader
from app.reporting import ReportExporter, ReportStructuringEngine


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(50, 10, n)
    out = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": rng.integers(18, 70, n),
            "income": x * 1000,
            "income_copy": x * 1000 * 1.01,          # near-duplicate feature
            "region": rng.choice(["North", "South", "East"], n),
            "signup": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "notes": [None if i % 3 == 0 else f"note {i}" for i in range(n)],
            "constant": "same",
        }
    )
    out["spend"] = out["income"] * 0.1 + rng.normal(0, 50, n)
    return pd.concat([out, out.iloc[[0]]], ignore_index=True)  # one duplicate row


def test_engine_builds_all_sections(df):
    report = ReportStructuringEngine().build(df, source_file="demo.csv")
    d = report.to_dict()

    assert list(d["sections"]) == ["profiling", "quality", "eda", "feature_insights"]
    assert all(s["status"] == "ok" for s in d["sections"].values())
    assert d["summary"]["rows"] == len(df)
    assert d["summary"]["duplicate_rows"] == 1
    assert d["summary"]["quality_score"] is not None
    assert d["summary"]["key_findings"]

    fi = d["sections"]["feature_insights"]["data"]
    roles = {f["name"]: f["role"] for f in fi["features"]}
    assert roles["customer_id"] == "identifier"
    assert roles["constant"] == "constant"
    assert any({"income", "income_copy"} == {p["feature_a"], p["feature_b"]}
               for p in fi["redundant_pairs"])


def test_target_analysis_ranks_related_feature_first(df):
    report = ReportStructuringEngine().build(df, source_file="demo.csv", target="spend")
    ta = report.to_dict()["sections"]["feature_insights"]["data"]["target_analysis"]
    assert ta["target"] == "spend"
    assert ta["rankings"][0]["feature"] in {"income", "income_copy"}


def test_unknown_target_raises(df):
    with pytest.raises(ValueError):
        ReportStructuringEngine().build(df, target="nope")


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        ReportStructuringEngine().build(pd.DataFrame())


def test_failing_section_is_isolated(df):
    class Boom:
        def analyze(self, *a, **k):
            raise RuntimeError("kaboom")

    report = ReportStructuringEngine(eda=Boom()).build(df)
    d = report.to_dict()
    assert d["sections"]["eda"]["status"] == "error"
    assert "kaboom" in d["sections"]["eda"]["error"]
    assert d["sections"]["quality"]["status"] == "ok"
    # exporters still work with a failed section
    exp = ReportExporter()
    assert "kaboom" in exp.to_html(report, "/tmp/_x.html").read_text()


def test_uses_loader_inferred_types(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,day,amount\n" + "\n".join(f"{i},2024-01-{i:02d},{i * 1.5}" for i in range(1, 25)))
    df, prof = DatasetLoader().load(p)
    d = ReportStructuringEngine().build(df, source_file="s.csv", loader_profile=prof).to_dict()
    cols = {c["name"]: c["inferred_type"] for c in d["sections"]["profiling"]["data"]["columns"]}
    assert cols["day"] == "datetime"
    assert d["sections"]["eda"]["data"]["datetime_summary"][0]["span_days"] == 23


def test_json_export_is_strict_json(tmp_path, df):
    df.loc[0, "income"] = np.inf  # would break naive json.dump
    report = ReportStructuringEngine().build(df, source_file="demo.csv")
    path = ReportExporter().to_json(report, tmp_path / "r.json")
    data = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"bad constant {c}"))
    assert data["metadata"]["row_count"] == len(df)


def test_csv_export_has_sections(tmp_path, df):
    report = ReportStructuringEngine().build(df, source_file="demo.csv", target="spend")
    path = ReportExporter().to_csv(report, tmp_path / "r.csv")
    text = path.read_text(encoding="utf-8-sig")
    for heading in ("Key Findings", "Profiling: Columns", "Quality: Issues",
                    "EDA: Numeric Columns", "Features: Recommendations",
                    "Features: Association with Target 'spend'"):
        assert heading in text
    list(csv.reader(io.StringIO(text)))  # parses cleanly


def test_html_export_is_escaped_and_complete(tmp_path):
    evil = pd.DataFrame({"<script>alert(1)</script>": ["<b>x</b>", "y", "z", "x"], "n": [1, 2, 3, 4]})
    report = ReportStructuringEngine().build(evil, source_file="evil.csv")
    html_text = ReportExporter().to_html(report, tmp_path / "r.html").read_text()
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    for anchor in ('id="profiling"', 'id="quality"', 'id="eda"', 'id="feature_insights"'):
        assert anchor in html_text


def test_export_all_and_bad_format(tmp_path, df):
    report = ReportStructuringEngine().build(df, source_file="data/raw/demo.csv")
    paths = ReportExporter().export_all(report, tmp_path / "out")
    assert set(paths) == {"html", "json", "csv"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
    assert paths["json"].name == "demo_report.json"
    with pytest.raises(ValueError):
        ReportExporter().export_all(report, tmp_path, formats=["pdf"])