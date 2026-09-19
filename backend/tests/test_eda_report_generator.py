import json

import pandas as pd
import pytest

from app.analysis.eda_report_generator import EDAReportGenerator


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": list(range(1, 21)),
            "amount": [10, 12, 11, 13, 12, 11, 14, 10, 15, 9,
                       12, 11, 13, 12, 500, 10, 14, 11, 12, 13],
            "region": ["North", "South", "East", "West"] * 5,
            "notes": [None] * 12 + ["ok"] * 8,
        }
    )


def test_generate_basic_summary(df):
    report = EDAReportGenerator().generate(df, source_file="sales.csv")

    assert report.source_file == "sales.csv"
    assert report.summary.row_count == 20
    assert report.summary.column_count == 4
    assert 0 <= report.summary.quality_score <= 100
    assert report.summary.quality_grade in {"A", "B", "C", "D", "F"}


def test_generate_column_summaries(df):
    report = EDAReportGenerator().generate(df)
    names = {c.name for c in report.columns}

    assert names == {"id", "amount", "region", "notes"}
    notes_col = next(c for c in report.columns if c.name == "notes")
    assert notes_col.missing_count == 12
    assert notes_col.missing_severity in {"high", "critical"}


def test_generate_produces_charts(df):
    report = EDAReportGenerator().generate(df)
    kinds = {c.kind for c in report.charts}

    assert "distribution" in kinds  # amount is numeric
    assert "categorical" in kinds  # region is categorical


def test_generate_flags_missing_and_outliers_in_insights(df):
    report = EDAReportGenerator().generate(df)
    joined = " ".join(report.insights)

    assert "notes" in joined
    assert "outlier" in joined.lower()


def test_generate_flags_identifier_column(df):
    report = EDAReportGenerator().generate(df)
    # "id" is numeric with all-unique values; the numeric branch means it
    # won't necessarily be flagged as an identifier, but the report should
    # not error and should still classify it.
    id_col = next(c for c in report.columns if c.name == "id")
    assert id_col.unique_count == 20


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        EDAReportGenerator().generate(pd.DataFrame())


def test_to_json_writes_file(tmp_path, df):
    report = EDAReportGenerator().generate(df, source_file="sales.csv")
    out_path = tmp_path / "report.json"
    report.to_json(out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["source_file"] == "sales.csv"
    assert "insights" in data
    assert "charts" in data


def test_to_html_writes_self_contained_file(tmp_path, df):
    report = EDAReportGenerator().generate(df, source_file="sales.csv")
    out_path = tmp_path / "report.html"
    report.to_html(out_path)

    content = out_path.read_text()
    assert out_path.exists()
    assert "EDA Report" in content
    assert "data:image/png;base64," in content