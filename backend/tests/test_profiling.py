import json

import pandas as pd
import pytest

from app.profiling.profiler import DatasetProfiler
from app.profiling.report_generator import ReportGenerator


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", None, "Alice"],
            "score": [91.5, 88.0, 76.2, 91.5],
        }
    )


@pytest.fixture
def duplicate_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 1],
            "name": ["Alice", "Bob", "Alice"],
        }
    )


def test_profile_basic_counts(sample_df):
    report = DatasetProfiler().profile(sample_df, source_file="sample.csv")
    assert report.row_count == 4
    assert report.column_count == 3
    assert report.data_types["id"] == "int64"
    assert report.missing_values["name"]["count"] == 1
    assert report.missing_values["name"]["percentage"] == 25.0


def test_profile_detects_duplicates(duplicate_df):
    report = DatasetProfiler().profile(duplicate_df, source_file="dupes.csv")
    assert report.duplicate_row_count == 1
    assert report.duplicate_row_indices == [2]


def test_profile_no_duplicates(sample_df):
    report = DatasetProfiler().profile(sample_df)
    assert report.duplicate_row_count == 0
    assert report.duplicate_row_indices == []


def test_report_to_json(tmp_path, sample_df):
    report = DatasetProfiler().profile(sample_df, source_file="sample.csv")
    out_path = tmp_path / "profile.json"
    ReportGenerator().to_json(report, out_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["row_count"] == 4
    assert data["column_count"] == 3
    assert "name" in data["missing_values"]


def test_report_to_csv(tmp_path, sample_df):
    report = DatasetProfiler().profile(sample_df, source_file="sample.csv")
    out_path = tmp_path / "profile.csv"
    ReportGenerator().to_csv(report, out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert "Row Count" in content
    assert "Column Name,Data Type,Missing Count,Missing Percentage" in content
    assert "name" in content