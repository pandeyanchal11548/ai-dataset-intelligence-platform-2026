import pandas as pd
import pytest

from app.analysis.column_classifier import ColumnClassifier
from app.analysis.missing_value_analyzer import MissingValueAnalyzer
from app.analysis.duplicate_detector import DuplicateDetector


def test_classifier_numerical():
    df = pd.DataFrame({"age": [25, 30, 45, 22, 31]})
    report = ColumnClassifier().classify(df)
    assert report.to_dict()["age"] == "numerical"


def test_classifier_categorical():
    df = pd.DataFrame({"region": ["North", "South", "North", "East", "North"] * 4})
    report = ColumnClassifier().classify(df)
    assert report.to_dict()["region"] == "categorical"


def test_classifier_text():
    df = pd.DataFrame(
        {
            "comment": [
                "Great product, fast shipping",
                "Not what I expected at all",
                "Would buy again, very happy",
                "Terrible packaging but item was fine",
                "Excellent customer service experience",
            ]
        }
    )
    report = ColumnClassifier().classify(df)
    assert report.to_dict()["comment"] == "text"


def test_classifier_datetime():
    df = pd.DataFrame({"signup_date": ["2024-01-05", "2024-02-14", "2024-03-01"]})
    report = ColumnClassifier().classify(df)
    assert report.to_dict()["signup_date"] == "datetime"


def test_classifier_boolean_is_categorical():
    df = pd.DataFrame({"is_active": [True, False, True, True]})
    report = ColumnClassifier().classify(df)
    assert report.to_dict()["is_active"] == "categorical"


def test_missing_value_severity_buckets():
    df = pd.DataFrame(
        {
            "col_none": list(range(10)),
            "col_low": [1] * 9 + [None],
            "col_critical": [1] * 4 + [None] * 6,
        }
    )
    report = MissingValueAnalyzer().analyze(df)
    stats = {c.name: c for c in report.columns}

    assert stats["col_none"].severity == "none"
    assert stats["col_low"].null_percentage == 10.0
    assert stats["col_low"].severity == "moderate"
    assert stats["col_critical"].null_percentage == 60.0
    assert stats["col_critical"].severity == "critical"


def test_missing_value_worst_columns_ordering():
    df = pd.DataFrame(
        {
            "ok": [1, 2, 3, 4],
            "bad": [1, None, None, None],
        }
    )
    report = MissingValueAnalyzer().analyze(df)
    assert report.worst_columns == ["bad"]


def test_full_duplicate_detection():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Alice"],
        }
    )
    df_exact_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = DuplicateDetector().detect(df_exact_dupe)
    assert report.full_duplicate_count == 1
    assert report.full_duplicate_indices == [3]


def test_partial_duplicate_detection_with_explicit_subset():
    df = pd.DataFrame(
        {
            "order_id": [101, 102, 103],
            "customer": ["Alice", "Alice", "Bob"],
            "region": ["North", "North", "South"],
        }
    )
    report = DuplicateDetector().detect(df, partial_subset=["customer", "region"])

    assert report.partial_duplicate_subset == ["customer", "region"]
    assert report.partial_duplicate_count == 1
    assert report.partial_duplicate_indices == [1]


def test_partial_duplicate_auto_subset_excludes_unique_id():
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "customer": ["Alice", "Alice", "Bob"],
        }
    )
    report = DuplicateDetector().detect(df)

    assert "order_id" not in report.partial_duplicate_subset
    assert "customer" in report.partial_duplicate_subset
    assert report.partial_duplicate_count == 1