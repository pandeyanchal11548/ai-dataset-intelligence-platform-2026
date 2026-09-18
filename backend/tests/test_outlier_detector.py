import pandas as pd
import pytest

from app.analysis.outlier_detector import OutlierDetector


@pytest.fixture
def df_with_outliers():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "amount": [10, 12, 11, 13, 12, 11, 500, 10],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )


def test_iqr_detects_obvious_outlier(df_with_outliers):
    report = OutlierDetector().detect(df_with_outliers, method="iqr")
    amount_stats = next(c for c in report.columns if c.name == "amount")

    assert amount_stats.outlier_count == 1
    assert 6 in amount_stats.outlier_indices
    assert amount_stats.lower_bound is not None
    assert amount_stats.upper_bound is not None


def test_zscore_detects_obvious_outlier(df_with_outliers):
    detector = OutlierDetector(zscore_threshold=1.5)
    report = detector.detect(df_with_outliers, method="zscore")
    amount_stats = next(c for c in report.columns if c.name == "amount")

    assert amount_stats.outlier_count >= 1
    assert 6 in amount_stats.outlier_indices


def test_non_numeric_columns_are_skipped(df_with_outliers):
    report = OutlierDetector().detect(df_with_outliers, method="iqr")
    names = [c.name for c in report.columns]

    assert "category" not in names
    assert "amount" in names


def test_specific_columns_only(df_with_outliers):
    report = OutlierDetector().detect(
        df_with_outliers, method="iqr", columns=["amount"]
    )

    assert len(report.columns) == 1
    assert report.columns[0].name == "amount"


def test_unknown_method_raises(df_with_outliers):
    with pytest.raises(ValueError):
        OutlierDetector().detect(df_with_outliers, method="bogus")


def test_zero_variance_column_has_no_outliers():
    df = pd.DataFrame({"constant": [5, 5, 5, 5, 5]})
    report = OutlierDetector().detect(df, method="zscore")

    assert report.columns[0].outlier_count == 0


def test_total_outlier_rows_and_indices(df_with_outliers):
    report = OutlierDetector().detect(df_with_outliers, method="iqr")

    assert report.total_outlier_rows == 1
    assert report.all_outlier_indices == [6]


def test_report_to_dict_shape(df_with_outliers):
    report = OutlierDetector().detect(
        df_with_outliers, method="iqr", source_file="sales.csv"
    )
    data = report.to_dict()

    assert data["source_file"] == "sales.csv"
    assert data["method"] == "iqr"
    assert "total_outlier_rows" in data
    assert isinstance(data["columns"], list)