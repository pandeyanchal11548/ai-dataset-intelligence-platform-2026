import pandas as pd
import pytest

from app.analysis.feature_correlation_analyzer import FeatureCorrelationAnalyzer


@pytest.fixture
def df():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return pd.DataFrame(
        {
            "x": x,
            "x_copy": x,  # exact duplicate of x
            "y": [v * 2 for v in x],  # perfectly correlated with x (and thus a near-duplicate)
            "noise": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],  # uncorrelated
            "label": ["a", "b"] * 5,  # non-numeric, excluded from correlation
        }
    )


def test_exact_duplicate_column_detected(df):
    report = FeatureCorrelationAnalyzer().analyze(df)
    exact = [r for r in report.redundant_columns if r.reason == "exact_duplicate"]

    assert len(exact) == 1
    assert exact[0].kept_column == "x"
    assert exact[0].dropped_column == "x_copy"


def test_near_duplicate_correlation_detected(df):
    report = FeatureCorrelationAnalyzer().analyze(df)
    near = [r for r in report.redundant_columns if r.reason == "near_duplicate_correlation"]

    # y is perfectly correlated with x; x_copy is already dropped as an exact
    # duplicate of x, so y should be recommended to drop against x (not x_copy).
    assert any(r.dropped_column == "y" and r.kept_column == "x" for r in near)


def test_recommended_drops_includes_both_kinds(df):
    report = FeatureCorrelationAnalyzer().analyze(df)

    assert "x_copy" in report.recommended_drops
    assert "y" in report.recommended_drops
    assert "noise" not in report.recommended_drops
    assert "x" not in report.recommended_drops  # kept, not dropped


def test_highly_correlated_pairs_include_perfect_pair(df):
    report = FeatureCorrelationAnalyzer().analyze(df)
    pair_columns = [{p.column_a, p.column_b} for p in report.highly_correlated_pairs]

    assert {"x", "y"} in pair_columns


def test_low_correlation_columns_not_flagged(df):
    report = FeatureCorrelationAnalyzer().analyze(df)
    pair_columns = [{p.column_a, p.column_b} for p in report.highly_correlated_pairs]

    assert {"x", "noise"} not in pair_columns
    assert not any("noise" in pair for pair in pair_columns)


def test_custom_thresholds_are_respected():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.2],  # strongly but not perfectly correlated
        }
    )
    strict = FeatureCorrelationAnalyzer(high_threshold=0.999, redundant_threshold=0.9999)
    report = strict.analyze(df)

    assert report.highly_correlated_pairs == []
    assert report.redundant_columns == []


def test_invalid_threshold_ordering_raises():
    with pytest.raises(ValueError):
        FeatureCorrelationAnalyzer(high_threshold=0.9, redundant_threshold=0.8)


def test_too_few_columns_raises():
    df = pd.DataFrame({"only_col": [1, 2, 3]})
    with pytest.raises(ValueError):
        FeatureCorrelationAnalyzer().analyze(df)


def test_no_numeric_columns_still_finds_exact_duplicates():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["x", "y", "z"]})
    report = FeatureCorrelationAnalyzer().analyze(df)

    assert report.recommended_drops == ["b"]
    assert report.highly_correlated_pairs == []