import pandas as pd
import pytest

from app.analysis.ml_readiness_scorer import MLReadinessScorer


def test_clean_dataset_scores_high():
    df = pd.DataFrame(
        {
            "feature_a": list(range(1, 51)),
            "feature_b": [v * 1.5 + 2 for v in range(1, 51)],
            "feature_c": ["group_a", "group_b"] * 25,
            "target": [0, 1] * 25,
        }
    )
    report = MLReadinessScorer().score(df, target_column="target")

    assert report.overall_score >= 70
    assert report.readiness_level in {"Ready for ML", "Needs Minor Cleanup"}
    assert report.row_count == 50
    assert report.column_count == 4
    assert len(report.dimensions) == 3


def test_messy_dataset_scores_low():
    df = pd.DataFrame(
        {
            "customer_id": list(range(1, 9)),
            "constant_col": [1] * 8,
            "mostly_missing": [None] * 7 + [1],
            "dup_of_id": list(range(1, 9)),  # exact duplicate column of customer_id
            "amount": [10, 12, 11, 13, 12, 5000, 14, 10],
        }
    )
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)  # duplicate rows

    report = MLReadinessScorer().score(df)

    assert report.overall_score < 70
    assert report.readiness_level in {"Needs Significant Cleanup", "Not Ready"}


def test_dimension_names_and_weights_sum_to_one():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    report = MLReadinessScorer().score(df)
    names = {d.name for d in report.dimensions}

    assert names == {"completeness", "feature_quality", "noise_level"}
    assert abs(sum(d.weight for d in report.dimensions) - 1.0) < 0.001


def test_recommendations_flag_duplicates_and_outliers():
    df = pd.DataFrame(
        {
            "amount": [10, 12, 11, 13, 12, 11, 14, 10, 5000, 10],
            "region": ["N", "S", "E", "W", "N", "S", "E", "W", "N", "S"],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = MLReadinessScorer().score(df)
    joined = " ".join(report.recommendations)

    assert "duplicate" in joined.lower()
    assert "outlier" in joined.lower()


def test_no_issues_gives_positive_recommendation():
    df = pd.DataFrame(
        {
            "a": list(range(1, 101)),
            "b": [v + 0.5 for v in range(1, 101)],
        }
    )
    report = MLReadinessScorer().score(df)

    if report.overall_score >= 90:
        assert any("ready for ml" in r.lower() for r in report.recommendations)


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        MLReadinessScorer().score(pd.DataFrame())


def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        MLReadinessScorer(weights={"completeness": 0.5, "feature_quality": 0.6, "noise_level": 0.1})


def test_unknown_target_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(KeyError):
        MLReadinessScorer().score(df, target_column="nope")


def test_readiness_level_boundaries():
    scorer = MLReadinessScorer()
    assert scorer._readiness_level(90) == "Ready for ML"
    assert scorer._readiness_level(75) == "Needs Minor Cleanup"
    assert scorer._readiness_level(55) == "Needs Significant Cleanup"
    assert scorer._readiness_level(30) == "Not Ready"