import pandas as pd
import pytest

from app.analysis.quality_scorer import DataQualityScorer


def test_clean_dataset_scores_high():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "amount": [10, 12, 11, 13, 12, 11, 14, 10],
        }
    )
    report = DataQualityScorer().score(df)

    assert report.overall_score >= 90
    assert report.grade == "A"
    assert report.row_count == 8
    assert report.column_count == 2


def test_missing_values_lower_completeness():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "value": [10, None, None, 40]})
    report = DataQualityScorer().score(df)
    completeness = next(d for d in report.dimensions if d.name == "completeness")

    assert completeness.score < 100


def test_duplicates_lower_uniqueness():
    df = pd.DataFrame({"id": [1, 1, 2, 3], "name": ["A", "A", "B", "C"]})
    report = DataQualityScorer().score(df)
    uniqueness = next(d for d in report.dimensions if d.name == "uniqueness")

    assert uniqueness.score < 100


def test_outliers_lower_validity():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "amount": [10, 12, 11, 13, 12, 11, 500, 10],
        }
    )
    report = DataQualityScorer().score(df)
    validity = next(d for d in report.dimensions if d.name == "validity")

    assert validity.score < 100


def test_mixed_type_column_lowers_consistency():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "amount": [10, 12, 11, "oops", 12, 11],
        }
    )
    report = DataQualityScorer().score(df)
    consistency = next(d for d in report.dimensions if d.name == "consistency")

    assert consistency.score < 100


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        DataQualityScorer().score(pd.DataFrame())


def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        DataQualityScorer(
            weights={
                "completeness": 0.5,
                "uniqueness": 0.6,
                "validity": 0.1,
                "consistency": 0.1,
            }
        )


def test_grade_boundaries():
    scorer = DataQualityScorer()
    assert scorer._grade(95) == "A"
    assert scorer._grade(85) == "B"
    assert scorer._grade(75) == "C"
    assert scorer._grade(65) == "D"
    assert scorer._grade(50) == "F"


def test_warnings_populated_for_poor_dataset():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "value": [10, None, None, None],
        }
    )
    report = DataQualityScorer().score(df)

    assert len(report.warnings) > 0