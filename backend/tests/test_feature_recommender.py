import pandas as pd
import pytest

from app.analysis.feature_recommender import FeatureRecommender


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "customer_id": list(range(1, 21)),           # identifier -> drop
            "signup_notes": [f"unique free text {i}" for i in range(20)],  # high-cardinality text -> drop
            "mostly_empty": [None] * 15 + [1] * 5,        # >50% missing -> drop
            "constant_flag": [1] * 20,                    # zero variance -> drop
            "region": ["North", "South", "East", "West"] * 5,       # low-cardinality categorical -> one-hot
            "is_active": [True, False] * 10,               # binary categorical
            "amount": [10, 12, 11, 13, 12, 11, 14, 10, 15, 9,
                       12, 11, 13, 12, 500, 10, 14, 11, 12, 13],  # numeric w/ outlier -> robust scale
            "score": [50, 52, 49, 51, 53, 48, 50, 52, 49, 51,
                      50, 52, 49, 51, 53, 48, 50, 52, 49, 51],  # symmetric, no outliers -> standard scale
            "churned": ["yes", "no"] * 10,  # categorical target
        }
    )


def test_drop_recommendations(df):
    recs = FeatureRecommender().recommend(df)
    dropped = {d.column for d in recs.drop_columns}

    assert "customer_id" in dropped
    assert "signup_notes" in dropped
    assert "mostly_empty" in dropped
    assert "constant_flag" in dropped
    assert "region" not in dropped
    assert "amount" not in dropped


def test_encode_recommendations_skip_dropped_columns(df):
    recs = FeatureRecommender().recommend(df)
    encoded = {e.column: e.strategy for e in recs.encode_columns}

    assert encoded["region"] == "one_hot"
    assert encoded["is_active"] == "binary"
    assert "constant_flag" not in encoded  # dropped, shouldn't be re-recommended


def test_scale_recommendations(df):
    recs = FeatureRecommender().recommend(df)
    scaled = {s.column: s.strategy for s in recs.scale_columns}

    assert scaled["amount"] == "robust"  # has an outlier
    assert scaled["score"] == "standard"  # roughly symmetric, low outliers


def test_importance_without_target_is_heuristic(df):
    recs = FeatureRecommender().recommend(df)

    assert recs.importance_is_heuristic is True
    methods = {f.method for f in recs.important_features}
    assert methods == {"coefficient_of_variation"}


def test_importance_with_numeric_target(df):
    recs = FeatureRecommender().recommend(df, target_column="amount")

    assert recs.importance_is_heuristic is False
    assert all(f.method == "abs_pearson_correlation_with_target" for f in recs.important_features)
    assert "amount" not in {f.column for f in recs.important_features}


def test_importance_with_categorical_target(df):
    recs = FeatureRecommender().recommend(df, target_column="churned")

    assert recs.importance_is_heuristic is False
    methods = {f.method for f in recs.important_features}
    assert methods == {"between_group_variance_ratio"}


def test_target_column_excluded_from_drops(df):
    recs = FeatureRecommender().recommend(df, target_column="constant_flag")
    dropped = {d.column for d in recs.drop_columns}

    assert "constant_flag" not in dropped


def test_unknown_target_column_raises(df):
    with pytest.raises(KeyError):
        FeatureRecommender().recommend(df, target_column="does_not_exist")


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        FeatureRecommender().recommend(pd.DataFrame())