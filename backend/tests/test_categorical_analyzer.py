import pandas as pd
import pytest

from app.analysis.categorical_analyzer import CategoricalAnalyzer


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "East", "North", "South", None],
            "amount": [10, 20, 30, 40, 50, 60, 70],
        }
    )


def test_value_counts_basic(df):
    summary = CategoricalAnalyzer().value_counts(df, "region")

    assert summary.column == "region"
    assert summary.total_count == 7
    assert summary.missing_count == 1
    assert summary.unique_count == 3
    top = summary.top_category
    assert top.category == "North"
    assert top.count == 3


def test_value_counts_percentages_exclude_missing(df):
    summary = CategoricalAnalyzer().value_counts(df, "region")
    total_pct = sum(c.percentage for c in summary.categories)

    assert total_pct == pytest.approx((6 / 7) * 100, abs=0.5)


def test_other_bucket_created_for_long_tail():
    df = pd.DataFrame({"cat": [f"item_{i}" for i in range(20)]})
    summary = CategoricalAnalyzer(default_top_n=5).value_counts(df, "cat")

    categories = [c.category for c in summary.categories]
    assert "Other" in categories
    assert len(categories) == 6  # top 5 + Other


def test_missing_column_raises(df):
    with pytest.raises(KeyError):
        CategoricalAnalyzer().value_counts(df, "nonexistent")


def test_distribution_plot_returns_base64_image(df):
    plot = CategoricalAnalyzer().distribution_plot(df, "region")

    assert len(plot.image_base64) > 100
    assert plot.as_data_uri().startswith("data:image/png;base64,")


def test_distribution_plot_saves_file(tmp_path, df):
    out_path = tmp_path / "dist.png"
    plot = CategoricalAnalyzer().distribution_plot(df, "region", output_path=out_path)

    assert out_path.exists()
    assert plot.saved_path == str(out_path)


def test_analyze_all_auto_detects_categorical_columns(df):
    summaries = CategoricalAnalyzer().analyze_all(df)
    columns_found = [s.column for s in summaries]

    assert "region" in columns_found
    assert "amount" not in columns_found  # numeric, should be excluded


def test_all_null_column_returns_empty_summary():
    df = pd.DataFrame({"empty": [None, None, None]})
    summary = CategoricalAnalyzer().value_counts(df, "empty")

    assert summary.categories == []
    assert summary.unique_count == 0