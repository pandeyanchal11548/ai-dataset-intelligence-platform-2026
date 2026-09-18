import pandas as pd
import pytest

from app.analysis.correlation_analyzer import CorrelationAnalyzer


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [2, 4, 6, 8, 10, 12, 14, 16],  # perfectly correlated with x
            "z": [8, 7, 6, 5, 4, 3, 2, 1],  # perfectly anti-correlated with x
            "noise": [3, 1, 4, 1, 5, 9, 2, 6],
            "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def test_analyze_returns_matrix(df):
    report = CorrelationAnalyzer().analyze(df)

    assert "x" in report.columns
    assert "label" not in report.columns  # non-numeric excluded
    assert report.matrix["x"]["x"] == 1.0
    assert report.matrix["x"]["y"] == pytest.approx(1.0, abs=0.01)


def test_strong_positive_pair_detected(df):
    report = CorrelationAnalyzer().analyze(df)
    pair = next(
        p for p in report.notable_pairs if {p.column_a, p.column_b} == {"x", "y"}
    )

    assert pair.correlation > 0
    assert pair.strength == "strong"


def test_strong_negative_pair_detected(df):
    report = CorrelationAnalyzer().analyze(df)
    pair = next(
        p for p in report.notable_pairs if {p.column_a, p.column_b} == {"x", "z"}
    )

    assert pair.correlation < 0
    assert pair.strength == "strong"


def test_too_few_numeric_columns_raises():
    df = pd.DataFrame({"label": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        CorrelationAnalyzer().analyze(df)


def test_heatmap_returns_base64_image(df):
    heatmap = CorrelationAnalyzer().heatmap(df)

    assert len(heatmap.image_base64) > 100
    assert heatmap.as_data_uri().startswith("data:image/png;base64,")


def test_heatmap_saves_file(tmp_path, df):
    out_path = tmp_path / "heatmap.png"
    heatmap = CorrelationAnalyzer().heatmap(df, output_path=out_path)

    assert out_path.exists()
    assert heatmap.saved_path == str(out_path)


def test_specific_columns_only(df):
    report = CorrelationAnalyzer().analyze(df, columns=["x", "y"])

    assert set(report.columns) == {"x", "y"}