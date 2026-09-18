import pandas as pd
import pytest

from app.analysis.distribution_plotter import DistributionPlotter


@pytest.fixture
def df():
    return pd.DataFrame({"amount": [10, 12, 11, 13, 12, 11, 14, 10, 15, 9]})


def test_histogram_returns_base64_image(df):
    result = DistributionPlotter().histogram(df, "amount")

    assert result.kind == "histogram"
    assert len(result.image_base64) > 100
    assert result.stats.count == 10


def test_density_returns_base64_image(df):
    result = DistributionPlotter().density(df, "amount")

    assert result.kind == "density"
    assert len(result.image_base64) > 100


def test_combined_plot(df):
    result = DistributionPlotter().combined(df, "amount")

    assert result.kind == "combined"
    assert len(result.image_base64) > 100


def test_missing_column_raises(df):
    with pytest.raises(KeyError):
        DistributionPlotter().histogram(df, "nonexistent")


def test_non_numeric_column_raises():
    df = pd.DataFrame({"name": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        DistributionPlotter().histogram(df, "name")


def test_saves_file_to_disk(tmp_path, df):
    out_path = tmp_path / "hist.png"
    result = DistributionPlotter().histogram(df, "amount", output_path=out_path)

    assert out_path.exists()
    assert result.saved_path == str(out_path)


def test_data_uri_prefix(df):
    result = DistributionPlotter().histogram(df, "amount")

    assert result.as_data_uri().startswith("data:image/png;base64,")