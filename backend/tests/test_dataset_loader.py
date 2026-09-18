import pandas as pd
import pytest

from app.loaders.dataset_loader import DatasetLoader
from app.loaders.exceptions import (
    EmptyDatasetError,
    FileValidationError,
    UnsupportedFileTypeError,
)


@pytest.fixture
def loader():
    return DatasetLoader()


def test_load_csv_basic(tmp_path, loader):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "id,name,signup_date,is_active,score\n"
        "1,Alice,2024-01-05,true,91.5\n"
        "2,Bob,2024-02-14,false,88\n"
        "3,Cara,2024-03-01,true,76.2\n"
    )

    df, profile = loader.load(csv_path)

    assert df.shape == (3, 5)
    types = {c.name: c.inferred_type for c in profile.columns}
    assert types["id"] == "integer"
    assert types["score"] == "float"
    assert types["is_active"] == "boolean"
    assert types["signup_date"] == "datetime"
    assert types["name"] == "categorical"


def test_unnamed_and_duplicate_columns_are_fixed(tmp_path, loader):
    csv_path = tmp_path / "messy.csv"
    csv_path.write_text("a,a,\n1,2,3\n4,5,6\n")

    df, profile = loader.load(csv_path)

    assert len(set(df.columns)) == len(df.columns)  # all unique
    assert any("renamed" in w or "auto-named" in w for w in profile.warnings)


def test_fully_empty_rows_and_columns_dropped(tmp_path, loader):
    csv_path = tmp_path / "with_blanks.csv"
    csv_path.write_text(
        "a,b,c\n1,,\n2,,\n,,\n3,,\n"  # column b/c fully empty, one row fully empty
    )

    df, profile = loader.load(csv_path)

    assert "b" not in df.columns or df["b"].isna().all() is False
    assert df.shape[0] <= 3


def test_unsupported_extension_raises(tmp_path, loader):
    bad_path = tmp_path / "data.json"
    bad_path.write_text("{}")

    with pytest.raises(UnsupportedFileTypeError):
        loader.load(bad_path)


def test_missing_file_raises(loader):
    with pytest.raises(FileValidationError):
        loader.load("does_not_exist.csv")


def test_empty_file_raises(tmp_path, loader):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")

    with pytest.raises(FileValidationError):
        loader.load(empty_path)


def test_excel_load(tmp_path, loader):
    xlsx_path = tmp_path / "sample.xlsx"
    df_in = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["A", "B", "A"],
            "value": [10.5, 20.1, 30.9],
        }
    )
    df_in.to_excel(xlsx_path, index=False)

    df, profile = loader.load(xlsx_path)

    assert df.shape == (3, 3)
    assert profile.file_type == "excel"
