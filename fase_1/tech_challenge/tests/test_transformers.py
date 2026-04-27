"""Testes unitários para transformadores e preprocessing."""

import numpy as np
import pandas as pd

from src.models.transformers import (
    BinaryEncoder,
    CategoricalEncoder,
    ColumnDropper,
    FeatureSelector,
    NumericalTransformer,
    TypeConverter,
)


# -----------------------------
# ColumnDropper
# -----------------------------
class TestColumnDropper:
    """Testes para ColumnDropper."""

    def test_drop_single_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        dropper = ColumnDropper(["a"])
        result = dropper.transform(df)
        assert "a" not in result.columns
        assert list(result.columns) == ["b", "c"]

    def test_drop_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        dropper = ColumnDropper(["a", "c"])
        result = dropper.transform(df)
        assert list(result.columns) == ["b"]

    def test_drop_nonexistent_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dropper = ColumnDropper(["nonexistent"])
        result = dropper.transform(df)
        assert result.equals(df)

    def test_get_feature_names(self):
        dropper = ColumnDropper(["b"])
        features = ["a", "b", "c"]
        result = dropper.get_feature_names_out(features)
        assert list(result) == ["a", "c"]


# -----------------------------
# TypeConverter
# -----------------------------
class TestTypeConverter:
    """Testes para TypeConverter."""

    def test_type_conversion_success(self):
        df = pd.DataFrame({"a": ["1", "2"]})
        converter = TypeConverter({"a": "int"})
        result = converter.transform(df)
        assert result["a"].dtype == "int64"

    def test_type_conversion_failure(self, capfd):
        df = pd.DataFrame({"a": ["x", "y"]})
        converter = TypeConverter({"a": "int"})
        result = converter.transform(df)

        captured = capfd.readouterr()
        assert "[WARN]" in captured.out
        assert result["a"].dtype == object


# -----------------------------
# BinaryEncoder
# -----------------------------
class TestBinaryEncoder:
    """Testes para BinaryEncoder."""

    def test_encode_yes_no(self):
        df = pd.DataFrame({"internet": ["Yes", "No", "Yes", "No"]})
        encoder = BinaryEncoder(["internet"])
        encoder.fit(df)
        result = encoder.transform(df)
        assert result["internet"].dtype in [np.int64, np.int32, int]
        assert set(result["internet"].unique()) == {0, 1}

    def test_auto_detect_binary(self):
        df = pd.DataFrame(
            {
                "binary": ["A", "B", "A", "B"],
                "non_binary": ["X", "Y", "Z", "X"],
                "numeric": [1, 2, 3, 4],
            }
        )
        encoder = BinaryEncoder()
        encoder.fit(df)
        assert "binary" in encoder.binary_columns

    def test_mapping_consistency(self):
        df = pd.DataFrame({"flag": ["No", "Yes"]})
        encoder = BinaryEncoder(["flag"])
        encoder.fit(df)
        result = encoder.transform(df)
        assert list(result["flag"]) == [0, 1]


# -----------------------------
# CategoricalEncoder
# -----------------------------
class TestCategoricalEncoder:
    """Testes para CategoricalEncoder."""

    def test_one_hot_encoding(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
        encoder = CategoricalEncoder(["color"])
        encoder.fit(df)
        result = encoder.transform(df)

        assert "color" not in result.columns
        assert any("color_" in col for col in result.columns)

    def test_drop_first_prevents_multicollinearity(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        encoder = CategoricalEncoder(["color"])
        encoder.fit(df)
        result = encoder.transform(df)

        color_cols = [c for c in result.columns if "color_" in c]
        assert len(color_cols) == 2  # n-1

    def test_auto_detect(self):
        df = pd.DataFrame({"color": ["red", "blue"]})
        encoder = CategoricalEncoder()
        encoder.fit(df)
        assert "color" in encoder.categorical_columns


# -----------------------------
# FeatureSelector
# -----------------------------
class TestFeatureSelector:
    """Testes para FeatureSelector."""

    def test_select_features(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        selector = FeatureSelector(["a", "c"])
        result = selector.transform(df)
        assert list(result.columns) == ["a", "c"]

    def test_ignore_missing_features(self):
        df = pd.DataFrame({"a": [1, 2]})
        selector = FeatureSelector(["a", "b"])
        result = selector.transform(df)
        assert list(result.columns) == ["a"]


# -----------------------------
# NumericalTransformer
# -----------------------------
class TestNumericalTransformer:
    """Testes para NumericalTransformer."""

    def test_full_transformation_flow(self):
        df = pd.DataFrame(
            {
                "drop_me": [1, 2],
                "binary": ["No", "Yes"],
                "category": ["A", "B"],
                "value": [10, 20],
            }
        )

        transformer = NumericalTransformer(
            binary_cols=["binary"],
            categorical_cols=["category"],
            drop_cols=["drop_me"],
        )

        result = transformer.transform(df)

        # Drop
        assert "drop_me" not in result.columns

        # Binary
        assert set(result["binary"].unique()) <= {0, 1}

        # Categorical
        assert "category" not in result.columns
        assert any(col.startswith("category_") for col in result.columns)

    def test_no_transformation(self):
        df = pd.DataFrame({"a": [1, 2]})
        transformer = NumericalTransformer()
        result = transformer.transform(df)

        assert result.equals(df)
