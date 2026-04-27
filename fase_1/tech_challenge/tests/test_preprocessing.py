"""Testes para TelcoDataPreprocessor.

Cobre:
  - Carregamento e remoção de colunas
  - Extração de target
  - Codificação binária e one-hot
  - Divisão treino/teste
  - Normalização
  - Pipeline completo
  - Inferência (fit_for_inference, transform_single, transform_batch)
  - Casos de erro
"""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import TelcoDataPreprocessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_df(n: int = 100) -> pd.DataFrame:
    """Cria um DataFrame mínimo que imita o CSV original."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "customerid": [f"id_{i}" for i in range(n)],
            "country": ["USA"] * n,
            "state": ["CA"] * n,
            "city": ["LA"] * n,
            "zip_code": rng.integers(10000, 99999, n).astype(str),
            "lat_long": ["0,0"] * n,
            "latitude": rng.uniform(-90, 90, n),
            "longitude": rng.uniform(-180, 180, n),
            "churn_label": rng.choice(["Yes", "No"], n),
            "churn_reason": ["price"] * n,
            "cltv": rng.integers(1000, 9000, n),
            "churn_score": rng.integers(0, 100, n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "total_charges": rng.uniform(100, 8000, n).round(2),
            "gender": rng.choice(["Male", "Female"], n),
            "partner": rng.choice(["Yes", "No"], n),
            "dependents": rng.choice(["Yes", "No"], n),
            "phone_service": rng.choice(["Yes", "No"], n),
            "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
            "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
            ),
            "Churn Value": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def raw_df():
    return _make_raw_df()


@pytest.fixture
def preprocessor():
    return TelcoDataPreprocessor(random_state=42)


@pytest.fixture
def features_df(raw_df, preprocessor):
    """DataFrame já sem leakage e sem target."""
    df = preprocessor.drop_leakage_columns(raw_df)
    X, _ = preprocessor.extract_target(df)
    return X


@pytest.fixture
def csv_path(tmp_path, raw_df):
    """Salva o DataFrame de teste como CSV temporário."""
    path = tmp_path / "telco_test.csv"
    raw_df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_returns_dataframe(self, preprocessor, csv_path):
        df = preprocessor.load_data(csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_shape_matches(self, preprocessor, csv_path, raw_df):
        df = preprocessor.load_data(csv_path)
        assert df.shape == raw_df.shape

    def test_missing_file_raises(self, preprocessor):
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/nonexistent/path/file.csv")


# ---------------------------------------------------------------------------
# drop_leakage_columns
# ---------------------------------------------------------------------------


class TestDropLeakageColumns:
    LEAKAGE_COLS: ClassVar[list[str]] = [
        "customerid",
        "country",
        "state",
        "city",
        "zip_code",
        "lat_long",
        "latitude",
        "longitude",
        "churn_label",
        "churn_reason",
        "cltv",
        "churn_score",
    ]

    def test_default_leakage_cols_removed(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        for col in self.LEAKAGE_COLS:
            assert col not in df.columns

    def test_non_leakage_cols_kept(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        assert "tenure_months" in df.columns
        assert "monthly_charges" in df.columns

    def test_custom_drop_columns(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df, drop_columns=["tenure_months"])
        assert "tenure_months" not in df.columns
        # customerid should NOT be dropped (not in custom list)
        assert "customerid" in df.columns

    def test_ignores_missing_columns(self, preprocessor, raw_df):
        # Should not raise even if a column in drop list doesn't exist
        df = preprocessor.drop_leakage_columns(raw_df, drop_columns=["nonexistent_col"])
        assert df.shape == raw_df.shape


# ---------------------------------------------------------------------------
# extract_target
# ---------------------------------------------------------------------------


class TestExtractTarget:
    def test_returns_X_and_y(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        X, y = preprocessor.extract_target(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        X, _ = preprocessor.extract_target(df)
        assert "churn_value" not in X.columns

    def test_y_values_are_binary(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        _, y = preprocessor.extract_target(df)
        assert set(y.unique()).issubset({0, 1})

    def test_missing_target_raises(self, preprocessor, raw_df):
        df = raw_df.drop(columns=["Churn Value"])
        df = preprocessor.drop_leakage_columns(df)
        with pytest.raises(ValueError, match="churn_value"):
            preprocessor.extract_target(df)


# ---------------------------------------------------------------------------
# encode_binary_features
# ---------------------------------------------------------------------------


class TestEncodeBinaryFeatures:
    def test_binary_cols_become_int(self, preprocessor, features_df):
        X = preprocessor.encode_binary_features(features_df)
        binary_cols = [
            c
            for c in features_df.columns
            if features_df[c].dtype == "object" and features_df[c].nunique() == 2
        ]
        for col in binary_cols:
            assert X[col].dtype in [np.int64, np.int32, int, np.bool_]

    def test_values_are_zero_or_one(self, preprocessor, features_df):
        X = preprocessor.encode_binary_features(features_df)
        binary_cols = [
            c
            for c in features_df.columns
            if features_df[c].dtype == "object" and features_df[c].nunique() == 2
        ]
        for col in binary_cols:
            assert set(X[col].dropna().unique()).issubset({0, 1})

    def test_does_not_modify_original(self, preprocessor, features_df):
        original_vals = features_df["gender"].copy()
        preprocessor.encode_binary_features(features_df)
        pd.testing.assert_series_equal(features_df["gender"], original_vals)

    def test_yes_maps_to_one(self, preprocessor):
        df = pd.DataFrame({"col": ["Yes", "No", "Yes"]})
        preprocessor.binary_mappings = {"col": {"Yes": 1, "No": 0}}
        preprocessor.binary_columns = ["col"]
        result = preprocessor.encode_binary_features(df)
        assert result.loc[0, "col"] == 1
        assert result.loc[1, "col"] == 0


# ---------------------------------------------------------------------------
# encode_categorical_features
# ---------------------------------------------------------------------------


class TestEncodeCategoricalFeatures:
    def test_object_cols_removed(self, preprocessor, features_df):
        X = preprocessor.encode_binary_features(features_df)
        X = preprocessor.encode_categorical_features(X)
        remaining_object_cols = [c for c in X.columns if X[c].dtype == "object"]
        assert remaining_object_cols == []

    def test_number_of_columns_increases(self, preprocessor, features_df):
        X_before = preprocessor.encode_binary_features(features_df)
        n_before = X_before.shape[1]
        X_after = preprocessor.encode_categorical_features(X_before)
        assert X_after.shape[1] >= n_before

    def test_drop_first_avoids_full_rank(self, preprocessor):
        """com drop_first=True, k categorias geram k-1 colunas."""
        df = pd.DataFrame({"cat": ["A", "B", "C", "A", "B"]})
        result = preprocessor.encode_categorical_features(df)
        # Esperado: 2 colunas (cat_B, cat_C) — A é descartado
        assert result.shape[1] == 2

    def test_encoded_values_are_numeric(self, preprocessor, features_df):
        """All columns must be numeric after encoding.

        Binary columns mapped via dict.map() return float64 when NaN is
        present (pandas upcasts to accommodate NaN in integer series).
        One-hot columns produced by get_dummies(dtype=int) are int.
        Both are acceptable — the key requirement is no object columns remain.
        """
        X = preprocessor.encode_binary_features(features_df)
        X = preprocessor.encode_categorical_features(X)
        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), (
                f"Column '{col}' has non-numeric dtype {X[col].dtype}"
            )


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    def test_correct_sizes(self, preprocessor, features_df, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        X, y = preprocessor.extract_target(df)
        X = preprocessor.encode_binary_features(X)
        X = preprocessor.encode_categorical_features(X)

        X_train, X_test, _y_train, _y_test = preprocessor.split_data(X, y, test_size=0.2)

        assert X_train.shape[0] + X_test.shape[0] == len(X)
        assert abs(X_test.shape[0] - round(0.2 * len(X))) <= 1

    def test_stratification_preserves_ratio(self, preprocessor, features_df, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        X, y = preprocessor.extract_target(df)
        X = preprocessor.encode_binary_features(X)
        X = preprocessor.encode_categorical_features(X)

        _, _, y_train, y_test = preprocessor.split_data(X, y)

        ratio_train = y_train.mean()
        ratio_test = y_test.mean()
        assert abs(ratio_train - ratio_test) < 0.05  # within 5%

    def test_reproducibility(self, features_df, raw_df):
        p1 = TelcoDataPreprocessor(random_state=99)
        p2 = TelcoDataPreprocessor(random_state=99)

        df1 = p1.drop_leakage_columns(raw_df.copy())
        X1, y1 = p1.extract_target(df1)
        X1 = p1.encode_binary_features(X1)
        X1 = p1.encode_categorical_features(X1)

        df2 = p2.drop_leakage_columns(raw_df.copy())
        X2, y2 = p2.extract_target(df2)
        X2 = p2.encode_binary_features(X2)
        X2 = p2.encode_categorical_features(X2)

        _, _, y_train1, _ = p1.split_data(X1, y1)
        _, _, y_train2, _ = p2.split_data(X2, y2)

        pd.testing.assert_series_equal(
            y_train1.reset_index(drop=True), y_train2.reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------


class TestScaleFeatures:
    @pytest.fixture
    def split_arrays(self, preprocessor, raw_df):
        df = preprocessor.drop_leakage_columns(raw_df)
        X, y = preprocessor.extract_target(df)
        X = preprocessor.encode_binary_features(X)
        X = preprocessor.encode_categorical_features(X)
        X_train, X_test, _, _ = preprocessor.split_data(X, y)
        return X_train, X_test

    def test_returns_numpy_arrays(self, preprocessor, split_arrays):
        X_train, X_test = split_arrays
        X_tr_sc, X_te_sc = preprocessor.scale_features(X_train, X_test)
        assert isinstance(X_tr_sc, np.ndarray)
        assert isinstance(X_te_sc, np.ndarray)

    def test_train_mean_near_zero(self, preprocessor, split_arrays):
        X_train, X_test = split_arrays
        X_tr_sc, _ = preprocessor.scale_features(X_train, X_test)
        assert np.abs(X_tr_sc.mean(axis=0)).max() < 1e-10

    def test_train_std_near_one(self, preprocessor, split_arrays):
        X_train, X_test = split_arrays
        X_tr_sc, _ = preprocessor.scale_features(X_train, X_test)
        # Columns with zero variance are edge cases; skip them
        stds = X_tr_sc.std(axis=0)
        nonzero = stds > 0
        assert np.allclose(stds[nonzero], 1.0, atol=1e-6)

    def test_scaler_stored(self, preprocessor, split_arrays):
        X_train, X_test = split_arrays
        preprocessor.scale_features(X_train, X_test)
        assert preprocessor.scaler is not None

    def test_feature_names_stored(self, preprocessor, split_arrays):
        X_train, X_test = split_arrays
        preprocessor.scale_features(X_train, X_test)
        assert preprocessor.feature_names == X_train.columns.tolist()


# ---------------------------------------------------------------------------
# pipeline_completo
# ---------------------------------------------------------------------------


class TestPipelineCompleto:
    def test_returns_four_items(self, preprocessor, csv_path):
        result = preprocessor.pipeline_completo(csv_path)
        assert len(result) == 4

    def test_shapes_consistent(self, preprocessor, csv_path):
        X_train, X_test, y_train, y_test = preprocessor.pipeline_completo(csv_path)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_outputs_are_numpy(self, preprocessor, csv_path):
        X_train, X_test, _y_train, _y_test = preprocessor.pipeline_completo(csv_path)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)

    def test_no_nan_in_output(self, preprocessor, csv_path):
        X_train, X_test, _, _ = preprocessor.pipeline_completo(csv_path)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()


# ---------------------------------------------------------------------------
# fit_for_inference / transform_single / transform_batch
# ---------------------------------------------------------------------------


class TestInferencePipeline:
    @pytest.fixture
    def fitted_preprocessor(self, csv_path):
        p = TelcoDataPreprocessor(random_state=42)
        p.fit_for_inference(csv_path)
        return p

    @pytest.fixture
    def sample_record(self):
        return {
            "tenure_months": 12,
            "monthly_charges": 65.5,
            "total_charges": 786.0,
            "gender": "Male",
            "partner": "Yes",
            "dependents": "No",
            "phone_service": "Yes",
            "internet_service": "Fiber optic",
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
        }

    def test_fitted_flag_set(self, fitted_preprocessor):
        assert fitted_preprocessor._fitted_for_inference is True

    def test_transform_single_returns_2d_array(self, fitted_preprocessor, sample_record):
        result = fitted_preprocessor.transform_single(sample_record)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_transform_single_feature_count_matches(self, fitted_preprocessor, sample_record):
        result = fitted_preprocessor.transform_single(sample_record)
        assert result.shape[1] == len(fitted_preprocessor.feature_names)

    def test_transform_batch_shape(self, fitted_preprocessor, sample_record):
        batch = [sample_record, sample_record, sample_record]
        result = fitted_preprocessor.transform_batch(batch)
        assert result.shape == (3, len(fitted_preprocessor.feature_names))

    def test_transform_single_without_fit_raises(self, sample_record):
        p = TelcoDataPreprocessor()
        with pytest.raises(ValueError, match="fit_for_inference"):
            p.transform_single(sample_record)

    def test_transform_batch_without_fit_raises(self, sample_record):
        p = TelcoDataPreprocessor()
        with pytest.raises(ValueError, match="fit_for_inference"):
            p.transform_batch([sample_record])

    def test_no_nan_in_single_output(self, fitted_preprocessor, sample_record):
        result = fitted_preprocessor.transform_single(sample_record)
        assert not np.isnan(result).any()

    def test_missing_columns_filled_with_zero(self, fitted_preprocessor):
        """Colunas one-hot ausentes no input devem virar 0 via reindex fill_value=0.

        The preprocessor iterates binary_mappings only for columns that are
        actually present in the DataFrame, so passing a record that contains
        all binary/categorical keys (even with a single known value) lets the
        encode step succeed and reindex fills any remaining one-hot columns.
        """
        # Provide all required raw columns; only numeric values differ.
        # Unknown/unseen categorical values simply produce no one-hot column,
        # which reindex then fills with 0.
        minimal = {
            "tenure_months": 6,
            "monthly_charges": 50.0,
            "total_charges": 300.0,
            "gender": "Male",
            "partner": "No",
            "dependents": "No",
            "phone_service": "No",
            "internet_service": "No",  # valid category that has fewest dummies
            "contract": "Month-to-month",
            "payment_method": "Mailed check",
        }
        result = fitted_preprocessor.transform_single(minimal)
        assert result.shape[1] == len(fitted_preprocessor.feature_names)


# ---------------------------------------------------------------------------
# inverse_transform_features
# ---------------------------------------------------------------------------


class TestInverseTransform:
    def test_roundtrip(self, preprocessor, csv_path):
        X_train, _, _, _ = preprocessor.pipeline_completo(csv_path)
        X_back = preprocessor.inverse_transform_features(X_train)
        # Shape preserved
        assert X_back.shape == X_train.shape

    def test_raises_without_scaler(self, preprocessor):
        with pytest.raises(ValueError, match="Scaler"):
            preprocessor.inverse_transform_features(np.zeros((5, 3)))
