"""Testes unitários para transformadores e preprocessing."""

import pytest
import numpy as np
import pandas as pd
from src.models.transformers import (
    ColumnDropper, BinaryEncoder, CategoricalEncoder, FeatureSelector
)
from src.data.preprocessing import TelcoDataPreprocessor


class TestColumnDropper:
    """Testes para ColumnDropper."""
    
    def test_drop_single_column(self):
        """Testa remoção de coluna única."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        dropper = ColumnDropper(['a'])
        result = dropper.transform(df)
        assert 'a' not in result.columns
        assert list(result.columns) == ['b', 'c']
    
    def test_drop_multiple_columns(self):
        """Testa remoção de múltiplas colunas."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        dropper = ColumnDropper(['a', 'c'])
        result = dropper.transform(df)
        assert list(result.columns) == ['b']
    
    def test_drop_nonexistent_column(self):
        """Testa que não erro ao dropar coluna inexistente."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        dropper = ColumnDropper(['nonexistent'])
        result = dropper.transform(df)
        assert result.equals(df)


class TestBinaryEncoder:
    """Testes para BinaryEncoder."""
    
    def test_encode_yes_no(self):
        """Testa codificação Yes/No."""
        df = pd.DataFrame({'internet': ['Yes', 'No', 'Yes', 'No']})
        encoder = BinaryEncoder(['internet'])
        encoder.fit(df)
        result = encoder.transform(df)
        assert result['internet'].dtype in [np.int64, np.int32, int]
        assert set(result['internet'].unique()) == {0, 1}
    
    def test_auto_detect_binary(self):
        """Testa auto-detecção de colunas binárias."""
        df = pd.DataFrame({
            'binary': ['A', 'B', 'A', 'B'],
            'non_binary': ['X', 'Y', 'Z', 'X'],
            'numeric': [1, 2, 3, 4]
        })
        encoder = BinaryEncoder()  # Sem especificar colunas
        encoder.fit(df)
        assert 'binary' in encoder.binary_columns


class TestCategoricalEncoder:
    """Testes para CategoricalEncoder."""
    
    def test_one_hot_encoding(self):
        """Testa one-hot encoding."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
        encoder = CategoricalEncoder(['color'])
        encoder.fit(df)
        result = encoder.transform(df)
        
        # Deve ter criado colunas dummy
        assert 'color' not in result.columns
        assert any('color_' in col for col in result.columns)
    
    def test_drop_first_prevents_multicollinearity(self):
        """Testa que drop_first=True previne multicolinearidade."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green']})
        encoder = CategoricalEncoder(['color'])
        encoder.fit(df)
        result = encoder.transform(df)
        
        # Deve ter n-1 colunas (drop_first=True)
        color_cols = [c for c in result.columns if 'color_' in c]
        assert len(color_cols) == 2  # 3 categorias - 1 = 2 colunas


class TestFeatureSelector:
    """Testes para FeatureSelector."""
    
    def test_select_features(self):
        """Testa seleção de features."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        selector = FeatureSelector(['a', 'c'])
        result = selector.transform(df)
        assert list(result.columns) == ['a', 'c']


class TestTelcoDataPreprocessor:
    """Testes para TelcoDataPreprocessor."""
    
    def test_load_data(self, sample_csv):
        """Testa carregamento de dados."""
        preprocessor = TelcoDataPreprocessor()
        df = preprocessor.load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
    
    def test_extract_target(self, sample_dataframe):
        """Testa extração de target."""
        preprocessor = TelcoDataPreprocessor()
        X, y = preprocessor.extract_target(sample_dataframe)
        
        assert 'Churn Value' not in X.columns
        assert len(X) == len(y)
        assert len(X.columns) == len(sample_dataframe.columns) - 1
    
    def test_encode_binary_features(self, sample_dataframe):
        """Testa codificação de features binárias."""
        preprocessor = TelcoDataPreprocessor()
        result = preprocessor.encode_binary_features(sample_dataframe)
        
        assert result['internet_service'].dtype in [np.int64, np.int32, int]
        assert set(result['internet_service'].unique()) == {0, 1}
    
    def test_split_data_stratification(self, sample_dataframe):
        """Testa que split mantém proporções de classe."""
        X, y = sample_dataframe.drop('Churn Value', axis=1), sample_dataframe['Churn Value']
        
        preprocessor = TelcoDataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.25)
        
        # Deve ter proporção similar de classes
        original_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        # Aceitar tolerância de 0.2 (20%) de desvio
        assert abs(original_ratio - train_ratio) < 0.2
        assert abs(original_ratio - test_ratio) < 0.2
    
    def test_scale_features(self, X_train_sample, X_test_sample):
        """Testa normalização de features."""
        preprocessor = TelcoDataPreprocessor()
        X_train_df = pd.DataFrame(X_train_sample, columns=['a', 'b', 'c'])
        X_test_df = pd.DataFrame(X_test_sample, columns=['a', 'b', 'c'])
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train_df, X_test_df)
        
        # Features deve ter média ~0 e std ~1
        assert np.abs(X_train_scaled.mean(axis=0)).max() < 1e-10
        assert np.abs(X_train_scaled.std(axis=0) - 1.0).max() < 0.1


class TestDataPreprocessorPipeline:
    """Testes de integração para pipeline de preprocessing."""
    
    def test_pipeline_returns_correct_shapes(self, sample_csv):
        """Testa que pipeline retorna shapes corretos."""
        preprocessor = TelcoDataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.pipeline_completo(
            sample_csv, test_size=0.4
        )
        
        assert X_train.shape[0] + X_test.shape[0] == 5  # Total de amostras
        assert X_train.shape[1] == X_test.shape[1]  # Mesmo número de features
        assert len(y_train) + len(y_test) == 5
    
    def test_pipeline_preserves_label_distribution(self, sample_csv):
        """Testa que pipeline preserva distribuição de labels."""
        preprocessor = TelcoDataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.pipeline_completo(
            sample_csv, test_size=0.4
        )
        
        # Deve ter positivos e negativos em ambos sets
        assert 0 in y_train and 1 in y_train
        assert 0 in y_test and 1 in y_test


class TestPreprocessorInference:
    """Testes para métodos de inferência do TelcoDataPreprocessor."""
    
    def test_fit_for_inference_creates_scaler(self, real_data_path):
        """Testa que fit_for_inference() cria scaler e feature_names."""
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference(real_data_path)
        
        assert preprocessor.scaler is not None
        assert preprocessor.feature_names is not None
        assert isinstance(preprocessor.feature_names, list)
        assert len(preprocessor.feature_names) == 30
        assert preprocessor._fitted_for_inference is True
    
    def test_fit_for_inference_detects_columns(self, real_data_path):
        """Testa que fit_for_inference() detecta colunas binárias e categóricas."""
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference(real_data_path)
        
        assert preprocessor.binary_columns is not None
        assert preprocessor.categorical_columns is not None
        assert len(preprocessor.binary_columns) > 0
        assert len(preprocessor.categorical_columns) > 0
    
    def test_transform_single_returns_correct_shape(self, real_data_path, sample_features_dict):
        """Testa que transform_single() retorna shape (1, 30)."""
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference(real_data_path)
        
        result = preprocessor.transform_single(sample_features_dict)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 30)
    
    def test_transform_single_without_fit_raises_error(self, sample_features_dict):
        """Testa que transform_single() sem fit_for_inference() levanta ValueError."""
        preprocessor = TelcoDataPreprocessor()
        
        with pytest.raises(ValueError, match="fit_for_inference"):
            preprocessor.transform_single(sample_features_dict)
    
    def test_transform_batch_returns_correct_shape(self, real_data_path, sample_features_dict):
        """Testa que transform_batch() retorna shape (n, 30)."""
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference(real_data_path)
        
        # Passar 3 dicionários
        features_list = [sample_features_dict] * 3
        result = preprocessor.transform_batch(features_list)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 30)
    
    def test_transform_batch_without_fit_raises_error(self, sample_features_dict):
        """Testa que transform_batch() sem fit_for_inference() levanta ValueError."""
        preprocessor = TelcoDataPreprocessor()
        features_list = [sample_features_dict]
        
        with pytest.raises(ValueError, match="fit_for_inference"):
            preprocessor.transform_batch(features_list)
    
    def test_transform_single_output_is_normalized(self, real_data_path, sample_features_dict):
        """Testa que output de transform_single() foi normalizado."""
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference(real_data_path)
        
        result = preprocessor.transform_single(sample_features_dict)
        
        # Não deve ser tudo zeros
        assert not np.allclose(result, 0)
        
        # Valores devem estar no range típico de dados normalizados (não valores brutos)
        # Esperamos valores menores em magnitude depois do StandardScaler
        assert np.max(np.abs(result)) < 100  # Valores típicos são muito menores
    
    def test_transform_consistency(self, real_data_path, sample_features_dict):
        """Testa que transform_single() é determinístico."""
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference(real_data_path)
        
        result1 = preprocessor.transform_single(sample_features_dict)
        result2 = preprocessor.transform_single(sample_features_dict)
        
        assert np.allclose(result1, result2)
    
    def test_transform_single_matches_pipeline(self, real_data_path, sample_features_dict):
        """Teste de integração: transform_single deve corresponder ao pipeline_completo.
        
        Este é o teste mais importante — garante que treinamento e inferência
        aplicam o mesmo preprocessamento.
        """
        # Usar pipeline_completo normalmente
        preprocessor1 = TelcoDataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor1.pipeline_completo(
            real_data_path, test_size=0.2
        )
        
        # Usar transform_single para um registro
        preprocessor2 = TelcoDataPreprocessor()
        preprocessor2.fit_for_inference(real_data_path)
        result = preprocessor2.transform_single(sample_features_dict)
        
        # Ambos devem ter 30 features
        assert X_train.shape[1] == 30
        assert result.shape == (1, 30)
        
        # Os nomes de features devem coincidir
        assert preprocessor1.feature_names == preprocessor2.feature_names

