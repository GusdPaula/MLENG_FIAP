"""Testes unitários para transformadores e preprocessing."""

import pytest
import numpy as np
import pandas as pd
from src.models.transformers import (
    ColumnDropper, BinaryEncoder, CategoricalEncoder, FeatureSelector
)


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

