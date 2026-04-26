"""Transformadores custom para sklearn pipeline."""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Remove colunas especificadas."""

    def __init__(self, columns: list[str]):
        """
        Args:
            columns: Lista de colunas para remover
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Remove colunas."""
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors='ignore')

    def get_feature_names_out(self, input_features=None):
        """Retorna nomes das features após transformação."""
        if input_features is None:
            return np.array([f for f in input_features if f not in self.columns])
        return np.array([f for f in input_features if f not in self.columns])


class TypeConverter(BaseEstimator, TransformerMixin):
    """Converte tipos de colunas."""

    def __init__(self, type_mapping: dict):
        """
        Args:
            type_mapping: Dict com nome da coluna -> tipo desejado
                Ex: {'tenure': 'float', 'monthly_charges': 'float'}
        """
        self.type_mapping = type_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Converte tipos."""
        X = X.copy()
        for col, dtype in self.type_mapping.items():
            if col in X.columns:
                try:
                    X[col] = X[col].astype(dtype)
                except (ValueError, TypeError):
                    print(f"[WARN] Não foi possível converter {col} para {dtype}")
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Codifica colunas binárias (Yes/No -> 1/0)."""

    def __init__(self, binary_columns: list[str] | None = None):
        """
        Args:
            binary_columns: Lista de colunas binárias. Se None, detecta automaticamente.
        """
        self.binary_columns = binary_columns
        self.binary_mapping = {}

    def fit(self, X, y=None):
        """Aprende mapping de valores binários."""
        if self.binary_columns is None:
            # Auto-detectar colunas binárias
            self.binary_columns = [
                col for col in X.columns
                if X[col].dtype == 'object' and X[col].nunique() == 2
            ]

        # Aprender mapeamento
        for col in self.binary_columns:
            unique_vals = sorted(X[col].unique())
            if len(unique_vals) == 2:
                # Assumir que a primeira é 0 e segunda é 1
                self.binary_mapping[col] = {unique_vals[0]: 0, unique_vals[1]: 1}

        return self

    def transform(self, X):
        """Codifica colunas binárias."""
        X = X.copy()
        for col, mapping in self.binary_mapping.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode colunas categóricas."""

    def __init__(self, categorical_columns: list[str] | None = None):
        """
        Args:
            categorical_columns: Lista de colunas categóricas
        """
        self.categorical_columns = categorical_columns
        self.categories_ = {}

    def fit(self, X, y=None):
        """Aprende categorias."""
        if self.categorical_columns is None:
            self.categorical_columns = [
                col for col in X.columns if X[col].dtype == 'object'
            ]

        # Armazenar categorias únicas
        for col in self.categorical_columns:
            self.categories_[col] = sorted(X[col].unique())

        return self

    def transform(self, X):
        """One-hot encode."""
        X = X.copy()
        for col in self.categorical_columns:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Seleciona subset de features."""

    def __init__(self, features: list[str]):
        """
        Args:
            features: Lista de features a manter
        """
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Seleciona features."""
        return X[[f for f in self.features if f in X.columns]]


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """Pipelines numéricos padrão."""

    def __init__(self,
                 binary_cols: list[str] | None = None,
                 categorical_cols: list[str] | None = None,
                 drop_cols: list[str] | None = None):
        self.binary_cols = binary_cols or []
        self.categorical_cols = categorical_cols or []
        self.drop_cols = drop_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Aplica todas as transformações."""
        # 1. Drop colunas indesejadas
        if self.drop_cols:
            X = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors='ignore')

        # 2. Codificar binárias
        if self.binary_cols:
            binary_encoder = BinaryEncoder(self.binary_cols)
            X = binary_encoder.fit_transform(X)

        # 3. One-hot encode categóricas
        if self.categorical_cols:
            cat_encoder = CategoricalEncoder(self.categorical_cols)
            X = cat_encoder.fit_transform(X)

        return X
