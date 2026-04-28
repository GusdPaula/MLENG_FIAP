"""Preprocessamento de dados reprodutível para Telco Churn.

Orquesta todo o pipeline de preparação de dados:
  1. Carregamento de CSV
  2. Remoção de leakage + features irrelevantes
  3. Codificação de features binárias (Yes/No -> 1/0)
  4. Codificação de features categóricas (one-hot com drop_first)
  5. Divisão treino/teste estratificada
  6. Normalização com StandardScaler
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import set_all_seeds
set_all_seeds()


class TelcoDataPreprocessor:
    """Preprocessa dados de churn de telecom.

    Implementa todo o pipeline de preparação que transforma dados brutos
    em arrays numéricos padronizados prontos para ML.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
        self.binary_columns = None
        self.binary_mappings = {}  # Armazenar mappings de colunas binárias
        self.categorical_columns = None
        self._fitted_for_inference = False

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Carrega dataset CSV."""
        df = pd.read_csv(data_path)
        print(f"[OK] Dataset carregado: {df.shape[0]} linhas X {df.shape[1]} colunas")
        return df

    def drop_leakage_columns(
        self, df: pd.DataFrame, drop_columns: list | None = None
    ) -> pd.DataFrame:
        """Remove colunas com leakage (churn label, reason, score, CLTV) e localização."""
        if drop_columns is None:
            drop_columns = [
                "customerid",
                "count",
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

        cols_to_drop = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        print(f"[OK] {len(cols_to_drop)} colunas removidas (leakage/inúteis)")
        return df

    def extract_target(
        self, df: pd.DataFrame, target_col: str = "churn_value"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separa target (churn_value) das features (X, y)."""
        df.rename(columns={"Churn Value": "churn_value"}, inplace=True)
        if target_col not in df.columns:
            raise ValueError(f"Coluna '{target_col}' não encontrada")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        print(f"[OK] Target extraído: {y.value_counts().to_dict()}")
        return X, y

    def encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica colunas binárias (Yes/No) em 1/0."""
        df = df.copy()

        # Usar colunas binárias pré-detectadas (para inferência) ou detectar automaticamente
        if self.binary_columns is not None:
            binary_cols = self.binary_columns
        else:
            binary_cols = [
                col for col in df.columns if df[col].dtype == "object" and df[col].nunique() == 2
            ]

        # Mapeamento Yes/No -> 1/0
        for col in binary_cols:
            # Usar mapping armazenado se disponível, caso contrário criar novo
            if col in self.binary_mappings:
                mapping = self.binary_mappings[col]
                df[col] = df[col].map(mapping)
            else:
                unique_vals = df[col].unique()
                if len(unique_vals) >= 2:
                    # Ordenar para garantir consistência
                    sorted_vals = sorted(unique_vals)
                    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                    df[col] = df[col].map(mapping)
                else:
                    # Inferência com apenas 1 valor: mapear conforme aprendido
                    if col in self.binary_mappings:
                        mapping = self.binary_mappings[col]
                        df[col] = df[col].map(mapping)
                    else:
                        # Fallback: Yes -> 1, outros -> 0
                        val = unique_vals[0]
                        mapping = {val: 1 if val == "Yes" else 0}
                        df[col] = df[col].map(mapping)

        print(f"[OK] {len(binary_cols)} colunas binárias codificadas")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica one-hot encoding em features categóricas com drop_first=True."""
        df = df.copy()

        # Usar colunas categóricas pré-detectadas (para inferência) ou detectar automaticamente
        if self.categorical_columns is not None:
            categorical_cols = self.categorical_columns
        else:
            categorical_cols = [
                col for col in df.columns if df[col].dtype == "object" and df[col].nunique() > 2
            ]

        if categorical_cols:
            # One-hot encoding com drop_first=True para evitar multicolinearidade
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            print(f"[OK] {len(categorical_cols)} colunas categóricas codificadas (one-hot)")

        return df

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide dados em treino/teste com estratificação."""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,  # Garante proporções de classe iguais
        )

        print(f"[OK] Dados divididos: treino {X_train.shape[0]}, teste {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normaliza features com StandardScaler."""
        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Armazenar nomes de features
        self.feature_names = X_train.columns.tolist()

        print("[OK] Features normalizadas (mean=0, std=1)")
        return X_train_scaled, X_test_scaled

    def pipeline_completo(
        self, data_path: str, test_size: float = 0.2, drop_columns: list | None = None
    ) -> tuple:
        """
        Pipeline completo: load -> drop -> encode -> split -> scale.

        Returns:
            (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        # 1. Carregar
        df = self.load_data(data_path)

        # 2. Remover leakage
        df = self.drop_leakage_columns(df, drop_columns)

        df.rename(columns={"Churn Value": "churn_value"}, inplace=True)
        # 3. Extrair target
        X, y = self.extract_target(df)

        # 4. Codificar features
        X = self.encode_binary_features(X)
        X = self.encode_categorical_features(X)

        print(f"Features após codificação: {X.columns.tolist()}")

        # 5. Dividir dados
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)

        # 6. Normalizar
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("\n✅ Pipeline completo finalizado!")
        print(f"   Features: {X_train_scaled.shape[1]}")
        print(f"   Treino: {X_train_scaled.shape[0]} amostras")
        print(f"   Teste: {X_test_scaled.shape[0]} amostras")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def fit_for_inference(self, data_path: str) -> None:
        """Prepara preprocessador para uso em inferência.

        Carrega dataset, aplica encoding, e treina StandardScaler.
        Deve ser chamado uma vez na inicialização da API.

        Args:
            data_path: Caminho para o CSV processado
        """
        # 1. Carregar dados
        df = self.load_data(data_path)

        # 2. Remover leakage
        df = self.drop_leakage_columns(df)

        # 3. Extrair features (sem target)
        X, _ = self.extract_target(df)

        # 4. Detectar e armazenar colunas binárias e categóricas
        self.binary_columns = [
            col for col in X.columns if X[col].dtype == "object" and X[col].nunique() == 2
        ]
        self.categorical_columns = [
            col for col in X.columns if X[col].dtype == "object" and X[col].nunique() > 2
        ]

        # 5. Aprender os mappings das colunas binárias antes de codificar
        for col in self.binary_columns:
            unique_vals = X[col].unique()
            if "Yes" in unique_vals:
                self.binary_mappings[col] = {"Yes": 1, "No": 0}
            else:
                # Ordenar para garantir consistência
                sorted_vals = sorted(unique_vals)
                self.binary_mappings[col] = {sorted_vals[0]: 0, sorted_vals[1]: 1}

        # 6. Codificar features
        X = self.encode_binary_features(X)
        X = self.encode_categorical_features(X)

        # 7. Armazenar nomes das features
        self.feature_names = X.columns.tolist()

        # 8. Criar e treinar StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        # 9. Marcar como preparado para inferência
        self._fitted_for_inference = True

        print(
            f"[OK] Preprocessador preparado para inferência com {len(self.feature_names)} features"
        )

    def transform_single(self, features_dict: dict[str, str | int | float]) -> np.ndarray:
        """Transforma dicionário de features em array 30D normalizado.

        Args:
            features_dict: Dicionário com 19 features nomeadas

        Returns:
            Array numpy (1, 30) normalizado e pronto para predição

        Raises:
            ValueError: Se fit_for_inference() não foi chamado antes
        """
        if not self._fitted_for_inference:
            raise ValueError(
                "fit_for_inference() deve ser chamado antes de usar transform_single()"
            )

        # 1. Criar DataFrame de 1 linha
        df = pd.DataFrame([features_dict])

        # 2. Codificar
        df = self.encode_binary_features(df)
        df = self.encode_categorical_features(df)

        # 3. Reordenar colunas
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # 4. Normalizar
        X_scaled = self.scaler.transform(df)

        return X_scaled

    def transform_batch(self, features_list: list[dict[str, str | int | float]]) -> np.ndarray:
        """Transforma lista de dicionários em array 2D normalizado.

        Args:
            features_list: Lista de dicionários com features nomeadas

        Returns:
            Array numpy (n_samples, 30) normalizado

        Raises:
            ValueError: Se fit_for_inference() não foi chamado antes
        """
        if not self._fitted_for_inference:
            raise ValueError("fit_for_inference() deve ser chamado antes de usar transform_batch()")

        # 1. Criar DataFrame a partir da lista
        df = pd.DataFrame(features_list)

        # 2. Codificar
        df = self.encode_binary_features(df)
        df = self.encode_categorical_features(df)

        # 3. Reordenar colunas
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # 4. Normalizar
        X_scaled = self.scaler.transform(df)

        return X_scaled

    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverte normalização."""
        if self.scaler is None:
            raise ValueError("Scaler não foi ajustado. Execute pipeline_completo primeiro.")
        return self.scaler.inverse_transform(X_scaled)
