"""Preprocessamento de dados limpo e reprodutível."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class TelcoDataPreprocessor:
    """Preprocessa dados Telco Churn de forma reprodutível."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Carrega dados CSV."""
        df = pd.read_csv(data_path)
        print(f"[OK] Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
        return df
    
    def drop_leakage_columns(self, df: pd.DataFrame, 
                            drop_columns: Optional[list] = None) -> pd.DataFrame:
        """Remove colunas com leakage ou não relevantes."""
        if drop_columns is None:
            drop_columns = [
                'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
                'Lat Long', 'Latitude', 'Longitude',
                'Churn Label', 'Churn Reason',
                'CLTV', 'Churn Score',
            ]
        
        cols_to_drop = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        print(f"[OK] {len(cols_to_drop)} colunas removidas (leakage/inúteis)")
        return df
    
    def extract_target(self, df: pd.DataFrame, 
                      target_col: str = 'Churn Value') -> Tuple[pd.DataFrame, pd.Series]:
        """Extrai target e features."""
        if target_col not in df.columns:
            raise ValueError(f"Coluna '{target_col}' não encontrada")
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        print(f"[OK] Target extraído: {y.value_counts().to_dict()}")
        return X, y
    
    def encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica features binárias (Yes/No -> 1/0)."""
        df = df.copy()
        
        # Detectar colunas binárias
        binary_cols = [
            col for col in df.columns 
            if df[col].dtype == 'object' and df[col].nunique() == 2
        ]
        
        # Mapeamento Yes/No -> 1/0
        for col in binary_cols:
            unique_vals = df[col].unique()
            if 'Yes' in unique_vals:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
            else:
                # Alternativa: primeiro valor -> 0, segundo -> 1
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df[col] = df[col].map(mapping)
        
        print(f"[OK] {len(binary_cols)} colunas binárias codificadas")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode features categóricas."""
        df = df.copy()
        
        # Detectar colunas categóricas (excluindo numéricas e já codificadas)
        categorical_cols = [
            col for col in df.columns 
            if df[col].dtype == 'object' and df[col].nunique() > 2
        ]
        
        if categorical_cols:
            # One-hot encoding com drop_first=True para evitar multicolinearidade
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            print(f"[OK] {len(categorical_cols)} colunas categóricas codificadas (one-hot)")
        
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                     pd.Series, pd.Series]:
        """Divide dados em treino/teste com estratificação."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y  # Garante proporções de classe iguais
        )
        
        print(f"[OK] Dados divididos: treino {X_train.shape[0]}, teste {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, 
                      X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Normaliza features com StandardScaler."""
        self.scaler = StandardScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Armazenar nomes de features
        self.feature_names = X_train.columns.tolist()
        
        print(f"[OK] Features normalizadas (mean=0, std=1)")
        return X_train_scaled, X_test_scaled
    
    def pipeline_completo(self, data_path: str, 
                         test_size: float = 0.2,
                         drop_columns: Optional[list] = None) -> Tuple:
        """
        Pipeline completo: load -> drop -> encode -> split -> scale.
        
        Returns:
            (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        # 1. Carregar
        df = self.load_data(data_path)
        
        # 2. Remover leakage
        df = self.drop_leakage_columns(df, drop_columns)
        
        # 3. Extrair target
        X, y = self.extract_target(df)
        
        # 4. Codificar features
        X = self.encode_binary_features(X)
        X = self.encode_categorical_features(X)
        
        # 5. Dividir dados
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # 6. Normalizar
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"\n✅ Pipeline completo finalizado!")
        print(f"   Features: {X_train_scaled.shape[1]}")
        print(f"   Treino: {X_train_scaled.shape[0]} amostras")
        print(f"   Teste: {X_test_scaled.shape[0]} amostras")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverte normalização."""
        if self.scaler is None:
            raise ValueError("Scaler não foi ajustado. Execute pipeline_completo primeiro.")
        return self.scaler.inverse_transform(X_scaled)
