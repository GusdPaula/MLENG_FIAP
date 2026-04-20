"""Carregamento e préprocessamento de dados para Telco Churn."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, List


class TelcoDataLoader:
    """Carrega e processa dados do dataset Telco Churn."""

    def __init__(self, data_path: str):
        """
        Inicializa o loader.

        Args:
            data_path: Caminho para o arquivo CSV processado
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.le_dict = {}

    def carregar(self) -> pd.DataFrame:
        """Carrega o dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"[OK] Dataset carregado: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas")
        return self.df

    def preparar_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target.

        Target: Churn Value (0 = Não Churn, 1 = Churn)
        """
        # Remover colunas não relevantes
        drop_cols = [
            'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
            'Lat Long', 'Latitude', 'Longitude',  # Localização inútil
            'Churn Label',  # Usar Churn Value em vez disso
            'Churn Reason',  # Razão subjetiva
            'CLTV',  # Leakage - correlacionado com churn
            'Churn Score',  # Leakage - score de churn externo
        ]

        X = self.df.drop(columns=drop_cols + ['Churn Value'])
        y = self.df['Churn Value']

        print(f"[OK] Features selecionadas: {X.shape[1]}")
        print(f"  - Distribuicao de Churn: {y.value_counts().to_dict()}")
        print(f"  - Taxa de Churn: {y.mean()*100:.2f}%")

        return X, y

    def codificar_categoricas(self, X: pd.DataFrame, fit=True) -> pd.DataFrame:
        """
        Codifica variáveis categóricas com LabelEncoder.

        Args:
            X: DataFrame com features
            fit: Se True, treina LabelEncoder. Se False, aplica transformação.

        Returns:
            DataFrame com categorias codificadas
        """
        X_encoded = X.copy()

        categoricas = X_encoded.select_dtypes(include=['object']).columns

        for col in categoricas:
            if fit:
                self.le_dict[col] = LabelEncoder()
                X_encoded[col] = self.le_dict[col].fit_transform(X_encoded[col])
            else:
                X_encoded[col] = self.le_dict[col].transform(X_encoded[col])

        print(f"[OK] Variaveis categoricas codificadas: {len(categoricas)}")
        return X_encoded

    def normalizar_numericas(self, X: pd.DataFrame, fit=True) -> pd.DataFrame:
        """
        Normaliza variáveis numéricas com StandardScaler.

        Args:
            X: DataFrame com features
            fit: Se True, treina scaler. Se False, aplica transformação.

        Returns:
            DataFrame com features normalizadas
        """
        X_scaled = X.copy()

        numericas = X_scaled.select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            X_scaled[numericas] = self.scaler.fit_transform(X_scaled[numericas])
        else:
            X_scaled[numericas] = self.scaler.transform(X_scaled[numericas])

        print(f"[OK] Variaveis numericas normalizadas: {len(numericas)}")
        return X_scaled

    def split_treino_teste(self, X: pd.DataFrame, y: pd.Series,
                          test_size=0.2, random_state=42) -> None:
        """
        Divide dados em treino e teste.

        Usa stratified split para manter a proporção de churn.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"[OK] Split treino/teste (80/20):")
        print(f"  - Treino: {self.X_train.shape[0]} amostras")
        print(f"  - Teste: {self.X_test.shape[0]} amostras")
        print(f"  - Taxa churn treino: {self.y_train.mean()*100:.2f}%")
        print(f"  - Taxa churn teste: {self.y_test.mean()*100:.2f}%")

    def pipeline_completo(self, test_size=0.2, random_state=42) -> Tuple:
        """
        Executa pipeline completo: carregar -> processar -> dividir.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("PIPELINE DE PREPARACAO DE DADOS")
        print("="*60 + "\n")
        self.carregar()
        X, y = self.preparar_features_target()

        # 1. PRIMEIRO O SPLIT (Antes de qualquer transformação estatística)
        self.split_treino_teste(X, y, test_size, random_state)

        # 2. TRABALHAR NO TREINO (Fit e Transform)
        # Codificar
        self.X_train = self.codificar_categoricas(self.X_train, fit=True)
        # Imputar
        self.X_train = pd.DataFrame(
            self.imputer.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        # Normalizar
        self.X_train = self.normalizar_numericas(self.X_train, fit=True)

        # 3. TRABALHAR NO TESTE (APENAS Transform)
        # Usamos as réguas aprendidas no treino para aplicar no teste
        self.X_test = self.codificar_categoricas(self.X_test, fit=False)
        self.X_test = pd.DataFrame(
            self.imputer.transform(self.X_test),
            columns=self.X_test.columns
        )
        self.X_test = self.normalizar_numericas(self.X_test, fit=False)

        print("\n" + "="*60)
        return self.X_train, self.X_test, self.y_train, self.y_test