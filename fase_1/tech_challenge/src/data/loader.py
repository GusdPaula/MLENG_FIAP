"""Carregamento e préprocessamento de dados para Telco Churn."""


from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
        self.feature_names = None
        self._fitted_for_inference = False
        self.categorical_values = {}  # Armazenar valores válidos para cada coluna categórica

    def carregar(self) -> pd.DataFrame:
        """Carrega o dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"[OK] Dataset carregado: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas")
        return self.df

    def preparar_features_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target.

        Target: churn_value (0 = Não Churn, 1 = Churn)
        """
        # Remover colunas não relevantes
        drop_cols = [
            'customerid', 'count', 'country', 'state', 'city', 'zip_code',
            'lat_long', 'latitude', 'longitude',  # localização inútil
            'churn_label',  # usar churn_value em vez disso
            'churn_reason',  # razão subjetiva
            'cltv',  # leakage - correlacionado com churn
            'churn_score',  # leakage - score de churn externo
        ]
        self.df.rename(columns={'Churn Value': 'churn_value'}, inplace=True)
        X = self.df.drop(columns=[*drop_cols, 'churn_value'])
        y = self.df['churn_value']

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

        print("[OK] Split treino/teste (80/20):")
        print(f"  - Treino: {self.X_train.shape[0]} amostras")
        print(f"  - Teste: {self.X_test.shape[0]} amostras")
        print(f"  - Taxa churn treino: {self.y_train.mean()*100:.2f}%")
        print(f"  - Taxa churn teste: {self.y_test.mean()*100:.2f}%")

    def pipeline_completo(self, test_size=0.2, random_state=42) -> tuple:
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

    def fit_for_inference(self, data_path: str | None = None) -> None:
        """Prepara loader para uso em inferência.

        Carrega dataset e treina os transformadores (encoders, imputer, scaler).
        Deve ser chamado uma vez na inicialização da API.

        Args:
            data_path: Caminho para o CSV processado (usa self.data_path se não fornecido)
        """
        if data_path:
            self.data_path = data_path

        # 1. Carregar dados
        self.carregar()

        # 2. Preparar features (sem target)
        X, _ = self.preparar_features_target()

        # 2.5. Armazenar valores válidos para cada coluna categórica ANTES de transformar
        categoricas = X.select_dtypes(include=['object']).columns
        for col in categoricas:
            self.categorical_values[col] = set(X[col].unique())

        # 3. Codificar categoricas (fit=True para aprender os encoders)
        X = self.codificar_categoricas(X, fit=True)

        # 4. Imputar (fit=True para aprender as médias)
        X = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )

        # 5. Normalizar (fit=True para aprender média e desvio)
        X = self.normalizar_numericas(X, fit=True)

        # 6. Armazenar nomes das features
        self.feature_names = X.columns.tolist()

        # 7. Marcar como preparado para inferência
        self._fitted_for_inference = True

        print(f"[OK] Loader preparado para inferência com {len(self.feature_names)} features")

    def transform_single(self, features_dict: Any) -> np.ndarray:
        """
        Transforma o input da API em um array pronto para predição,
        mantendo os nomes originais (snake_case).
        """
        if not self._fitted_for_inference:
            raise ValueError("fit_for_inference() deve ser chamado antes de usar transform_single()")

        # 1. Converter Pydantic para dicionário real (se necessário)
        if hasattr(features_dict, "dict"):
            features_dict = features_dict.dict()
        elif hasattr(features_dict, "model_dump"):
            features_dict = features_dict.model_dump()

        # 2. Validar valores categóricos (usando as chaves snake_case originais)
        for col, valid_values in self.categorical_values.items():
            if col in features_dict:
                value = features_dict[col]
                if value not in valid_values:
                    raise ValueError(
                        f"Valor inválido para '{col}': '{value}'. "
                        f"Valores válidos: {sorted(valid_values)}"
                    )

        # 3. Criar DataFrame (Pandas usará as chaves do dict como nomes de colunas)
        df = pd.DataFrame([features_dict])

        # 4. Fluxo de processamento
        # Agora o 'col' dentro do codificar_categoricas deve bater com o snake_case
        df = self.codificar_categoricas(df, fit=False)

        df = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns
        )

        X_scaled = self.normalizar_numericas(df, fit=False)

        # Garante a ordem correta das features conforme o treinamento
        X_scaled = X_scaled[self.feature_names]

        return X_scaled.values

    def transform_batch(self, samples: list[Any]) -> np.ndarray:
        """
        Transforms a list of Pydantic objects or dicts into a normalized 2D array.
        Optimized for batch performance using vectorized operations.
        """
        if not self._fitted_for_inference:
            raise ValueError("fit_for_inference() must be called before transform_batch()")

        # 1. Convert Pydantic objects to dicts efficiently
        # We check the first item to decide the conversion strategy
        if len(samples) > 0:
            first_item = samples[0]
            if hasattr(first_item, "dict"):
                data = [s.dict() for s in samples]
            elif hasattr(first_item, "model_dump"):
                data = [s.model_dump() for s in samples]
            else:
                data = samples
        else:
            return np.array([]).reshape(0, len(self.feature_names))

        # 2. Create DataFrame (Vectorized operation)
        df = pd.DataFrame(data)

        # 3. Encode Categoricals (Vectorized)
        # This will now correctly find the snake_case columns
        df = self.codificar_categoricas(df, fit=False)

        # 4. Impute missing values
        df = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns
        )

        # 5. Normalize numericals
        X_scaled = self.normalizar_numericas(df, fit=False)

        # 6. Ensure column alignment and order
        X_scaled = X_scaled[self.feature_names]

        return X_scaled.values
