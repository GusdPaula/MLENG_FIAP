"""
Script de Treinamento Refatorado com Pipeline Completo.

Cria um Pipeline sklearn que une:
1. ColumnTransformer para pré-processamento (categóricas + numéricas)
2. LogisticRegression balanceado para classificação

O Pipeline completo é logado no MLflow para fácil reprodução e deploy.
"""

import os
import sys
import logging

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath("")))

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Construtor do Pipeline completo de ML com pré-processamento integrado."""

    def __init__(self, random_state: int = 42, data_path: str = "../data/processed/telco_churn_processed.csv"):
        self.random_state = random_state
        self.categorical_columns = None
        self.numerical_columns = None
        self.pipeline = None
        self.data_path = data_path        

    def identify_feature_types(self, X: pd.DataFrame) -> tuple[list, list]:
        """
        Identifica colunas categóricas e numéricas.

        Args:
            X: DataFrame com features

        Returns:
            (categorical_columns, numerical_columns)
        """
        logger.info("Identificando tipos de features... ")
        logger.info(X.head(3))
        self.categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
        self.numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()


        logger.info(f"✓ Colunas identificadas:")
        logger.info(f"  - Categóricas: {len(self.categorical_columns)}")
        logger.info(f"  - Numéricas: {len(self.numerical_columns)}")
        logger.info(f"  - Categóricas: {self.categorical_columns}")
        logger.info(f"  - Numéricas: {self.numerical_columns}")

        return self.categorical_columns, self.numerical_columns

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Cria o ColumnTransformer para pré-processamento.

        Categorias: OneHotEncoder (melhor interpretabilidade)
        Numéricas: SimpleImputer (média) + StandardScaler

        Returns:
            ColumnTransformer configurado
        """
        if self.categorical_columns is None or self.numerical_columns is None:
            raise ValueError("Chame identify_feature_types() primeiro")

        # Pipeline para colunas categóricas
        # OneHotEncoder cria colunas binárias para cada categoria
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        sparse_output=False,  # Retorna array denso (não sparse)
                        drop="if_binary",  # Remove 1 coluna em features binárias
                        handle_unknown="ignore",  # ✅ Ignora valores novos em inferência
                    ),
                ),
            ]
        )

        # Pipeline para colunas numéricas
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),  # ✅ Trata NaN
                ("scaler", StandardScaler()),
            ]
        )

        # ColumnTransformer une ambos
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, self.categorical_columns),
                ("numerical", numerical_transformer, self.numerical_columns),
            ],
            remainder="drop",  # Descarta colunas não selecionadas
            verbose_feature_names_out=True,  # Nomes descritivos das features
        )

        logger.info("✓ ColumnTransformer criado")
        return preprocessor

    def create_full_pipeline(self) -> Pipeline:
        """
        Cria o Pipeline completo: Preprocessor + Classifier.

        Returns:
            Pipeline sklearn pronto para treino
        """
        preprocessor = self.create_preprocessor()

        # Pipeline completo: Preprocessor + LogisticRegression
        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        self.pipeline = full_pipeline
        logger.info("✓ Pipeline completo criado")
        return full_pipeline

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PipelineBuilder":
        """
        Treina o Pipeline.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento

        Returns:
            Self para encadeamento
        """
        if self.pipeline is None:
            raise ValueError("Chame create_full_pipeline() primeiro")

        logger.info("🔄 Iniciando treinamento do Pipeline...")
        self.pipeline.fit(X_train, y_train)
        logger.info("✓ Pipeline treinado com sucesso")

        return self

    def get_pipeline(self) -> Pipeline:
        """Retorna o Pipeline treinado."""
        return self.pipeline
    
    def preparar_features_target(self) -> tuple[pd.DataFrame, pd.Series]:
            """
            Separa features e target.

            Target: churn_value (0 = Não Churn, 1 = Churn)
            """

            self.df[self.df.select_dtypes(include=['Int64']).columns] = self.df.select_dtypes(include=['Int64']).astype('int64')

            # Remover colunas não relevantes
            drop_cols = [
                "customerid",
                "count",
                "country",
                "state",
                "city",
                "zip_code",
                "lat_long",
                "latitude",
                "longitude",  # localização inútil
                "churn_label",  # usar churn_value em vez disso
                "churn_reason",  # razão subjetiva
                "cltv",  # leakage - correlacionado com churn
                "churn_score",  # leakage - score de churn externo
            ]
            self.df.rename(columns={"Churn Value": "churn_value"}, inplace=True)
            X = self.df.drop(columns=[*drop_cols, "churn_value"])
            y = self.df["churn_value"]            

            logger.info(f"[OK] Features selecionadas: {X.shape[1]}")
            logger.info(f"  - Distribuicao de Churn: {y.value_counts().to_dict()}")
            logger.info(f"  - Taxa de Churn: {y.mean() * 100:.2f}%")

            return X, y

    def carregar(self) -> pd.DataFrame:
            """Carrega o dataset."""
            self.df = pd.read_csv(self.data_path)
            logger.info(f"[OK] Dataset carregado: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas")
            return self.df

    def split_treino_teste(
            self, X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42
        ) -> None:
            """
            Divide dados em treino e teste.

            Usa stratified split para manter a proporção de churn.
            """
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            logger.info("[OK] Split treino/teste (80/20):")
            logger.info(f"  - Treino: {self.X_train.shape[0]} amostras")
            logger.info(f"  - Teste: {self.X_test.shape[0]} amostras")
            logger.info(f"  - Taxa churn treino: {self.y_train.mean() * 100:.2f}%")
            logger.info(f"  - Taxa churn teste: {self.y_test.mean() * 100:.2f}%")        
        

def main():
    """Função principal do script de treinamento."""

    # ============ CARREGAR E PREPARAR DADOS ============

    # Conectar ao MLflow na porta 5000
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"📊 MLflow Tracking URI: {tracking_uri}")

    # Criar ou usar experimento existente
    experiment_name = "TelcoChurnPipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        logger.info(f"✓ Experimento criado: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    logger.info(f"✓ Usando experimento: {experiment_name}")    

    # ============ CONSTRUIR PIPELINE ============
    logger.info("\n1️⃣  Construindo Pipeline...")
    builder = PipelineBuilder(random_state=42)
    
        
    builder.carregar()
    logger.info("Tipo de dados carregados:  ")
    logger.info(builder.df.dtypes)

    X, y = builder.preparar_features_target()
    builder.split_treino_teste(X, y)
    builder.identify_feature_types(builder.X_train)
    builder.create_full_pipeline()

    # ============ TREINAR PIPELINE ============
    logger.info("\n2️⃣  Treinando Pipeline...")
    builder.train(builder.X_train, builder.y_train)
    logger.info("✓ Pipeline treinado com sucesso")


    # ============ AVALIAR NO CONJUNTO DE TESTE ============
    logger.info("\n3️⃣  Avaliando Pipeline no conjunto de teste...")
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    y_pred = builder.pipeline.predict(builder.X_test)
    y_pred_proba = builder.pipeline.predict_proba(builder.X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(builder.y_test, y_pred),
        "precision": precision_score(builder.y_test, y_pred),
        "recall": recall_score(builder.y_test, y_pred),
        "f1_score": f1_score(builder.y_test, y_pred),
        "auc_roc": roc_auc_score(builder.y_test, y_pred_proba),
    }

    logger.info("\n📊 Métricas do Pipeline:")
    for metric, value in metrics.items():
        logger.info(f"  {metric:.<30} {value:.4f}")

    # ============ LOGAR NO MLFLOW ============
    logger.info("\n4️⃣  Logando Pipeline no MLflow...")

    # Encerrar runs anteriores
    if mlflow.active_run():
        mlflow.end_run()

    # ✅ Criar um input_example RAW (antes de transformações)
    # Isso garante que o MLflow registra o schema correto para inferência
    # Carregamos os dados brutos antes de qualquer transformação
    raw_example = builder.df.head(1)
    # Pegar apenas colunas categóricas + numéricas (sem extras)
    raw_example = raw_example[builder.categorical_columns + builder.numerical_columns]
    
    logger.info(f"✓ Input example (RAW): {raw_example.shape}")
    logger.info(f"  Colunas categóricas no exemplo: {builder.categorical_columns[:3]}...")
    logger.info(f"  Colunas numéricas no exemplo: {builder.numerical_columns[:3]}...")

    # Iniciar nova run
    with mlflow.start_run(run_name="telco-churn-pipeline-logistic"):
        # Log de parâmetros
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("categorical_encoding", "OneHotEncoder")
        mlflow.log_param("categorical_imputation", "most_frequent")
        mlflow.log_param("numerical_scaling", "StandardScaler")
        mlflow.log_param("numerical_imputation", "mean")
        mlflow.log_param("n_categorical_features", len(builder.categorical_columns))
        mlflow.log_param("n_numerical_features", len(builder.numerical_columns))

        # Log de métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log do Pipeline completo
        mlflow.sklearn.log_model(
            builder.pipeline,
            name="model",
            registered_model_name="TelcoChurnPipeline",
            input_example=raw_example,  # ✅ Usar dados RAW, não transformados
            metadata={
                "categorical_columns": builder.categorical_columns,
                "numerical_columns": builder.numerical_columns,
                "feature_count": builder.X_train.shape[1],
                "training_samples": builder.X_train.shape[0],
                "test_samples": builder.X_test.shape[0],
            },
        )

        logger.info("✓ Pipeline logado no MLflow")
        logger.info(f"  - Modelo registrado: TelcoChurnPipeline")
        logger.info(f"  - Artifact Path: model")

    logger.info("\n" + "=" * 70)
    logger.info("✅ TREINAMENTO CONCLUÍDO COM SUCESSO")
    logger.info("=" * 70)

    # ============ INFORMAÇÕES DE FEATURES ============
    logger.info("\n📋 Resumo de Features:")
    logger.info(f"\nColunas Categóricas ({len(builder.categorical_columns)}):")
    for col in builder.categorical_columns:
        logger.info(f"  - {col}")

    logger.info(f"\nColunas Numéricas ({len(builder.numerical_columns)}):")
    for col in builder.numerical_columns:
        logger.info(f"  - {col}")

    return builder.pipeline


if __name__ == "__main__":
    pipeline = main()
