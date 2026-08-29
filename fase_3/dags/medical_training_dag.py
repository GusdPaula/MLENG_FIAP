"""DAG do Apache Airflow para orquestração da pipeline de treinamento de classificação médica.

Agendamento: @weekly
Etapas: Ingestão -> Validação -> Treinamento -> Avaliação
"""

from datetime import datetime, timedelta
import logging

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIRFLOW_AVAILABLE = False
    DAG = object
    PythonOperator = object

from treino_modelo.data.ingest import load_data
from treino_modelo.data.validate import validate_data
from treino_modelo.train.evaluate import evaluate_model
from treino_modelo.train.train import train_model

logger = logging.getLogger(__name__)

# Default arguments para as tasks da DAG
DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def run_ingestion(**kwargs) -> dict:
    """Task 1: Ingestão dos dados via kagglehub."""
    logger.info("Executando task de ingestão de dados...")
    data = load_data()
    logger.info("Ingestão concluída com sucesso.")
    return {"train_shape": data["train"].shape, "test_shape": data["test"].shape}


def run_validation(**kwargs) -> None:
    """Task 2: Validação de consistência e schema dos dados."""
    logger.info("Executando task de validação de dados...")
    data = load_data()
    validate_data(data["train"], data["test"])
    logger.info("Validação concluída com sucesso.")


def run_training(**kwargs) -> None:
    """Task 3: Treinamento do modelo TF-IDF + Classificador."""
    logger.info("Executando task de treinamento do modelo...")
    data = load_data()
    train_model(data["train"])
    logger.info("Treinamento e persistência do modelo concluídos.")


def run_evaluation(**kwargs) -> dict:
    """Task 4: Avaliação do modelo e geração de relatórios/métricas."""
    logger.info("Executando task de avaliação do modelo...")
    import joblib
    from treino_modelo.train.train import MODEL_PATH

    data = load_data()
    pipeline = joblib.load(MODEL_PATH)
    results = evaluate_model(pipeline, data["test"])
    logger.info(f"Avaliação concluída. Acurácia: {results['accuracy']:.4f}")
    return results


def create_dag() -> DAG:
    """Fábrica de criação da DAG para execução e testes."""
    dag = DAG(
        dag_id="medical_classification_training_pipeline",
        default_args=DEFAULT_ARGS,
        description="Pipeline semanal de retreinamento do modelo de triagem de laudos médicos",
        schedule_interval="@weekly",
        catchup=False,
        tags=["mlops", "nlp", "medical", "fase_3"],
    )

    with dag:
        ingest_task = PythonOperator(
            task_id="ingest_data",
            python_callable=run_ingestion,
        )

        validate_task = PythonOperator(
            task_id="validate_data",
            python_callable=run_validation,
        )

        train_task = PythonOperator(
            task_id="train_model",
            python_callable=run_training,
        )

        evaluate_task = PythonOperator(
            task_id="evaluate_model",
            python_callable=run_evaluation,
        )

        # Ordem de dependência das tarefas
        ingest_task >> validate_task >> train_task >> evaluate_task

    return dag


# Instância global da DAG para descoberta automática pelo Airflow
if AIRFLOW_AVAILABLE:
    dag = create_dag()
