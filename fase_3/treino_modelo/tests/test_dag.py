"""Testes unitários para a DAG do Airflow (dags/medical_training_dag.py)."""

from unittest.mock import patch

import pandas as pd
import pytest

from dags.medical_training_dag import (
    AIRFLOW_AVAILABLE,
    DEFAULT_ARGS,
    create_dag,
    run_evaluation,
    run_ingestion,
    run_training,
    run_validation,
)


def test_default_args():
    """Valida se os argumentos padrão da DAG contêm retries e schedule adequados."""
    assert DEFAULT_ARGS["owner"] == "mlops"
    assert DEFAULT_ARGS["retries"] >= 1
    assert DEFAULT_ARGS["depends_on_past"] is False


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="Apache Airflow não está instalado")
def test_dag_structure_and_dependencies():
    """Valida a estrutura de tasks e dependências na DAG do Airflow."""
    dag = create_dag()
    assert dag.dag_id == "medical_classification_training_pipeline"
    assert dag.schedule_interval == "@weekly"
    assert dag.catchup is False

    expected_tasks = {"ingest_data", "validate_data", "train_model", "evaluate_model"}
    assert set(dag.task_dict.keys()) == expected_tasks

    # Validação da ordem sequencial das dependências
    ingest_task = dag.get_task("ingest_data")
    validate_task = dag.get_task("validate_data")
    train_task = dag.get_task("train_model")
    evaluate_task = dag.get_task("evaluate_model")

    assert validate_task in ingest_task.downstream_list
    assert train_task in validate_task.downstream_list
    assert evaluate_task in train_task.downstream_list


@patch("dags.medical_training_dag.load_data")
def test_run_ingestion_callable(mock_load_data):
    """Testa a função de ingestão executada pelo PythonOperator."""
    dummy_df = pd.DataFrame({"medical_abstract": ["text"], "condition_label": [1]})
    mock_load_data.return_value = {"train": dummy_df, "test": dummy_df}

    result = run_ingestion()
    assert mock_load_data.called
    assert "train_shape" in result
    assert result["train_shape"] == (1, 2)


@patch("dags.medical_training_dag.validate_data")
@patch("dags.medical_training_dag.load_data")
def test_run_validation_callable(mock_load_data, mock_validate_data):
    """Testa a função de validação executada pelo PythonOperator."""
    dummy_df = pd.DataFrame({"medical_abstract": ["text"], "condition_label": [1]})
    mock_load_data.return_value = {"train": dummy_df, "test": dummy_df}

    run_validation()
    assert mock_load_data.called
    assert mock_validate_data.called


@patch("dags.medical_training_dag.train_model")
@patch("dags.medical_training_dag.load_data")
def test_run_training_callable(mock_load_data, mock_train_model):
    """Testa a função de treinamento executada pelo PythonOperator."""
    dummy_df = pd.DataFrame({"medical_abstract": ["text"], "condition_label": [1]})
    mock_load_data.return_value = {"train": dummy_df, "test": dummy_df}

    run_training()
    assert mock_load_data.called
    assert mock_train_model.called


@patch("joblib.load")
@patch("dags.medical_training_dag.evaluate_model")
@patch("dags.medical_training_dag.load_data")
def test_run_evaluation_callable(mock_load_data, mock_evaluate_model, mock_joblib_load):
    """Testa a função de avaliação executada pelo PythonOperator."""
    dummy_df = pd.DataFrame({"medical_abstract": ["text"], "condition_label": [1]})
    mock_load_data.return_value = {"train": dummy_df, "test": dummy_df}
    mock_evaluate_model.return_value = {"accuracy": 0.85, "report": "dummy report"}

    result = run_evaluation()
    assert mock_load_data.called
    assert mock_joblib_load.called
    assert mock_evaluate_model.called
    assert result["accuracy"] == 0.85
