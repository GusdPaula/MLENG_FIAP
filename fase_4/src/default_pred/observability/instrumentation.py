import logging
from collections.abc import Callable, Sequence
from typing import Any

from default_pred.config import EXPERIMENT_INFERENCE, EXPERIMENT_TRAINING
from default_pred.observability.metrics import (
    as_float_metrics,
    inference_metrics,
    training_params,
)
from default_pred.observability.run import tracked_run
from default_pred.observability.timing import timed
from default_pred.observability.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

TRAINING_RUN_NAME = "logistic_regression_training"
INFERENCE_RUN_NAME = "batch_inference"

TrainFn = Callable[..., tuple[Any, dict[str, float], Any, Any]]


def track_training(
    tracker: ExperimentTracker,
    train_fn: TrainFn,
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    max_iter: int,
    class_weight: Any,
    test_size: float,
    random_state: int,
):
    """Train through `train_fn` while recording params, metrics and the model."""
    with tracked_run(tracker, EXPERIMENT_TRAINING, TRAINING_RUN_NAME) as outcome:
        outcome.rows = len(X_train)
        tracker.log_params(
            training_params(
                max_iter=max_iter,
                class_weight=class_weight,
                test_size=test_size,
                random_state=random_state,
                n_train=len(X_train),
                n_features=X_train.shape[1],
            )
        )

        model, metrics, _class_report, _conf_matrix = train_fn(
            X_train,
            X_test,
            y_train,
            y_test,
            max_iter=max_iter,
            class_weight=class_weight,
        )

        tracker.log_metrics(as_float_metrics(metrics))
        tracker.log_model(model, "model")
        return model


def track_inference(
    tracker: ExperimentTracker, infer_fn: Callable[[], Sequence[float]]
) -> Sequence[float]:
    """Run `infer_fn` while recording rows, latency and predicted default rate."""
    with tracked_run(tracker, EXPERIMENT_INFERENCE, INFERENCE_RUN_NAME) as outcome:
        with timed("inference") as timer:
            predictions = infer_fn()

        outcome.rows = len(predictions)
        tracker.log_metrics(inference_metrics(predictions, timer.elapsed_ms))
        return predictions
