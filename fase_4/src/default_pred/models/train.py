from sklearn.linear_model import LogisticRegression
from default_pred.models.evaluate import evaluate_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train, X_test, y_train, y_test, max_iter=1000, class_weight=None
):
    """Train a logistic regression model and evaluate its performance."""
    logger.info("Starting model training...")
    # Initialize model
    model = LogisticRegression(
        max_iter=max_iter, class_weight=class_weight, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    metrics, class_report, conf_matrix = evaluate_model(model, X_test, y_test)

    logger.info("Evaluation Metrics")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info(class_report)
    logger.info("\nConfusion Matrix:")
    logger.info(conf_matrix)

    return model, metrics, class_report, conf_matrix
