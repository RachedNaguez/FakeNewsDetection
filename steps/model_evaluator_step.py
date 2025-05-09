import logging
from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix  # Import csr_matrix
from sklearn.linear_model import LogisticRegression
from zenml import step
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluatorStrategy


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: LogisticRegression,
    xv_test: csr_matrix,  # Changed to csr_matrix
    y_test: pd.Series
) -> Tuple[float, str]:  # Return accuracy and classification report separately
    """
    Evaluates the trained model using the specified evaluation strategy.

    Args:
        model (LogisticRegression): The trained logistic regression model.
        x_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels for the test set.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    logging.info("Initializing the model evaluator.")
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluatorStrategy())

    logging.info("Evaluating the model.")
    metrics = evaluator.evaluate(trained_model, xv_test, y_test)
    accuracy = metrics["accuracy"]
    classification_rep = metrics["classification_report"]

    return accuracy, classification_rep
