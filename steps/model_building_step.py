from zenml.enums import ArtifactType
from zenml.client import Client
from zenml import ArtifactConfig, step
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy.sparse import csr_matrix  # Import csr_matrix
import mlflow
from typing import Annotated
import logging
from zenml import Model


experiment_tracker = Client().active_stack.experiment_tracker


model = Model(
    name="fake_news_detector",
    version=None,
    license="Apache 2.0",
    description="Pipeline to detect fake news based on text content."
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(xv_train: csr_matrix, y_train: pd.Series) -> Annotated[LogisticRegression, ArtifactConfig(name="fake_news_detector_model", artifact_type=ArtifactType.MODEL)]:
    """
    Trains a Logistic Regression model on the transformed training data.
    """
    print("Model building step: Training Logistic Regression model")

    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train must be a pandas Series")
    if xv_train.shape[0] == 0 or y_train.empty:  # Check if sparse matrix is empty
        raise ValueError("Training data is empty")
    logging.info("Training Logistic Regression model")
    model = LogisticRegression()
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Training Logistic Regression model")
        # Convert y_train to NumPy array to avoid MLflow warning with autolog
        model.fit(xv_train, y_train.values)
        logging.info("Model training complete")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    finally:
        mlflow.end_run()
    return model
