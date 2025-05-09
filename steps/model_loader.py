from zenml import step, Model
from sklearn.linear_model import LogisticRegression
import logging
# from typing import Any # Or a more specific model type like sklearn.base.BaseEstimator, tensorflow.keras.Model, etc.

# Configure logging
logger = logging.getLogger(__name__)


@step
# -> Any: Add return type hint
def model_loader(model_name: str) -> LogisticRegression:
    """
    Loads a pre-trained model from the model registry.
    """
    logger.info(f"Loading model: {model_name}")

    model = Model(name=model_name, version="production")

    model_artifact = model.load_artifact("fake_news_detector_model")
    logger.info(f"Model loaded successfully: {model_artifact}")

    return model_artifact
