import joblib
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

vectorizer = joblib.load('vectorizer.pkl')


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: pd.DataFrame,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (pd.DataFrame): DataFrame containing text data for prediction.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Ensure we have the text column
    if 'text' not in input_data.columns:
        raise ValueError(
            "Input data must contain a 'text' column for prediction")

    # Extract just the text column for prediction
    texts = input_data['text'].values

    # Create a text processing pipeline similar to what was used in training
    # This should match the vectorization process used during model training

    # Transform the text data
    X_vectorized = vectorizer.transform(texts)

    # Convert to array format if needed
    # Some MLFlow models expect dense arrays
    X_array = X_vectorized.toarray()

    # Run the prediction with vectorized data
    prediction = service.predict(X_array)

    return prediction
