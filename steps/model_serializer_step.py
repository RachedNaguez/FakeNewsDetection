from zenml import step
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# For type hinting, can be LogisticRegression or TfidfVectorizer
from typing import Union


@step
def model_serializer_step(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    model_path: str = "model.pkl",
    vectorizer_path: str = "vectorizer.pkl"
) -> None:
    """
    Serializes (saves) the trained model and the TF-IDF vectorizer to disk.
    """
    print("Model serializer step: Saving model and vectorizer")

    if model is None:
        print(
            f"Warning: Model is None. Skipping model serialization to {model_path}.")
    else:
        try:
            joblib.dump(model, model_path)
            print(f"Trained model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model to {model_path}: {e}")

    if vectorizer is None:
        print(
            f"Warning: Vectorizer is None. Skipping vectorizer serialization to {vectorizer_path}.")
    else:
        try:
            joblib.dump(vectorizer, vectorizer_path)
            print(f"Vectorizer saved to {vectorizer_path}")
        except Exception as e:
            print(f"Error saving vectorizer to {vectorizer_path}: {e}")

    print("Model and vectorizer serialization step complete.")
