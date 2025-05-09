from zenml import step
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix  # Import csr_matrix
from src.feature_engineering import TfidfFeatureEngineeringStrategy


@step
def feature_engineering_step(
    x_train: pd.DataFrame,  # This will be a Series after data cleaning
    x_test: pd.DataFrame,  # This will be a Series after data cleaning
) -> Tuple[TfidfVectorizer, csr_matrix, csr_matrix]:  # Adjusted return types for sparse matrices
    """
    Performs feature engineering on the training and test text data using a strategy pattern.
    """
    print("Feature engineering step: Performing Feature Engineering")

    tfidf_strategy = TfidfFeatureEngineeringStrategy()

    try:
       # Fit and transform the training data
        xv_train = tfidf_strategy.fit_transform(x_train)
        # Transform the test data
        xv_test = tfidf_strategy.transform(x_test)

        # Get the fitted vectorizer
        vectorizer = tfidf_strategy.get_vectorizer()
        print("Feature engineering step: TF-IDF transformation complete.")
    except Exception as e:
        print(
            f"Feature engineering step: Error during TF-IDF transformation: {e}")

        raise
    return vectorizer, xv_train, xv_test
