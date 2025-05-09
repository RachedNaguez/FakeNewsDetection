import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureEngineeringStrategy(ABC):
    """
    Abstract base class for feature engineering strategies.
    """

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to transform the data.
        """
        pass


class TfidfFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    """
    Concrete strategy for TF-IDF feature engineering.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the TF-IDF vectorizer and transform the data.
        """
        logging.info(
            "Fitting and transforming data using TF-IDF vectorization.")
        tfidf_matrix = self.vectorizer.fit_transform(data['text'])
        return tfidf_matrix  # Return sparse matrix

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # Return type will be sparse matrix
        """
        Transform the data using TF-IDF vectorization.
        """
        logging.info("Transforming data using TF-IDF vectorization.")
        tfidf_matrix = self.vectorizer.transform(data['text'])
        # Return sparse matrix
        return tfidf_matrix

    def get_vectorizer(self) -> TfidfVectorizer:
        """
        Get the fitted TF-IDF vectorizer.
        """
        return self.vectorizer
