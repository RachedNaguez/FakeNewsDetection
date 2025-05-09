import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ModelEvaluatorStrategy(ABC):
    @abstractmethod
    def evaluate(self, model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model and return evaluation metrics.
        """
        pass


class RegressionModelEvaluatorStrategy(ModelEvaluatorStrategy):
    def evaluate(self, model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the logistic regression model and return evaluation metrics.
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)
        logging.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'classification_report': report
        }
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluatorStrategy):
        """ Initializes the ModelEvaluator with a specific model evaluation strategy.
        This allows for flexibility in how models are evaluated, making it easy to switch strategies
        or extend functionality in the future.
        Parameters:
        strategy (ModelEvaluatorStrategy): The evaluation strategy to use.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: ModelEvaluatorStrategy):
        """
        Sets the evaluation strategy.

        Parameters:
        strategy (ModelEvaluatorStrategy): The evaluation strategy to use.
        """
        logging.info(
            f"Setting evaluation strategy to {strategy.__class__.__name__}.")
        self.strategy = strategy

    def evaluate(self, model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the model using the current strategy.
        """
        return self.strategy.evaluate(model, X_test, y_test)
