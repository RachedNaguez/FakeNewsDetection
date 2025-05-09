import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DataSplitterStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, text_column: str, target_column: str):
        """Abstract method to split data into training and testing sets.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - target_column (str): The name of the target column.

        Returns:
        X_train,X_test, y_train, y_test: The training and testing sets for features and target variable.
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplitterStrategy):
    def __init__(self, test_size: float, random_state: int):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, text_column: str, target_column: str):
        """
        Splits the data into training and testing sets using a simple train-test split.
        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the target column.
        Returns:
        X_train, X_test, y_train, y_test: The training and testing sets for features and target variable.
        """
        logging.info("Splitting data into training and testing sets.")
        X = df[[text_column]]  # Changed to return DataFrame
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        logging.info("Training and testing sets created successfully.")
        return X_train, X_test, y_train, y_test


class DataSplitter:
    def __init__(self, strategy: DataSplitterStrategy):
        """
        Initializes the DataSplitter with a specific strategy.
        Parameters:
        strategy (DataSplitterStrategy): The strategy to use for splitting the data.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: DataSplitterStrategy):
        """
        Sets a new strategy for the DataSplitter.
        Parameters:
        strategy (DataSplitterStrategy): The new strategy to use for splitting the data.
        """
        self.strategy = strategy
        logging.info(f"Strategy set to {type(strategy).__name__}.")

    def split(self, df: pd.DataFrame, text_column: str, target_column: str):
        """
        Splits the data using the current strategy.
        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the target column.
        Returns:
        X_train, X_test, y_train, y_test: The training and testing sets for features and target variable.
        """
        logging.info("Starting data split.")
        return self.strategy.split_data(df, text_column, target_column)
