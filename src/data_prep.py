import logging
from abc import ABC, abstractmethod
import pandas as pd

# setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Abstract base class for data preparation steps


class DataPreparationStrategy(ABC):
    """
    Abstract base class for data preparation strategies.
    Each concrete strategy should implement the `prepare_data` method.
    """
    @abstractmethod
    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle data preparation.
        :param data: Input DataFrame to be processed.
        :return: Processed DataFrame.
        """
        pass


class AddClassColumn(DataPreparationStrategy):
    """
    Concrete strategy to add a 'class' column to the DataFrame.
    """

    def __init__(self, class_value: int):
        """
        Initialize the strategy with the class value to be added.
        :param class_value: The value to be added in the 'class' column.
        """
        self.class_value = class_value

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'class' column to the DataFrame with the specified class value.
        :param data: Input DataFrame to be processed.
        :return: DataFrame with the 'class' column added.
        """
        logging.info(f"Adding 'class' column with value {self.class_value}.")
        data['class'] = self.class_value
        return data


class ConcatenateDataFrames(DataPreparationStrategy):
    """
    Concrete strategy to concatenate two DataFrames.
    """

    def __init__(self, data_frame_to_concatenate: pd.DataFrame):
        """
        Initialize the strategy with the DataFrame to be concatenated.
        :param data_frame_to_concatenate: The DataFrame to be concatenated.
        """
        self.data_frame_to_concatenate = data_frame_to_concatenate

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate the input DataFrame with another DataFrame.
        :param data: Input DataFrame to be processed.
        :return: Concatenated DataFrame.
        """
        logging.info("Concatenating DataFrames.")
        return pd.concat([data, self.data_frame_to_concatenate], ignore_index=True)


class DropColumns(DataPreparationStrategy):
    """
    Concrete strategy to drop specified columns from the DataFrame.
    """

    def __init__(self, columns_to_drop: list):
        """
        Initialize the strategy with the columns to be dropped.
        :param columns_to_drop: List of column names to be dropped.
        """
        self.columns_to_drop = columns_to_drop

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.
        :param data: Input DataFrame to be processed.
        :return: DataFrame with specified columns dropped.
        """
        logging.info(f"Dropping columns: {self.columns_to_drop}.")
        return data.drop(columns=self.columns_to_drop)


class ResetIndex(DataPreparationStrategy):
    """ Concrete strategy to reset the index of the DataFrame."""

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reset the index of the DataFrame.
        :param data: Input DataFrame to be processed.
        :return: DataFrame with reset index.
        """
        logging.info("Resetting index.")
        return data.reset_index(drop=True)


class DataPreparationContext:
    """
    Context class to manage the data preparation strategies.
    It allows adding multiple strategies and executing them in order.
    """

    def __init__(self):
        self.strategies = []

    def add_strategy(self, strategy: DataPreparationStrategy):
        """
        Add a data preparation strategy to the context.
        :param strategy: An instance of a DataPreparationStrategy.
        """
        self.strategies.append(strategy)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all added strategies on the input DataFrame.
        :param data: Input DataFrame to be processed.
        :return: Processed DataFrame after applying all strategies.
        """
        for strategy in self.strategies:
            data = strategy.handle(data)
        return data
