import os
from abc import ABC, abstractmethod

import pandas as pd


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a file path and return a DataFrame."""
        pass


class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a CSV file."""
        if not file_path.endswith('.csv'):
            raise ValueError("File path must end with .csv")
        df = pd.read_csv(file_path)
        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestor:
        """Factory method to get the appropriate data ingestor based on file type."""
        if file_path.endswith('.csv'):
            return CSVDataIngestor()
        else:
            raise ValueError(
                "Unsupported file type. Only .csv files are supported.")
