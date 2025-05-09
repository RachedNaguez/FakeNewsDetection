import pandas as pd
from src.data_ingest import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a file path and return a DataFrame."""
    # Determine the file extension
    file_extension = ".csv"
    # Get the appropriate data ingestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    # Ingest the data
    df = data_ingestor.ingest(file_path)
    return df
