from zenml import step
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy


@step
def data_splitter_step(
    df: pd.DataFrame,
    text_column: str = "text",
    target_column: str = "class",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:  # Updated return type
    """Splits the data into training and testing sets."""

    splitter = DataSplitter(
        strategy=SimpleTrainTestSplitStrategy(test_size=test_size, random_state=random_state))

    X_train, X_test, y_train, y_test = splitter.split(
        df, text_column, target_column)
    return X_train, X_test, y_train, y_test
