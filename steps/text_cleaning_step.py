from zenml import step
import pandas as pd
from src.data_clean import TextCleaner


@step
def text_cleaning_step(data: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Cleans text data in the specified column using TextCleaner.

    Args:
        data: Input DataFrame containing text to clean
        text_column: Column containing text to clean

    Returns:
        DataFrame with cleaned text
    """
    print(f"Cleaning text in column: {text_column}")

    # Create a copy to avoid modifying the original
    result = data.copy()

    # Apply the TextCleaner to the specified column
    result[text_column] = result[text_column].apply(
        lambda text: TextCleaner(text).clean_text().get_cleaned_text()
    )

    print("Text cleaning complete")
    return result
