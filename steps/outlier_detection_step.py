from zenml import step
import pandas as pd


@step
def outlier_detection_step(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Detects and handles outliers in the specified column.
    TODO: Determine if and how outlier detection is applicable for your text features.
          This step is more common for numerical features. If using numerical representations
          of text (e.g., certain embedding properties or derived metrics), it might be relevant.
          Otherwise, this step might be removed or significantly adapted.
    """
    print(f"Outlier detection step: Processing column '{column_name}'")
    # Placeholder: returning data as is
    # Replace with actual outlier detection and handling logic if applicable
    # Example for numerical data (conceptual):
    # Q1 = data[column_name].quantile(0.25)
    # Q3 = data[column_name].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # data_cleaned = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    print(
        f"Warning: Placeholder outlier detection for column '{column_name}'. Returning data as is.")
    return data
