from zenml import step
import pandas as pd
from typing import Tuple
# Assuming src.data_prep is accessible via PYTHONPATH
from src.data_prep import (
    DataPreparationContext,
    AddClassColumn,
    ConcatenateDataFrames,
    DropColumns,
    ResetIndex
)


@step
def data_prep_step(true_raw_df: pd.DataFrame, fake_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the raw true and fake news DataFrames using strategies
    from src.data_prep.py.
    - Adds 'class' column (1 to true_raw_df, 0 to fake_raw_df).
    - Concatenates the DataFrames.
    - Drops 'title', 'subject', 'date' columns.
    - Resets the index.
    """
    print("Data preparation step (using src.data_prep): Starting...")

    if true_raw_df.empty and fake_raw_df.empty:
        print(
            "Both input DataFrames to data_prep_step are empty. Returning empty DataFrame.")
        return pd.DataFrame()
    if true_raw_df.empty:
        print("Warning: true_raw_df is empty.")
        # Decide handling: maybe process only fake_df or return fake_df if that's desired
    if fake_raw_df.empty:
        print("Warning: fake_raw_df is empty.")
        # Decide handling

    try:
        # Prepare true_df: Add class column
        true_prep_context = DataPreparationContext()
        true_prep_context.add_strategy(AddClassColumn(class_value=1))
        processed_true_df = true_prep_context.prepare_data(true_raw_df.copy())
        print(f"Added 'class'=1 to true_df. Shape: {processed_true_df.shape}")

        # Prepare fake_df: Add class column
        fake_prep_context = DataPreparationContext()
        fake_prep_context.add_strategy(AddClassColumn(class_value=0))
        processed_fake_df = fake_prep_context.prepare_data(fake_raw_df.copy())
        print(f"Added 'class'=0 to fake_df. Shape: {processed_fake_df.shape}")

        # Combine and further process using main context
        main_preparation_context = DataPreparationContext()
        # Strategy to concatenate processed_fake_df to processed_true_df
        main_preparation_context.add_strategy(ConcatenateDataFrames(
            data_frame_to_concatenate=processed_fake_df))
        # Strategy to drop columns
        columns_to_drop = ['title', 'subject', 'date']
        main_preparation_context.add_strategy(
            DropColumns(columns_to_drop=columns_to_drop))
        # Strategy to reset index
        main_preparation_context.add_strategy(ResetIndex())

        # Apply concatenation, drop, and reset_index to processed_true_df
        final_prepared_df = main_preparation_context.prepare_data(
            processed_true_df)

        print("Data preparation strategies from src.data_prep applied successfully.")
        print(
            f"Shape after all data_prep_step strategies: {final_prepared_df.shape}")
        # print(final_prepared_df.head()) # Avoid for large data

    except Exception as e:
        print(f"Error during data_prep_step execution: {e}")
        # Return an empty DataFrame or re-raise, depending on desired pipeline behavior
        return pd.DataFrame()

    return final_prepared_df
