from steps.data_ingestion_step import data_ingestion_step
from steps.data_prep_step import data_prep_step  # Added
from steps.text_cleaning_step import text_cleaning_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.model_serializer_step import model_serializer_step
from zenml import Model, pipeline


@pipeline(
    model=Model(
        name="fake_news_detector",
    ),
)
def fake_news_detection_pipeline(
    true_csv_path: str = "./data/True.csv",
    fake_csv_path: str = "./data/Fake.csv",
    text_column: str = "text",
    target_column: str = "class",
    test_size: float = 0.25,
    random_state: int = 42,
    model_output_path: str = "model.pkl",
    vectorizer_output_path: str = "vectorizer.pkl"
):
    """Defines an end-to-end machine learning pipeline for fake news detection."""

    # Data Ingestion Step
    true_data_df = data_ingestion_step(
        file_path=true_csv_path
    )
    fake_data_df = data_ingestion_step(
        file_path=fake_csv_path
    )

    # Data Preparation Step
    prepared_data_df = data_prep_step(
        true_raw_df=true_data_df,
        fake_raw_df=fake_data_df
    )

    # Text Cleaning Step
    cleaned_data_df = text_cleaning_step(
        data=prepared_data_df,  # Changed from raw_data_df
        text_column=text_column
    )

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(
        df=cleaned_data_df,
        text_column=text_column,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )

    # Feature Engineering Step (TF-IDF)
    vectorizer, xv_train, xv_test = feature_engineering_step(
        x_train=X_train,
        x_test=X_test
    )

    # Model Building Step (Logistic Regression)
    trained_model = model_building_step(
        xv_train=xv_train,
        y_train=y_train
    )

    # Model Evaluation Step
    accuracy, classification_report = model_evaluator_step(  # Unpack directly
        trained_model=trained_model,
        xv_test=xv_test,
        y_test=y_test
    )
    # accuracy = metrics["accuracy"] # No longer needed
    # classification_report = metrics["classification_report"] # No longer needed

    # Model Serialization Step
    model_serializer_step(
        model=trained_model,
        vectorizer=vectorizer,
        model_path=model_output_path,
        vectorizer_path=vectorizer_output_path
    )

    # Return the trained model first, so it can be used by deployment pipelines
    return trained_model, accuracy, classification_report


if __name__ == "__main__":
    print("Attempting to run the Fake News Detection ML pipeline...")
    try:
        # Example of running the pipeline with default parameters
        # In a real ZenML setup, you'd typically run this from the CLI
        # or a dedicated script after `zenml init`.
        # When running a ZenML pipeline directly like this, it doesn't return values for direct unpacking.
        # The outputs (accuracy, classification_report) are registered as ZenML artifacts.
        fake_news_detection_pipeline(random_state=46)  # Cache busting
        print("Pipeline run initiated. Check the ZenML dashboard for status and artifacts.")
        print(f"Model should be saved to model.pkl and vectorizer to vectorizer.pkl")

    except ImportError as ie:
        print(
            f"ImportError: {ie}. Make sure ZenML and all step dependencies are installed.")
        print("Try: pip install zenml scikit-learn pandas joblib")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("This might be due to ZenML not being initialized ('zenml init'),")
        print("or issues within the implemented steps (e.g., file paths, data issues).")
        print("Ensure 'True.csv' and 'Fake.csv' are in the root directory or provide correct paths.")
