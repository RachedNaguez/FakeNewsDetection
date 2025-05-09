import click
from pipelines.training_pipeline import fake_news_detection_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


click.command()


def main():
    """
    Run the ML pipeline and start the MLflow UI.
    """

    # Run the ML pipeline
    run = fake_news_detection_pipeline()
    # Print the run ID
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the `ml_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )


if __name__ == "__main__":
    main()
