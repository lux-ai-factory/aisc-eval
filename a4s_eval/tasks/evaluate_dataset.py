"""Task implementation for evaluating dataset drift over time.

This module provides functionality to evaluate data drift in a dataset by comparing
a reference dataset against a test dataset over various time windows. The evaluation
results are posted to an API endpoint for tracking.
"""

import pandas as pd
import requests

from a4s_eval.data_model.metric import Metric
from a4s_eval.data_model.project import Project
from a4s_eval.datasets.drift import data_drift_test
from a4s_eval.utils.env import API_URL
from a4s_eval.utils.files import auto_get_read_dataset_file


def post_metrics(project_id: int, metrics: list[Metric]) -> None:
    """Post computed metrics to the API endpoint.

    Args:
        project_id: ID of the project these metrics belong to
        metrics: List of computed metrics to post
    """
    response = requests.post(
        f"{API_URL}/api/projects/{project_id}/metrics",
        json=[m.model_dump(mode="json") for m in metrics],
    )
    if response.status_code == 201:
        print("Metrics sent successfully")
        print(response.json())
    else:
        print(f"Failed to send metrics {response.status_code}")


def evaluate_dataset(project_id: int) -> None:
    """Evaluate data drift for a project's dataset.

    This function:
    1. Retrieves project configuration from the API
    2. Loads reference and test datasets
    3. Computes drift metrics over time windows
    4. Posts results back to the API

    Args:
        project_id: ID of the project to evaluate
    """
    # Retrieve the data
    # Run the evaluations
    response = requests.get(f"{API_URL}/api/projects/{project_id}")

    if response.status_code == 200:
        project = Project(**response.json())  # Convert API response to Pydantic object

        # Load and preprocess datasets
        x_ref = auto_get_read_dataset_file(project.dataset.train_file_path)
        x_new = auto_get_read_dataset_file(project.dataset.test_file_path)

        # Convert date columns to datetime
        x_ref[project.dataset.date_feature.name] = pd.to_datetime(
            x_ref[project.dataset.date_feature.name]
        )
        x_new[project.dataset.date_feature.name] = pd.to_datetime(
            x_new[project.dataset.date_feature.name]
        )

        # Compute drift metrics
        metrics = data_drift_test(
            project,
            x_ref,
            x_new,
            project.dataset.features,
            project.dataset.date_feature,
        )

        print(metrics)

        # post_metrics(project_id, metrics)  # Commented out for testing
    else:
        print(f"Failed to fetch data {response.status_code}")


if __name__ == "__main__":
    # Example usage
    evaluate_dataset(1)  # First evaluation
    evaluate_dataset(1)  # Second evaluation
