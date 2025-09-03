"""Task implementation for evaluating model performance over time.

This module provides functionality to evaluate a machine learning model's performance
by computing various classification metrics over time windows. The model is loaded
from an ONNX format file and evaluated against test data. Results are posted to
an API endpoint for tracking.
"""

import numpy as np
import onnxruntime as rt
import pandas as pd
import requests

from a4s_eval.data_model.metric import Metric
from a4s_eval.data_model.project import Project
from a4s_eval.evaluations.model_evaluation.metrics import prediction_test
from a4s_eval.utils.env import API_URL
from a4s_eval.utils.files import auto_get_read_dataset_file
from a4s_eval.utils.logging import get_logger


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
        get_logger().debug("Metrics sent successfully")
    else:
        get_logger().error(f"Failed to send metrics {response.status_code}")


def evaluate_dataset(project_id: int) -> None:
    """Evaluate model performance on a project's dataset.

    This function:
    1. Retrieves project configuration from the API
    2. Loads the model and test dataset
    3. Makes predictions using the model
    4. Computes performance metrics over time windows
    5. Posts results back to the API

    Args:
        project_id: ID of the project to evaluate

    Note:
        The model is expected to be in ONNX format and available as 'random_forest.onnx'
    """
    get_logger().debug(f"Starting evaluation for project {project_id}")

    # Get project configuration from API
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

        # Prepare feature matrix for prediction
        x_test = x_new[[f.name for f in project.dataset.features]].to_numpy()

        # Load and run ONNX model
        sess = rt.InferenceSession(
            "random_forest.onnx", providers=["CPUExecutionProvider"]
        )

        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name
        pred_onx = sess.run([label_name], {input_name: x_test})[0]
        pred_proba = np.array([list(d.values()) for d in pred_onx])

        # Compute metrics over time windows
        metrics = prediction_test(
            project,
            x_new,
            project.dataset.date_feature,
            project.dataset.target,
            pred_proba,
        )

        # Add dataset ID to metrics and save
        for m in metrics:
            m.dataset_id = project.dataset.id
        get_logger().debug(f"Saving {len(metrics)} metrics.")

        post_metrics(project_id, metrics)
    else:
        get_logger().error(f"Failed to fetch data {response.status_code}")


if __name__ == "__main__":
    evaluate_dataset(1)  # Run evaluation for project ID 1
    get_logger().debug("Task done, exiting.")
