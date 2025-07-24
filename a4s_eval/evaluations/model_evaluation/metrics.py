"""Implementation of model evaluation metrics for classification models.

This module provides functions to calculate various classification metrics
for evaluating model performance, including both prediction-based metrics
and probability-based metrics.
"""

import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from a4s_eval.data_model.dataset import Feature
from a4s_eval.data_model.metric import Metric
from a4s_eval.data_model.project import Project
from a4s_eval.utils.dates import DateIterator


def robust_roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """Calculate ROC AUC score with handling for binary classification probabilities.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities (can be 2D for binary classification)

    Returns:
        np.ndarray: ROC AUC score
    """
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[
            :, 1
        ]  # Use probability of positive class for binary classification
    return roc_auc_score(y_true, y_pred_proba)


# Dictionary mapping metric names to their sklearn implementations
pred_classification_metric = {
    "F1": f1_score,  # F1 score (harmonic mean of precision and recall)
    "MCC": matthews_corrcoef,  # Matthews Correlation Coefficient
    "Accuracy": accuracy_score,  # Simple accuracy
    "Precision": precision_score,  # Precision (positive predictive value)
    "Recall": recall_score,  # Recall (sensitivity, true positive rate)
}

# Dictionary for probability-based metrics
str2proba_classification_metric = {
    "ROCAUC": robust_roc_auc_score  # Area Under the ROC Curve
}


def prediction_metric_test(
    y_true: np.ndarray, y_pred_proba: np.ndarray, current_time: Timestamp
) -> list[Metric]:
    """Calculate all classification metrics for a set of predictions.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        current_time: Timestamp for the metrics

    Returns:
        list[Metric]: List of computed metrics
    """
    out = []
    y_pred = np.argmax(
        y_pred_proba, axis=1
    )  # Convert probabilities to class predictions

    # Calculate prediction-based metrics
    for name, score_f in pred_classification_metric.items():
        out.append(
            Metric(
                name=name,
                score=score_f(y_true, y_pred),
                time=current_time,
            )
        )

    # Calculate probability-based metrics
    for name, score_f in str2proba_classification_metric.items():
        out.append(
            Metric(
                name=name,
                score=score_f(y_true, y_pred_proba),
                time=current_time,
            )
        )

    return out


def prediction_test(
    project: Project,
    x_new: pd.DataFrame,
    date_feature: Feature,
    target_feature: Feature,
    y_new_pred_proba: np.ndarray,
) -> list[Metric]:
    """Evaluate model predictions over time windows.

    Args:
        project: Project configuration with window size and frequency
        x_new: Test dataset
        date_feature: Feature containing temporal information
        target_feature: Target feature being predicted
        y_new_pred_proba: Model's probability predictions

    Returns:
        list[Metric]: List of evaluation metrics for each time window
    """
    metrics = []
    for end_date, x_curr in DateIterator(
        date_round="1 D",
        window=project.window_size,
        freq=project.frequency,
        df=x_new,
        date_feature=date_feature.name,
    ):
        y_curr = x_curr[target_feature.name].to_numpy()
        y_curr_pred_proba = y_new_pred_proba[x_curr.index]

        metrics_curr = prediction_metric_test(y_curr, y_curr_pred_proba, end_date)
        metrics.extend(metrics_curr)

    return metrics
