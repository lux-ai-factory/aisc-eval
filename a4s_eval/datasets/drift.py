"""Implementation of data drift detection methods for dataset evaluation.

This module provides functions to detect and measure data drift between reference
and new datasets using various statistical methods depending on feature types.

This module now delegates to the main drift evaluation functions to avoid code duplication.
"""

from datetime import datetime
import pandas as pd

from a4s_eval.data_model.dataset import Feature
from a4s_eval.data_model.metric import Metric
from a4s_eval.data_model.project import Project
from a4s_eval.utils.dates import DateIterator

# Import the actual drift calculation functions from the evaluation module
from a4s_eval.evaluations.data_evaluation.drift_evaluation import (
    numerical_drift_test,
    feature_drift_test,
)


def numerical_drift_metric(
    x_ref: pd.Series, x_new: pd.Series, time: datetime
) -> Metric:
    """Create a metric object for numerical drift using Wasserstein distance.

    Args:
        x_ref: Reference distribution
        x_new: New distribution to compare
        time: Timestamp for the metric

    Returns:
        Metric: Drift metric object with computed score
    """
    drift = numerical_drift_test(x_ref, x_new)
    return Metric(
        name="wasserstein_distance",
        score=drift,
        time=time,
    )


def data_drift_test(
    project: Project,
    x_ref: pd.DataFrame,
    x_new: pd.DataFrame,
    features: list[Feature],
    date_feature: Feature,
) -> list[Metric]:
    """Calculate drift metrics for all features in a dataset over time windows.

    This function maintains backward compatibility while using the improved
    drift calculation functions from the evaluation module.

    Args:
        project: Project configuration containing window size and frequency
        x_ref: Reference dataset
        x_new: New dataset to compare
        features: List of features to analyze
        date_feature: Feature containing temporal information

    Returns:
        list[Metric]: List of drift metrics for each feature and time window
    """
    metrics = []
    for end_date, x_curr in DateIterator(
        date_round="1 D",
        window=project.window_size,
        freq=project.frequency,
        df=x_new,
        date_feature=date_feature.name,
    ):
        for feature in features:
            feature_type = feature.feature_type
            x_ref_feature = x_ref[feature.name]
            x_new_feature = x_curr[feature.name]

            metric = feature_drift_test(
                x_ref_feature, x_new_feature, feature_type, end_date
            )
            metric.feature_pid = feature.pid
            metrics.append(metric)

    return metrics
