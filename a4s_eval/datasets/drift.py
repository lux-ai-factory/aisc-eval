"""Implementation of data drift detection methods for dataset evaluation.

This module provides functions to detect and measure data drift between reference
and new datasets using various statistical methods depending on feature types.
"""

from datetime import datetime
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

from a4s_eval.data_model.dataset import Feature, FeatureType
from a4s_eval.data_model.metric import Metric
from a4s_eval.data_model.project import Project
from a4s_eval.utils.dates import DateIterator


def numerical_drift_test(x_ref: pd.Series, x_new: pd.Series) -> float:
    """Calculate drift between two numerical distributions using Wasserstein distance.
    
    Args:
        x_ref: Reference distribution as pandas Series
        x_new: New distribution to compare against reference
    
    Returns:
        float: Wasserstein distance between the distributions
    """
    return wasserstein_distance(x_ref, x_new)


def categorical_drift_test(x_ref: pd.Series, x_new: pd.Series) -> float:
    """Calculate drift between two categorical distributions using Jensen-Shannon distance.
    
    Args:
        x_ref: Reference distribution as pandas Series
        x_new: New distribution to compare against reference
    
    Returns:
        float: Jensen-Shannon distance between the distributions
    """
    return jensenshannon(x_ref, x_new)


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


def feature_drift_test(
    x_ref: pd.Series,
    x_new: pd.Series,
    feature_type: FeatureType,
    date: pd.Timestamp,
) -> Metric:
    """Calculate drift for a specific feature based on its type.
    
    Args:
        x_ref: Reference distribution for the feature
        x_new: New distribution to compare
        feature_type: Type of the feature (numerical or categorical)
        date: Timestamp for the metric
    
    Returns:
        Metric: Drift metric object with computed score
    
    Raises:
        ValueError: If feature type is not supported
    """
    if feature_type == FeatureType.INTEGER or feature_type == FeatureType.FLOAT:
        return Metric(
            name="wasserstein_distance",
            score=numerical_drift_test(x_ref, x_new),
            time=date.to_pydatetime(),
        )

    elif feature_type == FeatureType.CATEGORICAL:
        return Metric(
            name="jensenshannon",
            score=numerical_drift_test(x_ref, x_new),
            time=date.to_pydatetime(),
        )
    else:
        raise ValueError(f"Feature type {feature_type} not supported")


def data_drift_test(
    project: Project,
    x_ref: pd.DataFrame,
    x_new: pd.DataFrame,
    features: list[Feature],
    date_feature: Feature,
) -> list[Metric]:
    """Calculate drift metrics for all features in a dataset over time windows.
    
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
        x_curr_distribution = pd.concat([x_ref, x_curr], axis=0)

        for feature in features:
            feature_type = feature.feature_type
            x_ref_feature = x_ref[feature.name]
            x_new_feature = x_curr_distribution[feature.name]

            metric = feature_drift_test(
                x_ref_feature, x_new_feature, feature_type, end_date
            )
            metric.feature_id = feature.id
            metric.dataset_id = project.dataset.id
            metrics.append(metric)
    
    return metrics
