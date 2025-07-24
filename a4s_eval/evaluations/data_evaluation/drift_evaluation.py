import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from a4s_eval.data_model.evaluation import Dataset, FeatureType
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluations.data_evaluation.registry import data_evaluator


@data_evaluator(name="Empty data evaluator")
def empty_data_evaluator(reference: Dataset, evaluated: Dataset) -> list[Metric]:
    return []


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
    # Get all unique values from both series
    all_categories = pd.Index(x_ref.unique()).union(pd.Index(x_new.unique()))

    # Compute normalized value counts for both distributions
    ref_counts = x_ref.value_counts(normalize=True)
    new_counts = x_new.value_counts(normalize=True)

    # Reindex to ensure both have the same categories (fill missing with 0)
    ref_dist = ref_counts.reindex(all_categories, fill_value=0.0)
    new_dist = new_counts.reindex(all_categories, fill_value=0.0)

    return jensenshannon(ref_dist.values, new_dist.values)


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
            score=categorical_drift_test(
                x_ref, x_new
            ),  # Fixed: was using numerical_drift_test
            time=date.to_pydatetime(),
        )
    else:
        raise ValueError(f"Feature type {feature_type} not supported")


@data_evaluator(name="Data drift")
def data_drift_evaluator(reference: Dataset, evaluated: Dataset) -> list[Metric]:
    """Calculate drift for all features between reference and evaluated datasets.

    This evaluator compares the reference dataset against the evaluated dataset
    for the current time window. The time windowing is handled at a higher level
    by the evaluation_tasks.py DateIterator.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Metric]: List of drift metrics for each feature
    """

    # Get the current date from the evaluated dataset
    date = pd.to_datetime(evaluated.data[reference.shape.date.name]).max()

    metrics = []

    for feature in reference.shape.features:
        feature_type = feature.feature_type
        x_ref_feature = reference.data[feature.name]
        x_new_feature = evaluated.data[feature.name]

        metric = feature_drift_test(x_ref_feature, x_new_feature, feature_type, date)
        metric.feature_pid = feature.pid
        metrics.append(metric)

    return metrics
