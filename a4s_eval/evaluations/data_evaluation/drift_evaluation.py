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


@data_evaluator(name="Data drift")
def data_drift_evaluator(reference: Dataset, evaluated: Dataset) -> list[Metric]:
    """Calculate drift for a specific feature based on its type."""

    date = pd.to_datetime(evaluated.data[reference.shape.date.name]).max()

    out = []

    for feature in reference.shape.features:
        feature_type = feature.feature_type
        if feature_type == FeatureType.INTEGER or feature_type == FeatureType.FLOAT:
            out.append(
                Metric(
                    name="wasserstein_distance",
                    score=numerical_drift_test(
                        reference.data[feature.name],
                        evaluated.data[feature.name],
                    ),
                    time=date.to_pydatetime(),
                    feature_pid=feature.pid,
                )
            )

        elif feature_type == FeatureType.CATEGORICAL:
            out.append(
                Metric(
                    name="jensenshannon",
                    score=categorical_drift_test(
                        reference.data[feature.name],
                        evaluated.data[feature.name],
                    ),
                    time=date.to_pydatetime(),
                    feature_pid=feature.pid,
                )
            )

    return out
