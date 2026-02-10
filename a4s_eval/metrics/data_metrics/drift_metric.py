import uuid
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric
from a4s_eval.utils.logging import get_logger

logger = get_logger()


def numerical_drift_test(x_ref: "pd.Series[float]", x_new: "pd.Series[float]") -> float:
    """Calculate drift between two numerical distributions using Wasserstein distance.

    Args:
        x_ref: Reference distribution as pandas Series
        x_new: New distribution to compare against reference

    Returns:
        float: Wasserstein distance between the distributions
    """
    logger.debug(
        f"Computing numerical drift test - Reference shape: {x_ref.shape}, "
        f"New shape: {x_new.shape}"
    )
    distance = wasserstein_distance(x_ref.to_numpy(), x_new.to_numpy())
    logger.debug(f"Wasserstein distance computed: {distance}")
    return distance


def categorical_drift_test(x_ref: "pd.Series[int]", x_new: "pd.Series[int]") -> float:
    """Calculate drift between two categorical distributions using Jensen-Shannon distance.

    Args:
        x_ref: Reference distribution as pandas Series
        x_new: New distribution to compare against reference

    Returns:
        float: Jensen-Shannon distance between the distributions
    """
    logger.debug(
        f"Computing categorical drift test - Reference shape: {x_ref.shape}, "
        f"New shape: {x_new.shape}"
    )

    # Get all unique values from both series
    all_categories = pd.Index(x_ref.unique()).union(pd.Index(x_new.unique()))
    logger.debug(f"Total unique categories: {len(all_categories)}")

    # Compute normalized value counts for both distributions
    ref_counts = x_ref.value_counts(normalize=True)
    new_counts = x_new.value_counts(normalize=True)

    # Reindex to ensure both have the same categories (fill missing with 0)
    ref_dist = ref_counts.reindex(all_categories, fill_value=0.0)
    new_dist = new_counts.reindex(all_categories, fill_value=0.0)

    distance = jensenshannon(ref_dist.to_numpy(), new_dist.to_numpy())
    logger.debug(f"Jensen-Shannon distance computed: {distance}")
    return distance


def feature_drift_test(
    feature: Feature,
    feature_pid: uuid.UUID,
    date: pd.Timestamp,
    reference: pd.DataFrame,
    evaluated: pd.DataFrame,
) -> Measure:
    """Calculate drift for a specific feature based on its type.

    Args:
        feature: Feature
        feature_pid: uuid.UUID
        date: Timestamp for the metric
        reference: Reference dataframe
        evaluated: Evaluated dataframe

    Returns:
        Measure: Drift metric object with computed score

    Raises:
        ValueError: If feature type is not supported
    """
    feature_type = feature.feature_type
    logger.debug(f"Processing feature: {feature.name} (type: {feature_type})")
    logger.debug(f"Computing feature drift test for feature type: {feature_type}")

    score = None
    metric_name = None

    if feature_type == FeatureType.INTEGER or feature_type == FeatureType.FLOAT:
        score = numerical_drift_test(reference[feature.name], evaluated[feature.name])
        metric_name = "wasserstein_distance"
        logger.debug(f"Created numerical drift metric: {metric_name} = {score}")

    elif feature_type == FeatureType.CATEGORICAL:
        score = categorical_drift_test(reference[feature.name], evaluated[feature.name])
        metric_name = "jensenshannon"
        logger.debug(f"Created categorical drift metric: {metric_name} = {score}")

    else:
        logger.error(f"Unsupported feature type: {feature_type}")
        raise ValueError(f"Feature type {feature_type} not supported")

    if score is not None:
        logger.debug(
            f"Added metric for feature {feature.name}: {metric_name} = {score}"
        )
        return Measure(
            name=metric_name, score=score, time=date, feature_pid=feature_pid
        )


@data_metric(name="Data drift")
def data_drift_metric(
    datashape: DataShape, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """Calculate drift for all features between reference and evaluated datasets.

    This metric compares the reference dataset against the evaluated dataset
    for the current time window. The time windowing is handled at a higher level
    by the evaluation_tasks.py DateIterator.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Measure]: List of drift metrics for each feature
    """
    logger.debug(
        "Starting data drift evaluation - "
        f"Reference shape: {reference.data.shape}, Evaluated shape: {evaluated.data.shape}"
    )

    # Get the current date from the evaluated dataset
    date_feature = datashape.date.name
    date = pd.to_datetime(evaluated.data[date_feature]).max()
    logger.debug(f"Evaluation date: {date}")

    # Get the current date from the evaluated dataset
    logger.debug(f"Evaluation date: {date}")

    logger.debug(f"Processing {len(reference.shape.features)} features")

    # Get feature name / feature pid mapping from test dataset
    test_feature_name_pid_mapping = {
        _feature.name: _feature.pid for _feature in evaluated.shape.features
    }

    compute_metric = partial(
        feature_drift_test,
        date=date,
        reference=reference.data,
        evaluated=evaluated.data,
    )
    it = (
        (feature, test_feature_name_pid_mapping.get(feature.name, None))
        for feature in datashape.features
    )

    # FIXME: perhaps set max_workers
    with ThreadPoolExecutor() as pool:
        metrics = list(pool.map(compute_metric, *(zip(*it))))

    logger.debug(f"Data drift evaluation completed - Generated {len(metrics)} metrics")
    return metrics
