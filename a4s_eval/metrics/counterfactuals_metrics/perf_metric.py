import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, jensenshannon

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.counterfactuals_metric_registry import counterfactual_metric
from a4s_eval.utils.logging import get_logger

logger = get_logger()

@counterfactual_metric(name="Empty counterfactual metric")
def empty_counterfactual_metric(
    expected_datashape: DataShape, factual_scaled: pd.DataFrame, counterfactuals: pd.DataFrame
) -> list[Measure]:
    return []


def numerical_distance(factuals: pd.Series, counterfactuals: pd.Series) -> float:
    """Calculate Euclidean distance between two numerical data points.

    Args:
        factuals: query instances
        counterfactuals: generated counterfactual instances

    Returns:
        float: Euclidean distance between the points
    """
    logger.debug(
        f"Computing numerical distance test - factuals shape: {factuals.shape}, New shape: {counterfactuals.shape}"
    )
    distance = euclidean(factuals, counterfactuals)
    logger.debug(f"Euclidean distance computed: {distance}")
    return distance



def feature_distance_test(
    factuals: pd.Series,
    counterfactuals: pd.Series,
    date: datetime.datetime,
) -> Measure:
    """Calculate distances between factual and counterfactuals.

    Args:
        factuals: query instances
        counterfactuals: generated counterfactual instances
        feature_type: Type of the feature (numerical or categorical)
        date: Timestamp for the metric

    Returns:
        Measure: Distance metric object with computed score

    Raises:
        ValueError: If feature type is not supported
    """

    score = numerical_distance(factuals, counterfactuals)
    metric = Measure(
        name="euclidean",
        score=score,
        time=date,
    )
    logger.debug(f"Created numerical drift metric: {metric.name} = {metric.score}")
    return metric

    


@counterfactual_metric(name="Counterfactuals distance")
def counterfactual_distance_metric(
    expected_datashape: DataShape, factual_scaled: pd.DataFrame, counterfactuals: pd.DataFrame
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


    # Get the current date from the evaluated dataset
    date = datetime.datetime.now()
    logger.debug(f"Evaluation date: {date}")

    metrics = []
    logger.debug(f"Processing {len(expected_datashape.features)} features")

    # Identify feature types
    numeric_feats = [
        f.name for f in expected_datashape.features 
        if f.feature_type == FeatureType.INTEGER or f.feature_type == FeatureType.FLOAT
    ]
    # TODO: Add categorical distance computation
    
    if numeric_feats:
        # Loop through all features in the project expected datashape
        for idx in factual_scaled.index:
            factual_row = factual_scaled.loc[idx]
            counterfactual_row = counterfactuals.loc[idx]

            metric = feature_distance_test(factual_row, counterfactual_row, date)

            # Set correct feature pid (from test dataset)
            metrics.append(metric)
    else:
        logger.error(f"Unsupported feature types")
        raise ValueError(f"Non-numerical feature type not supported")
    
    logger.debug(f"Distance evaluation completed - Generated {len(metrics)} metrics")
    return metrics
