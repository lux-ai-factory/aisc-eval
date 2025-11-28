import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric


@data_metric(name="Distribution")
def data_distribution_metric(
    datashape: DataShape, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    Counts data points in and outside mean +/- 3*std.
    Checks evaluated datset for integer and float feature types.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Measure]: Metrics counting data points
    """
    # Get the current date from the evaluated dataset
    date = pd.to_datetime(evaluated.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    eval_df = evaluated.data
    measures: list[Measure] = []

    # For each feature in datashape, check dataset for outliers
    for feature in datashape.features:
        # Only process numeric features
        if feature.feature_type not in (FeatureType.INTEGER, FeatureType.FLOAT):
            continue

        feat_name = feature.name
        values = eval_df[feat_name]

        mean = values.mean()
        std = values.std()

        min_val = mean - 3 * std
        max_val = mean + 3 * std

        # check outliers
        feat_pass = (values >= min_val) & (values <= max_val)

        in_bounds = Measure(
            name="Distribution Inside",
            score=float(feat_pass.sum()),
            time=date,
            feature_pid=feature.pid,
        )
        measures.append(in_bounds)

        out_bounds = Measure(
            name="Distribution Outside",
            score=float((~feat_pass).sum()),
            time=date,
            feature_pid=feature.pid,
        )
        measures.append(out_bounds)

    return measures
