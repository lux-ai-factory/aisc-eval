import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric


def check_numeric_constraints(
    datashape: DataShape, dataset: Dataset, date, name: str
) -> list[Measure]:
    """
    Checks if any constraints are violated in the dataset.
    Based on the feature constrains in datashape.
    Checks integer and float feature types.
    """
    df = dataset.data
    measures: list[Measure] = []

    # Get feature name / feature pid mapping from test dataset
    # feature_name_pid_mapping = {
    #     _feature.name: _feature.pid for _feature in dataset.shape.features
    # }

    # For each feature in datashape, check if dataset meets constraints
    for feature in datashape.features:
        # Only process numeric features
        if feature.feature_type not in (FeatureType.INTEGER, FeatureType.FLOAT):
            continue

        feat_name = feature.name
        values = df[feat_name]

        # Some were stored as string
        min_val = float(feature.min_value) if feature.min_value is not None else None
        max_val = float(feature.max_value) if feature.max_value is not None else None

        # check min
        if min_val is not None:
            mask_min = values < min_val
            lower_metric = Measure(
                name=f"Number of {name} Constraint Violations (Lower)",
                score=float(mask_min.sum()),
                time=date,
                feature_pid=None,
                description=f"{feature.name}",
            )
            measures.append(lower_metric)

        # check max
        if max_val is not None:
            mask_max = values > max_val
            upper_metric = Measure(
                name=f"Number of {name} Constraint Violations (Upper)",
                score=float(mask_max.sum()),
                time=date,
                feature_pid=None,
                description=f"{feature.name}",
            )
            measures.append(upper_metric)

    return measures


@data_metric(name="Reference Constraints")
def data_reference_numeric_constraints_metric(
    datashape: DataShape, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    Checks if any constraints are violated in the reference dataset.
    Based on the feature constrains in datashape.
    Checks integer and float feature types.
    Date is taken from evaluated.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Measure]: Metrics counting constraint violations
    """
    # Get the current date from the evaluated dataset
    date = pd.to_datetime(evaluated.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    return check_numeric_constraints(datashape, reference, date, "Reference")


@data_metric(name="Evaluated Constraints")
def data_evaluated_numeric_constraints_metric(
    datashape: DataShape, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    Checks if any constraints are violated in the evaluated dataset.
    Based on the feature constrains in datashape.
    Checks integer and float feature types.
    Date is taken from evaluated.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Measure]: Metrics counting constraint violations
    """
    # Get the current date from the evaluated dataset
    date = pd.to_datetime(evaluated.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    return check_numeric_constraints(datashape, evaluated, date, "Evaluated")
