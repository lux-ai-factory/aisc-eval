import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric


@data_metric(name="New Category in Test Data")
def data_new_category_metric(
    datashape: DataShape, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    Calculates if there are new categories found in the test set not in the reference set.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Measure]: List of metrics about the category differences
    """
    # Get the current date from the evaluated dataset
    date = pd.to_datetime(evaluated.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    ref_df = reference.data
    eval_df = evaluated.data

    measures: list[Measure] = []
    total_instances: int = 0

    # For each feature in evaluated, check if category not in reference
    # Use expected datashape features to ensure consistent processing
    for feature in datashape.features:
        # Only process categorical features
        if feature.feature_type != FeatureType.CATEGORICAL:
            continue

        feat_name = feature.name

        # Categories
        ref_categories = set(ref_df[feat_name].unique())
        eval_categories = set(eval_df[feat_name].unique())

        new_categories = eval_categories - ref_categories  # set difference

        total_instances += len(new_categories)

        # Add measures
        diff_categories = Measure(
            name="Number of New Categories in Test Data Feature",
            score=float(len(new_categories)),
            time=date,
            # description too long
            # description=f"feature:{feat_name}, categories:{list(new_categories)}",
            description=f"feature:{feat_name}",
            feature_pid=feature.pid,
        )
        measures.append(diff_categories)

    diff_total_data = Measure(
        name="Number of New Category Instances in Test Data",
        score=total_instances,
        time=date,
    )
    measures.append(diff_total_data)

    return measures


@data_metric(name="Missing Category in Test Data")
def data_missing_category_metric(
    datashape: DataShape, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    Calculates if there are categories found in the reference set missing in the test set.

    Args:
        reference: The reference dataset (model dataset)
        evaluated: The evaluated dataset (current time window)

    Returns:
        list[Measure]: List of metrics about the category differences
    """
    # Get the current date from the evaluated dataset
    date = pd.to_datetime(evaluated.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    ref_df = reference.data
    eval_df = evaluated.data

    measures: list[Measure] = []
    total_instances: int = 0

    # For each feature in evaluated, check if category not in test
    # Use expected datashape features to ensure consistent processing
    for feature in datashape.features:
        # Only process categorical features
        if feature.feature_type != FeatureType.CATEGORICAL:
            continue

        feat_name = feature.name

        # Categories
        ref_categories = set(ref_df[feat_name].unique())
        eval_categories = set(eval_df[feat_name].unique())

        new_categories = ref_categories - eval_categories  # set difference

        total_instances += len(new_categories)

        # Add measures
        diff_categories = Measure(
            name="Number of Missing Categories in Test Data Feature",
            score=float(len(new_categories)),
            time=date,
            # description too long
            # description=(f"feature:{feat_name}, categories:{list(new_categories)}"),
            description=(f"feature:{feat_name}"),
            feature_pid=feature.pid,
        )
        measures.append(diff_categories)

    diff_total_data = Measure(
        name="Number of Missing Category Instances in Test Data",
        score=total_instances,
        time=date,
    )
    measures.append(diff_total_data)

    return measures
