import datetime
import pandas as pd

import pytest
import uuid

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.metrics.data_metrics.category_metrics import (
    data_new_category_metric,
    data_missing_category_metric,
)


@pytest.fixture
def data_shape_cat() -> DataShape:
    date = Feature(
        pid=uuid.uuid4(),
        name="date",
        feature_type=FeatureType.DATE,
        min_value=0,
        max_value=0,
    )

    target = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.FLOAT,
        min_value=0.0,
        max_value=5.0,
    )

    features: list[Feature] = []
    for i in range(1, 4):
        f = Feature(
            pid=uuid.uuid4(),
            name=f"f_{i}",
            feature_type=FeatureType.CATEGORICAL,
            min_value=None,
            max_value=None,
        )
        features.append(f)

    datashape = DataShape(features=features, date=date, target=target)

    return datashape


def gen_test_dataset_cat(data_shape_cat: DataShape, data: pd.DataFrame) -> Dataset:
    data["date"] = pd.to_datetime(data["date"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape_cat, data=data)


def gen_ref_dataset_cat(data_shape_cat: DataShape, data: pd.DataFrame) -> Dataset:
    # Training dataset does not have date column
    # data["col_timestamp"] = pd.to_datetime(data["col_timestamp"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape_cat,
        data=data,
    )


def test_data_new_category_metric_no_categories(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
) -> None:
    metrics = data_new_category_metric(data_shape, ref_dataset, test_dataset)
    for m in metrics:
        assert m.score == 0
        assert m.name in (
            "Number of New Categories in Test Data Feature",
            "Number of New Category Instances in Test Data",
        )
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)


def test_data_new_category_metric_categories_1(data_shape_cat: DataShape) -> None:
    # Create reference data
    data_ref = pd.DataFrame(
        {
            "f_1": [1, 1, 0, 1, 0],
            "f_2": [1, 0, 0, 2, 1],
            "f_3": [1, 2, 0, 0, 1],
            "target": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    # Create testing data
    data_test = pd.DataFrame(
        {
            "date": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
            ],
            "f_1": [0, 6, 7],
            "f_2": [-4, -4, 1],
            "f_3": [1, 0, 0],
            "target": [0, 1, 1],
        }
    )

    # Generate custom datasets
    ref_dataset: Dataset = gen_ref_dataset_cat(data_shape_cat, data_ref)
    test_dataset: Dataset = gen_test_dataset_cat(data_shape_cat, data_test)

    f_1, f_2, f_3, m = data_new_category_metric(
        data_shape_cat, ref_dataset, test_dataset
    )
    assert f_1.score == 2
    assert f_2.score == 1
    assert f_3.score == 0
    assert m.score == 3


def test_data_new_category_metric_categories_2(data_shape_cat: DataShape) -> None:
    # Create reference data
    data_ref = pd.DataFrame(
        {
            "f_1": ["a", "b", "a", "b"],
            "f_2": ["c", "d", "d", "c"],
            "f_3": ["e", "e", "f", "f"],
            "target": ["y", "y", "n", "n"],
        }
    )

    # Create testing data
    data_test = pd.DataFrame(
        {
            "date": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
            ],
            "f_1": ["purple", "b", "b"],
            "f_2": ["d", "c", "c"],
            "f_3": ["f", "e", "k"],
            "target": ["k", "e", "k"],
        }
    )

    # Generate custom datasets
    ref_dataset: Dataset = gen_ref_dataset_cat(data_shape_cat, data_ref)
    test_dataset: Dataset = gen_test_dataset_cat(data_shape_cat, data_test)

    f_1, f_2, f_3, m = data_new_category_metric(
        data_shape_cat, ref_dataset, test_dataset
    )
    assert f_1.score == 1
    assert f_2.score == 0
    assert f_3.score == 1
    assert m.score == 2


def test_data_missing_category_metric_no_categories(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
) -> None:
    metrics = data_missing_category_metric(data_shape, ref_dataset, test_dataset)
    for m in metrics:
        assert m.score == 0
        assert m.name in (
            "Number of Missing Categories in Test Data Feature",
            "Number of Missing Category Instances in Test Data",
        )
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)


def test_data_missing_category_metric_categories(data_shape_cat: DataShape) -> None:
    # Create reference data
    data_ref = pd.DataFrame(
        {
            "f_1": ["a", "b", "a", "b"],
            "f_2": ["c", "d", "d", "c"],
            "f_3": ["e", "e", "f", "f"],
            "target": ["y", "y", "n", "n"],
        }
    )

    # Create testing data
    data_test = pd.DataFrame(
        {
            "date": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
            ],
            "f_1": ["purple", "b", "b"],
            "f_2": ["d", "c", "c"],
            "f_3": ["f", "e", "e"],
            "target": ["k", "e", "k"],
        }
    )

    # Generate custom datasets
    ref_dataset: Dataset = gen_ref_dataset_cat(data_shape_cat, data_ref)
    test_dataset: Dataset = gen_test_dataset_cat(data_shape_cat, data_test)

    f_1, f_2, f_3, m = data_missing_category_metric(
        data_shape_cat, ref_dataset, test_dataset
    )
    assert f_1.score == 1
    assert f_2.score == 0
    assert f_3.score == 0
    assert m.score == 1
