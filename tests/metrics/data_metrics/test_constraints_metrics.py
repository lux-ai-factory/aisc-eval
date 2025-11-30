import datetime
import pandas as pd

import pytest
import uuid

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.metrics.data_metrics.constraints_metrics import (
    data_reference_numeric_constraints_metric,
    data_evaluated_numeric_constraints_metric,
)


@pytest.fixture
def data_shape_int() -> DataShape:
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
            feature_type=FeatureType.INTEGER,
            min_value=0,
            max_value=100,
        )
        features.append(f)

    datashape = DataShape(features=features, date=date, target=target)

    return datashape


@pytest.fixture
def test_dataset_int(data_shape_int: DataShape) -> Dataset:
    data = pd.DataFrame(
        {
            "date": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
            ],
            "f_1": [-9, 6, 7],
            "f_2": [-4, 120, 1],
            "f_3": [-9, 0, 0],
            "target": [0, 1, 1],
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape_int, data=data)


@pytest.fixture
def ref_dataset_int(data_shape_int: DataShape) -> Dataset:
    data = pd.DataFrame(
        {
            "f_1": [1, 1, 0, 100, 0],
            "f_2": [1, 0, 0, -2, 101],
            "f_3": [1, 200, 0, 0, 1],
            "target": [0, 0, 1, 1, 1],
        }
    )
    # Training dataset does not have date column
    # data["col_timestamp"] = pd.to_datetime(data["col_timestamp"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape_int,
        data=data,
    )


def test_data_reference_numeric_constraints_metric(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
) -> None:
    metrics = data_reference_numeric_constraints_metric(
        data_shape, ref_dataset, test_dataset
    )
    for m in metrics:
        assert m.score == 0
        assert m.name in (
            "Number of Reference Constraint Violations (Lower)",
            "Number of Reference Constraint Violations (Upper)",
        )
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)


def test_data_reference_numeric_constraints_metric_int(
    data_shape_int: DataShape, ref_dataset_int: Dataset, test_dataset_int: Dataset
) -> None:
    l_1, u_1, l_2, u_2, l_3, u_3 = data_reference_numeric_constraints_metric(
        data_shape_int, ref_dataset_int, test_dataset_int
    )

    assert l_1.score == 0
    assert u_1.score == 0
    assert l_2.score == 1
    assert u_2.score == 1
    assert l_3.score == 0
    assert u_3.score == 1


def test_data_evaluated_numeric_constraints_metric(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
) -> None:
    metrics = data_evaluated_numeric_constraints_metric(
        data_shape, ref_dataset, test_dataset
    )
    for m in metrics:
        assert m.score == 0
        assert m.name in (
            "Number of Evaluated Constraint Violations (Lower)",
            "Number of Evaluated Constraint Violations (Upper)",
        )
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)


def test_data_evaluated_numeric_constraints_metric_int(
    data_shape_int: DataShape, ref_dataset_int: Dataset, test_dataset_int: Dataset
) -> None:
    l_1, u_1, l_2, u_2, l_3, u_3 = data_evaluated_numeric_constraints_metric(
        data_shape_int, ref_dataset_int, test_dataset_int
    )

    assert l_1.score == 1
    assert u_1.score == 0
    assert l_2.score == 1
    assert u_2.score == 1
    assert l_3.score == 1
    assert u_3.score == 0
