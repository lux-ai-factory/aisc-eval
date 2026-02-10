import numpy as np
import pandas as pd
import pytest
import uuid

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
    Model,
)
from a4s_eval.metrics.regression_metrics.perf_metrics import (
    empty_regression_metric,
    simple_demo_metric,
    mean_squared_error_metric,
)


@pytest.fixture(scope="module")
def data_shape() -> DataShape:
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

    datashape = DataShape(features=[], date=date, target=target)

    return datashape


@pytest.fixture(scope="module")
def test_dataset(data_shape: DataShape) -> Dataset:
    data = pd.DataFrame(
        {
            "date": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
            ],
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
            "target": [2.5, 3.5, 4.5],
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture(scope="module")
def ref_dataset(data_shape: DataShape) -> Dataset:
    data = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [5.0, 6.0, 7.0, 8.0, 9.0],
            "target": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # Training dataset does not have date column
    # data["col_timestamp"] = pd.to_datetime(data["col_timestamp"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture(scope="module")
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture(scope="module")
def y_pred(test_dataset: Dataset) -> np.ndarray:
    y_pred = np.array([2.7, 3.6, 4.4])
    return y_pred


def test_smoke(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred: np.ndarray,
) -> None:
    metrics = empty_regression_metric(data_shape, ref_model, test_dataset, y_pred)
    assert len(metrics) == 0


def test_simple_demo_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred: np.ndarray,
) -> None:
    metrics = simple_demo_metric(data_shape, ref_model, test_dataset, y_pred)
    assert len(metrics) == 1
    assert metrics[0].name == "score_mean_value_metric"


def test_mean_squared_error_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred: np.ndarray,
) -> None:
    metrics = mean_squared_error_metric(data_shape, ref_model, test_dataset, y_pred)
    assert len(metrics) == 1
    assert metrics[0].score == pytest.approx(0.02)
