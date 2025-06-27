import uuid

import numpy as np
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.evaluations.data_evaluation.drift_evaluation import (
    data_drift_evaluator,
    empty_data_evaluator,
)


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(data_shape: DataShape) -> Dataset:
    data = pd.read_csv("./tests/data/lcld_v2_test_400.csv")
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(data_shape: DataShape) -> Dataset:
    data = pd.read_csv("./tests/data/lcld_v2_train_800.csv")
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


def test_smoke(ref_dataset: Dataset, test_dataset: Dataset):
    metrics = empty_data_evaluator(ref_dataset, test_dataset)
    assert len(metrics) == 0


def test_data_drift_evaluator_generates_metrics(
    ref_dataset: Dataset, test_dataset: Dataset
):
    """
    # This function tests the data drift evaluator to ensure it generates some metrics.
    """

    metrics = data_drift_evaluator(ref_dataset, test_dataset)
    assert len(metrics) == len(ref_dataset.shape.features)


def test_data_drift_evaluator_metrics_not_nan(
    ref_dataset: Dataset, test_dataset: Dataset
):
    metrics = data_drift_evaluator(ref_dataset, test_dataset)
    assert all(not np.isnan(metric.score) for metric in metrics)
