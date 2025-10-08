import uuid

import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.metric_registries.data_metric_registry import (
    DataMetric,
    data_metric_registry,
)


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv")
    metadata['feature_type'] = metadata['feature_type'].str.capitalize()
    metadata = metadata.to_dict(
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


def test_non_empty_registry():
    assert len(data_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", [e[1] for e in data_metric_registry])
def test_data_metric_registry_contains_evaluator(
    evaluator_function: DataMetric,
    data_shape: DataShape,
    ref_dataset: Dataset,
    test_dataset: Dataset,
):
    metrics = evaluator_function(data_shape, ref_dataset, test_dataset)

    assert len(metrics) > 0
