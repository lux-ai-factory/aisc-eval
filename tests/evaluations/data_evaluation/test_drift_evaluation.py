import uuid

import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.evaluations.data_evaluation.drift_evaluation import empty_data_evaluator


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["uuid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("Name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)

@pytest.fixture
def test_dataset(data_shape: DataShape) -> Dataset:
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=pd.read_csv("./tests/data/lcld_v2_test_400.csv")
    )


@pytest.fixture
def ref_dataset(data_shape: DataShape) -> Dataset:
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=pd.read_csv("./tests/data/lcld_v2_train_800.csv"),
    )


def test_smoke(ref_dataset:Dataset, test_dataset: Dataset):
    metrics = empty_data_evaluator(ref_dataset, test_dataset)
    assert len(metrics) == 0
