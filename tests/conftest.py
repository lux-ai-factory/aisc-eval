"""PyTest configuration and fixtures for A4S EVAL tests.

This module configures the test environment and provides shared fixtures
for all test modules. It ensures a clean database state before each test session.
"""

import uuid
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model


@pytest.fixture(scope="module")
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv")
    metadata["feature_type"] = metadata["feature_type"].str.capitalize()
    metadata = metadata.to_dict(orient="records")

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


@pytest.fixture(scope="module")
def test_dataset(data_shape: DataShape) -> Dataset:
    data = pd.read_csv("./tests/data/lcld_v2_test_400.csv")
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture(scope="module")
def ref_dataset(data_shape: DataShape) -> Dataset:
    data = pd.read_csv("./tests/data/lcld_v2_train_800.csv")
    data["issue_d"] = pd.to_datetime(data["issue_d"])
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
def y_pred_proba(ref_model: Model, test_dataset: Dataset) -> np.ndarray:
    session = ort.InferenceSession("./tests/data/lcld_v2_random_forest.onnx")
    df = test_dataset.data[[f.name for f in test_dataset.shape.features]]
    x_test = df.astype(np.double).to_numpy()

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[1].name
    pred_onx = session.run([label_name], {input_name: x_test})[0]
    y_pred_proba = np.array([list(d.values()) for d in pred_onx])

    return y_pred_proba
