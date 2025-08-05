import uuid
import datetime
import pytest

import numpy as np
import pandas as pd
import onnxruntime as ort

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.evaluations.model_evaluation.perf_evaluation import (
    empty_model_evaluator,
    classification_accuracy_evaluator,
    classification_roc_auc_evaluator,
    classification_f1_score_evaluator,
    classification_precision_evaluator,
    classification_recall_evaluator,
    classification_matthews_corrcoef_evaluator,
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


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def y_pred_proba(ref_model: Model, test_dataset: Dataset) -> np.ndarray:
    session = ort.InferenceSession("./tests/data/lcld_v2_random_forest.onnx")
    df = test_dataset.data[[f.name for f in test_dataset.shape.features]]
    x_test = df.astype(np.double).to_numpy()

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[1].name
    pred_onx = session.run([label_name], {input_name: x_test})[0]
    y_pred_proba = np.array([list(d.values()) for d in pred_onx])

    return y_pred_proba


def test_smoke(ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray):
    metrics = empty_model_evaluator(ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 0


def test_model_accuracy_evaluation(
    ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray
):
    metrics = classification_accuracy_evaluator(ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 1
    assert metrics[0].name == "Accuracy"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_f1_score_evaluation(
    ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray
):
    metrics = classification_f1_score_evaluator(ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 1
    assert metrics[0].name == "F1"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_precision_evaluation(
    ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray
):
    metrics = classification_precision_evaluator(ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 1
    assert metrics[0].name == "Precision"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_recall_evaluation(
    ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray
):
    metrics = classification_recall_evaluator(ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 1
    assert metrics[0].name == "Recall"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_matthews_corrcoef_evaluation(
    ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray
):
    metrics = classification_matthews_corrcoef_evaluator(
        ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "MCC"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_roc_auc_evaluation(
    ref_model: Model, test_dataset: Dataset, y_pred_proba: np.ndarray
):
    metrics = classification_roc_auc_evaluator(ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 1
    assert metrics[0].name == "ROCAUC"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)
