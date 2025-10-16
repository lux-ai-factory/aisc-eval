import datetime
import numpy as np

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.metrics.prediction_metrics.perf_metric import (
    classification_accuracy_metric,
    classification_f1_score_metric,
    classification_matthews_corrcoef_metric,
    classification_precision_metric,
    classification_recall_metric,
    classification_roc_auc_metric,
    empty_model_metric,
)


def test_smoke(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
) -> None:
    metrics = empty_model_metric(data_shape, ref_model, test_dataset, y_pred_proba)
    assert len(metrics) == 0


def test_model_accuracy_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_accuracy_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "Accuracy"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_f1_score_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_f1_score_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "F1"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_precision_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_precision_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "Precision"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_recall_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_recall_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "Recall"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_matthews_corrcoef_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_matthews_corrcoef_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "MCC"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_model_roc_auc_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_roc_auc_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "ROCAUC"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)
