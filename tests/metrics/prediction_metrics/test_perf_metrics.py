import datetime
import numpy as np

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.metrics.prediction_metrics.perf_metrics import (
    classification_accuracy_metric,
    classification_error_rate_metric,
    classification_balanced_accuracy_metric,
    classification_confusion_matrix_metric,
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


def test_classification_accuracy_metric(
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


def test_classification_error_rate_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_error_rate_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "Error Rate"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_classification_balanced_accuracy_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_balanced_accuracy_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) == 1
    assert metrics[0].name == "Balanced Accuracy"
    assert isinstance(metrics[0].score, float)
    assert isinstance(metrics[0].time, datetime.datetime)


def test_classification_confusion_matrix_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_confusion_matrix_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) >= 4  # minimum 2*2 matrix

    for metric in metrics:
        print(metric)
        assert metric.name == "Confusion Matrix"
        assert metric.description != ""
        assert isinstance(metric.score, float)
        assert isinstance(metric.time, datetime.datetime)


def test_f1_score_metric(
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


def test_precision_metric(
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


def test_recall_metric(
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


def test_matthews_corrcoef_metric(
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


def test_roc_auc_metric(
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
