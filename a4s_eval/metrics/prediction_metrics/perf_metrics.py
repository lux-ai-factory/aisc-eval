import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


def robust_roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """Calculate ROC AUC score with handling for binary classification probabilities.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities (can be 2D for binary classification)

    Returns:
        np.ndarray: ROC AUC score
    """
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[
            :, 1
        ]  # Use probability of positive class for binary classification
    return roc_auc_score(y_true, y_pred_proba)


@prediction_metric(name="Empty model pred proba metric")
def empty_model_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    return []


@prediction_metric(name="Accuracy")
def classification_accuracy_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="Accuracy",
            score=accuracy_score(y_true, y_pred),
            time=date,
        )
    ]


@prediction_metric(name="Error Rate")
def classification_error_rate_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    accuracy_metric: Measure = classification_accuracy_metric(
        datashape, model, dataset, y_pred_proba, y_pred
    )[0]

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="Error Rate",
            score=1 - accuracy_metric.score,
            time=accuracy_metric.time,
        )
    ]


@prediction_metric(name="Balanced Accuracy")
def classification_balanced_accuracy_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="Balanced Accuracy",
            score=balanced_accuracy_score(y_true, y_pred),
            time=date,
        )
    ]


@prediction_metric(name="Confusion Matrix")
def classification_confusion_matrix_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    matrix = confusion_matrix(y_true, y_pred)
    max_i, max_j = matrix.shape

    return [
        Measure(
            name="Confusion Matrix",
            description=f"({i},{j})/({max_i},{max_j})",
            score=matrix[i][j],
            time=date,
        )
        for i in range(max_i)
        for j in range(max_j)
    ]


@prediction_metric(name="F1 Score")
def classification_f1_score_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="F1",
            score=f1_score(y_true, y_pred, zero_division=0.0),
            time=date,
        )
    ]


@prediction_metric(name="Precision")
def classification_precision_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="Precision",
            score=precision_score(y_true, y_pred, zero_division=0.0),
            time=date,
        )
    ]


@prediction_metric(name="Recall")
def classification_recall_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="Recall",
            score=recall_score(y_true, y_pred, zero_division=0.0),
            time=date,
        )
    ]


@prediction_metric(name="Matthews Correlation Coefficient")
def classification_matthews_corrcoef_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="MCC",
            score=matthews_corrcoef(y_true, y_pred),
            time=date,
        )
    ]


@prediction_metric(name="ROCAUC")
def classification_roc_auc_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="ROCAUC",
            score=robust_roc_auc_score(y_true, y_pred_proba),
            time=date,
        )
    ]


@prediction_metric(name="log_loss")
def classification_log_loss_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="log_loss",
            score=log_loss(y_true, y_pred),
            time=date,
        )
    ]


@prediction_metric(name="Average Entropy")
def classification_average_entropy_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    """
    The entropy is used as a measure of (total) uncertainty
    """
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()

    return [
        Measure(
            name="average_entropy",
            score=stats.entropy(y_pred_proba, axis=1, base=2).mean(),
            time=date,
        )
    ]


@prediction_metric(name="brier_score_loss")
def classification_brier_score_loss_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    return [
        Measure(
            name="brier_score_loss",
            score=brier_score_loss(y_true, y_pred),
            time=date,
        )
    ]
