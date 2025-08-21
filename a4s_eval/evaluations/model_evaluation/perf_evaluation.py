import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluations.model_evaluation.registry import model_pred_proba_evaluator


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


@model_pred_proba_evaluator(name="Empty model pred proba evaluator")
def empty_model_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    return []


@model_pred_proba_evaluator(name="Classification Performance evaluator: Accuracy")
def classification_accuracy_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Metric(
        name="Accuracy",
        score=accuracy_score(y_true, y_pred),
        time=date,
    )

    return [metric]


@model_pred_proba_evaluator(name="Classification Performance evaluator: F1 Score")
def classification_f1_score_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Metric(
        name="F1",
        score=f1_score(y_true, y_pred),
        time=date,
    )

    return [metric]


@model_pred_proba_evaluator(name="Classification Performance evaluator: Precision")
def classification_precision_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Metric(
        name="Precision",
        score=precision_score(y_true, y_pred, zero_division=0.0),
        time=date,
    )

    return [metric]


@model_pred_proba_evaluator(name="Classification Performance evaluator: Recall")
def classification_recall_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Metric(
        name="Recall",
        score=recall_score(y_true, y_pred),
        time=date,
    )

    return [metric]


@model_pred_proba_evaluator(
    name="Classification Performance evaluator: Matthews Correlation Coefficient"
)
def classification_matthews_corrcoef_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Metric(
        name="MCC",
        score=matthews_corrcoef(y_true, y_pred),
        time=date,
    )

    return [metric]


@model_pred_proba_evaluator(name="Classification Performance evaluator: RROCAUC")
def classification_roc_auc_evaluator(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Metric]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    metric = Metric(
        name="ROCAUC",
        score=robust_roc_auc_score(y_true, y_pred_proba),
        time=date,
    )

    return [metric]
