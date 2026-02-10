import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0.0),
        "Precision": precision_score(y_true, y_pred, zero_division=0.0),
        "Recall": recall_score(y_true, y_pred, zero_division=0.0),
    }


@prediction_metric(name="Fairness")
def classification_fairness_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    n_bins: int = 10,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Get feature name / feature pid mapping from test dataset
    feature_name_pid_mapping = {
        _feature.name: _feature.pid for _feature in dataset.shape.features
    }

    categorical_features = (
        feature.name
        for feature in datashape.features
        if feature.feature_type == FeatureType.CATEGORICAL
    )

    return [
        Measure(
            name=metric_name,
            score=metric_score,
            time=date,
            description=f"feature:{feat_name}, category:{category}",
            feature_pid=feature_name_pid_mapping[feat_name],
        )
        for feat_name in categorical_features
        for category in dataset.data[feat_name].unique()
        if (mask := (dataset.data[feat_name] == category)) is not None
        for metric_name, metric_score in classification_metrics(
            y_true[mask], y_pred[mask]
        ).items()
    ]
