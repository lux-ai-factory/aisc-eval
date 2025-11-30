import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric

from a4s_eval.metrics.prediction_metrics.perf_metrics import (
    classification_accuracy_metric,
    classification_f1_score_metric,
    classification_precision_metric,
    classification_recall_metric,
)


@prediction_metric(name="Fairness")
def classification_fairness_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()

    metrics = [
        classification_accuracy_metric,
        classification_f1_score_metric,
        classification_precision_metric,
        classification_recall_metric,
    ]

    eval_df = dataset.data
    measures: list[Measure] = []

    # Get feature name / feature pid mapping from test dataset
    feature_name_pid_mapping = {
        _feature.name: _feature.pid for _feature in dataset.shape.features
    }

    # For each feature in datashape, check fairness
    for feature in datashape.features:
        # Only process categorical features
        if feature.feature_type != FeatureType.CATEGORICAL:
            continue

        feat_name = feature.name

        # Categories
        for category in eval_df[feat_name].unique():
            # Look specifically at the entries following the measure
            mask = eval_df[feat_name] == category

            # Apply mask
            df_masked = eval_df[mask]
            dataset_masked = Dataset(
                pid=dataset.pid,
                shape=datashape,
                data=df_masked,
            )
            y_pred_proba_masked = y_pred_proba[mask]

            # Metric
            for metric in metrics:
                m_list = metric(datashape, model, dataset_masked, y_pred_proba_masked)
                m = m_list[0]

                # Add measures
                fair_m = Measure(
                    name=f"Fairness_{m.name}",
                    score=m.score,
                    time=date,
                    description=f"feature:{feat_name}, category:{category}",
                    feature_pid=feature_name_pid_mapping[feat_name],
                )
                measures.append(fair_m)

    return measures
