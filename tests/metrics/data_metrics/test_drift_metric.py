import uuid

import numpy as np
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.metrics.data_metrics.drift_metric import (
    data_drift_metric,
)


def test_data_drift_metric_generates_metrics(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    """
    # This function tests the data drift metric to ensure it generates some metrics.
    """

    metrics = data_drift_metric(data_shape, ref_dataset, test_dataset)
    assert len(metrics) == len(ref_dataset.shape.features)


def test_data_drift_metric_metrics_not_nan(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    metrics = data_drift_metric(data_shape, ref_dataset, test_dataset)
    assert all(not np.isnan(metric.score) for metric in metrics)
