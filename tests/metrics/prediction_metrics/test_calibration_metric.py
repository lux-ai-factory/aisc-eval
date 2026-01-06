import datetime
import uuid

import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metrics.prediction_metrics.calibration_metric import (
    classification_calibration_score_metric,
)


def test_model_calibration_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_calibration_score_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    assert len(metrics) >= 2
    ECE_metric: Measure = metrics[-2]
    MCE_metric: Measure = metrics[-1]

    assert ECE_metric.name == "ECE"
    assert MCE_metric.name == "MCE"

    for metric in [ECE_metric, MCE_metric]:
        assert isinstance(metric.score, float)
        assert 0 <= metric.score <= 1
        assert isinstance(metric.time, datetime.datetime)


def generate_expected_formats(data_shape: DataShape, y_true: np.ndarray) -> Dataset:
    dates = pd.date_range("2025-01-01", periods=len(y_true), freq="D")
    df: pd.DataFrame = pd.DataFrame(
        {
            data_shape.target.name: y_true,
            data_shape.date.name: dates,
        }
    )

    dummy_dataset = Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=df,
    )

    return dummy_dataset


def test_model_calibration_value_evaluation_1(data_shape: DataShape, ref_model: Model):
    dummy_data_shape = data_shape
    dummy_model = ref_model

    # Hardcoded test data
    # Source from towardsdatascience.com
    y_pred_proba = np.array(
        [
            [0.78, 0.22],
            [0.36, 0.64],
            [0.08, 0.92],
            [0.58, 0.42],
            [0.49, 0.51],
            [0.85, 0.15],
            [0.30, 0.70],
            [0.63, 0.37],
            [0.17, 0.83],
        ]
    )

    y_true = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1])
    expected_ece = 0.10444
    expected_mce = 0.20000
    n_bins = 5

    # Generate expected formats
    dummy_dataset = generate_expected_formats(dummy_data_shape, y_true)

    # Run metrics
    metrics = classification_calibration_score_metric(
        dummy_data_shape, dummy_model, dummy_dataset, y_pred_proba, n_bins
    )

    # Run assertions
    assert len(metrics) >= 2
    ECE_metric: Measure = metrics[-2]
    MCE_metric: Measure = metrics[-1]

    assert ECE_metric.name == "ECE"
    assert MCE_metric.name == "MCE"

    for metric in [ECE_metric, MCE_metric]:
        assert isinstance(metric.score, float)
        assert 0 <= metric.score <= 1
        assert isinstance(metric.time, datetime.datetime)

    assert abs(ECE_metric.score - expected_ece) < 0.00001
    assert abs(MCE_metric.score - expected_mce) < 0.00001


def test_model_calibration_value_evaluation_2(data_shape: DataShape, ref_model: Model):
    dummy_data_shape = data_shape
    dummy_model = ref_model

    # Hardcoded test data
    # Source from towardsdatascience.com
    y_pred_proba = np.array(
        [
            [0.25, 0.2, 0.22, 0.18, 0.15],
            [0.16, 0.06, 0.5, 0.07, 0.21],
            [0.06, 0.03, 0.8, 0.07, 0.04],
            [0.02, 0.03, 0.01, 0.04, 0.9],
            [0.4, 0.15, 0.16, 0.14, 0.15],
            [0.15, 0.28, 0.18, 0.17, 0.22],
            [0.07, 0.8, 0.03, 0.06, 0.04],
            [0.1, 0.05, 0.03, 0.75, 0.07],
            [0.25, 0.22, 0.05, 0.3, 0.18],
            [0.12, 0.09, 0.02, 0.17, 0.6],
        ]
    )

    y_true = np.array([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])
    expected_ece = 0.312
    expected_mce = 0.600
    n_bins = 10

    # Generate expected formats
    # Generate expected formats
    dummy_dataset = generate_expected_formats(dummy_data_shape, y_true)

    # Run metrics
    metrics = classification_calibration_score_metric(
        dummy_data_shape, dummy_model, dummy_dataset, y_pred_proba, n_bins
    )

    # Run assertions
    assert len(metrics) >= 2
    ECE_metric: Measure = metrics[-2]
    MCE_metric: Measure = metrics[-1]

    assert ECE_metric.name == "ECE"
    assert MCE_metric.name == "MCE"

    for metric in [ECE_metric, MCE_metric]:
        assert isinstance(metric.score, float)
        assert 0 <= metric.score <= 1
        assert isinstance(metric.time, datetime.datetime)

    assert abs(ECE_metric.score - expected_ece) < 0.001
    assert abs(MCE_metric.score - expected_mce) < 0.001
