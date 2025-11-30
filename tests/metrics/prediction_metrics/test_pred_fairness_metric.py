import datetime
import pytest
import uuid

import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    Feature,
    FeatureType,
)
from a4s_eval.metrics.prediction_metrics.pred_fairness_metric import (
    classification_fairness_metric,
)


@pytest.fixture
def data_shape_cat() -> DataShape:
    date = Feature(
        pid=uuid.uuid4(),
        name="date",
        feature_type=FeatureType.DATE,
        min_value=0,
        max_value=0,
    )

    target = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.FLOAT,
        min_value=0.0,
        max_value=5.0,
    )

    features: list[Feature] = []
    for i in range(1, 4):
        f = Feature(
            pid=uuid.uuid4(),
            name=f"f_{i}",
            feature_type=FeatureType.CATEGORICAL,
            min_value=None,
            max_value=None,
        )
        features.append(f)

    datashape = DataShape(features=features, date=date, target=target)

    return datashape


@pytest.fixture
def test_dataset_cat(data_shape_cat: DataShape) -> Dataset:
    data = pd.DataFrame(
        {
            "date": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
            ],
            "f_1": ["purple", "orange", "orange", "purple", "orange"],
            "f_2": ["orange", "orange", "purple", "orange", "purple"],
            "f_3": ["orange", "purple", "orange", "purple", "orange"],
            "target": [0, 1, 1, 0, 1],
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape_cat, data=data)


def test_classification_fairness_metric(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred_proba: np.ndarray,
):
    metrics = classification_fairness_metric(
        data_shape, ref_model, test_dataset, y_pred_proba
    )
    for m in metrics:
        assert isinstance(m.name, str)
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)
        assert len(m.description) > 0


def test_classification_fairness_metric_cat(
    data_shape_cat: DataShape,
    ref_model: Model,
    test_dataset_cat: Dataset,
):
    y_pred_proba = np.array(
        [
            [0.78, 0.22],
            [0.36, 0.64],
            [0.08, 0.92],
            [0.58, 0.42],
            [0.51, 0.49],
        ]
    )

    metrics = classification_fairness_metric(
        data_shape_cat, ref_model, test_dataset_cat, y_pred_proba
    )
    for m in metrics:
        assert isinstance(m.name, str)
        assert isinstance(m.score, float)
        assert isinstance(m.time, datetime.datetime)
        assert len(m.description) > 0
