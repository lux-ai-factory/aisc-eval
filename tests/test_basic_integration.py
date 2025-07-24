"""Basic integration tests for A4S Evaluation service.

These tests verify core functionality without requiring external services.
They are designed to run in CI environments reliably.
"""

import uuid
from datetime import datetime

import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Feature, FeatureType
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluations.data_evaluation.drift_evaluation import (
    empty_data_evaluator,
)


def test_empty_evaluator():
    """Test that empty evaluator returns empty metrics."""
    from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType

    # Create minimal test data with proper Feature objects
    target_feature = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.CATEGORICAL,
        min_value=0,
        max_value=1,
    )

    date_feature = Feature(
        pid=uuid.uuid4(),
        name="date",
        feature_type=FeatureType.DATE,
        min_value="2023-01-01",
        max_value="2023-12-31",
    )

    data_shape = DataShape(features=[], target=target_feature, date=date_feature)

    test_data = pd.DataFrame(
        {
            "target": [0, 1, 0],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        }
    )

    ref_dataset = Dataset(pid=uuid.uuid4(), shape=data_shape, data=test_data)
    test_dataset = Dataset(pid=uuid.uuid4(), shape=data_shape, data=test_data)

    metrics = empty_data_evaluator(ref_dataset, test_dataset)
    assert len(metrics) == 0


def test_celery_tasks_import():
    """Test that Celery tasks can be imported without errors."""
    try:
        from a4s_eval import celery_tasks

        # Just verify the module loads, don't try to connect to Redis
        assert hasattr(celery_tasks, "celery_app")
    except ImportError as e:
        pytest.fail(f"Failed to import Celery tasks: {e}")


def test_evaluation_data_model():
    """Test that evaluation data models work correctly."""

    # Test metric creation
    metric = Metric(name="test_metric", score=0.95, time=datetime.now())
    assert metric.name == "test_metric"
    assert metric.score == 0.95

    # Test feature creation
    feature = Feature(
        pid=uuid.uuid4(),
        name="test_feature",
        feature_type=FeatureType.FLOAT,
        min_value=0.0,
        max_value=1.0,
    )
    assert feature.name == "test_feature"
    assert feature.feature_type == FeatureType.FLOAT


def test_drift_evaluation_import():
    """Test that drift evaluation functions can be imported."""
    try:
        from a4s_eval.evaluations.data_evaluation.drift_evaluation import (
            categorical_drift_test,
            data_drift_evaluator,
            empty_data_evaluator,
            numerical_drift_test,
        )

        # Just verify they can be imported
        assert callable(data_drift_evaluator)
        assert callable(empty_data_evaluator)
        assert callable(numerical_drift_test)
        assert callable(categorical_drift_test)
    except ImportError as e:
        pytest.fail(f"Failed to import drift evaluation functions: {e}")
