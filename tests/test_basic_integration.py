"""Basic integration tests for VERA Evaluation service.

These tests verify core functionality without requiring external services.
They are designed to run in CI environments reliably.
"""

import uuid
from datetime import datetime

import pytest

from aisc_eval.data_model.evaluation import Evaluation, Project


def test_celery_tasks_import():
    """Test that Celery tasks can be imported without errors."""
    try:
        from aisc_eval import celery_tasks

        # Just verify the module loads, don't try to connect to Redis
        assert hasattr(celery_tasks, "celery_app")
    except ImportError as e:
        pytest.fail(f"Failed to import Celery tasks: {e}")


def test_evaluation_data_model():
    """Test that evaluation data models work correctly."""
    from aisc_plugin_interface import Measure

    # Test metric creation
    metric = Measure(name="test_metric", score=0.95, time=datetime.now())
    assert metric.name == "test_metric"
    assert metric.score == 0.95

    # Test evaluation creation
    project = Project(pid=uuid.uuid4(), name="test-project")
    evaluation = Evaluation(pid=uuid.uuid4(), project=project, evaluation_plugins=[])
    assert evaluation.project.name == "test-project"
    assert evaluation.evaluation_plugins == []
