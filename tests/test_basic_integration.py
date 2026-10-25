"""Basic integration tests for AISC Evaluation service.

These tests verify core functionality without requiring external services.
They are designed to run in CI environments reliably.
"""

from datetime import datetime

import pytest
from aisc_plugin_interface import Measure


def test_celery_tasks_import():
    """Test that Celery tasks can be imported without errors."""
    try:
        from aisc_eval import celery_tasks

        # Just verify the module loads, don't try to connect to Redis
        assert hasattr(celery_tasks, "celery_app")
    except ImportError as e:
        pytest.fail(f"Failed to import Celery tasks: {e}")


def test_measure_from_plugin_interface():
    """Test that the plugin interface's measure model works for the service."""
    metric = Measure(name="test_metric", score=0.95, time=datetime.now())
    assert metric.name == "test_metric"
    assert metric.score == 0.95
