"""Basic integration tests for the AISC Evaluation service.

These tests verify core functionality without requiring external services.
They are designed to run in CI environments reliably.
"""

import uuid

import pytest
from pydantic import ValidationError

from aisc_eval.data_model.evaluation import Evaluation, Project
from aisc_eval.data_model.project import Plugin


def test_celery_tasks_import():
    """Test that Celery tasks can be imported without errors."""
    try:
        from aisc_eval import celery_tasks

        # Just verify the module loads, don't try to connect to Redis
        assert hasattr(celery_tasks, "celery_app")
    except ImportError as e:
        pytest.fail(f"Failed to import Celery tasks: {e}")


def test_evaluation_requires_a_project():
    """An evaluation without a project cannot be built."""
    with pytest.raises(ValidationError):
        Evaluation.model_validate({"pid": str(uuid.uuid4())})


def test_evaluation_model_carries_project_and_plugins():
    """An evaluation exposes its project and the plugins it will run."""
    evaluation = Evaluation.model_validate(
        {
            "pid": str(uuid.uuid4()),
            "project": {"pid": str(uuid.uuid4()), "name": "LCLD"},
            "evaluation_plugins": [
                {
                    "pid": str(uuid.uuid4()),
                    "name": "DemoPlugin",
                    "package_name": "demo-pkg",
                    "version": "0.1.0",
                    "evaluation_inputs": [],
                }
            ],
        }
    )

    assert isinstance(evaluation.project, Project)
    assert evaluation.project.name == "LCLD"
    assert isinstance(evaluation.evaluation_plugins[0], Plugin)
    assert evaluation.evaluation_plugins[0].name == "DemoPlugin"
