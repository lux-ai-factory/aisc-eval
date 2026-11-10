"""Basic integration tests for VERA Evaluation service.

These tests verify core functionality without requiring external services.
They are designed to run in CI environments reliably.
"""

import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from unittest import mock

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


def test_install_package_failure_marks_evaluation_failed():
    """Failed package install must fail the plugins AND the evaluation.

    Regression: the link_error errback may not fire for chord/chain members, so
    install_package itself must mark the evaluation failed or it stays Pending.
    """
    from aisc_eval import celery_tasks

    evaluation_pid = uuid.uuid4()
    plugin_pids = [uuid.uuid4(), uuid.uuid4()]

    marked_failed = []
    marked_plugins = []
    stub_loader = mock.MagicMock()
    stub_loader.discovered_packages = {
        "demo-pkg": {
            "1.0.0": {"source": "local", "pkg_root": Path(__file__).resolve().parent}
        }
    }

    def fail_install(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["uv", "pip", "install"], stderr="boom")

    with (
        mock.patch.object(
            celery_tasks, "mark_failed", side_effect=lambda pid: marked_failed.append(pid)
        ),
        mock.patch.object(celery_tasks, "mark_plugin_failed",
                          side_effect=lambda eid, pid, msg: marked_plugins.append(pid)),
        mock.patch.object(celery_tasks, "plugin_loader", stub_loader),
        mock.patch.object(subprocess, "run", side_effect=fail_install),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            celery_tasks.install_package.run(
                "demo-pkg", "1.0.0", evaluation_pid, plugin_pids
            )

    assert marked_failed == [evaluation_pid]
    assert marked_plugins == plugin_pids


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
