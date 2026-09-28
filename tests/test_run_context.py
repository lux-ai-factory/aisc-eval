import inspect
import uuid
from types import SimpleNamespace

from aisc_eval import celery_tasks, run_context


def test_seans_signatures_are_back():
    assert list(inspect.signature(celery_tasks.run_evaluation.run).parameters) == ["evaluation_pid"]
    assert "platform_pid" not in inspect.signature(celery_tasks.install_package.run).parameters
    assert "ticket" not in inspect.signature(celery_tasks.install_package.run).parameters


def test_run_of_reads_the_header_either_way():
    run = {"project": str(uuid.uuid4()), "evaluation": str(uuid.uuid4()), "ticket": "t"}
    assert run_context.run_of(SimpleNamespace(aisc_run=run)) == run
    assert run_context.run_of(SimpleNamespace(headers={"aisc_run": run})) == run
    assert run_context.run_of(SimpleNamespace(headers=None)) is None


def test_a_task_published_inside_a_run_inherits_it():
    run = {"project": "p", "evaluation": "e", "ticket": "t"}
    token = run_context._current.set(run)
    try:
        headers = {}
        run_context._forward(headers=headers)
        assert headers == {"aisc_run": run}
    finally:
        run_context._current.reset(token)


def test_standalone_publishes_no_header():
    headers = {}
    run_context._forward(headers=headers)
    assert headers == {}
