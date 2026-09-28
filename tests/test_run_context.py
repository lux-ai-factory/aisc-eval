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


def _task(run):
    return SimpleNamespace(request=SimpleNamespace(aisc_run=run))


def test_enter_sets_the_run_of_the_task_and_leave_clears_it():
    run = {"project": "p", "evaluation": "e", "ticket": "t"}
    task = _task(run)
    run_context._enter(task=task)
    assert run_context.current() == run
    run_context._leave(task=task)
    assert run_context.current() is None


def test_a_nested_enter_and_leave_restores_the_outer_run():
    outer_run = {"project": "p", "evaluation": "outer", "ticket": "t"}
    inner_run = {"project": "p", "evaluation": "inner", "ticket": "t"}
    outer, inner = _task(outer_run), _task(inner_run)
    run_context._enter(task=outer)
    run_context._enter(task=inner)  # an eager call inside a running task
    assert run_context.current() == inner_run
    run_context._leave(task=inner)
    assert run_context.current() == outer_run
    run_context._leave(task=outer)
    assert run_context.current() is None


def test_leave_without_a_stored_token_still_clears():
    token = run_context._current.set({"project": "p", "evaluation": "e", "ticket": "t"})
    try:
        run_context._leave(task=_task(None))
        assert run_context.current() is None
        run_context._leave()
        assert run_context.current() is None
    finally:
        run_context._current.reset(token)
