"""The dispatcher claims a run once, registers it for the sweep and shows it Processing.

The queue can deliver the run's dispatcher twice. The first delivery claims the run (one key, set only when absent),
dispatches its installs and tests, registers the run with its deadline so a later sweep can close what the worker
loses, and marks the run Processing. A second delivery finds the claim and dispatches nothing. A run a person
stopped before its dispatch is not dispatched. A dispatch that fails releases the claim and fails the run, and one
test that cannot be marked failed does not keep the others or the run from being marked.
"""

import contextlib
import fnmatch
import importlib
import time
import types
import uuid

import pytest
import requests

from aisc_eval import celery_tasks, run_context
from aisc_eval.data_model.evaluation import Evaluation

EVALUATION = "11111111-1111-1111-1111-111111111111"
PROJECT = "22222222-2222-2222-2222-222222222222"
FIRST, SECOND = (
    "33333333-3333-3333-3333-333333333331",
    "33333333-3333-3333-3333-333333333332",
)


def _registry():
    try:
        return importlib.import_module("aisc_eval.run_registry")
    except ModuleNotFoundError:
        raise AssertionError(
            "the worker keeps no registry of the runs it dispatched (aisc_eval.run_registry)"
        ) from None


class FakeRedis:
    """The result backend's Redis client, as a dict."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    def delete(self, *keys):
        return sum(1 for key in keys if self.data.pop(key, None) is not None)

    def scan_iter(self, match="*"):
        return [key for key in list(self.data) if fnmatch.fnmatch(key, match)]


def _run_of_three_tests():
    def test(package):
        return {
            "pid": str(uuid.uuid4()),
            "name": "Plugin",
            "package_name": package,
            "version": "1.0",
            "plugin_config": None,
            "evaluation_inputs": [],
        }

    return Evaluation.model_validate(
        {
            "pid": EVALUATION,
            "project": {"pid": PROJECT, "name": "mijke"},
            "evaluation_plugins": [test("pkg-a"), test("pkg-b"), test("pkg-c")],
        }
    )


class _Dispatch(types.SimpleNamespace):
    pass


@pytest.fixture
def dispatch(monkeypatch):
    seen = _Dispatch(
        redis=FakeRedis(),
        applied=0,
        apply_raises=None,
        status="Pending",
        processing=[],
        processing_raises=None,
        failed=[],
        plugin_failed=[],
    )

    class _Workflow:
        def __or__(self, other):
            return self

        def apply_async(self, *a, **kw):
            if seen.apply_raises:
                raise seen.apply_raises
            seen.applied += 1

    def mark_processing(pid):
        if seen.processing_raises:
            raise seen.processing_raises
        seen.processing.append(pid)
        return "Processing"

    run = _run_of_three_tests()
    monkeypatch.setattr(
        celery_tasks.plugin_loader, "list_packages", lambda refresh=False: {}
    )
    monkeypatch.setattr(celery_tasks, "get_evaluation", lambda pid: run)
    monkeypatch.setattr(
        celery_tasks,
        "get_evaluation_request",
        lambda pid: {
            "status": seen.status,
            "evaluation_plugins": [
                {"pid": str(t.pid), "status": "Pending"} for t in run.evaluation_plugins
            ],
        },
    )
    monkeypatch.setattr(celery_tasks, "get_evaluation_inputs", lambda pid: {})
    monkeypatch.setattr(celery_tasks, "get_project_settings_by_pid", lambda *a: [])
    monkeypatch.setattr(celery_tasks, "group", lambda *a: _Workflow())
    monkeypatch.setattr(celery_tasks, "chain", lambda *a: a)
    monkeypatch.setattr(celery_tasks, "mark_processing", mark_processing, raising=False)
    monkeypatch.setattr(
        celery_tasks, "mark_failed", lambda pid: seen.failed.append(pid)
    )
    monkeypatch.setattr(
        celery_tasks,
        "mark_plugin_failed",
        lambda pid, ep, message: seen.plugin_failed.append((pid, ep, message)),
    )
    fake_app = types.SimpleNamespace(
        backend=types.SimpleNamespace(client=seen.redis),
        conf=types.SimpleNamespace(task_time_limit=24300, task_soft_time_limit=24000),
    )
    monkeypatch.setattr(celery_tasks, "celery_app", fake_app)
    return seen


def test_the_first_delivery_claims_the_run_dispatches_it_and_marks_it_processing(
    dispatch,
):
    out = celery_tasks._start_evaluation(EVALUATION)
    assert dispatch.applied == 1
    assert dispatch.processing == [EVALUATION], "the run is never shown Processing"
    [entry] = _registry().registered(dispatch.redis)
    assert (
        set(entry.tests.values()) == set(out["plugin_task_ids"])
        and len(entry.installs) == 3
    )


def test_a_second_delivery_of_the_dispatcher_dispatches_nothing(dispatch):
    first = celery_tasks._start_evaluation(EVALUATION)
    second = celery_tasks._start_evaluation(EVALUATION)
    assert dispatch.applied == 1, "a second delivery dispatched the run's tests again"
    assert second["plugin_task_ids"] == first["plugin_task_ids"]


def test_a_run_stopped_before_its_dispatch_is_not_dispatched(dispatch):
    dispatch.status = "Failed"
    out = celery_tasks._start_evaluation(EVALUATION)
    assert dispatch.applied == 0, "a run a person stopped was dispatched"
    assert out["plugin_task_ids"] == []
    assert dispatch.redis.get(f"eval_tasks:{EVALUATION}") is None


def test_a_dispatch_that_fails_releases_the_claim_and_fails_the_run(dispatch):
    dispatch.apply_raises = ConnectionError("broker down")
    celery_tasks.run_evaluation.run(EVALUATION)
    assert dispatch.redis.get(f"eval_tasks:{EVALUATION}") is None
    assert dispatch.failed == [EVALUATION]
    assert all(
        m.startswith("the run could not start: ConnectionError")
        for _, _, m in dispatch.plugin_failed
    )
    assert _registry().registered(dispatch.redis) == []


def test_the_deadline_is_before_the_runs_ticket_expires(dispatch):
    registry = _registry()
    expiry = int(time.time()) + 3600
    acting_for = getattr(run_context, "acting_for", None)
    assert (
        acting_for is not None
    ), "the worker cannot act as a run outside a task (run_context.acting_for)"
    with acting_for(
        {"project": PROJECT, "evaluation": EVALUATION, "ticket": f"{expiry}.mac"}
    ):
        celery_tasks._start_evaluation(EVALUATION)
    [entry] = registry.registered(dispatch.redis)
    assert entry.deadline <= expiry - registry.DEADLINE_MARGIN_S


def test_a_processing_mark_that_fails_does_not_stop_the_run(dispatch):
    dispatch.processing_raises = requests.ConnectionError("x")
    out = celery_tasks._start_evaluation(EVALUATION)
    assert out["plugin_task_ids"] and dispatch.applied == 1
    assert _registry().registered(
        dispatch.redis
    ), "the run was not registered for the sweep"


def test_one_failed_mark_does_not_stop_the_others(dispatch, monkeypatch):
    monkeypatch.setattr(
        celery_tasks,
        "get_evaluation_request",
        lambda pid: {
            "status": "Pending",
            "evaluation_plugins": [{"pid": FIRST}, {"pid": SECOND}],
        },
    )

    def mark(pid, ep, message):
        if ep == FIRST:
            raise requests.ConnectionError("engine away")
        dispatch.plugin_failed.append((pid, ep, message))

    monkeypatch.setattr(celery_tasks, "mark_plugin_failed", mark)
    with contextlib.suppress(Exception):
        celery_tasks._fail_before_start(EVALUATION, RuntimeError("boom"))
    assert [ep for _, ep, _ in dispatch.plugin_failed] == [SECOND]
    assert dispatch.failed == [EVALUATION]
