"""When one test of a run fails, the run stops: whichever task failed (its package's install, the test itself, or
posting its measurements), the run's other tasks are revoked, so those running stop and those queued never start,
and each of its other tests still Pending or Running is marked failed as stopped. A test that already finished
keeps its result. The backend and the broker are
faked as test_plugin_process.py fakes them."""

import json
import types

import pytest

from aisc_eval import celery_tasks, plugin_process

from test_plugin_process import _Process, worker  # noqa: F401  (the fixture)

EVALUATION = "eval"
FAILING = "ep"
RUNNING, PENDING, DONE, FAILED = "ep-running", "ep-pending", "ep-done", "ep-failed"
TASKS = ["task-of-failing", "task-running", "task-pending", "install-other"]


def engine_run(status="Running", own_status="Running"):
    """GET /evaluations/{pid}?include=plugin as the engine answers it, reduced to what stopping reads."""
    return {"status": status,
            "evaluation_plugins": [{"pid": FAILING, "status": own_status},
                                   {"pid": RUNNING, "status": "Running"},
                                   {"pid": PENDING, "status": "Pending"},
                                   {"pid": DONE, "status": "Done"},
                                   {"pid": FAILED, "status": "Failed"}]}


@pytest.fixture
def broker(worker, monkeypatch):  # noqa: F811  (the fixture)
    """The run's task list in Redis and the revokes sent, with the stopping itself left real."""
    revoked = []
    keys = {f"eval_tasks:{EVALUATION}": json.dumps(TASKS)}
    fake_app = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            client=types.SimpleNamespace(get=keys.get),
            store_result=lambda *a, **kw: None,
        ),
        control=types.SimpleNamespace(revoke=lambda tid, **kw: revoked.append(tid)),
        conf=celery_tasks.celery_app.conf,
    )
    monkeypatch.setattr(celery_tasks, "celery_app", fake_app)
    monkeypatch.setattr(celery_tasks, "_revoke_sibling_tasks", REAL_REVOKE)
    monkeypatch.setattr(celery_tasks, "get_evaluation_request", lambda pid: engine_run())
    worker["revoked"] = revoked
    return worker


REAL_REVOKE = celery_tasks._revoke_sibling_tasks


def stopped(worker):
    return sorted(ep for _evaluation, ep, text in worker["failed"] if text == celery_tasks.STOPPED)


def test_a_test_that_fails_stops_the_runs_other_tests(broker, monkeypatch):
    monkeypatch.setattr(plugin_process.subprocess, "Popen", lambda argv, **kw: _Process(returncode=1))
    with pytest.raises(RuntimeError):
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], EVALUATION, FAILING)
    assert broker["failed"][0][:2] == (EVALUATION, FAILING)
    assert stopped(broker) == [PENDING, RUNNING]
    # run_plugin.run has no task id of its own here: every task of the list is revoked
    assert broker["revoked"] == TASKS


def test_a_failed_install_stops_the_tests_of_the_runs_other_packages(broker, monkeypatch):
    monkeypatch.setattr(plugin_process, "install_user",
                        lambda: plugin_process.PluginUser(0, 0, None, switch=False))
    monkeypatch.setattr(plugin_process, "prepare_install_folder", lambda *a, **kw: None)
    monkeypatch.setattr(plugin_process, "run_install",
                        lambda *a, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr="no such package"))
    with pytest.raises(RuntimeError):
        celery_tasks.install_package.run("pkg", "1.0", EVALUATION, [FAILING])
    assert broker["failed"][0] == (EVALUATION, FAILING, "no such package")
    assert stopped(broker) == [PENDING, RUNNING]
    assert broker["revoked"] == TASKS


def test_a_failed_post_of_the_measurements_stops_the_runs_other_tests(broker, monkeypatch):
    def refuse(*a, **kw):
        raise RuntimeError("500 posting measures")

    monkeypatch.setattr(celery_tasks, "post_measures", refuse)
    with pytest.raises(RuntimeError):
        celery_tasks.post_measurements.run([], EVALUATION, FAILING)
    assert broker["failed"][0] == (EVALUATION, FAILING, "500 posting measures")
    assert stopped(broker) == [PENDING, RUNNING]
    assert broker["revoked"] == TASKS


def test_a_run_whose_tests_cannot_be_read_still_revokes_its_tasks(broker, monkeypatch):
    def unreachable(*a):
        raise ConnectionError("engine not answering")

    monkeypatch.setattr(celery_tasks, "get_evaluation_request", unreachable)
    monkeypatch.setattr(celery_tasks, "post_measures", unreachable)
    with pytest.raises(ConnectionError):
        celery_tasks.post_measurements.run([], EVALUATION, FAILING)
    assert stopped(broker) == []
    assert broker["revoked"] == TASKS


def test_a_test_that_starts_after_the_run_failed_is_marked_stopped(broker, monkeypatch):
    monkeypatch.setattr(celery_tasks, "get_evaluation_request",
                        lambda pid: engine_run(status="Failed", own_status="Pending"))
    assert celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], EVALUATION, FAILING) == []
    assert broker["failed"] == [(EVALUATION, FAILING, celery_tasks.STOPPED)]


def test_a_test_already_marked_stopped_keeps_its_mark_when_it_starts(broker, monkeypatch):
    monkeypatch.setattr(celery_tasks, "get_evaluation_request",
                        lambda pid: engine_run(status="Failed", own_status="Failed"))
    assert celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], EVALUATION, FAILING) == []
    assert broker["failed"] == []
