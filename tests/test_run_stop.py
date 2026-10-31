"""When one test of a run fails, the run stops: whichever task failed (its package's install, the test itself, or
posting its measurements), the run's other tasks are revoked, so those running stop and those queued never start,
and each of its other tests still Pending or Running is marked failed as stopped. A test that already finished
keeps its result. The list of the run's tasks is kept as long as the run can last. The backend and the broker are
faked as test_plugin_process.py fakes them."""

import json
import types
import uuid

import pytest
import requests

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.data_model.evaluation import Evaluation

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


# ── The run's task list ──────────────────────────────────────────────────────

PROJECT = "22222222-2222-2222-2222-222222222222"


def two_packages():
    def test(pid, package):
        return {"pid": pid, "name": "Plugin", "package_name": package, "version": "1.0", "plugin_config": None,
                "evaluation_inputs": []}

    return Evaluation.model_validate({
        "pid": "11111111-1111-1111-1111-111111111111", "project": {"pid": PROJECT, "name": "mijke"},
        "evaluation_plugins": [test(str(uuid.uuid4()), "pkg-a"), test(str(uuid.uuid4()), "pkg-a"),
                               test(str(uuid.uuid4()), "pkg-b")]})


def test_the_runs_task_list_names_every_install_and_test_and_lasts_as_long_as_the_run(monkeypatch):
    stored = {}

    class _Workflow:
        def __or__(self, other):
            return self

        def apply_async(self):
            stored["dispatched"] = True

    monkeypatch.setattr(celery_tasks.plugin_loader, "list_packages", lambda refresh=False: {})
    monkeypatch.setattr(celery_tasks, "get_evaluation", lambda pid: two_packages())
    monkeypatch.setattr(celery_tasks, "get_evaluation_inputs", lambda pid: {})
    monkeypatch.setattr(celery_tasks, "get_project_settings_by_pid", lambda *a: [])
    monkeypatch.setattr(celery_tasks, "group", lambda *a: _Workflow())
    monkeypatch.setattr(celery_tasks, "chain", lambda *a: a)
    fake_app = types.SimpleNamespace(
        backend=types.SimpleNamespace(client=types.SimpleNamespace(
            set=lambda key, value, ex: stored.update(key=key, value=json.loads(value), ex=ex))),
        conf=types.SimpleNamespace(task_time_limit=24300))
    monkeypatch.setattr(celery_tasks, "celery_app", fake_app)

    out = celery_tasks._start_evaluation(EVALUATION)
    assert stored["dispatched"] and stored["key"] == f"eval_tasks:{EVALUATION}"
    # two installs and three tests, all of them revocable; the tests are what the run answers
    assert len(stored["value"]) == 5 and set(out["plugin_task_ids"]) < set(stored["value"])
    # one worker runs a task at a time: 2 installs, 3 tests, 3 posts of their measurements and the finalisation,
    # each as long as a task may take
    assert stored["ex"] == 9 * 24300


# ── Every way a test can fail stops the run once ────────────────────────────


def own_marks(worker):
    return [text for _evaluation, ep, text in worker["failed"] if ep == FAILING]


def test_an_input_that_cannot_be_fetched_fails_the_test_once_and_stops_the_run(broker, monkeypatch):
    def refused(name):
        raise requests.HTTPError("404 for the dataset")

    monkeypatch.setattr(celery_tasks, "get_dataset_file_content", refused)
    with pytest.raises(requests.HTTPError):
        celery_tasks.run_plugin.run(
            "pkg", "Plugin", "1.0", {},
            [{"component_type": "dataset", "name": "data", "data": "data.csv"}], [], EVALUATION, FAILING)
    assert own_marks(broker) == ["404 for the dataset"]
    assert stopped(broker) == [PENDING, RUNNING]
    assert broker["revoked"] == TASKS


def test_an_unsupported_input_fails_the_test_once_and_stops_the_run(broker):
    with pytest.raises(ValueError):
        celery_tasks.run_plugin.run(
            "pkg", "Plugin", "1.0", {}, [{"component_type": "telepathy", "name": "x"}], [], EVALUATION, FAILING)
    assert own_marks(broker) == ["Unsupported component type: telepathy"]
    assert stopped(broker) == [PENDING, RUNNING]


def test_a_failed_venv_fails_the_test_once_and_stops_the_run(broker, monkeypatch):
    monkeypatch.setattr(plugin_process, "run_install",
                        lambda *a, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr="no python"))
    with pytest.raises(RuntimeError):
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], EVALUATION, FAILING)
    assert own_marks(broker) == ["no python"]
    assert stopped(broker) == [PENDING, RUNNING]
    assert broker["revoked"] == TASKS


def test_a_minted_pass_is_revoked_when_the_test_fails_before_its_plugin_starts(broker, monkeypatch):
    from aisc_eval import run_context

    revoked_passes = []
    monkeypatch.setattr(celery_tasks.run_pass, "mint", lambda *a, **kw: "the-pass")
    monkeypatch.setattr(celery_tasks.run_pass, "revoke", revoked_passes.append)

    def unreachable(*a):
        raise ConnectionError("engine not answering")

    monkeypatch.setattr(celery_tasks, "mark_plugin_started", unreachable)
    token = run_context._current.set({"project": "p", "evaluation": EVALUATION, "ticket": "t"})
    try:
        with pytest.raises(ConnectionError):
            celery_tasks.run_plugin.run(
                "pkg", "Plugin", "1.0", {},
                [{"component_type": "resource", "name": "target",
                  "json_value": {"value": "target:p/system"}}], [], EVALUATION, FAILING)
    finally:
        run_context._current.reset(token)
    assert revoked_passes == ["the-pass"]
    assert own_marks(broker) == ["engine not answering"]
    assert stopped(broker) == [PENDING, RUNNING]
