"""Every way a test ends is reported once, with its reason.

A test runs inside one state machine (celery_tasks.test_run): a version the index does not list, an install that
fails, a run whose state cannot be read, a plugin that ends well but leaves no measure, a start the engine refuses,
a measure post the engine already has, and a run the platform stopped because its calls kept failing each end the
test in one known way. The run's other tasks are stopped with the signal Celery turns into the soft time limit, so
their own failure handlers run.

The platform stopping a run is the case of a night when the Wi-Fi dropped: the system under test became
unreachable, three calls in a row failed, the platform refused the run's next calls, and the plugin, which treats a
refused call as one more failed question, ended normally with partial results. Such a test is not Done.
"""

import inspect
import json
import types

import pytest
from aisc_service_client import ServiceRefused

from aisc_eval import celery_tasks, plugin_process

from test_plugin_process import (  # noqa: F401  (fixtures)
    TARGET,
    _Process,
    _resource,
    _run,
    _run_with,
    platform,
    worker,
)
from test_run_stop import (  # noqa: F401  (fixtures)
    EVALUATION,
    FAILING,
    PENDING,
    RUNNING,
    broker,
    own_marks,
    stopped,
)
from recorded_pins import recorded_pins

ONE_MEASURE = '[{"name": "score", "score": 1.0}]'


@pytest.fixture(autouse=True)
def the_index_can_be_listed_again(worker, monkeypatch):  # noqa: F811  (the fixture)
    """The faked index lists what it already knows when the worker asks it again."""
    loader = celery_tasks.plugin_loader
    monkeypatch.setattr(loader, "list_packages", lambda refresh=False: loader.discovered_packages, raising=False)


def _ends(call):
    """What a task call ended with: its return value, or the exception it raised."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001  (the test asserts what the exception was)
        return exc


def test_a_version_the_index_does_not_list_fails_the_test_and_stops_the_run(broker):
    ended = _ends(lambda: celery_tasks.run_plugin.run("pkg", "Plugin", "9.9", {}, [], [], EVALUATION, FAILING, pins=recorded_pins("pkg", "9.9")))
    assert isinstance(ended, LookupError)
    assert own_marks(broker) == ["Version '9.9' of package 'pkg' not found."]
    assert stopped(broker) == [PENDING, RUNNING]


def test_an_install_of_a_version_the_index_does_not_list_stops_the_run(broker):
    ended = _ends(lambda: celery_tasks.install_package.run("pkg", "9.9", EVALUATION, [FAILING], pins=recorded_pins("pkg", "9.9")))
    assert isinstance(ended, LookupError)
    assert broker["failed"][:1] == [(EVALUATION, FAILING, "Version '9.9' of package 'pkg' not found.")]
    assert stopped(broker) == [PENDING, RUNNING]


def test_a_run_whose_state_cannot_be_read_fails_the_test_without_starting_it(worker, monkeypatch):
    def unreachable(pid):
        raise ConnectionError("engine not answering")

    monkeypatch.setattr(celery_tasks, "get_evaluation_request", unreachable)
    _ends(_run)
    assert worker["popen"] == []
    reason = getattr(celery_tasks, "RUN_STATE_UNREADABLE", "<a reason that says the run's state could not be read>")
    assert worker["failed"] == [("eval", "ep", reason)]


@pytest.mark.parametrize("leave", [None, "[]"], ids=["no-file", "empty-list"])
def test_a_plugin_that_ends_well_without_measures_fails(worker, leave):
    worker["leave"] = lambda output: leave and (output / "measures.json").write_text(leave)
    ended = _ends(_run)
    assert isinstance(ended, RuntimeError), f"the test ended {ended!r}, as if Done"
    assert "without any measure" in str(ended)
    assert worker["failed"][-1][2] == celery_tasks.NO_MEASURES
    assert "plugin_execution.log" in worker["uploads"]


def test_a_test_the_engine_refuses_to_start_never_starts_its_plugin(worker, monkeypatch):
    worker["leave"] = lambda output: (output / "measures.json").write_text(ONE_MEASURE)

    def refused(*a):
        raise ServiceRefused(409, response=types.SimpleNamespace(status_code=409))

    monkeypatch.setattr(celery_tasks, "mark_plugin_started", refused)
    ended = _ends(_run)
    assert ended == [], f"a refused start ended the test with {ended!r}"
    assert worker["popen"] == [] and worker["failed"] == []


def test_a_refused_start_still_revokes_its_pass(worker, monkeypatch):
    revoked_passes = []
    monkeypatch.setattr(celery_tasks.run_pass, "mint", lambda *a, **kw: "the-pass")
    monkeypatch.setattr(celery_tasks.run_pass, "trace", lambda *a: None)
    monkeypatch.setattr(celery_tasks.run_pass, "revoke", lambda run_pass, run=None: revoked_passes.append(run_pass))

    def refused(*a):
        raise ServiceRefused(409, response=types.SimpleNamespace(status_code=409))

    monkeypatch.setattr(celery_tasks, "mark_plugin_started", refused)
    _ends(lambda: _run_with([_resource("target", TARGET)]))
    assert revoked_passes == ["the-pass"]
    assert worker["failed"] == [], "a start the engine refused was reported as a failure of the test"


def test_the_runs_other_tasks_are_revoked_with_the_stop_signal(broker, monkeypatch):
    sent = []
    monkeypatch.setattr(celery_tasks.celery_app.control, "revoke", lambda tid, **kw: sent.append(kw))
    monkeypatch.setattr(plugin_process.subprocess, "Popen", lambda argv, **kw: _Process(returncode=1))
    _ends(lambda: celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], EVALUATION, FAILING, pins=recorded_pins("pkg", "1.0")))
    assert sent
    assert all(kw == {"terminate": True, "signal": "SIGUSR1"} for kw in sent), sent


def test_a_skipped_test_posts_no_measures(broker, monkeypatch):
    posted = []
    monkeypatch.setattr(celery_tasks, "post_measures", lambda *a: posted.append(a))
    celery_tasks.post_measurements.run([], EVALUATION, FAILING)
    assert posted == [] and broker["failed"] == []


def test_measures_already_recorded_are_not_a_failure(broker, monkeypatch):
    def already(*a):
        raise ServiceRefused(409, response=types.SimpleNamespace(status_code=409))

    monkeypatch.setattr(celery_tasks, "post_measures", already)
    _ends(lambda: celery_tasks.post_measurements.run([{"name": "score", "score": 1.0}], EVALUATION, FAILING))
    assert broker["failed"] == []


def test_run_plugin_runs_inside_the_state_machine():
    assert callable(getattr(celery_tasks, "test_run", None)), "celery_tasks has no test_run state machine"
    assert "with test_run(" in inspect.getsource(celery_tasks.run_plugin)


# ── A run the platform stopped is not Done ───────────────────────────────────

PLATFORM_STOP = "the platform stopped this run: 3 calls in a row failed; the last: timeout"


def _trace_of_a_dropped_network():
    rows = [{"seq": 1, "outcome": "answered", "error": None}]
    rows += [{"seq": n, "outcome": "failed", "error": {"code": "timeout", "detail": "no answer"}} for n in (2, 3, 4)]
    rows += [{"seq": 5, "outcome": "not_sent", "error": {"code": "run_stopped", "detail": PLATFORM_STOP}}]
    lines = [json.dumps(r) for r in rows] + [json.dumps({"seal": {"calls": 5, "digest": "d"}})]
    return ("\n".join(lines) + "\n").encode()


def test_a_test_the_platform_stopped_is_failed_with_the_platforms_reason(worker, platform, monkeypatch):
    finished = []
    monkeypatch.setattr(celery_tasks, "mark_plugin_finished", lambda *a: finished.append(a))
    worker["leave"] = lambda output: (output / "measures.json").write_text(ONE_MEASURE)
    platform["trace"].content = _trace_of_a_dropped_network()
    _ends(lambda: _run_with([_resource("target", TARGET)]))
    assert finished == [], "a test the platform stopped was reported finished, so it shows Done"
    assert [reason for _e, _t, reason in worker["failed"]] == [PLATFORM_STOP]


def test_a_test_the_platform_stopped_still_keeps_its_calls_and_log(worker, platform):
    worker["leave"] = lambda output: (output / "measures.json").write_text(ONE_MEASURE)
    platform["trace"].content = _trace_of_a_dropped_network()
    _ends(lambda: _run_with([_resource("target", TARGET)]))
    assert worker["uploads"][celery_tasks.CALLS_FILE] == platform["trace"].content
    assert "plugin_execution.log" in worker["uploads"]
    assert worker["failed"], "the test was not failed"
