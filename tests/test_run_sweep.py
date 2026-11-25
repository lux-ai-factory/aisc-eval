"""The sweep closes the tests the worker lost, ran past their limit, or never started.

Every five minutes, and once soon after the worker starts, the sweep reads each run the dispatcher registered and
fails, with a stated reason, each of its tests that cannot end any more on its own: one whose task ended without
reporting (the worker stopped or its process was killed), one running past its time limit, one still queued when
the run's time ran out. A run whose tests all ended but which was never closed is finalized. A run that has ended
leaves the registry; one whose ticket expired is dropped. One run that cannot be read never stops the others.

The night the Wi-Fi dropped is the case behind it: the worker could reach neither the target nor the engine, so its
own reports of the failures were refused or never arrived. Once the engine answers again, the sweep closes each
of those tests once, with its reason, and the run with them.
"""

import datetime
import fnmatch
import importlib
import json
import types

import pytest

from aisc_eval import celery_tasks, run_context
from aisc_eval.celery_app import celery_app

NOW, LIMIT = 1_000_000.0, 24300
E1, E2 = "e1", "e2"
RUN_OF_E1 = {"project": "p", "evaluation": E1, "ticket": f"{int(NOW) + 86400}.mac"}


def _module(name):
    try:
        return importlib.import_module(f"aisc_eval.{name}")
    except ModuleNotFoundError:
        raise AssertionError(
            f"the worker has no aisc_eval.{name}: nothing closes the tests it loses"
        ) from None


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def running(started_ago=60):
    return {"pid": "t", "status": "Running", "started_at": iso(NOW - started_ago)}


PENDING_TEST = {"pid": "t", "status": "Pending", "started_at": None}


# ── Which test is lost, and why ──────────────────────────────────────────────


@pytest.mark.parametrize("state", ["FAILURE", "REVOKED"])
def test_a_test_whose_task_ended_while_it_was_running_is_lost(state):
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(running(), state, "SUCCESS", NOW, NOW + 10, LIMIT)
        == sweep.LOST
    )


def test_a_running_test_whose_task_reports_success_is_a_second_copy_and_is_left():
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(running(), "SUCCESS", "SUCCESS", NOW, NOW + 10, LIMIT) is None
    )


def test_a_pending_test_whose_task_ended_is_lost():
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(PENDING_TEST, "SUCCESS", "SUCCESS", NOW, NOW + 10, LIMIT)
        == sweep.LOST
    )


def test_a_pending_test_whose_install_failed_is_lost():
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(PENDING_TEST, "PENDING", "FAILURE", NOW, NOW + 10, LIMIT)
        == sweep.LOST
    )


def test_a_test_running_past_its_limit_is_timed_out():
    sweep, registry = _module("run_sweep"), _module("run_registry")
    long_ago = LIMIT + registry.DEADLINE_MARGIN_S + 1
    reason = sweep.lost_reason(
        running(long_ago), "RUNNING", "SUCCESS", NOW, NOW + 10, LIMIT
    )
    assert reason == sweep.TIMED_OUT.format(limit=LIMIT)


def test_a_test_still_queued_at_the_runs_deadline_never_started():
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(PENDING_TEST, "PENDING", "SUCCESS", NOW, NOW - 1, LIMIT)
        == sweep.NEVER_STARTED
    )


def test_a_queued_or_running_test_within_its_limits_is_left():
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(PENDING_TEST, "PENDING", "STARTED", NOW, NOW + 10, LIMIT)
        is None
    )
    assert (
        sweep.lost_reason(running(), "RUNNING", "SUCCESS", NOW, NOW + 10, LIMIT) is None
    )


@pytest.mark.parametrize("status", ["Done", "Failed"])
def test_an_ended_test_is_never_touched(status):
    sweep = _module("run_sweep")
    assert (
        sweep.lost_reason(
            {"pid": "t", "status": status}, "FAILURE", "FAILURE", NOW, NOW - 1, LIMIT
        )
        is None
    )


# ── The sweep over the registered runs ───────────────────────────────────────


class FakeRedis:
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


class _Swept(types.SimpleNamespace):
    def start(self):
        """Loads the sweep and registers E1, as the dispatcher would have."""
        self.sweep_module, self.registry = _module("run_sweep"), _module("run_registry")
        self.monkeypatch.setattr(
            self.sweep_module,
            "task_state",
            lambda task_id: self.states.get(task_id, "PENDING"),
        )
        self.monkeypatch.setattr(
            self.sweep_module, "get_evaluation_request", self.read, raising=False
        )
        self.register(E1, ticket=RUN_OF_E1["ticket"], states={})
        return self.sweep_module

    def register(self, evaluation, ticket=None, states=None, test="t"):
        run = {
            "project": "p",
            "evaluation": evaluation,
            "ticket": ticket or f"{int(NOW) + 86400}.mac",
        }
        self.registry.register(
            self.redis,
            self.registry.RegisteredRun(
                evaluation=evaluation,
                run=run,
                claimed_at=NOW - 120,
                deadline=NOW + 3600,
                tests={test: f"task-{test}"},
                installs={test: f"install-{test}"},
            ),
            ttl=86400,
        )
        self.states.update(states or {})
        self.engine.setdefault(
            evaluation,
            {
                "status": "Processing",
                "evaluation_plugins": [
                    {"pid": test, "status": "Running", "started_at": iso(NOW - 60)}
                ],
            },
        )

    def read(self, pid):
        self.read_as.append(run_context.current())
        if pid in self.unreadable:
            raise ConnectionError("engine not answering")
        return self.engine[pid]


@pytest.fixture
def swept(monkeypatch):
    seen = _Swept(
        monkeypatch=monkeypatch,
        redis=FakeRedis(),
        states={},
        engine={},
        read_as=[],
        stops=[],
        finalized=[],
        unreadable=set(),
    )

    def stop(evaluation, tests, reason, task_id):
        seen.stops.append((evaluation, list(tests), reason, task_id))
        for test in seen.engine[evaluation]["evaluation_plugins"]:
            if test["pid"] in tests:
                test["status"] = "Failed"

    monkeypatch.setattr(celery_tasks, "_stop_run", stop)
    monkeypatch.setattr(
        celery_tasks, "_finalize", lambda pid: seen.finalized.append(pid), raising=False
    )
    monkeypatch.setattr(celery_app.backend, "client", seen.redis, raising=False)
    return seen


def test_the_sweep_fails_a_lost_test_with_its_reason_and_stops_the_run(swept):
    sweep = swept.start()
    swept.states["task-t"] = "FAILURE"
    assert sweep.sweep(NOW) == [(E1, "t", sweep.LOST)]
    assert swept.stops == [(E1, ["t"], sweep.LOST, None)]


def test_the_sweep_acts_as_the_run(swept):
    sweep = swept.start()
    sweep.sweep(NOW)
    assert swept.read_as == [RUN_OF_E1]


def test_a_run_whose_tests_all_ended_but_was_never_closed_is_finalized(swept):
    sweep = swept.start()
    swept.engine[E1] = {
        "status": "Processing",
        "evaluation_plugins": [{"pid": "t", "status": "Done"}],
    }
    sweep.sweep(NOW)
    assert swept.finalized == [E1]


def test_an_ended_run_leaves_the_registry(swept):
    sweep = swept.start()
    swept.engine[E1] = {
        "status": "Done",
        "evaluation_plugins": [{"pid": "t", "status": "Done"}],
    }
    sweep.sweep(NOW)
    assert [r.evaluation for r in swept.registry.registered(swept.redis)] == []


def test_a_run_whose_ticket_expired_is_dropped_and_the_others_still_swept(swept):
    sweep = swept.start()
    swept.states["task-t"] = "FAILURE"
    swept.register(
        E2, ticket=f"{int(NOW) - 1}.mac", states={"task-u": "FAILURE"}, test="u"
    )
    sweep.sweep(NOW)
    assert [r.evaluation for r in swept.registry.registered(swept.redis)] == [E1]
    assert swept.stops and swept.stops[0][0] == E1


def test_one_run_that_cannot_be_read_does_not_stop_the_sweep(swept):
    sweep = swept.start()
    swept.register(E2, states={"task-u": "FAILURE"}, test="u")
    swept.unreadable = {E1}
    assert [e for e, _, _ in sweep.sweep(NOW)] == [E2]


def test_tests_the_worker_could_not_report_while_the_network_was_down_are_closed_once(
    swept,
):
    """Each run's test ended with a dropped connection; its failure never reached the engine, so it shows Running
    and the run Processing. Two sweeps after the network is back close each test once."""
    sweep = swept.start()
    swept.register(E2, states={"task-u": "FAILURE"}, test="u")
    swept.states["task-t"] = "FAILURE"
    sweep.sweep(NOW)
    sweep.sweep(NOW + 300)
    assert sorted((e, tests[0]) for e, tests, _, _ in swept.stops) == [
        (E1, "t"),
        (E2, "u"),
    ]
    assert {reason for _, _, reason, _ in swept.stops} == {sweep.LOST}


def test_the_sweep_runs_every_five_minutes_from_beat():
    _module("run_sweep")
    schedule = (celery_app.conf.beat_schedule or {}).get("sweep-runs")
    assert schedule == {"task": "aisc_eval.run_sweep.sweep_runs", "schedule": 300.0}


def test_the_worker_sweeps_once_after_it_starts(monkeypatch):
    sweep = _module("run_sweep")
    sent = []
    monkeypatch.setattr(sweep.sweep_runs, "apply_async", lambda **kw: sent.append(kw))
    sweep._sweep_after_start()
    assert sent == [{"countdown": sweep.STARTUP_SWEEP_DELAY_S}]


def test_the_worker_loads_the_sweep():
    source = importlib.import_module("aisc_eval.celery_worker").__file__
    with open(source) as f:
        assert (
            "run_sweep" in f.read()
        ), "the worker never imports the sweep, so neither beat nor start-up runs it"


def test_the_runs_task_list_is_json_the_stop_route_reads():
    """The registry keeps the key Stop already reads, so a person's Stop still revokes every task."""
    registry = _module("run_registry")
    redis = FakeRedis()
    assert registry.claim(redis, E1, ["install-t", "task-t"], ttl=60) is True
    assert json.loads(redis.get(f"eval_tasks:{E1}")) == ["install-t", "task-t"]
    assert registry.claim(redis, E1, ["other"], ttl=60) is False
