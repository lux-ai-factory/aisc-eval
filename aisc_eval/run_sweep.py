"""The sweep: it closes the tests the worker lost, ran past their limit, or never started.

Every SWEEP_EVERY_S (Celery beat) and once soon after the worker starts, the sweep reads each run the dispatcher
registered (run_registry) and fails, with a stated reason, each of its tests that can no longer end on its own:
one whose task ended without reporting (the worker stopped or its process was killed, or its report could not
reach the engine), one running past its time limit, one still queued when the run's time ran out. Failing it
stops the run as any failed test does. A run whose tests all ended but which was never closed is finalized; a run
that has ended leaves the registry, and one whose ticket expired is dropped (the worker can no longer act as it).
Each run is swept on its own: one that cannot be read never stops the others.
"""

import datetime
import time

from celery.result import AsyncResult
from celery.signals import worker_ready
from celery.states import READY_STATES

from aisc_eval import celery_tasks, run_context, run_registry
from aisc_eval.celery_app import celery_app
from aisc_eval.service.api_client import get_evaluation_request
from aisc_eval.utils.logging import get_logger

logger = get_logger()

SWEEP_EVERY_S = 300
STARTUP_SWEEP_DELAY_S = 120

LOST = (
    "The worker lost this test: its task ended without reporting (the worker stopped or its process was "
    "killed). It is not run again."
)
TIMED_OUT = "The test ran longer than its time limit ({limit} s) and no worker reported its end."
NEVER_STARTED = (
    "The test never started: no worker took it before the run's time ran out."
)

#: the states of a task that ended without its test ending: it can no longer report
_ENDED_BADLY = ("FAILURE", "REVOKED")
_RUN_OPEN = ("Pending", "Processing")

celery_app.conf.beat_schedule = {
    **(celery_app.conf.beat_schedule or {}),
    "sweep-runs": {
        "task": "aisc_eval.run_sweep.sweep_runs",
        "schedule": float(SWEEP_EVERY_S),
    },
}


def task_state(task_id: str) -> str:
    return AsyncResult(task_id, app=celery_app).state


def _started(test: dict) -> float | None:
    try:
        return datetime.datetime.fromisoformat(str(test.get("started_at"))).timestamp()
    except ValueError:
        return None


def lost_reason(
    test: dict,
    task_state: str,
    install_state: str | None,
    now: float,
    deadline: float,
    time_limit: int,
) -> str | None:
    """Why a test can no longer end on its own, or None while it still can (or has ended).

    A Running test whose task reports success is a second copy's: the queue delivered it twice, the refused copy
    ended well, and the first copy still runs it. It is left alone."""
    status = test.get("status")
    if status not in celery_tasks.UNFINISHED:
        return None
    if status == "Running" and task_state in _ENDED_BADLY:
        return LOST
    if status == "Pending" and (
        task_state in READY_STATES or install_state in _ENDED_BADLY
    ):
        return LOST
    if status == "Running":
        started = _started(test)
        if (
            started is not None
            and now - started > time_limit + run_registry.DEADLINE_MARGIN_S
        ):
            return TIMED_OUT.format(limit=time_limit)
    if now > deadline:
        return (
            NEVER_STARTED if status == "Pending" else TIMED_OUT.format(limit=time_limit)
        )
    return None


def _sweep_run(
    client, entry: run_registry.RegisteredRun, now: float, time_limit: int
) -> list[tuple[str, str, str]]:
    expiry = run_registry.ticket_expiry(entry.run)
    if expiry is not None and expiry <= now:
        logger.warning(
            f"Evaluation {entry.evaluation}: its run ticket expired, so it is no longer swept"
        )
        run_registry.forget(client, entry.evaluation)
        return []
    closed = []
    with run_context.acting_for(entry.run):
        state = get_evaluation_request(entry.evaluation)
        tests = [
            t for t in state.get("evaluation_plugins") or [] if isinstance(t, dict)
        ]
        for test in tests:
            pid = str(test.get("pid"))
            task, install = entry.tests.get(pid), entry.installs.get(pid)
            reason = lost_reason(
                test,
                task_state(task) if task else "PENDING",
                task_state(install) if install else None,
                now,
                entry.deadline,
                time_limit,
            )
            if reason:
                celery_tasks._stop_run(entry.evaluation, [test["pid"]], reason, None)
                closed.append((entry.evaluation, pid, reason))
        if not closed and not any(
            t.get("status") in celery_tasks.UNFINISHED for t in tests
        ):
            if state.get("status") in _RUN_OPEN:
                celery_tasks._finalize(entry.evaluation)
            else:
                run_registry.forget(client, entry.evaluation)
    return closed


def sweep(now: float | None = None) -> list[tuple[str, str, str]]:
    """Sweep every registered run; returns (evaluation, test, reason) for each test it failed."""
    now = time.time() if now is None else now
    client = celery_app.backend.client
    time_limit = int(celery_app.conf.task_time_limit)
    closed = []
    for entry in run_registry.registered(client):
        try:
            closed += _sweep_run(client, entry, now, time_limit)
        except Exception as exc:
            logger.warning(
                f"Evaluation {entry.evaluation} could not be swept ({type(exc).__name__}); tried again next sweep"
            )
    return closed


@celery_app.task
def sweep_runs() -> int:
    closed = sweep()
    for evaluation, test, reason in closed:
        logger.warning(
            f"Evaluation {evaluation}: test {test} failed by the sweep: {reason}"
        )
    return len(closed)


@worker_ready.connect
def _sweep_after_start(**_) -> None:
    sweep_runs.apply_async(countdown=STARTUP_SWEEP_DELAY_S)
