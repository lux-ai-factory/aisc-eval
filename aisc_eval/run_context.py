"""The run a task acts for (configurator only), carried in the Celery message headers, so
Sean's task signatures stay his (adapt plan 2026-09-28, items 5 and 6). The backend puts it
on run_evaluation; every task published while a task runs inherits it; the api_client reads
it for the X-AISC-Project, X-AISC-Evaluation and X-AISC-Run headers the door checks."""

from contextvars import ContextVar
from typing import Any

from celery.signals import before_task_publish, task_postrun, task_prerun

HEADER = "aisc_run"
RUN_KEYS = ("project", "evaluation", "ticket")
Run = dict[str, str]
_current: ContextVar[Run | None] = ContextVar(HEADER, default=None)


def run_of(request: Any) -> Run | None:
    value = getattr(request, HEADER, None)
    if value is None:
        value = (getattr(request, "headers", None) or {}).get(HEADER)
    if not isinstance(value, dict) or not all(key in value for key in RUN_KEYS):
        return (
            None  # a partial run would raise KeyError in every internal call's headers
        )
    return value


def current() -> Run | None:
    return _current.get()


#: where _enter keeps the token of its set(), on the task's request, so _leave can reset to
#: what was current before: when an eager task (.apply(), task_always_eager) runs inside
#: another task, the outer task gets its run back afterwards. The eager task itself runs with
#: no run, since it publishes no message and so carries no header; a task without a header
#: never inherits a run, on purpose, so a stale context cannot leak. Nothing calls .apply().
_TOKEN = "_aisc_run_token"


@task_prerun.connect
def _enter(task: Any = None, **_: Any) -> None:
    token = _current.set(run_of(task.request) if task is not None else None)
    if task is not None:
        setattr(task.request, _TOKEN, token)


@task_postrun.connect
def _leave(task: Any = None, **_: Any) -> None:
    request = getattr(task, "request", None)
    token = getattr(request, _TOKEN, None)
    if token is None:
        _current.set(None)
        return
    setattr(request, _TOKEN, None)
    try:
        _current.reset(token)
    except ValueError:  # the token belongs to another context: clear rather than guess
        _current.set(None)


@before_task_publish.connect
def _forward(headers: dict[str, Any] | None = None, **_: Any) -> None:
    run = _current.get()
    if run is not None and headers is not None and HEADER not in headers:
        headers[HEADER] = run
