"""When the worker ends a run's pass, the platform seals the run's calls in the ledger, citing the request and the
run the run's start cites. The backend puts both in the run's aisc_run header (request_id, run_id); the worker
sends them on the revoke as X-AISC-Request-Id and X-AISC-Run-Id. A run without them is still revoked."""

from aisc_eval import run_context

from test_plugin_process import (  # noqa: F401  (fixtures)
    PROJECT,
    TARGET,
    _resource,
    _run_with,
    platform,
    worker,
)

REQUEST = "11111111-2222-4333-8444-555555555555"
RUN = "0f0e0d0c-0b0a-4908-8706-050403020100"
REVOKE = ("DELETE", "http://platform:8000/internal/run-passes")


def _revokes(platform):
    return [
        headers
        for method, url, headers, _ in platform["calls"]
        if (method, url) == REVOKE
    ]


def test_the_revoke_names_the_request_and_the_run_of_the_runs_start(worker, platform):
    token = run_context._current.set(
        {
            "project": PROJECT,
            "evaluation": "eval",
            "ticket": "t",
            "request_id": REQUEST,
            "run_id": RUN,
        }
    )
    try:
        _run_with([_resource("target", TARGET)])
    finally:
        run_context._current.reset(token)
    assert _revokes(platform) == [
        {
            "X-AISC-Run-Pass": "the-pass",
            "X-AISC-Request-Id": REQUEST,
            "X-AISC-Run-Id": RUN,
        }
    ]


def test_a_run_without_them_is_still_revoked_with_the_pass_alone(worker, platform):
    _run_with([_resource("target", TARGET)])
    assert _revokes(platform) == [{"X-AISC-Run-Pass": "the-pass"}]
