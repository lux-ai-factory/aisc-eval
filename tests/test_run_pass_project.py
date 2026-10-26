"""The worker asks for a run pass for its run's own project: the platform pid the backend put in the
run's aisc_run header. A run that carries no project gets no pass, and fails before its plugin starts;
so does a run whose pass the platform refuses."""

import pytest

from aisc_eval import celery_tasks, run_context
from aisc_eval.service import run_pass

from test_plugin_process import (  # noqa: F401  (fixtures)
    PROJECT,
    TARGET,
    _Answer,
    _resource,
    _run_with,
    platform,
    worker,
)

NO_PROJECT = "no run pass: the run carries no platform project"


def test_the_pass_is_asked_for_the_runs_project(worker, platform):
    _run_with([_resource("target", TARGET)])
    method, url, _headers, body = platform["calls"][0]
    assert (method, url) == ("POST", "http://platform:8000/internal/run-passes")
    assert body == {"refs": [TARGET], "evaluation_plugin": "ep", "project": PROJECT}


def test_a_run_with_no_project_gets_no_pass_and_never_starts(worker, platform):
    token = run_context._current.set(None)
    try:
        with pytest.raises(run_pass.RunPassRefused) as refused:
            _run_with([_resource("target", TARGET)])
    finally:
        run_context._current.reset(token)
    assert str(refused.value) == NO_PROJECT
    assert worker["popen"] == []
    assert platform["calls"] == []
    [(_evaluation, _plugin, message)] = worker["failed"]
    assert message == NO_PROJECT


def test_a_pass_refused_for_a_foreign_reference_fails_the_run_before_the_plugin(
    worker, platform
):
    platform["mint"] = _Answer(
        403, {"detail": "'connection:B/x' is not of the run's project A"}
    )
    with pytest.raises(RuntimeError) as raised:
        _run_with([_resource("system", f"connection:{PROJECT}/x")])
    assert "403" in str(raised.value)
    assert worker["popen"] == []
    [(_evaluation, _plugin, message)] = worker["failed"]
    assert "403" in message


@pytest.mark.parametrize("project", [None, ""])
def test_mint_without_a_project_refuses_before_any_call(platform, project):
    with pytest.raises(run_pass.RunPassRefused) as refused:
        run_pass.mint([TARGET], "ep", project)
    assert str(refused.value) == NO_PROJECT
    assert platform["calls"] == []


def test_mint_sends_the_project_in_the_body(platform):
    assert run_pass.mint([TARGET], "ep", PROJECT) == "the-pass"
    assert platform["calls"][0][3] == {
        "refs": [TARGET],
        "evaluation_plugin": "ep",
        "project": PROJECT,
    }


def test_the_worker_reads_the_project_from_the_run_context(worker, platform, monkeypatch):
    seen = []

    def mint(refs, evaluation_plugin_uuid, project=None, **kwargs):
        seen.append(kwargs.get("project", project))
        return "the-pass"

    monkeypatch.setattr(celery_tasks.run_pass, "mint", mint)
    _run_with([_resource("target", TARGET)])
    assert seen == [PROJECT]
