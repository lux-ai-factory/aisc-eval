import pytest
from aisc_eval import deployment
from aisc_eval.service import api_client


def test_bad_values_stop_the_worker():
    with pytest.raises(SystemExit, match="AISC_DEPLOYMENT"):
        deployment.mode({"AISC_DEPLOYMENT": "config"})


def test_standalone_headers_name_no_project(monkeypatch):
    monkeypatch.setattr(deployment, "MODE", deployment.STANDALONE)
    assert "X-AISC-Project" not in api_client.headers(None)


def test_configurator_headers_name_the_run(monkeypatch):
    monkeypatch.setattr(deployment, "MODE", deployment.CONFIGURATOR)
    run = {
        "project": "00000000-0000-0000-0000-000000000001",
        "evaluation": "e",
        "ticket": "t",
    }
    h = api_client.headers(run)
    assert (
        h["X-AISC-Project"] == run["project"]
        and h["X-AISC-Run"] == "t"
        and h["X-AISC-Evaluation"] == "e"
    )


def test_configurator_headers_default_to_the_run_context(monkeypatch):
    from aisc_eval import run_context

    monkeypatch.setattr(deployment, "MODE", deployment.CONFIGURATOR)
    run = {
        "project": "00000000-0000-0000-0000-000000000001",
        "evaluation": "e",
        "ticket": "t",
    }
    assert "X-AISC-Project" not in api_client.headers()
    token = run_context._current.set(run)
    try:
        assert api_client.headers()["X-AISC-Run"] == "t"
    finally:
        run_context._current.reset(token)


def test_the_worker_refuses_to_start_without_the_internal_secret():
    """Every internal call, standalone or configurator, carries X-Internal-Secret; without it
    the worker would start and have every call refused, with only warnings in the log."""
    import pytest

    for env in (
        {"AISC_DEPLOYMENT": "standalone"},
        {"AISC_DEPLOYMENT": "configurator"},
        {"AISC_DEPLOYMENT": "configurator", "INTERNAL_API_KEY": "  "},
    ):
        with pytest.raises(RuntimeError, match="INTERNAL_API_KEY"):
            deployment.check_environment(env)
    deployment.check_environment(
        {"AISC_DEPLOYMENT": "configurator", "INTERNAL_API_KEY": "k"}
    )


def test_the_check_runs_when_a_worker_starts_not_when_flower_imports_the_app():
    from celery.signals import worker_init

    assert any(
        getattr(r[1](), "__name__", "") == "_check_on_worker_start"
        for r in worker_init.receivers
        if callable(r[1])
    )
