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
    run = api_client.Run(project="00000000-0000-0000-0000-000000000001", evaluation="e", ticket="t")
    h = api_client.headers(run)
    assert h["X-AISC-Project"] == run.project and h["X-AISC-Run"] == "t" and h["X-AISC-Evaluation"] == "e"
