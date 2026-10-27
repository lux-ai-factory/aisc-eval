"""A helper input reaches the worker as a reference; the worker resolves it on the platform before
the plugin starts and hands the plugin what an llm input gives: <name>.json with the endpoint, the
model and the secret's name, the key in the plugin's secret environment."""

import json
import logging

import pytest
import requests

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.service import helpers
from aisc_plugin_interface import BaseEvaluationPlugin, HelperModel, helper_model
from test_plugin_process import PROJECT, _Answer, platform, worker  # noqa: F401  (fixtures)

KEY = "sk-test-123"
ENDPOINT = "https://llm.example/v1"
REF = f"helper:{PROJECT}/judge-a"
RESOLVED = {
    "label": "Judge A",
    "endpoint_url": ENDPOINT,
    "roles": ["judge"],
    "api_key": KEY,
}


def _helper_input():
    return {
        "name": "helper:judge",
        "component_type": "helper",
        "helper_ref": REF,
        "value": {"model": "gpt-4o-mini", "label": "Judge A"},
    }


def _llm_input():
    return {
        "name": "helper:judge",
        "component_type": "llm",
        "json_value": {"endpoint_url": ENDPOINT, "secret_key": "AISC_HELPER_JUDGE_KEY"},
        "value": {"model": "gpt-4o-mini"},
        "secret_key": "AISC_HELPER_JUDGE_KEY",
        "secret_encrypted_value": "cipher",
    }


@pytest.fixture
def run(worker, platform, monkeypatch):  # noqa: F811
    """run_plugin with the platform's resolve route faked; what the plugin was given."""
    seen = {"resolve": _Answer(200, RESOLVED), "urls": [], "inputs": {}, "venvs": []}

    def get(url, headers=None, **kwargs):
        seen["urls"].append((url, dict(headers or {})))
        return seen["resolve"]

    monkeypatch.setattr(helpers.requests, "get", get)
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda value: KEY)
    real_install = plugin_process.run_install
    monkeypatch.setattr(
        plugin_process,
        "run_install",
        lambda argv, *a, **kw: seen["venvs"].append(argv)
        or real_install(argv, *a, **kw),
    )

    def leave(output):
        seen["inputs"] = {
            path.name: path.read_bytes() for path in (output.parent / "input").iterdir()
        }
        (output / "measures.json").write_text("[]")

    worker["leave"] = leave
    seen["worker"] = worker
    seen["go"] = lambda inputs: celery_tasks.run_plugin.run(
        "pkg", "Plugin", "1.0", {}, inputs, [], "eval", "ep"
    )
    return seen


def test_helper_input_writes_same_payload_as_llm_component(run):
    run["go"]([_helper_input()])
    from_platform = run["inputs"]["helper:judge.json"]
    [(_argv, kwargs)] = run["worker"]["popen"]
    assert kwargs["env"]["AISC_SECRET_AISC_HELPER_JUDGE_KEY"] == KEY

    run["go"]([_llm_input()])
    assert json.loads(from_platform) == json.loads(run["inputs"]["helper:judge.json"])
    assert KEY.encode() not in from_platform
    assert (
        run["worker"]["popen"][1][1]["env"]["AISC_SECRET_AISC_HELPER_JUDGE_KEY"] == KEY
    )


def test_the_resolve_call_carries_the_service_token_and_names_the_helper(run):
    run["go"]([_helper_input()])
    [(url, headers)] = run["urls"]
    assert (
        url
        == f"http://platform:8000/internal/projects/{PROJECT}/helpers/judge-a/resolve"
    )
    assert headers == {"X-AISC-Service-Token": "stack-token"}


@helper_model("judge", "Judge model", required=True)
class _Judging(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        return None

    def export_metrics(self, *a, **k):
        return []


def test_plugin_reads_helper_through_interface(run, monkeypatch):
    run["go"]([_helper_input()])
    [(_argv, kwargs)] = run["worker"]["popen"]
    monkeypatch.setenv(
        "AISC_SECRET_AISC_HELPER_JUDGE_KEY",
        kwargs["env"]["AISC_SECRET_AISC_HELPER_JUDGE_KEY"],
    )
    plugin = _Judging()
    plugin.set_input_content("helper:judge", run["inputs"]["helper:judge.json"])
    assert plugin.helper("judge") == HelperModel(
        role="judge", endpoint_url=ENDPOINT, model="gpt-4o-mini", api_key=KEY
    )


def test_unreadable_key_fails_run_before_plugin(run):
    detail = "the stored key for helper judge-a cannot be decrypted; enter it again"
    run["resolve"] = _Answer(409, {"detail": detail})
    with pytest.raises(RuntimeError, match="judge-a"):
        run["go"]([_helper_input()])
    assert run["worker"]["popen"] == []
    assert run["venvs"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert message == detail


def test_removed_helper_fails_run_with_its_name(run):
    run["resolve"] = _Answer(404, {"detail": "no helper judge-a in this project"})
    with pytest.raises(RuntimeError, match="judge-a"):
        run["go"]([_helper_input()])
    assert run["worker"]["popen"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert "judge-a" in message


@pytest.mark.parametrize("status", [401, 503, 500])
def test_any_other_answer_says_the_platform_did_not_answer(run, status):
    run["resolve"] = _Answer(status, {"detail": f"internal {KEY}"})
    with pytest.raises(RuntimeError, match="the platform did not answer"):
        run["go"]([_helper_input()])
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert message == "the platform did not answer"


def test_an_unreachable_platform_says_so_without_its_address(run, monkeypatch):
    def down(*_args, **_kwargs):
        raise requests.ConnectionError("http://platform:8000 refused")

    monkeypatch.setattr(helpers.requests, "get", down)
    with pytest.raises(RuntimeError, match="the platform did not answer") as raised:
        run["go"]([_helper_input()])
    assert "platform:8000" not in str(raised.value)
    assert run["worker"]["popen"] == []


def test_key_never_logged(run, caplog):
    with caplog.at_level(logging.DEBUG):
        run["go"]([_helper_input()])
    assert KEY not in caplog.text
    assert all(KEY not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "ref",
    [
        f"helper:{PROJECT}/..",
        f"helper:{PROJECT}/.",
        "helper:../x",
        f"helper:{PROJECT}/a/b",
        f"helper:{PROJECT}",
        "helper:not-a-uuid/judge-a",
        f"helper:{PROJECT}/Judge_A",
        f"helper:{PROJECT}/",
        "judge-a",
    ],
)
def test_a_malformed_reference_fails_the_run_without_calling_the_platform(run, ref):
    component = {**_helper_input(), "helper_ref": ref}
    with pytest.raises(RuntimeError, match="malformed"):
        run["go"]([component])
    assert run["urls"] == []
    assert run["worker"]["popen"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert "malformed" in message


def test_a_helper_input_without_a_reference_fails_the_run(run):
    component = _helper_input()
    del component["helper_ref"]
    with pytest.raises(RuntimeError, match="malformed"):
        run["go"]([component])
    assert run["urls"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert "malformed" in message
