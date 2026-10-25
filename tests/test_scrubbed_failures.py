"""What the worker reports of a failed run keeps no secret: the values it handed the plugin and anything shaped like
a key are redacted from every failure text it sends, every error it raises and the execution log it uploads.
The run is faked as test_plugin_process.py fakes it."""

import importlib
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.service import api_client

from test_plugin_process import _Process, worker  # noqa: F401  (the fixture)

JUDGE = "judge-secret-value-1"
SHAPED = "sk-proj-abcdefghijklmnopqrstu"
SECRET_SETTING = [{"category": "secrets", "key": "judge_key", "encrypted_value": "ciphertext"}]


def scrub():
    return importlib.import_module("aisc_eval.scrub")


def test_the_known_secrets_are_the_values_the_run_handed_out_longest_first():
    known = scrub().known_secrets(
        {"AISC_SECRET_JUDGE_KEY": JUDGE},
        {"OPENAI_API_KEY": "sk-worker-abcdefghijklmnop", "API_KEY_OPENAI": celery_tasks.OPENAI_PLACEHOLDER,
         "PATH": "/usr/bin"},
        {"api_key": "cfg-key-12345678", "temperature": "0.2", "nested": {"token": "short"}},
        "run-pass-abcdef123456")
    assert known == ["sk-worker-abcdefghijklmnop", "run-pass-abcdef123456", JUDGE, "cfg-key-12345678"]


@pytest.fixture
def with_secret(monkeypatch):
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda value: JUDGE)


def _plugin(monkeypatch, *, stdout=(), stderr="", returncode=0, leave=None):
    def popen(argv, **kwargs):
        kwargs["stderr"].write(stderr)
        kwargs["stderr"].flush()
        output = Path(kwargs["cwd"]) / "output"
        (output / "measures.json").write_text("[]")
        process = _Process(returncode=returncode)
        process.stdout = iter(list(stdout))
        return process

    monkeypatch.setattr(plugin_process.subprocess, "Popen", popen)


def _run(project_settings):
    return celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], project_settings, "eval", "ep")


def test_a_failed_plugins_stderr_is_reported_without_its_secrets(worker, with_secret, monkeypatch):
    _plugin(monkeypatch, stderr=f"auth failed for {JUDGE} and {SHAPED}\n", returncode=1)
    with pytest.raises(RuntimeError) as raised:
        _run(SECRET_SETTING)
    message = str(raised.value)
    assert "auth failed for [redacted] and [redacted]" in message
    assert JUDGE not in message and SHAPED not in message
    assert worker["failed"]
    for _evaluation, _plugin_pid, text in worker["failed"]:
        assert JUDGE not in text and SHAPED not in text
        assert "[redacted]" in text


def test_the_execution_log_keeps_no_secret_the_plugin_printed(worker, with_secret, monkeypatch):
    _plugin(monkeypatch, stdout=[f"using {JUDGE}\n"], stderr=f"warning: {JUDGE}\n", returncode=0)
    _run(SECRET_SETTING)
    log = worker["uploads"]["plugin_execution.log"]
    assert JUDGE.encode() not in log
    assert b"[redacted]" in log


class _Failed:
    returncode = 1
    stdout = ""

    def __init__(self, stderr):
        self.stderr = stderr


def test_a_failed_venv_step_is_reported_with_its_password_redacted(worker, monkeypatch):
    monkeypatch.setattr(plugin_process, "run_install", lambda *a, **kw: _Failed("error: password=hunter22 rejected"))
    with pytest.raises(RuntimeError) as raised:
        _run([])
    assert worker["failed"] == [("eval", "ep", "error: password=[redacted] rejected")]
    assert "hunter22" not in str(raised.value)


def test_a_failed_install_is_reported_and_logged_with_its_key_redacted(worker, monkeypatch, caplog):
    monkeypatch.setattr(plugin_process, "install_user", lambda: plugin_process.PluginUser(0, 0, None, switch=False))
    monkeypatch.setattr(plugin_process, "prepare_install_folder", lambda *a, **kw: None)
    monkeypatch.setattr(plugin_process, "run_install", lambda *a, **kw: _Failed("error: api_key=abcd1234efgh refused"))
    with pytest.raises(RuntimeError):
        celery_tasks.install_package.run("pkg", "1.0", "eval", ["ep"])
    assert worker["failed"] == [("eval", "ep", "error: api_key=[redacted] refused")]
    assert "abcd1234efgh" not in caplog.text


def test_the_api_client_sends_the_failure_text_scrubbed(monkeypatch):
    sent = []

    class _Answer:
        ok = True
        text = ""

    def patch(url, json=None, headers=None, **kwargs):
        sent.append(json)
        return _Answer()

    monkeypatch.setattr(api_client.requests, "patch", patch)
    api_client.mark_plugin_failed("e", "p", "x Bearer abc")
    assert sent == [{"error_message": "x [redacted]"}]

