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
SECRET_SETTING = [
    {"category": "secrets", "key": "judge_key", "encrypted_value": "ciphertext"}
]


def scrub():
    return importlib.import_module("aisc_eval.scrub")


def test_the_known_secrets_are_the_values_the_run_handed_out_longest_first():
    known = scrub().known_secrets(
        {"AISC_SECRET_JUDGE_KEY": JUDGE},
        {
            "OPENAI_API_KEY": "sk-worker-abcdefghijklmnop",
            "API_KEY_OPENAI": celery_tasks.OPENAI_PLACEHOLDER,
            "PATH": "/usr/bin",
        },
        {
            "api_key": "cfg-key-12345678",
            "temperature": "0.2",
            "nested": {"token": "short"},
        },
        "run-pass-abcdef123456",
    )
    assert known == [
        "sk-worker-abcdefghijklmnop",
        "run-pass-abcdef123456",
        JUDGE,
        "cfg-key-12345678",
    ]


@pytest.fixture
def with_secret(monkeypatch):
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda value: JUDGE)


def _plugin(
    monkeypatch, *, stdout=(), stderr="", returncode=0, leave=None, measures="[]"
):
    def popen(argv, **kwargs):
        kwargs["stderr"].write(stderr)
        kwargs["stderr"].flush()
        output = Path(kwargs["cwd"]) / "output"
        (output / "measures.json").write_text(measures)
        process = _Process(returncode=returncode)
        process.stdout = iter(list(stdout))
        return process

    monkeypatch.setattr(plugin_process.subprocess, "Popen", popen)


def _run(project_settings):
    return celery_tasks.run_plugin.run(
        "pkg", "Plugin", "1.0", {}, [], project_settings, "eval", "ep"
    )


def test_a_failed_plugins_stderr_is_reported_without_its_secrets(
    worker, with_secret, monkeypatch
):
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


def test_the_execution_log_keeps_no_secret_the_plugin_printed(
    worker, with_secret, monkeypatch
):
    _plugin(
        monkeypatch,
        stdout=[f"using {JUDGE}\n"],
        stderr=f"warning: {JUDGE}\n",
        returncode=0,
    )
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
    monkeypatch.setattr(
        plugin_process,
        "run_install",
        lambda *a, **kw: _Failed("error: password=hunter22 rejected"),
    )
    with pytest.raises(RuntimeError) as raised:
        _run([])
    assert worker["failed"] == [("eval", "ep", "error: password=[redacted] rejected")]
    assert "hunter22" not in str(raised.value)


def test_a_failed_install_is_reported_and_logged_with_its_key_redacted(
    worker, monkeypatch, caplog
):
    monkeypatch.setattr(
        plugin_process,
        "install_user",
        lambda: plugin_process.PluginUser(0, 0, None, switch=False),
    )
    monkeypatch.setattr(plugin_process, "prepare_install_folder", lambda *a, **kw: None)
    monkeypatch.setattr(
        plugin_process,
        "run_install",
        lambda *a, **kw: _Failed("error: api_key=abcd1234efgh refused"),
    )
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
    api_client.mark_plugin_failed("e", "p", "x Bearer abcDEF0123456789ghi")
    assert sent == [{"error_message": "x [redacted]"}]


# What else of a run leaves the worker: progress, other exceptions, the error handler


def test_the_progress_a_plugin_reports_is_stored_without_secrets(
    worker, with_secret, monkeypatch
):
    import json
    import types

    stored = []
    fake_app = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            store_result=lambda task_id, meta, state=None, **kw: stored.append(
                json.dumps(meta, default=str)
            )
        )
    )
    monkeypatch.setattr(celery_tasks, "celery_app", fake_app)
    line = json.dumps(
        {
            "progress": 0.5,
            "extra": {
                "note": f"judging with {JUDGE}",
                "nested": [{"header": f"Bearer {SHAPED}"}],
            },
        }
    )
    _plugin(monkeypatch, stdout=[line + "\n"], returncode=0)
    _run(SECRET_SETTING)
    assert stored, "the progress line was not stored"
    for meta in stored:
        assert JUDGE not in meta and SHAPED not in meta
        assert "0.5" in meta


def _everything_raised(exc):
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def test_an_exception_that_is_not_the_workers_own_is_raised_with_its_secrets_redacted(
    worker, with_secret, monkeypatch, caplog
):
    def refuse(evaluation, plugin, name, content):
        raise ConnectionError(f"upload refused for {JUDGE} with {SHAPED}")

    monkeypatch.setattr(celery_tasks, "upload_artifact", refuse)
    _plugin(monkeypatch, returncode=0)
    with pytest.raises(Exception) as raised:
        _run(SECRET_SETTING)
    text = _everything_raised(raised.value)
    assert "upload refused for" in text
    assert JUDGE not in text and SHAPED not in text
    assert JUDGE not in caplog.text and SHAPED not in caplog.text
    for _evaluation, _plugin_pid, failure in worker["failed"]:
        assert JUDGE not in failure and SHAPED not in failure


def test_a_refused_run_pass_is_raised_with_its_secrets_redacted(
    worker, monkeypatch, caplog
):
    from aisc_eval.service import run_pass

    def refused(refs, plugin_pid, *args, **kwargs):
        raise run_pass.RunPassRefused(
            f"the platform refused the pass: Authorization: Bearer {SHAPED}"
        )

    monkeypatch.setattr(
        celery_tasks.run_pass, "refs_of", lambda components: ["target:x/system"]
    )
    monkeypatch.setattr(celery_tasks.run_pass, "mint", refused)
    with pytest.raises(Exception) as raised:
        _run([])
    text = _everything_raised(raised.value)
    assert "refused the pass" in text
    assert SHAPED not in text
    assert SHAPED not in caplog.text


def test_the_error_handler_logs_neither_the_configuration_nor_the_failure_in_clear(
    monkeypatch, caplog
):
    import logging

    monkeypatch.setattr(celery_tasks, "mark_failed", lambda *a, **kw: None)
    request = {
        "args": [
            "pkg",
            "Plugin",
            "1.0",
            {"api_key": "cfg-key-12345678", "judge": f"Bearer {SHAPED}"},
        ]
    }
    caplog.set_level(logging.DEBUG)
    celery_tasks.handle_error.run(
        "eval", request, RuntimeError(f"api_key=cfg-key-12345678 {SHAPED}"), None
    )
    assert "cfg-key-12345678" not in caplog.text
    assert SHAPED not in caplog.text
    assert "eval" in caplog.text


# What still left the worker unscrubbed: a failing post, an exception's args, progress keys, a known value in a
# measure


URL_CREDENTIALS = "https://bob:hunter22url@llm.example/v1"


def _stored_by_celery(exc):
    """What Celery keeps of a failed task (type, args, repr) and logs (the whole traceback, chain included)."""
    import json

    return json.dumps(
        {
            "type": type(exc).__name__,
            "args": [repr(a) for a in exc.args],
            "repr": repr(exc),
        }
    ) + _everything_raised(exc)


class _PostRefused(Exception):
    pass


def test_a_failing_post_is_re_raised_scrubbed_with_nothing_chained(
    worker, monkeypatch, caplog
):
    import logging

    def refuse(*a, **kw):
        raise _PostRefused(
            f"500 posting measures; echoed password={JUDGE} and {URL_CREDENTIALS}"
        )

    monkeypatch.setattr(celery_tasks, "post_measures", refuse)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(Exception) as raised:
        celery_tasks.post_measurements.run([], "eval", "ep")
    seen = _stored_by_celery(raised.value)
    assert "500 posting measures" in seen
    for secret in (JUDGE, "hunter22url"):
        assert secret not in seen
        assert secret not in caplog.text
        for *_ids, text in worker["failed"]:
            assert secret not in text
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None or raised.value.__suppress_context__


class _CleanText(Exception):
    def __str__(self):
        return "upload refused"


def test_an_exception_whose_args_hold_a_secret_is_raised_scrubbed(
    worker, with_secret, monkeypatch, caplog
):
    def refuse(*a, **kw):
        raise _CleanText("upload refused", JUDGE, {"token": SHAPED})

    monkeypatch.setattr(celery_tasks, "upload_artifact", refuse)
    _plugin(monkeypatch, returncode=0)
    with pytest.raises(Exception) as raised:
        _run(SECRET_SETTING)
    seen = _stored_by_celery(raised.value)
    assert "upload refused" in seen
    assert JUDGE not in seen and SHAPED not in seen
    assert JUDGE not in caplog.text and SHAPED not in caplog.text


def test_a_secret_used_as_a_key_in_progress_is_stored_scrubbed(
    worker, with_secret, monkeypatch
):
    import json
    import types

    stored = []
    fake_app = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            store_result=lambda task_id, meta, state=None, **kw: stored.append(
                json.dumps(meta, default=str)
            )
        )
    )
    monkeypatch.setattr(celery_tasks, "celery_app", fake_app)
    line = json.dumps(
        {
            "progress": 0.3,
            "extra": {JUDGE: "as a key", "nested": {SHAPED: 1}, "plain": "kept"},
        }
    )
    _plugin(monkeypatch, stdout=[line + "\n"], returncode=0)
    _run(SECRET_SETTING)
    assert stored, "the progress line was not stored"
    for meta in stored:
        assert JUDGE not in meta and SHAPED not in meta
        assert "kept" in meta


def test_a_value_the_plugin_was_given_is_scrubbed_from_a_measure_failure(
    worker, with_secret, monkeypatch, caplog
):
    """The judge key has no recognisable shape; the worker knows it handed it over, so a measure that echoes it
    and fails to post leaves it in no stored failure text, raised error or log."""
    import json
    import logging

    _plugin(
        monkeypatch, returncode=0, measures=json.dumps([{"name": "m", "score": JUDGE}])
    )
    measures = _run(SECRET_SETTING)  # what the chain hands post_measurements
    caplog.set_level(logging.DEBUG)
    with pytest.raises(Exception) as raised:
        celery_tasks.post_measurements.run(measures, "eval", "ep")
    assert worker["failed"], "the failed post was not reported"
    for *_ids, text in worker["failed"]:
        assert JUDGE not in text
    assert JUDGE not in _stored_by_celery(raised.value)
    assert JUDGE not in caplog.text
