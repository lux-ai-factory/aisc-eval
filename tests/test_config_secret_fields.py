"""A key the backend kept as a project secret comes back into the plugin's configuration only in the plugin's own
process: the worker puts its value in the plugin's environment and names the field-to-variable pairs in
config.json, and AISC's runtime script fills the fields in memory just before evaluate. No value on disk.
The run is faked as test_plugin_process.py fakes it."""

import copy
import json
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process, plugin_runtime

from test_plugin_process import _Process, worker  # noqa: F401  (the fixture)

CRED = "cred-0123456789abcdef"
VARIABLE = "AISC_SECRET_CONFIG_MODEL_CREDENTIAL"


def s(category, plugin_config_key, encrypted_value):
    """One setting as the backend's by-pid route answers it."""
    return {
        "pid": "00000000-0000-4000-8000-000000000001",
        "key": plugin_config_key.replace(".", "_"),
        "plugin_config_key": plugin_config_key,
        "category": category,
        "encrypted_value": encrypted_value,
        "json_value": {},
    }


SETTINGS = [
    s("secrets", "config.model_credential", "c1"),
    s("secrets", "openai_key", "c2"),
    s("variables", "config.threshold", ""),
    s("secrets", "config.empty", ""),
]


def test_the_config_fields_are_the_secrets_mapped_to_a_config_path():
    assert celery_tasks.config_secret_fields(SETTINGS) == {"model_credential": VARIABLE}


def test_a_config_secret_is_in_the_environment_like_any_selected_secret(monkeypatch):
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda c: "dec:" + c)
    assert celery_tasks.build_secret_environment(SETTINGS) == {
        VARIABLE: "dec:c1",
        "AISC_SECRET_OPENAI_KEY": "dec:c2",
    }


def _plugin(monkeypatch, seen, *, stderr="", returncode=0):
    def popen(argv, **kwargs):
        workspace = Path(kwargs["cwd"])
        seen["config_bytes"] = (workspace / "config.json").read_bytes()
        seen["env"] = kwargs["env"]
        kwargs["stderr"].write(stderr)
        kwargs["stderr"].flush()
        (workspace / "output" / "measures.json").write_text("[]")
        return _Process(returncode=returncode)

    monkeypatch.setattr(plugin_process.subprocess, "Popen", popen)


def _run(plugin_config, project_settings):
    return celery_tasks.run_plugin.run(
        "pkg", "Plugin", "1.0", plugin_config, [], project_settings, "eval", "ep"
    )


def test_config_json_names_the_variable_and_never_holds_the_key(worker, monkeypatch):
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda c: CRED)
    seen = {}
    _plugin(monkeypatch, seen)
    _run({"threshold": 0.5}, [s("secrets", "config.model_credential", "c1")])
    written = json.loads(seen["config_bytes"])
    assert written["config_secret_fields"] == {"model_credential": VARIABLE}
    assert written["plugin_config"] == {"threshold": 0.5}
    assert CRED.encode() not in seen["config_bytes"]
    assert seen["env"][VARIABLE] == CRED

    _run({"threshold": 0.5}, [])
    assert json.loads(seen["config_bytes"])["config_secret_fields"] == {}


def test_a_failed_plugin_that_prints_the_key_is_reported_without_it(
    worker, monkeypatch
):
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda c: CRED)
    _plugin(monkeypatch, {}, stderr="auth failed for " + CRED + "\n", returncode=1)
    with pytest.raises(RuntimeError) as raised:
        _run({"threshold": 0.5}, [s("secrets", "config.model_credential", "c1")])
    assert CRED not in str(raised.value)
    assert "[redacted]" in str(raised.value)
    assert worker["failed"]
    for _evaluation, _plugin_pid, text in worker["failed"]:
        assert CRED not in text
        assert "[redacted]" in text


def test_the_runtime_fills_each_named_field_from_the_environment():
    config = {"threshold": 0.5, "judge": {"model": "m"}, "keys": ["", "plain"]}
    before = copy.deepcopy(config)
    filled = plugin_runtime.fill_config_secrets(
        config,
        {
            "model_credential": "E1",
            "judge.phrase": "E2",
            "keys[0]": "E3",
            "targets[1].token": "E4",
            "missing": "E5",
        },
        {"E1": "v1", "E2": "v2", "E3": "v3", "E4": "v4"},
    )
    assert filled == {
        "threshold": 0.5,
        "judge": {"model": "m", "phrase": "v2"},
        "keys": ["v3", "plain"],
        "model_credential": "v1",
    }
    assert config == before


class _Recorded:
    configs: list = []

    def _set_artifact_callback(self, callback):
        pass

    def _set_progress_callback(self, callback):
        pass

    def set_input_content(self, name, content):
        pass

    def evaluate(self, config):
        _Recorded.configs.append(config)
        return []

    def export_metrics(self, output):
        return []


def test_the_runtime_hands_evaluate_the_filled_config_and_prints_no_key(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "output").mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "plugin_source": "pkg:Plugin",
                "input_mapping": {},
                "project_settings": {},
                "plugin_config": {"threshold": 0.5},
                "config_secret_fields": {"model_credential": VARIABLE},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(VARIABLE, CRED)
    monkeypatch.setattr(plugin_runtime, "load_plugin", lambda spec: _Recorded)
    _Recorded.configs = []
    try:
        plugin_runtime.main()
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert _Recorded.configs == [{"threshold": 0.5, "model_credential": CRED}]
    assert CRED not in captured.out
    assert CRED not in captured.err
