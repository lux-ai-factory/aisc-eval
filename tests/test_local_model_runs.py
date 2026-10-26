"""A run on a local model file is never taken for an AI card run.

When the plugin's local-model input (the backend marks it, from the plugin's @assesses) is bound to
an uploaded model file, the worker writes input/target.json as {"local_model": {"file", "sha256"}},
stores the same on that evaluation input, and asks the platform for no run pass: a local file opens
nothing on the platform. A card run's pass names the evaluation plugin it is for, so the platform
can record which connection served the run.
"""

import hashlib
import json

import pytest
import requests

from aisc_eval import celery_tasks
from test_plugin_process import TARGET, _resource, platform, worker  # noqa: F401  (fixtures)

MODEL_BYTES = b"an onnx graph"
MARKER = {
    "local_model": {
        "file": "scoring.onnx",
        "sha256": hashlib.sha256(MODEL_BYTES).hexdigest(),
    }
}


def _model_input(local_model=True, unknown=False):
    return {
        "pid": "input-1",
        "component_type": "model",
        "name": "model",
        "component_name": "scoring.onnx",
        "data": "0c5e7a1b-stored.onnx",
        "local_model": None if unknown else local_model,
        "local_model_unknown": unknown,
    }


def _dataset():
    return {
        "pid": "input-2",
        "component_type": "dataset",
        "name": "prompts",
        "data": "p.csv",
        "local_model": False,
        "local_model_unknown": False,
    }


def _judge():
    return {
        "component_type": "llm",
        "name": "helper:judge",
        "json_value": {"endpoint_url": "http://judge/v1"},
        "value": {"model": "judge-model"},
        "secret_key": "judge_key",
        "secret_encrypted_value": "cipher",
    }


@pytest.fixture
def run(worker, platform, monkeypatch):  # noqa: F811
    """run_plugin with the model download and the backend's input store faked; what the plugin saw."""
    seen = {"stored": [], "inputs": {}}
    monkeypatch.setattr(
        celery_tasks, "get_model_file_content", lambda name: MODEL_BYTES
    )
    monkeypatch.setattr(
        celery_tasks, "get_dataset_file_content", lambda name: b"prompt\nhello\n"
    )
    monkeypatch.setattr(
        celery_tasks,
        "store_input_value",
        lambda evaluation, input_pid, value: seen["stored"].append(
            (evaluation, input_pid, value)
        ),
    )
    monkeypatch.setattr(celery_tasks, "decrypt_value", lambda value: "judge-secret")

    def leave(output):
        seen["inputs"] = {
            path.name: path.read_bytes() for path in (output.parent / "input").iterdir()
        }
        (output / "measures.json").write_text("[]")

    worker["leave"] = leave

    def go(inputs):
        celery_tasks.run_plugin.run(
            "pkg", "Plugin", "1.0", {}, inputs, [], "eval", "ep"
        )
        return seen

    seen["worker"], seen["platform"] = worker, platform
    seen["go"] = go
    return seen


def test_a_local_run_is_marked_with_the_file_and_its_hash(run):
    seen = run["go"]([_model_input()])
    assert json.loads(seen["inputs"]["target.json"]) == MARKER
    assert seen["inputs"]["0c5e7a1b-stored.onnx"] == MODEL_BYTES


def test_a_local_run_stores_the_marker_on_its_model_input(run):
    seen = run["go"]([_model_input()])
    assert seen["stored"] == [("eval", "input-1", MARKER)]


def test_a_local_run_gets_no_run_pass(run):
    seen = run["go"]([_model_input()])
    assert seen["platform"]["calls"] == []
    [(_argv, kwargs)] = seen["worker"]["popen"]
    assert "PLATFORM_RUN_PASS" not in kwargs["env"]


def test_a_model_input_that_is_not_the_plugins_local_model_marks_nothing(run):
    seen = run["go"]([_model_input(local_model=False)])
    assert "target.json" not in seen["inputs"]
    assert seen["stored"] == []


STORE_REFUSED = (
    "could not store the local model file's name and hash on the run's input"
)


def test_a_run_whose_marker_cannot_be_stored_fails_before_the_plugin_starts(
    run, monkeypatch
):
    def refused(*_args):
        answer = requests.Response()
        answer.status_code = 500
        raise requests.HTTPError(
            "500 Server Error for url: http://backend:8000/api/v1/internal/x",
            response=answer,
        )

    monkeypatch.setattr(celery_tasks, "store_input_value", refused)
    with pytest.raises(RuntimeError, match=STORE_REFUSED):
        run["go"]([_model_input()])
    assert run["worker"]["popen"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert message == f"{STORE_REFUSED} (HTTP 500)"


def test_a_store_that_gets_no_answer_says_so_without_the_address(run, monkeypatch):
    def unreachable(*_args):
        raise requests.ConnectionError(
            "http://backend:8000/api/v1/internal/x: connection refused"
        )

    monkeypatch.setattr(celery_tasks, "store_input_value", unreachable)
    with pytest.raises(RuntimeError, match=STORE_REFUSED):
        run["go"]([_model_input()])
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert message == f"{STORE_REFUSED} (no answer)"


UNKNOWN = "could not tell whether the model file is a local target (the plugin could not be loaded)"


def test_a_model_file_the_backend_could_not_place_refuses_the_run(run):
    with pytest.raises(
        RuntimeError, match=UNKNOWN.replace("(", r"\(").replace(")", r"\)")
    ):
        run["go"]([_model_input(unknown=True), _dataset()])
    assert run["worker"]["popen"] == []
    assert run["platform"]["calls"] == []
    assert run["stored"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert message == UNKNOWN


def test_a_run_without_a_model_file_still_runs_when_the_plugin_could_not_be_loaded(run):
    seen = run["go"]([_dataset()])
    assert seen["worker"]["popen"]
    assert seen["worker"]["failed"] == []


def test_a_run_binding_a_local_model_and_a_card_target_is_refused(run):
    with pytest.raises(ValueError, match="one of them"):
        run["go"]([_model_input(), _resource("target", TARGET)])
    assert run["worker"]["popen"] == []
    assert run["platform"]["calls"] == []
    [(_evaluation, _plugin, message)] = run["worker"]["failed"]
    assert message == (
        "the run binds both a local model file and an AI card target:"
        " a run assesses one of them"
    )


def test_helper_models_of_a_local_run_are_written_as_llm_inputs(run):
    seen = run["go"]([_model_input(), _judge()])
    assert json.loads(seen["inputs"]["helper:judge.json"]) == {
        "endpoint_url": "http://judge/v1",
        "model": "judge-model",
    }
    [(_argv, kwargs)] = seen["worker"]["popen"]
    assert kwargs["env"]["AISC_SECRET_JUDGE_KEY"] == "judge-secret"


def test_a_card_run_mints_its_pass_for_its_evaluation_plugin(run):
    seen = run["go"]([_resource("target", TARGET), _judge()])
    method, url, _headers, body = seen["platform"]["calls"][0]
    assert (method, url) == ("POST", "http://platform:8000/internal/run-passes")
    assert body == {"refs": [TARGET], "evaluation_plugin": "ep"}
    assert seen["stored"] == []
    assert "target.json" in seen["inputs"]
    assert json.loads(seen["inputs"]["target.json"]) == {"value": TARGET}


def test_the_marker_is_stored_through_the_backends_internal_api(monkeypatch):
    from aisc_eval.service import api_client

    calls = []

    class Answer:
        def raise_for_status(self):
            calls.append("checked")

    def patch(url, json=None, headers=None, **kwargs):
        calls.append((url, json))
        return Answer()

    monkeypatch.setattr(api_client.requests, "patch", patch)
    api_client.store_input_value("eval", "input-1", MARKER)
    assert calls == [
        (
            f"{api_client.API_URL_PREFIX}/evaluations/eval/inputs/input-1/value",
            {"value": MARKER},
        ),
        "checked",
    ]
