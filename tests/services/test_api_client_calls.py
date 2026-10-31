"""Every call the worker makes to the backend gives up after a set time, and a refusal is an error that
names the answer's status (and the backend's reason, when it gives one), never the backend's address. Marking
a test failed is the exception: it is the last report of a failure, so a refusal of it is only logged."""

import uuid

import pytest
import requests
from aisc_plugin_interface.models.measure import Measure

from aisc_eval.service import api_client

EVALUATION = uuid.UUID("afb49e3f-813d-4260-9919-ee179d1090e6")
PLUGIN_RUN = uuid.UUID("f29a97fb-b0b7-476e-a07e-da60741fc6e9")


class _Answer:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.ok = status < 400
        self._body = {} if body is None else body
        self.text = str(self._body)
        self.content = b"bytes"
        self.headers = {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} for url: {api_client.API_URL_PREFIX}/x")


@pytest.fixture
def backend(monkeypatch):
    """requests' verbs, recording the keyword arguments of each call; answers with seen['answer']."""
    seen = {"calls": [], "answer": _Answer()}
    for verb in ("get", "put", "patch", "post"):
        def call(url, _verb=verb, **kwargs):
            seen["calls"].append((_verb, url, kwargs))
            return seen["answer"]

        monkeypatch.setattr(api_client.requests, verb, call)
    return seen


#: every call of the api_client, with arguments that reach the network
CALLS = {
    "get_project_settings_by_pid": lambda: api_client.get_project_settings_by_pid(EVALUATION, []),
    "get_evaluation_plugins_status": lambda: api_client.get_evaluation_plugins_status(EVALUATION),
    "mark_completed": lambda: api_client.mark_completed(EVALUATION),
    "mark_failed": lambda: api_client.mark_failed(EVALUATION),
    "mark_plugin_failed": lambda: api_client.mark_plugin_failed(EVALUATION, PLUGIN_RUN, "x"),
    "mark_plugin_started": lambda: api_client.mark_plugin_started(EVALUATION, PLUGIN_RUN),
    "mark_plugin_finished": lambda: api_client.mark_plugin_finished(EVALUATION, PLUGIN_RUN),
    "get_dataset_file_content": lambda: api_client.get_dataset_file_content("d.csv"),
    "get_model_file_content": lambda: api_client.get_model_file_content("m.pkl"),
    "get_evaluation_request": lambda: api_client.get_evaluation_request(EVALUATION),
    "get_evaluation_inputs": lambda: api_client.get_evaluation_inputs(EVALUATION),
    "store_input_value": lambda: api_client.store_input_value(EVALUATION, PLUGIN_RUN, {}),
    "upload_artifact": lambda: api_client.upload_artifact(EVALUATION, PLUGIN_RUN, "a.txt", b"a"),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_call_gives_up_in_time(backend, name):
    CALLS[name]()
    [(_verb, _url, kwargs)] = backend["calls"]
    assert kwargs["timeout"] == api_client.TIMEOUT


def test_posting_measures_gives_up_in_time(backend):
    backend["answer"] = _Answer(201)
    api_client.post_measures(EVALUATION, PLUGIN_RUN, [Measure(name="m", score=1.0)])
    assert backend["calls"][0][2]["timeout"] == api_client.TIMEOUT


@pytest.mark.parametrize(
    "name", ["mark_completed", "mark_plugin_started", "mark_plugin_finished", "upload_artifact"]
)
def test_a_refused_report_is_an_error_with_its_status_and_not_the_backends_address(backend, name):
    backend["answer"] = _Answer(503, {"detail": "the engine is restarting"})
    with pytest.raises(requests.HTTPError) as refused:
        CALLS[name]()
    assert str(refused.value) == "the backend answered HTTP 503: the engine is restarting"
    assert api_client.API_URL_PREFIX not in str(refused.value)
    assert refused.value.response.status_code == 503


def test_a_refused_read_does_not_name_the_backends_address(backend):
    backend["answer"] = _Answer(404, ["not", "a", "detail"])
    with pytest.raises(requests.HTTPError) as refused:
        api_client.get_dataset_file_content("d.csv")
    assert str(refused.value) == "the backend answered HTTP 404"


def test_refused_measures_say_why(backend):
    backend["answer"] = _Answer(400, {"detail": "name too long"})
    with pytest.raises(requests.HTTPError, match="HTTP 400: name too long"):
        api_client.post_measures(EVALUATION, PLUGIN_RUN, [Measure(name="m", score=1.0)])


def test_a_refused_failure_mark_is_only_logged(backend):
    backend["answer"] = _Answer(500)
    api_client.mark_plugin_failed(EVALUATION, PLUGIN_RUN, "x")


def test_a_refusal_chains_nothing_that_names_the_backends_address(backend):
    backend["answer"] = _Answer(500)
    with pytest.raises(requests.HTTPError) as refused:
        api_client.mark_completed(EVALUATION)
    assert refused.value.__context__ is None and refused.value.__cause__ is None
