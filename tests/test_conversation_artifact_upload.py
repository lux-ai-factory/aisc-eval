"""The worker marks the files it uploads that hold a conversation with the system under test: the run's
recorded calls and the plugin's execution log go to the engine with holds_conversation=true, so the engine
shows them to the project's owners and editors only. Any other file goes unmarked (the engine still applies its
own name rule)."""

import uuid

import http_fake

from aisc_eval import celery_tasks
from aisc_eval.service import api_client

EVALUATION = uuid.UUID("11111111-2222-4333-8444-555555555555")
PLUGIN_RUN = uuid.UUID("66666666-7777-4888-9999-aaaaaaaaaaaa")


class _Ok:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"file_name": "x"}


def _sent_forms(monkeypatch):
    sent = []
    http_fake.route(
        monkeypatch,
        post=lambda url, data=None, files=None, **k: sent.append(data or {}) or _Ok(),
    )
    return sent


def test_the_upload_says_when_a_file_holds_a_conversation(monkeypatch):
    sent = _sent_forms(monkeypatch)
    try:
        api_client.upload_artifact(
            EVALUATION,
            PLUGIN_RUN,
            celery_tasks.CALLS_FILE,
            b"{}",
            holds_conversation=True,
        )
    except TypeError as exc:
        raise AssertionError(
            f"the upload cannot say a file holds a conversation: {exc}"
        ) from None
    assert sent[0].get("holds_conversation") == "true"


def test_an_ordinary_file_is_uploaded_unmarked(monkeypatch):
    sent = _sent_forms(monkeypatch)
    api_client.upload_artifact(EVALUATION, PLUGIN_RUN, "results.csv", b"a,b\n")
    assert sent[0].get("holds_conversation") in (None, "false")


def test_the_runs_recorded_calls_are_uploaded_as_a_conversation(monkeypatch):
    uploads = []
    monkeypatch.setattr(
        celery_tasks, "upload_artifact", lambda *a, **k: uploads.append((a, k))
    )
    celery_tasks._keep_the_runs_calls(EVALUATION, PLUGIN_RUN, b'{"prompt": "hi"}\n', [])
    assert len(uploads) == 1
    args, kwargs = uploads[0]
    assert args[2] == celery_tasks.CALLS_FILE
    assert (
        kwargs.get("holds_conversation") is True
    ), "the run's recorded calls are uploaded unmarked"
