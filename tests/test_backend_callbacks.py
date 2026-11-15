"""The worker's calls to the backend survive a backend that is briefly away.

A call that cannot reach the backend (a restart, a dropped network) is tried again after 1, 2, 4, 8 and 16 seconds,
then gives up and raises, so the failure path and the sweep take over. A refusal (the backend answered and said no)
is not retried. The worker can ask for
Processing and learns the run's actual status. A test is never failed without a reason.
"""

import inspect

import pytest
import requests

from aisc_eval.service import api_client


class _Answer:
    def __init__(self, status, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = text
        self.headers = {}

    def json(self):
        return self.text

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}", response=self)


def ok(text="ok"):
    return _Answer(200, text)


def refused(status):
    return _Answer(status, {"detail": "no"})


class _Calls:
    def __init__(self):
        self.put, self.patch, self.get, self.post = [], [], [], []
        self.urls, self.sleeps = [], []


@pytest.fixture
def calls(monkeypatch):
    seen = _Calls()

    def method(name):
        def call(url, *args, **kwargs):
            seen.urls.append(url)
            queue = getattr(seen, name)
            if not queue:
                raise AssertionError(f"one {name.upper()} more than the test expected: {url}")
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        return call

    for name in ("put", "patch", "get", "post"):
        monkeypatch.setattr(api_client.requests, name, method(name))
    monkeypatch.setattr(api_client, "sleep", seen.sleeps.append, raising=False)
    return seen


def test_a_callback_retries_when_the_backend_cannot_be_reached(calls):
    calls.put = [requests.ConnectionError("refused"), requests.ConnectionError("refused"), ok("Done")]
    try:
        api_client.mark_completed("e1")
    except requests.ConnectionError:
        pass
    assert calls.sleeps == [1, 2]


def test_a_callback_gives_up_after_its_last_wait(calls):
    waits = getattr(api_client, "RETRY_WAITS", (1, 2, 4, 8, 16))
    calls.put = [requests.ConnectionError("refused")] * (len(waits) + 1)
    with pytest.raises(requests.ConnectionError):
        api_client.mark_completed("e1")
    assert calls.sleeps == list(waits)


def test_a_refusal_is_not_retried(calls):
    calls.put = [refused(500)]
    with pytest.raises(requests.HTTPError):
        api_client.mark_completed("e1")
    assert calls.sleeps == []


def test_the_failure_marks_retry_too(calls):
    calls.put = [requests.ConnectionError("x"), ok("Failed")]
    calls.patch = [requests.ConnectionError("x"), ok("ok")]
    try:
        api_client.mark_failed("e1")
        api_client.mark_plugin_failed("e1", "ep", "boom")
    except requests.ConnectionError:
        pass
    assert calls.sleeps == [1, 1]


def test_processing_is_asked_with_the_runs_status(calls):
    calls.put = [ok("Failed")]
    mark_processing = getattr(api_client, "mark_processing", None)
    assert mark_processing is not None, "the worker cannot ask for Processing"
    assert mark_processing("e1") == "Failed"
    assert calls.urls[-1].endswith("/evaluations/e1?status=Processing")


def test_a_test_is_never_failed_without_a_reason():
    default = inspect.signature(api_client.mark_plugin_failed).parameters["error_message"].default
    assert default is inspect.Parameter.empty
