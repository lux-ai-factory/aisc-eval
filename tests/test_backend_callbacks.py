"""The worker's calls to the backend survive a backend that is briefly away.

A call that cannot reach the backend (a restart, a dropped network) is tried again after 1, 2, 4, 8 and 16 seconds,
then gives up and raises, so the failure path and the sweep take over. A refusal (the backend answered and said no)
is not retried, except the gateway answers a proxy gives while the backend restarts. Marking a test failed is never
silent: a refused mark raises instead of leaving the test Running and its run Pending. The worker can ask for
Processing and learns the run's actual status. A test is never failed without a reason.
"""

import inspect

import http_fake
import pytest
from aisc_service_client import ServiceRefused, ServiceUnreachable

from aisc_eval.service import api_client


class _Answer:
    def __init__(self, status, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = text
        self.headers = {}

    def json(self):
        return self.text


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

    http_fake.route(monkeypatch, **{name: method(name) for name in ("put", "patch", "get", "post")})
    monkeypatch.setattr(api_client, "sleep", seen.sleeps.append, raising=False)
    return seen


def test_a_callback_retries_when_the_backend_cannot_be_reached(calls):
    calls.put = [http_fake.unreachable("refused"), http_fake.unreachable("refused"), ok("Done")]
    try:
        api_client.mark_completed("e1")
    except ServiceUnreachable:
        pass
    assert calls.sleeps == [1, 2]


def test_a_callback_gives_up_after_its_last_wait(calls):
    waits = getattr(api_client, "RETRY_WAITS", (1, 2, 4, 8, 16))
    calls.put = [http_fake.unreachable("refused")] * (len(waits) + 1)
    with pytest.raises(ServiceUnreachable):
        api_client.mark_completed("e1")
    assert calls.sleeps == list(waits)


def test_a_refusal_is_not_retried(calls):
    calls.put = [refused(500)]
    with pytest.raises(ServiceRefused):
        api_client.mark_completed("e1")
    assert calls.sleeps == []


def test_the_failure_marks_retry_too(calls):
    calls.put = [http_fake.unreachable("x"), ok("Failed")]
    calls.patch = [http_fake.unreachable("x"), ok("ok")]
    try:
        api_client.mark_failed("e1")
        api_client.mark_plugin_failed("e1", "ep", "boom")
    except ServiceUnreachable:
        pass
    assert calls.sleeps == [1, 1]


def test_a_refused_fail_mark_is_not_only_logged(calls):
    calls.patch = [refused(500)]
    with pytest.raises(ServiceRefused):
        api_client.mark_plugin_failed("e1", "ep", "boom")


@pytest.mark.parametrize("status", [502, 503])
def test_a_fail_mark_refused_while_the_backend_restarts_is_tried_again(calls, status):
    calls.patch = [refused(status), ok("ok")]
    try:
        api_client.mark_plugin_failed("e1", "ep", "boom")
    except ServiceRefused:
        pass
    assert len(calls.urls) == 2, "the test stays Running: its failure was sent once and refused"


def test_processing_is_asked_with_the_runs_status(calls):
    calls.put = [ok("Failed")]
    mark_processing = getattr(api_client, "mark_processing", None)
    assert mark_processing is not None, "the worker cannot ask for Processing"
    assert mark_processing("e1") == "Failed"
    assert calls.urls[-1].endswith("/evaluations/e1?status=Processing")


def test_a_test_is_never_failed_without_a_reason():
    default = inspect.signature(api_client.mark_plugin_failed).parameters["error_message"].default
    assert default is inspect.Parameter.empty
