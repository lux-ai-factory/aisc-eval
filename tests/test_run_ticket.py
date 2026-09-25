"""I7.3 (isolation 2026-09-25): the worker names its project and carries a run ticket.

The engine's data lives in one database per project. The worker has no database
access: it calls the backend's /api/v1/internal/* routes, and the backend's door
decides which project database to open from X-AISC-Project, and accepts the
internal call only when X-AISC-Run is the ticket the backend minted for that
project and that evaluation (HMAC-SHA256 over platform_pid + "." +
evaluation_pid with DJANGO_SECRET_KEY, minted by the backend, never here).

So, from 01-specs.md I7.3:
- the task is run_evaluation(platform_pid, evaluation_pid, ticket);
- every call to /api/v1/internal/* sends X-Internal-Secret, X-AISC-Project and
  X-AISC-Run, including the calls made by the tasks run_evaluation dispatches;
- tasks never carry a DSN;
- a task called the old way, run_evaluation(evaluation_pid), fails clearly and
  calls nothing.

Decided in stage 2 (isolation 02-tests.md, gap E-G1): three internal routes
carry no evaluation in their path (/projects/settings/{project_pid}[/by-pid],
/files/dataset/{name}, /files/model/{name}), so the door could not bind the
ticket to an evaluation there. Every internal call therefore also sends
X-AISC-Evaluation: <evaluation pid>; the door checks the ticket against that
header and, where the path names an evaluation, requires the two to be equal.

No broker, no backend, no database: requests and the Celery canvas are faked.
These tests need none of tests/conftest.py (run them with --noconftest while
that file's imports are broken, see isolation 02-tests.md).
"""
import inspect
import re
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from celery import canvas

from aisc_eval import celery_tasks
from aisc_eval.utils import env

PLATFORM_PID = "3f2b8c1e-0d4a-4e7b-9a55-1c2d3e4f5a6b"
ENGINE_PROJECT_PID = "afb49e3f-813d-8888-9919-ee179d1090e6"
EVALUATION_PID = "afb49e3f-813d-4260-9919-ee179d1090e6"
PLUGIN_PID = "750ec557-fd8c-4f94-92b9-28796591fd40"
TICKET = "0" * 16 + "a-ticket-minted-by-the-backend" + "f" * 16

WORKER_ROOT = Path(__file__).resolve().parents[1]
DSN_PATTERN = re.compile(r"postgres(ql)?(\+\w+)?://|dbname=|DATABASE_URL|_DSN\b", re.IGNORECASE)


class FakeBackend:
    """Stands in for the backend's internal API; records every call."""

    def __init__(self, has_failed_plugins=False):
        self.calls = []
        self.has_failed_plugins = has_failed_plugins

    def _response(self, status, body):
        response = MagicMock(spec=requests.Response)
        response.status_code = status
        response.ok = 200 <= status < 300
        response.json.return_value = body
        response.text = ""
        response.content = b""
        response.headers = {}
        response.raise_for_status.return_value = None
        return response

    def handle(self, method, url, headers=None, **_kwargs):
        self.calls.append((method, url, dict(headers or {})))
        if method == "GET" and url.endswith(f"/evaluations/{EVALUATION_PID}?include=project,plugin"):
            return self._response(200, {
                "pid": EVALUATION_PID,
                "project": {"pid": ENGINE_PROJECT_PID, "name": "MCAS"},
                "evaluation_plugins": [{
                    "pid": PLUGIN_PID, "name": "probe", "package_name": "aisc-probe", "version": "1.0.0",
                }],
            })
        if method == "GET" and url.endswith("/inputs"):
            return self._response(200, {PLUGIN_PID: [
                {"component_type": "dataset", "name": "train", "data": "d0c5e7a1-dataset.csv"},
                {"component_type": "model", "name": "clf", "data": "m0c5e7a1-model.onnx"},
            ]})
        if url.endswith("/by-pid"):
            return self._response(200, [])
        if url.endswith("/plugins/status"):
            return self._response(200, {"has_failed_plugins": self.has_failed_plugins})
        if method == "POST" and url.endswith("/measures"):
            return self._response(201, {})
        return self._response(200, {})

    def internal_calls(self):
        return [c for c in self.calls if "/api/v1/internal/" in c[1] or c[1].startswith(env.API_URL_PREFIX)]


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    for verb in ("get", "post", "put", "patch", "delete"):
        monkeypatch.setattr(
            requests, verb,
            (lambda method: (lambda url, *a, **kw: fake.handle(method, url, **kw)))(verb.upper()),
        )
    return fake


@pytest.fixture
def dispatched(monkeypatch):
    """Everything run_evaluation hands to Celery, captured instead of sent."""
    sent = []

    def capture(self, *args, **kwargs):
        sent.append(self)
        return MagicMock()

    for cls in (canvas.Signature, canvas._chain, canvas.group, canvas._chord):
        monkeypatch.setattr(cls, "apply_async", capture)
        monkeypatch.setattr(cls, "delay", lambda self, *a, **kw: capture(self))
    fake_app = MagicMock()
    monkeypatch.setattr(celery_tasks, "celery_app", fake_app)
    monkeypatch.setattr(celery_tasks.plugin_loader, "list_packages", lambda *a, **kw: {})
    return sent


def _leaves(sig):
    if sig is None:
        return
    if isinstance(sig, (list, tuple)):
        for item in sig:
            yield from _leaves(item)
        return
    sig = canvas.maybe_signature(sig)
    if isinstance(sig, canvas._chord):
        yield from _leaves(sig.tasks)
        yield from _leaves(sig.body)
    elif isinstance(sig, (canvas.group, canvas._chain)):
        yield from _leaves(list(sig.tasks))
    else:
        yield sig


def _values(sig):
    return [str(v) for v in list(sig.args) + list(sig.kwargs.values())]


def _run_with_ticket():
    run = celery_tasks.run_evaluation.run
    try:
        return run(uuid.UUID(PLATFORM_PID), uuid.UUID(EVALUATION_PID), TICKET)
    except TypeError as exc:
        pytest.fail(f"I7.3: run_evaluation does not take (platform_pid, evaluation_pid, ticket): {exc}")


def _assert_headers(calls, what):
    assert calls, f"I7.3: {what} made no internal call at all"
    for method, url, headers in calls:
        assert headers.get("X-Internal-Secret") == env.INTERNAL_API_KEY, (
            f"I7.3: {what}: {method} {url} lacks X-Internal-Secret")
        assert headers.get("X-AISC-Project") == PLATFORM_PID, (
            f"I7.3: {what}: {method} {url} does not name its project in X-AISC-Project")
        assert headers.get("X-AISC-Run") == TICKET, (
            f"I7.3: {what}: {method} {url} does not carry the run ticket in X-AISC-Run")
        assert headers.get("X-AISC-Evaluation") == EVALUATION_PID, (
            f"I7.3 (E-G1): {what}: {method} {url} does not name its evaluation in X-AISC-Evaluation")


def test_i7_3_run_evaluation_takes_platform_pid_evaluation_pid_and_ticket():
    parameters = [
        p.name for p in inspect.signature(celery_tasks.run_evaluation.run).parameters.values()
        if p.name != "self"
    ]
    assert parameters == ["platform_pid", "evaluation_pid", "ticket"], (
        f"I7.3: run_evaluation's parameters are {parameters}")


def test_i7_3_the_old_one_argument_call_fails_clearly_and_calls_nothing(backend, dispatched):
    with pytest.raises(TypeError):
        celery_tasks.run_evaluation.run(uuid.UUID(EVALUATION_PID))
    assert backend.calls == [], "I7.3: an old-style task reached the backend"
    assert dispatched == [], "I7.3: an old-style task dispatched work"


def test_i7_3_every_internal_call_of_run_evaluation_names_the_project_and_the_ticket(backend, dispatched):
    _run_with_ticket()
    _assert_headers(backend.internal_calls(), "run_evaluation")


def test_i7_3_every_dispatched_task_carries_the_project_and_the_ticket(backend, dispatched):
    _run_with_ticket()
    assert dispatched, "I7.3: run_evaluation dispatched nothing"
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    names = {leaf.task.rsplit(".", 1)[-1] for leaf in leaves}
    assert {"install_package", "run_plugin", "post_measurements", "finalize_evaluation"} <= names
    for leaf in leaves:
        values = _values(leaf)
        assert PLATFORM_PID in values, f"I7.3: {leaf.task} is dispatched without the platform pid"
        assert TICKET in values, f"I7.3: {leaf.task} is dispatched without the run ticket"


def test_i7_3_no_dispatched_task_carries_a_dsn(backend, dispatched):
    _run_with_ticket()
    for sent in dispatched:
        for leaf in _leaves(sent):
            for value in _values(leaf):
                assert not DSN_PATTERN.search(value), f"I7.3: {leaf.task} carries a DSN-like argument"


@pytest.mark.parametrize("has_failed_plugins", [False, True], ids=["completed", "failed"])
def test_i7_3_finalize_and_post_measurements_send_the_project_and_the_ticket(
    backend, dispatched, has_failed_plugins
):
    _run_with_ticket()
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    backend.has_failed_plugins = has_failed_plugins
    backend.calls.clear()
    finals = [leaf for leaf in leaves if leaf.task.endswith("finalize_evaluation")]
    assert finals, "I7.3: no finalize_evaluation dispatched"
    for leaf in finals:
        celery_tasks.finalize_evaluation.run(*leaf.args, **leaf.kwargs)
    _assert_headers(backend.internal_calls(), "finalize_evaluation")

    backend.calls.clear()
    posts = [leaf for leaf in leaves if leaf.task.endswith("post_measurements")]
    assert posts, "I7.3: no post_measurements dispatched"
    for leaf in posts:
        # a chained .s(): the previous task's result (the measures) comes first
        celery_tasks.post_measurements.run([], *leaf.args, **leaf.kwargs)
    _assert_headers(backend.internal_calls(), "post_measurements")


class _Failed:
    returncode = 1
    stdout = ""
    stderr = "simulated: no venv here"


def test_i7_3_run_plugin_downloads_and_reports_with_the_project_and_the_ticket(
    backend, dispatched, monkeypatch, tmp_path
):
    """run_plugin fetches its input files and reports a failure through the
    internal API; each of those calls names the project and carries the ticket.
    The venv step is faked to fail, so no plugin runs."""
    _run_with_ticket()
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    runs = [leaf for leaf in leaves if leaf.task.endswith("run_plugin")]
    assert runs, "I7.3: no run_plugin dispatched"
    monkeypatch.setattr(celery_tasks.plugin_loader, "discovered_packages",
                        {"aisc-probe": {"1.0.0": {"source": "local", "pkg_root": tmp_path}}})
    monkeypatch.setattr(celery_tasks.subprocess, "run", lambda *a, **kw: _Failed())
    backend.calls.clear()
    for leaf in runs:
        with pytest.raises(Exception):
            celery_tasks.run_plugin.run(*leaf.args, **leaf.kwargs)
    urls = [url for _, url, _ in backend.internal_calls()]
    assert any("/files/dataset/" in u for u in urls), "the dataset download was not reached"
    assert any("/files/model/" in u for u in urls), "the model download was not reached"
    assert any(u.endswith("/fail") for u in urls), "the failure report was not reached"
    _assert_headers(backend.internal_calls(), "run_plugin")


def test_i7_3_install_package_reports_a_failure_with_the_project_and_the_ticket(
    backend, dispatched, monkeypatch, tmp_path
):
    _run_with_ticket()
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    installs = [leaf for leaf in leaves if leaf.task.endswith("install_package")]
    assert installs, "I7.3: no install_package dispatched"
    monkeypatch.setattr(celery_tasks.plugin_loader, "discovered_packages",
                        {"aisc-probe": {"1.0.0": {"source": "local", "pkg_root": tmp_path}}})

    def boom(*_a, **_kw):
        raise RuntimeError("simulated: uv is not here")

    monkeypatch.setattr(celery_tasks.subprocess, "run", boom)
    backend.calls.clear()
    for leaf in installs:
        with pytest.raises(Exception):
            celery_tasks.install_package.run(*leaf.args, **leaf.kwargs)
    _assert_headers(backend.internal_calls(), "install_package")


def test_i7_3_the_worker_settings_name_no_database_and_no_ticket_key():
    """Guard (passes today, must keep passing): no DSN in the worker's settings,
    and the worker cannot mint tickets (DJANGO_SECRET_KEY is the backend's)."""
    sources = [WORKER_ROOT / "aisc_eval" / "utils" / "env.py", WORKER_ROOT / "aisc_eval" / "celery_app.py"]
    sources += [p for p in (WORKER_ROOT / "env.development",) if p.exists()]
    for source in sources:
        text = source.read_text()
        assert not DSN_PATTERN.search(text), f"I7.3: {source.name} names a database"
        if source.suffix == ".py":
            assert "DJANGO_SECRET_KEY" not in text, f"I7.3: {source.name} reads the ticket key"
    names = inspect.signature(celery_tasks.run_evaluation.run).parameters
    assert not any("dsn" in n.lower() or "database" in n.lower() for n in names)
