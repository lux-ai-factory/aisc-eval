"""I7.3, by mode: the configurator worker names its project and carries a run
ticket on every internal call; the standalone worker keeps master's shape
untouched (a single evaluation_pid, no project headers).

Since adapt Task 4 (2026-09-28) the run no longer travels as task arguments:
the backend puts it in the `aisc_run` Celery header of run_evaluation, every
task published while a task runs inherits it (aisc_eval.run_context), and the
api_client reads it from run_context.current(). So the task signatures are
Meril's tip (96a8ec7) exactly, in both modes, and these tests set the run
context the way task_prerun does on a real worker (a broker-backed check of
the inheritance itself is tests/test_run_context_broker.py).

deployment.MODE (read once at import) picks which half of this file runs; run
this file twice, once per mode, to cover both halves.

The engine's data lives in one database per project. The worker has no database
access: it calls the backend's /api/v1/internal/* routes, and the backend's door
decides which project database to open from X-AISC-Project, and accepts the
internal call only when X-AISC-Run is the ticket the backend minted for that
project and that evaluation. In standalone there is no such door: every internal
call carries only Sean's X-Internal-Secret, and no task carries a project or a
ticket.

No broker, no backend, no database: requests and the Celery canvas are faked.
These tests need none of tests/conftest.py (its imports are broken on master
itself, independently of this task): run with --noconftest.
"""
import re
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from celery import canvas

from aisc_eval import celery_tasks, deployment, run_context
from aisc_eval.utils import env

PLATFORM_PID = "3f2b8c1e-0d4a-4e7b-9a55-1c2d3e4f5a6b"
ENGINE_PROJECT_PID = "afb49e3f-813d-8888-9919-ee179d1090e6"
EVALUATION_PID = "afb49e3f-813d-4260-9919-ee179d1090e6"
PLUGIN_PID = "750ec557-fd8c-4f94-92b9-28796591fd40"
TICKET = "0" * 16 + "a-ticket-minted-by-the-backend" + "f" * 16
RUN = {"project": PLATFORM_PID, "evaluation": EVALUATION_PID, "ticket": TICKET}

WORKER_ROOT = Path(__file__).resolve().parents[1]
DSN_PATTERN = re.compile(r"postgres(ql)?(\+\w+)?://|dbname=|DATABASE_URL|_DSN\b", re.IGNORECASE)

configurator_only = pytest.mark.skipif(deployment.MODE != deployment.CONFIGURATOR, reason="configurator run")
standalone_only = pytest.mark.skipif(deployment.MODE != deployment.STANDALONE, reason="standalone run")


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


class _Sent(list):
    """What was handed to Celery, and the run context each piece was published in
    (what run_context._forward copies into its aisc_run header)."""

    def __init__(self):
        super().__init__()
        self.published_in = []


@pytest.fixture
def acting():
    """The run context task_prerun sets on a real worker from the aisc_run header."""
    token = run_context._current.set(RUN)
    try:
        yield RUN
    finally:
        run_context._current.reset(token)


@pytest.fixture
def dispatched(monkeypatch):
    """Everything run_evaluation hands to Celery, captured instead of sent."""
    sent = _Sent()
    published_in = sent.published_in

    def capture(self, *args, **kwargs):
        sent.append(self)
        published_in.append(run_context.current())
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


def _run():
    """Sean's call, in both modes: run_evaluation(evaluation_pid)."""
    run = celery_tasks.run_evaluation.run
    try:
        return run(uuid.UUID(EVALUATION_PID))
    except TypeError as exc:
        pytest.fail(f"run_evaluation does not take (evaluation_pid), as Sean's worker does: {exc}")


def _assert_configurator_headers(calls, what):
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


def _assert_standalone_headers(calls, what):
    assert calls, f"standalone: {what} made no internal call at all"
    for method, url, headers in calls:
        assert headers.get("X-Internal-Secret") == env.INTERNAL_API_KEY, (
            f"standalone: {what}: {method} {url} lacks X-Internal-Secret")
        assert "X-AISC-Project" not in headers, f"standalone: {what}: {method} {url} names a project"
        assert "X-AISC-Run" not in headers, f"standalone: {what}: {method} {url} carries a run ticket"
        assert "X-AISC-Evaluation" not in headers, f"standalone: {what}: {method} {url} names an evaluation"


def _assert_arguments_carry_no_run(dispatched, what):
    """The run is in the headers, never in the arguments (adapt item 5)."""
    assert dispatched, f"{what}: run_evaluation dispatched nothing"
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    names = {leaf.task.rsplit(".", 1)[-1] for leaf in leaves}
    assert {"install_package", "run_plugin", "post_measurements", "finalize_evaluation"} <= names
    for leaf in leaves:
        values = _values(leaf)
        assert PLATFORM_PID not in values, f"{what}: {leaf.task} carries a platform pid argument"
        assert TICKET not in values, f"{what}: {leaf.task} carries a run ticket argument"
        assert "None" not in values[-2:], f"{what}: {leaf.task} carries trailing None arguments"
    return leaves


# ---------------------------------------------------------------------------
# Both modes
# ---------------------------------------------------------------------------

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


def test_the_three_argument_call_fails_clearly_and_calls_nothing(backend, dispatched):
    """Both modes now take Sean's single argument; the old configurator shape is a stale caller."""
    with pytest.raises(TypeError):
        celery_tasks.run_evaluation.run(uuid.UUID(PLATFORM_PID), uuid.UUID(EVALUATION_PID), TICKET)
    assert backend.calls == [], "a three-argument call reached the backend"
    assert list(dispatched) == [], "a three-argument call dispatched work"


def test_every_task_keeps_merils_signature():
    """The task signatures are 96a8ec7's: no platform_pid, no ticket, anywhere."""
    import inspect

    for task in (celery_tasks.install_package, celery_tasks.run_evaluation, celery_tasks.run_plugin,
                 celery_tasks.post_measurements, celery_tasks.finalize_evaluation, celery_tasks.handle_error):
        params = inspect.signature(task.run).parameters
        assert "platform_pid" not in params and "ticket" not in params, task.name


# ---------------------------------------------------------------------------
# Configurator only
# ---------------------------------------------------------------------------

@configurator_only
def test_i7_3_configurator_run_evaluation_calls_with_the_run_of_its_header(backend, dispatched, acting):
    _run()
    _assert_configurator_headers(backend.internal_calls(), "run_evaluation")


@configurator_only
def test_i7_3_configurator_every_dispatched_task_is_published_inside_the_run(backend, dispatched, acting):
    """The context is the run at each publish. This cannot show that a child inherits the run:
    nothing is sent over a broker here. Only test_run_context_broker.py proves inheritance."""
    _run()
    _assert_arguments_carry_no_run(dispatched, "configurator")
    assert dispatched.published_in and all(run == RUN for run in dispatched.published_in), (
        "I7.3: run_evaluation published work outside its run, so the children would lose the ticket")


@configurator_only
def test_i7_3_configurator_without_a_header_names_no_project(backend, dispatched):
    """No aisc_run header, no run: the calls go out without the door headers (and the door refuses them)."""
    _run()
    assert backend.internal_calls()
    for _, _, headers in backend.internal_calls():
        assert "X-AISC-Project" not in headers and "X-AISC-Run" not in headers


@configurator_only
def test_i7_3_no_dispatched_task_carries_a_dsn(backend, dispatched, acting):
    _run()
    for sent in dispatched:
        for leaf in _leaves(sent):
            for value in _values(leaf):
                assert not DSN_PATTERN.search(value), f"I7.3: {leaf.task} carries a DSN-like argument"


@configurator_only
@pytest.mark.parametrize("has_failed_plugins", [False, True], ids=["completed", "failed"])
def test_i7_3_configurator_finalize_and_post_measurements_send_the_project_and_the_ticket(
    backend, dispatched, acting, has_failed_plugins
):
    _run()
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    backend.has_failed_plugins = has_failed_plugins
    backend.calls.clear()
    finals = [leaf for leaf in leaves if leaf.task.endswith("finalize_evaluation")]
    assert finals, "I7.3: no finalize_evaluation dispatched"
    for leaf in finals:
        celery_tasks.finalize_evaluation.run(*leaf.args, **leaf.kwargs)
    _assert_configurator_headers(backend.internal_calls(), "finalize_evaluation")

    backend.calls.clear()
    posts = [leaf for leaf in leaves if leaf.task.endswith("post_measurements")]
    assert posts, "I7.3: no post_measurements dispatched"
    for leaf in posts:
        # a chained .s(): the previous task's result (the measures) comes first
        celery_tasks.post_measurements.run([], *leaf.args, **leaf.kwargs)
    _assert_configurator_headers(backend.internal_calls(), "post_measurements")


class _Failed:
    returncode = 1
    stdout = ""
    stderr = "simulated: no venv here"


@configurator_only
def test_i7_3_configurator_run_plugin_downloads_and_reports_with_the_project_and_the_ticket(
    backend, dispatched, acting, monkeypatch, tmp_path
):
    """run_plugin fetches its input files and reports a failure through the
    internal API; each of those calls names the project and carries the ticket.
    The venv step is faked to fail, so no plugin runs."""
    _run()
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
    _assert_configurator_headers(backend.internal_calls(), "run_plugin")


@configurator_only
def test_i7_3_configurator_install_package_reports_a_failure_with_the_project_and_the_ticket(
    backend, dispatched, acting, monkeypatch, tmp_path
):
    _run()
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
    _assert_configurator_headers(backend.internal_calls(), "install_package")


# ---------------------------------------------------------------------------
# Standalone only
# ---------------------------------------------------------------------------

@standalone_only
def test_i7_3_standalone_run_evaluation_takes_only_evaluation_pid(backend, dispatched):
    """As Sean's worker always took it: master's call, master's headers."""
    _run()
    _assert_standalone_headers(backend.internal_calls(), "run_evaluation")


@standalone_only
def test_i7_3_standalone_ignores_a_run_header_if_one_ever_arrives(backend, dispatched, acting):
    """Standalone never sets the header; even if a message carried one, no door header goes out."""
    _run()
    _assert_standalone_headers(backend.internal_calls(), "run_evaluation")


@standalone_only
def test_i7_3_standalone_dispatched_tasks_carry_no_project_or_ticket(backend, dispatched):
    _run()
    _assert_arguments_carry_no_run(dispatched, "standalone")
    assert all(run is None for run in dispatched.published_in), "standalone: work published inside a run"


@standalone_only
@pytest.mark.parametrize("has_failed_plugins", [False, True], ids=["completed", "failed"])
def test_i7_3_standalone_finalize_and_post_measurements_send_no_project_headers(
    backend, dispatched, has_failed_plugins
):
    _run()
    leaves = [leaf for sent in dispatched for leaf in _leaves(sent)]
    backend.has_failed_plugins = has_failed_plugins
    backend.calls.clear()
    finals = [leaf for leaf in leaves if leaf.task.endswith("finalize_evaluation")]
    assert finals, "standalone: no finalize_evaluation dispatched"
    for leaf in finals:
        celery_tasks.finalize_evaluation.run(*leaf.args, **leaf.kwargs)
    _assert_standalone_headers(backend.internal_calls(), "finalize_evaluation")

    backend.calls.clear()
    posts = [leaf for leaf in leaves if leaf.task.endswith("post_measurements")]
    assert posts, "standalone: no post_measurements dispatched"
    for leaf in posts:
        celery_tasks.post_measurements.run([], *leaf.args, **leaf.kwargs)
    _assert_standalone_headers(backend.internal_calls(), "post_measurements")
