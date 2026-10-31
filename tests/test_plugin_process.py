"""A worker that runs as root starts each plugin as the image's unprivileged plugin user, so
the kernel refuses the plugin the worker's /proc/<pid>/environ (DJANGO_SECRET_KEY,
INTERNAL_API_KEY) and the worker's files. The worker then reads what the plugin left in its
output folder as root, so that folder must not let the plugin point it at anything else."""

import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.plugin_process import PluginUser

PLUGIN = PluginUser(uid=10001, gid=10001, home="/home/aisc-plugin", switch=True)


class _Entry:
    def __init__(self, uid, gid=10001, home="/home/aisc-plugin"):
        self.pw_uid, self.pw_gid, self.pw_dir = uid, gid, home


def _captured_popen(monkeypatch):
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return "process"

    monkeypatch.setattr(plugin_process.subprocess, "Popen", popen)
    return calls


def test_a_root_worker_runs_plugins_as_the_plugin_user(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)
    monkeypatch.setattr(plugin_process.pwd, "getpwnam", lambda name: _Entry(10001))
    assert plugin_process.plugin_user("aisc-plugin") == PLUGIN


def test_a_root_worker_without_the_plugin_user_refuses_to_run_the_plugin(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)

    def missing(name):
        raise KeyError(name)

    monkeypatch.setattr(plugin_process.pwd, "getpwnam", missing)
    with pytest.raises(KeyError):
        plugin_process.plugin_user("aisc-plugin")


def test_the_plugin_user_is_never_root(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        plugin_process.pwd, "getpwnam", lambda name: _Entry(0, gid=0, home="/root")
    )
    with pytest.raises(RuntimeError):
        plugin_process.plugin_user("root")


def test_a_worker_that_is_not_root_runs_plugins_as_itself(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(plugin_process.os, "getegid", lambda: 1000)
    assert plugin_process.plugin_user("aisc-plugin") == PluginUser(
        1000, 1000, None, switch=False
    )


def test_the_plugin_starts_as_its_user_with_no_groups_a_private_umask_and_its_own_home(
    monkeypatch, tmp_path
):
    calls = _captured_popen(monkeypatch)
    worker_env = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "DJANGO_SECRET_KEY": "django",
        "INTERNAL_API_KEY": "internal",
    }
    plugin_env = celery_tasks.plugin_environment(worker_env, {})
    with open(tmp_path / "stderr.tmp", "w") as stderr:
        plugin_process.start(["python", "run.py"], tmp_path, stderr, plugin_env, PLUGIN)

    [(argv, kwargs)] = calls
    assert argv == ["python", "run.py"]
    assert kwargs["user"] == 10001
    assert kwargs["group"] == 10001
    assert kwargs["extra_groups"] == []  # root's supplementary groups dropped
    assert kwargs["umask"] == 0o077
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["env"]["HOME"] == "/home/aisc-plugin"
    assert kwargs["env"]["PATH"] == "/usr/bin"
    assert "DJANGO_SECRET_KEY" not in kwargs["env"]
    assert "INTERNAL_API_KEY" not in kwargs["env"]


def test_a_worker_that_is_not_root_starts_the_plugin_without_switching(
    monkeypatch, tmp_path
):
    calls = _captured_popen(monkeypatch)
    me = PluginUser(1000, 1000, None, switch=False)
    with open(tmp_path / "stderr.tmp", "w") as stderr:
        plugin_process.start(["python"], tmp_path, stderr, {"HOME": "/home/dev"}, me)

    [(_, kwargs)] = calls
    assert kwargs["user"] is None
    assert kwargs["group"] is None
    assert kwargs["extra_groups"] is None
    assert kwargs["umask"] == 0o077
    assert kwargs["env"]["HOME"] == "/home/dev"


def test_the_run_folder_and_output_stay_roots_and_are_shared_with_the_plugin_group(
    monkeypatch, tmp_path
):
    chowns = []
    monkeypatch.setattr(
        plugin_process.os,
        "chown",
        lambda path, uid, gid: chowns.append((path, uid, gid)),
    )
    output = tmp_path / "output"
    output.mkdir()
    plugin_process.prepare_workspace(tmp_path, output, PLUGIN)

    # owner unchanged (-1): with the sticky bit the plugin can add files but not rename or
    # replace config.json, input/, venv/ or output/ itself
    assert chowns == [(tmp_path, -1, 10001), (output, -1, 10001)]
    for folder in (tmp_path, output):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o1770


def test_the_runtime_script_is_copied_into_the_run_folder(tmp_path):
    script = plugin_process.copy_runtime(tmp_path)
    assert script.parent == tmp_path / "runtime"
    assert (
        script.read_bytes() == Path(plugin_process.plugin_runtime.__file__).read_bytes()
    )


# A plugin is the leader of its own process group: what it starts is ended with it.

#: a plugin that starts a sleeping grandchild, writes its pid, then (when told) waits itself
_STARTS_A_GRANDCHILD = """
import subprocess, sys, time
child = subprocess.Popen(["sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
open(sys.argv[1], "w").write(str(child.pid))
if sys.argv[2] == "wait":
    time.sleep(60)
"""


def _gone(pid: int, within: float = 5.0) -> bool:
    """True once pid has ended (a zombie waiting for its new parent counts as ended)."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def _start_grandparent(tmp_path, then):
    pid_file = tmp_path / "grandchild.pid"
    with open(tmp_path / "stderr.tmp", "w") as stderr:
        process = plugin_process.start(
            [sys.executable, "-c", _STARTS_A_GRANDCHILD, str(pid_file), then],
            tmp_path,
            stderr,
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            _me(),
        )
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "the plugin never started its grandchild"
        time.sleep(0.05)
    return process, int(pid_file.read_text())


def _kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_the_plugin_starts_in_a_session_of_its_own(monkeypatch, tmp_path):
    calls = _captured_popen(monkeypatch)
    with open(tmp_path / "stderr.tmp", "w") as stderr:
        plugin_process.start(["python"], tmp_path, stderr, {}, _me())
    [(_, kwargs)] = calls
    assert kwargs["start_new_session"] is True


def test_what_a_plugin_left_running_ends_when_the_plugin_ends(tmp_path):
    process, grandchild = _start_grandparent(tmp_path, "exit")
    try:
        assert process.stdout is not None and process.stdout.read() == ""
        assert plugin_process.wait(process) == 0
        assert _gone(grandchild)
    finally:
        _kill(grandchild)


def test_a_terminated_plugin_takes_what_it_started_with_it(tmp_path):
    process, grandchild = _start_grandparent(tmp_path, "wait")
    try:
        plugin_process.terminate(process)
        assert process.poll() is not None
        assert _gone(grandchild)
    finally:
        _kill(grandchild)
        _kill(process.pid)


def test_terminating_no_plugin_or_one_that_ended_is_harmless(tmp_path):
    plugin_process.terminate(None)
    process, grandchild = _start_grandparent(tmp_path, "exit")
    try:
        process.wait()
        plugin_process.terminate(process)
        assert _gone(grandchild)
    finally:
        _kill(grandchild)


def _me():
    return PluginUser(os.getuid(), os.getgid(), None, switch=False)


def test_the_plugins_regular_files_are_read(tmp_path):
    (tmp_path / "measures.json").write_text("[]")
    (tmp_path / "report.pdf").write_bytes(b"pdf")
    assert plugin_process.read_outputs(tmp_path, _me()) == {
        "measures.json": b"[]",
        "report.pdf": b"pdf",
    }


def test_a_symlink_in_the_output_is_refused(tmp_path):
    (tmp_path / "environ").symlink_to("/proc/1/environ")
    with pytest.raises(RuntimeError, match="environ"):
        plugin_process.read_outputs(tmp_path, _me())


def test_a_file_the_plugin_does_not_own_is_refused(tmp_path):
    (tmp_path / "data.csv").write_text("a,b")
    someone_else = PluginUser(os.getuid() + 1, os.getgid(), None, switch=True)
    with pytest.raises(RuntimeError, match="data.csv"):
        plugin_process.read_outputs(tmp_path, someone_else)


def test_a_hard_link_in_the_output_is_refused(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "secret").write_text("s")
    os.link(tmp_path / "secret", output / "secret")
    with pytest.raises(RuntimeError, match="secret"):
        plugin_process.read_outputs(output, _me())


def test_a_folder_or_fifo_in_the_output_is_refused_without_blocking(tmp_path):
    (tmp_path / "pipe").mkdir()
    with pytest.raises(RuntimeError, match="pipe"):
        plugin_process.read_outputs(tmp_path, _me())
    (tmp_path / "pipe").rmdir()
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(RuntimeError, match="pipe"):
        plugin_process.read_outputs(tmp_path, _me())


class _Process:
    #: no group: killpg refuses a negative group, and the worker fixture records group signals anyway
    pid = -1

    def __init__(self, returncode=0):
        self.stdout = iter(["plain line\n"])
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class _Ok:
    returncode = 0
    stdout = ""
    stderr = ""


class _Loader:
    discovered_packages = {"pkg": {"1.0": {"source": "registry"}}}

    class devpi_client:
        simple_index_url = None


@pytest.fixture
def worker(monkeypatch):
    """run_plugin with the backend, the installs and the plugin process faked."""
    seen = {"uploads": {}, "failed": [], "popen": [], "group_signals": []}
    monkeypatch.setattr(celery_tasks, "plugin_loader", _Loader())
    monkeypatch.setattr(
        plugin_process.os,
        "killpg",
        lambda pgid, sig: seen["group_signals"].append((pgid, sig)),
    )
    monkeypatch.setattr(
        celery_tasks, "get_evaluation_request", lambda pid: {"status": "Running"}
    )
    monkeypatch.setattr(celery_tasks.subprocess, "run", lambda *a, **kw: _Ok())
    monkeypatch.setattr(celery_tasks, "mark_plugin_started", lambda *a: None)
    monkeypatch.setattr(celery_tasks, "mark_plugin_finished", lambda *a: None)
    monkeypatch.setattr(
        celery_tasks, "mark_plugin_failed", lambda *a: seen["failed"].append(a)
    )
    monkeypatch.setattr(celery_tasks, "_revoke_sibling_tasks", lambda *a: None)
    monkeypatch.setattr(
        celery_tasks,
        "upload_artifact",
        lambda evaluation, plugin, name, content: seen["uploads"].__setitem__(
            name, content
        ),
    )
    monkeypatch.setattr(
        plugin_process,
        "plugin_user",
        lambda: PluginUser(os.getuid(), os.getgid(), "/home/aisc-plugin", switch=True),
    )
    seen["leave"] = lambda output: (output / "measures.json").write_text("[]")

    def popen(argv, **kwargs):
        seen["popen"].append((argv, kwargs))
        seen["leave"](Path(kwargs["cwd"]) / "output")
        return _Process()

    monkeypatch.setattr(plugin_process.subprocess, "Popen", popen)
    return seen


def _run():
    return celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep")


def test_run_plugin_starts_the_plugin_as_the_plugin_user(worker):
    assert _run() == []
    [(argv, kwargs)] = worker["popen"]
    assert kwargs["user"] == os.getuid()
    assert kwargs["extra_groups"] == []
    assert kwargs["env"]["HOME"] == "/home/aisc-plugin"
    # the runtime script is read from the run folder: the plugin user cannot read /app
    assert Path(argv[1]).parent == Path(kwargs["cwd"]) / "runtime"


def test_run_plugin_ends_what_the_plugin_left_running(worker):
    _run()
    assert worker["group_signals"] == [(-1, signal.SIGKILL)]


def test_run_plugin_uploads_its_own_log_and_the_plugins_files(worker):
    def leave(output):
        (output / "measures.json").write_text("[]")
        (output / "chart.png").write_bytes(b"png")
        (output / "plugin_execution.log").write_text("forged")

    worker["leave"] = leave
    _run()
    assert worker["uploads"]["chart.png"] == b"png"
    assert b"=== Plugin Execution Log ===" in worker["uploads"]["plugin_execution.log"]
    assert "measures.json" not in worker["uploads"]


def test_run_plugin_fails_a_plugin_that_points_its_output_elsewhere(worker):
    def leave(output):
        (output / "measures.json").symlink_to("/proc/1/environ")

    worker["leave"] = leave
    with pytest.raises(RuntimeError, match="measures.json"):
        _run()
    assert worker["failed"]
    assert worker["uploads"] == {}


class _Answer:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


#: the platform pid of the project the faked run is of, as the backend puts it in the run's header
PROJECT = "3f2b8c1e-0d4a-4e7b-9a55-1c2d3e4f5a6b"


@pytest.fixture
def platform(monkeypatch):
    """The platform's run-pass routes, faked; records every call. The run is of PROJECT: its run
    context is set as the backend's aisc_run header sets it, and reset afterwards."""
    from aisc_eval import run_context

    seen = {
        "calls": [],
        "mint": _Answer(
            201, {"pass": "the-pass", "expires_at": "2026-10-19T12:00:00Z"}
        ),
    }
    monkeypatch.setenv("PLATFORM_URL", "http://platform:8000")
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "stack-token")

    def post(url, json=None, headers=None, **kwargs):
        seen["calls"].append(("POST", url, dict(headers or {}), json))
        return seen["mint"]

    def delete(url, headers=None, **kwargs):
        seen["calls"].append(("DELETE", url, dict(headers or {}), None))
        return _Answer(204)

    monkeypatch.setattr(celery_tasks.run_pass.requests, "post", post)
    monkeypatch.setattr(celery_tasks.run_pass.requests, "delete", delete)
    token = run_context._current.set(
        {"project": PROJECT, "evaluation": "eval", "ticket": "t"}
    )
    yield seen
    run_context._current.reset(token)


TARGET = f"target:{PROJECT}/system"


def _run_with(inputs):
    return celery_tasks.run_plugin.run(
        "pkg", "Plugin", "1.0", {}, inputs, [], "eval", "ep"
    )


def _resource(name, value):
    return {"component_type": "resource", "name": name, "json_value": {"value": value}}


def test_the_plugin_never_gets_the_stack_connections_token(worker, platform):
    _run_with([_resource("target", TARGET)])
    [(_argv, kwargs)] = worker["popen"]
    assert "PLATFORM_CONNECTIONS_TOKEN" not in kwargs["env"]
    assert kwargs["env"]["PLATFORM_URL"] == "http://platform:8000"


def test_a_run_with_a_target_mints_a_pass_for_exactly_that_ref(worker, platform):
    _run_with(
        [
            _resource("target", TARGET),
            {"component_type": "datashape", "name": "shape", "json_value": {}},
        ]
    )
    method, url, headers, body = platform["calls"][0]
    assert (method, url) == ("POST", "http://platform:8000/internal/run-passes")
    assert headers["X-AISC-Service-Token"] == "stack-token"
    assert body == {"refs": [TARGET], "evaluation_plugin": "ep", "project": PROJECT}
    [(_argv, kwargs)] = worker["popen"]
    assert kwargs["env"]["PLATFORM_RUN_PASS"] == "the-pass"


def test_a_connection_input_is_a_ref_too(worker, platform):
    _run_with(
        [_resource("system", "connection:3f2b8c1e-0d4a-4e7b-9a55-1c2d3e4f5a6b/chatbot")]
    )
    assert platform["calls"][0][3]["refs"] == [
        "connection:3f2b8c1e-0d4a-4e7b-9a55-1c2d3e4f5a6b/chatbot"
    ]


def test_a_run_without_target_or_connection_inputs_mints_nothing(worker, platform):
    _run_with([_resource("other", "something-else")])
    assert platform["calls"] == []
    [(_argv, kwargs)] = worker["popen"]
    assert "PLATFORM_RUN_PASS" not in kwargs["env"]


def test_a_refused_mint_fails_the_run_before_the_plugin_starts(worker, platform):
    platform["mint"] = _Answer(403, {"detail": "not this project"})
    with pytest.raises(RuntimeError, match="run pass"):
        _run_with([_resource("target", TARGET)])
    assert worker["popen"] == []
    [(_evaluation, _plugin, message)] = worker["failed"]
    assert "run pass" in message and "403" in message
    assert [call[0] for call in platform["calls"]] == ["POST"]


def test_the_pass_is_revoked_after_the_run(worker, platform):
    _run_with([_resource("target", TARGET)])
    method, url, headers, _body = platform["calls"][-1]
    assert (method, url) == ("DELETE", "http://platform:8000/internal/run-passes")
    assert headers["X-AISC-Run-Pass"] == "the-pass"


def test_the_pass_is_revoked_also_when_the_plugin_fails(worker, platform, monkeypatch):
    monkeypatch.setattr(
        plugin_process.subprocess,
        "Popen",
        lambda argv, **kwargs: _Process(returncode=1),
    )
    with pytest.raises(RuntimeError, match="Plugin failed"):
        _run_with([_resource("target", TARGET)])
    assert platform["calls"][-1][0] == "DELETE"
    assert platform["calls"][-1][2]["X-AISC-Run-Pass"] == "the-pass"


def test_a_failed_revoke_does_not_fail_the_run(worker, platform, monkeypatch):
    def unreachable(*args, **kwargs):
        raise celery_tasks.run_pass.requests.ConnectionError("down")

    monkeypatch.setattr(celery_tasks.run_pass.requests, "delete", unreachable)
    assert _run_with([_resource("target", TARGET)]) == []
    assert worker["failed"] == []
