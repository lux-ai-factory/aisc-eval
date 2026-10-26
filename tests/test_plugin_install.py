"""Installing a plugin runs its code: a source package builds itself, so its build code runs at install
time. The worker, root and holding DJANGO_SECRET_KEY and INTERNAL_API_KEY, installs as the image's install
user, with nothing of its own environment. That user is not the plugin user either: the install cache is
its own, and a plugin that could write to it would plant a package in the next run's install. The run's
environment is the install user's, so the plugin may read and run it but not change it."""

import os
import stat
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.plugin_process import PluginUser

INSTALLER = PluginUser(uid=10002, gid=10002, home="/home/aisc-install", switch=True)


class _Entry:
    def __init__(self, uid, gid=10002, home="/home/aisc-install"):
        self.pw_uid, self.pw_gid, self.pw_dir = uid, gid, home


def test_a_root_worker_installs_as_the_install_user(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)
    users = {
        "aisc-install": _Entry(10002),
        "aisc-plugin": _Entry(10001, gid=10001, home="/home/aisc-plugin"),
    }
    monkeypatch.setattr(plugin_process.pwd, "getpwnam", lambda name: users[name])
    assert plugin_process.install_user("aisc-install") == INSTALLER


def test_the_install_user_is_never_root_nor_the_plugin_user(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)
    for uid in (0, 10001):
        monkeypatch.setattr(
            plugin_process.pwd, "getpwnam", lambda name, uid=uid: _Entry(uid)
        )
        monkeypatch.setattr(
            plugin_process,
            "plugin_user",
            lambda: PluginUser(10001, 10001, None, switch=True),
        )
        with pytest.raises(RuntimeError):
            plugin_process.install_user("aisc-install")


def test_an_install_runs_as_its_user_with_its_own_cache_and_none_of_the_workers_environment(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        plugin_process.subprocess,
        "run",
        lambda argv, **kw: calls.append((argv, kw)) or "done",
    )
    monkeypatch.setenv("DJANGO_SECRET_KEY", "canary")
    monkeypatch.setenv("INTERNAL_API_KEY", "canary")
    assert (
        plugin_process.run_install(["uv", "pip", "install", "x"], INSTALLER, timeout=5)
        == "done"
    )
    [(argv, kw)] = calls
    assert argv == ["uv", "pip", "install", "x"]
    assert (kw["user"], kw["group"], kw["extra_groups"]) == (10002, 10002, [])
    assert kw["env"] == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aisc-install",
        "UV_CACHE_DIR": "/home/aisc-install/.cache/uv",
        "UV_NO_CONFIG": "1",
    }
    # not the worker's folder (/app, root's only): uv reads configuration from where it starts
    assert kw["cwd"] == "/home/aisc-install"
    # what it installs, the plugin user reads and runs
    assert kw["umask"] == 0o022
    assert kw["timeout"] == 5 and kw["capture_output"] and kw["text"]


def test_a_worker_that_is_not_root_installs_as_itself(monkeypatch):
    calls = []
    monkeypatch.setattr(
        plugin_process.subprocess, "run", lambda argv, **kw: calls.append(kw)
    )
    plugin_process.run_install(
        ["uv"], PluginUser(1000, 1000, None, switch=False), timeout=5
    )
    assert calls[0]["user"] is None and calls[0]["extra_groups"] is None
    assert calls[0]["env"]["PATH"] == "/usr/local/bin:/usr/bin:/bin"


def test_the_install_user_can_reach_the_folder_through_a_parent_that_is_roots(tmp_path):
    """A run folder is root's and closed until the plugin's group is let in, after the install: the install
    user may pass through it to the venv, without listing it."""
    me = PluginUser(os.getuid(), os.getgid(), None, switch=True)
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    plugin_process.prepare_install_folder(run / "venv", me)
    assert stat.S_IMODE(run.stat().st_mode) == 0o711


def test_the_venv_folder_is_the_install_users_and_closed_to_the_plugin(tmp_path):
    me = PluginUser(os.getuid(), os.getgid(), None, switch=True)
    venv = tmp_path / "venv"
    plugin_process.prepare_install_folder(venv, me)
    info = venv.stat()
    assert (info.st_uid, info.st_gid) == (os.getuid(), os.getgid())
    assert stat.S_IMODE(info.st_mode) == 0o755


class _Ok:
    returncode = 0
    stdout = ""
    stderr = ""


class _Loader:
    discovered_packages = {"pkg": {"1.0": {"source": "registry"}}}

    class devpi_client:
        simple_index_url = None


@pytest.fixture
def installs(monkeypatch):
    seen = []
    monkeypatch.setattr(celery_tasks, "plugin_loader", _Loader())
    monkeypatch.setattr(
        plugin_process,
        "install_user",
        lambda: PluginUser(os.getuid(), os.getgid(), None, switch=True),
    )

    def run_install(argv, user, timeout):
        seen.append((argv, user))
        return _Ok()

    monkeypatch.setattr(plugin_process, "run_install", run_install)
    # nothing installs as the worker
    monkeypatch.setattr(
        celery_tasks.subprocess,
        "run",
        lambda *a, **kw: pytest.fail(f"run as the worker: {a}"),
    )
    return seen


def test_the_cache_warming_install_runs_as_the_install_user(installs):
    celery_tasks.install_package.run("pkg", "1.0", "eval", ["ep"])
    assert [argv[:3] for argv, _ in installs] == [
        ["uv", "venv", installs[0][0][2]],
        ["uv", "pip", "install"],
    ]
    assert all(user.uid == os.getuid() for _, user in installs)


def test_a_runs_venv_is_made_and_filled_by_the_install_user(installs, monkeypatch):
    stopped = RuntimeError("stop before the plugin starts")
    monkeypatch.setattr(
        celery_tasks, "get_evaluation_request", lambda pid: {"status": "Running"}
    )
    monkeypatch.setattr(
        celery_tasks, "mark_plugin_started", lambda *a: (_ for _ in ()).throw(stopped)
    )
    monkeypatch.setattr(celery_tasks, "mark_plugin_failed", lambda *a: None)
    monkeypatch.setattr(celery_tasks, "_revoke_sibling_tasks", lambda *a: None)
    with pytest.raises(RuntimeError):
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep")
    venv_cmd, install_cmd = (argv for argv, _ in installs)
    assert venv_cmd[:2] == ["uv", "venv"] and "--allow-existing" in venv_cmd
    assert install_cmd[:3] == ["uv", "pip", "install"]
    assert Path(venv_cmd[2]).name == "venv"


def test_a_local_plugins_source_is_copied_where_the_install_user_may_read_it(tmp_path):
    """A local plugin's folder is under /app/plugins, and /app is root's only: the install reads a copy."""
    source = tmp_path / "app" / "plugins" / "pkg"
    (source / "src").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
    me = PluginUser(os.getuid(), os.getgid(), None, switch=True)
    copy = plugin_process.stage_local_source(source, tmp_path / "run" / "source", me)
    assert (copy / "pyproject.toml").read_text().startswith("[project]")
    assert all(p.stat().st_uid == os.getuid() for p in [copy, *copy.rglob("*")])


def test_a_runs_local_plugin_is_installed_from_its_copy(
    installs, monkeypatch, tmp_path
):
    source = tmp_path / "pkg"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
    monkeypatch.setattr(
        celery_tasks.plugin_loader,
        "discovered_packages",
        {"pkg": {"1.0": {"source": "local", "pkg_root": source}}},
    )
    stopped = RuntimeError("stop before the plugin starts")
    monkeypatch.setattr(
        celery_tasks, "get_evaluation_request", lambda pid: {"status": "Running"}
    )
    monkeypatch.setattr(
        celery_tasks, "mark_plugin_started", lambda *a: (_ for _ in ()).throw(stopped)
    )
    monkeypatch.setattr(celery_tasks, "mark_plugin_failed", lambda *a: None)
    monkeypatch.setattr(celery_tasks, "_revoke_sibling_tasks", lambda *a: None)
    with pytest.raises(RuntimeError):
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep")
    install_cmd = [
        argv for argv, _ in installs if argv[:3] == ["uv", "pip", "install"]
    ][0]
    assert Path(install_cmd[-1]) != source and Path(install_cmd[-1]).name == "source"
