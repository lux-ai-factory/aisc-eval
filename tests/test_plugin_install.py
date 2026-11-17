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
from recorded_pins import recorded_pins

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
    monkeypatch.setattr(plugin_process, "end_processes_of", lambda user: None)
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
    # what it installed is then given to root, which only a root worker can do
    monkeypatch.setattr(plugin_process.os, "lchown", lambda *a: None)

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
    celery_tasks.install_package.run("pkg", "1.0", "eval", ["ep"], pins=recorded_pins("pkg", "1.0"))
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
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep", pins=recorded_pins("pkg", "1.0"))
    venv_cmd, install_cmd, freeze_cmd = (argv for argv, _ in installs)
    assert venv_cmd[:2] == ["uv", "venv"] and "--allow-existing" in venv_cmd
    assert install_cmd[:3] == ["uv", "pip", "install"]
    # the run's environment is read by the install user too, to be kept as a file of the run
    assert freeze_cmd[:3] == ["uv", "pip", "freeze"]
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


# One way to build a plugin's environment, for the install that warms the cache and for each run


def _recorded_installs(monkeypatch):
    seen = []

    def run_install(argv, user, timeout):
        seen.append((argv, timeout))
        return _Ok()

    monkeypatch.setattr(plugin_process, "run_install", run_install)
    return seen


def test_a_venv_is_made_then_filled_from_the_index(monkeypatch, tmp_path):
    seen = _recorded_installs(monkeypatch)
    me = PluginUser(os.getuid(), os.getgid(), None, switch=False)
    venv = plugin_process.build_venv(
        "pkg", "1.0", {"source": "registry"}, tmp_path, me, "http://devpi/simple"
    )
    assert (
        venv.path == tmp_path / "venv"
        and venv.python == tmp_path / "venv" / "bin" / "python"
    )
    (made, made_timeout), (filled, filled_timeout) = seen
    assert made == [
        "uv",
        "venv",
        str(tmp_path / "venv"),
        "--allow-existing",
        "--python",
        "/usr/local/bin/python",
    ]
    assert filled == [
        "uv",
        "pip",
        "install",
        "--python",
        str(tmp_path / "venv"),
        # a catalogue plugin installs from wheels only
        "--only-binary",
        ":all:",
        "--extra-index-url",
        "http://devpi/simple",
        "pkg==1.0",
    ]
    # a first-time Python download can be slow: neither step is cut short at a minute
    assert made_timeout == filled_timeout == plugin_process.INSTALL_TIMEOUT


def test_a_failed_step_is_refused_with_its_message_scrubbed(monkeypatch, tmp_path):
    class _Failed(_Ok):
        returncode = 1
        stderr = "error: api_key=abcd1234efgh refused"

    monkeypatch.setattr(plugin_process, "run_install", lambda *a, **kw: _Failed())
    me = PluginUser(os.getuid(), os.getgid(), None, switch=False)
    with pytest.raises(plugin_process.InstallFailed) as refused:
        plugin_process.build_venv(
            "pkg", "1.0", {"source": "registry"}, tmp_path, me, None
        )
    assert str(refused.value) == "error: api_key=[redacted] refused"
    assert refused.value.step == "venv"


def test_both_tasks_build_the_venv_the_same_way(installs, monkeypatch):
    seen = _recorded_installs(monkeypatch)
    celery_tasks.install_package.run("pkg", "1.0", "eval", ["ep"], pins=recorded_pins("pkg", "1.0"))
    warmed = list(seen)
    seen.clear()
    monkeypatch.setattr(
        celery_tasks, "get_evaluation_request", lambda pid: {"status": "Running"}
    )
    monkeypatch.setattr(
        celery_tasks,
        "mark_plugin_started",
        lambda *a: (_ for _ in ()).throw(RuntimeError("stop before the plugin starts")),
    )
    monkeypatch.setattr(celery_tasks, "mark_plugin_failed", lambda *a: None)
    monkeypatch.setattr(celery_tasks, "_revoke_sibling_tasks", lambda *a: None)
    with pytest.raises(RuntimeError):
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep", pins=recorded_pins("pkg", "1.0"))

    def shape(calls):
        return [(argv[:2], timeout) for argv, timeout in calls]

    # the run then reads its environment, which the warm-up does not keep
    assert shape(warmed) == shape(seen[:2]) and len(seen) == 3
    assert seen[2][0][:3] == ["uv", "pip", "freeze"]


class _LoaderThatLearns:
    """A worker that listed the index before a new version was published: the index has it now. Like the real
    loader (aisc_plugin_manager.loader.Loader.list_packages), it keeps its list unless asked to refresh it."""

    def __init__(self):
        self.discovered_packages = {"pkg": {"1.0": {"source": "registry"}}}
        self.listed = 0

    def list_packages(self, refresh=False):
        if refresh or not self.discovered_packages:
            self.listed += 1
            self.discovered_packages = {
                "pkg": {"1.0": {"source": "registry"}, "1.1": {"source": "registry"}}
            }
        return self.discovered_packages

    class devpi_client:
        simple_index_url = None


def test_a_version_published_after_the_worker_listed_the_index_is_found(
    installs, monkeypatch
):
    loader = _LoaderThatLearns()
    monkeypatch.setattr(celery_tasks, "plugin_loader", loader)
    celery_tasks.install_package.run("pkg", "1.1", "eval", ["ep"], pins=recorded_pins("pkg", "1.1"))
    assert loader.listed == 1
    assert [argv[:3] for argv, _ in installs][-1] == ["uv", "pip", "install"]


def test_a_version_the_index_does_not_have_is_still_refused(installs, monkeypatch):
    loader = _LoaderThatLearns()
    monkeypatch.setattr(celery_tasks, "plugin_loader", loader)
    stops = []
    monkeypatch.setattr(celery_tasks, "_stop_run", lambda *a: stops.append(a))
    with pytest.raises(KeyError, match="Version '9.9'"):
        celery_tasks.install_package.run("pkg", "9.9", "eval", ["ep"], pins=recorded_pins("pkg", "9.9"))
    assert loader.listed == 1
    # the refusal fails the run's tests with its reason, as any failed install does
    assert [(evaluation, tests, reason) for evaluation, tests, reason, _task in stops] == [
        ("eval", ["ep"], "Version '9.9' of package 'pkg' not found.")
    ]


# What an install step leaves running (a build backend may start a daemon) is ended with the step


def test_every_process_an_install_step_left_is_ended_when_the_step_ends(monkeypatch):
    events = []
    monkeypatch.setattr(
        plugin_process.subprocess, "run", lambda argv, **kw: events.append("step") or _Ok()
    )
    monkeypatch.setattr(
        plugin_process, "end_processes_of", lambda user: events.append(("ended", user))
    )
    plugin_process.run_install(["uv", "pip", "install", "x"], INSTALLER, timeout=5)
    assert events == ["step", ("ended", INSTALLER)]


def test_a_step_cut_short_by_its_time_limit_still_has_its_processes_ended(monkeypatch):
    ended = []

    def too_slow(argv, **kw):
        raise plugin_process.subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(plugin_process.subprocess, "run", too_slow)
    monkeypatch.setattr(plugin_process, "end_processes_of", ended.append)
    with pytest.raises(plugin_process.subprocess.TimeoutExpired):
        plugin_process.run_install(["uv", "venv"], INSTALLER, timeout=5)
    assert ended == [INSTALLER]


# Before a test starts, what it runs is root's: the user who installed it can no longer change it


def _builds_files(monkeypatch):
    """run_install faked: each step leaves files in the venv as an installer would, some writable by others."""

    def run_install(argv, user, timeout):
        if argv[:2] == ["uv", "venv"]:
            venv = Path(argv[2])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "activate").write_text("")
            os.chmod(venv / "bin" / "activate", 0o777)
            (venv / "bin" / "python").symlink_to("/usr/local/bin/python")
        elif argv[:3] == ["uv", "pip", "install"]:
            site = Path(argv[argv.index("--python") + 1]) / "lib" / "site-packages" / "pkg"
            site.mkdir(parents=True, exist_ok=True)
            os.chmod(site, 0o777)
            (site / "__init__.py").write_text("")
            os.chmod(site / "__init__.py", 0o666)
        return _Ok()

    monkeypatch.setattr(plugin_process, "run_install", run_install)
    owners = {}
    monkeypatch.setattr(
        plugin_process.os, "lchown", lambda path, uid, gid: owners.__setitem__(Path(path), (uid, gid))
    )
    monkeypatch.setattr(plugin_process.os, "chown", lambda path, uid, gid: None)
    return owners


def _assert_sealed(folder: Path, owners: dict):
    paths = [folder, *folder.rglob("*")]
    assert paths and all(owners.get(path) == (0, 0) for path in paths), "not every file was given to root"
    for path in paths:
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_dir():
            assert mode == 0o755, path
        else:
            assert mode in (0o644, 0o755), (path, oct(mode))


def test_a_built_venv_is_roots_and_closed_to_its_installer(monkeypatch, tmp_path):
    owners = _builds_files(monkeypatch)
    me = PluginUser(os.getuid(), os.getgid(), None, switch=True)
    venv = plugin_process.build_venv(
        "pkg", "1.0", {"source": "registry"}, tmp_path, me, None,
        pins=plugin_process.Pins("a" * 64, "pkg==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"),
    )
    _assert_sealed(venv.path, owners)
    # what could run keeps its execute bit, nothing else gains one
    assert stat.S_IMODE((venv.path / "bin" / "activate").stat().st_mode) == 0o755
    init = venv.path / "lib" / "site-packages" / "pkg" / "__init__.py"
    assert stat.S_IMODE(init.stat().st_mode) == 0o644


def test_a_local_plugins_copy_is_roots_too(monkeypatch, tmp_path):
    owners = _builds_files(monkeypatch)
    source = tmp_path / "pkg"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
    me = PluginUser(os.getuid(), os.getgid(), None, switch=True)
    (tmp_path / "run").mkdir()
    plugin_process.build_venv(
        "pkg", "1.0", {"source": "local", "pkg_root": source}, tmp_path / "run", me, None
    )
    _assert_sealed(tmp_path / "run" / "source", owners)


def test_a_kept_environment_is_roots_once_it_is_built(monkeypatch, tmp_path):
    """The environment a manifest and the form hooks run in is kept for every later call: its lock, venv and
    copy are root's before it is marked ready."""
    owners = _builds_files(monkeypatch)
    real_run = plugin_process.run_install

    def run_install(argv, user, timeout):
        if argv[:3] == ["uv", "pip", "compile"]:
            Path(argv[argv.index("-o") + 1]).write_text("pkg==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n")
            return _Ok()
        return real_run(argv, user, timeout)

    monkeypatch.setattr(plugin_process, "run_install", run_install)
    monkeypatch.setattr(
        plugin_process, "install_user", lambda: PluginUser(os.getuid(), os.getgid(), None, switch=True)
    )
    monkeypatch.setattr(celery_tasks, "_envs_root", lambda: tmp_path)
    monkeypatch.setattr(celery_tasks, "plugin_loader", _Loader())
    with celery_tasks._plugin_env("pkg", "1.0", {"source": "registry"}, "a" * 64):
        pass
    folder = tmp_path / ("a" * 64)
    for part in ("venv", "lock"):
        _assert_sealed(folder / part, owners)
