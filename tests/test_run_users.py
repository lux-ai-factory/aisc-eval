"""Each plugin test runs as a user of its own, leased from a pool, in folders only that user (and root) can
read: two tests at once never share a uid, a home, a tmp or a readable input. A lease is released however
the test ends, also when the process that held it died. A full pool refuses with a reason."""

import contextlib
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.plugin_process import PluginUser
from test_plugin_process import _run, worker  # noqa: F401  (fixtures)
from recorded_pins import recorded_pins


def need(name):
    found = getattr(plugin_process, name, None)
    assert (
        found is not None
    ), f"plugin_process has no {name}: runs share one plugin user"
    return found


@pytest.fixture
def as_root(monkeypatch):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)


def test_the_pool_is_a_hundred_uids_above_the_image_users():
    assert need("RUN_UIDS") == range(10100, 10200)


def test_a_worker_takes_its_pool_from_its_setting():
    """Two worker containers must never lease the same uid: the kernel counts a uid's processes across
    containers, and a shared volume would show one owner for two runs. Each names its own pool."""
    pool = need("run_uid_pool")
    assert pool(None) == pool("") == range(10100, 10200)
    assert pool("10200-10209") == range(10200, 10210)
    assert pool(" 10200 - 10200 ") == range(10200, 10201)


@pytest.mark.parametrize("value", ["10209-10200", "10000-10099", "10003-10010", "10200", "a-b", "10200-"])
def test_a_pool_that_is_reversed_reaches_the_image_users_or_is_not_a_range_is_refused(value):
    with pytest.raises(ValueError, match=plugin_process.RUN_UIDS_SETTING):
        need("run_uid_pool")(value)


def test_the_firewall_covers_the_pool_the_setting_names():
    code = ("from aisc_eval import egress, plugin_process; "
            "print(plugin_process.RUN_UIDS.start, plugin_process.RUN_UIDS.stop, "
            "*{rule[rule.index('--uid-owner') + 1] for rule in egress.firewall_rules()})")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PLUGIN_RUN_UIDS": "10200-10209"})
    assert done.stdout.split() == ["10200", "10210", "10200-10209"], done.stderr


def test_two_leases_get_two_different_users(tmp_path, as_root):
    lease = need("lease_run_user")
    with (
        lease(range(10100, 10102), tmp_path) as a,
        lease(range(10100, 10102), tmp_path) as b,
    ):
        assert {a.uid, b.uid} == {10100, 10101}
        assert a.gid == a.uid and a.switch


def test_a_full_pool_refuses_with_a_reason(tmp_path, as_root):
    lease = need("lease_run_user")
    busy = need("NoRunUserFree")
    with lease(range(10100, 10101), tmp_path):
        with pytest.raises(busy, match="busy"):
            with lease(range(10100, 10101), tmp_path):
                pass


def test_a_released_lease_is_free_again(tmp_path, as_root):
    lease = need("lease_run_user")
    with lease(range(10100, 10101), tmp_path) as first:
        pass
    with lease(range(10100, 10101), tmp_path) as second:
        assert second.uid == first.uid


def test_a_lease_held_by_a_process_that_died_is_free_again(tmp_path, as_root):
    lease = need("lease_run_user")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            f"f = open({str(tmp_path / '10100.lock')!r}, 'a')\n"
            "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        holder.send_signal(signal.SIGKILL)
        holder.wait(10)
        with lease(range(10100, 10101), tmp_path) as user:
            assert user.uid == 10100
    finally:
        with contextlib.suppress(ProcessLookupError):
            holder.kill()


def test_a_worker_that_is_not_root_runs_the_plugin_as_itself(tmp_path, monkeypatch):
    lease = need("lease_run_user")
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 4242)
    with lease(range(10100, 10102), tmp_path) as user:
        assert not user.switch


def test_run_folders_are_private_to_the_runs_user(tmp_path, monkeypatch):
    make = need("make_run_folders")
    chowns = []
    monkeypatch.setattr(
        plugin_process.os, "chown", lambda p, u, g: chowns.append((Path(p).name, u, g))
    )
    f = make(tmp_path / "run", PluginUser(10150, 10150, None, switch=True))
    modes = {
        p.name: stat.S_IMODE(p.stat().st_mode)
        for p in (f.root, f.work, f.input, f.output, f.home, f.tmp)
    }
    assert modes == {
        "run": 0o711,
        "work": 0o750,
        "input": 0o750,
        "output": 0o1770,
        "home": 0o700,
        "tmp": 0o700,
    }
    assert ("home", 10150, 10150) in chowns and ("tmp", 10150, 10150) in chowns
    assert ("work", -1, 10150) in chowns and ("output", -1, 10150) in chowns


def test_a_file_the_worker_writes_for_the_run_is_unreadable_to_others(
    tmp_path, monkeypatch
):
    write = need("write_for_run")
    chowns = []
    monkeypatch.setattr(
        plugin_process.os, "chown", lambda p, u, g: chowns.append((Path(p).name, u, g))
    )
    target = tmp_path / "config.json"
    write(target, b"{}", PluginUser(10150, 10150, None, switch=True))
    assert target.read_bytes() == b"{}"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert ("config.json", 0, 10150) in chowns


def test_a_run_user_has_a_home_a_name_and_a_tmp_of_its_own(tmp_path, monkeypatch):
    make = need("make_run_folders")
    monkeypatch.setattr(plugin_process.os, "chown", lambda *a: None)
    user = PluginUser(10150, 10150, None, switch=True)
    folders = make(tmp_path / "run", user)
    started = []
    monkeypatch.setattr(
        plugin_process.subprocess, "Popen", lambda argv, **kw: started.append(kw)
    )
    with open(os.devnull, "w") as stderr:
        plugin_process.start(
            ["python"], folders.work, stderr, {"PATH": "/usr/bin"}, user
        )
    env = started[0]["env"]
    assert env.get("HOME") == str(folders.home)
    assert env.get("TMPDIR") == str(folders.tmp)
    assert env.get("USER") == env.get("LOGNAME") == "aisc-run-10150"


# the worker: each run takes its own lease and gives it back


@pytest.fixture
def leases(worker, monkeypatch):  # noqa: F811
    """A fake pool handing out 10100, 10101, ... and counting what was taken and given back."""
    seen = {"taken": [], "given_back": []}
    uids = iter(range(10100, 10200))

    @contextlib.contextmanager
    def lease(*a, **k):
        user = PluginUser(next(uids), os.getgid(), None, switch=True)
        seen["taken"].append(user.uid)
        try:
            yield user
        finally:
            seen["given_back"].append(user.uid)

    monkeypatch.setattr(plugin_process, "lease_run_user", lease, raising=False)
    # the pool's uids are not this test's to give folders to, nor to own the files the fake plugin leaves
    monkeypatch.setattr(plugin_process.os, "chown", lambda *a: None)
    read_outputs = plugin_process.read_outputs
    monkeypatch.setattr(
        plugin_process,
        "read_outputs",
        lambda folder, user, **kw: read_outputs(
            folder, PluginUser(os.getuid(), os.getgid(), None, switch=True), **kw
        ),
    )
    seen["worker"] = worker
    return seen


def test_two_runs_start_as_two_different_users(leases):
    _run()
    _run()
    popen = leases["worker"]["popen"]
    assert (
        popen[0][1]["user"] != popen[1][1]["user"]
    ), "both runs started as the same user"


def test_the_lease_is_released_when_the_plugin_fails(leases, monkeypatch):
    def fail(argv, **kwargs):
        raise OSError("the plugin could not start")

    monkeypatch.setattr(plugin_process.subprocess, "Popen", fail)
    with contextlib.suppress(Exception):
        celery_tasks.run_plugin.run("pkg", "Plugin", "1.0", {}, [], [], "eval", "ep", pins=recorded_pins("pkg", "1.0"))
    assert leases["taken"], "the run took no lease"
    assert leases["given_back"] == leases["taken"]


def test_the_runs_processes_are_ended_before_its_user_is_given_back(
    leases, monkeypatch
):
    ended = []

    def end(user):
        assert (
            user.uid not in leases["given_back"]
        ), "the user was given back with its processes running"
        ended.append(user.uid)

    monkeypatch.setattr(plugin_process, "end_processes_of", end)
    _run()
    assert ended == leases["taken"] == leases["given_back"]


# what a leased user leaves behind


def test_a_released_run_user_has_no_process_left(monkeypatch):
    ran = []
    monkeypatch.setattr(
        plugin_process.subprocess, "run", lambda argv, **kw: ran.append(kw)
    )
    plugin_process.end_processes_of(PluginUser(10150, 10150, None, switch=True))
    [kwargs] = ran
    assert (kwargs["user"], kwargs["group"], kwargs["extra_groups"]) == (
        10150,
        10150,
        [],
    )
    killed = []
    monkeypatch.setattr(
        plugin_process.os, "kill", lambda pid, sig: killed.append((pid, sig))
    )
    kwargs["preexec_fn"]()
    assert killed == [(-1, signal.SIGKILL)]


def test_a_worker_that_is_not_root_never_kills_its_own_processes(monkeypatch):
    monkeypatch.setattr(
        plugin_process.subprocess,
        "run",
        lambda *a, **k: pytest.fail("ended the worker's own processes"),
    )
    plugin_process.end_processes_of(
        PluginUser(os.getuid(), os.getgid(), None, switch=False)
    )
