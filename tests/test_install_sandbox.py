"""An install runs the package's build code, so it is sandboxed as a test is: each install takes a uid of its
own from the run users' pool, whose firewall lets it reach only the egress proxy, and the proxy admits only the
package index hosts for it (the stack's index and PyPI). Its home and uv cache are its own and gone when it ends,
and so is every process it left."""

import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, egress, plugin_process
from aisc_eval.plugin_process import PluginUser
from recorded_pins import recorded_pins

INDEX = "http://devpi:3141/root/public/+simple/"
A = "a" * 64


def test_the_index_hosts_are_all_an_install_may_reach():
    allowed = egress.index_egress(INDEX)
    assert allowed.hosts == frozenset({"devpi:3141", "pypi.org:443", "files.pythonhosted.org:443"})
    # the stack's index is an internal address: allowed for this install, by its own rule
    assert allowed.rule["allowed_hosts"] == ["devpi:3141"]
    assert allowed.rule["denied_addresses"] == []


def test_without_an_index_of_its_own_an_install_reaches_pypi_only():
    assert egress.index_egress(None).hosts == frozenset({"pypi.org:443", "files.pythonhosted.org:443"})


def test_an_install_step_goes_through_the_proxy_it_is_given(monkeypatch):
    calls = []
    monkeypatch.setattr(
        plugin_process.subprocess,
        "run",
        lambda argv, **kw: calls.append(kw) or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    monkeypatch.setattr(plugin_process, "end_processes_of", lambda user: None)
    proxy = egress.proxy_env("t0ken")
    user = PluginUser(10150, 10150, "/tmp/install-home", switch=True, proxy=proxy)
    plugin_process.run_install(["uv", "pip", "install", "x"], user, timeout=5)
    env = calls[0]["env"]
    assert {name: env[name] for name in proxy} == proxy
    assert env["HOME"] == "/tmp/install-home" and env["UV_CACHE_DIR"] == "/tmp/install-home/.cache/uv"
    assert (calls[0]["user"], calls[0]["group"]) == (10150, 10150)


class _Ok:
    returncode = 0
    stdout = ""
    stderr = ""


@pytest.fixture
def root_worker(monkeypatch, tmp_path):
    """A root worker with its firewall and proxy, its pool faked: each lease is the test's own uid, recorded."""
    seen = {"leased": [], "given_back": [], "ended": [], "steps": []}
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 0)
    monkeypatch.setattr(plugin_process.os, "chown", lambda *a: None)
    monkeypatch.setattr(plugin_process.os, "lchown", lambda *a: None)
    monkeypatch.setattr(egress, "in_force", lambda: True)
    monkeypatch.setattr(egress, "proxy_answers", lambda: True)
    monkeypatch.setattr(egress.grp, "getgrnam", lambda name: type("G", (), {"gr_gid": os.getgid()}))
    policies = tmp_path / "policies"
    policies.mkdir()
    monkeypatch.setattr(egress, "POLICIES", policies)
    monkeypatch.setattr(celery_tasks, "_envs_root", lambda: tmp_path / "envs")
    (tmp_path / "envs").mkdir()

    @contextmanager
    def lease():
        user = PluginUser(os.getuid(), os.getgid(), None, switch=True)
        seen["leased"].append(user.uid)
        try:
            yield user
        finally:
            seen["given_back"].append(user.uid)

    monkeypatch.setattr(plugin_process, "lease_run_user", lease)
    monkeypatch.setattr(
        plugin_process, "install_user", lambda: pytest.fail("installed as the image's shared install user")
    )
    monkeypatch.setattr(plugin_process, "end_processes_of", lambda user: seen["ended"].append(user.uid))

    def run_install(argv, user, timeout):
        token = user.proxy["HTTP_PROXY"].split(":")[2].split("@")[0]
        rule = policies / (hashlib.sha256(token.encode()).hexdigest() + ".json")
        seen["steps"].append({
            "argv": argv,
            "user": user,
            "hosts": json.loads(rule.read_text())["hosts"] if rule.exists() else None,
            "home_exists": Path(user.home).is_dir(),
        })
        if argv[:3] == ["uv", "pip", "compile"]:
            Path(argv[argv.index("-o") + 1]).write_text(f"pkg==1.0 \\\n    --hash=sha256:{A}\n")
        return _Ok()

    monkeypatch.setattr(plugin_process, "run_install", run_install)

    class _Loader:
        discovered_packages = {"pkg": {"1.0": {"source": "registry"}}}

        def list_packages(self, refresh=False):
            return self.discovered_packages

        class devpi_client:
            simple_index_url = INDEX

    monkeypatch.setattr(celery_tasks, "plugin_loader", _Loader())
    seen["policies"] = policies
    return seen


def _warm(seen):
    celery_tasks.install_package.run("pkg", "1.0", "eval", ["ep"], pins=recorded_pins("pkg", "1.0"))


def _kept(seen):
    with celery_tasks._plugin_env("pkg", "1.0", {"source": "registry"}, A):
        pass


@pytest.mark.parametrize("install", [_warm, _kept], ids=["cache warm-up", "kept environment"])
def test_a_root_worker_installs_as_a_leased_user_that_reaches_only_the_index(root_worker, install):
    install(root_worker)
    steps = root_worker["steps"]
    assert steps, "nothing was installed"
    for step in steps:
        assert step["user"].uid in root_worker["leased"]
        assert sorted(step["hosts"]) == ["devpi:3141", "files.pythonhosted.org:443", "pypi.org:443"]
        # a home and uv cache of its own, not the shared install user's
        assert step["home_exists"] and not step["user"].home.startswith("/home/")
        assert step["user"].proxy["NO_PROXY"] == ""
    # once it ended: no rule left, its processes ended before its uid went back, its home gone
    assert list(root_worker["policies"].iterdir()) == []
    assert root_worker["ended"] and root_worker["given_back"] == root_worker["leased"]
    assert not Path(steps[0]["user"].home).exists()


def test_one_lease_covers_every_step_of_one_install(root_worker):
    _kept(root_worker)
    assert len(root_worker["leased"]) == 1
    assert {step["user"].uid for step in root_worker["steps"]} == set(root_worker["leased"])


def test_a_root_worker_without_its_firewall_installs_nothing(root_worker, monkeypatch):
    monkeypatch.setattr(egress, "in_force", lambda: False)
    with pytest.raises(RuntimeError, match="isolation is not in force"):
        _kept(root_worker)
    assert root_worker["steps"] == [] and root_worker["leased"] == []


def test_a_worker_that_is_not_root_installs_as_itself_without_a_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin_process.os, "geteuid", lambda: 4242)
    monkeypatch.setattr(plugin_process.os, "getegid", lambda: 4242)
    with celery_tasks._installer(INDEX) as user:
        assert (user.uid, user.switch, user.proxy) == (4242, False, None)


def test_the_proxy_token_of_an_install_is_kept_out_of_its_failure(monkeypatch):
    monkeypatch.setattr(plugin_process, "end_processes_of", lambda user: None)
    proxy = egress.proxy_env("s3cret-t0ken")

    def failed(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, "", f"error: proxy {proxy['HTTP_PROXY']} refused")

    monkeypatch.setattr(plugin_process.subprocess, "run", failed)
    user = PluginUser(10150, 10150, "/tmp/h", switch=True, proxy=proxy)
    done = plugin_process.run_install(["uv", "venv"], user, timeout=5)
    assert "s3cret-t0ken" not in done.stderr
