"""The worker reads a plugin's manifest and runs its form hooks for the engine, so the engine never imports
plugin code. The plugin is installed from a hashed lock that must name the wheel hash recorded for it; a
wheel whose hash differs, or a registry plugin with no hash at all, is refused before any of its code runs.
Reading a manifest and a hook each have a time limit."""

import subprocess
from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process
from aisc_eval.plugin_process import PluginUser

A, B = "a" * 64, "b" * 64
INSTALLER = PluginUser(4242, 4242, "/home/aisc-install", switch=False)


def need(module, name):
    found = getattr(module, name, None)
    assert found is not None, f"{module.__name__} has no {name}: the engine still imports plugin code"
    return found


def lock_listing(digest):
    return (f"demo-pkg==1.0 \\\n    --hash=sha256:{digest}\n"
            f"pydantic==2.9.2 \\\n    --hash=sha256:{'d' * 64}\n")


@pytest.fixture
def installs(monkeypatch):
    """run_install faked: each argv recorded; `uv pip compile` writes the lock it is told to."""
    seen = {"argv": [], "lock": lock_listing(A)}

    def run(argv, user, timeout):
        seen["argv"].append(argv)
        if argv[:3] == ["uv", "pip", "compile"] and "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_text(seen["lock"])
        return subprocess.CompletedProcess(argv, 0, stdout=seen["lock"] if "-o" not in argv else "", stderr="")

    monkeypatch.setattr(plugin_process, "run_install", run)
    return seen


def test_the_lock_is_compiled_with_hashes_as_the_install_user(installs, tmp_path):
    lock = need(plugin_process, "lock_dependencies")
    text = lock("demo-pkg", "1.0", {"source": "registry"}, A, tmp_path, INSTALLER, "http://devpi/root/public/+simple/")
    [compile_argv] = [a for a in installs["argv"] if a[:3] == ["uv", "pip", "compile"]]
    assert "--generate-hashes" in compile_argv
    assert f"--hash=sha256:{A}" in text


def test_a_wheel_whose_hash_differs_from_the_recorded_one_is_refused(installs, tmp_path):
    lock = need(plugin_process, "lock_dependencies")
    installs["lock"] = lock_listing(B)
    with pytest.raises(plugin_process.InstallFailed, match="not the one recorded"):
        lock("demo-pkg", "1.0", {"source": "registry"}, A, tmp_path, INSTALLER, None)


def test_a_pinned_install_takes_only_hashed_packages(installs, tmp_path):
    pins = need(plugin_process, "Pins")(wheel_sha256=A, lock=lock_listing(A))
    try:
        plugin_process.build_venv("demo-pkg", "1.0", {"source": "registry"}, tmp_path, INSTALLER, None, pins=pins)
    except TypeError as exc:
        pytest.fail(f"an environment cannot be built from pins: {exc}")
    [install] = [a for a in installs["argv"] if a[:3] == ["uv", "pip", "install"]]
    assert "--require-hashes" in install and "--no-deps" in install


def test_the_manifest_and_the_form_hooks_are_worker_tasks_with_time_limits():
    need(celery_tasks, "read_manifest")
    need(celery_tasks, "plugin_hook")
    assert need(celery_tasks, "MANIFEST_TIMEOUT_S") == 120
    assert need(celery_tasks, "HOOK_TIMEOUT_S") == 30


def test_a_registry_plugin_with_no_hash_anywhere_is_refused(installs, monkeypatch):
    read_manifest = need(celery_tasks, "read_manifest")
    monkeypatch.setattr(celery_tasks, "plugin_loader", type("L", (), {
        "list_packages": lambda self, refresh=False: {"demo-pkg": {"1.0": {"source": "registry"}}},
        "devpi_client": type("D", (), {"wheel_sha256": lambda self, p, v: None})(),
    })())
    with pytest.raises(Exception, match="hash"):
        read_manifest.run("demo-pkg", "1.0", None)
    assert not [a for a in installs["argv"] if a[:3] == ["uv", "pip", "install"]]


ME = PluginUser(0, 0, None, switch=False)


def answer_of(code, tmp_path, timeout=5.0, cap=1024):
    import sys

    return plugin_process.run_for_answer([sys.executable, "-c", code], tmp_path, {"PATH": "/usr/bin:/bin"}, ME,
                                         b"", timeout, cap)


def test_a_command_that_does_not_answer_in_time_is_killed(tmp_path):
    with pytest.raises(plugin_process.NoAnswer, match="within 1 seconds"):
        answer_of("import time; time.sleep(30)", tmp_path, timeout=1)


def test_an_answer_over_its_cap_is_refused(tmp_path):
    with pytest.raises(plugin_process.NoAnswer, match="larger than"):
        answer_of("print('x' * 5000)", tmp_path)


def test_a_failed_command_gives_the_last_line_of_its_reason(tmp_path):
    with pytest.raises(plugin_process.NoAnswer, match="ValueError: no class here"):
        answer_of("import sys; sys.stderr.write('noise\\nValueError: no class here\\n'); sys.exit(1)", tmp_path)


def test_a_command_gets_its_stdin_and_answers_on_stdout(tmp_path):
    import sys

    got = plugin_process.run_for_answer([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                                        tmp_path, {"PATH": "/usr/bin:/bin"}, ME, b"hook", 5, 1024)
    assert got.strip() == b"HOOK"


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """The listing, the environment and the command faked: what the command was given is recorded."""
    seen = {"builds": 0, "calls": []}
    monkeypatch.setattr(celery_tasks, "plugin_loader", type("L", (), {
        "list_packages": lambda self, refresh=False: {"demo-pkg": {"1.0": {"source": "registry",
                                                                             "module_name": "demo_pkg"}}},
        "devpi_client": type("D", (), {"wheel_sha256": lambda self, p, v: A, "simple_index_url": None})(),
    })())
    monkeypatch.setattr(celery_tasks, "_envs_root", lambda: tmp_path / "envs")
    (tmp_path / "envs").mkdir()

    def lock(package_name, version, info, expected, folder, installer, index_url):
        return lock_listing(expected)

    def build(package_name, version, info, folder, installer, index_url, pins=None):
        seen["builds"] += 1
        (folder / "venv" / "bin").mkdir(parents=True)
        (folder / plugin_process.LOCK_FILE).write_text(pins.lock)

    def run(argv, workspace, env, user, stdin, timeout_s, max_bytes, limits=None):
        seen["calls"].append({"argv": argv, "env": env, "stdin": stdin, "timeout": timeout_s})
        if argv[-1] == "hook":
            return b'[{}, {"type": "object"}, {}]'
        return b'{"plugins": {"DemoPlugin": {"hooks": []}}}'

    monkeypatch.setattr(plugin_process, "lock_dependencies", lock)
    monkeypatch.setattr(plugin_process, "build_venv", build)
    monkeypatch.setattr(plugin_process, "install_user", lambda: INSTALLER)
    monkeypatch.setattr(plugin_process, "run_for_answer", run)
    return seen


def test_the_manifest_is_read_with_no_network_pass_or_secret_of_the_worker(sandbox, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_KEY", "worker-only")
    got = celery_tasks.read_manifest.run("demo-pkg", "1.0", None)
    assert (got["wheel_sha256"], got["code"]) == (A, None)
    assert "DemoPlugin" in got["manifest"]["plugins"] and f"--hash=sha256:{A}" in got["dependency_lock"]
    [call] = sandbox["calls"]
    assert call["argv"][-2:] == ["describe", "demo_pkg"] and call["timeout"] == 120
    env = call["env"]
    assert not {"HTTP_PROXY", "HTTPS_PROXY", "PLATFORM_RUN_PASS", "INTERNAL_API_KEY"} & set(env)


def test_a_hook_reuses_the_environment_built_for_its_hash(sandbox):
    celery_tasks.read_manifest.run("demo-pkg", "1.0", A)
    got = celery_tasks.plugin_hook.run("demo-pkg", "1.0", "DemoPlugin", "on_config_change", {"config": {}},
                                       {"wheel_sha256": A})
    assert got == [{}, {"type": "object"}, {}]
    assert sandbox["builds"] == 1
    assert sandbox["calls"][-1]["timeout"] == 30


def test_a_hook_other_than_the_form_hooks_is_refused_before_anything_runs(sandbox):
    with pytest.raises(ValueError, match="evaluate"):
        celery_tasks.plugin_hook.run("demo-pkg", "1.0", "DemoPlugin", "evaluate", {}, {"wheel_sha256": A})
    assert sandbox["calls"] == [] and sandbox["builds"] == 0


def test_a_manifest_that_is_not_json_is_refused(sandbox, monkeypatch):
    monkeypatch.setattr(plugin_process, "run_for_answer", lambda *a, **k: b"not json")
    with pytest.raises(plugin_process.NoAnswer, match="not JSON"):
        celery_tasks.read_manifest.run("demo-pkg", "1.0", A)
