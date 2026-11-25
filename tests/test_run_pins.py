"""Every run installs its plugin from the hashes recorded for it: the wheel's sha256 and a hashed dependency
lock, with no other resolution. A registry run without them is refused before anything is installed. The
environment the run ran in is kept as a file of the run, and a local plugin's code digest, commit and dirty
flag are reported when it starts."""

from pathlib import Path

import pytest

from aisc_eval import celery_tasks, plugin_process
from test_plugin_process import _Ok, worker  # noqa: F401  (fixtures)

HASH = "a" * 64
LOCK = f"pkg==1.0 \\\n    --hash=sha256:{HASH}\n"
FREEZE = "pkg==1.0\npydantic==2.9.2\n"
CODE = {"digest": "d" * 64, "git_commit": "c" * 40, "git_dirty": True}


@pytest.fixture
def installs(worker, monkeypatch):  # noqa: F811
    """Each install step's argv; `uv pip freeze` answers FREEZE. The lock an install reads is kept as it was
    then (worker["locks"]): the run's folder is gone once the run has ended."""
    seen = []
    worker["locks"] = []

    def run(argv, user, timeout):
        seen.append(list(argv))
        if "-r" in argv:
            worker["locks"].append(Path(argv[argv.index("-r") + 1]).read_text())
        done = _Ok()
        if argv[:3] == ["uv", "pip", "freeze"]:
            done.stdout = FREEZE
        return done

    monkeypatch.setattr(plugin_process, "run_install", run)
    worker["installs"] = seen
    return seen


def run_with(pins=None, **kw):
    try:
        return celery_tasks.run_plugin.run(
            "pkg", "Plugin", "1.0", {}, [], [], "eval", "ep", pins=pins, **kw
        )
    except TypeError as exc:
        pytest.fail(f"a run takes no recorded hashes: {exc}")


def test_a_registry_run_installs_only_from_its_recorded_hashes(worker, installs):  # noqa: F811
    run_with({"wheel_sha256": HASH, "dependency_lock": LOCK})
    [install] = [argv for argv in installs if argv[:3] == ["uv", "pip", "install"]]
    assert "--require-hashes" in install and "--no-deps" in install and "-r" in install
    assert worker["locks"] == [LOCK]


def test_a_registry_run_without_pins_is_refused_before_install(worker, installs):  # noqa: F811
    with pytest.raises(Exception, match="no recorded hash"):
        run_with(None)
    assert not [argv for argv in installs if argv[:3] == ["uv", "pip", "install"]]
    assert worker["popen"] == []


def test_the_runs_environment_is_kept_as_a_file(worker, installs):  # noqa: F811
    run_with({"wheel_sha256": HASH, "dependency_lock": LOCK})
    assert worker["uploads"].get("plugin-environment.txt") == FREEZE.encode()


def test_a_local_run_reports_its_digest_commit_and_dirty_flag(
    worker, installs, monkeypatch, tmp_path
):  # noqa: F811
    source = tmp_path / "pkg"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "1.0"\n')
    monkeypatch.setattr(
        celery_tasks.plugin_loader,
        "discovered_packages",
        {"pkg": {"1.0": {"source": "local", "pkg_root": source, **CODE}}},
        raising=False,
    )
    started = []
    monkeypatch.setattr(
        celery_tasks, "mark_plugin_started", lambda *a, **k: started.append((a, k))
    )
    run_with(None)
    assert started, "the run never reported its start"
    args, kwargs = started[0]
    assert kwargs.get("code", args[2] if len(args) > 2 else None) == CODE, started


def test_a_lock_that_does_not_name_the_recorded_wheel_is_refused_before_install(
    worker, installs
):  # noqa: F811
    with pytest.raises(Exception, match="does not name its recorded wheel"):
        run_with({"wheel_sha256": "b" * 64, "dependency_lock": LOCK})
    assert not [argv for argv in installs if argv[:3] == ["uv", "pip", "install"]]


def test_a_file_whose_hash_is_not_the_recorded_one_is_refused_naming_why(
    monkeypatch, tmp_path
):
    class Refused(_Ok):
        returncode = 1
        stderr = "error: Hash mismatch for `pkg==1.0`"

    monkeypatch.setattr(
        plugin_process,
        "run_install",
        lambda argv, user, timeout: Refused()
        if argv[:3] == ["uv", "pip", "install"]
        else _Ok(),
    )
    me = plugin_process.PluginUser(0, 0, None, switch=False)
    with pytest.raises(plugin_process.InstallFailed, match="is not the one recorded"):
        plugin_process.build_venv(
            "pkg",
            "1.0",
            {"source": "registry"},
            tmp_path,
            me,
            None,
            pins=plugin_process.Pins(HASH, LOCK),
        )


def test_a_plugin_cannot_leave_a_file_named_as_the_runs_environment(worker, installs):  # noqa: F811
    def leave(output):
        (output / "measures.json").write_text('[{"name": "m", "value": 1}]')
        (output / "plugin-environment.txt").write_text("forged")

    worker["leave"] = leave
    with pytest.raises(Exception, match="name of a file the worker keeps"):
        run_with({"wheel_sha256": HASH, "dependency_lock": LOCK})
