"""A catalogue plugin and its dependencies install from wheels only, so no build code runs at install time:
building is the publisher's job. A local plugin keeps its source builds (it is the operator's own code). A
dependency with no wheel fails the install with a message that says so and names it."""

import os
import subprocess
from pathlib import Path

import pytest

from aisc_eval import plugin_process
from aisc_eval.plugin_process import PluginUser

A = "a" * 64
ME = PluginUser(os.getuid(), os.getgid(), None, switch=False)
LOCK = f"pkg==1.0 \\\n    --hash=sha256:{A}\n"


@pytest.fixture
def steps(monkeypatch):
    seen = {"argv": [], "fail": None}

    def run(argv, user, timeout):
        seen["argv"].append(argv)
        if seen["fail"] and argv[:3] == seen["fail"][0]:
            return subprocess.CompletedProcess(argv, 1, "", seen["fail"][1])
        if argv[:3] == ["uv", "pip", "compile"]:
            Path(argv[argv.index("-o") + 1]).write_text(LOCK)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(plugin_process, "run_install", run)
    return seen


def _wheels_only(argv):
    i = argv.index("--only-binary") if "--only-binary" in argv else -1
    return i >= 0 and argv[i + 1] == ":all:"


def _local(tmp_path):
    source = tmp_path / "pkg"
    source.mkdir(exist_ok=True)
    (source / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
    return {"source": "local", "pkg_root": source}


def test_a_catalogue_plugin_is_locked_and_installed_from_wheels_only(steps, tmp_path):
    lock = plugin_process.lock_dependencies(
        "pkg", "1.0", {"source": "registry"}, A, tmp_path, ME, None
    )
    plugin_process.build_venv(
        "pkg",
        "1.0",
        {"source": "registry"},
        tmp_path,
        ME,
        None,
        pins=plugin_process.Pins(A, lock),
    )
    (tmp_path / "unpinned").mkdir()
    plugin_process.build_venv(
        "pkg", "1.0", {"source": "registry"}, tmp_path / "unpinned", ME, None
    )
    resolving = [
        a
        for a in steps["argv"]
        if a[:3] in (["uv", "pip", "compile"], ["uv", "pip", "install"])
    ]
    assert len(resolving) == 3 and all(_wheels_only(a) for a in resolving)


def test_a_local_plugin_keeps_its_source_builds(steps, tmp_path):
    info = _local(tmp_path)
    lock = plugin_process.lock_dependencies(
        "pkg", "1.0", info, None, tmp_path, ME, None
    )
    (tmp_path / "run").mkdir()
    plugin_process.build_venv(
        "pkg",
        "1.0",
        info,
        tmp_path / "run",
        ME,
        None,
        pins=plugin_process.Pins(None, lock),
    )
    assert not any("--only-binary" in a for a in steps["argv"])


def test_a_dependency_with_no_wheel_fails_naming_it(steps, tmp_path):
    steps["fail"] = (
        ["uv", "pip", "compile"],
        "  x No solution found when resolving dependencies:\n  Because oldlib==0.1 has no usable "
        "wheels and building from source is disabled, we can conclude that pkg==1.0 cannot be used.",
    )
    with pytest.raises(plugin_process.InstallFailed) as failed:
        plugin_process.lock_dependencies(
            "pkg", "1.0", {"source": "registry"}, A, tmp_path, ME, None
        )
    text = str(failed.value)
    assert "wheels only" in text and "oldlib==0.1" in text


def test_uvs_own_wrapped_message_is_recognised(steps, tmp_path):
    """uv wraps its message at the terminal width: the words that say a wheel is missing may span lines."""
    steps["fail"] = (
        ["uv", "pip", "compile"],
        "  x No solution found when resolving dependencies:\n  -> Because aisc-bed-oldlib==0.1 has no "
        "usable wheels and needs-oldlib==1.0\n      depends on aisc-bed-oldlib==0.1, we can conclude that "
        "needs-oldlib==1.0\n      cannot be used.\n\n      hint: Wheels are required for `aisc-bed-oldlib` "
        "because building from\n      source is disabled for all packages (i.e., with `--no-build`)\n",
    )
    with pytest.raises(plugin_process.InstallFailed) as failed:
        plugin_process.lock_dependencies(
            "needs-oldlib", "1.0", {"source": "registry"}, A, tmp_path, ME, None
        )
    assert str(failed.value).startswith("a catalogue plugin installs from wheels only")
    assert "aisc-bed-oldlib" in str(failed.value)
