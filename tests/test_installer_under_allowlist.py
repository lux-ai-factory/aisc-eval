"""ADR 0003 §4 — the installer is held to the same rule as the run.

The plugin child's environment was closed to a small allowlist, but the steps that
*install* a plugin (`uv venv`, `uv pip install`) are subprocesses too, and they used
to inherit the worker's environment: the database URL, the registry password, the
signing key. These tests pin both halves of that gate — that the install steps get
the allowlist, and that with nothing but the allowlist a real install and a real run
still work.

The loader/registry is faked (it stands in for a devpi index a test must not need);
the wheel, the virtualenv, the install and the run are real.
"""
import ast
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from aisc_eval import celery_tasks
from aisc_eval.celery_tasks import plugin_child_env

WORKER_ONLY = ("DJANGO_SECRET_KEY", "DATABASE_URL", "PACKAGE_REGISTRY_PASSWORD")
SPAWNERS = {"run", "Popen", "call", "check_call", "check_output"}


def _plugin_wheel(directory: Path, name: str = "smoke_plugin", answer: str = "the plugin ran") -> Path:
    """A real, dependency-free wheel, written by hand so no index is needed."""
    dist = f"{name}-1.0.dist-info"
    files = {
        f"{name}/__init__.py": f"ANSWER = {answer!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    records = [
        f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b'=').decode()},{len(text)}"
        for path, text in files.items()
    ]
    files[f"{dist}/RECORD"] = "\n".join(records) + f"\n{dist}/RECORD,,\n"
    wheel = directory / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel


def test_the_install_steps_get_the_allowlist_not_the_worker_environment(monkeypatch):
    """`uv venv` and `uv pip install` are children too: they get the allowlist."""
    for name in WORKER_ONLY:
        monkeypatch.setenv(name, "worker-only-value")
    monkeypatch.setenv("PATH", "/usr/bin")

    seen: list[dict] = []

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def record(cmd, **kwargs):
        seen.append(kwargs.get("env"))
        return _Done()

    loader = SimpleNamespace(
        discovered_packages={"demo-pkg": {"1.0.0": {"source": "registry"}}},
        devpi_client=SimpleNamespace(simple_index_url=None),
        list_packages=lambda *args, **kwargs: None,
    )
    with (
        mock.patch.object(celery_tasks, "plugin_loader", loader),
        mock.patch.object(subprocess, "run", record),
    ):
        celery_tasks.install_package.run("demo-pkg", "1.0.0", None, [])

    assert len(seen) == 2, "expected a venv step and an install step"
    for env in seen:
        assert env is not None, "an install step would inherit the worker's environment"
        for name in WORKER_ONLY:
            assert name not in env
        assert env["PATH"] == "/usr/bin"


def test_every_subprocess_the_worker_starts_passes_an_environment():
    """A guard for the next spawn point: every child is given an environment, so a
    new installer or run step cannot quietly inherit the worker's credentials."""
    tree = ast.parse(Path(celery_tasks.__file__).read_text())
    spawns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in SPAWNERS
        and getattr(node.func.value, "id", None) == "subprocess"
    ]
    assert spawns, "no subprocess spawn point found — did the module move?"
    for node in spawns:
        assert any(keyword.arg == "env" for keyword in node.keywords), (
            f"the subprocess at line {node.lineno} passes no environment"
        )


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_a_plugin_installs_and_runs_with_nothing_but_the_allowlist(tmp_path, monkeypatch):
    """The point of the gate: with the worker's secrets present, an install and a run
    succeed on the allowlist alone, and the child cannot see the worker's secrets."""
    uv = shutil.which("uv")
    for name in WORKER_ONLY:
        monkeypatch.setenv(name, "worker-only-value")
    env = plugin_child_env()
    assert env, "the allowlist is empty; nothing could run"

    wheel = _plugin_wheel(tmp_path)
    venv = tmp_path / "venv"

    made = subprocess.run(
        [uv, "venv", str(venv), "--python", sys.executable],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert made.returncode == 0, made.stderr

    install = subprocess.run(
        [uv, "pip", "install", "--python", str(venv), "--offline", str(wheel)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert install.returncode == 0, install.stderr

    program = (
        "import json, os, smoke_plugin;"
        "print(json.dumps({'answer': smoke_plugin.ANSWER,"
        " 'worker_secrets': sorted(n for n in os.environ if n.startswith('DJANGO') or n.endswith('_PASSWORD'))}))"
    )
    ran = subprocess.run(
        [str(venv / "bin" / "python"), "-c", program],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    result = json.loads(ran.stdout.strip().splitlines()[-1])
    assert result == {"answer": "the plugin ran", "worker_secrets": []}
