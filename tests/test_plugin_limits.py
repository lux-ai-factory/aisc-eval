"""A plugin runs under resource limits, its outputs are capped, and its run pass lasts as long as a test may.

Memory, processes, file size and CPU time are capped in the plugin's own process (set in preexec, after the switch
to the plugin user), each from a setting read at call time. The files a plugin leaves are refused when their total
is above a cap, before any is read. The pass the worker mints for the platform names its own lifetime: the test's
hard time limit plus a margin, so no call made with it can land after the test has ended.
"""

import inspect
import os
import resource
import sys

import pytest

from aisc_eval import celery_tasks, plugin_process

from test_plugin_process import TARGET, _me, _resource, _run_with, platform, worker  # noqa: F401  (fixtures)


def test_the_limits_come_from_the_settings(monkeypatch):
    plugin_limits = getattr(plugin_process, "plugin_limits", None)
    assert plugin_limits is not None, "the plugin runs with no resource limit"
    monkeypatch.setenv("PLUGIN_MAX_MEMORY_MB", "1024")
    monkeypatch.setenv("PLUGIN_MAX_PROCESSES", "64")
    monkeypatch.setenv("PLUGIN_MAX_FILE_MB", "8")
    limits = plugin_limits(24000)
    assert limits[resource.RLIMIT_AS] == (1024 * 2**20,) * 2
    assert limits[resource.RLIMIT_NPROC] == (64, 64)
    assert limits[resource.RLIMIT_FSIZE] == (8 * 2**20,) * 2
    assert limits[resource.RLIMIT_CPU] == (24000, 24060)


def test_a_started_plugin_runs_under_its_limits(tmp_path):
    assert (
        "limits" in inspect.signature(plugin_process.start).parameters
    ), "start() takes no limits"
    with open(tmp_path / "err", "w+") as err:
        process = plugin_process.start(
            [
                sys.executable,
                "-c",
                "import resource; print(resource.getrlimit(resource.RLIMIT_FSIZE))",
            ],
            tmp_path,
            err,
            {"PATH": os.environ["PATH"]},
            _me(),
            limits={resource.RLIMIT_FSIZE: (4096, 4096)},
        )
        assert process.stdout.read().strip() == "(4096, 4096)"
        plugin_process.wait(process)


def test_outputs_larger_than_the_total_cap_are_refused(tmp_path):
    for name in ("a.bin", "b.bin"):
        (tmp_path / name).write_bytes(b"x" * 600)
    assert (
        "max_total_bytes" in inspect.signature(plugin_process.read_outputs).parameters
    ), "a plugin's outputs are read whatever their size"
    with pytest.raises(RuntimeError, match="larger than"):
        plugin_process.read_outputs(tmp_path, _me(), max_total_bytes=1000)


def test_the_pass_lives_as_long_as_a_test_may_run(worker, platform):
    _run_with([_resource("target", TARGET)])
    _method, _url, _headers, body = platform["calls"][0]
    margin = getattr(celery_tasks, "RUN_PASS_MARGIN_S", 300)
    assert (
        body.get("ttl_s") == int(celery_tasks.celery_app.conf.task_time_limit) + margin
    ), "the pass does not say how long it lasts, so the platform's default (12 h) outlives the test"


def test_the_worker_starts_its_plugin_under_the_limits(worker, platform):
    _run_with([_resource("target", TARGET)])
    [(_argv, kwargs)] = worker["popen"]
    assert (
        kwargs["preexec_fn"] is not None
    ), "the worker starts its plugin with no limits"


def test_a_plugin_running_as_the_worker_is_not_given_the_process_cap(tmp_path):
    """The process cap counts every process of the user: as the worker's own user it would count the worker's."""
    script = "import resource; print(resource.getrlimit(resource.RLIMIT_NPROC))"
    before = resource.getrlimit(resource.RLIMIT_NPROC)
    with open(tmp_path / "err", "w+") as err:
        process = plugin_process.start(
            [sys.executable, "-c", script],
            tmp_path,
            err,
            {"PATH": os.environ["PATH"]},
            _me(),
            limits={resource.RLIMIT_NPROC: (1, 1)},
        )
        assert process.stdout.read().strip() == str(before)
        plugin_process.wait(process)


def test_cpu_time_is_counted_for_each_core_a_plugin_may_keep_busy(monkeypatch):
    monkeypatch.setenv("PLUGIN_CPU_CORES", "2")
    assert celery_tasks.plugin_cpu_seconds() == 2 * int(
        celery_tasks.celery_app.conf.task_soft_time_limit
    )
    monkeypatch.setenv("PLUGIN_CPU_CORES", "none")
    with pytest.raises(RuntimeError, match="PLUGIN_CPU_CORES"):
        celery_tasks.plugin_cpu_seconds()


@pytest.mark.parametrize("value", ["lots", "0", "-5"])
def test_a_limit_that_is_not_a_positive_whole_number_is_refused(monkeypatch, value):
    monkeypatch.setenv("PLUGIN_MAX_MEMORY_MB", value)
    with pytest.raises(RuntimeError, match="PLUGIN_MAX_MEMORY_MB"):
        plugin_process.plugin_limits(60)
