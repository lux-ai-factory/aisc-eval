"""A run stopped from the progress bar: the backend revokes the running test with SIGUSR1, which Celery turns
into SoftTimeLimitExceeded inside the task while it reads the plugin's progress. The worker's failure handler then
ends the plugin's whole process group (SIGTERM, then SIGKILL) and stops the run; the plugin is never left running."""
import signal

import pytest
from billiard.exceptions import SoftTimeLimitExceeded

from aisc_eval import celery_tasks, plugin_process

from test_plugin_process import _Process, _run, worker  # noqa: F401  (the fixture)


class _StoppedWhileRunning(_Process):
    """A plugin still running when the stop arrives: reading its next progress line raises."""

    def __init__(self):
        super().__init__()
        self.ended = False

        def lines():
            yield '{"progress": 0.1}\n'
            raise SoftTimeLimitExceeded()
        self.stdout = lines()

    def poll(self):
        return None if not self.ended else -15

    def wait(self, timeout=None):
        self.ended = True
        return -15


def test_a_stopped_test_ends_its_plugins_processes_and_stops_the_run(worker, monkeypatch):  # noqa: F811
    stops = []
    monkeypatch.setattr(celery_tasks, "_stop_run", lambda *a: stops.append(a))
    monkeypatch.setattr(plugin_process.subprocess, "Popen", lambda argv, **kwargs: _StoppedWhileRunning())
    with pytest.raises(SoftTimeLimitExceeded):
        _run()
    assert (-1, signal.SIGTERM) in worker["group_signals"]
    assert worker["group_signals"][-1] == (-1, signal.SIGKILL)
    [(evaluation, failed, _reason, _task)] = stops
    assert (evaluation, failed) == ("eval", ["ep"])


def test_the_stop_signal_the_backend_sends_is_the_one_celery_turns_into_the_soft_limit():
    from billiard.pool import SIG_SOFT_TIMEOUT
    assert SIG_SOFT_TIMEOUT == signal.SIGUSR1
