"""The Redis password never reaches the worker's log or a plugin.

The worker logs at debug level while its Celery app is made; that line names the Redis host, never the
backend URL, which carries the password once the shared Redis asks for one.
"""

import importlib
import logging

from aisc_eval import celery_tasks
from aisc_eval.utils import env as env_module

FAKE_PASSWORD = "redis-pw-for-the-test"


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_the_worker_never_logs_the_redis_password(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", FAKE_PASSWORD)
    logger = logging.getLogger("aisc_eval")
    recorder = _Recorder()
    old_level = logger.level
    logger.addHandler(recorder)
    logger.setLevel(logging.DEBUG)
    from aisc_eval import celery_app as celery_app_module

    # the app every task module was made with; the reload below makes another, which must not outlive this test
    the_app = celery_app_module.celery_app
    try:
        importlib.reload(env_module)
        assert (
            FAKE_PASSWORD in env_module.REDIS_BACKEND_URL
        ), "the test password is not in the backend URL"
        importlib.reload(celery_app_module)
    finally:
        celery_app_module.celery_app = the_app
        logger.removeHandler(recorder)
        logger.setLevel(old_level)
        monkeypatch.delenv("REDIS_PASSWORD")
        importlib.reload(env_module)
    logged = "\n".join(recorder.lines)
    assert (
        "redis" in logged.lower()
    ), "nothing about Redis was logged: the check saw no line"
    assert (
        FAKE_PASSWORD not in logged
    ), "the Redis backend URL, password and all, was logged"


def test_a_plugin_never_gets_the_redis_password():
    got = celery_tasks.plugin_environment({"REDIS_PASSWORD": "x", "PATH": "/bin"}, {})
    assert "REDIS_PASSWORD" not in got
