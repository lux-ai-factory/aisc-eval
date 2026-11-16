"""The worker logs one JSON object per line, without colour codes, and every line written while a test runs
names its evaluation: the formatter of the worker's stdout handler is the shared JSON one, and a task sets the
evaluation pid of its run before it starts and clears it after."""
import importlib.util
import json
import logging
import logging.config
import weakref
from pathlib import Path

import yaml

LOGGING_YAML = Path(__file__).resolve().parents[1] / "config/logging.yaml"
PID = "11111111-2222-4333-8444-555555555555"


def _stdout_formatter() -> logging.Formatter:
    cfg = yaml.safe_load(LOGGING_YAML.read_text())
    spec = dict(cfg["formatters"][cfg["handlers"]["stdout"]["formatter"]])
    return logging.config.DictConfigurator({}).configure_formatter(spec)


def _logs():
    assert importlib.util.find_spec("aisc_identity.logs") is not None, \
        "there is no shared JSON log module (aisc_identity.logs)"
    from aisc_identity import logs
    return logs


def test_a_line_is_json_without_colour_and_names_the_evaluation():
    logs = _logs()
    formatter = _stdout_formatter()
    token = logs.evaluation_pid.set(PID)
    try:
        record = logging.LogRecord("aisc_eval.celery_tasks", logging.WARNING, __file__, 1, "hello", None, None)
        out = formatter.format(record)
    finally:
        logs.evaluation_pid.reset(token)
    assert "\x1b[" not in out, "the worker's log lines carry colour codes"
    line = json.loads(out)
    assert line["evaluation_pid"] == PID and line["message"] == "hello" and line["level"] == "WARNING"


def test_a_task_sets_its_evaluation_pid_for_its_lines():
    logs = _logs()
    from celery.signals import task_postrun, task_prerun

    from aisc_eval import celery_app, celery_tasks  # noqa: F401  (connects the worker's signal handlers)

    # each receiver is ((ids), weak reference to the function); its repr names the function, not its module
    receivers = [ref() if isinstance(ref, weakref.ref) else ref for _, ref in task_prerun.receivers]
    ours = [r for r in receivers if (getattr(r, "__module__", None) or "").startswith("aisc_eval")]
    assert ours, "the worker connects nothing to task_prerun, so no line names the run"
    args = ("pkg", "Plugin", "1.0", {}, [], [], PID, "ep")
    task_prerun.send(sender=celery_tasks.run_plugin, task_id="t1", task=celery_tasks.run_plugin, args=args,
                     kwargs={})
    try:
        assert logs.evaluation_pid.get() == PID
    finally:
        task_postrun.send(sender=celery_tasks.run_plugin, task_id="t1", task=celery_tasks.run_plugin, args=args,
                          kwargs={}, retval=None, state="SUCCESS")
    assert logs.evaluation_pid.get() is None
