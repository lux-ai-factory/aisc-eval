"""Reading a plugin's manifest and running its form hooks go to a queue of their own, which a worker process of
its own serves one task at a time, so a configure page never waits behind the runs. No run task is sent there,
and the run worker does not read that queue (the compose files say so, scripts/tests checks them)."""

from aisc_eval import celery_tasks  # noqa: F401  (registers the tasks)
from aisc_eval.celery_app import celery_app

DECLARATIONS_QUEUE = "plugin-declarations"
DECLARATION_TASKS = {
    "aisc_eval.celery_tasks.read_manifest",
    "aisc_eval.celery_tasks.plugin_hook",
}


def _queue_of(task_name: str) -> str:
    return celery_app.amqp.router.route({}, task_name)["queue"].name


def test_the_manifest_read_and_the_form_hooks_go_to_their_own_queue():
    assert {name: _queue_of(name) for name in DECLARATION_TASKS} == dict.fromkeys(
        DECLARATION_TASKS, DECLARATIONS_QUEUE
    )


def test_no_run_task_goes_to_the_declarations_queue():
    from aisc_eval import run_sweep  # noqa: F401  (the sweep's task)

    run_tasks = sorted(
        name
        for name in celery_app.tasks
        if name.startswith("aisc_eval.") and name not in DECLARATION_TASKS
    )
    assert "aisc_eval.celery_tasks.run_evaluation" in run_tasks
    assert {name: _queue_of(name) for name in run_tasks} == dict.fromkeys(
        run_tasks, celery_app.conf.task_default_queue
    )
    assert celery_app.conf.task_default_queue != DECLARATIONS_QUEUE
