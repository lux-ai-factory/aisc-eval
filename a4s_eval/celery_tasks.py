from celery import Celery, group
from a4s_eval.service.api_client import (
    fetch_pending_evaluation,
    mark_completed,
    mark_failed
)

from a4s_eval.tasks.evaluation_tasks import dataset_evaluation_task

from a4s_eval.utils import env

celery_app = Celery(__name__, broker=env.CELERY_BROKER_URL, backend=env.REDIS_BACKEND_URL)

@celery_app.task
def poll_and_run_evaluation():
    eval_id = fetch_pending_evaluation()
    if not eval_id:
        eval_id = 1

    tasks = group([
        dataset_evaluation_task.s(eval_id),
    ])
    (tasks | finalize_evaluation.s(eval_id)).apply_async()

@celery_app.task
def finalize_evaluation(results, evaluation_id):
    try:
        mark_completed(evaluation_id)
    except Exception:
        mark_failed(evaluation_id)
