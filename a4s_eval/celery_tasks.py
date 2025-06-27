import uuid

from celery import group

from a4s_eval.celery_app import celery_app
from a4s_eval.service.api_client import (
    fetch_pending_evaluations,
    mark_completed,
    mark_failed,
)
from a4s_eval.tasks.evaluation_tasks import dataset_evaluation_task


@celery_app.task
def poll_and_run_evaluation() -> None:
    eval_ids = fetch_pending_evaluations()
    if not eval_ids:
        eval_ids = []  # Default evaluation ID if none are pending

    # Create a group for each evaluation ID
    print(eval_ids)
    groups = [group(dataset_evaluation_task.s(eval_id)) for eval_id in eval_ids]

    # Apply each group in parallel
    for eval_id, g in zip(eval_ids, groups):
        (g | finalize_evaluation.si(eval_id)).apply_async()


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
    try:
        mark_completed(evaluation_id)
    except Exception:
        mark_failed(evaluation_id)
