import uuid

from celery import group

from a4s_eval.celery_app import celery_app
from a4s_eval.service.api_client import (
    fetch_pending_evaluations,
    mark_completed,
    mark_failed,
)
from a4s_eval.tasks.evaluation_tasks import (
    model_evaluation_task,
)


@celery_app.task
def poll_and_run_evaluation() -> None:
    print("Polling for pending evaluations...")
    eval_ids = fetch_pending_evaluations()
    print(f"Found {len(eval_ids)} pending evaluations: {eval_ids}")

    if not eval_ids:
        print("No pending evaluations found")
        return

    # Create a group for each evaluation ID
    print(f"Creating tasks for {len(eval_ids)} evaluations...")
    groups = [
        group(
            [
                # dataset_evaluation_task.s(eval_id).on_error(handle_error.s(eval_id)),
                model_evaluation_task.s(eval_id).on_error(handle_error.s(eval_id)),
            ]
        )
        for eval_id in eval_ids
    ]

    # Apply each group in parallel
    for eval_id, g in zip(eval_ids, groups):
        print(f"Launching evaluation task for {eval_id}")
        (g | finalize_evaluation.si(eval_id)).apply_async()
        print(f"Task launched for {eval_id}")


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
    print(f"Finalizing evaluation {evaluation_id}")
    try:
        response = mark_completed(evaluation_id)
        print(
            f"Evaluation {evaluation_id} marked as completed, status: {response.status_code}"
        )
    except Exception as e:
        print(f"Failed to mark evaluation {evaluation_id} as completed: {e}")
        mark_failed(evaluation_id)


@celery_app.task
def handle_error(evaluation_id, request, exc, traceback) -> None:
    print(f"Error in evaluation {evaluation_id}:")

    print(f"--\n\n{request} {exc} {traceback}")

    mark_failed(evaluation_id)
    print(f"Evaluation {evaluation_id} marked as failed due to error.")
