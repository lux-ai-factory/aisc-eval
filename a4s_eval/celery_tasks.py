import pathlib
import uuid

from celery import group
import yaml

from a4s_eval.celery_app import celery_app
from a4s_eval.service.api_client import (
    fetch_pending_evaluations,
    mark_completed,
    mark_failed,
)
from a4s_eval.tasks.data_metric_tasks import dataset_evaluation_task
from a4s_eval.tasks.metric_tasks import metric_task
from a4s_eval.tasks.prediction_metric_tasks import (
    model_evaluation_task,
)
from a4s_eval.utils.logging import get_logger

logger = get_logger()


@celery_app.task
def poll_and_run_evaluation() -> None:
    try:
        logger.debug("=== POLL_AND_RUN_EVALUATION START ===")
        logger.debug("1. Starting poll_and_run_evaluation task")

        logger.debug("2. About to call fetch_pending_evaluations()")
        eval_ids = fetch_pending_evaluations()
        logger.debug(
            f"3. fetch_pending_evaluations() completed. Found {len(eval_ids)} evaluations: {eval_ids}"
        )

        if not eval_ids:
            logger.debug("4. No pending evaluations found, returning")
            return

        logger.debug(f"5. Creating signatures for {len(eval_ids)} evaluations...")
        signatures = [
            generate_evaluation_signature(eval_id) for eval_id in eval_ids
        ]
        logger.debug(f"6. Signatures created: {len(signatures)}")

        logger.debug("7. Starting to apply signatures...")
        # Apply each signature in parallel
        for i, (eval_id, sig) in enumerate(zip(eval_ids, signatures)):
            logger.debug(f"8.{i + 1} About to launch evaluation task for {eval_id}")
            try:
                sig.apply_async()
                logger.debug(f"9.{i + 1} Task launched successfully for {eval_id}")
            except Exception as e:
                logger.error(f"ERROR launching task for {eval_id}: {str(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

        logger.debug("10. All tasks processed")

    except Exception as e:
        logger.error(f"ERROR in poll_and_run_evaluation: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        logger.debug("=== POLL_AND_RUN_EVALUATION END ===")


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
    logger.debug(f"Finalizing evaluation {evaluation_id}")
    try:
        response = mark_completed(evaluation_id)
        logger.debug(
            f"Evaluation {evaluation_id} marked as completed, status: {response.status_code}"
        )
    except Exception as e:
        logger.error(f"Failed to mark evaluation {evaluation_id} as completed: {e}")
        mark_failed(evaluation_id)


@celery_app.task
def handle_error(
    evaluation_id: uuid.UUID,
    request: object,
    exc: BaseException,
    traceback: object,
) -> None:
    logger.error(f"Error in evaluation {evaluation_id}:")
    logger.error(f"--\n\n{request} {exc} {traceback}")
    mark_failed(evaluation_id)
    logger.error(f"Evaluation {evaluation_id} marked as failed due to error.")


def generate_evaluation_signature(evaluation_pid: uuid.UUID) -> None:
    config_file = pathlib.Path("config/eval_config.yaml")
    with open(config_file) as f_in:
        eval_config = yaml.safe_load(f_in)

    task_signatures = []
    for registry_name, metric_list in eval_config.items():
        task_signatures.append(
            metric_task.s(
                evaluation_pid, 
                registry_name, 
                metric_list
            ).on_error(
                handle_error.s(evaluation_pid)
            ),
        )

    return (group(task_signatures) | finalize_evaluation.si(evaluation_pid))