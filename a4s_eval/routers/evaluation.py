from fastapi import APIRouter
from a4s_eval.celery_tasks import poll_and_run_evaluation
from a4s_eval.utils.logging import get_logger

router = APIRouter()
logger = get_logger()


@router.get("/evaluate")
async def evaluate() -> dict[str, str]:
    """Trigger evaluation of pending evaluatsions"""
    try:
        logger.debug("=== EVALUATE ENDPOINT START ===")
        logger.debug("1. About to call poll_and_run_evaluation.delay()")

        # Launch the evaluation task asynchronously
        task = poll_and_run_evaluation.delay()

        logger.debug("2. poll_and_run_evaluation.delay() completed successfully")
        logger.debug(f"3. Task ID: {task.id}")
        logger.debug("4. About to return response")

        return {
            "message": "Evaluation started.",
            "task_id": str(task.id),
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"ERROR in evaluate endpoint: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"message": f"Failed to start evaluation: {str(e)}", "status": "error"}
    finally:
        logger.debug("=== EVALUATE ENDPOINT END ===")
