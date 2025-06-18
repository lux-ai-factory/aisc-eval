from fastapi import APIRouter

from a4s_eval.celery_tasks import poll_and_run_evaluation

router = APIRouter(tags=["evaluations"])



@router.get("/evaluate")
async def evaluate() -> str:

    # Return the number of eval that we found and will performs.
    poll_and_run_evaluation()
    # Async run the evaluation
    return {"message": "Evaluation started."}
