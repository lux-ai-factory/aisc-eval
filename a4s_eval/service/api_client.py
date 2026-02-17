import uuid
from typing import Any

import requests
from pydantic import BaseModel

from a4s_eval.data_model.evaluation import Evaluation
from a4s_eval.data_model.measure import Measure
from a4s_eval.utils.env import API_URL_PREFIX
from a4s_eval.utils.logging import get_logger

logger = get_logger()

class EvaluationStatusUpdateDTO(BaseModel):
    status: str



def mark_completed(evaluation_pid: uuid.UUID) -> requests.Response:
    return requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Done")


def mark_failed(evaluation_pid: uuid.UUID) -> None:
    payload = EvaluationStatusUpdateDTO(status="failed").model_dump()
    requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}", json=payload)


def get_dataset_file_content(file_name: str) -> bytes:
    resp = requests.get(f"{API_URL_PREFIX}/files/dataset/{file_name}", stream=True)
    resp.raise_for_status()

    return resp.content


def get_model_file_content(file_name: str) -> bytes:
    resp = requests.get(f"{API_URL_PREFIX}/files/model/{file_name}", stream=True)
    resp.raise_for_status()

    return resp.content


def get_evaluation_request(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    resp = requests.get(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?include=project,dataset,model,datashape,plugin"
    )
    resp.raise_for_status()
    return resp.json()


def get_evaluation(
    evaluation_pid: uuid.UUID,
) -> Evaluation:
    return Evaluation.model_validate(get_evaluation_request(evaluation_pid))


def post_measures(
    evaluation_pid: uuid.UUID, metrics: list[Measure]
) -> requests.Response:
    """Post metrics to the API for a specific evaluation.

    Args:
        evaluation_pid: UUID of the evaluation to post metrics for
        metrics: List of metrics to post

    Returns:
        requests.Response: The API response
    """
    logger.debug(
        f"post_metrics called with {len(metrics)} metrics for evaluation {evaluation_pid}"
    )

    payload = [m.model_dump() for m in metrics]
    logger.debug(f"Payload prepared, size: {len(payload)}")
    if len(payload) > 0:
        logger.debug(f"Sample payload item: {payload[0]}")

    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/measures"
    logger.debug(f"Posting to URL: {url}")

    response = requests.post(url, json=payload)
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {dict(response.headers)}")
    logger.debug(f"Response content: {response.text}")

    if response.status_code != 201:
        logger.error(f"ERROR: Expected status 201, got {response.status_code}")
        logger.error(f"Response content: {response.text}")
        raise ValueError(response.text)

    return response
