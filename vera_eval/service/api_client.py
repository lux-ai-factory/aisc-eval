import uuid
from typing import Any

import requests
from pydantic import BaseModel

from vera_eval.data_model.evaluation import Evaluation
from vera_eval.data_model.measure import Measure
from vera_eval.utils.env import API_URL_PREFIX
from vera_eval.utils.logging import get_logger

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
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?include=project,plugin"
    )
    resp.raise_for_status()
    return resp.json()


def get_evaluation(
    evaluation_pid: uuid.UUID,
) -> Evaluation:
    return Evaluation.model_validate(get_evaluation_request(evaluation_pid))


def post_measures(
    evaluation_pid: uuid.UUID, plugin_name: str, metrics: list[Measure]
) -> requests.Response:

    logger.debug(
        f"post_metrics called with {len(metrics)} metrics for evaluation {evaluation_pid}"
    )

    payload = {
        plugin_name: [m.model_dump() for m in metrics]
    }
    logger.debug(f"Payload prepared, size: {len(payload)}")

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

def upload_artifact(evaluation_pid: uuid.UUID, plugin_name: str, name: str, content: bytes) -> requests.Response:
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/artifacts"

    files = {'file': (name, content)}
    data = {'plugin_name': plugin_name}
    response = requests.post(url, files=files, data=data)

    return response