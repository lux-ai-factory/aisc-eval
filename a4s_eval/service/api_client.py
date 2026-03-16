import uuid
from typing import Any

import requests
from pydantic import BaseModel

from a4s_eval.audit.events import (
    API_CALL_GET_DATASET,
    API_CALL_GET_EVALUATION,
    API_CALL_GET_MODEL,
    API_CALL_MARK_COMPLETED,
    API_CALL_MARK_FAILED,
    API_CALL_POST_MEASURES,
    AuditTimer,
    log_audit_event,
)
from a4s_eval.data_model.evaluation import Evaluation
from a4s_eval.data_model.measure import Measure
from a4s_eval.utils.env import API_URL_PREFIX
from a4s_eval.utils.logging import get_logger

logger = get_logger()


class EvaluationStatusUpdateDTO(BaseModel):
    status: str


def mark_completed(evaluation_pid: uuid.UUID) -> requests.Response:
    with AuditTimer() as timer:
        response = requests.put(
            f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Done"
        )
    log_audit_event(
        API_CALL_MARK_COMPLETED,
        evaluation_id=str(evaluation_pid),
        duration_ms=timer.duration_ms,
        details={"status_code": response.status_code},
    )
    return response


def mark_failed(evaluation_pid: uuid.UUID) -> None:
    payload = EvaluationStatusUpdateDTO(status="failed").model_dump()
    with AuditTimer() as timer:
        requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}", json=payload)
    log_audit_event(
        API_CALL_MARK_FAILED,
        evaluation_id=str(evaluation_pid),
        duration_ms=timer.duration_ms,
    )


def get_dataset_file_content(file_name: str, evaluation_id: str = "") -> bytes:
    with AuditTimer() as timer:
        resp = requests.get(f"{API_URL_PREFIX}/files/dataset/{file_name}", stream=True)
        resp.raise_for_status()
        content = resp.content
    log_audit_event(
        API_CALL_GET_DATASET,
        evaluation_id=evaluation_id,
        duration_ms=timer.duration_ms,
        details={"file_name": file_name, "content_length": len(content)},
    )
    return content


def get_model_file_content(file_name: str, evaluation_id: str = "") -> bytes:
    with AuditTimer() as timer:
        resp = requests.get(f"{API_URL_PREFIX}/files/model/{file_name}", stream=True)
        resp.raise_for_status()
        content = resp.content
    log_audit_event(
        API_CALL_GET_MODEL,
        evaluation_id=evaluation_id,
        duration_ms=timer.duration_ms,
        details={"file_name": file_name, "content_length": len(content)},
    )
    return content


def get_evaluation_request(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    with AuditTimer() as timer:
        resp = requests.get(
            f"{API_URL_PREFIX}/evaluations/{evaluation_pid}"
            "?include=project,dataset,model,datashape,plugin"
        )
        resp.raise_for_status()
        data = resp.json()
    log_audit_event(
        API_CALL_GET_EVALUATION,
        evaluation_id=str(evaluation_pid),
        duration_ms=timer.duration_ms,
    )
    return data


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

    with AuditTimer() as timer:
        response = requests.post(url, json=payload)

    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {dict(response.headers)}")
    logger.debug(f"Response content: {response.text}")

    log_audit_event(
        API_CALL_POST_MEASURES,
        evaluation_id=str(evaluation_pid),
        duration_ms=timer.duration_ms,
        details={"metric_count": len(payload), "status_code": response.status_code},
    )

    if response.status_code != 201:
        logger.error(f"ERROR: Expected status 201, got {response.status_code}")
        logger.error(f"Response content: {response.text}")
        raise ValueError(response.text)

    return response
