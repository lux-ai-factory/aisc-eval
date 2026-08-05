import uuid
from typing import Any

import requests
from pydantic import BaseModel
from aisc_plugin_interface import Measure

from aisc_eval.data_model.evaluation import Evaluation
from aisc_eval.utils.env import API_URL_PREFIX, INTERNAL_API_KEY
from aisc_eval.utils.logging import get_logger

logger = get_logger()

headers = {
    "X-Internal-Secret": INTERNAL_API_KEY
}


def get_project_settings(project_pid: uuid.UUID) -> list[dict]:
    resp = requests.get(f"{API_URL_PREFIX}/projects/settings/{project_pid}", headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_project_settings_by_pid(
    project_pid: uuid.UUID, project_setting_selections: list[dict]
) -> list[dict]:
    resp = requests.post(
        f"{API_URL_PREFIX}/projects/settings/{project_pid}/by-pid",
        json={"project_setting_selections": project_setting_selections},
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


class EvaluationStatusUpdateDTO(BaseModel):
    status: str


def get_evaluation_plugins_status(evaluation_pid: uuid.UUID) -> dict:
    """Get plugin status information for an evaluation."""
    resp = requests.get(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/status", headers=headers)
    resp.raise_for_status()
    return resp.json()


def mark_completed(evaluation_pid: uuid.UUID) -> requests.Response:
    return requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Done", headers = headers)


def mark_failed(evaluation_pid: uuid.UUID) -> None:
    payload = EvaluationStatusUpdateDTO(status="failed").model_dump()
    requests.put(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}",
        json=payload,
        headers=headers
    )
    resp = requests.put(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Failed",
        headers=headers
    )
    if not resp.ok:
        logger.warning(
            f"Failed to mark evaluation {evaluation_pid} as failed: {resp.text}"
        )


def mark_plugin_failed(
    evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID, error_message: str = ""
) -> None:
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/{evaluation_plugin_pid}/fail"
    resp = requests.patch(url, json={"error_message": error_message}, headers=headers)
    if not resp.ok:
        logger.warning(f"Failed to mark plugin {evaluation_plugin_pid} as failed: {resp.text}")


def mark_plugin_started(evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID) -> None:
    url = (
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/{evaluation_plugin_pid}/timestamp"
    )
    resp = requests.patch(url, json={"field": "started_at"}, headers=headers)
    if not resp.ok:
        logger.warning(f"Failed to mark plugin {evaluation_plugin_pid} as started: {resp.text}")


def mark_plugin_finished(evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID) -> None:
    url = (
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/{evaluation_plugin_pid}/timestamp"
    )
    resp = requests.patch(url, json={"field": "finished_at"}, headers=headers)
    if not resp.ok:
        logger.warning(f"Failed to mark plugin {evaluation_plugin_pid} as finished: {resp.text}")


def get_dataset_file_content(file_name: str) -> bytes:
    resp = requests.get(f"{API_URL_PREFIX}/files/dataset/{file_name}", stream=True, headers=headers)
    resp.raise_for_status()

    return resp.content


def get_model_file_content(file_name: str) -> bytes:
    resp = requests.get(f"{API_URL_PREFIX}/files/model/{file_name}", stream=True, headers=headers)
    resp.raise_for_status()

    return resp.content


def get_evaluation_request(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    resp = requests.get(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?include=project,plugin", headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_evaluation(
    evaluation_pid: uuid.UUID,
) -> Evaluation:
    return Evaluation.model_validate(get_evaluation_request(evaluation_pid))


def post_measures(
    evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID, metrics: list[Measure]
) -> requests.Response:
    logger.debug(
        f"post_metrics called with {len(metrics)} metrics for evaluation {evaluation_pid}"
    )

    payload = {str(evaluation_plugin_uuid): [m.model_dump(mode="json") for m in metrics]}
    logger.debug(f"Payload prepared, size: {len(payload)}")

    url = f"{API_URL_PREFIX}/evaluations/{str(evaluation_pid)}/measures"
    logger.debug(f"Posting to URL: {url}")

    response = requests.post(url, json=payload, headers=headers)
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {dict(response.headers)}")
    logger.debug(f"Response content: {response.text}")

    if response.status_code != 201:
        logger.error(f"ERROR: Expected status 201, got {response.status_code}")
        logger.error(f"Response content: {response.text}")
        raise ValueError(response.text)

    return response


def upload_artifact(
    evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID, name: str, content: bytes
) -> requests.Response:
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/artifacts"

    files = {'file': (name, content)}
    data = {'evaluation_plugin_uuid': str(evaluation_plugin_uuid)}
    response = requests.post(url, files=files, data=data, headers=headers)

    return response
