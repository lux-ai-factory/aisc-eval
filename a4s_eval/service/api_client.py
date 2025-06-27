import datetime
import json
import uuid
from io import BytesIO
from typing import Annotated, Any, Callable

import pandas as pd
import requests
from fastapi import Depends
from pydantic import BaseModel

from a4s_eval.data_model.evaluation import Evaluation
from a4s_eval.data_model.metric import Metric
from a4s_eval.utils.env import API_URL_PREFIX


class EvaluationStatusUpdateDTO(BaseModel):
    status: str


class MetricDTO(BaseModel):
    name: str
    value: float | str


def fetch_pending_evaluations() -> list[uuid.UUID]:
    resp = requests.get(f"{API_URL_PREFIX}/evaluations?status=pending")
    if resp.status_code != 200:
        return []
    evaluations = resp.json()
    claimed_pids = []
    print(evaluations)
    for e in evaluations:
        print(e)
        if claim_evaluation(e["evaluation_pid"]):
            claimed_pids.append(uuid.UUID(e["evaluation_pid"]))
    print(claimed_pids)
    return claimed_pids


def claim_evaluation(evaluation_pid: uuid.UUID) -> bool:
    # payload = EvaluationStatusUpdateDTO(status="processing").model_dump()
    # resp = requests.put(
    #     f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=processing"
    # )
    # print(resp.status_code)
    # return resp.status_code == 200
    return True


def mark_completed(evaluation_pid: uuid.UUID) -> requests.Response:
    payload = EvaluationStatusUpdateDTO(status="done").model_dump()
    return requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=done")


def mark_failed(evaluation_pid: uuid.UUID) -> None:
    payload = EvaluationStatusUpdateDTO(status="failed").model_dump()
    return requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}", json=payload)


def get_dataset_data(dataset_pid: str) -> pd.DataFrame:
    resp = requests.get(f"{API_URL_PREFIX}/datasets/{dataset_pid}/data", stream=True)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")

    if "parquet" in content_type:
        content_buffer = BytesIO(resp.content)
        content_buffer.seek(0)
        return pd.read_parquet(content_buffer)
    elif "csv" in content_type:
        content_buffer = BytesIO(resp.content)
        content_buffer.seek(0)
        return pd.read_csv(content_buffer)
    else:
        raise ValueError("Unsupported dataset format")


def get_evaluation_request(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    resp = requests.get(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?include=project,dataset,model,datashape"
    )
    resp.raise_for_status()
    return resp.json()


def get_evaluation(
    evaluation_pid: uuid.UUID,
) -> Evaluation:
    return Evaluation.model_validate(get_evaluation_request(evaluation_pid))


def post_metrics(evaluation_pid: uuid.UUID, metrics: list[Metric]) -> requests.Response:
    payload = [metric.model_dump() for metric in metrics]

    response = requests.post(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/metrics", json=payload
    )

    if response.status_code != 201:
        raise ValueError(response.content)

    return requests.post(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/metrics", json=payload
    )
