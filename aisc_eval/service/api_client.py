import hashlib
import uuid
from collections.abc import Callable
from time import sleep
from typing import Any

import requests
from aisc_plugin_interface import Measure

from aisc_eval import deployment, run_context
from aisc_eval.scrub import scrub_text
from aisc_eval.data_model.evaluation import Evaluation
from aisc_eval.utils.env import API_URL_PREFIX, INTERNAL_API_KEY
from aisc_eval.utils.logging import get_logger

logger = get_logger()

#: (connect, read) seconds a call to the backend may take before the worker gives up on it
TIMEOUT = (5, 60)
#: the seconds waited before each new try of a call that could not reach the backend (a restart, a dropped
#: network); after the last one the call gives up and raises, so the failure path and the sweep take over
RETRY_WAITS = (1, 2, 4, 8, 16)
#: the answers a proxy gives while the backend restarts: a mark that a test or a run failed is tried again on them
#: as on no answer (it can be sent twice: the engine keeps the first reason)
BACKEND_RESTARTING = (502, 503, 504)


def _with_retries(
    call: Callable[[], requests.Response], retry_statuses: tuple[int, ...] = ()
) -> requests.Response:
    """`call`, tried again after each of RETRY_WAITS while the backend cannot be reached, or answers one of
    `retry_statuses`. Any other answer, even a refusal, is returned as it is: the backend has said no, and asking
    again would change nothing."""
    for wait in RETRY_WAITS:
        try:
            resp = call()
        except requests.ConnectionError:
            sleep(wait)
            continue
        if not retry_statuses or resp.status_code not in retry_statuses:
            return resp
        sleep(wait)
    return call()


def headers(run: dict[str, str] | None = None) -> dict[str, str | None]:
    """The headers of an internal call, read at call time.

    Standalone: the upstream secret alone. Configurator: the same secret, plus the
    project, run ticket and evaluation the backend's door checks. `run`
    defaults to the run of the task being executed (run_context.current(), set
    from the task's aisc_run Celery header); with none, no door header is sent.
    """
    found = {"X-Internal-Secret": INTERNAL_API_KEY}
    if not deployment.is_configurator():
        return found
    if run is None:
        run = run_context.current()
    if run is not None:
        found.update(
            {
                "X-AISC-Project": str(run["project"]),
                "X-AISC-Run": str(run["ticket"]),
                "X-AISC-Evaluation": str(run["evaluation"]),
            }
        )
    return found


def _refusal(resp: requests.Response) -> requests.HTTPError:
    """The error for a refused call: its status and the backend's reason, never its address."""
    text = f"the backend answered HTTP {resp.status_code}"
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str) and detail:
        text += f": {detail[:300]}"
    return requests.HTTPError(text, response=resp)


def _send(
    method: str, url: str, retry_statuses: tuple[int, ...] = (), **kwargs: Any
) -> requests.Response:
    """One call to the backend with the internal headers, given up after TIMEOUT; a refusal raises."""
    resp = _with_retries(
        lambda: getattr(requests, method)(url, headers=headers(), timeout=TIMEOUT, **kwargs),
        retry_statuses,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        pass
    else:
        return resp
    # raised outside the except: requests' own error names the URL, the backend's address, and is not chained
    raise _refusal(resp)


def get_project_settings_by_pid(
    project_pid: uuid.UUID, project_config_selections: list[dict]
) -> list[dict]:
    resp = _with_retries(
        lambda: requests.post(
            f"{API_URL_PREFIX}/projects/settings/{project_pid}/by-pid",
            json={"project_config_selections": project_config_selections},
            headers=headers(),
            timeout=TIMEOUT,
        )
    )
    resp.raise_for_status()
    return resp.json()


def get_evaluation_plugins_status(evaluation_pid: uuid.UUID) -> dict:
    """Get plugin status information for an evaluation."""
    return _send(
        "get", f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/status"
    ).json()


def mark_completed(evaluation_pid: uuid.UUID) -> requests.Response:
    return _send("put", f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Done")


def mark_processing(evaluation_pid: uuid.UUID) -> str:
    """Ask for the run to be shown Processing; the backend answers the run's status, which stays as it is
    when the run is no longer Pending (a run a person stopped stays Failed)."""
    return _send(
        "put", f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Processing"
    ).json()


def mark_failed(evaluation_pid: uuid.UUID) -> None:
    resp = _with_retries(
        lambda: requests.put(
            f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=Failed",
            headers=headers(),
            timeout=TIMEOUT,
        ),
        BACKEND_RESTARTING,
    )
    if not resp.ok:
        logger.warning(
            f"Failed to mark evaluation {evaluation_pid} as failed: {resp.text}"
        )


def mark_plugin_failed(
    evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID, error_message: str
) -> None:
    """Mark a test failed with its reason. A refusal raises: logged only, it would leave the test Running and its
    run Pending with no reason; raised, the caller says so and the sweep closes the test later."""
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/{evaluation_plugin_pid}/fail"
    # a last guard for any caller: key shapes and secret assignments never leave the worker
    _send(
        "patch",
        url,
        retry_statuses=BACKEND_RESTARTING,
        json={"error_message": scrub_text(error_message)},
    )


def mark_plugin_started(
    evaluation_pid: uuid.UUID,
    evaluation_plugin_pid: uuid.UUID,
    code: dict | None = None,
) -> None:
    """The test started; `code`, for a local plugin, is the code it runs: {"digest", "git_commit", "git_dirty"}."""
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/{evaluation_plugin_pid}/timestamp"
    body = {"field": "started_at"}
    if code is not None:
        body["code"] = code
    _send("patch", url, json=body)


def mark_plugin_finished(
    evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID
) -> None:
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/plugins/{evaluation_plugin_pid}/timestamp"
    _send("patch", url, json={"field": "finished_at"})


def get_dataset_file_content(file_name: str) -> bytes:
    return _send(
        "get", f"{API_URL_PREFIX}/files/dataset/{file_name}", stream=True
    ).content


def get_model_file_content(file_name: str) -> bytes:
    return _send(
        "get", f"{API_URL_PREFIX}/files/model/{file_name}", stream=True
    ).content


def get_evaluation_request(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    return _send(
        "get", f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?include=project,plugin"
    ).json()


def get_evaluation_inputs(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    resp = _with_retries(
        lambda: requests.get(
            f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/inputs",
            headers=headers(),
            timeout=TIMEOUT,
        )
    )
    resp.raise_for_status()
    return resp.json()


def store_input_value(
    evaluation_pid: uuid.UUID, input_pid: uuid.UUID, value: dict[str, Any]
) -> None:
    """Store what the run was on (a local model file's name and hash) on one of its inputs."""
    _send(
        "patch",
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/inputs/{input_pid}/value",
        json={"value": value},
    )


def get_evaluation(
    evaluation_pid: uuid.UUID,
) -> Evaluation:
    return Evaluation.model_validate(get_evaluation_request(evaluation_pid))


#: the engine's Measurement text columns and their length; one longer text makes it refuse the whole batch
MEASURE_TEXT_FIELDS = ("name", "description", "unit", "error")
MEASURE_TEXT_MAX = 255


def fit_measure(measure: dict[str, Any]) -> dict[str, Any]:
    """A dumped measure whose texts the engine can store: no NUL (Postgres refuses it) and
    within MEASURE_TEXT_MAX, cut with an ellipsis. The engine finds a metric by its exact
    name, so a cut name ends with a short hash of the whole one: two long names that start
    alike stay two metrics, and the same name is always cut the same way."""
    out = dict(measure)
    for field in MEASURE_TEXT_FIELDS:
        text = out.get(field)
        if not isinstance(text, str):
            continue
        text = out[field] = text.replace("\x00", "")
        if len(text) > MEASURE_TEXT_MAX:
            if field == "name":
                tag = "…[" + hashlib.sha256(text.encode()).hexdigest()[:8] + "]"
                out[field] = text[: MEASURE_TEXT_MAX - len(tag)].rstrip() + tag
            else:
                out[field] = text[: MEASURE_TEXT_MAX - 1].rstrip() + "…"
            logger.warning(
                f"measure {field} cut from {len(text)} to {MEASURE_TEXT_MAX} characters: {text[:80]!r}…"
            )
    return out


def post_measures(
    evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID, metrics: list[Measure]
) -> requests.Response:
    logger.debug(
        f"post_metrics called with {len(metrics)} metrics for evaluation {evaluation_pid}"
    )

    payload = {
        str(evaluation_plugin_uuid): [
            fit_measure(m.model_dump(mode="json")) for m in metrics
        ]
    }
    logger.debug(f"Payload prepared, size: {len(payload)}")

    url = f"{API_URL_PREFIX}/evaluations/{str(evaluation_pid)}/measures"
    logger.debug(f"Posting to URL: {url}")

    response = _send("post", url, json=payload)
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {dict(response.headers)}")
    logger.debug(f"Response content: {response.text}")

    if response.status_code != 201:
        logger.error(f"ERROR: Expected status 201, got {response.status_code}")
        logger.error(f"Response content: {response.text}")
        raise ValueError(response.text)

    return response


def upload_artifact(
    evaluation_pid: uuid.UUID,
    evaluation_plugin_uuid: uuid.UUID,
    name: str,
    content: bytes,
) -> requests.Response:
    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/artifacts"

    files = {"file": (name, content)}
    data = {"evaluation_plugin_uuid": str(evaluation_plugin_uuid)}
    return _send("post", url, files=files, data=data)
