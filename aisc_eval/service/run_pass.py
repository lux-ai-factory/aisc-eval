"""A plugin's pass to the platform's connection routes, for one run.

The stack's PLATFORM_CONNECTIONS_TOKEN opens every project's connections, so it stays with the
worker. Before a plugin starts, the worker asks the platform for a pass that opens only the
targets and connections this run's resource inputs name; the plugin gets the pass as
PLATFORM_RUN_PASS, and the worker revokes it when the plugin has ended (it also expires).
"""

import os
from collections.abc import Iterable

import requests

from aisc_eval.utils.logging import get_logger

logger = get_logger()

REF_PREFIXES = ("target:", "connection:")


class RunPassRefused(RuntimeError):
    """The platform did not mint a pass for the run."""


def refs_of(input_components: Iterable[dict]) -> list[str]:
    """The target: and connection: values of a run's resource inputs, in order, once each."""
    refs = []
    for component in input_components:
        if component.get("component_type") != "resource":
            continue
        value = (component.get("json_value") or {}).get("value")
        if isinstance(value, str) and value.startswith(REF_PREFIXES):
            refs.append(value)
    return list(dict.fromkeys(refs))


def _platform_url() -> str:
    return (os.environ.get("PLATFORM_URL") or "").rstrip("/")


def mint(refs: list[str], evaluation_plugin_uuid: str, project: str | None) -> str:
    """A pass for exactly these refs, for one evaluation plugin's run of `project` (the platform pid of
    the evaluation's project: the platform refuses a ref of another project, and records which
    connection served the run); RunPassRefused when the platform does not give one."""
    if not project:
        raise RunPassRefused("no run pass: the run carries no platform project")
    base = _platform_url()
    token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
    if not base or not token:
        raise RunPassRefused(
            "no run pass: PLATFORM_URL and PLATFORM_CONNECTIONS_TOKEN must be set on the worker"
        )
    try:
        resp = requests.post(
            f"{base}/internal/run-passes",
            json={
                "refs": refs,
                "evaluation_plugin": evaluation_plugin_uuid,
                "project": project,
            },
            headers={"X-AISC-Service-Token": token},
            timeout=30,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RunPassRefused(
            f"no run pass: the platform could not be reached ({exc.__class__.__name__})"
        ) from None
    if resp.status_code != 201:
        raise RunPassRefused(
            f"no run pass: the platform refused it ({resp.status_code}): {resp.text[:500]}"
        )
    try:
        run_pass = (resp.json() or {}).get("pass")
    except (ValueError, AttributeError):
        run_pass = None
    if not isinstance(run_pass, str) or not run_pass:
        raise RunPassRefused("no run pass: the platform answered without one")
    return run_pass


def trace(run_pass: str) -> bytes | None:
    """The run's calls to the system under test, as the platform recorded them (JSON lines ending with the
    seal), read with the run's pass before it is revoked. None when the platform cannot give them; the
    platform keeps them anyway, so a failure here is only logged."""
    try:
        resp = requests.get(
            f"{_platform_url()}/internal/run-passes/trace",
            headers={"X-AISC-Run-Pass": run_pass},
            timeout=60,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        logger.warning(
            f"Could not read the run's calls from the platform ({exc.__class__.__name__}); they stay in its trace"
        )
        return None
    if resp.status_code != 200:
        logger.warning(
            f"The platform did not give the run's calls ({resp.status_code}); they stay in its trace"
        )
        return None
    return resp.content


def revoke(run_pass: str, run: dict | None = None) -> None:
    """Revoke a pass; a failure is only logged, the pass expires anyway. The platform seals the run's calls
    in the ledger as it ends the pass, citing the request and the run of the run's start: the request_id and
    run_id the backend put in the run's aisc_run header (`run`), sent when the run has them."""
    headers = {"X-AISC-Run-Pass": run_pass}
    for key, header in (
        ("request_id", "X-AISC-Request-Id"),
        ("run_id", "X-AISC-Run-Id"),
    ):
        if (run or {}).get(key):
            headers[header] = str(run[key])
    try:
        resp = requests.delete(
            f"{_platform_url()}/internal/run-passes",
            headers=headers,
            timeout=30,
            allow_redirects=False,
        )
        if not 200 <= resp.status_code < 300:
            logger.warning(
                f"The platform did not revoke a run pass ({resp.status_code}); it expires on its own"
            )
    except requests.RequestException as exc:
        logger.warning(
            f"Could not revoke a run pass ({exc.__class__.__name__}); it expires on its own"
        )
