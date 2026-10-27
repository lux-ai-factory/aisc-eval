"""A run's helper models, resolved from the platform just before the plugin starts.

The engine hands the worker a helper input as a reference (helper:<platform project>/<name>) and
the model picked for the run, never an endpoint or a key. The worker asks the platform for the
helper's endpoint and key with the stack's service token; the key goes only into the plugin's
secret environment, never into a file, a log record or an error message.
"""

import os
from urllib.parse import quote

import requests

from aisc_eval.utils.logging import get_logger

logger = get_logger()

REF_PREFIX = "helper:"
DID_NOT_ANSWER = "the platform did not answer"


class HelperUnavailable(RuntimeError):
    """The platform cannot give the run this helper; the message is fit to show the person."""


def parse_ref(ref: str) -> tuple[str, str]:
    """(platform project, helper name) of helper:<project>/<name>."""
    project, _, name = (ref or "").removeprefix(REF_PREFIX).partition("/")
    if not ref.startswith(REF_PREFIX) or not project or not name:
        raise HelperUnavailable(f"the helper reference {ref!r} is malformed")
    return project, name


def resolve_helper(ref: str) -> dict:
    """The platform's {"label", "endpoint_url", "roles", "api_key"} for the helper `ref` names;
    HelperUnavailable with the platform's own reason when it is removed or its key unreadable."""
    project, name = parse_ref(ref)
    base = (os.environ.get("PLATFORM_URL") or "").rstrip("/")
    token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
    if not base or not token:
        raise HelperUnavailable(DID_NOT_ANSWER)
    try:
        resp = requests.get(
            f"{base}/internal/projects/{quote(project, safe='')}/helpers/{quote(name, safe='')}/resolve",
            headers={"X-AISC-Service-Token": token},
            timeout=30,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        logger.warning(f"Could not resolve a helper ({exc.__class__.__name__})")
        raise HelperUnavailable(DID_NOT_ANSWER) from None
    if resp.status_code in (404, 409):
        try:
            detail = (resp.json() or {}).get("detail")
        except (ValueError, AttributeError):
            detail = None
        if not isinstance(detail, str) or not detail:
            detail = (
                f"the helper {name} no longer exists"
                if resp.status_code == 404
                else f"the stored key for helper {name} cannot be decrypted; enter it again"
            )
        raise HelperUnavailable(detail)
    if resp.status_code != 200:
        raise HelperUnavailable(DID_NOT_ANSWER)
    try:
        answer = resp.json()
    except ValueError:
        raise HelperUnavailable(DID_NOT_ANSWER) from None
    if not isinstance(answer, dict) or not answer.get("endpoint_url"):
        raise HelperUnavailable(DID_NOT_ANSWER)
    return answer
