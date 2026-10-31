"""A run's helper models, resolved from the platform just before the plugin starts.

The engine hands the worker a helper input as a reference (helper:<platform project>/<name>) and
the model picked for the run, never an endpoint or a key. The worker asks the platform for the
helper's endpoint and key with the stack's service token, naming the run's own project, and never
asks for a helper of another project; the key goes only into the plugin's secret environment, never
into a file, a log record or an error message.
"""

import os
import re
from urllib.parse import quote

import requests

from aisc_eval.utils.logging import get_logger

logger = get_logger()

REF_PREFIX = "helper:"
DID_NOT_ANSWER = "the platform did not answer"


class HelperUnavailable(RuntimeError):
    """The platform cannot give the run this helper; the message is fit to show the person."""


_PROJECT = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
#: the platform's rule for a helper's name
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")


def parse_ref(ref: str | None) -> tuple[str, str]:
    """(platform project, helper name) of helper:<project>/<name>. Both parts are checked against
    the platform's own formats: they go into a URL path that carries the service token, so a
    segment such as .. must never reach it."""
    if isinstance(ref, str) and ref.startswith(REF_PREFIX):
        project, _, name = ref[len(REF_PREFIX) :].partition("/")
        if _PROJECT.fullmatch(project) and _NAME.fullmatch(name):
            return project, name
    raise HelperUnavailable("the helper reference is malformed")


def resolve_helper(ref: str, run_project: str | None) -> dict:
    """The platform's {"label", "endpoint_url", "roles", "api_key"} for the helper `ref` names, for a
    run of `run_project` (the platform pid of the evaluation's project); HelperUnavailable with the
    platform's own reason when it is removed or its key unreadable, and without asking the platform
    when the run has no project or the helper is of another one."""
    project, name = parse_ref(ref)
    if not run_project:
        raise HelperUnavailable("no helper: the run carries no platform project")
    if project.lower() != str(run_project).lower():
        raise HelperUnavailable("the helper is of another project than the run")
    base = (os.environ.get("PLATFORM_URL") or "").rstrip("/")
    token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
    if not base or not token:
        raise HelperUnavailable(DID_NOT_ANSWER)
    try:
        resp = requests.get(
            f"{base}/internal/projects/{quote(project, safe='')}/helpers/{quote(name, safe='')}/resolve",
            # the platform refuses a helper of another project than the run's
            headers={"X-AISC-Service-Token": token, "X-AISC-Project": str(run_project)},
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
