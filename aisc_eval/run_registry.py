"""The runs the dispatcher sent off, kept in the result backend's Redis so the sweep can close what the worker loses.

A run is claimed once: its task list is set under eval_tasks:<pid> only when absent (the key Stop already reads to
revoke a run's tasks), so a second delivery of the dispatcher finds the claim and dispatches nothing. Beside it, the
run is registered under eval_run:<pid> with its run header (to act as the run), when it was claimed, the deadline
after which nothing of it may still be waiting, and the task ids of each test and of its package's install.
"""

import json
from dataclasses import asdict, dataclass

from aisc_eval.utils.logging import get_logger

logger = get_logger()

#: how long before its run ticket expires a run must have ended: after that the worker can no longer report it
DEADLINE_MARGIN_S = 600
_TASKS = "eval_tasks:"
_REGISTERED = "eval_run:"


@dataclass(frozen=True)
class RegisteredRun:
    evaluation: str
    #: the run's aisc_run header (project, evaluation, ticket), None in standalone
    run: dict | None
    claimed_at: float
    deadline: float
    #: each test pid and the id of its run_plugin task
    tests: dict[str, str]
    #: each test pid and the id of its package's install_package task
    installs: dict[str, str]


def tasks_key(evaluation_pid) -> str:
    return f"{_TASKS}{evaluation_pid}"


def _registered_key(evaluation_pid) -> str:
    return f"{_REGISTERED}{evaluation_pid}"


def claim(client, evaluation_pid, task_ids: list[str], ttl: int) -> bool:
    """Claim the run for this delivery: True when no delivery claimed it before."""
    return bool(
        client.set(tasks_key(evaluation_pid), json.dumps(task_ids), ex=ttl, nx=True)
    )


def register(client, entry: RegisteredRun, ttl: int) -> None:
    client.set(_registered_key(entry.evaluation), json.dumps(asdict(entry)), ex=ttl)


def release(client, evaluation_pid) -> None:
    """Undo a claim whose dispatch failed: the run's task list and its registration."""
    client.delete(tasks_key(evaluation_pid), _registered_key(evaluation_pid))


def forget(client, evaluation_pid) -> None:
    """The sweep has nothing more to do for the run; its task list stays for Stop until it expires."""
    client.delete(_registered_key(evaluation_pid))


def entry_of(client, evaluation_pid) -> RegisteredRun | None:
    return _parse(client.get(_registered_key(evaluation_pid)))


def registered(client) -> list[RegisteredRun]:
    """Every registered run; an entry that cannot be read is logged and left out."""
    entries = []
    for key in client.scan_iter(match=f"{_REGISTERED}*"):
        entry = _parse(client.get(key))
        if entry is not None:
            entries.append(entry)
    return entries


def _parse(raw) -> RegisteredRun | None:
    if not raw:
        return None
    try:
        return RegisteredRun(**json.loads(raw))
    except (TypeError, ValueError) as exc:
        logger.warning(f"A registered run could not be read ({type(exc).__name__})")
        return None


def ticket_expiry(run: dict | None) -> float | None:
    """When the run's ticket expires: the number before the first dot of `<expiry>.<mac>`, None without one."""
    head, _, _ = str((run or {}).get("ticket") or "").partition(".")
    return float(head) if head.isdigit() else None
