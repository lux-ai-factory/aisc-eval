"""Audit event types and logging function for immudb audit trail.

Defines all auditable event types and provides log_audit_event() that
writes events to immudb with cryptographic verification. If the write
or verification fails, AuditWriteError is raised — callers must NOT
swallow this exception.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any

from immudb.datatypesv2 import PrimaryKeyIntValue

from a4s_eval.audit.client import (
    AuditConnectionError,
    AuditWriteError,
    get_audit_client,
)
from a4s_eval.utils.logging import get_logger

logger = get_logger()

# ── Task lifecycle events ──────────────────────────────────────────
EVALUATION_STARTED = "EVALUATION_STARTED"
EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
EVALUATION_FAILED = "EVALUATION_FAILED"
PLUGIN_STARTED = "PLUGIN_STARTED"
PLUGIN_COMPLETED = "PLUGIN_COMPLETED"
PLUGIN_FAILED = "PLUGIN_FAILED"
MEASUREMENTS_POSTED = "MEASUREMENTS_POSTED"

# ── API call events ────────────────────────────────────────────────
API_CALL_GET_EVALUATION = "API_CALL_GET_EVALUATION"
API_CALL_GET_DATASET = "API_CALL_GET_DATASET"
API_CALL_GET_MODEL = "API_CALL_GET_MODEL"
API_CALL_POST_MEASURES = "API_CALL_POST_MEASURES"
API_CALL_MARK_COMPLETED = "API_CALL_MARK_COMPLETED"
API_CALL_MARK_FAILED = "API_CALL_MARK_FAILED"

_INSERT_SQL = """
    INSERT INTO audit_events (
        timestamp, event_type, evaluation_id, task_id,
        plugin_name, status, duration_ms, details, error_message
    ) VALUES (
        @timestamp, @event_type, @evaluation_id, @task_id,
        @plugin_name, @status, @duration_ms, @details, @error_message
    );
"""


def _truncate(value: str | None, max_len: int) -> str:
    """Truncate a string to max_len, appending '...' if truncated."""
    if value is None:
        return ""
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def log_audit_event(
    event_type: str,
    *,
    evaluation_id: str | None = None,
    task_id: str | None = None,
    plugin_name: str | None = None,
    status: str = "success",
    duration_ms: int | None = None,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """Write a single audit event to immudb.

    Raises:
        AuditConnectionError: If immudb is unreachable.
        AuditWriteError: If the INSERT fails.
    """
    try:
        client = get_audit_client()

        details_str = json.dumps(details, default=str) if details else ""
        if len(details_str) > 4000:
            details_str = details_str[:3997] + "..."

        params = {
            "timestamp": datetime.now(timezone.utc),
            "event_type": event_type,
            "evaluation_id": str(evaluation_id) if evaluation_id else "",
            "task_id": str(task_id) if task_id else "",
            "plugin_name": plugin_name or "",
            "status": status,
            "duration_ms": duration_ms if duration_ms is not None else 0,
            "details": details_str,
            "error_message": _truncate(error_message, 2000),
        }

        result = client.sqlExec(_INSERT_SQL, params)

        # Verified read-back: cryptographically prove the row was written
        # correctly and hasn't been tampered with.
        if result.txs:
            tx = result.txs[0]
            last_pks = tx.lastInsertedPKs
            if "audit_events" in last_pks:
                inserted_id = last_pks["audit_events"].n
                verification = client.verifiableSQLGet(
                    "audit_events", [PrimaryKeyIntValue(inserted_id)]
                )
                if not verification.verified:
                    raise AuditWriteError(
                        f"Verification failed for audit event {event_type} "
                        f"(id={inserted_id})"
                    )
                logger.debug(
                    f"Audit event verified: {event_type} (id={inserted_id})"
                )
            else:
                logger.debug(f"Audit event logged: {event_type} (no PK returned)")
        else:
            logger.debug(f"Audit event logged: {event_type} (no tx info)")

    except (AuditConnectionError, AuditWriteError):
        raise
    except Exception as e:
        logger.error(f"Failed to write audit event {event_type}: {e}")
        raise AuditWriteError(
            f"Audit logging failed for event {event_type}: {e}"
        ) from e


class AuditTimer:
    """Context manager for timing operations.

    Usage:
        with AuditTimer() as timer:
            do_work()
        log_audit_event(..., duration_ms=timer.duration_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "AuditTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.monotonic()

    @property
    def duration_ms(self) -> int:
        return int((self._end - self._start) * 1000)
