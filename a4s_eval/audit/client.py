"""Immudb audit client for tamper-proof audit logging.

Provides a singleton ImmudbClient wrapper that manages connection
lifecycle and ensures the audit_events table exists.
"""

import threading

from immudb import ImmudbClient

from a4s_eval.utils import env
from a4s_eval.utils.logging import get_logger

logger = get_logger()

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS audit_events (
        id              INTEGER AUTO_INCREMENT,
        timestamp       TIMESTAMP,
        event_type      VARCHAR[64],
        evaluation_id   VARCHAR[64],
        task_id         VARCHAR[255],
        plugin_name     VARCHAR[255],
        status          VARCHAR[32],
        duration_ms     INTEGER,
        details         VARCHAR[4096],
        error_message   VARCHAR[2048],
        PRIMARY KEY id
    );
"""

_client: ImmudbClient | None = None
_lock = threading.Lock()
_table_created = False


class AuditConnectionError(Exception):
    """Raised when the audit system cannot connect to immudb."""


class AuditWriteError(Exception):
    """Raised when an audit event fails to be written to immudb."""


def get_audit_client() -> ImmudbClient:
    """Get or create the singleton ImmudbClient instance.

    Connects to immudb, authenticates, selects the database, and creates
    the audit_events table if it does not exist. Thread-safe.

    Raises:
        AuditConnectionError: If connection, authentication, or table
            creation fails.
    """
    global _client, _table_created

    if _client is not None and _table_created:
        return _client

    with _lock:
        if _client is not None and _table_created:
            return _client

        try:
            immudb_url = f"{env.IMMUDB_HOST}:{env.IMMUDB_PORT}"
            logger.info(f"Connecting to immudb at {immudb_url}")

            client = ImmudbClient(immudb_url)
            client.login(env.IMMUDB_USER, env.IMMUDB_PASSWORD)
            client.useDatabase(env.IMMUDB_DATABASE.encode())

            client.sqlExec(_CREATE_TABLE_SQL)
            logger.info("Audit table verified/created successfully")

            _client = client
            _table_created = True
            return _client

        except Exception as e:
            logger.error(f"Failed to connect to immudb: {e}")
            raise AuditConnectionError(
                f"Cannot establish immudb audit connection: {e}"
            ) from e


def reset_client() -> None:
    """Reset the singleton client. Used for testing and reconnection."""
    global _client, _table_created
    with _lock:
        if _client is not None:
            try:
                _client.logout()
            except Exception:
                pass
        _client = None
        _table_created = False
