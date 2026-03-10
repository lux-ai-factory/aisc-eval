"""Tests for immudb audit logging module."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from a4s_eval.audit.client import (
    AuditConnectionError,
    AuditWriteError,
    get_audit_client,
    reset_client,
    _CREATE_TABLE_SQL,
)
from a4s_eval.audit.events import (
    log_audit_event,
    AuditTimer,
    EVALUATION_STARTED,
    PLUGIN_COMPLETED,
    _truncate,
)


def _mock_sqlexec_result(inserted_id: int = 1) -> MagicMock:
    """Build a mock sqlExec return value with transaction metadata."""
    pk_value = MagicMock()
    pk_value.n = inserted_id

    tx = MagicMock()
    tx.lastInsertedPKs = {"audit_events": pk_value}

    result = MagicMock()
    result.txs = [tx]
    return result


def _mock_verified_entry(verified: bool = True) -> MagicMock:
    """Build a mock verifiableSQLGet return value."""
    entry = MagicMock()
    entry.verified = verified
    return entry


class TestAuditClient:
    """Tests for the immudb client singleton."""

    def setup_method(self):
        reset_client()

    def teardown_method(self):
        reset_client()

    @patch("a4s_eval.audit.client.ImmudbClient")
    def test_get_audit_client_connects_and_creates_table(self, mock_immudb_cls):
        mock_client = MagicMock()
        mock_immudb_cls.return_value = mock_client

        client = get_audit_client()

        assert client is mock_client
        mock_client.login.assert_called_once()
        mock_client.useDatabase.assert_called_once()
        mock_client.sqlExec.assert_called_once_with(_CREATE_TABLE_SQL)

    @patch("a4s_eval.audit.client.ImmudbClient")
    def test_get_audit_client_returns_singleton(self, mock_immudb_cls):
        mock_client = MagicMock()
        mock_immudb_cls.return_value = mock_client

        client1 = get_audit_client()
        client2 = get_audit_client()

        assert client1 is client2
        assert mock_immudb_cls.call_count == 1

    @patch("a4s_eval.audit.client.ImmudbClient")
    def test_get_audit_client_raises_on_connection_failure(self, mock_immudb_cls):
        mock_immudb_cls.side_effect = ConnectionError("refused")

        with pytest.raises(AuditConnectionError):
            get_audit_client()

    @patch("a4s_eval.audit.client.ImmudbClient")
    def test_reset_client_allows_reconnection(self, mock_immudb_cls):
        mock_client = MagicMock()
        mock_immudb_cls.return_value = mock_client

        client1 = get_audit_client()
        reset_client()
        client2 = get_audit_client()

        assert mock_immudb_cls.call_count == 2


class TestLogAuditEvent:
    """Tests for the log_audit_event function."""

    def setup_method(self):
        reset_client()

    def teardown_method(self):
        reset_client()

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_inserts_and_verifies(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.sqlExec.return_value = _mock_sqlexec_result(inserted_id=42)
        mock_client.verifiableSQLGet.return_value = _mock_verified_entry(verified=True)
        mock_get_client.return_value = mock_client

        log_audit_event(
            EVALUATION_STARTED,
            evaluation_id="abc-123",
            task_id="task-456",
            status="success",
        )

        mock_client.sqlExec.assert_called_once()
        params = mock_client.sqlExec.call_args[0][1]
        assert params["event_type"] == "EVALUATION_STARTED"
        assert params["evaluation_id"] == "abc-123"
        assert params["task_id"] == "task-456"
        assert params["status"] == "success"

        # Verified read-back was called with the inserted PK
        mock_client.verifiableSQLGet.assert_called_once()
        call_args = mock_client.verifiableSQLGet.call_args
        assert call_args[0][0] == "audit_events"

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_raises_on_verification_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.sqlExec.return_value = _mock_sqlexec_result(inserted_id=1)
        mock_client.verifiableSQLGet.return_value = _mock_verified_entry(verified=False)
        mock_get_client.return_value = mock_client

        with pytest.raises(AuditWriteError, match="Verification failed"):
            log_audit_event(EVALUATION_STARTED)

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_raises_on_write_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.sqlExec.side_effect = Exception("write failed")
        mock_get_client.return_value = mock_client

        with pytest.raises(AuditWriteError):
            log_audit_event(EVALUATION_STARTED)

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_propagates_connection_error(self, mock_get_client):
        mock_get_client.side_effect = AuditConnectionError("no immudb")

        with pytest.raises(AuditConnectionError):
            log_audit_event(EVALUATION_STARTED)

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_serializes_details_to_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.sqlExec.return_value = _mock_sqlexec_result()
        mock_client.verifiableSQLGet.return_value = _mock_verified_entry()
        mock_get_client.return_value = mock_client

        log_audit_event(
            PLUGIN_COMPLETED,
            details={"metric_count": 5, "plugin": "drift"},
        )

        params = mock_client.sqlExec.call_args[0][1]
        parsed = json.loads(params["details"])
        assert parsed["metric_count"] == 5
        assert parsed["plugin"] == "drift"

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_defaults(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.sqlExec.return_value = _mock_sqlexec_result()
        mock_client.verifiableSQLGet.return_value = _mock_verified_entry()
        mock_get_client.return_value = mock_client

        log_audit_event(EVALUATION_STARTED)

        params = mock_client.sqlExec.call_args[0][1]
        assert params["evaluation_id"] == ""
        assert params["task_id"] == ""
        assert params["plugin_name"] == ""
        assert params["status"] == "success"
        assert params["duration_ms"] == 0
        assert params["details"] == ""
        assert params["error_message"] == ""

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_truncates_large_details(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.sqlExec.return_value = _mock_sqlexec_result()
        mock_client.verifiableSQLGet.return_value = _mock_verified_entry()
        mock_get_client.return_value = mock_client

        large_details = {"data": "x" * 5000}
        log_audit_event(EVALUATION_STARTED, details=large_details)

        params = mock_client.sqlExec.call_args[0][1]
        assert len(params["details"]) <= 4000
        assert params["details"].endswith("...")

    @patch("a4s_eval.audit.events.get_audit_client")
    def test_log_audit_event_graceful_when_no_tx_info(self, mock_get_client):
        """If sqlExec returns no transaction info, event is still logged."""
        mock_client = MagicMock()
        result = MagicMock()
        result.txs = []
        mock_client.sqlExec.return_value = result
        mock_get_client.return_value = mock_client

        # Should not raise — just skips verification
        log_audit_event(EVALUATION_STARTED)
        mock_client.verifiableSQLGet.assert_not_called()


class TestAuditTimer:
    """Tests for the AuditTimer context manager."""

    def test_timer_measures_duration(self):
        with AuditTimer() as timer:
            time.sleep(0.05)

        assert timer.duration_ms >= 40

    def test_timer_zero_duration(self):
        with AuditTimer() as timer:
            pass

        assert timer.duration_ms >= 0


class TestTruncate:
    """Tests for the _truncate helper."""

    def test_short_string_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_string_truncated(self):
        result = _truncate("a" * 200, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_none_returns_empty(self):
        assert _truncate(None, 100) == ""

    def test_exact_length_unchanged(self):
        assert _truncate("abcde", 5) == "abcde"
