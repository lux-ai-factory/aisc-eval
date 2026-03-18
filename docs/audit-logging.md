# Audit Logging — Architecture & Technical Reference

## 1. Why audit logging?

The evaluation service executes evaluations as distributed Celery tasks. Before this change, **all observability was ephemeral**,  logs went to stdout and could be lost, modified, or tampered with. There was no persistent record of:

- Which evaluations ran, when, and what happened
- Which plugins executed and how long they took
- Which API calls were made to the backend
- What failed and why

For compliance, traceability, and team debugging we needed an **immutable, tamper-proof audit trail** that the a4s backend can read later.

## 2. Why immudb?

[immudb](https://github.com/codenotary/immudb) is an immutable database: once data is written, it cannot be modified or deleted. Every entry is cryptographically verified. This gives us:

- **Tamper-proof storage** — no one can alter historical audit records
- **Cryptographic verification** — built-in proof that data has not been modified
- **SQL interface** — familiar `INSERT`/`SELECT` via the Python SDK, easy for the backend to query later
- **Lightweight** — single binary, Docker-ready, minimal operational overhead
- **Separation of concerns** — audit data lives outside the main application database

## 3. Architecture overview

```
                                    ┌──────────────┐
                                    │   immudb     │
                                    │  (port 3322) │
                                    └──────┬───────┘
                                           │
                          writes           │         reads (later)
                     ┌─────────────────────┤──────────────────────┐
                     │                     │                      │
              ┌──────┴───────┐             │            ┌─────────┴────────┐
              │  a4s-eval    │             │            │   a4s backend    │
              │  (this repo) │             │            │   (separate)     │
              └──────────────┘             │            └──────────────────┘
                                           │
                                    audit_events table
```

**This repo only writes.** The a4s backend will read from the same `audit_events` table later.

## 4. Module structure

```
a4s_eval/
└── audit/
    ├── client.py          # immudb connection singleton
    └── events.py          # event constants + log_audit_event()
```


### `client.py` — Connection management

**Pattern:** Thread-safe singleton with double-checked locking.

```python
get_audit_client() -> ImmudbClient   # connect, auth, create table (once)
reset_client() -> None               # teardown for tests / reconnection
```

**Why a singleton?** Celery workers are long-lived processes. Creating a new gRPC connection per audit event would add ~50ms latency each time. The singleton connects once and reuses.

**Why double-checked locking?** Celery workers can be multi-threaded. Two threads could race to initialize the client. The `threading.Lock` + double-check prevents duplicate connections without penalizing the hot path.

**Table creation** happens lazily on first `get_audit_client()` call via `CREATE TABLE IF NOT EXISTS` — idempotent, no migration tooling needed.

**Custom exceptions:**
- `AuditConnectionError` — immudb unreachable or auth failed
- `AuditWriteError` — INSERT failed

These are distinct so callers (and logs) clearly indicate whether the problem is connectivity or data.

### `events.py` — Event logging

**Core function:**

```python
log_audit_event(
    event_type: str,              # one of the constants below
    *,                            # keyword-only after this
    evaluation_id: str | None,
    task_id: str | None,
    plugin_name: str | None,
    status: str = "success",
    duration_ms: int | None,
    details: dict | None,         # serialized to JSON string
    error_message: str | None,
)
```

**Design decisions:**

- **Keyword-only arguments** (`*` separator) — prevents positional mistakes with 8 optional params
- **`details` as `dict` -> JSON string** — flexible schema for different event types, stored as `VARCHAR[4096]` for broad compatibility
- **`_truncate()` helper** — immudb `VARCHAR` columns have fixed max sizes. Truncation prevents INSERT failures on long error messages or detail payloads
- **Exception behavior** — always raises on failure. **Audit failures must fail the task**. This means `log_audit_event()` must never be wrapped in a silent `try/except`
- **Verified writes** — after each INSERT, the function reads back the row via `verifiableSQLGet()` to get cryptographic proof that the write was stored correctly and hasn't been tampered with. If verification fails, `AuditWriteError` is raised. This adds ~10-20ms per write but guarantees end-to-end integrity

**`AuditTimer` context manager:**

```python
with AuditTimer() as timer:
    do_work()
log_audit_event(..., duration_ms=timer.duration_ms)
```

Uses `time.monotonic()` for reliable duration measurement.

## 5. Audit events catalog

### Task lifecycle events (in `celery_tasks.py`)

| Constant | When | Key fields |
|---|---|---|
| `EVALUATION_STARTED` | `run_evaluation` begins | evaluation_id, task_id |
| `EVALUATION_COMPLETED` | `finalize_evaluation` succeeds | evaluation_id, task_id |
| `EVALUATION_FAILED` | `handle_error` called | evaluation_id, task_id, error_message, traceback in details |
| `PLUGIN_STARTED` | `run_plugin` begins | plugin_name, task_id |
| `PLUGIN_COMPLETED` | `run_plugin` returns | plugin_name, task_id, duration_ms, measurement_count in details |
| `PLUGIN_FAILED` | `run_plugin` raises | plugin_name, task_id, error_message |
| `MEASUREMENTS_POSTED` | `post_measurements` succeeds | evaluation_id, task_id, measurement_count in details |

### API call events (in `api_client.py`)

| Constant | Function | Key fields |
|---|---|---|
| `API_CALL_GET_EVALUATION` | `get_evaluation_request()` | evaluation_id, duration_ms |
| `API_CALL_GET_DATASET` | `get_dataset_file_content()` | duration_ms, file_name + content_length in details |
| `API_CALL_GET_MODEL` | `get_model_file_content()` | duration_ms, file_name + content_length in details |
| `API_CALL_POST_MEASURES` | `post_measures()` | evaluation_id, duration_ms, metric_count + status_code in details |
| `API_CALL_MARK_COMPLETED` | `mark_completed()` | evaluation_id, duration_ms, status_code in details |
| `API_CALL_MARK_FAILED` | `mark_failed()` | evaluation_id, duration_ms |

**Note:** `get_evaluation()` delegates to `get_evaluation_request()` — audit happens only in the latter to avoid double-logging.

## 6. Database schema

```sql
CREATE TABLE IF NOT EXISTS audit_events (
    id              INTEGER AUTO_INCREMENT,
    timestamp       TIMESTAMP,
    event_type      VARCHAR[64],
    evaluation_id   VARCHAR[64],
    task_id         VARCHAR[255],
    plugin_name     VARCHAR[255],
    status          VARCHAR[32],       -- "success" | "failure" | "error"
    duration_ms     INTEGER,
    details         VARCHAR[4096],     -- JSON string
    error_message   VARCHAR[2048],
    PRIMARY KEY id
);
```

**Why `VARCHAR` instead of `JSON` type?** Broader compatibility for when the a4s backend queries this table — `VARCHAR` works universally while `JSON[n]` is immudb-specific.

**Why fixed sizes?** immudb requires explicit `VARCHAR` sizes. The `_truncate()` helper ensures values never exceed column limits.

## 7. Changes to existing files

### `celery_tasks.py`

- **3 tasks became `bind=True`**: `post_measurements`, `finalize_evaluation`, `handle_error` — previously unbound, now need `self.request.id` for the task_id audit field
- **`run_plugin` wrapped in try/except**: to log `PLUGIN_FAILED` before re-raising the exception. The original code had no try/except — the exception just propagated to Celery's error handler
- **All audit calls are at the application level**: no Celery signals or decorators — explicit `log_audit_event()` calls at precise locations for clarity

**Why explicit calls instead of Celery signals?** Signals (`task_prerun`, `task_postrun`) fire for all tasks generically. Explicit calls let us include context-specific data (plugin_name, evaluation_id, measurement_count) that signals don't have access to.

### `api_client.py`

- Each HTTP function wrapped with `AuditTimer` around the `requests` call
- `log_audit_event()` called after the HTTP response is received
- If the HTTP call itself fails (connection error), the exception propagates before audit — this is acceptable because the Celery task will fail and `handle_error` will log `EVALUATION_FAILED`

### `utils/env.py`

5 new environment variables following the existing `os.getenv("VAR", "default")` pattern:

| Variable | Default | Notes |
|---|---|---|
| `IMMUDB_HOST` | `"immudb"` | Docker service name convention (like `"rabbitmq"`, `"redis"`) |
| `IMMUDB_PORT` | `3322` | immudb default gRPC port |
| `IMMUDB_USER` | `"immudb"` | immudb factory default |
| `IMMUDB_PASSWORD` | `"immudb"` | immudb factory default |
| `IMMUDB_DATABASE` | `"defaultdb"` | immudb factory default |

### `pyproject.toml`

Added `"immudb-py>=1.4.0"` to dependencies. Installed version: `1.5.0`.

## 8. Failure behavior

**Audit failures fail the task.** 

If immudb is down or an INSERT fails:
1. `log_audit_event()` raises `AuditConnectionError` or `AuditWriteError`
2. The exception propagates up through the Celery task
3. Celery marks the task as failed
4. `handle_error` is invoked (which also tries to audit — if immudb is still down, this will also fail, and the evaluation gets marked as failed via the API)

**Trade-off:** This means immudb availability is required for evaluations to run. If this becomes a problem, the team can revisit and switch to fire-and-forget mode.

## 9. Configuration for deployment

Add immudb to your Docker Compose:

```yaml
services:
  immudb:
    image: codenotary/immudb:latest
    ports:
      - "3322:3322"   # gRPC
      - "8080:8080"   # web console (optional)
    volumes:
      - immudb_data:/var/lib/immudb

volumes:
  immudb_data:
```

Set environment variables on the eval worker:

```yaml
  eval-worker:
    environment:
      IMMUDB_HOST: immudb
      IMMUDB_PORT: 3322
      IMMUDB_USER: immudb
      IMMUDB_PASSWORD: <change-in-production>
      IMMUDB_DATABASE: defaultdb
```

## 10. Testing

```bash
uv run pytest tests/test_audit.py -v
```

All tests mock `ImmudbClient` — no running immudb instance needed.

| Test | What it verifies |
|---|---|
| `test_get_audit_client_connects_and_creates_table` | Login, DB selection, table creation on first call |
| `test_get_audit_client_returns_singleton` | Second call reuses same client |
| `test_get_audit_client_raises_on_connection_failure` | `AuditConnectionError` on connection failure |
| `test_reset_client_allows_reconnection` | Fresh connection after `reset_client()` |
| `test_log_audit_event_inserts_and_verifies` | Correct SQL params + `verifiableSQLGet` called after insert |
| `test_log_audit_event_raises_on_verification_failure` | `AuditWriteError` when verification returns `verified=False` |
| `test_log_audit_event_raises_on_write_failure` | `AuditWriteError` on INSERT failure |
| `test_log_audit_event_propagates_connection_error` | `AuditConnectionError` propagated |
| `test_log_audit_event_serializes_details_to_json` | `details` dict -> JSON string |
| `test_log_audit_event_defaults` | Empty strings / 0 for omitted fields |
| `test_log_audit_event_truncates_large_details` | Details > 4000 chars truncated |
| `test_log_audit_event_graceful_when_no_tx_info` | No verification attempted if sqlExec returns empty txs |
| `test_timer_measures_duration` | `AuditTimer` measures ~50ms sleep |
| `test_timer_zero_duration` | Timer works with instant operations |
| `test_short_string_unchanged` / `test_long_string_truncated` / `test_none_returns_empty` / `test_exact_length_unchanged` | `_truncate()` edge cases |

## 11. Querying audit logs (for the backend team)

The a4s backend can query audit data via the immudb Python SDK:

```python
from immudb import ImmudbClient

client = ImmudbClient("immudb:3322")
client.login("immudb", "immudb")
client.useDatabase(b"defaultdb")

# All events for an evaluation
results = client.sqlQuery(
    "SELECT * FROM audit_events WHERE evaluation_id = @eval_id ORDER BY id",
    params={"eval_id": "your-evaluation-uuid"}
)

# All failures in the last 24h
results = client.sqlQuery(
    "SELECT * FROM audit_events WHERE status = 'failure' ORDER BY id DESC"
)

# Plugin performance
results = client.sqlQuery(
    "SELECT plugin_name, duration_ms FROM audit_events WHERE event_type = 'PLUGIN_COMPLETED'"
)
```

## 12. Future considerations

- **Connection health checks**: The singleton client may lose its gRPC connection in long-lived workers. A periodic ping or retry-on-failure wrapper could be added.
- **Fire-and-forget mode**: If immudb availability becomes a concern, add an `IMMUDB_FAIL_OPEN` env var to switch to logging a warning instead of raising.
- **Structured indexing**: immudb supports secondary indexes — if query performance matters, add indexes on `evaluation_id` and `event_type`.
- **Verified reads on the backend**: **Implemented.** The a4s-backend now has audit endpoints at `/api/v1/audit/` with verified reads via `verifiableSQLGet()`. See `a4s-backend/a4s_backend/services/audit_service.py` and `a4s-backend/a4s_backend/routers/audit.py`. The end-to-end chain is complete: verified writes (eval) -> immutable storage -> verified reads (backend).
