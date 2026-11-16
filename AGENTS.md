# AGENTS.md — aisc-eval

The **evaluation worker**: a Celery worker that runs plugins and posts results. Not an API.

## Read first

The binding rules live in the parent repository, `lux-ai-factory/aisc`:
`ARCHITECTURE.md` (invariants I1–I11), `docs/refactor-plan.md` and `tests/architecture/`.

## What binds this repo

- **I1** — a run is self-contained: dispatch hands the worker every input (bytes included); the run
  path never fetches from the backend mid-run; results post at the end.
- **I2** — a plugin is untrusted catalogue code: it gets an allowlisted environment plus the run's
  own secrets (`AISC_SECRET_*`), never the worker's environment.
- **I7** — stdout, stderr, the execution log, artifacts and error messages are scrubbed before they
  leave the worker.
- **I3** — the worker reaches the engine over its HTTP API, never by importing backend internals.

## Build and test

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Do not

- add a backend fetch inside `run_plugin` (I1);
- hand a plugin a copy of `os.environ` (I2);
- write an outbound path without the scrub helper (I7).