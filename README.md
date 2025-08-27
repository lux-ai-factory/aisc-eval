# A4S Evaluation module

# Quickstart for Evaluation module

## How to run within local development environment

### Prerequisites
To run the local development environment, you need first to launch the services containers (database, redis, etc.). Please checkout API repo for instructions on how to do this.

### Configuration of development environment
We use `uv` as environment manager, you can configure python dependencies with the following command:

```bash
uv sync --frozen --group dev
```

<!-- We provide a pre-commit hook to automatically check and format your code before each commit. You can install the pre-commit hooks with the following command:

```bash
uv run pre-commit install
``` -->

### Launching locally the A4S Evaluation API
With the services running, you can now launch the A4S Evaluation API locally.

```bash
uv sync --frozen --group dev
bash tasks/start_api.sh
```

### Launching locally the A4S Evaluation Worker
With the services running, you can now launch the A4S Evaluation Worker locally.

```bash
uv sync --frozen --group dev
bash tasks/start_worker.sh
```

### How to manually run linter
We use ruff for linting. This step is automatically run before each commit if the pre-commit hooks are configured.

To manually run the linter, you can use the following command:

```bash
uv sync --frozen --group dev
uv run ruff check .
uv run ruff format .
```

### How to run tests
To run the unit tests, you can use the following command:

```bash
uv sync --frozen --group test
uv run pytest tests/
```
