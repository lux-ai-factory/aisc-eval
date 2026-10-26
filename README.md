# AISC Evaluation module

`aisc-eval` is the Celery worker of the AISC execution engine (step 4 of the AI Assessment
Sandbox Configurator). It takes evaluation tasks from the broker (RabbitMQ), installs each
plugin into its own virtual environment from the stack's package index (devpi), runs it, and
reports status and results back to the backend's internal API (`/api/v1/internal`). It has no
database of its own.

## In the AISC stack

From the root of the [aisc](https://github.com/lux-ai-factory/aisc) repository, follow its
README (`uv run scripts/aisc.py start` makes the secrets if they are missing
and starts the stack). This repo is built into the image `aisc-eval`, used by two
services of `docker-compose.development.yml`:

- **`aisc-eval-worker`**: `celery -A aisc_eval.celery_worker worker`, after installing the
  `shared/plugin-manager` and `shared/plugin-interface` packages of the aisc repo.
- **`aisc-eval-flower`**: Flower, the Celery monitor, behind the gateway at `/flower/`.

Both are started with `AISC_DEPLOYMENT: configurator`. To run the engine on its own, the aisc
repo has `docker-compose.engine-standalone.yml`, where the variable is unset.

## Two deployment modes

`AISC_DEPLOYMENT`, read once in `aisc_eval/deployment.py`:

- `standalone` (default, variable unset): calls to the backend carry only `X-Internal-Secret`.
- `configurator`: every internal call also carries `X-AISC-Project`, `X-AISC-Evaluation` and
  `X-AISC-Run`, which the backend checks before it opens the project's database. The backend
  puts these values in the Celery message headers of `run_evaluation`
  (`aisc_eval/run_context.py`), and every task published during the run inherits them.

Any other value stops the worker.

## Configuration

Read from the environment (`aisc_eval/utils/env.py`, `aisc_eval/deployment.py`):

| Variable | Meaning | Default |
|---|---|---|
| `AISC_DEPLOYMENT` | `standalone` or `configurator` | `standalone` |
| `API_URL` | address of the backend | `http://backend` |
| `API_PREFIX` | path of the backend's internal API | `/api/v1/internal` |
| `INTERNAL_API_KEY` | shared secret sent as `X-Internal-Secret`; the worker refuses to start without it | none (required) |
| `CELERY_BROKER_URL`, `REDIS_BACKEND_URL` | broker and result backend | see `utils/env.py` |
| `PLUGIN_PATH` | folder with local plugins | empty |
| `PACKAGE_REGISTRY_URL`, `PACKAGE_REGISTRY_INDEX`, `PACKAGE_REGISTRY_USER`, `PACKAGE_REGISTRY_PASSWORD` | package index plugins are installed from | empty |
| `CACHE_DIR` | cache of plugin environments | `/tmp/cache` |

A plugin does not get the worker's environment. It runs with an allowlist
(`PLUGIN_ENV_NAMES` in `aisc_eval/celery_tasks.py`): `PATH`, `HOME`, locale, proxy and CA
variables, `PLATFORM_URL` (the connection resolver), `CONNECTIONS_ALLOWED_HOSTS`,
`AISC_TARGET_*`, the OpenAI variables, and its project's secrets as `AISC_SECRET_*`. The broker
and Redis URLs, `INTERNAL_API_KEY`, `DJANGO_SECRET_KEY`, the package registry login and
`PLATFORM_CONNECTIONS_TOKEN` stay with the worker. When a run's resource inputs name targets or
connections, the worker mints a pass for exactly those from the platform
(`aisc_eval/service/run_pass.py`), gives it to the plugin as `PLATFORM_RUN_PASS` and revokes it
when the plugin has ended. A plugin that needs another variable needs it
added to that list.

## How to run within local development environment

### Prerequisites
Start the infrastructure (RabbitMQ and Redis) first:

1. Set `AISC_ADMIN_PORTS=on`, in the environment or in `env.local` in the aisc folder. The infrastructure
   ports (5672 for RabbitMQ, 6379 for Redis, and the others) are published, on `127.0.0.1` only, when it is set.
2. In the aisc folder, run `uv run scripts/aisc.py secrets`. It prints this install's exact compose
   command, which ends in `up -d`.
3. Append the service names to that command: `rabbitmq redis`.

`uv run scripts/aisc.py start` instead brings up everything, the eval containers included. Use it to run
this module inside the full stack.

The backend must also be running for the worker to report to.

### Configuration of development environment
We use `uv` as environment manager (Python 3.12), you can configure python dependencies with the following command:

```bash
uv sync --frozen --group dev
```

To use the local `plugin-manager` and `plugin-interface` from the aisc repo:

```bash
uv pip install --no-deps -e ../../shared/plugin-manager -e ../../shared/plugin-interface
```

The lock already takes `aisc-plugin-interface` 0.4.0, the plugin-manager and `aisc-identity` from wheels under `vendor/`, built from this checkout and refreshed by the root's `scripts/vendor-plugin-interface.sh` when one of them changes; the editable install is only for working on the interface or the plugin-manager itself.

### Launching locally the AISC Evaluation Worker

```bash
uv sync --frozen --group dev
bash tasks/start_worker.sh
```

The script loads `.env.dev` and then `.env.dev-local` if they exist (`env.development` is an
example of the variables), and starts the worker with `--pool=solo --concurrency=1`.

### How to manually run linter
We use ruff for linting. This step is automatically run before each commit if the pre-commit hooks are configured.

```bash
uv sync --frozen --group dev
uv run ruff check .
uv run ruff format .
```

### How to run tests

```bash
uv sync --frozen --group test
uv run pytest tests/
```

The tests need no database and no running stack. `tests/test_run_context_broker.py` skips
unless `AISC_TEST_BROKER_URL` and `AISC_TEST_RESULT_BACKEND` point at a throwaway broker and
Redis; never point them at a running stack. For example:

```bash
docker run -d --rm --name eval-test-rabbit -p 127.0.0.1:35672:5672 rabbitmq:3-management
docker run -d --rm --name eval-test-redis -p 127.0.0.1:36379:6379 redis:7-alpine
AISC_TEST_BROKER_URL=amqp://guest:guest@127.0.0.1:35672// \
AISC_TEST_RESULT_BACKEND=redis://127.0.0.1:36379/0 uv run pytest tests/
```

Run the suite once more with `AISC_DEPLOYMENT=configurator`: some tests only run in one mode.
`uv run pytest tests/` runs the whole suite as it is; no test reads the Git LFS files under
`tests/data`, so it passes without `git lfs pull`.

### How to log and customise logs

We use a single package-wide logger named as the main package: `aisc_eval`.

To log, simply import the logger and use it:

```python
from aisc_eval.utils import get_logger

get_logger().info("This is an info message")
get_logger().error("This is an error message")
```

You can customize the logging configuration by modifying the `./config/logging.yaml` file: the
level of each logger in its `loggers` section, and the message format in its `formatters`
section. See [official Python documentation on LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes) for a full list of available fields.

Please do not push your local changes, except if necessary. For instance, DEBUG log in `logging.yaml` should not be pushed.

## Changes on feat/unified-modules

`feat/unified-modules` is the branch the Configurator uses. It is not held to
`origin/master`. Compared with it:

- **Deployment mode and run headers**: `aisc_eval/deployment.py`, `aisc_eval/run_context.py`, and
  `aisc_eval/service/api_client.py`, which builds the headers of each internal call at call time.
  Task signatures are as on master.
- **Plugin installs** (from the `feat/dev-catalogue-staging` branch): plugins are installed online
  from the package index in `run_plugin` (no `--offline`), and the index is always passed to
  `uv pip install` instead of being probed first.
- **Plugin environment**: an allowlist instead of a copy of the worker's environment (see
  Configuration), with a placeholder `API_KEY_OPENAI` when none is set, for plugins that build an
  OpenAI client even when a local model is used.
- **Measures the engine can store**: texts lose NUL characters and are cut to 255 characters; a
  cut metric name ends with a hash of the whole name, so two long names stay two metrics.
- **Start-up check**: the worker (not flower) stops when `INTERNAL_API_KEY` is unset.
- **Task time limits** raised to 400 minutes soft, 405 hard (`aisc_eval/celery_app.py`), for long
  first runs that download a model.
- **Users**: a root worker installs plugins as `aisc-install` (`PLUGIN_INSTALL_USER`), with an
  environment and a uv cache of its own, and runs them as `aisc-plugin` (`PLUGIN_USER`), which can
  run the installed venv but not change it, and cannot read the worker's environment or `/app`
  (`aisc_eval/plugin_process.py`).
- **Tests**: `tests/test_deployment_mode.py`, `tests/test_run_context.py`,
  `tests/test_run_context_broker.py`, `tests/test_run_ticket.py`,
  `tests/test_plugin_environment.py`, `tests/services/test_post_measures_fit.py`,
  `tests/test_plugin_process.py`, `tests/test_plugin_install.py`.

## Contributing

We welcome community contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

By submitting contributions, you agree to the contributor license agreement
([individuals](<AISC ICLA (Individuals).txt>), [entities](<AISC CCLA (Entities).txt>)) and license
your work under [Apache 2.0](LICENSE.md).

---

## License

This project is licensed under the [Apache License 2.0](LICENSE.md).
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).
