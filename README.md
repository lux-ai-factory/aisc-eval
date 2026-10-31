# AISC Evaluation module

`aisc-eval` is the Celery worker of **the engine** of AISC (the AI Assessment Sandbox Configurator). It runs the
tests of step 4 (Collect evidence) as the compose service `aisc-eval-worker`, next to `aisc-backend` (the
engine's API) and `aisc-webapp` (its pages).

## How it works

When someone runs a test, `aisc-backend` records the run and queues it on the broker (RabbitMQ). The worker
takes the task, installs the test's package in a fresh virtual environment (from the stack's package index,
devpi, or, for a local plugin, from its folder), runs it in its own process, and posts the status, the
measurements and the files the test writes to the backend's internal API (`/api/v1/internal`). The backend keeps
the measurements in the engine's schema of the project's database and the files in MinIO. The worker has no
database and no object store settings of its own.

What the test assesses is reached through the platform, never with its secrets in the worker or the plugin:

- **What is assessed** (the system or a component of the AI card): the worker asks the platform for a run pass
  that opens only the targets and connections the run names, and gives it to the plugin as `PLATFORM_RUN_PASS`.
  With it the plugin gets a run key and calls one of the platform's doors, which adds the connection's secrets
  and forwards the call. The worker revokes the pass when the plugin has ended. A run on a local model file gets
  no pass; a run that names both a local model file and an AI card target is refused.
- **Helper models** (judge, attacker, grader, embedder, generator): the worker asks the platform for each helper
  model's endpoint and key just before the plugin starts, and gives the key to the plugin as a secret environment
  variable. The plugin calls the helper model directly.

There is no resume. Tasks are acknowledged only when they end (`task_acks_late`), so if the worker stops in the
middle, the broker hands each unfinished task (an install, or one test) to a worker again, and it starts over
from the beginning. When a test fails while it runs, the run's other tests that are still running are stopped.

For the whole picture, see `docs/architecture.md` ("Running tests") and `docs/appendices/4-collect-evidence.md`
("Run tests") in the aisc repository; how a run pass and run key work is in `docs/appendices/manage.md`.

## In the AISC stack

From the root of the [aisc](https://github.com/lux-ai-factory/aisc) repository, run
`uv run scripts/aisc.py start` (see that README). It makes the secrets if they are missing and starts the stack.
This repo is built into the image `aisc-eval`, used by two services of `docker-compose.development.yml`:

- **`aisc-eval-worker`**: `celery -A aisc_eval.celery_worker worker`, after installing the
  `shared/plugin-manager` and `shared/plugin-interface` packages of the aisc repo in editable mode.
- **`aisc-eval-flower`**: Flower, the Celery monitor, behind the gateway at `/flower/`. It starts only with the
  compose profile `inspect`; its link is under Manage, Background tasks, for admins.

Both are started with `AISC_DEPLOYMENT: configurator`. To run the engine on its own, the aisc
repo has `docker-compose.engine-standalone.yml`, where the variable is unset.

## Two deployment modes

`AISC_DEPLOYMENT`, read once in `aisc_eval/deployment.py`:

- `standalone` (default, variable unset): calls to the backend carry only `X-Internal-Secret`.
- `configurator`: every internal call also carries `X-AISC-Project`, `X-AISC-Evaluation` and
  `X-AISC-Run` (the run ticket), which the backend checks before it opens the project's database. The backend
  puts these values in the Celery message headers of `run_evaluation` (`aisc_eval/run_context.py`), and every
  task published during the run inherits them.

Any other value stops the worker.

## Configuration

Read from the environment (`aisc_eval/utils/env.py`, `aisc_eval/deployment.py`, `aisc_eval/service/`):

| Variable | Meaning | Default | In the AISC stack |
|---|---|---|---|
| `AISC_DEPLOYMENT` | `standalone` or `configurator` | `standalone` | `configurator` |
| `API_URL` | address of the backend | `http://backend` | `http://aisc-backend:8000` |
| `API_PREFIX` | path of the backend's internal API | `/api/v1/internal` | same |
| `INTERNAL_API_KEY` | shared secret sent as `X-Internal-Secret`; the worker (not Flower) refuses to start without it | none (required) | from the secrets |
| `DJANGO_SECRET_KEY` | decrypts the stored plugin settings the backend hands over | none | the backend's |
| `CELERY_BROKER_URL` | the broker; Celery reads it directly. Unset, the URL is built from `MQ_HOST`, `MQ_PORT`, `MQ_USERNAME`, `MQ_PASSWORD`, `MQ_USE_SSL` | `amqp://rabbitmq:5672` | `rabbitmq` with its account |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_USE_SSL` | the result store (the worker does not read `REDIS_BACKEND_URL`) | `redis`, `6379`, no login | defaults |
| `PLUGIN_PATH` | folder of local plugins | empty | `./local_plugins/`, mounted at `/app/plugins` |
| `PACKAGE_REGISTRY_URL`, `PACKAGE_REGISTRY_INDEX`, `PACKAGE_REGISTRY_USER`, `PACKAGE_REGISTRY_PASSWORD` | package index tests are installed from | empty | `http://devpi:3141`, `root/public`, no login |
| `PLATFORM_URL` | the platform, for run passes and helper models | empty | `http://platform:8000` |
| `PLATFORM_CONNECTIONS_TOKEN` | the worker's service token to the platform; never given to a plugin | empty | from the secrets |
| `PLUGIN_USER`, `PLUGIN_INSTALL_USER` | the users a root worker runs and installs plugins as | `aisc-plugin`, `aisc-install` | defaults |

A plugin does not get the worker's environment. It runs with an allowlist (`PLUGIN_ENV_NAMES` in
`aisc_eval/celery_tasks.py`): `PATH`, `HOME`, `LANG`, `LC_ALL`, `TZ`, `TMPDIR`, the CA bundle and proxy
variables, `PLATFORM_URL`, `CONNECTIONS_ALLOWED_HOSTS`, `AISC_TARGET_BASE_URL` and `OPENAI_BASE_URL`; plus its
project's secrets and its helper models' keys as `AISC_SECRET_*`, and its run pass as `PLATFORM_RUN_PASS`. When
no `API_KEY_OPENAI` is set it gets a placeholder, for plugins that build an OpenAI client even when they never
call one. The broker URL, `INTERNAL_API_KEY`, `DJANGO_SECRET_KEY`, the package index login and
`PLATFORM_CONNECTIONS_TOKEN` stay with the worker. A plugin that needs another variable needs it added to that
list.

## How to run within local development environment

### Prerequisites
Start the infrastructure (RabbitMQ and Redis) first:

1. Set `AISC_ADMIN_PORTS=on`, in the environment or in `env.local` in the aisc folder. The infrastructure
   ports (5672 for RabbitMQ, 6379 for Redis, and the others) are then published, on `127.0.0.1` only.
2. In the aisc folder, run `uv run scripts/aisc.py secrets`. It prints this install's exact compose
   command, which ends in `up -d`.
3. Append the service names to that command: `rabbitmq redis`.

`uv run scripts/aisc.py start` instead brings up everything, the eval containers included. Use it to run
this module inside the full stack.

The backend must also be running for the worker to report to.

### Configuration of development environment
We use `uv` as environment manager (Python 3.12). Install the dependencies with:

```bash
uv sync --frozen --group dev
```

The lock takes `aisc-plugin-interface` 0.4.0, `aisc-plugin-manager` 0.3.0+aisc.1 and `aisc-identity` 0.1.0
from the wheels under `vendor/`, built from the aisc checkout by the root's `scripts/vendor-plugin-interface.sh`,
which refreshes them when one of those libraries changes. To work on the interface or the plugin manager
themselves, install them from the aisc repo in editable mode:

```bash
uv pip install --no-deps -e ../../shared/plugin-manager -e ../../shared/plugin-interface
```

### Launching locally the AISC Evaluation Worker

```bash
uv sync --frozen --group dev
bash tasks/start_worker.sh
```

The script loads `.env.dev` and then `.env.dev-local` if they exist (`env.development` is an
example of the variables), and starts the worker with `--pool=solo --concurrency=1`.

### How to manually run linter
We use ruff for linting. This step runs before each commit if the pre-commit hooks are configured.

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
No test reads the Git LFS files under `tests/data`, so the suite passes without `git lfs pull`.

### How to log and customise logs

We use a single package-wide logger named as the main package: `aisc_eval`.

To log, import the logger and use it:

```python
from aisc_eval.utils import get_logger

get_logger().info("This is an info message")
get_logger().error("This is an error message")
```

You can customise the logging configuration in `./config/logging.yaml`: the
level of each logger in its `loggers` section, and the message format in its `formatters`
section. See [official Python documentation on LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes) for a full list of available fields.

Please do not push your local changes, except if necessary. For instance, DEBUG log in `logging.yaml` should not be pushed.

## Changes on pilot

`pilot` is the branch AISC uses. It is not held to
`origin/master`. Compared with it:

- **Deployment mode and run headers**: `aisc_eval/deployment.py`, `aisc_eval/run_context.py`, and
  `aisc_eval/service/api_client.py`, which builds the headers of each internal call at call time.
  Task signatures are as on master.
- **Plugin installs**: plugins are installed online from the package index in `run_plugin` (no `--offline`),
  and the index is always passed to `uv pip install` instead of being probed first. A local plugin is
  installed from a copy of its folder.
- **Plugin environment**: an allowlist instead of a copy of the worker's environment (see
  Configuration), with a placeholder `API_KEY_OPENAI` when none is set, for plugins that build an
  OpenAI client even when a local model is used.
- **Run passes and helper models**: `aisc_eval/service/run_pass.py` mints and revokes the run's pass to the
  platform; `aisc_eval/service/helpers.py` resolves the run's helper models (`helper:<platform pid>/<name>`) into
  an endpoint, the model picked for the run and a key in the plugin's secret environment.
- **Local model files**: a run on a local model file records the file's name and SHA-256 on its input, and opens
  nothing on the platform.
- **Measures the engine can store**: texts lose NUL characters and are cut to 255 characters; a
  cut metric name ends with a hash of the whole name, so two long names stay two metrics.
- **Secrets kept out of results**: installer messages, failure texts and artifacts are scrubbed of known keys
  before they reach the backend or a log.
- **Start-up check**: the worker (not Flower) stops when `INTERNAL_API_KEY` is unset.
- **Task time limits**: 400 minutes soft, 405 hard (`aisc_eval/celery_app.py`), for long
  first runs that download a model.
- **Users**: a root worker installs plugins as `aisc-install` (`PLUGIN_INSTALL_USER`), with an
  environment and a uv cache of its own, and runs them as `aisc-plugin` (`PLUGIN_USER`), which can
  run the installed venv but not change it, and cannot read the worker's environment or `/app`
  (`aisc_eval/plugin_process.py`).
- **Tests** under `tests/` for each of these, for example `test_deployment_mode.py`, `test_run_context.py`,
  `test_run_context_broker.py`, `test_run_ticket.py`, `test_run_pass_project.py`, `test_helper_resolution.py`,
  `test_plugin_environment.py`, `test_local_model_runs.py`, `test_plugin_process.py`, `test_plugin_install.py`
  and `services/test_post_measures_fit.py`.

## Contributing

We welcome community contributions. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

By submitting contributions, you agree to the contributor license agreement
([individuals](<AISC ICLA (Individuals).txt>), [entities](<AISC CCLA (Entities).txt>)) and license
your work under [Apache 2.0](LICENSE.md).

---

## License

This project is licensed under the [Apache License 2.0](LICENSE.md).
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).
