import fcntl
import hashlib
import re
import json
import os
import shutil
import secrets
import subprocess
import tempfile
import traceback as tracebacks

import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path

import requests
from celery import group, chain

from aisc_eval import egress, plugin_process, run_registry
from aisc_eval import run_context  # also registers the run header handlers
from aisc_eval.scrub import REDACTED, known_secrets, scrub_text, without_known_bytes
from aisc_eval.celery_app import celery_app
from aisc_eval.data_model.evaluation import Evaluation
from aisc_eval.service.api_client import (
    mark_completed,
    mark_failed,
    mark_processing,
    mark_plugin_started,
    mark_plugin_finished,
    mark_plugin_failed,
    post_measures,
    get_evaluation,
    get_evaluation_request,
    get_evaluation_plugins_status,
    get_dataset_file_content,
    get_model_file_content,
    upload_artifact,
    get_project_settings_by_pid,
    get_evaluation_inputs,
    store_input_value,
)
from aisc_eval.service import run_pass
from aisc_eval.service.helpers import check_endpoint, resolve_helper
from aisc_eval.utils.logging import get_logger

from aisc_plugin_manager.loader import Loader, local_version
import aisc_plugin_interface.manifest as plugin_manifest
from aisc_plugin_interface import TaskProgress, Measure
from aisc_eval.utils import env
from aisc_eval.utils.encryption import decrypt_value

logger = get_logger()

SECRET_ENV_PREFIX = "AISC_SECRET_"


def build_project_settings(settings: list[dict]) -> dict:
    """Build the non-secret runtime settings exposed through the plugin API."""
    values: dict = {}
    for setting in settings:
        if setting["category"] in {"variables", "datashape"}:
            values[setting["plugin_config_key"]] = (
                setting.get("json_value", {}).get("value")
                if setting["category"] == "variables"
                else setting.get("json_value", {})
            )
    return values


def build_secret_environment(settings: list[dict]) -> dict[str, str]:
    """Decrypt selected secrets under their environment names.

    A secret's env var is named after the plugin declared key (plugin_config_key)
    when present, otherwise the project config key under which it is stored.
    """
    values = {}
    for setting in settings:
        if setting["category"] == "secrets" and setting.get("encrypted_value"):
            name_source = setting.get("plugin_config_key") or setting.get("key")
            if not name_source:
                continue
            values[_secret_env_name(name_source)] = decrypt_value(
                setting["encrypted_value"]
            )
    return values


def _secret_env_name(name_source: str) -> str:
    return SECRET_ENV_PREFIX + "".join(
        character.upper() if character.isalnum() else "_" for character in name_source
    )


#: The plugin_config_key prefix of a configuration field the backend kept as a project secret.
CONFIG_SECRET_PREFIX = "config."


def config_secret_fields(settings: list[dict]) -> dict[str, str]:
    """The configuration fields kept as project secrets, as path -> the environment variable that
    holds the key in the plugin's process (the name build_secret_environment gives it), sorted by
    path. Names only: config.json carries this, never a value."""
    fields = {}
    for setting in settings:
        key = setting.get("plugin_config_key") or ""
        if (
            setting["category"] == "secrets"
            and setting.get("encrypted_value")
            and key.startswith(CONFIG_SECRET_PREFIX)
        ):
            fields[key[len(CONFIG_SECRET_PREFIX) :]] = _secret_env_name(key)
    return dict(sorted(fields.items()))


#: What a plugin may see of the worker's environment. A plugin is catalogue code: the worker's
#: own credentials (broker and Redis URLs, INTERNAL_API_KEY, DJANGO_SECRET_KEY, the package
#: registry login) stay out, or a plugin could read other runs' queued tasks, lift their run
#: tickets and call the backend as another project. PLATFORM_CONNECTIONS_TOKEN stays out too: it
#: opens every project's connections; a plugin gets its run's pass instead (service/run_pass.py).
PLUGIN_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # the platform's connection resolver and the model addresses (aisc_plugin_interface.connections);
        # the worker's own model keys never pass: a plugin's target key comes from its run's pass
        "PLATFORM_URL",
        "CONNECTIONS_ALLOWED_HOSTS",
        "AISC_TARGET_BASE_URL",
        "OPENAI_BASE_URL",
    }
)
PLUGIN_ENV_PROXIES = frozenset({"http_proxy", "https_proxy", "no_proxy", "all_proxy"})
#: LangBiTe builds an OpenAI client at init even with a local model and no judge; the client
#: checks the key only on a real request, so a non-empty placeholder lets it start
OPENAI_PLACEHOLDER = "sk-local-model-no-openai-call"
#: LiteLLM reads the price map it ships with instead of fetching a 4 MB copy from GitHub at every run
LITELLM_PRICE_MAP = {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}


def plugin_environment(
    worker_env: Mapping[str, str],
    secrets: Mapping[str, str],
    run_pass: str | None = None,
) -> dict[str, str]:
    """The environment a plugin runs with: the allowed part of the worker's, its project's secrets
    and, when one was minted, its run's pass to the platform."""
    env = {
        name: value
        for name, value in worker_env.items()
        if name in PLUGIN_ENV_NAMES
        or name.lower() in PLUGIN_ENV_PROXIES
        or name.startswith(SECRET_ENV_PREFIX)
    }
    env.update(secrets)
    if run_pass:
        env["PLATFORM_RUN_PASS"] = run_pass
    if not env.get("API_KEY_OPENAI"):
        env["API_KEY_OPENAI"] = OPENAI_PLACEHOLDER
    env.update(LITELLM_PRICE_MAP)
    return env


LOCAL_MODEL_UNKNOWN = "could not tell whether the model file is a local target (the plugin could not be loaded)"
BOTH_KINDS = "the run binds both a local model file and an AI card target: a run assesses one of them"
MARKER_NOT_STORED = (
    "could not store the local model file's name and hash on the run's input"
)


def local_model_marker(component: dict, file_content: bytes) -> dict:
    """What a run on a local model file was on: the file's name and the hash of the bytes the plugin gets."""
    return {
        "local_model": {
            "file": component.get("component_name") or component["data"],
            "sha256": hashlib.sha256(file_content).hexdigest(),
        }
    }


plugin_loader: Loader = Loader(
    env.PLUGIN_PATH,
    env.PACKAGE_REGISTRY_URL,
    env.PACKAGE_REGISTRY_INDEX,
    env.PACKAGE_REGISTRY_USER,
    env.PACKAGE_REGISTRY_PASSWORD,
)


def _scrubbed(value, known=()):
    """`value` with every string in it, at any depth, dict keys included, passed through scrub_text."""
    if isinstance(value, str):
        return scrub_text(value, known)
    if isinstance(value, dict):
        return {
            _scrubbed(key, known): _scrubbed(inner, known)
            for key, inner in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrubbed(item, known) for item in value]
    return value


def progress_callback(
    task_progress: TaskProgress, plugin_name: str, task_id: str, known=()
):
    # the plugin writes its progress (its `extra` too) itself: stored scrubbed, like everything else it prints
    meta = _scrubbed({**task_progress.model_dump(), "plugin_name": plugin_name}, known)
    celery_app.backend.store_result(task_id, meta, state="RUNNING")


def _without_known(value, known):
    """`value` with each known value (a secret the run handed its plugin) replaced, at any depth, dict keys included.
    Only the known values: the rest of a measure is the plugin's result and is posted as it is."""
    if isinstance(value, str):
        for secret in known:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {
            _without_known(key, known): _without_known(inner, known)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_without_known(item, known) for item in value]
    return value


def _scrubbed_error(exc: BaseException, known=()) -> BaseException:
    """An exception to raise in place of `exc`, with no secret in its text, its args or its chain.

    `exc` itself when its text and args are clean and it chains nothing; else a RuntimeError of its scrubbed text,
    raised `from None` by the caller (what Celery stores of a failed task is its type, args and text). The original
    traceback goes to the log, scrubbed."""
    text = scrub_text(str(exc), known)
    args_clean = all(scrub_text(repr(arg), known) == repr(arg) for arg in exc.args)
    if (
        text == str(exc)
        and args_clean
        and exc.__cause__ is None
        and exc.__context__ is None
    ):
        return exc
    logger.error(scrub_text("".join(tracebacks.format_exception(exc)), known))
    return RuntimeError(f"{type(exc).__name__}: {text}")


#: What a test of a run that failed is marked with when it is stopped, or never starts.
STOPPED = "Stopped: another test of this run failed"
#: a test the queue handed out again after the worker stopped while it ran (task_acks_late): it is not run twice
INTERRUPTED = (
    "Interrupted: the worker stopped while this test was running. It is not run again, so the system "
    "under test is not sent everything a second time; its record of calls and any files it saved are kept."
)
#: The states of a test that has not finished: the ones a failure of the run stops.
UNFINISHED = ("Pending", "Running")
#: the signal the run's other tasks are revoked with: Celery turns it into the soft time limit inside each task, so
#: its own failure handler ends its plugin and its pass (the signal the backend's Stop sends too)
STOP_SIGNAL = "SIGUSR1"
RUN_STATE_UNREADABLE = "The run's state could not be read, so the test did not start."
NO_ISOLATION = (
    "plugin network isolation is not in force on this worker (its firewall is not set: does the container have"
    " NET_ADMIN?), so the test did not start"
)
NO_PROXY_RUNNING = "the egress proxy is not running on this worker, so the test did not start"
NO_MEASURES = "The plugin finished without any measure."


class NotInIndex(KeyError):
    """The package index does not list the package, or this version of it."""

    def __str__(self) -> str:
        # KeyError quotes its text; the person reads it as it was written
        return str(self.args[0]) if self.args else ""


class StartRefused(RuntimeError):
    """The engine refused the test's start: another copy of it started first, or the test has already ended."""


#: seconds a plugin's run pass outlives the test's hard time limit: the worker still reads the run's
#: calls with it and revokes it once the plugin has ended
RUN_PASS_MARGIN_S = 300
#: cores a plugin's CPU time is counted for (PLUGIN_CPU_CORES): CPU time adds up over its threads, so a
#: plugin busy on four cores uses four seconds of it per second of the clock
PLUGIN_CPU_CORES = 4


def plugin_cpu_seconds() -> int:
    """The CPU time a plugin may use: its test's soft time limit on each core it may keep busy."""
    raw = os.environ.get("PLUGIN_CPU_CORES", "").strip()
    try:
        cores = int(raw) if raw else PLUGIN_CPU_CORES
    except ValueError:
        raise RuntimeError(f"PLUGIN_CPU_CORES must be a whole number, not {raw!r}") from None
    if cores <= 0:
        raise RuntimeError(f"PLUGIN_CPU_CORES must be above 0, not {cores}")
    return int(celery_app.conf.task_soft_time_limit) * cores


def _run_tasks_key(evaluation_pid: uuid.UUID) -> str:
    return run_registry.tasks_key(evaluation_pid)


def _run_tasks_ttl(task_count: int) -> int:
    """How long a run's task list is kept: as long as the run can last, each of its tasks taking the longest a
    task may (task_time_limit), one after the other as the worker runs them."""
    return task_count * int(celery_app.conf.task_time_limit)


def _revoke_sibling_tasks(
    evaluation_pid: uuid.UUID, current_task_id: str | None
) -> None:
    try:
        raw = celery_app.backend.client.get(_run_tasks_key(evaluation_pid))
        if raw:
            for tid in json.loads(raw):
                if tid != str(current_task_id):
                    celery_app.control.revoke(tid, terminate=True, signal=STOP_SIGNAL)
    except Exception as e:
        logger.warning(f"Could not revoke sibling tasks: {e}")


def _stop_run(
    evaluation_pid: uuid.UUID,
    failed_pids: list[uuid.UUID],
    error_msg: str,
    current_task_id: str | None,
) -> None:
    """A test of the run failed: failed_pids are marked failed with error_msg, each of its other tests not yet
    finished is marked failed as STOPPED, and the run's other tasks are revoked (those running stop, those queued
    never start). The marks come first, so a revoked test that reports its own end finds its reason already kept.
    A test that already finished keeps its result."""
    for pid in failed_pids:
        try:
            mark_plugin_failed(evaluation_pid, pid, error_msg)
        except Exception as e:
            logger.warning(f"Could not mark test {pid} failed: {type(e).__name__}")
    try:
        _mark_the_others_stopped(evaluation_pid, failed_pids)
    finally:
        _revoke_sibling_tasks(evaluation_pid, current_task_id)


def _mark_the_others_stopped(
    evaluation_pid: uuid.UUID, failed_pids: list[uuid.UUID]
) -> None:
    try:
        tests = get_evaluation_request(evaluation_pid).get("evaluation_plugins") or []
    except Exception as e:
        logger.warning(
            f"Evaluation {evaluation_pid}: its other tests could not be read ({type(e).__name__})"
        )
        return
    failed = {str(pid) for pid in failed_pids}
    for test in tests:
        if (
            not isinstance(test, dict)
            or str(test.get("pid")) in failed
            or test.get("status") not in UNFINISHED
        ):
            continue
        try:
            mark_plugin_failed(evaluation_pid, test["pid"], STOPPED)
        except Exception as e:
            logger.warning(
                f"Could not mark test {test['pid']} stopped: {type(e).__name__}"
            )


@celery_app.task(bind=True)
def install_package(
    self,
    package_name: str,
    version: str,
    evaluation_pid: uuid.UUID,
    evaluation_plugin_pids: list[uuid.UUID],
):
    """Install a package once using uv run to cache dependencies."""
    logger.info(f"Caching package {package_name}=={version}")

    try:
        plugin_info = _versions_of(package_name, version)[version]
        with tempfile.TemporaryDirectory(delete=True) as cache_tmp:
            plugin_process.build_venv(
                package_name,
                version,
                plugin_info,
                Path(cache_tmp),
                plugin_process.install_user(),
                plugin_loader.devpi_client.simple_index_url,
            )

        logger.info(
            f"Successfully cached {package_name}=={version} and all dependencies"
        )
    except Exception as e:
        # no traceback: an installer's message can hold a key, and only its scrubbed text is kept
        text = scrub_text(str(e))
        logger.error(
            f"Failed to cache {package_name}=={version}: {type(e).__name__}: {text}"
        )
        if evaluation_pid and evaluation_plugin_pids:
            _stop_run(evaluation_pid, evaluation_plugin_pids, text, self.request.id)
        raise


@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")
    try:
        return _start_evaluation(evaluation_pid)
    except Exception as exc:
        # nothing was dispatched, so no later task will close the run: without this it stays Pending
        _fail_before_start(evaluation_pid, exc)
        return {"evaluation_pid": evaluation_pid, "plugin_task_ids": []}


def _fail_before_start(evaluation_pid: uuid.UUID, exc: BaseException) -> None:
    """A run that could not start: each of its tests is marked failed with the reason (when the run's tests can
    be read at all), then the run itself."""
    reason = f"the run could not start: {type(exc).__name__}: {str(exc)}"[:500]
    logger.error(f"Evaluation {evaluation_pid}: {scrub_text(reason)}")
    try:
        try:
            tests = (
                get_evaluation_request(evaluation_pid).get("evaluation_plugins") or []
            )
        except Exception as read_failure:
            logger.error(
                f"Evaluation {evaluation_pid}: its tests could not be read ({type(read_failure).__name__})"
            )
            tests = []
        for test in tests:
            if not (isinstance(test, dict) and test.get("pid")):
                continue
            # one test that cannot be marked keeps neither the others nor the run from being marked
            try:
                mark_plugin_failed(evaluation_pid, test["pid"], reason)
            except Exception as mark_failure:
                logger.error(
                    f"Test {test['pid']} could not be marked failed ({type(mark_failure).__name__})"
                )
    finally:
        mark_failed(evaluation_pid)


def _start_evaluation(evaluation_pid: uuid.UUID) -> dict:
    """run_evaluation's work: read the run, claim it, register it for the sweep, dispatch each package's install
    and its tests, and show it Processing. A run that has already ended (a person stopped it) is not dispatched;
    a second delivery finds the claim and dispatches nothing."""
    plugin_loader.list_packages(refresh=True)

    status = get_evaluation_request(evaluation_pid).get("status")
    if status in ("Failed", "Done"):
        logger.info(f"Evaluation {evaluation_pid} is {status}: nothing dispatched")
        return {"evaluation_pid": evaluation_pid, "plugin_task_ids": []}

    evaluation: Evaluation = get_evaluation(evaluation_pid)
    inputs_by_plugin: dict[str, list[dict]] = get_evaluation_inputs(evaluation_pid)

    # group plugins by package
    plugins_by_pkg = {}
    for evaluation_plugin in evaluation.evaluation_plugins:
        pkg_key = f"{evaluation_plugin.package_name}=={evaluation_plugin.version}"
        if pkg_key not in plugins_by_pkg:
            plugins_by_pkg[pkg_key] = {
                "package_name": evaluation_plugin.package_name,
                "version": evaluation_plugin.version,
                "plugins": [],
            }
        plugins_by_pkg[pkg_key]["plugins"].append(evaluation_plugin)

    logger.info(
        f"Found {len(plugins_by_pkg)} unique packages for {len(evaluation.evaluation_plugins)} plugins"
    )

    # build per-package chains: install_pkg -> group(run_plugin -> post_measurements)
    package_chains = []
    plugin_task_ids = []
    install_task_ids = []
    tests: dict[str, str] = {}
    installs: dict[str, str] = {}
    for pkg_key, pkg_info in plugins_by_pkg.items():
        install_sig = install_package.si(
            pkg_info["package_name"],
            pkg_info["version"],
            evaluation_pid,
            [ep.pid for ep in pkg_info["plugins"]],
        )
        install_sig.freeze()
        install_task_ids.append(str(install_sig.id))

        plugin_chain_list = []
        for evaluation_plugin in pkg_info["plugins"]:
            config = None
            project_config_selections = []
            if evaluation_plugin.plugin_config:
                config = evaluation_plugin.plugin_config.config
                project_config_selections = (
                    evaluation_plugin.plugin_config.project_config_selections
                )

            input_components = inputs_by_plugin.get(str(evaluation_plugin.pid), [])

            run_plugin_sig = run_plugin.si(
                evaluation_plugin.package_name,
                evaluation_plugin.name,
                evaluation_plugin.version,
                config,
                input_components,
                get_project_settings_by_pid(
                    evaluation.project.pid, project_config_selections
                ),
                evaluation_pid,
                evaluation_plugin.pid,
            )

            # freeze to get a stable task id before dispatching
            run_plugin_sig.freeze()
            plugin_task_ids.append(str(run_plugin_sig.id))
            tests[str(evaluation_plugin.pid)] = str(run_plugin_sig.id)
            installs[str(evaluation_plugin.pid)] = str(install_sig.id)

            post_measurements_sig = post_measurements.s(
                evaluation_pid, evaluation_plugin.pid
            )
            plugin_chain_list.append(chain(run_plugin_sig, post_measurements_sig))

        package_chains.append(chain(install_sig, group(plugin_chain_list)))

    # the run's installs and tests, so a failure or a Stop revokes them; kept as long as the run can last: every
    # install, test and post of its measurements, and the finalisation. Set only when absent: the claim
    client = celery_app.backend.client
    ttl = _run_tasks_ttl(len(install_task_ids) + 2 * len(plugin_task_ids) + 1)
    if not run_registry.claim(
        client, evaluation_pid, install_task_ids + plugin_task_ids, ttl
    ):
        entry = run_registry.entry_of(client, evaluation_pid)
        logger.warning(
            f"Evaluation {evaluation_pid} was already dispatched: this delivery dispatches nothing"
        )
        return {
            "evaluation_pid": evaluation_pid,
            "plugin_task_ids": list(entry.tests.values()) if entry else [],
        }

    run = run_context.current()
    claimed_at = time.time()
    deadline = claimed_at + ttl
    expiry = run_registry.ticket_expiry(run)
    if expiry is not None:
        deadline = min(deadline, expiry - run_registry.DEADLINE_MARGIN_S)
    run_registry.register(
        client,
        run_registry.RegisteredRun(
            evaluation=str(evaluation_pid),
            run=run,
            claimed_at=claimed_at,
            deadline=deadline,
            tests=tests,
            installs=installs,
        ),
        ttl + run_registry.DEADLINE_MARGIN_S,
    )

    # dispatch: each package's plugins start after its install, no waiting for other packages
    workflow = group(package_chains) | finalize_evaluation.si(evaluation_pid)
    try:
        workflow.apply_async()
    except Exception:
        # nothing was sent: the claim goes, and run_evaluation fails the run
        run_registry.release(client, evaluation_pid)
        raise

    try:
        mark_processing(evaluation_pid)
    except Exception as exc:
        # the run goes on; it only keeps showing Pending
        logger.warning(
            f"Evaluation {evaluation_pid} could not be shown Processing ({type(exc).__name__})"
        )

    return {"evaluation_pid": evaluation_pid, "plugin_task_ids": plugin_task_ids}


@celery_app.task(bind=True)
def run_plugin(
    self,
    package_name: str,
    plugin_name: str,
    version: str,
    plugin_config: dict,
    input_components: list[dict],
    project_settings: list[dict],
    evaluation_pid: uuid.UUID,
    evaluation_plugin_pid: uuid.UUID,
) -> list[dict]:
    if not plugin_loader.discovered_packages:
        logger.debug("Worker cache empty. Fetching packages from Devpi...")
        plugin_loader.list_packages()

    with test_run(evaluation_pid, evaluation_plugin_pid, self.request.id) as reporter:
        if reporter is None:
            return []
        return _run_the_plugin(
            reporter,
            package_name,
            plugin_name,
            version,
            plugin_config,
            input_components,
            project_settings,
            str(self.request.id),
        )
    # the engine refused the start: another copy of this test runs it, or it has already ended
    return []


def _may_start(
    evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID, eval_data: dict
) -> bool:
    """Whether the test may start, from its run as the engine has it. A test of a run that already failed never
    starts (marked stopped, unless already failed); a test the queue hands out a second time (its first delivery
    was never acknowledged: the worker stopped, or RabbitMQ closed a delivery held past its consumer timeout) is
    never run again: once finished it has nothing left to do, and one cut off is marked interrupted."""
    own = next(
        (
            t
            for t in eval_data.get("evaluation_plugins") or []
            if isinstance(t, dict) and str(t.get("pid")) == str(evaluation_plugin_pid)
        ),
        {},
    )
    if eval_data.get("status") == "Failed":
        logger.info(
            f"Evaluation {evaluation_pid} already failed: test {evaluation_plugin_pid} not started"
        )
        if own.get("status") != "Failed":
            try:
                mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, STOPPED)
            except (
                Exception
            ) as exc:  # not marked: the run has failed, the test still never starts
                logger.error(
                    f"Test {evaluation_plugin_pid} could not be marked stopped: {type(exc).__name__}"
                )
        return False
    if own.get("finished_at"):
        logger.warning(
            f"Test {evaluation_plugin_pid} of {evaluation_pid} already finished: not run again"
        )
        return False
    if own.get("started_at"):
        logger.warning(
            f"Test {evaluation_plugin_pid} of {evaluation_pid} was cut off: marked interrupted"
        )
        try:
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, INTERRUPTED)
        except Exception as exc:  # not marked: still never run a second time
            logger.error(
                f"Test {evaluation_plugin_pid} could not be marked interrupted: {type(exc).__name__}"
            )
        return False
    return True


class RunReporter:
    """What one test reports to the engine, and what its failure handler must end: its plugin's process and the
    values the run handed the plugin (removed from every text reported)."""

    def __init__(
        self,
        evaluation_pid: uuid.UUID,
        test_pid: uuid.UUID,
        task_id: str | None,
        workspace: Path,
    ):
        self.evaluation_pid = evaluation_pid
        self.test_pid = test_pid
        self.task_id = task_id
        self.workspace = workspace
        self.known: list[str] = []
        self.process: subprocess.Popen | None = None
        #: the user the test runs as, and its folders, once leased (test_run)
        self.user: plugin_process.PluginUser | None = None
        self.folders: plugin_process.RunFolders | None = None
        #: the token its plugin names itself with to the egress proxy, once its rule is registered
        self.egress_token: str | None = None
        self.minted: str | None = None
        self._calls: bytes | None = None
        self._calls_read = False

    def calls(self) -> bytes | None:
        """The run's calls to the system under test, as the platform recorded them, read once with the run's
        pass; None when no pass was minted or the platform cannot give them."""
        if self.minted and not self._calls_read:
            self._calls = run_pass.trace(self.minted)
            self._calls_read = True
        return self._calls

    def started(self) -> None:
        try:
            mark_plugin_started(self.evaluation_pid, self.test_pid)
        except requests.HTTPError as exc:
            if _status_of(exc) == 409:
                raise StartRefused(
                    f"the engine refused the start of test {self.test_pid}"
                ) from None
            raise

    def finished(self) -> None:
        mark_plugin_finished(self.evaluation_pid, self.test_pid)

    def fail(self, exc: BaseException) -> BaseException:
        """The test failed with `exc`: its plugin is ended, it is marked failed with the reason and the run is
        stopped. Returns the exception to raise, with no value the run handed its plugin in it."""
        plugin_process.terminate(self.process)
        _stop_run(
            self.evaluation_pid,
            [self.test_pid],
            scrub_text(str(exc), self.known) or exc.__class__.__name__,
            self.task_id,
        )
        return _scrubbed_error(exc, self.known)


@contextmanager
def test_run(
    evaluation_pid: uuid.UUID, test_pid: uuid.UUID, task_id: str | None
) -> Iterator[RunReporter | None]:
    """One test, from the check that it may start to its end, each way it ends reported once.

    Yields None when the test must not run (its run failed, it already finished, or it was cut off). A run whose
    state cannot be read fails the test without starting it. A start the engine refuses ends this copy quietly:
    the test is another copy's, or has ended. Every other failure is reported once, with its reason, and raised
    without any value the run handed its plugin. Whatever happens, the run's calls are kept and its pass revoked
    when one was minted, and the workspace is removed. The test runs as a user of its own, leased here and given
    back last, once its plugin and everything it left running have ended and its folders are gone."""
    workspace = tempfile.TemporaryDirectory(delete=True)
    reporter = RunReporter(evaluation_pid, test_pid, task_id, Path(workspace.name))
    lease = ExitStack()
    try:
        unreadable = None
        try:
            eval_data = get_evaluation_request(evaluation_pid)
        except Exception as exc:
            unreadable = type(exc).__name__
        if unreadable is not None:
            logger.warning(
                f"Run {evaluation_pid} could not be read ({unreadable}): test {test_pid} not started"
            )
            raise RuntimeError(RUN_STATE_UNREADABLE)
        if not _may_start(evaluation_pid, test_pid, eval_data):
            yield None
        else:
            _check_isolation()
            reporter.user = lease.enter_context(plugin_process.lease_run_user())
            reporter.folders = plugin_process.make_run_folders(
                reporter.workspace, reporter.user
            )
            yield reporter
    except StartRefused as refused:
        logger.warning(f"{refused}: this copy ends without touching the test")
    except Exception as exc:
        clean = reporter.fail(exc)
        if clean is exc:
            raise
        raise clean from None
    finally:
        if reporter.minted:
            _keep_the_runs_calls(
                evaluation_pid, test_pid, reporter.calls(), reporter.known
            )
            run_pass.revoke(reporter.minted, run_context.current())
        try:
            if reporter.egress_token:
                egress.unregister(reporter.egress_token)
            if reporter.user is not None:
                plugin_process.end_processes_of(reporter.user)
            workspace.cleanup()
        finally:
            lease.close()


def _check_isolation() -> None:
    """A root worker starts plugins only behind its firewall and proxy (egress); one started by hand, not root,
    has neither and runs them as the developer."""
    if os.geteuid() != 0:
        return
    if not egress.in_force():
        raise RuntimeError(NO_ISOLATION)
    if not egress.proxy_answers():
        raise RuntimeError(NO_PROXY_RUNNING)


def _run_the_plugin(
    reporter: RunReporter,
    package_name: str,
    plugin_name: str,
    version: str,
    plugin_config: dict,
    input_components: list[dict],
    project_settings: list[dict],
    task_id: str,
) -> list[dict]:
    """What is specific to one run of a plugin: its inputs, configuration, environment, pass, process and
    outputs. Every failure raises, and test_run reports it."""
    evaluation_pid, evaluation_plugin_pid = reporter.evaluation_pid, reporter.test_pid
    plugin_info = _versions_of(package_name, version)[version]
    folders, user = reporter.folders, reporter.user
    work, input_dir, output_dir = folders.work, folders.input, folders.output

    # the backend could not load the plugin to say whether its model file is the local target:
    # running anyway could store a local run as if it were an AI card run
    if any(component.get("local_model_unknown") for component in input_components):
        raise RuntimeError(LOCAL_MODEL_UNKNOWN)

    input_mapping = {}
    #: what the plugin may reach besides the platform: its helpers' and LLM inputs' endpoints, under the rule
    #: the helpers came with
    endpoints: list[str] = []
    helper_rules: list[dict] = []
    llm_secret_settings = []
    helper_secret_env: dict[str, str] = {}
    local_model: dict | None = None
    for component in input_components:
        component_type = component["component_type"]
        name = component["name"]
        if component_type == "dataset":
            file_content = get_dataset_file_content(component["data"])
            relative_path = component["data"]
        elif component_type == "model":
            file_content = get_model_file_content(component["data"])
            relative_path = component["data"]
            if component.get("local_model"):
                local_model = {
                    "input": component.get("pid"),
                    "marker": local_model_marker(component, file_content),
                }
        elif component_type == "helper":
            # the engine names the helper; its endpoint and key come from the platform, and the
            # key goes to the plugin's environment only, under the name self.helper() reads
            role = name.removeprefix("helper:").upper()
            secret_key = f"AISC_HELPER_{role}_KEY"
            # a helper the platform cannot give fails the run with its reason, fit to show the person
            resolved = resolve_helper(
                component.get("helper_ref"),
                (run_context.current() or {}).get("project"),
            )
            # its address under the project's network rule, now, before anything is built
            check_endpoint(resolved)
            endpoints.append(resolved["endpoint_url"])
            helper_rules.append(resolved)
            if resolved.get("api_key"):
                helper_secret_env[_secret_env_name(secret_key)] = resolved[
                    "api_key"
                ]
                payload = {"secret_key": secret_key}
            else:
                payload = {}
            payload.update(
                endpoint_url=resolved["endpoint_url"],
                model=(component.get("value") or {}).get("model"),
            )
            file_content = json.dumps(payload).encode("utf-8")
            relative_path = f"{name}.json"
        elif component_type in {"datashape", "llm", "resource"}:
            payload = dict(component.get("json_value") or {})
            if component_type == "llm":
                if component.get("endpoint_url"):
                    endpoints.append(component["endpoint_url"])
                run_model = (component.get("value") or {}).get("model")
                if run_model:
                    payload["model"] = run_model
                if component.get("secret_key") and component.get(
                    "secret_encrypted_value"
                ):
                    llm_secret_settings.append(
                        {
                            "category": "secrets",
                            "key": component["secret_key"],
                            "encrypted_value": component["secret_encrypted_value"],
                        }
                    )
            file_content = json.dumps(payload).encode("utf-8")
            relative_path = f"{name}.json"
        else:
            raise ValueError(f"Unsupported component type: {component_type}")

        plugin_process.write_for_run(input_dir / relative_path, file_content, user)
        input_mapping[name] = relative_path

    # A run on a local model file says so, here for the plugin and on its input for the results:
    # it is never an AI card run, and it opens nothing on the platform (no run pass)
    if local_model is not None:
        if run_pass.refs_of(input_components):
            raise ValueError(BOTH_KINDS)
        plugin_process.write_for_run(
            input_dir / "target.json", json.dumps(local_model["marker"]).encode(), user
        )
        not_stored = None
        try:
            store_input_value(
                evaluation_pid, local_model["input"], local_model["marker"]
            )
        except Exception as exc:
            # the person reads this: the answer's status, never the backend's address
            status = getattr(getattr(exc, "response", None), "status_code", None)
            not_stored = f"{MARKER_NOT_STORED} ({f'HTTP {status}' if status else 'no answer'})"
        if not_stored:
            # raised outside the except: the backend's error, which names its address, is not chained
            raise RuntimeError(not_stored)

    config_data = {
        "plugin_source": f"{package_name}:{plugin_name}",
        "input_mapping": input_mapping,
        "project_settings": build_project_settings(project_settings),
        "plugin_config": plugin_config or {},
        "config_secret_fields": config_secret_fields(project_settings),
    }

    plugin_process.write_for_run(
        work / "config.json", json.dumps(config_data, indent=2).encode(), user
    )

    runtime_script = plugin_process.copy_runtime(work, user)

    # Steps 1 and 2: the venv, made and filled by the install user: the plugin may run it but not change it
    logger.debug(f"Installing {package_name}=={version} into an isolated venv")
    venv = plugin_process.build_venv(
        package_name,
        version,
        plugin_info,
        folders.root,
        plugin_process.install_user(),
        plugin_loader.devpi_client.simple_index_url,
    )
    venv_python = venv.python
    venv_result, install_result = venv.made, venv.installed

    # A pass to the platform for exactly the targets and connections this run names, all of the
    # run's own project: the platform pid the backend put in the run's aisc_run header
    refs = run_pass.refs_of(input_components)
    minted: str | None = None
    if refs:
        minted = reporter.minted = run_pass.mint(
            refs,
            str(evaluation_plugin_pid),
            project=(run_context.current() or {}).get("project"),
            ttl_s=int(celery_app.conf.task_time_limit) + RUN_PASS_MARGIN_S,
        )

    # Step 3: Run plugin script with the isolated venv's Python
    logger.debug(f"Running plugin with: {venv_python} {runtime_script}")
    start_time = time.perf_counter()
    reporter.started()

    stdout_lines: list[str] = []
    stderr_path = folders.root / "stderr.tmp"
    returncode: int | None = None
    stderr_content = ""
    secret_env = build_secret_environment(project_settings + llm_secret_settings)
    secret_env.update(helper_secret_env)
    # the plugin's way out: the proxy admits what this test may reach, named by a token of its own
    proxy_token = reporter.egress_token = secrets.token_urlsafe(32)
    egress.register(
        proxy_token,
        egress.run_egress(
            os.environ.get("PLATFORM_URL"),
            endpoints,
            helper_rules,
            egress.operator_hosts(),
        ),
    )
    known = reporter.known = known_secrets(
        secret_env, os.environ, plugin_config, minted, also=[proxy_token]
    )
    # the worker's only (0600): the run folder can be passed through by every plugin user
    with open(stderr_path, "w+", opener=_private) as stderr_file:
        child_env = {
            **plugin_environment(os.environ, secret_env, minted),
            **egress.proxy_env(proxy_token),
        }
        process = reporter.process = plugin_process.start(
            [str(venv_python), str(runtime_script)],
            work,
            stderr_file,
            child_env,
            user,
            limits=plugin_process.plugin_limits(plugin_cpu_seconds()),
        )

        if (stdout := process.stdout) is not None:
            for line in stdout:
                stdout_lines.append(line)
                stripped_line = line.strip()

                if stripped_line.startswith("{") and stripped_line.endswith("}"):
                    try:
                        progress_data = json.loads(stripped_line)
                        task_progress = TaskProgress(**progress_data)

                        progress_callback(
                            task_progress,
                            plugin_name,
                            task_id,
                            known,
                        )
                    except Exception:
                        pass

        returncode = plugin_process.wait(process)

        stderr_file.seek(0)
        stderr_content = stderr_file.read()

    end_time = time.perf_counter()
    duration = end_time - start_time
    stdout_content = "".join(stdout_lines)

    log_stdout_content = (
        "====== Step 1: Venv Creation ======\n"
        + (venv_result.stdout or "")
        + "\n\n"
        "====== Step 2: Plugin Install ======\n"
        + (install_result.stdout or "")
        + "\n\n"
        "====== Step 3: Plugin Run ======\n" + stdout_content + "\n\n"
    )
    log_stderr_content = (
        "====== Step 1: Venv Creation ======\n"
        + (venv_result.stderr or "")
        + "\n\n"
        "====== Step 2: Plugin Install ======\n"
        + (install_result.stderr or "")
        + "\n\n"
        "====== Step 3: Plugin Run ======\n" + stderr_content + "\n\n"
    )

    log_content = "=== Plugin Execution Log ===\n\n"
    log_content += f"Execution Time: {duration:.2f} seconds\n\n"
    log_content += f"=== STDOUT ===\n{log_stdout_content}\n\n"
    log_content += f"=== STDERR ===\n{log_stderr_content}\n\n"
    log_content += f"=== Return Code ===\n{returncode}\n"
    log_content = scrub_text(log_content, known)

    if returncode != 0:
        # the whole log is the run's file, failed or not; the run's error is the plugin's own reason
        try:
            upload_artifact(
                evaluation_pid,
                evaluation_plugin_pid,
                "plugin_execution.log",
                log_content.encode(),
            )
        except Exception as exc:
            logger.warning(
                f"The failed run's log was not kept as a file ({exc.__class__.__name__})"
            )
        # the run is stopped once, with this message, by test_run
        raise RuntimeError(
            f"Plugin failed: {plugin_reason(scrub_text(stderr_content, known), returncode)}"
        )

    outputs = plugin_process.read_outputs(output_dir, user)
    # the run's log and its recorded calls are the worker's files: a plugin file of either name would sit
    # beside them (or stand in for the calls when the trace cannot be read), so the run fails instead
    for name in outputs:
        if name.casefold() in _WORKERS_FILES:
            raise RuntimeError(
                f"Plugin output {name} has the name of a file the worker keeps for the run"
            )
    measures = json.loads(outputs.pop("measures.json", b"[]"))

    # the log is the worker's, kept out of output/ where the plugin could plant a link
    outputs["plugin_execution.log"] = log_content.encode()
    # every file without the values the run handed its plugin, in its name and its bytes
    for name, content in outputs.items():
        upload_artifact(
            evaluation_pid,
            evaluation_plugin_pid,
            _without_known(name, known),
            without_known_bytes(content, known),
        )

    # a plugin can end well after the platform stopped its run's calls (it counts each refused call as one
    # more failed question): its results are partial, so the test fails with the platform's reason
    stopped = platform_stop_reason(reporter.calls())
    if stopped:
        raise RuntimeError(stopped)
    # a plugin that ended well but measured nothing has no result: the test is not Done
    if not measures:
        raise RuntimeError(NO_MEASURES)

    reporter.finished()
    # what the chain hands post_measurements (and the result backend keeps): no value the plugin was given
    return _without_known(measures, known)


def _private(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def _versions_of(package_name: str, version: str) -> dict:
    """The known versions of a package, which must hold `version`. The worker lists the index once and keeps
    the list, so a version published since is not in it: the index is listed again (refreshed, not the kept
    list) before it is refused."""
    known = plugin_loader.discovered_packages
    if version not in (known.get(package_name) or {}):
        plugin_loader.list_packages(refresh=True)
        known = plugin_loader.discovered_packages
    if package_name not in known:
        raise NotInIndex(f"Package '{package_name}' not found.")
    if version not in known[package_name]:
        raise NotInIndex(f"Version '{version}' of package '{package_name}' not found.")
    return known[package_name]


# ── what the engine shows of a plugin, read here so the engine never imports plugin code ─────────────────

#: seconds a plugin may take to describe itself (import its module, build its form)
MANIFEST_TIMEOUT_S = 120
#: seconds a form hook may take (a person waits on the form for it)
HOOK_TIMEOUT_S = 30
#: the largest answer a manifest or a hook may give
ANSWER_MAX_BYTES = 2**20
#: where each plugin version's environment is kept for its manifest and form hooks, by the hash of its wheel
#: or of its local folder: root's, each environment the install user's, readable and runnable by run users
PLUGIN_ENVS = Path("/var/cache/aisc-plugin-envs")
#: the manifest command, copied next to each call: run users cannot read the worker's own files
MANIFEST_SCRIPT = "plugin_manifest.py"
NO_WHEEL_HASH = (
    "{package} {version} has no wheel hash, recorded or on the index: a catalogue plugin is installed only "
    "from a wheel whose hash is known"
)


def _listed(package_name: str, version: str) -> dict:
    """The listing's entry of one version, listed afresh (a version published since is found); a local
    plugin stored under its declared version is found too."""
    versions = plugin_loader.list_packages(refresh=True).get(package_name) or {}
    found = versions.get(version) or versions.get(local_version(version))
    if found is None:
        raise NotInIndex(f"Version '{version}' of package '{package_name}' not found.")
    return found


def _expected_hash(package_name: str, version: str, info: dict, recorded: str | None) -> str | None:
    """The wheel hash a catalogue plugin must be installed from: the one recorded, else the index's. A
    local plugin has none (it is installed from its folder)."""
    if info.get("source") == "local":
        return None
    expected = recorded or plugin_loader.devpi_client.wheel_sha256(package_name, version)
    if not expected:
        raise RuntimeError(NO_WHEEL_HASH.format(package=package_name, version=version))
    return expected.lower()


def _envs_root() -> Path:
    """PLUGIN_ENVS, or a folder under the temporary one for a worker that is not root."""
    root = PLUGIN_ENVS if os.geteuid() == 0 else Path(tempfile.gettempdir()) / "aisc-plugin-envs"
    root.mkdir(mode=0o711, parents=True, exist_ok=True)
    return root


@contextmanager
def _plugin_env(
    package_name: str, version: str, info: dict, expected: str | None
) -> Iterator[tuple[Path, str]]:
    """(venv python, dependency lock) of the plugin's environment for its manifest and form hooks, built
    once per wheel hash or local folder digest from the hashed lock, and kept. A local folder whose digest
    cannot be read gets an environment of its own, removed after the call."""
    key = expected or (info.get("digest") if info.get("source") == "local" else None)
    with ExitStack() as stack:
        if key is None:
            folder = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            root = _envs_root()
            folder = root / key
            held = stack.enter_context(open(root / f"{key}.lock", "a"))
            fcntl.flock(held, fcntl.LOCK_EX)
        ready = folder / "ready"
        if not ready.is_file():
            if folder.exists() and key is not None:
                shutil.rmtree(folder)
            folder.mkdir(mode=0o711, exist_ok=True)
            os.chmod(folder, 0o711)
            installer = plugin_process.install_user()
            index_url = plugin_loader.devpi_client.simple_index_url
            lock = plugin_process.lock_dependencies(
                package_name, version, info, expected, folder, installer, index_url
            )
            plugin_process.build_venv(
                package_name,
                version,
                info,
                folder,
                installer,
                index_url,
                pins=plugin_process.Pins(expected, lock),
            )
            ready.write_text("")
        yield folder / "venv" / "bin" / "python", (
            folder / plugin_process.LOCK_FILE
        ).read_text()


def _sandboxed(venv_python: Path, argv: list[str], stdin: bytes, timeout_s: int) -> bytes:
    """The manifest command run as a leased run user, in folders of its own, with no network (no proxy
    token: the firewall lets a run user reach only the proxy), no pass and none of the worker's secrets."""
    _check_isolation()
    with ExitStack() as stack:
        workspace = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        user = stack.enter_context(plugin_process.lease_run_user())
        stack.callback(plugin_process.end_processes_of, user)
        folders = plugin_process.make_run_folders(workspace, user)
        script = folders.work / MANIFEST_SCRIPT
        plugin_process.write_for_run(
            script, Path(plugin_manifest.__file__).read_bytes(), user
        )
        return plugin_process.run_for_answer(
            [str(venv_python), str(script), *argv],
            folders.work,
            plugin_environment(os.environ, {}),
            user,
            stdin,
            timeout_s,
            ANSWER_MAX_BYTES,
            limits=plugin_process.plugin_limits(timeout_s),
        )


def _json_answer(answer: bytes, what: str):
    try:
        return json.loads(answer)
    except ValueError:
        raise plugin_process.NoAnswer(f"the plugin's {what} is not JSON") from None


@celery_app.task
def read_manifest(package_name: str, version: str, wheel_sha256: str | None = None) -> dict:
    """What the engine stores of a plugin version: {"wheel_sha256", "dependency_lock", "manifest", "code"}.
    The plugin is installed from a hashed lock that must name the recorded wheel (or the index's, when none
    is recorded; a catalogue plugin with neither is refused before anything is installed), then describes
    itself as a run user with no network. "code" is a local folder's digest and git state, None otherwise."""
    info = _listed(package_name, version)
    expected = _expected_hash(package_name, version, info, wheel_sha256)
    with _plugin_env(package_name, version, info, expected) as (python, lock):
        answer = _sandboxed(python, ["describe", info["module_name"]], b"", MANIFEST_TIMEOUT_S)
    manifest = _json_answer(answer, "manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("plugins"), dict):
        raise plugin_process.NoAnswer("the plugin's manifest names no plugin class")
    code = None
    if info.get("source") == "local":
        code = {name: info.get(name) for name in ("digest", "git_commit", "git_dirty")}
    return {"wheel_sha256": expected, "dependency_lock": lock, "manifest": manifest, "code": code}


@celery_app.task
def plugin_hook(
    package_name: str, version: str, plugin_name: str, hook: str, args: dict, pins: dict
):
    """One form hook of one plugin class (aisc_plugin_interface.manifest.HOOKS only), run as a run user with
    no network, in the environment of the wheel recorded for it (pins["wheel_sha256"])."""
    if hook not in plugin_manifest.HOOKS:
        raise ValueError(f"{hook} is not a form hook")
    info = _listed(package_name, version)
    expected = _expected_hash(package_name, version, info, (pins or {}).get("wheel_sha256"))
    asked = json.dumps({"module": info["module_name"], "plugin": plugin_name, "hook": hook, "args": args})
    with _plugin_env(package_name, version, info, expected) as (python, _lock):
        answer = _sandboxed(python, ["hook"], asked.encode(), HOOK_TIMEOUT_S)
    return _json_answer(answer, f"answer to {hook}")


#: how much of a failed plugin's reason a run's error keeps (the whole log is the run's file)
REASON_MAX = 500
#: an exception's class before its message ("pkg.module.RunRefused: "), left out of the reason
_EXCEPTION_CLASS = re.compile(r"^(?:\w+\.)*[A-Z]\w*: ")


def plugin_reason(stderr_text: str, returncode: int) -> str:
    """What a failed plugin said last on its error stream (its exception's message, without the class),
    cut to REASON_MAX: the run's error names the reason, not the environment setup or the installs."""
    lines = [line.strip() for line in (stderr_text or "").splitlines() if line.strip()]
    if not lines:
        return f"it ended with code {returncode} and no error message; its log is the run's file"
    reason = _EXCEPTION_CLASS.sub("", lines[-1]) or lines[-1]
    return reason if len(reason) <= REASON_MAX else reason[: REASON_MAX - 1] + "…"


#: the run's calls to the system under test, as the platform recorded them, kept with the run's files
CALLS_FILE = "system-under-test-calls.jsonl"
#: the names of the files the worker itself keeps with a run, which no plugin output may take
_WORKERS_FILES = {CALLS_FILE.casefold(), "plugin_execution.log"}


#: the code of a call the platform refused because it had stopped the run, in the run's recorded calls
RUN_STOPPED_CODE = "run_stopped"
PLATFORM_STOPPED = "The platform stopped this run's calls to the system under test."


def platform_stop_reason(calls: bytes | None) -> str | None:
    """The reason the platform gave for stopping the run, read from the run's recorded calls: the detail of
    the first call it refused as stopped. None when it never stopped the run."""
    for line in (calls or b"").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        error = row.get("error") if isinstance(row, dict) else None
        if isinstance(error, dict) and error.get("code") == RUN_STOPPED_CODE:
            detail = error.get("detail")
            return detail if isinstance(detail, str) and detail.strip() else PLATFORM_STOPPED
    return None


def _keep_the_runs_calls(evaluation_pid, evaluation_plugin_pid, calls, known) -> None:
    """Upload the run's recorded calls as one of its files, whether the plugin succeeded or failed. Never
    fails the run: the platform's trace still holds every call."""
    if calls is None:
        return
    try:
        upload_artifact(
            evaluation_pid,
            evaluation_plugin_pid,
            CALLS_FILE,
            without_known_bytes(calls, known),
        )
    except Exception as exc:
        logger.warning(
            f"The run's calls were not kept as a file ({exc.__class__.__name__}); they stay in the platform's trace"
        )


@celery_app.task
def post_measurements(
    measurements_dict: list[dict],
    evaluation_pid: uuid.UUID,
    evaluation_plugin_uuid: uuid.UUID,
):
    # a test that did not run (skipped, or its start refused) hands on nothing: nothing is recorded or failed
    if not measurements_dict:
        return
    # run_plugin returns the measures without the values it handed the plugin; key shapes are scrubbed here
    try:
        measurements = [Measure(**m) for m in measurements_dict]
        post_measures(evaluation_pid, evaluation_plugin_uuid, measurements)
    except Exception as e:
        if isinstance(e, requests.HTTPError) and _status_of(e) == 409:
            # a second delivery of this post: the engine already has the test's measures
            logger.warning(
                f"Measures of test {evaluation_plugin_uuid} were already recorded: not posted again"
            )
            return
        # post_measurements is in no run's task list: every one of them is revoked
        _stop_run(evaluation_pid, [evaluation_plugin_uuid], scrub_text(str(e)), None)
        clean = _scrubbed_error(e)
        if clean is e:
            raise
        raise clean from None


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status of the answer a refused call raised with, if any."""
    return getattr(getattr(exc, "response", None), "status_code", None)


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
    _finalize(evaluation_id)


def _finalize(evaluation_id: uuid.UUID) -> None:
    """Close a run whose tests have all ended: Failed when one of them failed, Done otherwise."""
    logger.debug(f"Finalizing evaluation {evaluation_id}")
    try:
        # check if any plugins failed before marking as completed
        plugin_status = get_evaluation_plugins_status(evaluation_id)

        if plugin_status.get("has_failed_plugins", False):
            logger.info(
                f"Evaluation {evaluation_id} has failed plugins, marking as failed"
            )
            mark_failed(evaluation_id)
        else:
            response = mark_completed(evaluation_id)
            logger.debug(
                f"Evaluation {evaluation_id} marked as completed, status: {response.status_code}"
            )
    except Exception as e:
        logger.error(f"Failed to finalize evaluation {evaluation_id}: {e}")
        mark_failed(evaluation_id)
