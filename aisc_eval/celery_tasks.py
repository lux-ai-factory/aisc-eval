import hashlib
import json
import os
import subprocess
import tempfile
import traceback as tracebacks

import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from celery import group, chain

from aisc_eval import plugin_process
from aisc_eval import run_context  # noqa: F401  (registers the run header handlers)
from aisc_eval.scrub import REDACTED, known_secrets, scrub_text
from aisc_eval.celery_app import celery_app
from aisc_eval.data_model.evaluation import Evaluation
from aisc_eval.service.api_client import (
    mark_completed,
    mark_failed,
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
from aisc_eval.utils.logging import get_logger

from aisc_plugin_manager.loader import Loader
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
            env_key = SECRET_ENV_PREFIX + "".join(
                character.upper() if character.isalnum() else "_"
                for character in name_source
            )
            values[env_key] = decrypt_value(setting["encrypted_value"])
    return values



#: What a plugin may see of the worker's environment. A plugin is catalogue code: the worker's
#: own credentials (broker and Redis URLs, INTERNAL_API_KEY, DJANGO_SECRET_KEY, the package
#: registry login) stay out, or a plugin could read other runs' queued tasks, lift their run
#: tickets and call the backend as another project. PLATFORM_CONNECTIONS_TOKEN stays out too: it
#: opens every project's connections; a plugin gets its run's pass instead (service/run_pass.py).
PLUGIN_ENV_NAMES = frozenset({
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    # the platform's connection resolver and target settings (aisc_plugin_interface.connections)
    "PLATFORM_URL", "CONNECTIONS_ALLOWED_HOSTS",
    "AISC_TARGET_BASE_URL", "AISC_TARGET_API_KEY",
    "OPENAI_BASE_URL", "API_KEY_OPENAI", "OPENAI_API_KEY",
})
PLUGIN_ENV_PROXIES = frozenset({"http_proxy", "https_proxy", "no_proxy", "all_proxy"})
#: LangBiTe builds an OpenAI client at init even with a local model and no judge; the client
#: checks the key only on a real request, so a non-empty placeholder lets it start
OPENAI_PLACEHOLDER = "sk-local-model-no-openai-call"


def plugin_environment(worker_env: Mapping[str, str], secrets: Mapping[str, str],
                       run_pass: str | None = None) -> dict[str, str]:
    """The environment a plugin runs with: the allowed part of the worker's, its project's secrets
    and, when one was minted, its run's pass to the platform."""
    env = {
        name: value for name, value in worker_env.items()
        if name in PLUGIN_ENV_NAMES or name.lower() in PLUGIN_ENV_PROXIES
        or name.startswith(SECRET_ENV_PREFIX)
    }
    env.update(secrets)
    if run_pass:
        env["PLATFORM_RUN_PASS"] = run_pass
    if not env.get("API_KEY_OPENAI"):
        env["API_KEY_OPENAI"] = OPENAI_PLACEHOLDER
    return env

LOCAL_MODEL_UNKNOWN = "could not tell whether the model file is a local target (the plugin could not be loaded)"
BOTH_KINDS = "the run binds both a local model file and an AI card target: a run assesses one of them"
MARKER_NOT_STORED = "could not store the local model file's name and hash on the run's input"


def local_model_marker(component: dict, file_content: bytes) -> dict:
    """What a run on a local model file was on: the file's name and the hash of the bytes the plugin gets."""
    return {"local_model": {"file": component.get("component_name") or component["data"],
                            "sha256": hashlib.sha256(file_content).hexdigest()}}


plugin_loader: Loader = Loader(env.PLUGIN_PATH, env.PACKAGE_REGISTRY_URL, env.PACKAGE_REGISTRY_INDEX,
                               env.PACKAGE_REGISTRY_USER,
                               env.PACKAGE_REGISTRY_PASSWORD)


def _scrubbed(value, known=()):
    """`value` with every string in it, at any depth, dict keys included, passed through scrub_text."""
    if isinstance(value, str):
        return scrub_text(value, known)
    if isinstance(value, dict):
        return {_scrubbed(key, known): _scrubbed(inner, known) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrubbed(item, known) for item in value]
    return value


def progress_callback(task_progress: TaskProgress, plugin_name: str, task_id: str, known=()):
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
        return {_without_known(key, known): _without_known(inner, known) for key, inner in value.items()}
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
    if text == str(exc) and args_clean and exc.__cause__ is None and exc.__context__ is None:
        return exc
    logger.error(scrub_text("".join(tracebacks.format_exception(exc)), known))
    return RuntimeError(f"{type(exc).__name__}: {text}")


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        except Exception:
            pass


def _revoke_sibling_tasks(evaluation_pid: uuid.UUID, current_task_id: str) -> None:
    try:
        raw = celery_app.backend.client.get(f"eval_tasks:{evaluation_pid}")
        if raw:
            for tid in json.loads(raw):
                if tid != str(current_task_id):
                    celery_app.control.revoke(tid, terminate=True, signal="SIGTERM")
    except Exception as e:
        logger.warning(f"Could not revoke sibling tasks: {e}")


def _fail_plugin_and_revoke(
    evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID, error_msg: str, current_task_id: str
) -> None:
    try:
        mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, error_msg)
    except Exception:
        pass
    _revoke_sibling_tasks(evaluation_pid, current_task_id)


@celery_app.task(bind=True)
def install_package(self, package_name: str, version: str, evaluation_pid: uuid.UUID, evaluation_plugin_pids: list[uuid.UUID]):
    """Install a package once using uv run to cache dependencies."""
    logger.info(f"Caching package {package_name}=={version}")

    if not plugin_loader.discovered_packages:
        plugin_loader.list_packages()

    if package_name not in plugin_loader.discovered_packages:
        raise KeyError(f"Package '{package_name}' not found.")

    available_versions = plugin_loader.discovered_packages[package_name]
    if version not in available_versions:
        raise KeyError(f"Version '{version}' of package '{package_name}' not found.")

    plugin_info = available_versions[version]

    try:
        installer = plugin_process.install_user()
        with tempfile.TemporaryDirectory(delete=True) as cache_tmp:
            if plugin_info["source"] == "local":
                install_target = str(plugin_process.stage_local_source(
                    plugin_info["pkg_root"].resolve(), Path(cache_tmp) / "source", installer))
            else:
                install_target = f"{package_name}=={version}"
            cache_venv = Path(cache_tmp) / "venv"
            plugin_process.prepare_install_folder(cache_venv, installer)
            made = plugin_process.run_install(
                ["uv", "venv", str(cache_venv), "--allow-existing", "--python", "/usr/local/bin/python"],
                installer, timeout=60,
            )
            if made.returncode != 0:
                raise RuntimeError(scrub_text(made.stderr))
            # Always pass the devpi index. (We used to probe it with a 2s urlopen and drop
            # --extra-index-url on timeout, but that probe hits the FULL /+simple/ root — which
            # lists the whole PyPI mirror and is slow — so it intermittently timed out and silently
            # dropped devpi, making registry plugins "not found". If devpi is truly down, let uv
            # fail loudly instead.)
            install_cmd = ["uv", "pip", "install", "--python", str(cache_venv)]
            extra_url = plugin_loader.devpi_client.simple_index_url
            if extra_url:
                install_cmd.extend(["--extra-index-url", extra_url])
            install_cmd.append(install_target)

            result = plugin_process.run_install(install_cmd, installer, timeout=1000)
            if result.returncode != 0:
                raise RuntimeError(scrub_text(result.stderr))

        logger.info(f"Successfully cached {package_name}=={version} and all dependencies")
    except Exception as e:
        # no traceback: an installer's message can hold a key, and only its scrubbed text is kept
        text = scrub_text(str(e))
        logger.error(f"Failed to cache {package_name}=={version}: {type(e).__name__}: {text}")
        if evaluation_pid and evaluation_plugin_pids:
            for pid in evaluation_plugin_pids:
                mark_plugin_failed(evaluation_pid, pid, text)
        raise


@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    plugin_loader.list_packages(refresh=True)

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

    logger.info(f"Found {len(plugins_by_pkg)} unique packages for {len(evaluation.evaluation_plugins)} plugins")

    # build per-package chains: install_pkg -> group(run_plugin -> post_measurements)
    package_chains = []
    plugin_task_ids = []
    for pkg_key, pkg_info in plugins_by_pkg.items():
        install_sig = install_package.si(
            pkg_info["package_name"],
            pkg_info["version"],
            evaluation_pid,
            [ep.pid for ep in pkg_info["plugins"]],
        )

        plugin_chain_list = []
        for evaluation_plugin in pkg_info["plugins"]:
            config = None
            project_config_selections = []
            if evaluation_plugin.plugin_config:
                config = evaluation_plugin.plugin_config.config
                project_config_selections = evaluation_plugin.plugin_config.project_config_selections

            input_components = inputs_by_plugin.get(str(evaluation_plugin.pid), [])

            run_plugin_sig = run_plugin.si(
                evaluation_plugin.package_name,
                evaluation_plugin.name,
                evaluation_plugin.version,
                config,
                input_components,
                get_project_settings_by_pid(evaluation.project.pid, project_config_selections),
                evaluation_pid,
                evaluation_plugin.pid,
            )

            # freeze to get a stable task id before dispatching
            run_plugin_sig.freeze()
            plugin_task_ids.append(str(run_plugin_sig.id))

            post_measurements_sig = post_measurements.s(evaluation_pid, evaluation_plugin.pid)
            plugin_chain_list.append(chain(run_plugin_sig, post_measurements_sig))

        package_chains.append(chain(install_sig, group(plugin_chain_list)))

    # dispatch: each package's plugins start after its install, no waiting for other packages
    workflow = group(package_chains) | finalize_evaluation.si(evaluation_pid)
    workflow.apply_async()

    # store plugin task ids in redis so siblings can be revoked on failure
    celery_app.backend.client.set(
        f"eval_tasks:{evaluation_pid}", json.dumps(plugin_task_ids), ex=7200
    )

    return {"evaluation_pid": evaluation_pid, "plugin_task_ids": plugin_task_ids}


@celery_app.task(bind=True)
def run_plugin(self, package_name: str, plugin_name: str, version: str, plugin_config: dict,
               input_components: list[dict], project_settings: list[dict],
               evaluation_pid: uuid.UUID,
               evaluation_plugin_pid: uuid.UUID) -> list[dict]:

    if not plugin_loader.discovered_packages:
        logger.debug("Worker cache empty. Fetching packages from Devpi...")
        plugin_loader.list_packages()

    # abort early if another plugin already failed this evaluation
    try:
        eval_data = get_evaluation_request(evaluation_pid)
        if eval_data.get("status") == "Failed":
            logger.info(f"Evaluation {evaluation_pid} already failed — skipping plugin {plugin_name}")
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, "Aborted: evaluation already failed")
            return []
    except Exception:
        pass

    if package_name not in plugin_loader.discovered_packages:
        raise KeyError(f"Package '{package_name}' not found.")

    available_versions = plugin_loader.discovered_packages[package_name]
    if version not in available_versions:
        raise KeyError(f"Version '{version}' of package '{package_name}' not found.")

    plugin_info = available_versions[version]

    with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
        workspace_path = Path(tmp_dir)

        # Setup internal workspace structure
        input_dir = workspace_path / "input"
        output_dir = workspace_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # the backend could not load the plugin to say whether its model file is the local target:
        # running anyway could store a local run as if it were an AI card run
        if any(component.get("local_model_unknown") for component in input_components):
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, LOCAL_MODEL_UNKNOWN)
            raise RuntimeError(LOCAL_MODEL_UNKNOWN)

        input_mapping = {}
        llm_secret_settings = []
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
                    local_model = {"input": component.get("pid"), "marker": local_model_marker(component, file_content)}
            elif component_type in {"datashape", "llm", "resource"}:
                payload = dict(component.get("json_value") or {})
                if component_type == "llm":
                    run_model = (component.get("value") or {}).get("model")
                    if run_model:
                        payload["model"] = run_model
                    if component.get("secret_key") and component.get("secret_encrypted_value"):
                        llm_secret_settings.append({
                            "category": "secrets",
                            "key": component["secret_key"],
                            "encrypted_value": component["secret_encrypted_value"],
                        })
                file_content = json.dumps(payload).encode("utf-8")
                relative_path = f"{name}.json"
            else:
                raise ValueError(
                    f"Unsupported component type: {component_type}"
                )

            (input_dir / relative_path).write_bytes(file_content)
            input_mapping[name] = relative_path

        # A run on a local model file says so, here for the plugin and on its input for the results:
        # it is never an AI card run, and it opens nothing on the platform (no run pass)
        if local_model is not None:
            if run_pass.refs_of(input_components):
                mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, BOTH_KINDS)
                raise ValueError(BOTH_KINDS)
            (input_dir / "target.json").write_text(json.dumps(local_model["marker"]))
            try:
                store_input_value(evaluation_pid, local_model["input"], local_model["marker"])
            except Exception as exc:
                # the person reads this: the answer's status, never the backend's address
                status = getattr(getattr(exc, "response", None), "status_code", None)
                message = f"{MARKER_NOT_STORED} ({f'HTTP {status}' if status else 'no answer'})"
                mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, message)
                raise RuntimeError(message) from None

        config_data = {
            "plugin_source": f"{package_name}:{plugin_name}",
            "input_mapping": input_mapping,
            "project_settings": build_project_settings(project_settings),
            "plugin_config": plugin_config or {},
        }

        config_path = workspace_path / "config.json"
        config_path.write_text(json.dumps(config_data, indent=2))

        runtime_script = plugin_process.copy_runtime(workspace_path)

        installer = plugin_process.install_user()
        if plugin_info["source"] == "local":
            install_target = str(plugin_process.stage_local_source(
                plugin_info["pkg_root"].resolve(), workspace_path / "source", installer))
        else:
            install_target = f"{package_name}=={version}"

        venv_dir = workspace_path / "venv"
        venv_python = venv_dir / "bin" / "python"

        # Step 1: Create isolated venv, as the install user: the plugin may run it but not change it
        logger.debug(f"Creating isolated venv at {venv_dir}")
        plugin_process.prepare_install_folder(venv_dir, installer)
        venv_result = plugin_process.run_install(
            ["uv", "venv", str(venv_dir), "--allow-existing", "--python", "/usr/local/bin/python"],
            installer, timeout=1000,
        )
        if venv_result.returncode != 0:
            failure = scrub_text(venv_result.stderr or "")
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, failure)
            raise RuntimeError(f"Failed to create venv: {failure}")

        # Step 2: Install plugin and dependencies into the venv (online, from devpi).
        # NOTE: we do NOT use --offline here. The evaluation run needs the network anyway (e.g.
        # GPT4ALL downloads its model at run time), so offline isolation gave nothing, and it broke
        # for registry plugins whose shared dep (aisc-plugin-interface) is resolved via a local
        # workspace source and therefore never cached under the devpi index for the offline resolver.
        # Installing online resolves everything from the devpi index (root/public bases root/pypi),
        # exactly like install_package already does successfully.
        logger.debug(f"Installing {install_target} into isolated venv")
        install_cmd = ["uv", "pip", "install", "--python", str(venv_dir)]
        extra_url = plugin_loader.devpi_client.simple_index_url
        if extra_url:
            install_cmd.extend(["--extra-index-url", extra_url])
        install_cmd.append(install_target)
        install_result = plugin_process.run_install(install_cmd, installer, timeout=1000)
        if install_result.returncode != 0:
            failure = scrub_text(install_result.stderr or "")
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, failure)
            raise RuntimeError(f"Failed to install plugin: {failure}")

        # A pass to the platform for exactly the targets and connections this run names
        minted: str | None = None
        refs = run_pass.refs_of(input_components)
        if refs:
            try:
                minted = run_pass.mint(refs, str(evaluation_plugin_pid))
            except run_pass.RunPassRefused as exc:
                mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, scrub_text(str(exc)))
                raise run_pass.RunPassRefused(scrub_text(str(exc))) from None

        # Step 3: Run plugin script with the isolated venv's Python
        logger.debug(f"Running plugin with: {venv_python} {runtime_script}")
        start_time = time.perf_counter()
        mark_plugin_started(evaluation_pid, evaluation_plugin_pid)

        stdout_lines: list[str] = []
        stderr_path = workspace_path / "stderr.tmp"
        process: subprocess.Popen | None = None
        returncode: int | None = None
        stderr_content = ""
        # the values this run hands its plugin, removed from every text the worker reports of it
        known: list[str] = []
        try:
            secret_env = build_secret_environment(project_settings + llm_secret_settings)
            known = known_secrets(secret_env, os.environ, plugin_config, minted)
            user = plugin_process.plugin_user()
            plugin_process.prepare_workspace(workspace_path, output_dir, user)
            with open(stderr_path, "w+") as stderr_file:
                child_env = plugin_environment(os.environ, secret_env, minted)
                process = plugin_process.start(
                    [str(venv_python), str(runtime_script)], workspace_path, stderr_file,
                    child_env, user,
                )

                if (stdout := process.stdout) is not None:
                    for line in stdout:
                        stdout_lines.append(line)
                        stripped_line = line.strip()

                        if stripped_line.startswith("{") and stripped_line.endswith("}"):
                            try:
                                progress_data = json.loads(stripped_line)
                                task_progress = TaskProgress(**progress_data)

                                progress_callback(task_progress, plugin_name, str(self.request.id), known)
                            except Exception:
                                pass

                returncode = process.wait()

                stderr_file.seek(0)
                stderr_content = stderr_file.read()

            end_time = time.perf_counter()
            duration = end_time - start_time
            stdout_content = "".join(stdout_lines)

            log_stdout_content = (
                "====== Step 1: Venv Creation ======\n" + (venv_result.stdout or "") + "\n\n"
                "====== Step 2: Plugin Install ======\n" + (install_result.stdout or "") + "\n\n"
                "====== Step 3: Plugin Run ======\n" + stdout_content + "\n\n"
            )
            log_stderr_content = (
                "====== Step 1: Venv Creation ======\n" + (venv_result.stderr or "") + "\n\n"
                "====== Step 2: Plugin Install ======\n" + (install_result.stderr or "") + "\n\n"
                "====== Step 3: Plugin Run ======\n" + stderr_content + "\n\n"
            )

            log_content = "=== Plugin Execution Log ===\n\n"
            log_content += f"Execution Time: {duration:.2f} seconds\n\n"
            log_content += f"=== STDOUT ===\n{log_stdout_content}\n\n"
            log_content += f"=== STDERR ===\n{log_stderr_content}\n\n"
            log_content += f"=== Return Code ===\n{returncode}\n"
            log_content = scrub_text(log_content, known)

            if returncode != 0:
                _fail_plugin_and_revoke(
                    evaluation_pid, evaluation_plugin_pid, scrub_text(stderr_content, known), str(self.request.id)
                )
                raise RuntimeError(f"Plugin failed: {scrub_text(log_stderr_content, known)}")

            outputs = plugin_process.read_outputs(output_dir, user)
            measures = json.loads(outputs.pop("measures.json", b"[]"))

            # the log is the worker's, kept out of output/ where the plugin could plant a link
            outputs["plugin_execution.log"] = log_content.encode()
            for name, content in outputs.items():
                upload_artifact(evaluation_pid, evaluation_plugin_pid, name, content)

            mark_plugin_finished(evaluation_pid, evaluation_plugin_pid)
            # what the chain hands post_measurements (and the result backend keeps): no value the plugin was given
            return _without_known(measures, known)
        except Exception as exc:
            _terminate_process(process)
            _fail_plugin_and_revoke(
                evaluation_pid, evaluation_plugin_pid, scrub_text(str(exc), known) or exc.__class__.__name__,
                str(self.request.id)
            )
            clean = _scrubbed_error(exc, known)
            if clean is exc:
                raise
            raise clean from None
        finally:
            if minted:
                run_pass.revoke(minted)


@celery_app.task
def post_measurements(measurements_dict: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID):
    # run_plugin returns the measures without the values it handed the plugin; key shapes are scrubbed here
    try:
        measurements = [Measure(**m) for m in measurements_dict]
        post_measures(evaluation_pid, evaluation_plugin_uuid, measurements)
    except Exception as e:
        mark_plugin_failed(evaluation_pid, evaluation_plugin_uuid, scrub_text(str(e)))
        clean = _scrubbed_error(e)
        if clean is e:
            raise
        raise clean from None


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
    logger.debug(f"Finalizing evaluation {evaluation_id}")
    try:
        # check if any plugins failed before marking as completed
        plugin_status = get_evaluation_plugins_status(evaluation_id)

        if plugin_status.get("has_failed_plugins", False):
            logger.info(f"Evaluation {evaluation_id} has failed plugins, marking as failed")
            mark_failed(evaluation_id)
        else:
            response = mark_completed(evaluation_id)
            logger.debug(
                f"Evaluation {evaluation_id} marked as completed, status: {response.status_code}"
            )
    except Exception as e:
        logger.error(f"Failed to finalize evaluation {evaluation_id}: {e}")
        mark_failed(evaluation_id)


@celery_app.task
def handle_error(
        evaluation_id: uuid.UUID,
        request: object,
        exc: BaseException,
        traceback: object,
) -> None:
    # never the request (it holds the plugin configuration) nor the raw failure: its type and scrubbed text
    known = known_secrets({}, {}, _request_config(request), None)
    logger.error(f"Error in evaluation {evaluation_id}: {type(exc).__name__}: {scrub_text(str(exc), known)}")
    mark_failed(evaluation_id)
    logger.error(f"Evaluation {evaluation_id} marked as failed due to error.")


def _request_config(request: object):
    """The plugin configuration among a task request's arguments (run_plugin's fourth), if one is there."""
    args = request.get("args") if isinstance(request, dict) else getattr(request, "args", None)
    if isinstance(args, (list, tuple)):
        return [arg for arg in args if isinstance(arg, dict)]
    return None
