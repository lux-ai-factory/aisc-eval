import base64
import json
import os
import io
import secrets as pysecrets
import ssl
import subprocess
import tempfile
import urllib
import urllib.request
from urllib.error import URLError, HTTPError

import time
import uuid
from pathlib import Path

from aisc_eval import door, plugin_runtime
from aisc_eval.celery_app import celery_app, StopAwareTask
from aisc_eval.service.api_client import (
    mark_failed,
    mark_plugin_started,
    mark_plugin_finished,
    mark_plugin_failed,
    post_measures,
    post_provenance,
    upload_artifact,
)
from aisc_eval.utils.logging import get_logger
from aisc_eval.utils.scrub import scrub_bytes, scrub_secrets

from aisc_plugin_manager.loader import Loader
from aisc_plugin_interface import TaskProgress, Measure
from aisc_eval.utils import env
from aisc_eval.utils.encryption import decrypt_value
from aisc_eval.services.datashape_validation import validate_dataframe_against_datashape

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


_PLUGIN_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TZ",
    "UV_CACHE_DIR",
    "UV_PYTHON",
)


def plugin_child_env() -> dict[str, str]:
    """The environment a plugin subprocess may see.

    A plugin is untrusted catalogue code. It gets a small allowlist of the
    worker's environment instead of a copy of it, so no worker credential
    (database, registry, keycloak) reaches the plugin. Run secrets are added by
    the caller under the ``AISC_SECRET_`` prefix.
    """
    return {
        name: os.environ[name] for name in _PLUGIN_ENV_ALLOWLIST if name in os.environ
    }


def plugin_run_env(
    run_secrets: dict[str, str], *, run_local: bool = False
) -> dict[str, str]:
    """The child environment for a plugin run (EXEC-01 #120).

    Always the SEC-01 allowlist (:func:`plugin_child_env`). A normal run adds the
    per-run target secrets the dispatcher decrypted under ``AISC_SECRET_*``. A
    *run-local* (native) run executes on the worker's own host, beside the system
    it assesses, so it withholds those credentials: the run cannot exfiltrate the
    target's key/secret.
    """
    child_env = plugin_child_env()
    if run_local:
        return {
            name: value
            for name, value in child_env.items()
            if name not in run_secrets and not name.startswith(SECRET_ENV_PREFIX)
        }
    child_env.update(run_secrets)
    return child_env


def _decrypt_or_empty(value: str | None) -> str:
    if not value:
        return ""
    try:
        return decrypt_value(value)
    except Exception:
        return ""


def _adapter_spec(component: dict) -> dict:
    """How the plugin runtime resolves a data adapter for one input (ADR 0009)."""
    target = component.get("install_target") or {}
    return {
        "adapter": component.get("adapter") or "",
        "config": component.get("adapter_config") or {},
        "module": target.get("module", ""),
        "class": target.get("class_name", ""),
    }


def _door_slots(input_components: list[dict]) -> tuple[dict, dict]:
    """The door spec's slots, and the run tokens the plugin is handed (ADR 0009).

    The real key is decrypted here only to put it on the door's stdin; it is never
    added to the plugin's environment. Each slot gets its own run token.
    """
    slots: dict[str, dict] = {}
    tokens: dict[str, str] = {}
    for component in input_components:
        if component.get("component_type") != "endpoint":
            continue
        name = component["name"]
        token_key = f"door_{name}"
        token = pysecrets.token_urlsafe(24)
        # The plugin reads the token with `self.get_secret(llm.secret_key)`, which
        # looks up `AISC_SECRET_<normalized key>` — so the child env var must carry
        # that name, not the bare key (ADR 0009 D-16).
        token_env = SECRET_ENV_PREFIX + "".join(
            character.upper() if character.isalnum() else "_"
            for character in token_key
        )
        tokens[token_env] = token
        target = component.get("install_target") or {}
        slots[name] = {
            "adapter": component.get("adapter") or "",
            "version": component.get("adapter_version") or "",
            "module": target.get("module", ""),
            "class": target.get("class_name", ""),
            "face": component.get("face") or component.get("protocol") or "openai-chat",
            "protocol": component.get("protocol") or "",
            "url": component.get("url") or "",
            "config": component.get("adapter_config") or {},
            "secret": _decrypt_or_empty(component.get("secret_encrypted_value")),
            "secret_headers": {
                header: _decrypt_or_empty(value)
                for header, value in (component.get("secret_headers") or {}).items()
            },
            "token": token,
            "allowed": list(component.get("allowed_hosts") or []),
            "floor": list(component.get("floor_hosts") or []),
            "stack": list(component.get("stack_hosts") or []),
            "config_hash": component.get("config_hash") or "",
            "timeout_s": 60,
            "max_concurrency": 8,
        }
    return slots, tokens


def _endpoint_input_file(component: dict, door_url: str, token_key: str) -> dict:
    """The input a plugin reads for an endpoint slot: a door URL and a run token."""
    face = component.get("face") or component.get("protocol") or "openai-chat"
    model = (
        (component.get("value") or {}).get("model")
        or (component.get("adapter_config") or {}).get("model")
        or ""
    )
    if face == "openai-chat":
        return {"endpoint_url": door_url, "secret_key": token_key, "model": model}
    return {"url": door_url, "protocol": face, "model": model, "token_key": token_key}


def _adapter_install_specs(input_components: list[dict], loader) -> list[str]:
    """The extra packages a run's non-builtin adapters need, as uv install targets."""
    specs: list[str] = []
    seen: set[tuple[str, str]] = set()
    for component in input_components:
        target = component.get("install_target")
        if not target:
            continue
        package, version = target.get("package"), target.get("version")
        if not package or (package, version) in seen:
            continue
        seen.add((package, version))
        if target.get("source") == "local":
            info = (loader.discovered_adapters.get(package) or {}).get(version)
            if info and info.get("pkg_root"):
                specs.append(str(info["pkg_root"].resolve()))
                continue
        specs.append(f"{package}=={version}")
    return specs


def _start_door(venv_python, workspace_path, slots: dict, log_path) -> tuple[subprocess.Popen, int]:
    """Start the run door in the plugin's venv and return it and its port."""
    process = subprocess.Popen(
        [str(venv_python), os.path.abspath(door.__file__)],
        cwd=str(workspace_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(log_path, "w"),
        env=plugin_child_env(),
        text=True,
        bufsize=1,
    )
    process.stdin.write(json.dumps({"workspace": str(workspace_path), "slots": slots}))
    process.stdin.close()
    line = process.stdout.readline()
    if not line:
        _terminate_process(process)
        raise RuntimeError("the run door did not start")
    return process, int(json.loads(line)["port"])


def _send_door_provenance(evaluation_pid, evaluation_plugin_pid, workspace_path) -> None:
    """Send the door's call counts (best-effort: the run has already finished)."""
    path = Path(workspace_path) / "door_provenance.json"
    if not path.exists():
        return
    try:
        provenance = json.loads(path.read_text())
        if provenance:
            post_provenance(evaluation_pid, {str(evaluation_plugin_pid): provenance})
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning(f"Could not post door provenance: {exc}")


def _decode_input_file(component: dict) -> bytes:
    """The bytes the run was dispatched with; a run is self-contained (I1)."""
    encoded = component.get("content_b64")
    if not encoded:
        raise ValueError(
            f"input '{component.get('name')}' was dispatched without its bytes; "
            "the run must be self-contained"
        )
    return base64.b64decode(encoded)


def validate_input_file(content: bytes, file_name: str, datashape: dict) -> dict:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        frame = __import__("pandas").read_csv(io.BytesIO(content))
    elif suffix == ".parquet":
        frame = __import__("pandas").read_parquet(io.BytesIO(content))
    else:
        return {"errors": [], "warnings": []}
    return validate_dataframe_against_datashape(frame, datashape)


plugin_loader: Loader = Loader(
    env.PLUGIN_PATH,
    env.PACKAGE_REGISTRY_URL,
    env.PACKAGE_REGISTRY_INDEX,
    env.PACKAGE_REGISTRY_USER,
    env.PACKAGE_REGISTRY_PASSWORD,
)


def progress_callback(task_progress: TaskProgress, plugin_name: str, task_id: str):
    meta = {**task_progress.model_dump(), "plugin_name": plugin_name}
    celery_app.backend.store_result(task_id, meta, state="RUNNING")


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
    evaluation_pid: uuid.UUID,
    evaluation_plugin_pid: uuid.UUID,
    error_msg: str,
    current_task_id: str,
) -> None:
    try:
        mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, error_msg)
    except Exception:
        pass
    _revoke_sibling_tasks(evaluation_pid, current_task_id)


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

    if not plugin_loader.discovered_packages:
        plugin_loader.list_packages()

    if package_name not in plugin_loader.discovered_packages:
        raise KeyError(f"Package '{package_name}' not found.")

    available_versions = plugin_loader.discovered_packages[package_name]
    if version not in available_versions:
        raise KeyError(f"Version '{version}' of package '{package_name}' not found.")

    plugin_info = available_versions[version]

    try:
        if plugin_info["source"] == "local":
            install_target = str(plugin_info["pkg_root"].resolve())
        else:
            install_target = f"{package_name}=={version}"

        with tempfile.TemporaryDirectory(delete=True) as cache_tmp:
            cache_venv = Path(cache_tmp)
            subprocess.run(
                ["uv", "venv", str(cache_venv), "--python", "/usr/local/bin/python"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
                env=plugin_child_env(),
            )
            install_cmd = ["uv", "pip", "install", "--python", str(cache_venv)]
            extra_url = plugin_loader.devpi_client.simple_index_url
            if extra_url:
                try:
                    urllib.request.urlopen(
                        extra_url, timeout=2.0, context=ssl._create_unverified_context()
                    )
                except HTTPError:
                    pass
                except (URLError, TimeoutError, ValueError):
                    logger.warning(
                        f"'{extra_url}' is unreachable. Skipping '--extra-index-url'."
                    )
                    extra_url = None
                if extra_url:
                    install_cmd.extend(["--extra-index-url", extra_url])
            install_cmd.append(install_target)

            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=1000,
                env=plugin_child_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)

        logger.info(
            f"Successfully cached {package_name}=={version} and all dependencies"
        )
    except Exception as e:
        logger.error(f"Failed to cache {package_name}=={version}: {e}", exc_info=True)
        if evaluation_plugin_pids:
            for pid in evaluation_plugin_pids:
                mark_plugin_failed(evaluation_pid, pid, str(e))
        if evaluation_pid:
            # Mark the eval failed here too: the link_error errback may not fire
            # for chord/chain members, which would leave the eval Pending forever.
            mark_failed(evaluation_pid)
        raise


@celery_app.task(bind=True, base=StopAwareTask)
def run_plugin(self, package_name: str, plugin_name: str, version: str, plugin_config: dict,
               input_components: list[dict], project_settings: list[dict],
               evaluation_pid: uuid.UUID,
               evaluation_plugin_pid: uuid.UUID,
               run_local: bool = False) -> list[dict]:

    if not plugin_loader.discovered_packages:
        logger.debug("Worker cache empty. Fetching packages from Devpi...")
        plugin_loader.list_packages()

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

        input_mapping = {}
        input_adapters: dict[str, dict] = {}
        pending_inputs: dict[str, bytes] = {}
        try:
            for component in input_components:
                component_type = component["component_type"]
                name = component["name"]
                if component_type in ("dataset", "model"):
                    pending_inputs[name] = _decode_input_file(component)
                    input_mapping[name] = component["data"]
                    if component.get("adapter"):
                        input_adapters[name] = _adapter_spec(component)
                elif component_type in ("datashape", "resource"):
                    pending_inputs[name] = json.dumps(
                        dict(component.get("json_value") or {})
                    ).encode("utf-8")
                    input_mapping[name] = f"{name}.json"
                elif component_type == "endpoint":
                    # written once the door's port is known
                    input_mapping[name] = f"{name}.json"
                else:
                    raise ValueError(f"Unsupported component type: {component_type}")
        except Exception as exc:
            _fail_plugin_and_revoke(
                evaluation_pid, evaluation_plugin_pid,
                str(exc) or exc.__class__.__name__, str(self.request.id)
            )
            raise

        door_slots, door_tokens = _door_slots(input_components)

        config_data = {
            "plugin_source": f"{package_name}:{plugin_name}",
            "input_mapping": input_mapping,
            "input_adapters": input_adapters,
            "project_settings": build_project_settings(project_settings),
            "plugin_config": plugin_config or {},
        }

        config_path = workspace_path / "config.json"
        config_path.write_text(json.dumps(config_data, indent=2))

        runtime_script = os.path.abspath(plugin_runtime.__file__)

        if plugin_info["source"] == "local":
            install_target = str(plugin_info["pkg_root"].resolve())
        else:
            install_target = f"{package_name}=={version}"

        venv_dir = workspace_path / "venv"
        venv_python = venv_dir / "bin" / "python"

        # Step 1: Create isolated venv
        logger.debug(f"Creating isolated venv at {venv_dir}")
        venv_result = subprocess.run(
            ["uv", "venv", str(venv_dir), "--python", "/usr/local/bin/python"],
            capture_output=True,
            text=True,
            timeout=1000,
            env=plugin_child_env(),
        )
        if venv_result.returncode != 0:
            mark_plugin_failed(
                evaluation_pid, evaluation_plugin_pid, venv_result.stderr
            )
            raise RuntimeError(f"Failed to create venv: {venv_result.stderr}")

        # Step 2: Install plugin and dependencies into the venv
        adapter_specs = _adapter_install_specs(input_components, plugin_loader)
        endpoint_extra = (
            ["aisc-plugin-interface[endpoint]"]
            if any(c.get("component_type") == "endpoint" for c in input_components)
            else []
        )
        logger.debug(f"Installing {install_target} into isolated venv")
        install_result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_dir),
                "--offline",
                install_target,
                *adapter_specs,
                *endpoint_extra,
            ],
            capture_output=True,
            text=True,
            timeout=1000,
            env=plugin_child_env(),
        )
        if install_result.returncode != 0:
            mark_plugin_failed(
                evaluation_pid, evaluation_plugin_pid, install_result.stderr
            )
            raise RuntimeError(f"Failed to install plugin: {install_result.stderr}")

        # The run door: started before the plugin, in the same venv, and stopped
        # with it. Its spec (which carries the real key) goes on stdin, never the
        # plugin's environment (ADR 0009, I2).
        door_process: subprocess.Popen | None = None
        door_port: int | None = None
        try:
            if door_slots:
                door_process, door_port = _start_door(
                    venv_python, workspace_path, door_slots, workspace_path / "door.tmp"
                )
            for component in input_components:
                name = component["name"]
                path = input_dir / input_mapping[name]
                if component["component_type"] == "endpoint":
                    door_url = f"http://127.0.0.1:{door_port}/{name}"
                    path.write_bytes(
                        json.dumps(
                            _endpoint_input_file(component, door_url, f"door_{name}")
                        ).encode("utf-8")
                    )
                else:
                    path.write_bytes(pending_inputs[name])
        except Exception as exc:
            _terminate_process(door_process)
            _fail_plugin_and_revoke(
                evaluation_pid, evaluation_plugin_pid,
                str(exc) or exc.__class__.__name__, str(self.request.id)
            )
            raise

        # Step 3: Run plugin script with the isolated venv's Python
        logger.debug(f"Running plugin with: {venv_python} {runtime_script}")
        start_time = time.perf_counter()
        mark_plugin_started(evaluation_pid, evaluation_plugin_pid)

        # The plugin's own secrets plus its run tokens; the target's real key is
        # only ever in the door's stdin. A run-local run still gets the tokens (they
        # are not the target's credential).
        plugin_secrets = build_secret_environment(project_settings)
        run_secrets = {**plugin_secrets, **door_tokens}
        secret_values = list(run_secrets.values())
        stdout_lines: list[str] = []
        stderr_path = workspace_path / "stderr.tmp"
        process: subprocess.Popen | None = None
        returncode: int | None = None
        stderr_content = ""
        try:
            with open(stderr_path, "w+") as stderr_file:
                child_env = plugin_run_env(plugin_secrets, run_local=run_local)
                child_env.update(door_tokens)
                process = subprocess.Popen(
                    [str(venv_python), str(runtime_script)],
                    cwd=str(workspace_path),
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    env=child_env,
                    text=True,
                    bufsize=1,
                )

                if (stdout := process.stdout) is not None:
                    for line in stdout:
                        stdout_lines.append(line)
                        stripped_line = line.strip()

                        if stripped_line.startswith("{") and stripped_line.endswith(
                            "}"
                        ):
                            try:
                                progress_data = json.loads(stripped_line)
                                task_progress = TaskProgress(**progress_data)

                                progress_callback(
                                    task_progress, plugin_name, str(self.request.id)
                                )
                            except Exception:
                                pass

                returncode = process.wait()
                _terminate_process(door_process)
                _send_door_provenance(evaluation_pid, evaluation_plugin_pid, workspace_path)

                stderr_file.seek(0)
                stderr_content = stderr_file.read()

            end_time = time.perf_counter()
            duration = end_time - start_time
            stdout_content = scrub_secrets("".join(stdout_lines), secret_values)
            stderr_content = scrub_secrets(stderr_content, secret_values)

            log_stdout_content = scrub_secrets(
                "====== Step 1: Venv Creation ======\n"
                + (venv_result.stdout or "")
                + "\n\n"
                "====== Step 2: Plugin Install ======\n"
                + (install_result.stdout or "")
                + "\n\n"
                "====== Step 3: Plugin Run ======\n" + stdout_content + "\n\n",
                secret_values,
            )
            log_stderr_content = scrub_secrets(
                "====== Step 1: Venv Creation ======\n"
                + (venv_result.stderr or "")
                + "\n\n"
                "====== Step 2: Plugin Install ======\n"
                + (install_result.stderr or "")
                + "\n\n"
                "====== Step 3: Plugin Run ======\n" + stderr_content + "\n\n",
                secret_values,
            )

            log_file = output_dir / "plugin_execution.log"
            log_content = "=== Plugin Execution Log ===\n\n"
            log_content += f"Execution Time: {duration:.2f} seconds\n\n"
            log_content += f"=== STDOUT ===\n{log_stdout_content}\n\n"
            log_content += f"=== STDERR ===\n{log_stderr_content}\n\n"
            log_content += f"=== Return Code ===\n{returncode}\n"
            log_file.write_text(log_content)

            if returncode != 0:
                _fail_plugin_and_revoke(
                    evaluation_pid,
                    evaluation_plugin_pid,
                    stderr_content,
                    str(self.request.id),
                )
                raise RuntimeError(f"Plugin failed: {log_stderr_content}")

            measures_file = output_dir / "measures.json"
            measures = (
                json.loads(measures_file.read_text()) if measures_file.exists() else []
            )

            for file in output_dir.iterdir():
                if file.name != "measures.json" and file.name != "stderr.tmp":
                    upload_artifact(
                        evaluation_pid,
                        evaluation_plugin_pid,
                        file.name,
                        scrub_bytes(file.read_bytes(), secret_values),
                    )

            mark_plugin_finished(evaluation_pid, evaluation_plugin_pid)
            return measures
        except (Exception, SystemExit) as exc:
            # SystemExit = revoked with terminate=True: kill the plugin's
            # subprocess before the base task converts the abort to a
            # serializable failure, otherwise it keeps running orphaned.
            _terminate_process(process)
            _terminate_process(door_process)
            _fail_plugin_and_revoke(
                evaluation_pid,
                evaluation_plugin_pid,
                scrub_secrets(str(exc) or exc.__class__.__name__, secret_values),
                str(self.request.id),
            )
            raise


@celery_app.task
def post_measurements(
    measurements_dict: list[dict],
    evaluation_pid: uuid.UUID,
    evaluation_plugin_uuid: uuid.UUID,
):
    try:
        measurements = [Measure(**m) for m in measurements_dict]
        post_measures(evaluation_pid, evaluation_plugin_uuid, measurements)
    except Exception as e:
        mark_plugin_failed(evaluation_pid, evaluation_plugin_uuid, str(e))
        raise


@celery_app.task
def handle_error(
    evaluation_id: uuid.UUID,
    request: object,
    exc: BaseException,
    traceback: object,
) -> None:
    logger.error(f"Error in evaluation {evaluation_id}:")
    logger.error(f"--\n\n{request} {exc} {traceback}")
    mark_failed(evaluation_id)
    logger.error(f"Evaluation {evaluation_id} marked as failed due to error.")
