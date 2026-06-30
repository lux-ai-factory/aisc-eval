import json
import os
import ssl
import subprocess
import tempfile
import urllib
import urllib.request
from urllib.error import URLError, HTTPError

import time
import uuid
from pathlib import Path

from celery import group, chain

from aisc_eval import plugin_runtime
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
)
from aisc_eval.utils.logging import get_logger

from aisc_plugin_manager.loader import Loader
from aisc_plugin_interface import TaskProgress, Measure
from aisc_eval.utils import env

logger = get_logger()

plugin_loader: Loader = Loader(env.PLUGIN_PATH, env.PACKAGE_REGISTRY_URL, env.PACKAGE_REGISTRY_INDEX,
                               env.PACKAGE_REGISTRY_USER,
                               env.PACKAGE_REGISTRY_PASSWORD)


def progress_callback(task_progress: TaskProgress, plugin_name: str, task_id: str):
    meta = {**task_progress.model_dump(), "plugin_name": plugin_name}
    celery_app.backend.store_result(task_id, meta, state="RUNNING")


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
        if plugin_info["source"] == "local":
            install_target = str(plugin_info["pkg_root"].resolve())
        else:
            install_target = f"{package_name}=={version}"

        with tempfile.TemporaryDirectory(delete=True) as cache_tmp:
            cache_venv = Path(cache_tmp)
            subprocess.run(
                ["uv", "venv", str(cache_venv), "--python", "/usr/local/bin/python"],
                capture_output=True, text=True, timeout=60, check=True,
            )
            install_cmd = ["uv", "pip", "install", "--python", str(cache_venv)]
            extra_url = plugin_loader.devpi_client.simple_index_url
            if extra_url:
                try:
                    urllib.request.urlopen(extra_url, timeout=2.0, context=ssl._create_unverified_context())
                except HTTPError:
                    pass
                except (URLError, TimeoutError, ValueError):
                    logger.warning(f"'{extra_url}' is unreachable. Skipping '--extra-index-url'.")
                    extra_url = None
                if extra_url:
                    install_cmd.extend(["--extra-index-url", extra_url])
            install_cmd.append(install_target)

            result = subprocess.run(
                install_cmd,
                capture_output=True, text=True, timeout=1000,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)

        logger.info(f"Successfully cached {package_name}=={version} and all dependencies")
    except Exception as e:
        logger.error(f"Failed to cache {package_name}=={version}: {e}", exc_info=True)
        if evaluation_pid and evaluation_plugin_pids:
            for pid in evaluation_plugin_pids:
                mark_plugin_failed(evaluation_pid, pid, str(e))
        raise


@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    plugin_loader.list_packages(refresh=True)

    evaluation: Evaluation = get_evaluation(evaluation_pid)

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
            if evaluation_plugin.plugin_config:
                config = evaluation_plugin.plugin_config.config

            input_file_definitions = []
            for input_file in evaluation_plugin.input_files:
                input_file_definitions.append(
                    {
                        "name": input_file.name,
                        "input_type": input_file.input_type,
                        "data": input_file.input_file.data,
                    }
                )

            run_plugin_sig = run_plugin.si(
                evaluation_plugin.package_name,
                evaluation_plugin.name,
                evaluation_plugin.version,
                config,
                input_file_definitions,
                evaluation_pid,
                evaluation_plugin.pid
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
               input_file_definitions: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID) -> list[dict]:

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

        input_mapping = {}
        for input_file_definition in input_file_definitions:
            if input_file_definition["input_type"] == "dataset":
                file_content = get_dataset_file_content(input_file_definition["data"])
            elif input_file_definition["input_type"] == "model":
                file_content = get_model_file_content(input_file_definition["data"])
            else:
                raise ValueError(
                    f"Unsupported file type: {input_file_definition['input_type']}"
                )

            file_path = input_dir / input_file_definition['data']
            file_path.write_bytes(file_content)

            input_mapping[input_file_definition["name"]] = f"{input_file_definition['data']}"

        config_data = {
            "plugin_source": f"{package_name}:{plugin_name}",
            "input_mapping": input_mapping,
            "plugin_config": plugin_config
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
            capture_output=True, text=True, timeout=1000,
        )
        if venv_result.returncode != 0:
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, venv_result.stderr)
            raise RuntimeError(f"Failed to create venv: {venv_result.stderr}")

        # Step 2: Install plugin and dependencies into the venv
        logger.debug(f"Installing {install_target} into isolated venv")
        install_result = subprocess.run(
            ["uv", "pip", "install", "--python", str(venv_dir), "--offline", install_target],
            capture_output=True, text=True, timeout=1000,
        )
        if install_result.returncode != 0:
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, install_result.stderr)
            raise RuntimeError(f"Failed to install plugin: {install_result.stderr}")

        # Step 3: Run plugin script with the isolated venv's Python
        logger.debug(f"Running plugin with: {venv_python} {runtime_script}")
        start_time = time.perf_counter()
        mark_plugin_started(evaluation_pid, evaluation_plugin_pid)

        stdout_lines = []
        stderr_path = workspace_path / "stderr.tmp"

        with open(stderr_path, "w+") as stderr_file:
            process = subprocess.Popen(
                [str(venv_python), str(runtime_script)],
                cwd=str(workspace_path),
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                bufsize=1
            )

            if (stdout := process.stdout) is not None:
                for line in stdout:
                    stdout_lines.append(line)
                    stripped_line = line.strip()

                    if stripped_line.startswith("{") and stripped_line.endswith("}"):
                        try:
                            progress_data = json.loads(stripped_line)
                            task_progress = TaskProgress(**progress_data)

                            progress_callback(task_progress, plugin_name, str(self.request.id))
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

        log_file = output_dir / "plugin_execution.log"
        log_content = "=== Plugin Execution Log ===\n\n"
        log_content += f"Execution Time: {duration:.2f} seconds\n\n"
        log_content += f"=== STDOUT ===\n{log_stdout_content}\n\n"
        log_content += f"=== STDERR ===\n{log_stderr_content}\n\n"
        log_content += f"=== Return Code ===\n{returncode}\n"
        log_file.write_text(log_content)

        if returncode != 0:
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, stderr_content)
            # revoke all sibling plugin tasks
            try:
                raw = celery_app.backend.client.get(f"eval_tasks:{evaluation_pid}")
                if raw:
                    for tid in json.loads(raw):
                        if tid != str(self.request.id):
                            celery_app.control.revoke(tid, terminate=True, signal="SIGTERM")
            except Exception as e:
                logger.warning(f"Could not revoke sibling tasks: {e}")
            raise RuntimeError(f"Plugin failed: {log_stderr_content}")

        measures_file = output_dir / "measures.json"
        measures = json.loads(measures_file.read_text()) if measures_file.exists() else []

        for file in output_dir.iterdir():
            if file.name != "measures.json" and file.name != "stderr.tmp":
                upload_artifact(evaluation_pid, evaluation_plugin_pid, file.name, file.read_bytes())

        mark_plugin_finished(evaluation_pid, evaluation_plugin_pid)
        return measures


@celery_app.task
def post_measurements(measurements_dict: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID):
    try:
        measurements = [Measure(**m) for m in measurements_dict]
        post_measures(evaluation_pid, evaluation_plugin_uuid, measurements)
    except Exception as e:
        mark_plugin_failed(evaluation_pid, evaluation_plugin_uuid, str(e))
        raise


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
    logger.error(f"Error in evaluation {evaluation_id}:")
    logger.error(f"--\n\n{request} {exc} {traceback}")
    mark_failed(evaluation_id)
    logger.error(f"Evaluation {evaluation_id} marked as failed due to error.")
