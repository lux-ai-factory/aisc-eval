import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from celery import group, chain

from vera_eval import plugin_runtime
from vera_eval.celery_app import celery_app
from vera_eval.data_model.evaluation import Evaluation
from vera_eval.data_model.measure import Measure
from vera_eval.service.api_client import (
    mark_completed,
    mark_failed,
    mark_plugin_started,
    mark_plugin_finished,
    mark_plugin_failed,
    post_measures,
    get_evaluation,
    get_dataset_file_content,
    get_model_file_content,
    upload_artifact,
)
from vera_eval.utils.logging import get_logger

from vera_plugin_manager.loader import Loader
from vera_plugin_interface import TaskProgress
from vera_eval.utils import env

logger = get_logger()

plugin_loader: Loader = Loader(env.PLUGIN_PATH, env.PACKAGE_REGISTRY_URL, env.PACKAGE_REGISTRY_INDEX,
                               env.PACKAGE_REGISTRY_USER,
                               env.PACKAGE_REGISTRY_PASSWORD)


def progress_callback(task_progress: TaskProgress, plugin_name: str, task_id: str):
    meta = {**task_progress.model_dump(), "plugin_name": plugin_name}
    celery_app.backend.store_result(task_id, meta, state="RUNNING")


@celery_app.task(bind=True)
def install_package(self, package_name: str, version: str):
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
        # Use uv run with --with to cache package and all dependencies
        if plugin_info["source"] == "local":
            install_target = str(plugin_info["pkg_root"].resolve())
        else:
            install_target = f"{package_name}=={version}"

        cmd = [
            "uv", "run", "-v",
            "--no-project",
            "--extra-index-url", plugin_loader.devpi_client.simple_index_url,
            "--with", install_target,
        ]

        if plugin_info["source"] != "local":
            cmd.extend(["--with", "vera-plugin-interface @ git+https://github.com/lux-ai-factory/vera-plugin-interface.git@v0.2.3"])

        # run a print command
        cmd.extend(["python", "-c", "print('Package and dependencies cached successfully')"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1000,
        )

        if result.returncode != 0:
            logger.error(f"Failed to cache {package_name}=={version}: {result.stderr}", exc_info=True)
            raise RuntimeError(f"uv run failed: {result.stderr}")

        logger.info(f"Successfully cached {package_name}=={version} and all dependencies")
    except Exception as e:
        logger.error(f"Failed to cache {package_name}=={version}: {e}", exc_info=True)
        raise


@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    plugin_loader.list_packages(refresh=True)

    evaluation: Evaluation = get_evaluation(evaluation_pid)

    # collect all unique packages
    unique_packages = {}
    for evaluation_plugin in evaluation.evaluation_plugins:
        pkg_key = f"{evaluation_plugin.package_name}=={evaluation_plugin.version}"
        if pkg_key not in unique_packages:
            unique_packages[pkg_key] = {
                "package_name": evaluation_plugin.package_name,
                "version": evaluation_plugin.version
            }

    logger.info(f"Found {len(unique_packages)} unique packages for {len(evaluation.evaluation_plugins)} plugins")

    # tasks: install packages
    install_tasks = []
    for pkg_info in unique_packages.values():
        install_task = install_package.si(pkg_info["package_name"], pkg_info["version"])
        install_tasks.append(install_task)

    # tasks: run plugins
    plugin_chains = []
    plugin_task_ids = []
    for evaluation_plugin in evaluation.evaluation_plugins:
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
        plugin_chain = chain(run_plugin_sig, post_measurements_sig)
        plugin_chains.append(plugin_chain)

    # chain: install all packages -> run all plugins -> finalize
    workflow = group(install_tasks) | group(plugin_chains) | finalize_evaluation.si(evaluation_pid)
    workflow.apply_async()

    return {"evaluation_pid": evaluation_pid, "plugin_task_ids": plugin_task_ids}


@celery_app.task(bind=True)
def run_plugin(self, package_name: str, plugin_name: str, version: str, plugin_config: dict,
               input_file_definitions: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID) -> list[dict]:

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

        # Create environment for subprocess
        env_vars = dict(os.environ)
        env_vars.pop("UV_NO_SYNC", None)
        env_vars.pop("VIRTUAL_ENV", None)

        if plugin_info["source"] == "local":
            install_target = str(plugin_info["pkg_root"].resolve())
        else:
            install_target = f"{package_name}=={version}"

        cmd = [
            "uv", "run", "-v",
            "--directory", str(workspace_path),
            "--extra-index-url", plugin_loader.devpi_client.simple_index_url,
            "--with", install_target
        ]

        if plugin_info["source"] != "local":
            cmd.extend(["--with", "vera-plugin-interface @ git+https://github.com/lux-ai-factory/vera-plugin-interface.git@v0.2.3"])

        cmd.append(str(runtime_script))

        logger.debug(f"Running uv command: {' '.join(cmd)}")
        start_time = time.perf_counter()
        mark_plugin_started(evaluation_pid, evaluation_plugin_pid)

        stdout_lines = []
        stderr_path = workspace_path / "stderr.tmp"

        with open(stderr_path, "w+") as stderr_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                bufsize=1,
                env=env_vars
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

        log_file = output_dir / "plugin_execution.log"
        log_content = "=== Plugin Execution Log ===\n\n"
        log_content += f"Execution Time: {duration:.2f} seconds\n\n"
        log_content += f"=== STDOUT ===\n{stdout_content}\n\n"
        log_content += f"=== STDERR ===\n{stderr_content}\n\n"
        log_content += f"=== Return Code ===\n{returncode}\n"
        log_file.write_text(log_content)

        if returncode != 0:
            mark_plugin_failed(evaluation_pid, evaluation_plugin_pid, stderr_content)
            raise RuntimeError(f"Plugin failed: {stderr_content}")

        measures_file = output_dir / "measures.json"
        measures = json.loads(measures_file.read_text()) if measures_file.exists() else []

        for file in output_dir.iterdir():
            if file.name != "measures.json" and file.name != "stderr.tmp":
                upload_artifact(evaluation_pid, evaluation_plugin_pid, file.name, file.read_bytes())

        mark_plugin_finished(evaluation_pid, evaluation_plugin_pid)
        return measures



@celery_app.task
def post_measurements(measurements_dict: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID):
    measurements = [Measure(**m) for m in measurements_dict]
    post_measures(evaluation_pid, evaluation_plugin_uuid, measurements)


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
    logger.debug(f"Finalizing evaluation {evaluation_id}")
    try:
        response = mark_completed(evaluation_id)
        logger.debug(
            f"Evaluation {evaluation_id} marked as completed, status: {response.status_code}"
        )
    except Exception as e:
        logger.error(f"Failed to mark evaluation {evaluation_id} as completed: {e}")
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
