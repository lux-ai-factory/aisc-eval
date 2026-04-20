import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any
from functools import partial

from celery import group, chain, Task

from vera_eval.celery_app import celery_app
from vera_eval.data_model.evaluation import Evaluation
from vera_eval.data_model.measure import Measure
from vera_eval.service.api_client import (
    mark_completed,
    mark_failed,
    post_measures,
    get_evaluation,
    get_dataset_file_content,
    get_model_file_content, upload_artifact,
)
from vera_eval.utils.logging import get_logger

from vera_plugin_manager.loader import Loader
from vera_plugin_interface import BaseEvaluationPlugin, TaskProgress
from vera_eval.utils import env

logger = get_logger()

plugin_loader: Loader = Loader(env.PLUGIN_PATH, env.PACKAGE_REGISTRY_URL, env.PACKAGE_REGISTRY_INDEX, env.PACKAGE_REGISTRY_USER,
                               env.PACKAGE_REGISTRY_PASSWORD)

def artifact_callback(name: str, content: bytes, evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID):
    upload_evaluation_artifact.delay(name, content, evaluation_pid, evaluation_plugin_pid)

def progress_callback(task_progress: TaskProgress, plugin_name: str, task: Task):
    meta = task_progress.model_dump()
    meta["plugin_name"] = plugin_name
    task.update_state(state='RUNNING', meta=meta)


@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    evaluation: Evaluation = get_evaluation(evaluation_pid)

    plugin_chains = []
    for evaluation_plugin in evaluation.evaluation_plugins:
        config = None
        if evaluation_plugin.plugin_config:
            config = evaluation_plugin.plugin_config.config

        input_file_definitions = []
        for input_file in evaluation_plugin.input_files:
            input_file_definitions.append(
                {
                    'name': input_file.name,
                    'input_type': input_file.input_type,
                    'data': input_file.input_file.data,
                }
            )

        run_plugin_sig = run_plugin.s(
            evaluation_plugin.package_name,
            evaluation_plugin.name,
            evaluation_plugin.version,
            config,
            input_file_definitions,
            evaluation_pid,
            evaluation_plugin.pid
        )
        post_measurements_sig = post_measurements.s(evaluation_pid, evaluation_plugin.pid)
        plugin_chain = chain(run_plugin_sig, post_measurements_sig)
        plugin_chains.append(plugin_chain)

    group_task = group(plugin_chains) | finalize_evaluation.si(evaluation_pid)

    group_result = group_task.apply_async()

    return {'evaluation_pid': evaluation_pid}


@celery_app.task(bind=True)
def run_plugin(self, package_name: str, plugin_name: str, version: str, plugin_config: dict, input_file_definitions: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_pid: uuid.UUID) -> list[dict]:
    if not plugin_loader.discovered_packages:
        plugin_loader.list_packages()

    if package_name not in plugin_loader.discovered_packages:
        raise KeyError(f"Package '{package_name}' not found.")

    available_versions = plugin_loader.discovered_packages[package_name]
    if version not in available_versions:
        raise KeyError(f"Version '{version}' of package '{package_name}' not found.")

    plugin_info = available_versions[version]

    with tempfile.TemporaryDirectory(delete=False) as workspace_path:
        workspace = Path(workspace_path)

        input_dir = workspace / "input"
        output_dir = workspace / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_mapping = {}
        for input_file_definition in input_file_definitions:
            if input_file_definition['input_type'] == "dataset":
                file_content = get_dataset_file_content(input_file_definition['data'])
            elif input_file_definition['input_type'] == "model":
                file_content = get_model_file_content(input_file_definition['data'])
            else:
                raise ValueError(f"Unsupported file type: {input_file_definition['input_type']}")

            file_path = input_dir / input_file_definition['data']
            file_path.write_bytes(file_content)

            input_mapping[input_file_definition['name']] = f"{input_file_definition['data']}"

        config_data = {
            "plugin_source": f"{package_name}:{plugin_name}",
            "is_registry": plugin_info["source"] != "local",
            "input_mapping": input_mapping,
            "plugin_config": plugin_config,
        }

        config_path = workspace / "config.json"
        config_path.write_text(json.dumps(config_data, indent=2))

        runtime_script = Path(__file__).parent / "plugin_runtime.py"

        # Create environment for subprocess that allows UV to sync dependencies
        # Remove environment variables that interfere with UV's isolated execution
        env_vars = dict(os.environ)
        env_vars.pop("UV_NO_SYNC", None)  # Remove UV_NO_SYNC to allow dependency installation
        env_vars.pop("VIRTUAL_ENV", None)  # Remove VIRTUAL_ENV to prevent path mismatch warnings

        # Handle local vs registry plugins
        if plugin_info["source"] == "local":
            # For local plugins, copy plugin source to workspace and create isolated venv
            plugin_dir = workspace / "plugin"

            # Copy plugin files to workspace (excluding .venv and other cache dirs)
            logger.debug(f"Copying plugin from {plugin_info['pkg_root']} to {plugin_dir}")
            shutil.copytree(
                plugin_info["pkg_root"],
                plugin_dir,
                ignore=shutil.ignore_patterns('.venv', '__pycache__', '*.pyc', '.pytest_cache', '.git')
            )

            # Sync dependencies AND install the plugin package itself
            # By default, uv sync installs both dependencies and the project itself
            sync_cmd = ["uv", "sync", "-v", "--directory", str(plugin_dir)]
            logger.debug(f"Syncing plugin and dependencies: {' '.join(sync_cmd)}")
            sync_result = subprocess.run(sync_cmd, capture_output=True, text=True, env=env_vars)

            if sync_result.returncode != 0:
                logger.error(f"Failed to sync plugin: {sync_result.stderr}")
                raise RuntimeError(f"Failed to sync plugin: {sync_result.stderr}")

            # Run from the workspace plugin directory
            cmd = ["uv", "run", "-v", "--directory", str(plugin_dir), str(runtime_script), "--working-dir", str(workspace)]
        else:
            # For registry plugins, set working directory and install package
            cmd = [
                "uv", "run", "-v",
                "--directory", str(workspace),
                "--extra-index-url", plugin_loader.client.full_index_url,

                "--with", f"{package_name}=={version}",

                # We have to forcefully inject the interface here, once its publicly published somewhere we can remove this
                "--with", "vera-plugin-interface @ git+https://github.com/lux-ai-factory/vera-plugin-interface.git@v0.2.2",

                str(runtime_script),
                "--working-dir", str(workspace)
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env_vars)

        log_file = output_dir / "plugin_execution.log"
        log_content = f"=== Plugin Execution Log ===\n\n"
        log_content += f"=== STDOUT ===\n{result.stdout}\n\n"
        log_content += f"=== STDERR ===\n{result.stderr}\n\n"
        log_content += f"=== Return Code ===\n{result.returncode}\n"
        log_file.write_text(log_content)

        if result.returncode != 0:
            raise RuntimeError(f"Plugin failed: {result.stderr}")

        measures_file = output_dir / "measures.json"
        measures = json.loads(measures_file.read_text()) if measures_file.exists() else []

        for file in output_dir.iterdir():
            if file.name != "measures.json":
                upload_artifact(evaluation_pid, evaluation_plugin_pid, file.name, file.read_bytes())

        return measures


@celery_app.task
def post_measurements(measurements_dict: list[dict], evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID):
    measurements = [Measure(**m) for m in measurements_dict]
    response = post_measures(evaluation_pid, evaluation_plugin_uuid, measurements)

@celery_app.task
def upload_evaluation_artifact(name: str, content: bytes, evaluation_pid: uuid.UUID, evaluation_plugin_uuid: uuid.UUID):
    upload_artifact(evaluation_pid, evaluation_plugin_uuid, name, content)


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
