"""Dispatch an evaluation as a self-contained job (I1).

The dispatcher is the backend-facing coordination around a run: before the run
it resolves everything the run will need and hands it to the run path, and after
the run it reconciles the plugin status and posts the final result. The run path
itself (``aisc_eval.celery_tasks``) only consumes what it was dispatched with, so
a backend or platform outage mid-run cannot fail the run.
"""

import base64
import json
import uuid

from celery import chain, group

from aisc_eval.celery_app import celery_app
from aisc_eval.celery_tasks import (
    handle_error,
    install_package,
    plugin_loader,
    post_measurements,
    run_plugin,
)
from aisc_eval.service.api_client import (
    get_dataset_file_content,
    get_evaluation,
    get_evaluation_inputs,
    get_evaluation_plugins_status,
    get_model_file_content,
    get_project_settings_by_pid,
    mark_completed,
    mark_failed,
)
from aisc_eval.utils.logging import get_logger

logger = get_logger()


def _embed_input_files(input_components: list[dict]) -> None:
    """Fetch dataset/model bytes at dispatch so the run needs no mid-run call (I1)."""
    for component in input_components:
        component_type = component.get("component_type")
        if component_type == "dataset":
            content = get_dataset_file_content(component["data"])
        elif component_type == "model":
            content = get_model_file_content(component["data"])
        else:
            continue
        component["content_b64"] = base64.b64encode(content).decode("ascii")


def build_dispatch_payload(evaluation_pid: uuid.UUID) -> dict:
    """Resolve the whole job at dispatch time.

    Fetches the evaluation, each plugin's inputs (with the dataset/model bytes
    embedded as ``content_b64``) and its project settings, so the run path is
    handed everything it needs and performs no backend fetch of its own.
    """
    evaluation = get_evaluation(evaluation_pid)
    inputs_by_plugin: dict[str, list[dict]] = get_evaluation_inputs(evaluation_pid)

    settings_by_plugin: dict[str, list[dict]] = {}
    for evaluation_plugin in evaluation.evaluation_plugins:
        input_components = inputs_by_plugin.get(str(evaluation_plugin.pid), [])
        _embed_input_files(input_components)
        project_config_selections = (
            evaluation_plugin.plugin_config.project_config_selections
            if evaluation_plugin.plugin_config
            else []
        )
        settings_by_plugin[str(evaluation_plugin.pid)] = get_project_settings_by_pid(
            evaluation.project.pid, project_config_selections
        )

    return {
        "evaluation": evaluation,
        "inputs_by_plugin": inputs_by_plugin,
        "settings_by_plugin": settings_by_plugin,
    }


@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    plugin_loader.list_packages(refresh=True)

    payload = build_dispatch_payload(evaluation_pid)
    evaluation = payload["evaluation"]
    inputs_by_plugin = payload["inputs_by_plugin"]
    settings_by_plugin = payload["settings_by_plugin"]

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
    for pkg_key, pkg_info in plugins_by_pkg.items():
        install_sig = install_package.si(
            pkg_info["package_name"],
            pkg_info["version"],
            evaluation_pid,
            [ep.pid for ep in pkg_info["plugins"]],
        )
        # errback: chord body is skipped on task error, so this must mark the eval failed or it stays Pending forever.
        install_sig.link_error(handle_error.s(evaluation_pid))

        plugin_chain_list = []
        for evaluation_plugin in pkg_info["plugins"]:
            config = None
            if evaluation_plugin.plugin_config:
                config = evaluation_plugin.plugin_config.config

            input_components = inputs_by_plugin.get(str(evaluation_plugin.pid), [])

            run_plugin_sig = run_plugin.si(
                evaluation_plugin.package_name,
                evaluation_plugin.name,
                evaluation_plugin.version,
                config,
                input_components,
                settings_by_plugin[str(evaluation_plugin.pid)],
                evaluation_pid,
                evaluation_plugin.pid,
            )
            # errback: mark evaluation failed if run_plugin ever errors
            run_plugin_sig.link_error(handle_error.s(evaluation_pid))

            # freeze to get a stable task id before dispatching
            run_plugin_sig.freeze()
            plugin_task_ids.append(str(run_plugin_sig.id))

            post_measurements_sig = post_measurements.s(
                evaluation_pid, evaluation_plugin.pid
            )
            post_measurements_sig.link_error(handle_error.s(evaluation_pid))
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


@celery_app.task
def finalize_evaluation(evaluation_id: uuid.UUID) -> None:
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
