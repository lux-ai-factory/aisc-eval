import uuid
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
    get_model_file_content,
    upload_artifact,
)
from vera_eval.utils.logging import get_logger

from vera_plugin_manager.loader import Loader
from vera_plugin_interface import BaseEvaluationPlugin, TaskProgress
from vera_eval.utils import env

logger = get_logger()

plugin_loader: Loader = Loader(env.PLUGIN_PATH)


def artifact_callback(
    name: str, content: bytes, evaluation_pid: uuid.UUID, plugin_name: str
):
    upload_evaluation_artifact.delay(name, content, evaluation_pid, plugin_name)


def progress_callback(task_progress: TaskProgress, plugin_name: str, task: Task):
    meta = task_progress.model_dump()
    meta["plugin_name"] = plugin_name
    task.update_state(state="RUNNING", meta=meta)


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
                    "name": input_file.name,
                    "input_type": input_file.input_type,
                    "data": input_file.input_file.data,
                }
            )

        run_plugin_sig = run_plugin.s(
            evaluation_plugin.name, config, input_file_definitions, evaluation_pid
        )
        post_measurements_sig = post_measurements.s(
            evaluation_plugin.name, evaluation_pid
        )
        plugin_chain = chain(run_plugin_sig, post_measurements_sig)
        plugin_chains.append(plugin_chain)

    group_task = group(plugin_chains) | finalize_evaluation.si(evaluation_pid)
    group_task.apply_async()

    return {"evaluation_pid": evaluation_pid}


@celery_app.task(bind=True)
def run_plugin(
    self,
    plugin_name: str,
    plugin_config: dict,
    input_file_definitions: list[dict],
    evaluation_pid: uuid.UUID,
) -> list[dict]:
    plugin: BaseEvaluationPlugin = plugin_loader.load(plugin_name)

    plugin._set_progress_callback(
        partial(progress_callback, plugin_name=plugin_name, task=self)
    )

    plugin._set_artifact_callback(
        partial(
            artifact_callback, plugin_name=plugin_name, evaluation_pid=evaluation_pid
        )
    )

    for input_file_definition in input_file_definitions:
        if input_file_definition["input_type"] == "dataset":
            file_content = get_dataset_file_content(input_file_definition["data"])
        elif input_file_definition["input_type"] == "model":
            file_content = get_model_file_content(input_file_definition["data"])
        else:
            raise ValueError(
                f"Unsupported file type: {input_file_definition['input_type']}"
            )

        plugin.set_input_content(input_file_definition["name"], file_content)

    evaluation_output: Any = plugin.evaluate(plugin_config)

    measurements: list[Measure] = plugin.export_metrics(evaluation_output)
    measurements_dict = [m.model_dump() for m in measurements]
    return measurements_dict


@celery_app.task
def post_measurements(
    measurements_dict: list[dict], plugin_name: str, evaluation_pid: uuid.UUID
):
    measurements = [Measure(**m) for m in measurements_dict]
    post_measures(evaluation_pid, plugin_name, measurements)


@celery_app.task
def upload_evaluation_artifact(
    name: str, content: bytes, evaluation_pid: uuid.UUID, plugin_name: str
):
    upload_artifact(evaluation_pid, plugin_name, name, content)


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
