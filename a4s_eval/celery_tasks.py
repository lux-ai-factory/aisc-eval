import uuid
from typing import Any

from celery import group, chain

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.evaluation import Evaluation
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.project import InputFile
from a4s_eval.service.api_client import (
    mark_completed,
    mark_failed,
    post_measures,
    get_evaluation,
    get_dataset_file_content,
    get_model_file_content,
)
from a4s_eval.utils.logging import get_logger

from a4s_plugin_manager.loader import Loader
from a4s_plugin_interface import BaseEvaluationPlugin, TaskProgress
from a4s_eval.utils import env

logger = get_logger()

plugin_loader: Loader = Loader(env.PLUGIN_PATH)

@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    evaluation: Evaluation = get_evaluation(evaluation_pid)

    plugin_chains = []
    for evaluation_plugin in evaluation.evaluation_plugins:
        config = None
        if evaluation_plugin.plugin_config:
            config = evaluation_plugin.plugin_config.config

        run_plugin_sig = run_plugin.s(evaluation_plugin.name, config, evaluation_plugin.dataset_filename, evaluation_plugin.model_filename)
        post_measurements_sig = post_measurements.s(evaluation_pid)
        plugin_chain = chain(run_plugin_sig, post_measurements_sig)
        plugin_chains.append(plugin_chain)

    group_task = group(plugin_chains) | finalize_evaluation.si(evaluation_pid)

    group_result = group_task.apply_async()

    return {'evaluation_pid': evaluation_pid}

@celery_app.task(bind=True)
def run_plugin(self, plugin_name: str, plugin_config: dict, input_files: list[InputFile]) -> list[dict]:
    plugin: BaseEvaluationPlugin = plugin_loader.load(plugin_name)

    def progress_callback(task_progress: TaskProgress):
        meta = task_progress.model_dump()
        meta["plugin_name"] = plugin_name
        self.update_state(state='RUNNING', meta=meta)

    plugin._set_progress_callback(progress_callback)

    if dataset_file:
        dataset_file_contents = get_dataset_file_content(dataset_file)
        plugin.set_dataset_input_provider(dataset_file_contents)

    if model_filename:
        model_file_contents = get_model_file_content(model_filename)
        plugin.set_model_input_provider(model_file_contents)

    evaluation_output: Any = plugin.evaluate(plugin_config)

    measurements: list[Measure] = plugin.export_metrics(evaluation_output)
    measurements_dict = [m.model_dump() for m in measurements]
    return measurements_dict

@celery_app.task
def post_measurements(measurements_dict: list[dict], evaluation_pid: uuid.UUID):
    measurements = [Measure(**m) for m in measurements_dict]
    response = post_measures(evaluation_pid, measurements)


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
