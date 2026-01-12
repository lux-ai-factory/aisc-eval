import pathlib
import uuid
from typing import Any

from celery import group, Signature, chain
import yaml

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.api_client import (
    fetch_pending_evaluations,
    mark_completed,
    mark_failed, post_measures, get_evaluation, get_dataset_file_content, get_model_file_content,
)
from a4s_eval.tasks.metric_tasks import metric_task
from a4s_eval.metric_registries import map_registries_to_supported_metrics
from a4s_eval.utils.logging import get_logger

from a4s_plugin_manager.loader import Loader
from a4s_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin
from a4s_eval.utils import env

logger = get_logger()

plugin_loader: Loader = Loader(env.DEV_PLUGIN_PATH)

@celery_app.task(bind=True)
def run_evaluation(self, evaluation_pid: uuid.UUID) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    evaluation = get_evaluation(evaluation_pid)

    plugin_chains = []
    for plugin in evaluation.evaluation_plugins:
        run_plugin_sig = run_plugin.s(plugin.name, plugin.config, plugin.dataset_filename, plugin.model_filename)
        post_measurements_sig = post_measurements.s(evaluation_pid)
        plugin_chain = chain(run_plugin_sig, post_measurements_sig)
        plugin_chains.append(plugin_chain)

    group_task = group(plugin_chains) | finalize_evaluation.si(evaluation_pid)

    group_result = group_task.apply_async()

    return {'group_id': group_result.id}

@celery_app.task
def run_plugin(plugin_name: str, plugin_config: dict, dataset_file: str | None, model_filename: str | None) -> list[dict]:
    plugin: BaseEvaluationPlugin = plugin_loader.load(plugin_name)

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
def poll_and_run_evaluation() -> None:
    try:
        logger.debug("=== POLL_AND_RUN_EVALUATION START ===")
        logger.debug("1. Starting poll_and_run_evaluation task")

        logger.debug("2. About to call fetch_pending_evaluations()")
        eval_ids = fetch_pending_evaluations()
        logger.debug(
            f"3. fetch_pending_evaluations() completed. Found {len(eval_ids)} evaluations: {eval_ids}"
        )

        if not eval_ids:
            logger.debug("4. No pending evaluations found, returning")
            return

        logger.debug(f"5. Creating signatures for {len(eval_ids)} evaluations...")
        signatures = [generate_evaluation_signature(eval_id) for eval_id in eval_ids]
        logger.debug(f"6. Signatures created: {len(signatures)}")

        logger.debug("7. Starting to apply signatures...")
        # Apply each signature in parallel
        for i, (eval_id, sig) in enumerate(zip(eval_ids, signatures)):
            logger.debug(f"8.{i + 1} About to launch evaluation task for {eval_id}")
            try:
                sig.apply_async()
                logger.debug(f"9.{i + 1} Task launched successfully for {eval_id}")
            except Exception as e:
                logger.error(f"ERROR launching task for {eval_id}: {str(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

        logger.debug("10. All tasks processed")

    except Exception as e:
        logger.error(f"ERROR in poll_and_run_evaluation: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        logger.debug("=== POLL_AND_RUN_EVALUATION END ===")


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


def generate_evaluation_signature(evaluation_pid: uuid.UUID) -> Signature:
    # --- Load local config file. TODO: using evaluation config in api ---
    config_file = pathlib.Path("config/eval_config.yaml")
    with open(config_file) as f_in:
        eval_config = yaml.safe_load(f_in)

    registry_metric_pairs = map_registries_to_supported_metrics(eval_config)

    # --- Create metric tasks ---
    task_signatures = [
        metric_task.s(evaluation_pid, name, metrics).on_error(
            handle_error.s(evaluation_pid)
        )
        for name, metrics in registry_metric_pairs
    ]

    # --- Combine tasks into a group with finalization ---
    workflow = group(task_signatures) | finalize_evaluation.si(evaluation_pid)
    return workflow
