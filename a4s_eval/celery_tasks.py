import uuid
from typing import Any

from celery import group, chain

from datetime import datetime, timezone

from a4s_eval.audit.events import (
    EVALUATION_COMPLETED,
    EVALUATION_FAILED,
    EVALUATION_STARTED,
    MEASUREMENTS_POSTED,
    PLUGIN_COMPLETED,
    PLUGIN_FAILED,
    PLUGIN_STARTED,
    AuditTimer,
    log_audit_event,
)
from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
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
def run_evaluation(self, evaluation_pid: uuid.UUID, user_id: int | None = None) -> dict:
    logger.info(f"Running evaluation {evaluation_pid}")

    log_audit_event(
        EVALUATION_STARTED,
        evaluation_id=str(evaluation_pid),
        task_id=self.request.id,
        user_id=user_id,
    )

    evaluation = get_evaluation(evaluation_pid)

    plugin_chains = []
    for plugin in evaluation.evaluation_plugins:
        run_plugin_sig = run_plugin.s(str(evaluation_pid), plugin.name, plugin.config, plugin.dataset_filename, plugin.model_filename)
        post_measurements_sig = post_measurements.s(evaluation_pid)
        plugin_chain = chain(run_plugin_sig, post_measurements_sig)
        plugin_chains.append(plugin_chain)

    group_task = group(plugin_chains) | finalize_evaluation.si(evaluation_pid)

    group_task.apply_async()

    return {'evaluation_pid': evaluation_pid}

@celery_app.task(bind=True)
def run_plugin(self, evaluation_id: str, plugin_name: str, plugin_config: dict, dataset_file: str | None, model_filename: str | None) -> list[dict]:
    import json as _json
    config_str = _json.dumps(plugin_config, default=str) if plugin_config else ""
    execution_start = datetime.now(timezone.utc)

    log_audit_event(
        PLUGIN_STARTED,
        evaluation_id=evaluation_id,
        plugin_name=plugin_name,
        task_id=self.request.id,
        test_set=dataset_file or "",
        configuration=config_str,
        target_system=plugin_name,
        execution_start=execution_start,
    )

    try:
        with AuditTimer() as timer:
            plugin: BaseEvaluationPlugin = plugin_loader.load(plugin_name)

            def progress_callback(task_progress: TaskProgress):
                meta = task_progress.model_dump()
                meta["plugin_name"] = plugin_name
                self.update_state(state='RUNNING', meta=meta)

            plugin._set_progress_callback(progress_callback)

            if dataset_file:
                dataset_file_contents = get_dataset_file_content(dataset_file, evaluation_id=evaluation_id)
                plugin.set_dataset_input_provider(dataset_file_contents)

            if model_filename:
                model_file_contents = get_model_file_content(model_filename, evaluation_id=evaluation_id)
                plugin.set_model_input_provider(model_file_contents)

            evaluation_output: Any = plugin.evaluate(plugin_config)

            measurements: list[Measure] = plugin.export_metrics(evaluation_output)
            measurements_dict = [m.model_dump() for m in measurements]

        execution_end = datetime.now(timezone.utc)
        log_audit_event(
            PLUGIN_COMPLETED,
            evaluation_id=evaluation_id,
            plugin_name=plugin_name,
            task_id=self.request.id,
            duration_ms=timer.duration_ms,
            details={"measurement_count": len(measurements_dict)},
            test_set=dataset_file or "",
            configuration=config_str,
            target_system=plugin_name,
            execution_start=execution_start,
            execution_end=execution_end,
        )

        return measurements_dict

    except Exception as e:
        execution_end = datetime.now(timezone.utc)
        log_audit_event(
            PLUGIN_FAILED,
            evaluation_id=evaluation_id,
            plugin_name=plugin_name,
            task_id=self.request.id,
            status="failure",
            error_message=str(e),
            test_set=dataset_file or "",
            configuration=config_str,
            target_system=plugin_name,
            execution_start=execution_start,
            execution_end=execution_end,
        )
        raise

@celery_app.task(bind=True)
def post_measurements(self, measurements_dict: list[dict], evaluation_pid: uuid.UUID):
    measurements = [Measure(**m) for m in measurements_dict]
    post_measures(evaluation_pid, measurements)

    log_audit_event(
        MEASUREMENTS_POSTED,
        evaluation_id=str(evaluation_pid),
        task_id=self.request.id,
        details={"measurement_count": len(measurements)},
    )


@celery_app.task(bind=True)
def finalize_evaluation(self, evaluation_id: uuid.UUID) -> None:
    logger.debug(f"Finalizing evaluation {evaluation_id}")
    try:
        response = mark_completed(evaluation_id)
        logger.debug(
            f"Evaluation {evaluation_id} marked as completed, status: {response.status_code}"
        )
        log_audit_event(
            EVALUATION_COMPLETED,
            evaluation_id=str(evaluation_id),
            task_id=self.request.id,
        )
    except Exception as e:
        logger.error(f"Failed to mark evaluation {evaluation_id} as completed: {e}")
        mark_failed(evaluation_id)


@celery_app.task(bind=True)
def handle_error(
    self,
    evaluation_id: uuid.UUID,
    request: object,
    exc: BaseException,
    traceback: object,
) -> None:
    logger.error(f"Error in evaluation {evaluation_id}:")
    logger.error(f"--\n\n{request} {exc} {traceback}")

    log_audit_event(
        EVALUATION_FAILED,
        evaluation_id=str(evaluation_id),
        task_id=self.request.id,
        status="failure",
        error_message=str(exc),
        details={"traceback": str(traceback)},
    )

    mark_failed(evaluation_id)
    logger.error(f"Evaluation {evaluation_id} marked as failed due to error.")
