import traceback
import uuid
from multiprocessing.util import get_logger

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric_registry
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_evaluation,
    get_project_datashape,
    post_measures,
)
from a4s_eval.utils.dates import DateIterator


@celery_app.task
def dataset_evaluation_task(evaluation_pid: uuid.UUID) -> None:
    get_logger().info(f"Starting evaluation task for {evaluation_pid}.")

    # Check if any evaluators are registered
    evaluator_list = list(data_metric_registry)
    get_logger().info(f"Registered evaluators ({len(evaluator_list)}):")
    for name, _ in evaluator_list:
        get_logger().info(f"  - {name}")

    try:
        inputs_iterator = data_metric_registry.get_metric_inputs_dateiterator(evaluation_pid)
        metrics: list[Measure] = []
        for name, evaluator in data_metric_registry:
            get_logger().info(f"Running evaluator: {name}")
            for inputs in inputs_iterator:
                new_metrics = evaluator(*inputs)
                metrics.extend(new_metrics)

        get_logger().info(f"Total metrics generated: {len(metrics)}")
        get_logger().debug(f"Posting {len(metrics)} metrics to API...")
        try:
            response = post_measures(evaluation_pid, metrics)
            get_logger().info(
                f"Metrics posted successfully, status: {response.status_code}."
            )
        except Exception as e:
            get_logger().error(f"Error posting metrics: {e}")
            raise

        get_logger().info(f"Tasked complete for {evaluation_pid}")

    except Exception as e:
        get_logger().error(f"Error in evaluation task: {e}")
        raise
