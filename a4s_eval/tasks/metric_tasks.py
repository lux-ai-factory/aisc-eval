import traceback
import uuid
from multiprocessing.util import get_logger

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric_registry
from a4s_eval.utils.dates import DateIterator


def master_metric(evaluation_pid: uuid.UUID) -> None:
    get_logger().info(f"Starting master task for {evaluation_pid}.")

    evaluation_request = get_evaluation_request(evaluation_pid)
    evaluation = get_evaluation_from_request(evaluation_request)

    for metric_category, metric_dict in evaluation.target_metrics.items():



    return group



