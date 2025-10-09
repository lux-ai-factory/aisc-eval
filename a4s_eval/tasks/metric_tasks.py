import traceback
import uuid
from multiprocessing.util import get_logger
from celery import group

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric_registry
from a4s_eval.utils.dates import DateIterator
from a4s_eval.metric_registries import registries_dict


def master_metric(evaluation_pid: uuid.UUID) -> None:
    get_logger().info(f"Starting master task for {evaluation_pid}.")


    groups = []
    for metric_category, metric_dict in evaluation.target_metrics.items():
        group(
            [
                metric_dict(evaluation)
            ]
        )
        groups.append(group)

    return groups




for registry_name, category_registry in registries_dict.items():
    print(f"Running category: {registry_name}")
    # inputs = category_registry.get_inputs(eval_pid)
    for name, evaluator in category_registry:
        print(f"Running evaluator: {name}")
        # inputs = evaluator.get_inputs(eval_pid)
        # new_metrics = evaluator(
        #     *inputs
        # )


for name, evaluator in data_metric_registry:
    print(f"Running evaluator: {name}")