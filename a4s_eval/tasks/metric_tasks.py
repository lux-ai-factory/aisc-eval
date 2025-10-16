import uuid
from multiprocessing.util import get_logger

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.api_client import post_measures
from a4s_eval.metric_registries import registry_mapping


@celery_app.task
def metric_task(
    evaluation_pid: uuid.UUID, registry_name: str, metric_name_list: list[str]
) -> None:
    registry = registry_mapping.get(registry_name)
    inputs_iterator = registry.get_metric_inputs_dateiterator(evaluation_pid)
    measures: list[Measure] = []
    
    for inputs in inputs_iterator:
        for metric_name in metric_name_list:
            get_logger().info(f"Running evaluator: {metric_name}")
            evaluator = registry.get_functions().get(metric_name)
            new_measures = evaluator(*inputs)
            measures.extend(new_measures)

    response = post_measures(evaluation_pid, measures)
