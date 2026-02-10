import uuid
from multiprocessing.util import get_logger

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.api_client import post_measures
from a4s_eval.metric_registries import registry_mapping, input_generator_cls_mapping

logger = get_logger()


@celery_app.task
def metric_task(
    evaluation_pid: uuid.UUID, registry_name: str, metric_name_list: list[str]
) -> None:
    # retrieve registry and input generator class by registry_name
    registry = registry_mapping.get(registry_name)
    InputGenerator = input_generator_cls_mapping.get(registry_name)
    # Intantiate input generator with evaluation_pid and getting inputs iterator
    if registry is None or InputGenerator is None:
        raise ValueError(f"Unknown registry name: {registry_name}")
    input_generator = InputGenerator(evaluation_pid)
    inputs_iterator = input_generator.get_inputs_dateiterator()

    metrics_mapping = {
        metric_name: fn
        for metric_name in metric_name_list
        if (fn := registry.get_functions().get(metric_name)) is not None
    }

    measures: list[Measure] = []

    for inputs in inputs_iterator:
        for metric_name, metric_fn in metrics_mapping.items():
            logger.info(f"Running metric function: {metric_name}")
            measures.extend(metric_fn(*inputs))

    response = post_measures(evaluation_pid, measures)
    logger.info(f"Metrics posted successfully, status: {response.status_code}.")


@celery_app.task
def metric_one_shot_task(
    evaluation_pid: uuid.UUID, registry_name: str, metric_name_list: list[str]
) -> None:
    # retrieve registry and input generator class by registry_name
    registry = registry_mapping.get(registry_name)
    InputGenerator = input_generator_cls_mapping.get(registry_name)
    # Intantiate input generator with evaluation_pid and getting inputs iterator
    if registry is None or InputGenerator is None:
        raise ValueError(f"Unknown registry name: {registry_name}")
    input_generator = InputGenerator(evaluation_pid)
    inputs = input_generator.get_inputs()

    metrics_mapping = {
        metric_name: fn
        for metric_name in metric_name_list
        if (fn := registry.get_functions().get(metric_name)) is not None
    }

    measures: list[Measure] = []

    for metric_name, metric_fn in metrics_mapping.items():
        logger.info(f"Running metric function: {metric_name}")
        measures.extend(metric_fn(*inputs))

    response = post_measures(evaluation_pid, measures)
    logger.info(f"Metrics posted successfully, status: {response.status_code}.")
