import uuid

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluations.data_evaluation.registry import data_evaluator_registry
from a4s_eval.service.api_client import get_dataset_data, get_evaluation, post_metrics
from a4s_eval.utils.dates import DateIterator


@celery_app.task
def dataset_evaluation_task(evaluation_pid: uuid.UUID):
    evaluation = get_evaluation(evaluation_pid)

    evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
    evaluation.model.dataset.data = get_dataset_data(evaluation.model.dataset.pid)

    metrics: list[Metric] = []

    x_test = evaluation.dataset.data
    for i, (_, x_curr) in enumerate(DateIterator(
        date_round="1 D",
        window=evaluation.project.window_size,
        freq=evaluation.project.frequency,
        df=evaluation.dataset.data,
        date_feature=evaluation.model.dataset.shape.date.name,
    )):
        print(f"Iteration {i}")
        evaluation.dataset.data = x_curr
        for name, evaluator in data_evaluator_registry:
            metrics.extend(evaluator(evaluation.dataset, evaluation.model.dataset))
    evaluation.dataset.data = x_test
    post_metrics(evaluation_pid, metrics)
