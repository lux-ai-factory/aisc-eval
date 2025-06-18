


import uuid
from a4s_eval.service.api_client import get_dataset_data, get_evaluation
from celery import shared_task

from a4s_eval.evaluations.data_evaluation import data_evaluator_registry

@shared_task
def dataset_evaluation_task(evaluation_pid: uuid.UUID):

    evaluation = get_evaluation(evaluation_pid)

    test_data = get_dataset_data(evaluation.dataset.pid)
    train_data = get_dataset_data(evaluation.model.dataset.pid)

    for name, evaluator in data_evaluator_registry:
        evaluator(train_data, test_data)


