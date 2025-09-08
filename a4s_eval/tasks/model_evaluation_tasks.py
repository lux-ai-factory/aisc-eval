import uuid

import numpy as np

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluators.model_evaluator import (
    model_pred_proba_evaluator_registry,
)
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_evaluation,
    get_onnx_model,
    get_project_datashape,
    post_metrics,
)
from a4s_eval.utils.dates import DateIterator
from a4s_eval.utils.env import API_URL_PREFIX
from a4s_eval.utils.logging import get_logger

logger = get_logger()


@celery_app.task
def model_evaluation_task(evaluation_pid: uuid.UUID) -> None:
    get_logger().info(f"Starting evaluation task for {evaluation_pid}.")

    # Debug: Check registry and API configuration
    print(f"API_URL_PREFIX: {API_URL_PREFIX}")

    # Check if any evaluators are registered
    evaluator_list = list(model_pred_proba_evaluator_registry)
    get_logger().info(f"Registered evaluators ({len(evaluator_list)}):")
    for name, _ in evaluator_list:
        get_logger().info(f"  - {name}")

    try:
        evaluation = get_evaluation(evaluation_pid)
        evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
        session = get_onnx_model(evaluation.model.pid)

        metrics: list[Metric] = []

        datashape = get_project_datashape(evaluation.project.pid)

        x_test = evaluation.dataset.data
        x_test = x_test[[f.name for f in datashape.features]].to_numpy()

        iteration_count = 0

        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[1].name
        pred_onx = session.run([label_name], {input_name: x_test})[0]
        y_pred_proba = np.array([list(d.values()) for d in pred_onx])
        get_logger().info("Computation finished for Y prediction probability.")

        try:
            if not datashape.date:
                raise ValueError(
                    "Datashape is missing a date feature, which is required for time-based evaluation."
                )
            date_iterator = DateIterator(
                date_round="1 D",
                window=evaluation.project.window_size,
                freq=evaluation.project.frequency,
                df=evaluation.dataset.data,
                date_feature=datashape.date.name,
            )

            for i, (date_val, x_curr) in enumerate(date_iterator):
                iteration_count += 1
                print(f"Iteration {i}, date: {date_val}, data shape: {x_curr.shape}")
                evaluation.dataset.data = x_curr

                ## Get the current y_pred_proba for current date batch
                ## ATTENTION: This assumes that the index of x_test is not predifined
                y_curr_pred_proba = y_pred_proba[list(x_curr.index)]

                evaluator_count = 0
                for name, evaluator in model_pred_proba_evaluator_registry:
                    evaluator_count += 1
                    get_logger().info(f"Running evaluator: {name}")
                    new_metrics = evaluator(
                        datashape,
                        evaluation.model,
                        evaluation.dataset,
                        y_curr_pred_proba,
                    )
                    metrics.extend(new_metrics)

        except Exception as e:
            print(f"Error in DateIterator: {e}")
            import traceback

            traceback.print_exc()

        get_logger().info(f"Total iterations: {iteration_count}")
        get_logger().info(f"Total metrics generated: {len(metrics)}")

        evaluation.dataset.data = x_test

        get_logger().debug(f"Posting {len(metrics)} metrics to API...")
        try:
            response = post_metrics(evaluation_pid, metrics)
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
