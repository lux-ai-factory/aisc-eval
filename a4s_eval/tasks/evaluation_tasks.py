import uuid

import numpy as np

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluations.data_evaluation.registry import data_evaluator_registry
from a4s_eval.evaluations.model_evaluation.registry import (
    model_pred_proba_evaluator_registry,
)
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_evaluation,
    get_onnx_model,
    get_project_datashape,
    post_metrics,
    mark_failed,
    mark_completed,
)
from a4s_eval.utils.dates import DateIterator
from a4s_eval.utils.env import API_URL_PREFIX
from a4s_eval.utils.logging import get_logger

logger = get_logger()


@celery_app.task
def dataset_evaluation_task(evaluation_pid: uuid.UUID) -> None:
    logger.debug(f"Starting evaluation task for {evaluation_pid}")

    # Debug: Check registry and API configuration
    logger.debug(f"API_URL_PREFIX: {API_URL_PREFIX}")

    # Check if any evaluators are registered
    evaluator_list = list(data_evaluator_registry)
    logger.debug(f"Registered evaluators: {len(evaluator_list)}")
    for name, _ in evaluator_list:
        logger.debug(f"  - {name}")

    if len(evaluator_list) == 0:
        logger.warning("No evaluators registered!")
        return

    try:
        evaluation = get_evaluation(evaluation_pid)
        logger.debug(f"Evaluation loaded: {evaluation.pid}")

        evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
        evaluation.model.dataset.data = get_dataset_data(evaluation.model.dataset.pid)
        logger.debug("Data loaded for both datasets")

        metrics: list[Metric] = []

        x_test = evaluation.dataset.data
        logger.debug("Starting time iteration for evaluation...")

        datashape = get_project_datashape(evaluation.project.pid)

        # Debug DateIterator parameters
        logger.debug("DateIterator parameters:")
        logger.debug(f"   - window_size: {evaluation.project.window_size}")
        logger.debug(f"   - frequency: {evaluation.project.frequency}")
        logger.debug(f"   - date_feature: {datashape.date.name}")
        logger.debug(f"   - data shape: {evaluation.dataset.data.shape}")
        logger.debug(
            f"   - date column sample: {evaluation.dataset.data[datashape.date.name].head()}"
        )

        iteration_count = 0
        evaluator_count = 0  # Initialize here to avoid UnboundLocalError

        try:
            date_iterator = DateIterator(
                date_round="1 D",
                window=evaluation.project.window_size,
                freq=evaluation.project.frequency,
                df=evaluation.dataset.data,
                date_feature=datashape.date.name,
            )
            logger.debug("DateIterator created successfully")

            for i, (date_val, x_curr) in enumerate(date_iterator):
                iteration_count += 1
                logger.debug(
                    f"Iteration {i}, date: {date_val}, data shape: {x_curr.shape}"
                )
                evaluation.dataset.data = x_curr

                evaluator_count = 0
                for name, evaluator in data_evaluator_registry:
                    evaluator_count += 1
                    logger.debug(f"Running evaluator: {name}")
                    new_metrics = evaluator(
                        datashape, evaluation.model.dataset, evaluation.dataset
                    )
                    logger.debug(f"Generated {len(new_metrics)} metrics")
                    metrics.extend(new_metrics)

        except Exception as e:
            logger.error(f"Error in DateIterator: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

        logger.debug(f"Total iterations: {iteration_count}")
        logger.debug(f"Total evaluators per iteration: {evaluator_count}")
        logger.debug(f"Total metrics generated: {len(metrics)}")

        if len(metrics) > 0:
            logger.debug(f"Sample metric: {metrics[0].model_dump()}")
        else:
            logger.warning("No metrics generated!")

        evaluation.dataset.data = x_test

        logger.debug(f"Posting {len(metrics)} metrics to API...")
        try:
            response = post_metrics(evaluation_pid, metrics)
            logger.debug(f"Metrics posted successfully, status: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
        except Exception as e:
            logger.error(f"Error posting metrics: {e}")
            raise

        try:
            mark_completed(evaluation_pid)
        except Exception as mark_error:
            logger.error(f"Error marking evaluation as completed: {mark_error}")

        logger.debug("Evaluation task completed successfully")

    except Exception as e:
        logger.error(f"Error in evaluation task: {e}")
        logger.debug("Marking evaluation as failed...")
        try:
            mark_failed(evaluation_pid)
        except Exception as mark_error:
            logger.error(f"Error marking evaluation as failed: {mark_error}")
        raise


@celery_app.task
def model_evaluation_task(evaluation_pid: uuid.UUID) -> None:
    print(f"Starting evaluation task for {evaluation_pid}")

    # Debug: Check registry and API configuration
    print(f"API_URL_PREFIX: {API_URL_PREFIX}")

    # Check if any evaluators are registered
    evaluator_list = list(model_pred_proba_evaluator_registry)
    print(f"Registered evaluators: {len(evaluator_list)}")
    for name, _ in evaluator_list:
        print(f"  - {name}")

    if len(evaluator_list) == 0:
        print("WARNING: No evaluators registered!")
        return

    try:
        evaluation = get_evaluation(evaluation_pid)
        print(f"Evaluation loaded: {evaluation.pid}")

        evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
        session = get_onnx_model(evaluation.model.pid)
        print("Data loaded for both datasets")

        metrics: list[Metric] = []

        datashape = get_project_datashape(evaluation.project.pid)

        x_test = evaluation.dataset.data
        x_test = x_test[[f.name for f in datashape.features]].to_numpy()
        print("Starting time iteration for evaluation...")

        # Debug DateIterator parameters
        print("DateIterator parameters:")
        print(f"   - window_size: {evaluation.project.window_size}")
        print(f"   - frequency: {evaluation.project.frequency}")
        # print(f"   - date_feature: {evaluation.model.dataset.shape.date.name}")
        print(f"   - data shape: {evaluation.dataset.data.shape}")
        print(
            f"   - date column sample: {evaluation.dataset.data[datashape.date.name].head()}"
        )

        iteration_count = 0
        evaluator_count = 0  # Initialize here to avoid UnboundLocalError

        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[1].name
        pred_onx = session.run([label_name], {input_name: x_test})[0]
        y_pred_proba = np.array([list(d.values()) for d in pred_onx])
        print("Computation finished for Y prediction probability")

        try:
            date_iterator = DateIterator(
                date_round="1 D",
                window=evaluation.project.window_size,
                freq=evaluation.project.frequency,
                df=evaluation.dataset.data,
                date_feature=datashape.date.name,
            )
            print("DateIterator created successfully")

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
                    print(f"Running evaluator: {name}")
                    new_metrics = evaluator(
                        datashape,
                        evaluation.model,
                        evaluation.dataset,
                        y_curr_pred_proba,
                    )
                    print(f"Generated {len(new_metrics)} metrics")
                    metrics.extend(new_metrics)

        except Exception as e:
            print(f"Error in DateIterator: {e}")
            import traceback

            traceback.print_exc()

        print(f"Total iterations: {iteration_count}")
        print(f"Total evaluators per iteration: {evaluator_count}")
        print(f"Total metrics generated: {len(metrics)}")

        if len(metrics) > 0:
            print(f"Sample metric: {metrics[0].model_dump()}")
        else:
            print("WARNING: No metrics generated!")

        evaluation.dataset.data = x_test

        print(f"Posting {len(metrics)} metrics to API...")
        try:
            response = post_metrics(evaluation_pid, metrics)
            print(f"Metrics posted successfully, status: {response.status_code}")
            print(f"Response content: {response.text}")
        except Exception as e:
            print(f"Error posting metrics: {e}")
            raise

        print("Evaluation task completed successfully")

    except Exception as e:
        print(f"Error in evaluation task: {e}")
        print("Marking evaluation as failed...")
        raise
