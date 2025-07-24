import uuid

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.metric import Metric
from a4s_eval.evaluations.data_evaluation.registry import data_evaluator_registry
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_evaluation,
    post_metrics,
    mark_failed,
)
from a4s_eval.utils.dates import DateIterator
from a4s_eval.utils.env import API_URL_PREFIX


@celery_app.task
def dataset_evaluation_task(evaluation_pid: uuid.UUID):
    print(f"Starting evaluation task for {evaluation_pid}")

    # Debug: Check registry and API configuration
    print(f"API_URL_PREFIX: {API_URL_PREFIX}")

    # Check if any evaluators are registered
    evaluator_list = list(data_evaluator_registry)
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
        evaluation.model.dataset.data = get_dataset_data(evaluation.model.dataset.pid)
        print("Data loaded for both datasets")

        metrics: list[Metric] = []

        x_test = evaluation.dataset.data
        print("Starting time iteration for evaluation...")

        # Debug DateIterator parameters
        print("DateIterator parameters:")
        print(f"   - window_size: {evaluation.project.window_size}")
        print(f"   - frequency: {evaluation.project.frequency}")
        print(f"   - date_feature: {evaluation.model.dataset.shape.date.name}")
        print(f"   - data shape: {evaluation.dataset.data.shape}")
        print(
            f"   - date column sample: {evaluation.dataset.data[evaluation.model.dataset.shape.date.name].head()}"
        )

        iteration_count = 0
        evaluator_count = 0  # Initialize here to avoid UnboundLocalError

        try:
            date_iterator = DateIterator(
                date_round="1 D",
                window=evaluation.project.window_size,
                freq=evaluation.project.frequency,
                df=evaluation.dataset.data,
                date_feature=evaluation.model.dataset.shape.date.name,
            )
            print("DateIterator created successfully")

            for i, (date_val, x_curr) in enumerate(date_iterator):
                iteration_count += 1
                print(f"Iteration {i}, date: {date_val}, data shape: {x_curr.shape}")
                evaluation.dataset.data = x_curr

                evaluator_count = 0
                for name, evaluator in data_evaluator_registry:
                    evaluator_count += 1
                    print(f"Running evaluator: {name}")
                    new_metrics = evaluator(
                        evaluation.model.dataset, evaluation.dataset
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
        try:
            mark_failed(evaluation_pid)
        except Exception as mark_error:
            print(f"Error marking evaluation as failed: {mark_error}")
        raise
