# ruff: noqa: F401
from aisc_eval.celery_app import celery_app
# Importing the tasks module registers its tasks with the worker.
from aisc_eval.celery_tasks import run_evaluation
from aisc_eval.utils.logging import get_logger

get_logger().info("Starting worker...")
