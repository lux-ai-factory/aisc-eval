# ruff: noqa: F401
from aisc_eval.celery_app import celery_app

# Importing the tasks module registers its tasks with the worker.
from aisc_eval.celery_tasks import run_evaluation

# The sweep: its task, its beat schedule and the sweep once after the worker starts.
from aisc_eval import run_sweep
from aisc_eval.utils.logging import get_logger

get_logger().info("Starting worker...")
