# ruff: noqa: F401
from vera_eval.celery_app import celery_app
from vera_eval.celery_tasks import run_evaluation
from vera_eval.utils.logging import get_logger

get_logger().info("Starting worker...")
