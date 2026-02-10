# ruff: noqa: F401
from a4s_eval.celery_app import celery_app
from a4s_eval.celery_tasks import run_evaluation
from a4s_eval.utils.logging import get_logger

get_logger().info("Starting worker...")
