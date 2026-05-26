# ruff: noqa: F401
from aisc_eval.celery_app import celery_app
from aisc_eval.celery_tasks import run_evaluation
from aisc_eval.utils.logging import get_logger

get_logger().info("Starting worker...")
