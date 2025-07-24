from celery import Celery

from a4s_eval.utils import env

celery_app: Celery = Celery(
    __name__, broker=env.CELERY_BROKER_URL, backend=env.REDIS_BACKEND_URL
)

# Configure Celery settings for 30-minute task execution with heartbeat
celery_app.conf.update(
    broker_heartbeat=300,  # 5 minutes heartbeat interval
    broker_heartbeat_checkrate=2.0,  # Check heartbeat every 2 seconds
    task_acks_late=True,  # Acknowledge task only after completion
    worker_prefetch_multiplier=1,  # Process one task at a time
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2100,  # 35 minutes hard limit (buffer for cleanup)
)
