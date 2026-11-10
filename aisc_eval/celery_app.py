from celery import Celery, Task

from aisc_eval.utils import env
from aisc_eval.utils.logging import get_logger

logger = get_logger()


class StopAwareTask(Task):
    """Convert billiard's SIGTERM SystemExit into a serializable failure.

    When Celery revokes a running task with ``terminate=True`` (SIGTERM), the
    worker process is interrupted with a ``SystemExit``. ``SystemExit`` is a
    ``BaseException``, which the JSON result backend cannot serialize, so it
    reaches the task pool as ``EncodeError`` and crashes the worker. Turning
    it into a regular ``RuntimeError`` keeps the backend store working.
    """

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except SystemExit:
            raise RuntimeError("Aborted: task revoked/terminated") from None


logger.debug(f"CELERY REDIS BACKEND URL: {env.REDIS_BACKEND_URL}")
logger.debug("=== CELERY APP INITIALIZATION ===")

celery_app: Celery = Celery(
    env.CELERY_APP_NAME, broker=env.CELERY_BROKER_URL, backend=env.REDIS_BACKEND_URL
)

logger.debug("=== CELERY APP CREATED ===")

# Configure Celery settings for 30-minute task execution with heartbeat
celery_config = {
    "task_acks_late": True,  # Acknowledge task only after completion
    "worker_prefetch_multiplier": 1,  # Process one task at a time
    "task_soft_time_limit": 1800,  # 30 minutes soft limit
    "task_time_limit": 2100,  # 35 minutes hard limit (buffer for cleanup)
    "broker_connection_max_retries": 10,
    "redis_retry_on_timeout": True,
    "redis_socket_keepalive": True,
    "redis_socket_keepalive_options": {
        "TCP_KEEPIDLE": 1,
        "TCP_KEEPINTVL": 3,
        "TCP_KEEPCNT": 5,
    },
    "worker_hijack_root_logger": False,
    "broker_use_ssl": env.MQ_USE_SSL,
    "results_extended": True,
    "task_track_started": True,
}

# Only add SSL configuration in production
if env.MQ_USE_SSL:
    celery_config["broker_transport_options"] = {
        "ssl": {
            "cert_reqs": 0,  # CERT_NONE as integer
            "ca_certs": None,
            "certfile": None,
            "keyfile": None,
        }
    }

celery_app.conf.update(celery_config)

logger.debug("=== CELERY CONFIGURATION COMPLETED ===")
