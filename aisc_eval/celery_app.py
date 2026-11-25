import inspect
import os

from aisc_identity import logs
from celery import Celery
from celery.signals import task_postrun, task_prerun

from aisc_eval.utils import env
from aisc_eval.utils.logging import get_logger

logger = get_logger()


# the host only: the URL carries the Redis password
logger.debug(f"CELERY REDIS BACKEND HOST: {os.getenv('REDIS_HOST', 'redis')}")
logger.debug("=== CELERY APP INITIALIZATION ===")

celery_app: Celery = Celery(
    env.CELERY_APP_NAME, broker=env.CELERY_BROKER_URL, backend=env.REDIS_BACKEND_URL
)

logger.debug("=== CELERY APP CREATED ===")

#: the queue a plugin's manifest read and its form hooks go to: a worker process of its own serves it, one task at a
#: time, so a configure page never waits behind the runs; the run worker reads only the default queue
PLUGIN_DECLARATIONS_QUEUE = "plugin-declarations"
PLUGIN_DECLARATION_TASKS = (
    "aisc_eval.celery_tasks.read_manifest",
    "aisc_eval.celery_tasks.plugin_hook",
)

# Configure Celery settings for 30-minute task execution with heartbeat
celery_config = {
    "task_acks_late": True,  # Acknowledge task only after completion
    "worker_prefetch_multiplier": 1,  # Process one task at a time
    "task_soft_time_limit": 24000,  # 400 min soft limit (first GPT4ALL run downloads a ~2GB model + CPU inference)
    "task_time_limit": 24300,  # 405 min hard limit (buffer for cleanup)
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
    "task_routes": {
        name: {"queue": PLUGIN_DECLARATIONS_QUEUE} for name in PLUGIN_DECLARATION_TASKS
    },
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


@task_prerun.connect
def _name_the_run_in_log_lines(task=None, args=(), kwargs=None, **_) -> None:
    """Every line a task logs names the run (evaluation pid) it serves, when its arguments name one."""
    try:
        arguments = (
            inspect.signature(task.run)
            .bind_partial(*(args or ()), **(kwargs or {}))
            .arguments
        )
    except (TypeError, ValueError, AttributeError):
        arguments = {}
    pid = arguments.get("evaluation_pid")
    logs.evaluation_pid.set(str(pid) if pid is not None else None)


@task_postrun.connect
def _clear_the_run_in_log_lines(**_) -> None:
    logs.evaluation_pid.set(None)
