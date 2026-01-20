"""Environment configuration module for A4S Evaluation.

This module defines environment variables and their default values used throughout
the A4S evaluation system. These can be overridden by setting actual environment
variables.
"""

import os
from urllib.parse import quote

from a4s_eval.utils.logging import get_logger

logger = get_logger()


def handle_bool_var(envvar: str) -> bool:
    return str(envvar).lower() == "true"


API_URL = os.getenv("API_URL", "http://backend:8000")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
API_URL_PREFIX = f"{API_URL}{API_PREFIX}"
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/cache")

MQ_USE_SSL = handle_bool_var(os.getenv("MQ_USE_SSL", "false"))

CELERY_APP_NAME = os.getenv("CELERY_APP_NAME", "celery_app")

PLUGIN_PATH = os.getenv('PLUGIN_PATH', '')


def redis_handle_ssl_option(redis_url: str) -> str:
    # Only apply to ssl redis
    if not redis_url.startswith("rediss://"):
        return redis_url

    if "ssl_cert_reqs" not in redis_url:
        separator = "&" if "?" in redis_url else "?"
        return f"{redis_url}{separator}ssl_cert_reqs=none"

    return redis_url


# Redis configuration
def get_redis_backend_url() -> str:
    """Construct Redis backend URL for Celery from environment variables."""

    # For AWS, construct from individual components
    redis_host = os.getenv("REDIS_HOST", "redis")

    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_use_ssl = handle_bool_var(os.getenv("REDIS_USE_SSL", "false"))
    redis_user = os.getenv("REDIS_USERNAME", "")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    encoded_password = quote(redis_password)

    url_prexix = "rediss://" if redis_use_ssl else "redis://"

    url_login = redis_user
    if encoded_password:
        url_login += f":{encoded_password}"
    if url_login:
        url_login += "@"

    url_port = f":{redis_port}" if redis_port else ""
    url = f"{url_prexix}{url_login}{redis_host}{url_port}"

    return redis_handle_ssl_option(url)


REDIS_BACKEND_URL = get_redis_backend_url()


def get_celery_broker_url() -> str:
    """Construct Celery broker URL from environment variables."""

    # Construct from separate environment variables
    mq_host = os.getenv("MQ_HOST", "rabbitmq")
    mq_username = os.getenv("MQ_USERNAME", "")
    mq_password = os.getenv("MQ_PASSWORD", "")
    mq_use_ssl = handle_bool_var(os.getenv("MQ_USE_SSL", "false"))
    mq_port = os.getenv("MQ_PORT", "5672")
    encoded_password = quote(mq_password)

    url_prexix = "amqps://" if mq_use_ssl else "amqp://"

    url_login = mq_username
    if encoded_password:
        url_login += f":{encoded_password}"
    if url_login:
        url_login += "@"

    url_port = f":{mq_port}" if mq_port else ""
    url = f"{url_prexix}{url_login}{mq_host}{url_port}"
    return url


CELERY_BROKER_URL = get_celery_broker_url()
