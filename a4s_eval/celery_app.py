from celery import Celery

from a4s_eval.utils import env

celery_app: Celery = Celery(
    __name__, broker=env.CELERY_BROKER_URL, backend=env.REDIS_BACKEND_URL
)
