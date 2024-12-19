from celery import Celery
import os


def make_celery(app_name):
    return Celery(
        app_name,
        broker=os.getenv("CELERY_BROKER_URL"),
        backend=os.getenv("CELERY_RESULT_BACKEND"),
    )


celery = make_celery(__name__)
