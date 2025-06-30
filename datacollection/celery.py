import os
from celery import Celery
import logging


logger = logging.getLogger(__name__)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "datacollection.settings")

app = Celery("datacollection")
app.config_from_object("django.conf:settings", namespace="CELERY")



app.autodiscover_tasks()
