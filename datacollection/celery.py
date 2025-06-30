import os
from celery import Celery
import logging
from celery.schedules import crontab


logger = logging.getLogger(__name__)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "datacollection.settings")

app = Celery("iheardatacollection")
app.config_from_object("django.conf:settings", namespace="CELERY")

app.conf.beat_schedule = {}


app.autodiscover_tasks()
