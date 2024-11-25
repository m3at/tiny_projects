import logging
import os

from celery import Celery, signals
from django.conf import settings

logger = logging.getLogger("kitsune." + __file__)

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kitsune.settings")

app = Celery("tasks")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object(settings, namespace="CELERY", force=True)

# Prefix all keys with a custom string. Probably overkill when broker db are separated
app.conf.broker_transport_options = {"global_keyprefix": settings.CELERY_GLOBAL_PREFIX}

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


# By default, celery will setup it's own logger. Disable that, but it would be better to keep it above some level
# https://wdicc.com/logging-in-celery-and-django/
# https://stackoverflow.com/questions/48289809/celery-logger-configuration
@signals.setup_logging.connect
def on_celery_setup_logging(**kwargs):
    pass
