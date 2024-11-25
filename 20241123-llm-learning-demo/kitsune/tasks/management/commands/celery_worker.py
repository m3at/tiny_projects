import logging
import shlex
import subprocess

from django.core.management.base import BaseCommand
from django.utils import autoreload

logger = logging.getLogger("kitsune." + __file__)


class Command(BaseCommand):
    help = "Auto restart celery worker on file update"

    def handle(self, *_, **kwargs) -> None:
        logger.info("Starting celery worker with autoreload")

        autoreload.run_with_reloader(self.restart_celery)

    @staticmethod
    def restart_celery() -> None:
        cmd = "pkill celery"
        subprocess.call(shlex.split(cmd))
        cmd = "celery -A tasks worker --loglevel=debug --concurrency=5 --purge"
        subprocess.call(shlex.split(cmd))
