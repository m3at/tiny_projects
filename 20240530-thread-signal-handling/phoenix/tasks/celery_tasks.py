import logging

from celery import shared_task

from tasks.models import Birds

logger = logging.getLogger("phoenix." + __file__)


@shared_task
def t_count_birds():
    logger.debug(">>> Celery task: t_count_birds")
    count = Birds.objects.count()
    logger.info(f"There's like, {count} birds or something")
    return count
