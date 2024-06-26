import logging
from random import random
from time import sleep

from celery import shared_task

from tasks.models import Birds

logger = logging.getLogger("phoenix." + __file__)


@shared_task
def t_count_birds():
    logger.debug(">>> Celery task: t_count_birds")
    count = Birds.objects.count()
    logger.info(f"There's like, {count} birds or something")
    return count


# Run with: celery -A tasks worker --loglevel=info --concurrency=3 --pool=prefork
# Kill with: pkill -TERM -f 'celery'
# celery -A tasks purge -f
@shared_task(acks_late=True, reject_on_worker_lost=True)
def t_track_a_bird(name):
    logger.info(f">>> Celery task: t_track_a_bird, for {name=}")
    c = 0
    bird = Birds.objects.filter(name=name).latest()

    if not bird.is_active:
        logger.debug(
            f"Look like that bird has retired. There's no need to track it then, isn't it? {bird}"
        )
        return

    while True:
        try:
            # This shouldn't be necessary, should only need to call at startup
            # bird = Birds.objects.filter(color=color).latest()
            # print(bird.name)
            logger.debug(f"[age {c:>3}] {bird}")
            sleep(1 + random() * 2)
            c += 1
        except KeyboardInterrupt:
            logger.info("Task interrupted")
            return
