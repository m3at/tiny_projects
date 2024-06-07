import logging
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
def t_track_a_bird(color):
    logger.info(f">>> Celery task: t_track_a_bird, for {color=}")
    c = 0
    bird = Birds.objects.filter(color=color).latest()
    while True:
        try:
            # This shouldn't be necessary, should only need to call at startup
            # bird = Birds.objects.filter(color=color).latest()
            # print(bird.name)
            logger.debug(f"[{c:>4}]: {bird.name} {bird.emoji} {bird.color}")
            sleep(5)
            c += 1
        except KeyboardInterrupt:
            logger.info("Task interrupted")
            return
