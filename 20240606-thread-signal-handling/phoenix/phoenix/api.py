import logging

from celery.result import AsyncResult
from django.db.models.signals import post_save
from django.db.utils import IntegrityError
from django.dispatch import receiver
from ninja import NinjaAPI
from tasks.celery_instance import app as celery_app
from tasks.celery_tasks import t_track_a_bird
from tasks.models import Birds

from phoenix.api_specs.schemas import (
    FailMessage,
    SuccessMessage,
)

logger = logging.getLogger("phoenix." + __file__)

# Keep track of running tasks
CELERY_TASK_IDS: None | dict[str, AsyncResult] = None

api = NinjaAPI()


@api.get("/hello")
def hello(request):
    return "Birds aren't real"


@api.get("/count_birds")
def count_birds(request) -> int:
    return Birds.objects.count()


def _get_celery_task_ids():
    celery_task_ids = {}

    # Discard all waiting tasks.
    # c = celery_app.control.discard_all()

    # https://docs.celeryq.dev/en/stable/reference/celery.app.control.html#celery.app.control.Inspect.query_task
    active_tasks = celery_app.control.inspect().active()

    if active_tasks is None:
        return celery_task_ids

    for v in active_tasks.values():
        for task_params in v:
            celery_task_ids[task_params["args"][0]] = AsyncResult(
                id=task_params["id"], task_name=task_params["name"], app=celery_app
            )

    return celery_task_ids


@receiver(post_save, sender=Birds)
def restart_task(sender, instance: Birds, **_):
    """On change to `Birds`, trigger tasks."""

    global CELERY_TASK_IDS

    if CELERY_TASK_IDS is None:
        CELERY_TASK_IDS = _get_celery_task_ids()

    logger.info(f"Django signal on `post_save`, for table Birds: {sender=}")
    name = instance.name

    #
    if name in CELERY_TASK_IDS:
        task = CELERY_TASK_IDS[name]

        # https://docs.celeryq.dev/en/latest/reference/celery.result.html#celery.result.AsyncResult.revoke
        logger.debug(f"Task already running. Revoking {task=}")
        # Is it useful to send SIGTERM first without killing?
        # task.revoke(terminate=False, signal="TERM", wait=True, timeout=2)
        task.revoke(terminate=True, signal="KILL", wait=True, timeout=1)

    if not instance.is_active:
        logger.info(f"Not active, not triggering a restart: {name=}")
        return

    logger.debug(f"Starting new task for {name=}")
    task = t_track_a_bird.apply_async((name,))  # type:ignore

    # Keep track of the new task
    CELERY_TASK_IDS[name] = task


@api.get("/hatch", response={200: SuccessMessage, 418: FailMessage})
def hatch(
    request,
    name: str,
    speed: int | None = None,
):
    # This is very important
    emoji = dict(
        goose="🪿",
        dove="🕊️",
        eagle="🦅",
        duck="🦆",
        phoenix="🐦‍🔥",
        parrot="🦜",
        swan="🦢",
        rooster="🐓",
        flamingo="🦩",
    ).get(name, None)

    args = dict(name=name, speed=speed, emoji=emoji)
    args = {k: v for k, v in args.items() if v is not None}

    try:
        b = Birds(**args)
        b.save()
    except IntegrityError:
        logger.error(f"There can only be one {name}!!")
        return 418, {"message": "I'm a teapot"}

    logger.info(b)

    return 200, {"message": "New bird flying"}


@api.get("/update", response={200: SuccessMessage, 418: FailMessage})
def update(
    request,
    name: str,
    speed: int | None = None,
    is_active: bool | None = None,
):
    try:
        b = Birds.objects.get(name=name)
    except Birds.DoesNotExist:
        logger.error(f"Never heard of '{name}'. Is it real?")
        return 418, {"message": "I'm a teapot"}

    b.speed = b.speed if speed is None else speed
    b.is_active = b.is_active if is_active is None else is_active

    b.save()

    return 200, {
        "message": f"What does it mean to 'update' a bird? Strange concept. Anyway: {b}"
    }
