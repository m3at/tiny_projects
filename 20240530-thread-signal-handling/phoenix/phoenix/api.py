import logging

import redis
from django.db.models.signals import post_save
from django.db.utils import IntegrityError
from django.dispatch import receiver
from ninja import NinjaAPI
from tasks.models import Birds

from phoenix.api_specs.schemas import (
    FailMessage,
    SuccessMessage,
)

logger = logging.getLogger("phoenix." + __file__)

api = NinjaAPI()


@api.get("/hello")
def hello(request):
    return "Birds aren't real"


@api.get("/count_birds")
def count_birds(request) -> int:
    return Birds.objects.count()


def send_signal_to_command():
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
    # Something flying off a tree? I'm not an artist okay
    redis_client.publish("command_channel", "🌳💨🍃")


@api.get("/kill")
def kill(request):
    send_signal_to_command()


@receiver(post_save, sender=Birds)
def callback_send_signal(sender, **_):
    logger.error(f"Django signal on `post_save`, for table Birds: {sender=}")
    send_signal_to_command()


@api.get("/hatch", response={200: SuccessMessage, 418: FailMessage})
def hatch(
    request, name: str, flying_speed: int | None = None, emoji: str | None = None
):
    args = dict(name=name, flying_speed=flying_speed, emoji=emoji)
    args = {k: v for k, v in args.items() if v is not None}
    try:
        b = Birds(**args)
        b.save()
    except IntegrityError:
        logger.error(f"There can only be one {name}!!")
        return 418, {"message": "I'm a teapot"}

    logger.info(b)

    return 200, {"message": "I'm flying"}
