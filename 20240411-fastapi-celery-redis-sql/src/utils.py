import time
import random
from typing import Literal

from celery import Celery

# BROKER = "rabbitmq"
BROKER = "redis"


# @singledispatch
def randsleep(start: int, end: int | None = None):
    if end is None:
        time.sleep(random.randint(0, start))
    else:
        time.sleep(random.randint(start, end))


def get_veggies(*, broker: Literal["rabbitmq", "redis"] = BROKER) -> Celery:
    if broker == "rabbitmq":
        app = Celery(
            "worker", broker="amqp://guest:guest@localhost:5672//", backend="rpc://"
        )
    if broker == "redis":
        # https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#broker-redis
        # https://docs.celeryq.dev/en/stable/userguide/configuration.html#conf-redis-result-backend
        app = Celery(
            "worker",
            broker="redis://localhost:6379/0",
            # To be able to get return values
            result_backend="redis://localhost:6379/0",
            # Not required, add a prefix to all keys
            result_backend_transport_options={"global_keyprefix": "celery_"},
        )
    else:
        raise NotImplementedError(f"Can't support broker: {broker}")

    # https://docs.celeryq.dev/en/stable/userguide/configuration.html#configuration
    app.conf.update(
        timezone="Asia/Tokyo",
    )
    return app
