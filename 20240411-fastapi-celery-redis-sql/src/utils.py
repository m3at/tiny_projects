import time
import random
from typing import Literal

from celery import Celery

# BROKER = "rabbitmq"
BROKER = "redis"


# @singledispatch
def randsleep(start: int, end: int | None = None):
    if end is None:
        end, start = start, 0
    time.sleep(random.randint(start, end))


def randsleepf(start: float, end: float | None = None):
    if end is None:
        end, start = start, 0.0
    r = random.random() * (end - start)
    time.sleep(start + r)


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
        # TODO: use multiple queues instead? Seems easier, as
        # this allow an actual "prefetch" of 0 for high priority tasks
        worker_prefetch_multiplier=1,  # default: 4
        # Might affect some things for priorities:
        # https://docs.celeryq.dev/en/stable/faq.html#faq-acks-late-vs-retry
        # task_acks_late=True,
    )
    return app
