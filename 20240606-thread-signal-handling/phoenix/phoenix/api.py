import logging

import redis
from celery.result import AsyncResult
from django.db.models.signals import post_save
from django.db.utils import IntegrityError
from django.dispatch import receiver
from ninja import NinjaAPI

# from django_celery_results.models import TaskResult
from tasks.celery_instance import app as celery_app
from tasks.celery_tasks import t_track_a_bird
from tasks.models import Birds

from phoenix.api_specs.schemas import (
    FailMessage,
    SuccessMessage,
)

logger = logging.getLogger("phoenix." + __file__)

# TODO: fill with currently running task, per color at startup
TASKS_IDS = False

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


# @receiver(post_save, sender=Birds)
# def callback_send_signal(sender, **_):
#     logger.error(f"Django signal on `post_save`, for table Birds: {sender=}")
#     send_signal_to_command()

# # a = celery_app.control.inspect().active()
# a = TaskResult.objects.all()
# if a is None:
#     tasks_ids = {}
# else:
#     print(f">>>\n\n{a}\n\n<<<")
#
#     tasks_ids = {}

# {'celery@Pauls-Air.local': [{'id': 'b0360eeb-9c2c-45cf-ba87-20cc48290b82', 'name': 'tasks.celery_tasks.t_track_a_bird', 'args': ['red'], 'kwargs': {}, 'type': 'tasks.celery_tasks.t_track_a_bird', 'hostname': 'celery@Pauls-Air.local', 'time_start': 1717693916.9852536, 'acknowledged': True, 'delivery_info': {'exchange': '', 'routing_key': 'celery', 'priority': 0, 'redelivered': False}, 'worker_pid': 34939}]}
# buff = []
# for k, v in a.items():
#     buff.extend(v)
#
# tasks_ids = {
#     # e["args"][0]: AsyncResult(id=e["id"]) for e in buff
#     # Not getting the right task? Something is up
#     e["args"][0]: AsyncResult(**e) for e in buff
# }


# @api.get("/ping")
def get_tasks_ids():
    tasks_ids = {}

    # Discard all waiting tasks.
    # c = celery_app.control.discard_all()

    # a = celery_app.control.inspect().active()
    # https://docs.celeryq.dev/en/stable/reference/celery.app.control.html#celery.app.control.Inspect.query_task
    insp = celery_app.control.inspect()
    a = insp.active()

    print(f"???\n\n{insp.scheduled()}\n\n???")
    # print(f"!!!\n\n{insp.reserved()}\n\n!!!")
    # print(f"!!!\n\n{insp.registered()}\n\n!!!")

    # _ids = [x["id"] for _, v in insp.active().items() for x in v ]
    # a = insp.query_task(_ids)

    # {'celery@Pauls-Air.local': {
    #     '1e08c2f8-7bb4-44a6-8dc9-dbd81ab142c9': [
    #         'active',
    #         {'id': '1e08c2f8-7bb4-44a6-8dc9-dbd81ab142c9', 'name': 'tasks.celery_tasks.t_track_a_bird', 'args': ['green'], 'kwargs': {}, 'type': 'tasks.celery_tasks.t_track_a_bird', 'hostname': 'celery@Pauls-Air.local', 'time_start': 1717723479.630555, 'acknowledged': True, 'delivery_info': {'exchange': '', 'routing_key': 'celery', 'priority': 0, 'redelivered': False}, 'worker_pid': 42583}
    #     ],
    #     '84fb4426-a473-41d1-ad1c-94c1e18a22ab': ['active', {'id': '84fb4426-a473-41d1-ad1c-94c1e18a22ab', 'name': 'tasks.celery_tasks.t_track_a_bird', 'args': ['green'], 'kwargs': {}, 'type': 'tasks.celery_tasks.t_track_a_bird', 'hostname': 'celery@Pauls-Air.local', 'time_start': 1717723319.6865015, 'acknowledged': True, 'delivery_info': {'exchange': '', 'routing_key': 'celery', 'priority': 0, 'redelivered': False}, 'worker_pid': 42198}]
    # }}

    # a = celery_app.Worker
    # a = celery_app.control.objects().active()
    # a = TaskResult.objects.all()
    print(f">>>\n\n{a}\n\n<<<")
    if a is None:
        return {}

    # print(f">>>\n\n{a}\n\n<<<")

    # tasks_ids = {}
    # {'celery@Pauls-Air.local': [{
    #     'id': 'b0360eeb-9c2c-45cf-ba87-20cc48290b82',
    #     'name': 'tasks.celery_tasks.t_track_a_bird',
    #     'args': ['red'],
    #     'kwargs': {},
    #     'type': 'tasks.celery_tasks.t_track_a_bird',
    #     'hostname': 'celery@Pauls-Air.local',
    #     'time_start': 1717693916.9852536,
    #     'acknowledged': True,
    #     'delivery_info': {'exchange': '', 'routing_key': 'celery', 'priority': 0, 'redelivered': False},
    #     'worker_pid': 34939
    # }]}
    buff = []
    # for k, v in a.items():
    #     buff.extend(v)
    for v in a.values():
        buff.extend(v)

    # tasks_ids = {
    #     # e["args"][0]: AsyncResult(id=e["id"]) for e in buff
    #     # Not getting the right task? Something is up
    #     # e["args"][0]: AsyncResult(**e) for e in buff
    #     e["args"][0]: AsyncResult(id=e["id"], task_name=e["name"], app=celery_app) for e in buff
    # }
    for e in buff:
        tasks_ids[e["args"][0]] = AsyncResult(
            id=e["id"], task_name=e["name"], app=celery_app
        )

    return tasks_ids


# print(celery_app.control.inspect().active())
# # tasks_ids = {t for t in celery_app.control.inspect() if t.active()}
# t = [x for x in celery_app.tasks if x.startswith('tasks.celery_tasks.t_track_a_bird')]
# print(t)
# tasks_ids = {t.color: t for t in celery_app.tasks}


@receiver(post_save, sender=Birds)
def restart_task(sender, instance, **_):
    global TASKS_IDS

    if not TASKS_IDS:
        TASKS_IDS = get_tasks_ids()

    logger.error(f"Django signal on `post_save`, for table Birds: {sender=}")
    # t_track_a_bird.apply_async(countdown=0)
    # logger.error(f"{instance.color=} {instance=}")
    # celery_app.control.revoke("<TASK_ID>", terminate=True)
    # https://docs.celeryq.dev/en/latest/reference/celery.result.html#celery.result.AsyncResult.revoke
    color = instance.color

    if color in TASKS_IDS:
        task = TASKS_IDS[color]

        # async_result = AsyncResult(id=task.id)
        logger.info(f"Trying to revoke {task=}")
        # celery_app.control.revoke(task, terminate=True)
        # async_result.revoke(terminate=False, signal="KILL", wait=True, timeout=5)
        # terminate=True doesn't work for threads. Does it make sense anyway?

        # task.revoke(terminate=True, signal="TERM", wait=True, timeout=2)
        task.revoke(terminate=True, signal="KILL", wait=True, timeout=2)

        # task.revoke(terminate=False, signal="KILL", wait=True, timeout=2)

        # task.revoke(terminate=False, signal="KILL", wait=False)
        # celery_app.control.revoke(task.id)

    # task = t_track_a_bird.apply_async((color,), countdown=0)
    task = t_track_a_bird.apply_async((color,))  # type:ignore
    # tasks_ids[color] = task.id
    TASKS_IDS[color] = task


@api.get("/hatch", response={200: SuccessMessage, 418: FailMessage})
def hatch(
    request,
    name: str,
    flying_speed: int | None = None,
    emoji: str | None = None,
    color: str | None = None,
):
    args = dict(name=name, flying_speed=flying_speed, emoji=emoji, color=color)
    args = {k: v for k, v in args.items() if v is not None}
    try:
        b = Birds(**args)
        b.save()
    except IntegrityError:
        logger.error(f"There can only be one {name}!!")
        return 418, {"message": "I'm a teapot"}

    logger.info(b)

    return 200, {"message": "I'm flying"}
