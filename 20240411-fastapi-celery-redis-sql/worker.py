import time

import redis

# Local
from src.utils import randsleep, get_veggies
from src.tasks import TaskModel

# celery_app = Celery('worker', broker='amqp://guest:guest@localhost:5672//', backend='rpc://')
celery_app = get_veggies()

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)


# @celery_app.task(bind=True, ignore_result=False, track_started=False)
# @celery_app.task
# bind=True add self as first argument, containing things like self.request.id
@celery_app.task(bind=True)
# def process_task(task_model: TaskModel):
def process_task(self, task_model_json_str: str):
    start_time = time.perf_counter()

    # Update task status in Redis
    # task_id = _.request.id
    task_model = TaskModel.model_validate_json(task_model_json_str)
    # task_id = task_model.id
    task_id = self.request.id
    print(task_model)
    print(f"{task_id=}")
    name = task_model.name
    update_task_status(task_id, "STARTED", 0)

    randsleep(5)
    update_task_status(task_id, "PROGRESS 25", 25)

    # Simulate task progress
    randsleep(5)
    update_task_status(task_id, "PROGRESS 50", 50)

    randsleep(5)
    update_task_status(task_id, "PROGRESS 75", 75)

    randsleep(5)
    update_task_status(
        task_id,
        "SUCCESS",
        100,
        f"{name} completed in {time.perf_counter() - start_time} seconds",
    )

    return {"status": "completed", "time_taken": time.time() - start_time}


def update_task_status(task_id, status, progress, result=None):
    redis_client.hmset(
        task_id,
        {"status": status, "progress": progress, "result": result if result else ""},
    )
    # redis_client.expire(task_id, 3600)


if __name__ == "__main__":
    celery_app.start()
