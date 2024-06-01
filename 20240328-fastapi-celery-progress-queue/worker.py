import random
import time

import redis
from celery import Celery

app = Celery("worker", broker="amqp://guest:guest@localhost:5672//", backend="rpc://")

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)


# @app.task(bind=True, ignore_result=False, track_started=True)
@app.task(bind=True, ignore_result=False, track_started=False)
def process_task(_, name, value):
    start_time = time.perf_counter()

    # Update task status in Redis
    task_id = _.request.id
    update_task_status(task_id, "STARTED", 0)

    time.sleep(random.randint(1, 5))
    update_task_status(task_id, "PROGRESS 25", 25)

    # Simulate task progress
    time.sleep(random.randint(1, 5))
    update_task_status(task_id, "PROGRESS 50", 50)

    time.sleep(random.randint(1, 5))
    update_task_status(task_id, "PROGRESS 75", 75)

    time.sleep(random.randint(1, 5))
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
    app.start()
