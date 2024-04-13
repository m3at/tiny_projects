import time

import redis

# Local
from src.utils import randsleepf, get_veggies
from src.tasks import TaskModel

celery_app = get_veggies()

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)


# bind=True add self as first argument, containing things like self.request.id
@celery_app.task(bind=True)
def process_task(self, task_model_json_str: str):
    start_time = time.perf_counter()

    # Update task status in Redis
    task_model = TaskModel.model_validate_json(task_model_json_str)
    task_id = self.request.id
    print(f"{task_id=}, {task_model=}")

    def update_task_status(status, progress):
        redis_client.hmset(
            task_id,
            {"status": status, "progress": progress},
        )

    # name = task_model.name
    update_task_status("STARTED", 0)

    # Simulate task progress
    steps = 20
    for i in range(steps):
        randsleepf(0.02, 1)
        update_task_status("IN_PROGRESS", i / steps)

    randsleepf(1)
    update_task_status("SUCCESS", 1.0)

    # TODO: not actually using the returns, could disable it
    return {
        "status_task": "completed",
        "time_taken": round(time.perf_counter() - start_time, 2),
    }


if __name__ == "__main__":
    celery_app.start()
