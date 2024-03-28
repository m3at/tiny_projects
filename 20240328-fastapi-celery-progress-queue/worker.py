from celery import Celery
import time
import random
from pathlib import Path

app = Celery('worker', broker='amqp://guest:guest@localhost:5672//', backend='rpc://')
# app.conf.task_track_started = True
# In seconds
# Will raise: from celery.exceptions import SoftTimeLimitExceeded
app.conf.task_soft_time_limit = 30
app.conf.task_time_limit = 40

# @app.task(bind=True, track_started=True)
# @app.task(bind=True)
@app.task(bind=True, ignore_result=False, track_started=True)
def process_task(_, name, value):
    start_time = time.perf_counter()

    time.sleep(random.randint(0, 5))
    log_step("Step 1", name)

    # Step 2: Log name and value
    with open("task_log.txt", "a") as file:
        file.write(f"{name}: {value}\n")
    log_step("Step 2", name)

    time.sleep(random.randint(1, 5))
    log_step("Step 3", name)

    with Path("task_log.txt").open("a") as file:
        file.write(f"{name} completed in {time.perf_counter() - start_time} seconds\n")
    log_step("Step 4", name)

    return {"status": "completed", "time_taken": time.time() - start_time}

def log_step(step, name):
    print(f"{step} for task {name}")

if __name__ == '__main__':
    app.start()

