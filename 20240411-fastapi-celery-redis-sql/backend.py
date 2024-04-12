import redis
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy import create_engine, Column, String, Float, DateTime, func
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session

from sqlmodel import Session, SQLModel, create_engine, select

# Local
from src.tasks import TaskModel
from src.utils import get_veggies

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# SQL setup through SQLModel
DATABASE_URL = "sqlite:///./test.db"
# DATABASE_URL = "sqlite://"  # in-memory only, for debug
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)
# SQLModel.metadata.reflect(engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

celery_app = get_veggies()


def setup_db_session():
    # TODO: this is starting two sessions, one for each FastAPI route
    print(">>> setup_db_session")
    with Session(engine) as session:
        yield session


@app.post("/task/")
def create_task(task: TaskModel, db: Session = Depends(setup_db_session)):
    # TODO: can we do two steps, were we validate the task and queuing before submitting?

    # TODO: could serialize more "officially":
    # https://benninger.ca/posts/celery-serializer-pydantic/
    # https://gist.github.com/martinkozle/37c7eef95f9bbb5ace8bc6e32f379673
    ser_task = task.model_dump_json()
    # TODO: the task.id is of course not set yet

    # Add to Celery/RabbitMQ queue
    celery_task = celery_app.send_task(
        # "worker.process_task", args=[task.name, task.value]
        # "worker.process_task", args=[task.name, 0.0]
        "worker.process_task",
        args=(ser_task,),
    )

    # Prepare for DB
    db_task = TaskModel(
        id=str(celery_task.id),
        name=task.name,
        status="created in backend",
    )

    # Commit
    db.add(db_task)
    db.commit()

    # TODO: serialize and send to redis too, with the same model as json

    return {"task_id": str(celery_task.id)}


@app.get("/task/{task_id}")
def get_task(task_id: str, db: Session = Depends(setup_db_session)):
    # TODO: don't fail but report "not started" when missing?
    task_status: dict = redis_client.hgetall(task_id)  # type:ignore
    if not task_status:
        # raise HTTPException(status_code=404, detail="Task not found")
        return {"status": "unknown", "progress": 0, "result": "N/A"}

    res = db.exec(select(TaskModel).where(TaskModel.id == task_id)).first()

    if res is None:
        raise HTTPException(status_code=404, detail="Task not found in DB")

    print(f"{task_status=}")
    # TODO: don't keep bytes around
    res.status = task_status[b"status"]
    # res.result = task_status.get(b'result', b'').decode("utf-8")
    # res.progress = float(task_status.get(b"result", 0.0))
    _progress = task_status.get(b"result", 0.0).decode("utf-8")
    # _progress = -1.0 if _progress == "" else float(_progress)
    db.commit()

    # return {"status": res.status, "progress": res.progress}
    return {"status": res.status, "result": res.progress}


@app.get("/tasks/running")
def get_running_tasks(db: Session = Depends(setup_db_session)):
    # Currently, when empty raises:
    # sqlite3.OperationalError: no such table: taskmodel
    # TODO: check for table? Or populate it as empty?

    # tasks = (
    #     db.query(TaskModel)
    #     .filter(TaskModel.status.in_(["NEW", "PENDING", "STARTED", "RETRY"]))
    #     .all()
    # )
    # tasks = db.exec(select(TaskModel).where(TaskModel.progress < 1)).all()
    tasks = db.exec(select(TaskModel).where(TaskModel.status != "SUCCESS")).all()
    return [
        {"task_id": task.id, "name": task.name, "status": task.status} for task in tasks
    ]


@app.get("/tasks/completed")
def get_completed_tasks(db: Session = Depends(setup_db_session)):
    tasks = db.exec(
        # select(TaskModel).where(TaskModel.progress == 1)
        select(TaskModel).where(TaskModel.status == "SUCCESS")
    ).all()
    return [
        {
            "task_id": task.id,
            "name": task.name,
            "status": task.status,
            "result": task.progress,
        }
        for task in tasks
    ]
