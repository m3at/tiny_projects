from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import redis
from celery import Celery
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)


class TaskModel(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    value = Column(Float)
    status = Column(String)  # NEW, RUNNING, COMPLETED
    result = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


# SQLite setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

celery_app = Celery(
    "worker", broker="amqp://guest:guest@localhost:5672//", backend="rpc://"
)


class Task(BaseModel):
    name: str
    value: float


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/task/")
def create_task(task: Task, db: Session = Depends(get_db)):
    celery_task = celery_app.send_task(
        "worker.process_task", args=[task.name, task.value]
    )
    db_task = TaskModel(
        id=str(celery_task.id), name=task.name, value=task.value, status="NEW"
    )
    db.add(db_task)
    db.commit()
    return {"task_id": str(celery_task.id)}


@app.get("/task/{task_id}")
def get_task(task_id: str, db: Session = Depends(get_db)):
    # TODO: don't fail but report "not started" when missing?
    task_status = redis_client.hgetall(task_id)
    if not task_status:
        # raise HTTPException(status_code=404, detail="Task not found")
        return {"status": "unknown", "progress": 0, "result": "N/A"}
    
    # Update the database status based on Redis (optional, depending on your architecture)
    db_task = db.query(TaskModel).filter(TaskModel.id == task_id).first()
    if db_task:
        db_task.status = task_status[b'status'].decode("utf-8")
        db_task.result = task_status.get(b'result', b'').decode("utf-8")
        db.commit()

    return {"status": task_status[b'status'].decode("utf-8"), "progress": int(task_status[b'progress']), "result": task_status.get(b'result', b'').decode("utf-8")}


@app.get("/tasks/running")
def get_running_tasks(db: Session = Depends(get_db)):
    tasks = (
        db.query(TaskModel)
        .filter(TaskModel.status.in_(["NEW", "PENDING", "STARTED", "RETRY"]))
        .all()
    )
    return [
        {"task_id": task.id, "name": task.name, "status": task.status} for task in tasks
    ]


@app.get("/tasks/completed")
def get_completed_tasks(db: Session = Depends(get_db)):
    tasks = db.query(TaskModel).filter(TaskModel.status == "SUCCESS").all()
    return [
        {
            "task_id": task.id,
            "name": task.name,
            "status": task.status,
            "result": task.result,
        }
        for task in tasks
    ]
