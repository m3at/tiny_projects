"""Tasks and models."""

from datetime import datetime, timezone
from typing import TypeAlias

from sqlmodel import Field, SQLModel
from pydantic import BaseModel

# tasks_status = Literal["NEW", "PENDING", "STARTED", "SUCCESS"]
# tasks_status: TypeAlias = Literal["NEW", "PENDING", "STARTED", "SUCCESS"]
# This causes issues with SQLModel somehow
tasks_status: TypeAlias = str

# Can use instead from 3.12:
# type tasks_status = Literal["NEW", "PENDING", "STARTED", "SUCCESS"]

# from enum import StrEnum, auto
# class Status(StrEnum):
#     NEW = auto()
#     PENDING = auto()
#     STARTED = auto()
#     IN_PROGRESS = auto()
#     SUCCESS = auto()


# Used both for the db and serializing to the task, for funsies
class TaskModel(SQLModel, table=True):
    id: str | None = Field(default=None, primary_key=True, index=True)
    name: str
    status: tasks_status
    progress: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    high_priority: bool = False


# Used for passing progress message to redis
class TaskProgress(BaseModel):
    status: tasks_status
    progress: float = 0.0
