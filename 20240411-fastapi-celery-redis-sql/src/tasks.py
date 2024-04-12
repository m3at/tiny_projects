from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class TaskModel(SQLModel, table=True):
    id: str | None = Field(default=None, primary_key=True, index=True)
    name: str
    status: str
    progress: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
