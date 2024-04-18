Schedule, persist and process long running tasks with Celery, FastAPI, Redis and SQLite.

### Dev setup

Initial setup using [poetry](https://python-poetry.org/)
```bash
poetry env use 3.12
poetry shell
poetry install --with dev
# Linting and formatting with Ruff
pre-commit install
```

To manually run ruff: `ruff check --fix`.

```bash
# Add new dependencies
poetry add NAME [--group dev]
```

---


Setup:
```bash
# RabbitMQ in Docker
# Optional, can use Redis as the broker as well
# docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.13-management
# Redis 
docker run -p 6379:6379 redis:7.2.4-bookworm
# Python deps
# pip install sqlalchemy "pydantic[email]" uvicorn celery fastapi "redis[hiredis]"
# FastAPI
uvicorn backend:app --reload
# Workers
celery -A worker worker --loglevel=info --concurrency=3 --pool=threads --purge
# Or for actuall parallelism, launch multiple processes with:
# celery -A worker worker --loglevel=info --concurrency=1 --purge
# Optional, Flower for monitoring. Use the appropriate broker depending on what you picked
# celery --broker=amqp://guest:guest@localhost:5672// flower
celery --broker=redis://localhost:6379/0 flower
```

Test:
```bash
# Schedule a task
TASK_ID=$(curl --silent -X 'POST' 'http://127.0.0.1:8000/task/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{ "name": "TestTask" }' | jq '.task_id')
# Check the status of one task
curl -X 'GET' -H 'accept: application/json' "http://127.0.0.1:8000/task/$TASK_ID"
# Get all running tasks
curl -X 'GET' 'http://127.0.0.1:8000/tasks/running'
# Get all completed tasks
curl -X 'GET' 'http://127.0.0.1:8000/tasks/completed'
```

Test with a mock frontend in `./frontend/`:
```bash
cd frontend
uvicorn serve_frontend:app --reload --port 9090
# Then open: http://127.0.0.1:9090/
```

Use [litestream](https://litestream.io/getting-started/) to live replicate the DB:
```bash
# MacOS:
# brew install benbjohnson/litestream/litestream
# Debian:
# wget https://github.com/benbjohnson/litestream/releases/download/v0.3.13/litestream-v0.3.13-linux-amd64.deb && sudo dpkg -i litestream-v0.3.13-linux-amd64.deb
litestream replicate test.db sftp://USER:PASSWORD@HOST.rsync.net:PORT/PATH
```

### Demo

Example with basic task priority:

https://github.com/m3at/tiny_projects/assets/3440771/f94ac944-7113-40ef-99b2-00d324829c81

