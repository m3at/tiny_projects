Schedule, persist and process long running tasks with Celery, FastAPI and SQLite.


Setup:
```bash
# RabbitMQ in Docker
docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.13-management
# Python deps
pip install sqlalchemy "pydantic[email]" uvicorn celery fastapi
# FastAPI
uvicorn main:app --reload
# Workers
celery -A worker worker --loglevel=info --concurrency=3 --pool=threads --purge
```

Test:
```bash
# Schedule a task
TASK_ID=$(curl --silent -X 'POST' 'http://127.0.0.1:8000/task/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{ "name": "TestTask", "value": 123.45 }' | jq '.task_id')
# Check the status of one task
curl -X 'GET' -H 'accept: application/json' "http://127.0.0.1:8000/task/$TASK_ID"
# Get all running tasks
curl -X 'GET' 'http://127.0.0.1:8000/tasks/running'
# Get all completed tasks
curl -X 'GET' 'http://127.0.0.1:8000/tasks/completed'
```

Test with a mock frontend:
```bash
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

