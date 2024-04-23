Django setup

```bash
# Init
cd chousaheidan/
python manage.py startapp tasks
python manage.py makemigrations tasks
python manage.py migrate
DJANGO_SUPERUSER_PASSWORD=admin python manage.py createsuperuser --noinput --username admin --email admin@test.com
# Run the server interractively, check localhost:8000/api/docs
python manage.py runserver
# ASGI server with uvicorn
uvicorn --log-level warning --workers 4 chousaheidan.asgi:application
# TODO: how to set gunicorn up properly? Shouldn't have a big impact on db anyway
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker
# Check
curl -X POST -H "Content-Type: application/json" --data '{}' http://127.0.0.1:8000/api/add_task
```

See [wrk_tests/BENCHMARK.md](./wrk_tests/BENCHMARK.md) for a simple performance check.

---

### Dev setup

Initial setup using [poetry](https://python-poetry.org/)
```bash
poetry shell
poetry install --with dev
# Linting and formatting with Ruff
pre-commit install
```

To manually run, `pre-commit run --all-files`, or ruff only: `ruff check --fix`.

```bash
# Add new dependencies
poetry add NAME [--group dev]
```
