Copy of `../20240606-thread-signal-handling` with added prometheus monitoring.

```bash
# Start redis
redis-server
# Prometheus server
prometheus --config.file=prometheus.yml
# Run the backend
cd phoenix && ./migrate_and_sqlite_setup.sh && python manage.py runserver
# Run celery workers. Only works with prefork
cd phoenix && celery -A tasks worker --loglevel=info --concurrency=5 --pool=prefork
# Usage
curl 'http://127.0.0.1:8000/api/hatch?name=goose'
curl 'http://127.0.0.1:8000/api/hatch?name=duck'
curl 'http://127.0.0.1:8000/api/update?name=duck&speed=70'
# Kill workers, should pick up tasks that are still scheduled when starting again
pkill -TERM -f 'celery'
# Purge tasks:
cd phoenix && celery -A tasks purge -f
```
