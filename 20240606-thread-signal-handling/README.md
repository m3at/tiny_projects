Permanently running celery tasks

```bash
# Start redis
redis-server
# Run the backend
cd phoenix && ./migrate_and_sqlite_setup.sh && python manage.py runserver
# Run celery workers. Only works with prefork
cd phoenix && celery -A tasks worker --loglevel=info --concurrency=5 --pool=prefork
# 
curl 'http://127.0.0.1:8000/api/hatch?name=duck_red_1&color=red'
# Kill workers, should pick up tasks that are still scheduled when starting again
pkill -TERM -f 'celery'
# Purge tasks:
cd phoenix && celery -A tasks purge -f
```
