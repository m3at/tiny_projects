Example of SIGUSR1/SIGUSR2 signal handling in `example_signal_handle.py`.

Example of two steps signaling:
* django backend -> django command with redis
* djago command thread to main thread with unix's signal

Usage:
```bash
# Redis
redis-server
# In phoenix/
./migrate_and_sqlite_setup.sh
# Main server
python manage.py runserver
# Command
python manage.py runrespawn
# Create a bird
curl 'http://127.0.0.1:8000/api/hatch?name=dove&emoji=🕊️'
# See a new phoenix rise (the command restarting it's main loop)
# Kill the phoenix on new entry to the bird table
curl 'http://127.0.0.1:8000/api/hatch?name=goose&emoji=🪿'
```

Tried async ways too but ended up very clunky. Using celery workers would be much cleaner than either ways.



https://github.com/m3at/tiny_projects/assets/3440771/2ce70c16-4dcc-4620-bb11-9ba06b0f2a25

