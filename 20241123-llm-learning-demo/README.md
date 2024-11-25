Demo of a webapp using LLMs to create simple lessons on demand, then track the progress on each lesson.



https://github.com/user-attachments/assets/207acc9a-4794-4ce9-971c-b58e8d64180d



Standard django + celery setup, SQLite as db, redis as a broker, tailwind for css.

### Setup

This uses [uv](https://docs.astral.sh/uv/) as a package manager (require version >= 0.5.4), and common tools like `make`, `curl`, `ssh` (or `mosh`) and `rsync`.

For more details, read the makefile. It has nice title boxes, what more can you ask? ✨

Locally:
* Run `make setup`
* Copy .env.example to .env, fill up the required variables


For css generation with tailwind (only required if you modify css properties):
```bash
# Initial messy setup, there's probably a lighter way
make npmtailwindsetup
# Subsequently
make preparecss
```

### Run

```bash
# Prepare db
make migrate
# Run backend
make runserver
# Run celery worker
make runworker
```

TODO: in `content_generator.py`, actually call off to an LLM to generate lessons. For now, some lorem ipsum is generated.

### Deploy

Some key assumptions: nginx setup correctly, `.env` modified for prod, basic familiarity with linux servers.

Locally:
```bash
# Sync local files to remote
make sync_to_remote
```

In the server:
```bash
# Apply migration if necessary
make migrate
# Restart the systemd services
sudo make restart
# Look at the logs
journalctl -xeu django
journalctl -xeu celery
```

### Misc

Why 狐 (kitsune, fox in Japanese)? Because foxes are cute 🦊
