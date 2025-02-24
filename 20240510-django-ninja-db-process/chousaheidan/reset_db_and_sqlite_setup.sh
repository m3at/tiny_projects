#!/usr/bin/env bash

set -eo pipefail

rm -rf db.sqlite3* 2> /dev/null
rm -rf tasks/migrations/0*.py 2> /dev/null

python manage.py makemigrations tasks
python manage.py migrate

# https://www.sqlite.org/pragma.html
# Write-Ahead Logging, most significant
sqlite3 db.sqlite3 'PRAGMA journal_mode = WAL;'
# most common in WAL mode
sqlite3 db.sqlite3 'PRAGMA synchronous = NORMAL;'
# Limit the size of the journal; unlikely to be an issue
# sqlite3 db.sqlite3 'PRAGMA journal_size_limit = 6144000;'
# negative means size in kibibytes, default -2000
sqlite3 db.sqlite3 'PRAGMA cache_size = -8000;'
# keep temporary tables in memory
sqlite3 db.sqlite3 'PRAGMA temp_store = MEMORY;'
# reduce fragmentation
sqlite3 db.sqlite3 'PRAGMA auto_vacuum = INCREMENTAL;'
# Good to run every so often:
# sqlite3 db.sqlite3 'PRAGMA optimize;'
# Make sure relationships exists. Disabled by default for historical reasons
sqlite3 db.sqlite3 'PRAGMA foreign_keys = ON;'
# Longer timeout
sqlite3 db.sqlite3 'PRAGMA busy_timeout = 5000;'
