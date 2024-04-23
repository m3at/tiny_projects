Using [this wrk fork](https://github.com/rrva/wrk2), tests on MacBook Air M3.

```bash
# Example of wrk command
# Note that wrk calibrate for 10s, so anything under is not meaningful
# t=threads, c=connections, d=duration, R=throughput
wrk -t4 -c10 -d30s -R100 -sempty_add_task.lua [--latency] http://127.0.0.1:8000/api/add_task
```

Running server with `uvicorn --log-level critical --workers 4 chousaheidan.asgi:application`.  
`failed` typically means that the thread on the server crashed (non-2xx or 3xx responses).

Pushing default sqlite to failure, 30s tests:

threads | connections | throughput (client) | req/s | success | failed
---|---|---|---|---|---
4 | 10 | -R100 | 100 | 3003 | 0
4 | 100 | -R500 | 85.3 | 2564 | 123
4 | 200 | -R500 | 83.7 | 2528 | 519
4 | 300 | -R500 | 73.7 | 2528 | 519
8 | 300 | -R500 | 73.0 | 2207 | 722
8 | 400 | -R5000 | 8.5 | 2113 | 758


Tuning sqlite:
```txt
journal_mode = WAL
synchronous = normal
journal_size_limit = 6144000
page_size = 32768
```

Higher load:

threads | connections | throughput (client) | req/s | success | failed | socket errors
---|---|---|---|---|---|---
4 | 100 | 1000 | 999.3 | 29983 | 0 | 0
8 | 100 | 1000 | 1000.8 | 30032 | 0 | 0
8 | 100 | 2000 | 1999.3 | 59984 | 0 | 0
8 | 100 | 3000 | 2406.4 | 72204 | 0 | 0
8 | 200 | 3000 | 2282.0 | 68467 | 0 | 0
8 | 300 | 3000 | 2192.2 | 65795 | 0 | 765
8 | 400 | 3000 | 1836.4 | 55097 | 0 | 2325
8 | 400 | 5000 | 2284.8 | 68571 | 0 | 2326
16 | 200 | 3000 | 2238.7 | 67189 | 0 | 0

The socket errors are ~95% timeout, the rest is connection error.

Didn't manage to push past 2.4k req/s, some limitations likely due to my machine.
