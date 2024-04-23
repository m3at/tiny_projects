-- example HTTP POST script which demonstrates setting the
-- HTTP method, body, and adding a header

-- wrk -t1 -c3 -d10s -R5 -sempty_add_task.lua --latency http://127.0.0.1:8000/api/add_task
wrk.method                  = "POST"
wrk.body                    = '{}'
wrk.headers["Content-Type"] = "application/json"
