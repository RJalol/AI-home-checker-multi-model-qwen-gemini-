# Deployment

## What runs

| Process | Command | Purpose |
|---|---|---|
| api | `uvicorn api:app --host 0.0.0.0 --port 8000` | REST API (`/submissions`, `/metrics`, ...) |
| worker | `rq worker homework_grading webhook_delivery --with-scheduler` | grading and webhook delivery (run 2+ for load) |
| reconciler | `python reconcile.py --loop 60` | re-queues submissions that lost their job |
| retention | `python retention.py` once a day | deletes old student files |
| PostgreSQL 16, Redis 7 | | data and queue |

`docker compose up -d --build` starts all of it (see `docker-compose.yml`). The Docker image has not
been built in the environment this repo was prepared in: the CI job `docker` builds it on the first run.

## Required secrets (APP_ENV=production refuses to start without them)

`INTERNAL_API_KEY`, `WEBHOOK_SIGNING_SECRET` (different from the first), `DATABASE_URL`, `POSTGRES_PASSWORD`
(compose), and the provider keys (`QWEN_API_KEY`, `GEMINI_API_KEY`, `GLM_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`). Keep them in a secret manager or the host's protected environment, not in git. To
rotate `WEBHOOK_SIGNING_SECRET`, tell every CRM first: all webhook signatures change.

## In front of the API

* Terminate TLS at a reverse proxy (nginx, Caddy, a cloud load balancer).
* Limit the request body there (`client_max_body_size 30m;` in nginx). The application also caps uploads
  at `MAX_FILE_SIZE_MB` and answers HTTP 413, but the proxy stops oversized bodies earlier.
* Expose `/metrics` only to your monitoring network.
* Block the worker host's outbound traffic to private address ranges (webhook DNS-rebinding defence,
  see `webhook_security.py`).

## Database

* Schema changes are versioned (`migrations.py`) and applied by `db.init_db()` at API/worker start.
* Back up PostgreSQL daily and TEST a restore before launch: `pg_dump -Fc homework_checker > backup.dump`,
  then `pg_restore -d homework_checker_restore backup.dump` on a spare server.
* Redis holds the queue and counters. With `--appendonly yes` (compose) a restart keeps queued jobs; if
  Redis is lost, `reconcile.py` re-queues every submission still in `pending_ai`.
* `uploads/` is a volume: back it up if you need the originals, or rely on the retention job.

## Scaling

Grading is I/O-bound (waiting for model APIs): add worker containers (`docker compose up -d --scale worker=4`)
until `hw_queue_length` stays low. Respect the providers' rate limits (`HWCHECKER_RATE_LIMIT_PER_MINUTE`).
Load-test before launch: submit a burst the size of your busiest lesson end and watch `/metrics`.
