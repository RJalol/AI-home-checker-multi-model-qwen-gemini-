# Runbook

Start every incident with `GET /metrics` (admin key) and `GET /readyz`.

| Symptom | Metric / check | Likely cause | What to do |
|---|---|---|---|
| Submissions stay `pending_ai` | `hw_oldest_pending_ai_seconds` high, `hw_queue_length` high | no worker running, or workers too slow | `docker compose ps`; start/scale workers. The reconciler re-queues lost jobs by itself. |
| Many `grading_failed` | `hw_submissions{status="grading_failed"}` rising | a provider is down or its key/quota is exhausted | Check worker logs for the provider name. The router already falls back to other providers; if all fail, fix the key/quota, then `POST /submissions/{id}/regrade` (max 3 per submission) or let the teacher regrade. |
| One provider failing | log lines `circuit breaker is open for provider X` | provider outage | Nothing to do: the circuit breaker skips it and the next tier takes over. To force another model: `HWCHECKER_<TIER>_PROVIDER` / `_MODEL`, then restart workers. |
| Provider changed an API field | warning "field rejected, retrying without it" | provider API change | Grading continues without that field (e.g. thinking level) but quality/cost may shift: check the provider docs, update `hwchecker/providers/<name>.py`. |
| AI spend alert | `hw_ai_spend_usd`, log "budget" | retry loop, abuse, wrong tier | Look for one organization sending a burst (429s from `ORG_MAX_SUBMISSIONS_*`); check `HWCHECKER_*_MODEL` overrides; lower the limits. Spend is never blocked automatically. |
| Webhooks not reaching a CRM | `hw_queue_failed{queue="webhook_delivery"}`, `webhook_deliveries` table | CRM down, wrong URL, bad signature check | The CRM can poll `GET /submissions?status=approved&since=...`. Fix the endpoint, then re-run `rq requeue --all` for the failed registry. Delivery retries automatically for a while. |
| HTTP 429 for a school | `Retry-After` header | `ORG_MAX_SUBMISSIONS_PER_MINUTE/_DAY` reached | Check whether it is real traffic or a CRM retry loop; raise the limit for real traffic. |
| API will not start | log "Refusing to start with APP_ENV=production" | unsafe config | Fix every listed item (`config.production_config_problems`). |
| Disk filling up | `uploads/` size | retention job not running | Run `python retention.py` (or `POST /admin/retention`); check `UPLOAD_RETENTION_DAYS`. |
| Redis lost | queue empty, rows `pending_ai` | Redis restart without persistence | Restart Redis; the reconciler re-queues the rows within a few minutes (`POST /admin/reconcile` to force it). |
| Database migration error at start | log from `migrations.py` | a migration failed | Do not edit applied migrations. Restore the backup, fix the new migration, retry. |

## Kill switches (no code change)

* Send a tier to another model: `HWCHECKER_<TIER>_PROVIDER` + `_MODEL` (+ `_THINKING`), restart workers.
* Stop grading but keep accepting uploads: stop the `worker` container. Submissions wait in `pending_ai`
  (the reconciler gives up after `RECONCILE_MAX_REQUEUES` re-queues, then marks them `grading_failed`; they
  can be regraded later).
* Stop everything: stop `api`; nothing is lost, the CRM's retries are idempotent (`external_submission_id`).

## A school asks to delete a student's data

`DELETE /submissions/{id}?reviewer_external_id=...` with that school's key removes the row, the file and the
webhook log (an anonymous audit event stays). The grade in the CRM's own gradebook is the CRM's to remove.
See `PRIVACY_AND_RETENTION.md`.
