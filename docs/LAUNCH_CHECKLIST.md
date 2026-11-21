# Launch checklist

Items marked (code) are done in this repository; the rest need people, accounts or real traffic.

## Before any school
- [x] (code) Input limits, per-org limits, regrade cap, production-mode config check, audit trail, erase, retention, metrics, Docker, CI, migrations.
- [ ] CI is green on GitHub (the Docker build and the security scan have not run yet).
- [ ] Real provider keys in a secret manager; one live test per provider tier (model names, thinking fields, cached-token fields): `python -m hwchecker.benchmarks.golden_eval eval/injection_cases.jsonl` is a quick first live run.
- [ ] Golden set built with teachers and the thresholds accepted (`docs/EVALUATION.md`).
- [ ] Prompt-injection set passes; add real attempts.
- [ ] Legal and provider-terms review done (`docs/PRIVACY_AND_RETENTION.md`).
- [ ] PostgreSQL backup AND a restore test; Redis persistence on.
- [ ] Load test with a burst the size of your busiest lesson end; worker count chosen.
- [ ] Metrics scraped, alerts routed to a person who will answer (`deploy/alerts.yml`).
- [ ] Reverse proxy with TLS and a body-size limit; `/metrics` not public.
- [ ] Independent penetration test.
- [ ] CRM side: sends `external_assignment_id`, `reviewer_external_id`; verifies webhook signatures and timestamps; handles 429 with `Retry-After`; enforces who may approve what.

## Rollout
1. **Shadow mode (1-2 weeks)**: teachers grade as usual; the AI grades the same work in parallel; compare with `golden_eval` on the real pairs. Nothing reaches students.
2. **Pilot (1-2 schools)**: teachers review every AI result before students see it (the normal workflow). Track: share of results teachers edit, average points they change, time saved, student/teacher complaints.
3. **Widen** only while those numbers hold. Re-run the golden set after every model or prompt change.

## Cost
- [ ] After the pilot: replace the estimates in the pricing sheet with measured averages (tier share, tokens, `tokens.cached`, re-check rate) and compare with the provider invoices weekly.
