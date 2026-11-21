# Homework Checker AI

An AI teaching assistant that grades university-level homework across
many subjects and language skills (PDF, image, Word doc, or audio for
Speaking — typed or handwritten) question-by-question, using Gemini as
the grading engine. Produces HTML, PDF, and Excel reports, and is
usable both from a Streamlit teacher dashboard and from a REST API for
integrating with other apps/websites.

```
Upload Homework  (Streamlit UI  --or--  REST API call from another app)
      ↓
Python Backend
      ↓
Gemini API  (system prompt built per subject / language+skill)
      ↓
Homework Analyzer  →  JSON Response
      ↓
Generate: HTML report · PDF report · Excel gradebook
      ↓
Teacher Dashboard  --or--  JSON/file returned to the calling app
```

## Project structure

```
HomeworkCheckerAI/
├── app.py                 # Streamlit UI (upload, review, export)
├── api.py                  # FastAPI REST API (for other apps/websites)
├── config.py                # Paths, model list, backend switch, grading scale
├── prompts.py                 # Subject/language-skill system prompts
├── homework_checker.py         # Gemini backend + get_checker() factory
├── local_checker.py             # Self-hosted (vLLM/open-model) backend
├── cloud_checker.py               # OpenAI/DeepSeek/Qwen backend (get_checker() multi-provider mode)
├── anthropic_checker.py            # Claude backend (get_checker() multi-provider mode)
├── training_logger.py            # Logs data for future fine-tuning
├── token_usage_logger.py          # Logs input/output/cached tokens per submission
├── jobs.py                        # Redis/RQ queue for POST /jobs (async grading)
├── worker_tasks.py                 # The grading task run by `rq worker`
├── cache_registry.py                # Shared Gemini context-cache registry (Redis-backed)
├── batch_processor.py                # Gemini Batch API for bulk/overnight grading
├── db.py                              # PostgreSQL: orgs/teachers/students/submissions (Phase 3)
├── report_generator.py            # HTML / PDF / Excel report builders
├── utils.py                        # Upload validation, docx text extraction
├── hwchecker/                       # Phase 4: tiered multi-model grading engine (see below)
│   ├── providers/                     # AIProvider implementations + PROVIDER_REGISTRY
│   ├── subjects/                       # SubjectProfile data (rubrics, error taxonomy) + SUBJECT_REGISTRY
│   ├── checkers/                        # $0 deterministic checkers (SymPy, Pint, MCQ, ...)
│   ├── router/                           # difficulty routing + GradingRouter orchestration
│   ├── pipeline/                          # OCR routing + output JSON schema/validation
│   ├── infra/                              # retry, circuit breaker, rate limiter, cache, monitoring
│   ├── benchmarks/                          # 60-case benchmark loader + metrics runner
│   ├── language.py                            # FEEDBACK_LANGUAGE: which language the AI answers in
│   └── worker_entrypoint.py                  # RQ task entrypoint for this pipeline
├── tests/                          # unittest/pytest suite, runs offline with fake AI SDKs (see below)
├── uploads/                         # Saved student submissions (gitignored)
├── reports/                          # Generated reports (gitignored)
├── training_data/                     # Fine-tuning dataset (gitignored)
│   ├── inputs/                          # Copies of graded input files
│   ├── raw_log.jsonl                     # Every graded submission (unverified)
│   └── gold.jsonl                         # Teacher-verified examples
├── templates/
│   └── report.html                        # Jinja2 HTML report template
├── static/
│   └── fonts/                              # Bundled Unicode font for PDF reports
├── requirements.txt
├── .gitignore
└── .env.example
```



## Setup

Commands below are grouped by OS where they differ. Everything else
(the Python code itself, requirements.txt, the app's behavior) is
identical across all three — only how you create a venv, activate it,
and set environment variables differs.

### 1. Install Python 3.10+

- **Windows**: download from [python.org](https://www.python.org/downloads/) (check "Add python.exe to PATH" during install), or `winget install Python.Python.3.12`
- **macOS**: `brew install python@3.12` (or use the python.org installer)
- **Linux**: usually preinstalled; if not, `sudo apt install python3 python3-venv python3-pip` (Debian/Ubuntu)

### 2. Create and activate a virtual environment, install dependencies

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> If `Activate.ps1` is blocked by execution policy, run PowerShell as
> Administrator once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Get a Gemini API key

From Google AI Studio: https://aistudio.google.com/apikey

### 4. Set your API key

Either paste it directly into the sidebar text box when the app is
running (simplest for quickly trying things out), or set it as an
environment variable so you don't have to re-enter it each time:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-key-here"
```
This only lasts for the current terminal session. To set it
permanently: `setx GEMINI_API_KEY "your-key-here"` (then open a new terminal).

**Windows (cmd.exe):**
```cmd
set GEMINI_API_KEY=your-key-here
```

**macOS / Linux (bash/zsh):**
```bash
export GEMINI_API_KEY=your-key-here
```
To make this permanent, add that line to `~/.zshrc` (macOS default
shell) or `~/.bashrc` (Linux/bash), then restart your terminal.

**Recommended for all OSes: use a `.env` file instead.** It's loaded
automatically (via `python-dotenv`, already in requirements.txt) every
time the app starts, so you never need to re-run export/set/`$env:`
commands each session — and it's the same one file regardless of OS:
```bash
cp .env.example .env    # Windows: copy .env.example .env
# then edit .env and paste your key
```

### 5. Run the Streamlit app (teacher dashboard)

Same command on every OS, once the venv is activated:
```bash
streamlit run app.py
```
Streamlit will open the app in your browser (default: http://localhost:8501).

### 6. Run the REST API (optional — for other apps/websites to call)

```bash
uvicorn api:app --reload --port 8000
```
Interactive docs (Swagger UI) then live at http://localhost:8000/docs
— you can try every endpoint directly from the browser there.

## REST API (api.py)

Other apps/websites submit homework here instead of through the
Streamlit UI. It returns the exact same JSON schema the dashboard shows.

### Endpoints

| Method | Path                 | Purpose                                              |
|--------|----------------------|-------------------------------------------------------|
| GET    | `/health`             | Health check                                          |
| GET    | `/backend`             | Current MODEL_BACKEND + training-data stats           |
| GET    | `/usage`                | Token usage totals + breakdown by subject/model       |
| GET    | `/subjects`            | List standard subject keys/labels                     |
| GET    | `/languages`            | List language keys/labels                             |
| GET    | `/skills`                | List language-skill keys/labels                       |
| POST   | `/analyze`                 | Grade one file **synchronously** (blocks until done) |
| POST   | `/jobs`                      | **Enqueue** one file for grading, returns instantly  |
| GET    | `/jobs/{job_id}`               | Poll a job's status/result                          |
| POST   | `/jobs/batch`                    | Grade many files as one 50%-off Batch job          |
| GET    | `/jobs/batch/{batch_name}`         | Poll a batch job's status/results                |
| POST   | `/organizations`                     | **[Admin]** Register a new CRM tenant            |
| POST   | `/submissions`                         | Submit homework into the **teacher-review** workflow |
| GET    | `/submissions` / `/submissions/{id}`     | Query the review queue / one submission's state |
| POST   | `/submissions/{id}/review`                 | Teacher approves or edits + approves         |
| POST   | `/reports/html`              | Turn an `/analyze`-shaped response into an HTML file  |
| POST   | `/reports/pdf`                 | Turn an `/analyze`-shaped response into a PDF file  |
| POST   | `/reports/gradebook`             | Combine several such responses into one .xlsx     |

**Which one to use:**
- Testing / low-volume / no review needed → `/analyze`
- CRM sends many at once, AI result is trusted as final → `/jobs` (see
  "Handling many simultaneous submissions" below)
- Bulk/overnight grading where cost matters more than speed → `/jobs/batch`
  (see "Reducing cost at scale" below)
- **A teacher must review/approve before it counts as a real grade** →
  `/submissions` + `/submissions/{id}/review` (see "CRM review workflow"
  below) — this is the recommended path for a production CRM integration.

### Grading a standard subject

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@homework.pdf" \
  -F "subject_key=math" \
  -F "student_name=Ali" \
  -F "api_key=YOUR_GEMINI_API_KEY"
```

### Grading a language skill (e.g. English Speaking)

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@recording.mp3" \
  -F "language_key=english" \
  -F "skill_key=speaking" \
  -F "student_name=Ali" \
  -F "api_key=YOUR_GEMINI_API_KEY"
```

### Turning a result into a downloadable report

```bash
# save the /analyze response to result.json first, then:
curl -X POST http://localhost:8000/reports/pdf \
  -H "Content-Type: application/json" \
  -d @result.json \
  -o report.pdf
```

## Handling many simultaneous submissions (CRM integration)

If your CRM can send many homework submissions at roughly the same time
(e.g. a whole class submitting near a deadline), calling `/analyze`
directly for each one has real limits: each call blocks until Gemini
responds (often 10-30+ seconds), so many at once can queue up behind
each other, risk your CRM's own HTTP timeout, and can trigger Gemini's
rate limits if too many fire in the same moment.

**`/jobs` solves this with a queue.** It returns a `job_id` in
milliseconds — grading happens afterward, in a separate background
worker process, so the API itself never blocks no matter how many
submissions land at once.

### Setup (one-time)

1. **Install and run Redis** (the queue's backing store):

   **macOS:**
   ```bash
   brew install redis && brew services start redis
   ```

   **Linux:**
   ```bash
   sudo apt install redis-server && sudo systemctl start redis-server
   ```

   **Windows:** Redis itself isn't officially supported/maintained for
   Windows. Three practical options, easiest first:
   - **Docker** (recommended if you have Docker Desktop):
     ```powershell
     docker run -d --name redis -p 6379:6379 redis:latest
     ```
   - **WSL2** (Windows Subsystem for Linux) — install Ubuntu from the
     Microsoft Store, then inside the WSL terminal run the Linux
     commands above (`sudo apt install redis-server ...`). The API and
     workers can run on Windows and still reach Redis in WSL at
     `localhost:6379` (WSL2 shares localhost with Windows by default).
   - **[Memurai](https://www.memurai.com/)** — a native Windows-compatible
     Redis alternative with a free tier, if you'd rather avoid Docker/WSL.

   Whichever option you use, confirm it's reachable:
   ```bash
   redis-cli ping   # should print: PONG
   ```

2. **Start one or more workers** (separate terminal/process from the API):
   ```bash
   rq worker homework_grading
   ```
   Run this command multiple times (in separate terminals, or on
   separate machines, all pointed at the same `REDIS_URL`) to process
   several submissions in parallel — each worker instance handles one
   job at a time, so N workers = N submissions graded simultaneously.

   > **Windows note:** `rq worker`'s default process-forking mode relies
   > on Unix-only APIs and does not work on native Windows. Either run
   > workers inside WSL2 (recommended — same environment as Linux), or
   > on native Windows add `--worker-class rq.worker.SimpleWorker` to
   > the command above, which avoids forking at the cost of slightly
   > less isolation between jobs.

3. **Start the API as usual:**
   ```bash
   uvicorn api:app --port 8000
   ```

### Usage

```bash
# 1. Submit -- returns instantly:
curl -X POST http://localhost:8000/jobs \
  -F "file=@homework.pdf" -F "subject_key=math" \
  -F "student_name=Ali" -F "api_key=YOUR_GEMINI_API_KEY"
# -> {"job_id": "27bbae43-...", "status": "queued"}

# 2. Poll until done (your CRM does this on a timer, e.g. every few seconds):
curl http://localhost:8000/jobs/27bbae43-...
# while working:  {"job_id": "...", "status": "started"}
# when done:      {"job_id": "...", "status": "finished", "result": { ...same JSON /analyze returns... }}
# on failure:     {"job_id": "...", "status": "failed", "error": "..."}
```

### Running this in production

- Keep the worker(s) running continuously with a process manager —
  `systemd`, `supervisor`, or a Docker container with `restart: always` —
  so a crashed worker restarts automatically instead of silently
  stopping job processing.
- Scale throughput by running more `rq worker homework_grading`
  processes, not by adding more API instances — the API's job is just
  to enqueue quickly, which it already does.
- `JOB_TIMEOUT_SECONDS` (config.py, default 300) caps how long a single
  job can run before RQ marks it failed — raise it if you expect very
  large files or a slow local model.
- `JOB_RESULT_TTL_SECONDS` (default 24 hours) controls how long a
  finished job's result stays fetchable via `/jobs/{id}` before Redis
  expires it — make sure your CRM polls (or fetches) well within that
  window.

### Protecting the API (required)

The API **fails closed**: while `INTERNAL_API_KEY` is empty, every protected endpoint
(`/analyze`, `/jobs`, `/reports/*`, `/organizations`, `/backend`, `/usage`, `/admin/*`) answers
**HTTP 503** instead of serving anonymous callers. Set the key on the server:

```bash
# Windows (PowerShell):
$env:INTERNAL_API_KEY="some-long-random-string"

# Windows (cmd.exe):
set INTERNAL_API_KEY=some-long-random-string

# macOS / Linux:
export INTERNAL_API_KEY="some-long-random-string"
```

(In production, on any OS, this is usually set in `.env` or your process
manager's environment config rather than typed into a terminal each time.)

Every caller must then include it as a header:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-Internal-Api-Key: some-long-random-string" \
  -F "file=@homework.pdf" -F "subject_key=math" -F "api_key=YOUR_GEMINI_API_KEY"
```

This stops random internet traffic from burning through your Gemini
quota/billing once the API is reachable from outside your machine. It's
unrelated to `GEMINI_API_KEY`/the `api_key` field — that one authenticates
*to Gemini*; `INTERNAL_API_KEY` authenticates *callers of your API*. The
header is compared in constant time.

For **local development only** you can run without a key:
`export ALLOW_UNAUTHENTICATED_API=true` (the server logs a warning; never set this
where the port is reachable by others).

`/health` (liveness), `/readyz` (PostgreSQL + Redis reachable, 503 otherwise) and the
static discovery lists (`/subjects`, `/languages`, `/skills`) need no key.

CORS is **off** unless you list your frontend origin(s):
`export CORS_ALLOW_ORIGINS="https://crm.example.com,https://app.example.com"`.

`POST /jobs` no longer accepts a per-request `api_key` (job arguments are stored in Redis);
configure the provider keys on the server. `POST /analyze` still accepts one.

There is no rate limiting in the application: put the API behind a gateway/reverse proxy
that limits requests per key or IP.

## Reducing cost at scale (context caching + Batch API)

At high volume (thousands+ of submissions/day), two Gemini features cut
cost significantly without any change to grading quality. Both are
already wired into the app -- there's nothing extra to turn on beyond
what's described here.

### Explicit context caching (automatic, ~90% off the cached portion)

The system prompt for a given subject or language+skill is identical
across every request for it — that repeated content is exactly what
Gemini's explicit context caching is for: cached input tokens are billed
at a small fraction of the normal rate.

This is handled automatically by `cache_registry.py` and
`homework_checker.py` — nothing to configure to get the benefit:

1. The first request for a given (model, subject/skill) combination
   creates a cache and stores its name in Redis.
2. Every subsequent request for that same combination — from *any*
   process (any RQ worker, the API process, a Streamlit session) —
   looks up and reuses that one shared cache instead of creating its own.
3. If a prompt turns out to be too short to cache (Gemini has a minimum
   cacheable size), this is detected once and remembered, so the app
   doesn't retry cache creation on every single request — it just sends
   the prompt normally for that combination.
4. Without Redis reachable, caching still works, just per-process
   instead of shared across your whole deployment (fine for a single
   Streamlit session; for multiple workers/processes, run Redis so they
   share one cache instead of each creating their own).

Turn it off with `ENABLE_CONTEXT_CACHING=false` if you ever need to
(e.g. while debugging a prompt change, since a cached prompt keeps being
used for up to `CACHE_TTL_SECONDS` — default 1 hour — after you edit
`prompts.py`, until the cache expires and gets recreated with the new text).

### Batch API (50% off, non-real-time)

`POST /jobs/batch` submits many files as **one** Gemini Batch job — a
flat 50% discount on input+output tokens, in exchange for asynchronous
turnaround (typically minutes in practice, guaranteed within ~24h by
Google's SLA). This is for bulk/overnight grading runs — e.g. your CRM
exporting a whole day's submissions at once — not for a teacher waiting
on a single result in the Streamlit UI (use `/analyze` or `/jobs` for that).

Batching and caching combine automatically: each item in a batch reuses
the same shared cache as everything else, so both discounts stack.

```bash
# Submit up to many files together (must share one subject/language+skill/model):
curl -X POST http://localhost:8000/jobs/batch \
  -F "files=@student1.pdf" -F "files=@student2.pdf" -F "files=@student3.pdf" \
  -F "student_names=Ali,Vali,Guli" \
  -F "subject_key=math" \
  -F "api_key=YOUR_GEMINI_API_KEY"
# -> {"batch_name": "batches/...", "item_count": 3, "items": {"<id>": "Ali", ...}}

# Poll later (minutes to hours) for results:
curl http://localhost:8000/jobs/batch/batches/...
# -> {"state": "succeeded", "results": {"<id>": {...grading JSON...}, ...}}
```

All files in one batch call must share the same `subject_key` (or
`language_key`+`skill_key`) and `model_name`, since Gemini groups a
batch under one model. Submit separate batch calls if you need to mix
subjects within the same run (e.g. one batch per subject, submitted back
to back — they can all run concurrently server-side).

For a Python script/cron-style workflow instead of the REST endpoints,
use `batch_processor.py` directly — `submit_batch()`, `poll_batch()`, and
a blocking `wait_for_batch()` are all available to import.

### Many students, one homework (assignment sharing)

When one homework is submitted by many students, send the same `external_assignment_id` (your CRM's
id for that homework) with every `POST /submissions`. The first call can also carry `task_text` and
`answer_key` (each up to 20,000 characters); they are stored once on the assignment and reused for
every later student, so the CRM does not have to resend them. The answer key is only used inside the
grading prompt and is never returned by the API.

Why it saves money: the task text and answer key are placed in the prompt *before* the student's
file and are byte-identical for every student of that assignment, so the providers' automatic prompt
caching can reuse that shared prefix (Gemini implicit caching; Qwen implicit cache; OpenAI-protocol
`cached_tokens`). A reference key also lets the model compare instead of solving from scratch.
Caching is not guaranteed by the providers, so measure it: each result's `tokens.cached` (and the
submission's `token_usage.cached_tokens`) is the part of the input served from the cache.

Not done (yet): Gemini Flex (50% off) is documented only for the newer Interactions API, while this
code uses `generate_content`; Qwen Batch is 50% off but cannot be combined with caching. Both need a
separate decision once real `cached` numbers are known. An existing PostgreSQL database gets the new
`assignments` table and `submissions.assignment_id` column from `db.init_db()`.

## CRM review workflow (Phase 3): teacher approval, multi-tenant

This is the intended integration path for a real CRM platform: the AI
grades homework first, but nothing reaches a student's gradebook until
a **teacher reviews and approves it** — either as-is or with edits. This
is a different, more structured flow than `/analyze` or `/jobs` (which
just return the AI's result directly) — it adds a persistent review
queue and multi-tenant data isolation, backed by PostgreSQL (`db.py`).

### Why PostgreSQL, not the JSONL files used elsewhere

`training_logger.py` and `token_usage_logger.py` use simple JSONL files
— fine for append-only logs nobody needs to update. Submissions are
different: they need to be **updated** (pending → approved), **queried**
(a teacher's pending queue, a CRM's "approved since X" poll), and kept
**isolated per tenant** (school A must never see school B's data) — all
of which need a real relational database at this scale, not flat files.

### Concepts

- **Organization** = one tenant (one school / training center / CRM
  account). Has its own API key and, optionally, a webhook URL.
- **Submission** = one homework file, moving through:
  `pending_ai` → `pending_review` → `approved` (or `grading_failed`).
- **Teacher** / **Student** = referenced by *your* CRM's own IDs
  (`teacher_external_id`, `student_external_id`) — this app doesn't need
  its own user accounts for them, it just remembers the IDs you give it.

### One-time setup

1. **Install and run PostgreSQL:**
   ```bash
   # macOS: brew install postgresql@16 && brew services start postgresql@16
   # Linux: sudo apt install postgresql && sudo systemctl start postgresql
   # Windows: use Docker (docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16) or WSL2 + the Linux steps
   createdb homework_checker
   ```

2. **Point the app at it and create the tables:**
   ```bash
   export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/homework_checker"
   python3 -c "from db import init_db; init_db()"
   ```
   For schema *changes* later (adding a column, etc.), use a proper
   migration tool like [Alembic](https://alembic.sqlalchemy.org/) instead
   of relying on `init_db()` again — it only creates missing tables, it
   doesn't alter existing ones.

3. **Register your CRM as an organization** (admin operation, gated by
   `X-Internal-Api-Key`, not an organization key — the organization
   doesn't exist yet):
   ```bash
   curl -X POST http://localhost:8000/organizations \
     -H "X-Internal-Api-Key: your-internal-key" \
     -H "Content-Type: application/json" \
     -d '{"name": "Aniq Bilim Maktabi", "webhook_url": "https://your-crm.com/webhooks/homework"}'
   # -> {"organization_id": "...", "api_key": "hwck_...", "webhook_url": "...", "webhook_secret": "..."}
   ```
   `webhook_url` must be `https` and resolve to a public address (loopback, private,
   link-local and cloud-metadata addresses are refused with HTTP 422, and re-checked before
   every delivery). `webhook_secret` is used to verify webhook signatures (see below); it can be
   read again with `GET /organizations/me/webhook-secret`. For development against a local
   receiver set `WEBHOOK_ALLOW_PRIVATE_URLS=true`; to allow one named internal host set
   `WEBHOOK_ALLOWED_HOSTS=crm.internal`.
   **The `api_key` is shown once.** Store it in your CRM immediately —
   only its hash is kept server-side, so it can't be recovered later
   (register a new organization instead if it's lost). Every
   `/submissions*` call below authenticates with this key in an
   `X-Organization-Api-Key` header.

4. **Start a worker that also listens on the webhook queue — with the scheduler:**
   ```bash
   rq worker homework_grading webhook_delivery --with-scheduler
   ```
   (This is the same worker process type as the `/jobs` queue — just
   also listening on `webhook_delivery` now, so approved results get
   pushed out too.) **`--with-scheduler` is required for webhook retries:** RQ parks a failed
   delivery in its "scheduled" set and only a worker running the scheduler moves it back to the
   queue. Without the flag the first attempt is made but a failed delivery is never retried
   (verified by an audit probe in the original repository). Run the scheduler in one worker
   only.

5. **Run the reconciler** so a submission can never stay stuck. If a worker is killed mid-grading
   (OOM, deploy), Redis is flushed, or the queue was down when a submission arrived, the row would
   otherwise stay `pending_ai` forever. `reconcile.py` re-queues such submissions (those still
   `pending_ai` after `RECONCILE_REQUEUE_AFTER_SECONDS`, default 300, whose job is no longer
   queued or running) up to `RECONCILE_MAX_REQUEUES` times (default 3), then marks them
   `grading_failed` with a reason:
   ```bash
   python reconcile.py --loop 60            # a long-running process, one pass per minute
   # or from cron / a scheduler:
   curl -X POST -H "X-Internal-Api-Key: $INTERNAL_API_KEY" http://localhost:8000/admin/reconcile
   ```
   Grading jobs have a deterministic id (`grade-<submission_id>`), so a client retry or the
   reconciler never creates a second job next to a live one (no double spend).

**Shortcut for steps 4–5 and the API key.** `python setup_env.py` creates `.env` with a freshly
generated `INTERNAL_API_KEY` and `WEBHOOK_SIGNING_SECRET` (only empty values are filled, existing ones
are never changed, the secrets are not printed; add your provider key such as `GEMINI_API_KEY` to the
file yourself). The `Procfile` then starts the API, the worker (with the scheduler) and the reconciler
together: `pip install honcho && honcho start` (or Foreman, or a Heroku-style platform). If you run
several workers, start `--with-scheduler` on only one of them. This was smoke-tested against a real
PostgreSQL and Redis: the API answered 401 without the key and 200 with it, the worker picked a
submission off the queue, and the reconciler ran. Without provider keys the submission ended as
`grading_failed`, as it should.

### The workflow

**1. CRM submits homework** — grading happens in the background:

```bash
curl -X POST http://localhost:8000/submissions \
  -H "X-Organization-Api-Key: hwck_..." \
  -F "file=@homework.pdf" \
  -F "subject_key=math" \
  -F "teacher_external_id=teacher-101" -F "teacher_name=Nodira opa" \
  -F "student_external_id=student-55" -F "student_name=Sardor" \
  -F "external_submission_id=crm-hw-2026-001"
# -> {"id": "...", "status": "pending_ai", ...}
```

`external_submission_id` is optional but recommended: resubmitting the
same id returns the existing submission instead of creating a duplicate
— safe to retry on your CRM's side without double-grading.

**2. Your teacher-facing UI shows the review queue:**

```bash
curl "http://localhost:8000/submissions?status=pending_review&teacher_external_id=teacher-101" \
  -H "X-Organization-Api-Key: hwck_..."
```

Each item includes the full AI result (`ai_result`) — scores, mistakes,
suggestions, everything `/analyze` would have returned — for the teacher
to read before deciding.

**3. Teacher approves as-is, or edits and approves:**

```bash
# Approve exactly as the AI graded it:
curl -X POST http://localhost:8000/submissions/{id}/review \
  -H "X-Organization-Api-Key: hwck_..." -H "Content-Type: application/json" \
  -d '{"action": "approve"}'

# Or edit first (same JSON schema as ai_result), then approve:
curl -X POST http://localhost:8000/submissions/{id}/review \
  -H "X-Organization-Api-Key: hwck_..." -H "Content-Type: application/json" \
  -d '{"action": "edit", "corrected_result": {...}, "reviewer_note": "Partial credit for correct method."}'
```

`corrected_result` is validated: `overall_score` (a number 0–100) and a `questions` list are
required, each question's `score` must be 0–100, and text/list fields must have the right types —
otherwise HTTP 422 and nothing is stored. Only a `pending_review` submission can be reviewed (any
other state answers HTTP 409), and an approved result is final.

Either way, `status` becomes `"approved"` — that's the signal it's safe
to show in the student's gradebook. The AI's original result
(`ai_result`) is always preserved even after an edit, so you can see
what the AI actually got right or wrong.

**4. Result reaches your CRM — webhook first, polling as a safety net:**

If the organization has a `webhook_url`, we `POST` the approved result
there automatically (retrying with backoff — 10s, 30s, 2m, 10m, 30m —
on transient failures like your endpoint being briefly down):

```json
{
  "submission_id": "...",
  "external_submission_id": "crm-hw-2026-001",
  "status": "approved",
  "result": { "overall_score": 85, "grade": "Very Good", "questions": [...], "summary": {...} },
  "final_score": 85,
  "ai_score": 78,
  "edited_by_teacher": true,
  "reviewer_note": "Partial credit for correct method.",
  "reviewed_at": "2026-08-24T10:58:50Z",
  "approved_at": "2026-08-24T10:59:24Z"
}
```

For a gradebook, write `final_score` (the score the teacher approved: their edit when they
changed the result, otherwise the AI's). `ai_score` is what the AI originally gave, kept for audit
even after an edit; `edited_by_teacher` says whether the teacher changed it. Nothing else in the
body changed, so existing receivers keep working.

**Who sees what.** This service only ever *pushes* a result once it is `approved`. A result in
`pending_review` is visible through `GET /submissions` to your CRM (with the organization key), so
show those only to the teacher and release a result to the student only after the `approved`
webhook (or `status=approved` polling). The "one submission per student per homework" rule belongs
in your CRM: sending the same `external_submission_id` again returns the existing submission
instead of creating (and paying for) a second grading, so use your CRM's own id for the
student + homework as that value.

Every delivery is signed. The request carries `X-Webhook-Timestamp` and
`X-Webhook-Signature: sha256=<hex>`, an HMAC-SHA256 over `"<timestamp>.<raw body>"` with your
organization's `webhook_secret` (derived from `WEBHOOK_SIGNING_SECRET`, or `INTERNAL_API_KEY` if that
is unset; if neither is configured the webhook is sent unsigned). Verify it before trusting the call:

```python
import hmac, time
expected = "sha256=" + hmac.new(secret.encode(), f"{ts}.".encode() + raw_body, "sha256").hexdigest()
ok = hmac.compare_digest(expected, signature_header) and abs(time.time() - int(ts)) < 300
```

Redirects are never followed (a 3xx counts as a failed delivery).

**A failed AI grading** (all providers down, job timeout) ends as `grading_failed` with the reason in
`reviewer_note`. After the cause is fixed, `POST /submissions/{id}/regrade` grades it again; it also
works on a `pending_review` submission whose AI result a teacher does not trust, and it bypasses the
result cache so the models are really called again. Approved submissions cannot be regraded.

**Flags for the teacher.** `ai_result` carries `needs_attention` and `attention_reasons`
(`overall_score_differs_from_question_scores`, `classification_conflicts_with_score`,
`low_confidence`, `second_opinion_changed_classification`, `second_opinion_disagreed`), and
`verification` (the first-pass verdict when a second opinion changed it). They only *flag* a result for a
closer look; the model's own score is never silently rewritten.

Whether or not the webhook succeeds, this also remains available so
nothing is ever silently lost:

```bash
curl "http://localhost:8000/submissions?status=approved&since=2026-08-24T00:00:00Z" \
  -H "X-Organization-Api-Key: hwck_..."
```

### Fine-tuning data collection (OFF by default)

Student work is children's data, so nothing is copied into `training_data/` unless you turn
collection on: `export COLLECT_TRAINING_DATA=true`. Only then is every approved submission —
especially ones a teacher **edited** — logged to `training_data/gold.jsonl` (and the raw model output
to `raw_log.jsonl`) as a verified example. That is very good fine-tuning data, but enable it only
where you have the consent of the students/parents/school and a retention and deletion process for
`training_data/` — the application has none. `GET /backend` reports whether collection is on.

### Multi-tenancy and security

- Every `/submissions*` endpoint requires `X-Organization-Api-Key` and
  every database query is scoped to that organization — one tenant's
  data is structurally invisible to another's, not just hidden by the UI.
- A cross-tenant access attempt returns `404 Not Found` (not `403
  Forbidden`) — this avoids confirming to a caller that a submission ID
  exists at all if it doesn't belong to them.
- `X-Internal-Api-Key` (separate from organization keys) gates admin-only
  operations like creating a new organization.
- File storage note: uploaded homework currently lives on local disk
  (`uploads/`). This is fine for a single server; at real 100k-user
  multi-tenant scale with multiple API/worker instances, move this to
  shared object storage (S3-compatible) so every instance can read every
  file regardless of which one originally received the upload.

## Phase 4: multi-model tiered grading (hwchecker/)

A second, more elaborate grading engine lives in `hwchecker/` — a
difficulty-tiered, multi-provider pipeline with $0 deterministic
pre-checks and automatic secondary verification, matching a specific
architecture spec (`hwchecker/` mirrors that spec's file layout exactly:
`providers/`, `subjects/`, `checkers/`, `router/`, `pipeline/`, `infra/`,
plus `benchmarks/`). It runs **alongside**, not instead of, everything
above — the CRM review workflow (`POST /submissions`) automatically uses
whichever engine covers a given subject; nothing else changes.

### Architecture in one picture

```
Homework
   │
   ▼
Deterministic checkers ($0) ─── resolves it? ──▶ done (no AI call)
   │  (MCQ, SymPy, Pint units, equation balance, fuzzy-match, LanguageTool)
   ▼ (didn't resolve)
Difficulty router (metadata first, then Qwen3.7 Flash looking at the file)
   │
   ├─ easy   ──▶ Qwen3.5 Flash                (thinking off)
   ├─ medium ──▶ Qwen3.7 Plus                 (thinking low; Gemini Flash-Lite = "medium_alt" stand-in)
   ├─ hard   ──▶ GLM-5.2, typed work          (reasoning high)
   │             Gemini 3.6 Flash, photo/scan ("hard_vision" tier, thinking high)
   └─ audio  ──▶ Gemini 3.6 Flash             ("audio" tier)
   │
   ▼
Low confidence or borderline score? ──▶ Claude Haiku 5.5 (second opinion, always from a
   │                                     different vendor; never for hard-photo verdicts)
   │
   ▼
Provider failed entirely? ──▶ other usable providers (GPT-5.4 mini first)
   │
   ▼
Structured JSON result (classification/score/error_type/confidence/...)
```

### Which subjects use it

`worker_tasks._resolve_hwchecker_subject()` routes a CRM submission here
when its subject/language is one `hwchecker/subjects/registry.py`
actually covers: **math, physics, chemistry, biology, history**, and
**English/Russian/Uzbek/Turkish** (Writing/Reading/Listening — not
Speaking, see below). Everything else — Essay, Computer Science,
General/custom subjects, French, German, and Speaking in any language —
still uses the original single-call `get_checker()` engine described
earlier in this README. Both write the same way into `submissions.ai_result`
(JSONB), so the CRM review workflow, webhook delivery, and training-data
logging all work identically regardless of which engine actually graded it.

**Speaking was deliberately left out of `hwchecker/`'s language
profiles** (`subjects/languages.py`) — see that file's "SPEAKING
EXTENSION POINT" comment block for exactly what adding it later
involves. This mirrors the original spec's own MVP scope decision, not
a limitation of this project's Speaking support in general (Gemini-based
Speaking grading, via `get_checker()`, works fine today).

### Provider API keys

Beyond `GEMINI_API_KEY` (already required elsewhere in this project),
the new tiers need:

```bash
export QWEN_API_KEY="..."       # DashScope (Alibaba Cloud) -- easy tier + difficulty router
export GLM_API_KEY="..."         # Zhipu AI / Z.ai -- hard tier
export ANTHROPIC_API_KEY="..."    # verification tier (Haiku); optional -- Gemini/Qwen check each other without it
export OPENAI_API_KEY="..."        # fallback tier
```

Only set the keys for tiers you actually plan to use. A tier whose key is
missing is **skipped**: the submission goes to the next usable provider
(medium, then the Gemini stand-in `medium_alt`), so a deployment with only
`GEMINI_API_KEY` grades everything through Gemini. `GradingRouterError` is raised only when *no*
provider is usable, and its message lists what is missing. All variables
are listed in `.env.example`.

### Cost-optimised model set-up

The defaults implement five savings levers. Estimated saving of each one alone at
150 000 submissions a month (the pricing workbook's "Asosiy" scenario, ≈ $1 520/month
before any lever; the figures overlap, so they do not add up): smaller photos ≈ $108,
re-check share 25 % → 15 % ≈ $281, Haiku verifier ≈ $668, Flash-Lite medium tier ≈ $144.
The thinking limits were not priced separately. These are estimates from code/prompt size
and list prices, not measurements — real token use can differ by 30–50 %.

| # | Lever | Default |
|---|-------|---------|
| 1 | Smaller photos | `MAX_IMAGE_SIDE_PX=1568` |
| 2 | Limit hidden "thinking" tokens (they are billed as output) | router/easy off, medium low, hard `high`, audio low |
| 3 | Fewer second opinions | confidence < **0.70** (was 0.75) or score within **2** points (was 3) of 40/70/90 |
| 4 | Cheaper verifier | Claude Haiku 5.5 instead of Sonnet |
| 5 | Cheaper medium tier | Qwen3.7 Plus (Gemini 3.5 Flash-Lite before; Gemini 3.5/3.6 Flash before that) |
| 6 | Cheaper hard-photo tier | Gemini 3.6 Flash, thinking high (Claude Sonnet 5 before) |

Levers 5 and 6 were switched after the owner's own benchmark run found Qwen3.7 Plus and
Gemini 3.6 Flash grading well. Rough effect at 150 000 submissions a month (estimates from
list prices, not measurements): the medium tier ≈ $169 → ≈ $110, hard photos ≈ $194 →
≈ $83 (≈ $166 from 2027-01-01, when Gemini 3.6 Flash doubles to $1.50/$7.50), so the whole
set-up ≈ $575–670 → ≈ $405–500 a month today. **Check Qwen3.7 Plus's price on the Alibaba
Cloud pricing page**: `providers/qwen.py` flags it as not re-checked (the list price carries
a promo), and the medium-tier saving depends on it. To go back to the previous set-up for a
tier, use env only, e.g. `HWCHECKER_HARD_VISION_PROVIDER=claude`,
`HWCHECKER_HARD_VISION_MODEL=claude-sonnet-5`, `HWCHECKER_HARD_VISION_THINKING=default`, or
`HWCHECKER_MEDIUM_PROVIDER=gemini` + `HWCHECKER_MEDIUM_MODEL=gemini-3.5-flash-lite`.

Not included on purpose: provider batch APIs (50 % cheaper, but results arrive
hours later — wrong for a student waiting for feedback).

Everything is set per tier in `providers/registry.py` (`TIER_DEFAULTS`) and can be
changed without touching code — see `.env.example`:
`HWCHECKER_<TIER>_MODEL`, `HWCHECKER_<TIER>_PROVIDER`, `HWCHECKER_<TIER>_THINKING`,
`HWCHECKER_VERIFY_CONFIDENCE_THRESHOLD`, `HWCHECKER_VERIFY_SCORE_MARGIN`.
`providers.registry.describe_tiers()` returns what each tier is currently set to.

Things worth knowing:

- **Thinking knobs are sent per provider API**: Gemini `thinking_level`, Qwen
  `enable_thinking`/`thinking_budget`, GLM `thinking`/`reasoning_effort`, Claude
  `output_config.effort`. If an endpoint rejects the field the provider drops it
  and retries, so grading keeps working (just without that saving) — watch the
  logs if costs do not fall. Gemini's thinking tokens are now counted as output in
  the cost log (they were missing before).
- **The re-check rate is a guess until measured.** Count the share of results
  carrying a `verification` block over a week of real traffic; if it is much
  above 15 %, lower the confidence threshold or the margin further, or set
  `HWCHECKER_VERIFICATION_THINKING=low`. If quality drops, put the old value back via
  the env variable — no deploy needed.
- **The result cache knows the model set-up.** The cache key includes every tier's
  provider/model/thinking level and the re-check thresholds, so after you change a
  tier an old answer is never served for it (old entries just expire). Comparing two
  set-ups on the same files therefore needs no cache clearing.
- **Haiku also re-checks GLM's hard-tier (typed) verdicts**, and a second opinion
  from a cheaper model can overrule a stronger one there. Set
  `HWCHECKER_VERIFICATION_MODEL=claude-sonnet-5` if you want the stronger verifier
  for hard work again.
- **Re-run the benchmark with the exact production settings**: candidates in
  `python -m hwchecker.benchmarks.compare_candidates` accept `provider:model@thinking`
  (e.g. `gemini:gemini-3.5-flash-lite@low`); without `@…` the provider default is used,
  i.e. how the tier behaved before this set-up.
- The legacy single-call engine (`config.SUBJECT_MODEL_MAP` …, Speaking, other
  subjects) is a separate code path and is not changed by any of this.

### Second opinions, output cap and spend alerts (Qwen + Gemini set-up)

**Second opinions always come from a different vendor than the grader**, because a model
re-checking a sibling from its own vendor tends to repeat its mistakes. The router tries
`verification` (Claude Haiku 5.5), then `verification_gemini` (Gemini 3.5 Flash-Lite, low
thinking), then `verification_qwen` (Qwen3.7 Plus, low thinking) and takes the first one
that is configured, can read the input and is not from the grader's vendor. So:

| Keys you have | Qwen's grade is checked by | Gemini's grade is checked by |
|---|---|---|
| Anthropic + any | Haiku | Haiku |
| Qwen + Gemini only | Gemini Flash-Lite | Qwen3.7 Plus (only for Gemini graders on a verifiable tier; by default Gemini grades only hard photos and audio, which are not re-checked) |
| Only one vendor | nobody (result stays un-verified and is not cached) | nobody |

Recordings are only re-checked by a vendor that can hear them, so audio graded by Gemini
is not re-checked unless another audio-capable vendor is added. Cost note: Haiku
($0.10/$0.50 per 1M tokens) is the cheapest independent verifier — a Gemini Flash-Lite
check of a Qwen grade costs roughly 3–5× more per call (input 3×, output 5×). Keeping an Anthropic key just for
Haiku is usually cheaper than checking Qwen's work with Gemini; the Qwen/Gemini pair is the
fallback for deployments without one. Re-check each tier by env as usual, e.g.
`HWCHECKER_VERIFICATION_GEMINI_MODEL`, `HWCHECKER_VERIFICATION_QWEN_THINKING`.

**Output cap (optional).** `HWCHECKER_MAX_OUTPUT_TOKENS=4096` lowers every provider's
max reply size (default 8192 for Gemini/Claude, 8192/16384 for the OpenAI-protocol ones). It
is a guard against runaway replies, not a saving on normal ones, and is never set below
1024: Gemini counts its hidden thinking tokens against this limit and a JSON reply cut
off half-way would be retried, i.e. paid for twice.

**Spend alerts (never blocks grading).** `HWCHECKER_DAILY_BUDGET_USD` and
`HWCHECKER_MONTHLY_BUDGET_USD` (off by default) add up the estimated cost of every call
per UTC day/month — in Redis when reachable, else per process — and log a warning (plus a
Sentry message if installed) at 80 % and 100 % of the budget, once per period and level.
Nothing is throttled or downgraded when a budget is reached: set it to the level you
would call abnormal (e.g. 150 % of the cost you expect) to catch a retry loop, spam
uploads or a tier accidentally pointed at an expensive model. Costs are list-price
estimates (`estimate_cost`), not your invoice.

### How a submission is routed

- **By what the model can read.** Photos and scanned PDFs are only sent to
  vision-capable providers; GLM (text-only) is skipped for them, so a *hard*
  photo goes to the `hard_vision` tier (Gemini 3.6 Flash), then Qwen3.7 Plus,
  Gemini Flash-Lite, Haiku, OpenAI. Audio goes to the `audio` tier (Gemini 3.6 Flash —
  Flash-Lite is too weak for speech), then to the other Gemini tier. Typed/DOCX work can still
  use GLM.
- **By difficulty.** The CRM sends no question text, so the Qwen router
  looks at the student's file itself. If it cannot (no key, unreadable
  reply) the submission is treated as *medium* — grading is never blocked
  by the classifier.
- **On failure.** A failed call is retried (3 attempts, 2 s / 4 s backoff);
  a provider that keeps failing is skipped by the circuit breaker; then up
  to two other usable providers are tried (OpenAI, Gemini, Claude in that
  order, skipping the one that already failed). The second opinion comes
  from a **different vendor** than the grader (see "Second opinions" below) and is
  skipped when no such verifier is configured or can read the input, or when the result
  already came from `hard_vision` (Gemini 3.6 Flash) — a cheaper verifier must not
  overrule the strongest model — and an un-verified low-confidence result is
  not cached.
- **Repeat submissions** of the identical file/subject/prompts are served
  from a cache (`HWCHECKER_ENABLE_CACHE=false` to turn off). The key includes the
  model set-up, so changing a tier's model/thinking level starts a fresh cache.
- **Monitoring.** Every call logs provider, tokens, estimated cost and latency
  (`hwchecker.monitoring` logger; Prometheus/Sentry if installed).
  `HWCHECKER_RATE_LIMIT_PER_MINUTE` can pace calls per provider (off by default).

Redis is used for the circuit-breaker/cache/rate-limit state when
reachable and falls back to per-process memory otherwise.

### What the AI answers (language and structure)

- **Language.** Feedback is written in **Uzbek (Latin script) by default**.
  Set `FEEDBACK_LANGUAGE` to change it (`Russian`, `English`, … or `auto` to
  answer in the language of the student's work). This applies to both
  engines. JSON keys, classification values and error-category names stay
  English so the CRM and database see stable values; the student's own words and
  formulas are quoted as written, and for language homework the corrections
  stay in the studied language.
- **Per question.** The model returns a score, mistakes, analysis and tips
  for every question it finds in the file, plus a strengths/weaknesses
  summary. `worker_tasks.adapt_router_result()` maps that into the
  `questions[]` / `summary` shape the report generator and CRM already read
  (`grade` is the usual "Excellent / Good / …" band label). If the model gives
  no breakdown, the overall verdict is stored as a single question.
- **Blurry or cropped photos.** The prompt tells the model not to guess
  at illegible parts and to lower its confidence, which sends the answer to
  the second-opinion step.

### Photo uploads

Uploaded photos are normalised in `utils.save_uploaded_file()`: HEIC/HEIF
(iPhone default) is converted to JPEG, the EXIF rotation is applied (a
portrait page used to arrive sideways), and anything over `MAX_IMAGE_SIDE_PX`
(1568 by default; raise it to 2400 if tiny handwriting is misread) on its
longest side is downscaled. Images that need none of this are
stored untouched. Needs `Pillow` and `pillow-heif` (in `requirements.txt`);
an unreadable image answers HTTP 400 with the reason. One file per
submission: several photos of one homework must be sent as one PDF.

### Tests

```bash
python -m unittest discover -s tests -t . -v      # or: pytest tests
```

They need no API keys, network or Redis: the OpenAI/Anthropic/Google SDKs are
replaced by in-process fakes, and the real router, providers, prompts and
upload code run against them. They prove the plumbing (photo routing, retries,
fallback, verification, parameter handling, language, per-question output,
HEIC/rotation); they say nothing about how well a real model grades — that is
what the benchmark below is for.

`tests/test_crm_integration.py` runs the CRM workflow against a **real PostgreSQL and Redis/RQ**
(only the LLM SDKs stay faked). It is skipped unless you name throw-away services — **their contents
are wiped**, so the database name must contain `test` or `audit`:

```bash
export TEST_DATABASE_URL="postgresql+psycopg2://postgres@localhost:5432/homework_checker_test"
export TEST_REDIS_URL="redis://localhost:6379/15"
python -m pytest tests
```

### Running the mandatory benchmark before trusting a tier assignment

`providers/registry.py`'s tier→model assignments are meant to be
validated against real data before you rely on them in production —
this is exactly what `benchmarks/` (60 hand-verified math test cases,
`claude_math_benchmark_dataset_v1.md`) is for:

```bash
python -m hwchecker.benchmarks.runner /path/to/claude_math_benchmark_dataset_v1.md medium
```

This grades all 60 cases with the "medium" tier's provider and prints:

```
Benchmark results (60 cases):
  Classification Accuracy:    ...
  Correct Answer Accuracy:    ...
  Partial Credit Accuracy:    ...
  Error Detection Accuracy:   ...
  False Positive Rate:        ...   <- the most important number: how often
                                        a WRONG answer gets called CORRECT
  False Negative Rate:        ...
  Score Agreement (MAE):      ...
```

Run it against each tier ("easy", "medium", "hard", "hard_vision", "audio",
"verification", "fallback") before changing `providers/registry.py`'s assignments —
change based on what the numbers actually show, not assumption. A quick
smoke test on a handful of cases first: add a limit argument, e.g.
`run_benchmark(path, provider_tier="hard", limit=10)` from a Python shell.

### Extending it

- **New subject**: add a `SubjectProfile` (see `subjects/base.py`) to a
  new or existing file under `subjects/`, then register it in
  `subjects/registry.py`'s `SUBJECT_REGISTRY` — one line. Also add it to
  `worker_tasks._HWCHECKER_SUBJECTS` (or `_HWCHECKER_LANGUAGES`) so the
  CRM workflow actually routes to it.
- **New provider**: add a class implementing `providers/base.py`'s
  `AIProvider` interface, then point a tier at it in
  `providers/registry.py` — that's the only place a tier-to-provider
  assignment should ever be changed.
- **Self-hosted models**: `providers/self_hosted.py` is a ready-to-use
  extension point, disabled by default
  (`HWCHECKER_ENABLE_SELF_HOSTED=false`). This project's existing
  `local_checker.py` (see "Phase 2" above) is a fully working reference
  for the vLLM/Ollama deployment side of this, if/when you're ready to
  wire it in here too.

## How it works

- **Feedback text uses light markdown** — `analysis`, `mistakes`,
  `suggestions`, and `correct_answer` come back from Gemini with
  `**bold**`, `- bullet` points, and `` `code` `` where useful (the JSON
  structure itself is unchanged — this only affects the text inside those
  fields). `report_generator.py` renders this properly in both the HTML
  report (real `<strong>`/`<ul>` via the `markdown` package) and the PDF
  (a small built-in renderer, since fpdf2 doesn't consume HTML/CSS).
  Streamlit displays it with `st.markdown()` so it renders there too.
- **Token usage is tracked automatically** for every graded submission —
  input, output, and cached token counts, logged to
  `training_data/token_usage_log.jsonl` (`token_usage_logger.py`).
  View it in the Streamlit sidebar ("🔢 Token usage") or via `GET /usage`
  on the API (totals plus a breakdown by subject/skill and by model).
  Disable with `TRACK_TOKEN_USAGE=false` if you don't need it.
- **Sidebar → Assessment type**: choose **Standard subject** (Math, Physics,
  Chemistry, Biology, Computer Science, Essay/Literature, History, or a
  custom "General / Other" subject you type in) or **Language skill**
  (English / Russian / Turkish × Writing / Reading / Listening / Speaking).
  This swaps in a tailored grading system prompt — different roles,
  objectives, error categories, and score rubrics — while every mode still
  returns the same JSON schema, so reports work identically either way.
- **PDF / image / DOCX uploads** cover Math-type subjects and the Writing/
  Reading/Listening language skills (typed or handwritten work, completed
  worksheets, essays, etc).
- **Audio uploads** (.mp3 / .wav / .m4a / .ogg) are for the **Speaking**
  skill — upload a recording of the student talking, and Gemini listens to
  it directly to assess pronunciation, fluency, grammar, and vocabulary.
- **PDF / image uploads** are sent directly to Gemini as multimodal input —
  the model reads typeset or handwritten math (or text) straight off the
  page, no OCR pipeline needed.
- **DOCX uploads** are text-extracted locally with `python-docx` first
  (Gemini has no native Word-doc reader), then sent as text.
- Gemini is given a strict system prompt (built dynamically in
  `prompts.py`) instructing it to behave like an expert examiner for the
  chosen subject/skill, grade every detected question/task 0–100,
  categorize mistakes, and return **only** JSON in a fixed schema.
- `utils.parse_and_validate_json` strips any accidental markdown fences
  and checks the required keys are present; `homework_checker.py`
  automatically asks the model to self-correct once or twice if the
  JSON is malformed.
- `report_generator.py` turns that JSON into:
  - a styled **HTML** report (`templates/report.html`, via Jinja2)
  - a printable **PDF** report (via `fpdf2` with a bundled Unicode font,
    so math symbols like √, ∫, π render correctly — no system
    dependencies like wkhtmltopdf/weasyprint required)
  - a combined **Excel gradebook** (via `openpyxl`) across every
    submission graded in the current session, with an overview sheet
    and a per-question detail sheet.

## Customizing the grading rubric

- Standard subjects live in `prompts.py` as `SubjectProfile` entries
  (`MATH`, `PHYSICS`, `CHEMISTRY`, `BIOLOGY`, `COMPUTER_SCIENCE`, `ESSAY`,
  `HISTORY`, `GENERAL`). Edit objectives/error_categories/score_rubric on
  any of them, or add a new one and register it in the `SUBJECTS` dict.
- Languages live in `LANGUAGES` (add a 4th language by adding one line).
  Skills (Writing/Reading/Listening/Speaking) live in `SKILL_TEMPLATES` —
  edit objectives/error_categories/score_rubric there to change how any
  skill is graded across all languages at once.
- Keep the `OUTPUT FORMAT` JSON schema (`_OUTPUT_FORMAT_BLOCK`) unchanged
  unless you also update `utils.parse_and_validate_json` and
  `report_generator.py` to match, since they depend on those exact keys
  (`overall_score`, `grade`, `questions[]`, `summary`).

### Choosing which Gemini model grades which subject

Not every subject needs the same model. `config.SUBJECT_MODEL_MAP` and
`config.LANGUAGE_SKILL_MODEL_MAP` set a default model per subject/skill —
used automatically by `get_checker()` in every code path (`/analyze`,
`/jobs`, `/jobs/batch`, `/submissions`) whenever the caller doesn't pass
an explicit `model_name`:

```python
SUBJECT_MODEL_MAP = {
    "math": "gemini-3.5-flash",
    "computer_science": "gemini-3.1-pro-preview",  # logic/correctness reasoning
    "biology": "gemini-3.1-flash-lite",             # mostly fact/terminology checking
    ...
}
```

The idea: subjects that mostly involve checking facts or terminology
against a completed answer (biology, history) don't need the strongest
(most expensive) model; subjects that require judging whether a
multi-step *approach* was valid, not just the final answer (computer
science, advanced math/physics), benefit more from a stronger model even
at higher cost per request. **These starting values are reasonable
defaults, not settled science** — once you've collected enough
teacher-reviewed data per subject (`training_data/gold.jsonl`, fed
by the CRM review workflow above when collection is enabled), compare how often the
AI and teacher actually agreed across different models for that subject,
and adjust the mapping based on that evidence.

An explicit `model_name` (e.g. from the Streamlit sidebar's model
dropdown, or an explicit API parameter) always overrides this mapping.

## Notes / next steps

- `uploads/` and `reports/` are working directories — wire them up to
  cloud storage (S3, GCS) if you deploy this across multiple server
  instances (see the CRM review workflow's multi-tenancy section above
  for why this matters at scale).
- For very long, multi-page assignments, consider splitting into one
  Gemini call per page/question to stay well under output token limits.
- For production Postgres schema changes over time, add
  [Alembic](https://alembic.sqlalchemy.org/) migrations rather than
  relying on `db.init_db()`, which only creates missing tables.

## Phase 2: running your own model instead of Gemini

This project supports a full switch away from Gemini to a self-hosted,
open-weight model — for teams that want zero dependency on any external
AI company. This is a bigger undertaking than everything above, so read
this section fully before starting.

### What actually changes, and what doesn't

Only **one setting** decides which backend grades homework:

```bash
export MODEL_BACKEND=local   # or "gemini" (default)
```

`app.py`, `api.py`, `report_generator.py`, and the JSON schema never
change. `homework_checker.get_checker()` is the single place that reads
`MODEL_BACKEND` and hands back either the Gemini-backed `HomeworkChecker`
or the self-hosted `LocalModelChecker` (`local_checker.py`) — both expose
the exact same `.analyze(file_path, extra_instructions)` interface.

### Architecture

| Task | Local model |
|---|---|
| Text + vision (Math, Physics, Essay, Reading, Writing, etc.) | **Qwen2.5-VL** (32B or 72B) — reads PDFs/images/handwriting |
| Audio (Speaking) | **Qwen2-Audio** or **Qwen2.5-Omni** — listens to the recording directly |
| Serving | **vLLM**, exposing an OpenAI-compatible REST API on your GPU server |

One real difference from Gemini: open vision-language models take
**images**, not raw PDF bytes. `local_checker.py` converts PDFs to page
images automatically with PyMuPDF before sending them — this is
invisible to the rest of the app, but it's worth knowing about.

### Three different paths depending on your hardware and OS

- **NVIDIA GPU server, native Linux (A100/H100/etc.)** → use **vLLM**
  (Step 1 below). This is the production path: highest throughput,
  supports serving multiple models, LoRA adapters, many concurrent users.
- **NVIDIA GPU on Windows** → vLLM doesn't run on native Windows either
  (it needs a Linux CUDA environment). Install **WSL2** with Ubuntu
  (`wsl --install` in an admin PowerShell, then set up NVIDIA's
  [CUDA-on-WSL drivers](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)),
  and follow the Linux/vLLM instructions (Step 1 below) *inside* WSL2 —
  it gets near-native GPU performance this way. Alternatively, Ollama
  (below) has a native Windows installer and is far simpler to set up,
  at the cost of vLLM's production throughput/scaling features.
- **Apple Silicon Mac (M1/M2/M3, unified memory)** → vLLM does not run
  here at all (no NVIDIA CUDA on macOS, ever). Use **Ollama** instead —
  a genuinely different tool, not a smaller version of vLLM.

Ollama (whether on macOS, Windows, or Linux) is the right choice for
learning/prototyping on a single machine, not for serving many students
at once — see the honest limits note below.

#### Quick start with Ollama — for learning/prototyping only

**1. Install Ollama:**
```bash
# macOS:
brew install ollama

# Windows: download and run the installer from https://ollama.com/download/windows
# (installs as a background service; no separate "ollama serve" step needed)

# Linux:
curl -fsSL https://ollama.com/install.sh | sh
```

**2. Start it** (macOS/Linux only — Windows runs it as a service automatically):
```bash
ollama serve
```

**3. In another terminal, pull a vision-language model** (same command, every OS):
```bash
# Start small to confirm everything works:
ollama pull qwen2.5vl:7b
# Once that works, try a stronger one if your RAM/VRAM comfortably allows it:
ollama pull qwen2.5vl:32b
```

**4. Quick sanity check outside the app:**
```bash
ollama run qwen2.5vl:7b "Say hello"
```

**5. Point the app at it:**
```bash
# Windows (PowerShell):
$env:MODEL_BACKEND="local"
$env:LOCAL_MODEL_BASE_URL="http://localhost:11434/v1"
$env:LOCAL_MODEL_API_KEY="ollama"
$env:LOCAL_TEXT_MODEL="qwen2.5vl:7b"

# macOS / Linux:
export MODEL_BACKEND=local
export LOCAL_MODEL_BASE_URL="http://localhost:11434/v1"
export LOCAL_MODEL_API_KEY="ollama"          # Ollama ignores the value, but the field can't be empty
export LOCAL_TEXT_MODEL="qwen2.5vl:7b"       # or qwen2.5vl:32b once confirmed working
```
```bash
pip install openai pymupdf
streamlit run app.py
```

Speaking will keep using Gemini automatically (see `FORCE_GEMINI_FOR_SPEAKING`
in config.py) — mature, easy-to-run local audio models aren't really
available for consumer hardware yet, so this avoids a dead end there.

**Honest limits of this path**: a laptop or desktop is fine for trying
things out and building intuition, but it is not a real "server" — it
has far less raw compute than an A100/H100, needs to stay awake and on
the same network as whoever's using the app, and can only handle one or
a few requests at a time before things slow down noticeably. Treat this
as how you'd learn and prototype the workflow, not as infrastructure for
real students at scale. When you're ready for that, the vLLM path below
(on a rented or owned NVIDIA GPU server, or WSL2 on a Windows GPU
machine) is the one to use — everything else about the app (config,
`local_checker.py`, the rest of this guide) stays identical, you're only
swapping which server `LOCAL_MODEL_BASE_URL` points to.

### Step 1 — Serve a model with vLLM on your GPU server

On your A100/H100 server:

```bash
pip install vllm

# Vision-language model, for everything except Speaking:
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768

# Separately, for Speaking (different port, can run alongside):
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
  --port 8001 \
  --tensor-parallel-size 1
```

Adjust `--tensor-parallel-size` to however many GPUs you're splitting
the model across, and `--max-model-len` to your context needs (homework
pages + system prompt is usually a few thousand tokens; leave headroom).
72B-class models need more GPUs/VRAM than 32B — start with 32B, confirm
quality, then scale up if needed.


### Step 2 — Point the app at your server

This is run on whichever machine hosts the Streamlit app / API (your
own laptop or desktop, not the GPU server itself), since that's where
`streamlit run app.py` / `uvicorn api:app` actually execute:

```bash
# Windows (PowerShell):
$env:MODEL_BACKEND="local"
$env:LOCAL_MODEL_BASE_URL="http://YOUR_SERVER_IP:8000/v1"
$env:LOCAL_TEXT_MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
$env:LOCAL_AUDIO_MODEL="Qwen/Qwen2-Audio-7B-Instruct"

# macOS / Linux:
export MODEL_BACKEND=local
export LOCAL_MODEL_BASE_URL="http://YOUR_SERVER_IP:8000/v1"
export LOCAL_TEXT_MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
export LOCAL_AUDIO_MODEL="Qwen/Qwen2-Audio-7B-Instruct"
```

Run `streamlit run app.py` or `uvicorn api:app` as usual — the sidebar
will show "Model backend: your own server" and the Gemini API key field
disappears, since it's not needed anymore.

**Try the base (non-fine-tuned) model first.** Qwen2.5-VL is already a
strong general-purpose model — it may grade acceptably out of the box.
Fine-tuning (below) is for closing the remaining gap, not a prerequisite
to using your own model at all.

### Step 3 — Collect training data (opt-in)

`training_logger.py` logs every graded submission to
`training_data/raw_log.jsonl` (plus a copy of the input file in
`training_data/inputs/`) **only when you set `COLLECT_TRAINING_DATA=true`**
(it is off by default because it copies students' work to local disk —
enable it only with consent and a retention/deletion process). This
happens regardless of which backend is active — once enabled you can
start collecting data even while still using Gemini, and use that
data to fine-tune your local model later.

**Raw data isn't safe to train on as-is** — it may contain the model's
own mistakes (which is exactly the "too lenient" problem you ran into
earlier). Review a sample and promote only verified-correct examples to
the gold dataset:

```python
from training_logger import promote_to_gold

# Approve as-is:
promote_to_gold("the-example-id-from-raw_log.jsonl")

# Or supply a teacher-corrected version of the JSON:
promote_to_gold("the-example-id", corrected_output={...corrected JSON...})
```

(A simple review UI — a Streamlit page listing raw examples with an
edit box and an "approve" button — is a natural next addition once
you're ready to start reviewing at scale; ask if you'd like it built.)

Aim for at least a few hundred **gold** examples per subject/skill
before fine-tuning — more (thousands) is better. Quality matters more
than volume: 500 carefully verified examples beat 5,000 unreviewed ones.

### Step 4 — Fine-tune with LoRA once you have gold data

With 256GB VRAM across A100/H100s, LoRA/QLoRA fine-tuning of Qwen2.5-VL
(32B or even 72B) is very feasible — full fine-tuning is unnecessary and
slower for this kind of task-adaptation. Two well-maintained tools:

- **[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)** — YAML-config-driven, supports Qwen2-VL, good defaults.
- **[Unsloth](https://github.com/unslothai/unsloth)** — faster/lighter LoRA training, actively adds new Qwen support.

Rough shape of the workflow (exact commands depend on the tool version
at the time — check its current docs):

1. Convert `training_data/gold.jsonl` into the tool's expected
   conversation format (system prompt + user message with image/audio +
   assistant message = the gold JSON).
2. Write a LoRA config: rank 16-64, targeting attention + MLP projection
   layers, 1-3 epochs over your gold set (multimodal fine-tunes overfit
   fast with small datasets — watch a held-out validation split).
3. Train, then merge the LoRA adapter into the base weights (or serve
   the adapter directly — vLLM supports LoRA adapters at serving time
   via `--enable-lora`).
4. Point `LOCAL_TEXT_MODEL` / `LOCAL_AUDIO_MODEL` at the new
   checkpoint name and restart vLLM. Nothing else changes.

### Honest expectations

- Steps 1-2 (serving a base open model) can be working within a day or
  two on your hardware.
- Step 3 (collecting enough good gold data) is the real bottleneck —
  this typically takes weeks to months of real usage, not a one-time task.
- Step 4 (fine-tuning) itself is fast once you have data (hours, given
  your GPUs) — most of the effort is in steps 2-3, not the training run.
- A fine-tuned 32B open model can get very close to Gemini's grading
  quality for well-represented subjects/skills with enough gold data,
  but matching Gemini's raw general reasoning across every possible
  edge case is unlikely — the honest goal is "good enough and fully
  independent," not "identical to Gemini."
