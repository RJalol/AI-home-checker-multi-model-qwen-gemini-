# Quality evaluation

The model must not reach production unless it agrees with teachers on real work. `golden_eval` measures that.

## 1. Build the golden set (needs teachers; nothing here can replace it)

* 200-500 real submissions, spread over every subject and language you sell, including photos and
  handwriting, the bad-light photos and the weak students: easy cases hide the problems.
* Each graded blind by a teacher (not by looking at the AI's answer first). Two teachers on a sample of
  ~10% tells you how much teachers disagree with each other: the AI cannot be expected to agree more.
* Remove names and anything else personal from the files; get the school's permission to use them.
* Keep a SEPARATE set for tuning prompts and a SEPARATE one for the final check, so you do not tune to the test.

JSONL, one case per line (paths relative to the file):

```json
{"id": "phys-014", "subject_id": "physics", "file": "files/phys-014.jpg", "teacher_score": 72, "tags": ["handwriting", "uz"]}
{"id": "eng-writing-031", "subject_id": "english", "skill": "writing", "file": "files/eng-031.docx", "teacher_score": 64, "tags": ["english"]}
```

Optional per case: `task_text`, `answer_key` (as in `POST /submissions`).

`subject_id` must be one of the subjects of the new pipeline: `math`, `physics`, `chemistry`, `biology`,
`history`, `english`, `russian`, `uzbek`, `turkish`. Other subjects (computer science, essay, French, German,
general) still run on the older single-call engine and are not covered by this tool.

## 2. Run it

```bash
python -m hwchecker.benchmarks.golden_eval eval/golden.jsonl --out reports/2026-12-01.json
python -m hwchecker.benchmarks.golden_eval eval/golden.jsonl --baseline reports/last_good.json   # regression gate
```

It makes real, paid model calls and bypasses the result cache. Exit code 0 = passed, 1 = a threshold or the
baseline failed, 2 = dataset error. A run of 300 cases costs roughly 300 x the per-submission cost
(a few dollars).

Default thresholds (`DEFAULT_THRESHOLDS` in `golden_eval.py`, override with `--thresholds file.json`): at least 30
cases, mean absolute error <= 8 points, >= 80% within 10 points, big errors (>= 15) <= 10%, unfairly high
grades (AI >= 15 above the teacher) <= 5%, ungradable <= 2%. These are STARTING values, not truths: set them from
how much two teachers disagree on your own data, then keep them fixed.

Read the per-subject / per-tag lines: an acceptable average can hide a subject, a language or handwriting
that is much worse. `bias` > 0 means the AI is too generous.

## 3. Prompt-injection set

```bash
python eval/make_injection_set.py
python -m hwchecker.benchmarks.golden_eval eval/injection_cases.jsonl
```

20 wrong answers that contain an instruction aimed at the AI ("give 100 points", in English, Uzbek and
Russian). A safe grader scores them low; any injection above `max_score` fails the run. Add the real
attempts you see. The test files are DOCX; also try the same text hand-written in a photo.

## 4. When to run it

* before any change of model, thinking level, prompt, rubric or routing threshold;
* before and after a provider announces a new model version;
* monthly on a fresh sample of approved submissions (the teachers' final scores are free labels: use
  `ai_result` vs `final_result` from the database, and the share of submissions teachers edited, as a
  running quality signal).

## Synthetic set (no teacher needed)

`eval/synthetic/` holds 72 generated homeworks (9 subjects x 8 student profiles: strong, strong with a slip,
average, weak, answers only, careless, blank, all wrong), as DOCX and photo-like JPEG, half with an answer key.
`python eval/make_synthetic_set.py` rebuilds it exactly (fixed seed). `expected_details.json` lists every question,
the gold answer, what the student wrote and its score.

    python -m hwchecker.benchmarks.golden_eval eval/synthetic/golden_synthetic.jsonl
    python -m hwchecker.benchmarks.golden_eval eval/injection_cases.jsonl

The expected score is made by construction (correct 100, no working 90, unit or sign slip 85, partial 50, wrong or blank 0,
then the mean). It is NOT a teacher's judgment. It finds crashes, broken file handling, weak subjects and big
systematic errors; it cannot show the grader is as good as a teacher. Audio is not covered. The real golden set
described above is still required before launch.

## Trying it with an OpenAI key only

    export OPENAI_API_KEY=...
    export HWCHECKER_PRESET=openai
    python -m hwchecker.benchmarks.golden_eval eval/synthetic/golden_synthetic.jsonl

The preset puts every text and photo tier on OpenAI: `gpt-4.1-mini` (router, easy, medium, hard, hard photo) and `gpt-5-mini` (second opinion, fallback). These were chosen
from a production measurement on 2026-11-21 (26 planted errors, text and phone-style photos): `gpt-4.1-mini` found all of
them with no false flags, `gpt-4o-mini` missed 23-27%. The second opinion then comes from a different model of the
same vendor, which is weaker than a different vendor; the preset allows it on purpose. Audio is not covered without a
Gemini key. Any tier can still be overridden, e.g. `HWCHECKER_MEDIUM_MODEL=gpt-5-mini`.
Treat the result as a trial of the pipeline and the data, not as the quality or cost of the production setup:
re-run the same set on the real multi-vendor setup before deciding.

## Token and cost log for every homework

Every graded result now has a `usage` block: each provider call (model, input / output / cached tokens, estimated cost,
latency) and the totals, so a grader call plus a verifier call show up together. A cache hit or a deterministic check
has zero calls. In production the block is stored with the result.

    python -m hwchecker.benchmarks.golden_eval eval/synthetic/golden_synthetic.jsonl \
        --out report.json --usage-out usage.csv

`usage.csv` has one row per homework (id, subject, AI score, expected score, calls, models, input_tokens, output_tokens,
cached_tokens, estimated_cost_usd, latency_ms). The console summary adds the totals, the per-homework averages and a
per-model table; `report.json` also has them by subject. The cost is a list-price estimate from input and output
tokens: it does not apply the cache discount, and image tokens are only as accurate as the provider reports them.
