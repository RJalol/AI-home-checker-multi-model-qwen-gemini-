# Privacy and retention (technical measures; the legal side needs a lawyer)

## What is stored and where

| Data | Where | Kept |
|---|---|---|
| Student's original file (often a photo of a minor's handwritten work) | `uploads/` on the server | `UPLOAD_RETENTION_DAYS` (default 90) after approval; orphans after 1 day |
| AI result, teacher edit, student/teacher names and CRM ids | PostgreSQL | until erased (`DELETE /submissions/{id}`) |
| Audit trail (who approved/edited, scores) | `review_events` | kept, holds no student data (ids and scores only) |
| Fine-tuning copy of approved work | `training_data/` | only if `COLLECT_TRAINING_DATA=true` (off by default); NOT covered by the retention job |
| Provider error text | logs only | per your log retention |

## What leaves your server

The student's file, the task text and answer key, and the subject rubric are sent to the AI providers that
grade it: Alibaba Cloud (Qwen), Z.ai (GLM), Google (Gemini), Anthropic (Claude, second opinion) and OpenAI
(fallback). Student NAMES are not put into the prompt, but a student can write their name in the work itself.

## Before real students

These cannot be done in code:

1. Decide the legal basis and get the schools' (and, for minors, parents') consent. Check the applicable
   personal-data law (for Uzbekistan: the law on personal data) with a lawyer, including rules on sending data abroad.
2. For each provider, check its terms: data NOT used to train models, retention of API inputs, region, a
   data-processing agreement. Use the business/API terms, not consumer ones. Note that Alibaba and Z.ai are
   China-based vendors; decide whether that is acceptable for your schools, or route those tiers elsewhere
   (`HWCHECKER_<TIER>_PROVIDER`).
3. Write a privacy notice for schools and parents: what is sent, to whom, for how long.
4. Agree the retention period with the schools and set `UPLOAD_RETENTION_DAYS`.
5. Decide who may ask for deletion and how (`DELETE /submissions/{id}`); keep the request log.
6. Encrypt the disk/volume and the database backups; restrict who can read `uploads/`.
7. Keep `COLLECT_TRAINING_DATA` off unless consent explicitly covers model training.
