# Audit topilmalari holati (2026-11-21), PR #1: production hardening

PR: https://github.com/RJalol/AI-home-checker-multi-model-qwen-gemini-/pull/1 (branch `claude/production-hardening`, 240 test o'tdi, PostgreSQL/Redis bilan).
Asos hujjat: `audit_2026-11-21.md`.

| Topilma | Holat | Qanday |
|---|---|---|
| F1 uzun maydonlar 500 | Tuzatildi | `hardening.check_field_lengths`, 422; testlar bor |
| F2 `extra_instructions` cheksiz | Tuzatildi | `MAX_EXTRA_INSTRUCTIONS_CHARS` (4000) |
| F3 fayl xotiraga o'qiladi | Qisman | bo'laklab o'qish + 413, `MAX_IMAGE_PIXELS`; proxy chegarasi `docs/DEPLOYMENT.md`da (siz sozlaysiz); yuk ostida sinalmagan |
| F4 saqlash muddati / yetim fayllar | Tuzatildi | `retention.py`, `POST /admin/retention`, `DELETE /submissions/{id}`, qayta yuborishda fayl yozilmaydi |
| F5 webhook imzosi | Tuzatildi | `APP_ENV=production` da `WEBHOOK_SIGNING_SECRET` majburiy, admin kalitidan olinmaydi |
| F6 org kvotasi | Tuzatildi | `ORG_MAX_SUBMISSIONS_PER_MINUTE/_DAY` (429 + Retry-After); qiymatlarni real yuklamaga qarab sozlang |
| F7 CI/Docker/lock | Yozildi, sinalmagan | `Dockerfile`, `docker-compose.yml`, `requirements.lock`, `.github/workflows/ci.yml`; Docker build va CI hali ishga tushmagan |
| F8 xato matni | Tuzatildi | CRM'ga umumiy xabar, tafsilot logda |
| F9 regrade chegarasi | Tuzatildi | `MAX_REGRADES_PER_SUBMISSION` (3) |
| F10 legacy byudjet | Ochiq | legacy dvigatel hali byudjet kuzatuvchisidan tashqarida |
| F11 DNS rebinding | Ochiq | firewall bilan yopiladi (`docs/DEPLOYMENT.md`) |
| F12 umumiy kesh | Ochiq | biznes qarori |
| F13 default DB parol | Tuzatildi | production rejimida xato |
| I1 o'qituvchi shaxsi | CRM'da | `reviewer_external_id` audit jurnaliga yoziladi (`review_events`) |
| I2 yuridik | Ochiq (odam kerak) | `docs/PRIVACY_AND_RETENTION.md` |
| I3 real provayderlar | Ochiq | real kalitlar bilan `golden_eval` ishga tushiring |
| I4 qamrov | Qisman | yangi modullar 80–98%; umumiy 62% dan oshdi (aniq raqam o'lchanmagan) |
| I5 ikki dvigatel | Ochiq | legacy olib tashlanmadi |
| I6 prompt-injection | Vosita tayyor, natija yo'q | `eval/make_injection_set.py` (20 hujum), `golden_eval`; real modelda hali ishga tushmagan |

Qo'shimcha: `review_events` audit jurnali, versiyalangan migratsiyalar (`migrations.py`), `GET /metrics` + `deploy/alerts.yml`, `golden_eval` sifat darvozasi, `docs/` (DEPLOYMENT, RUNBOOK, EVALUATION, PRIVACY_AND_RETENTION, LAUNCH_CHECKLIST).

Sentetik to'plam (PR #2): `eval/synthetic/`, 9 fan, 72 uy ishi, tayyor sinov uchun (I3, I6 ga yordam beradi). Ballari qurilish bo'yicha kutilgan, o'qituvchi bahosi emas; audio qamrab olinmagan.

Kodga kirmaganlari (odam, hisob yoki real trafik kerak): o'qituvchilar bilan oltin to'plam, yuridik tekshiruv, yuk testi, zaxira tiklash sinovi, penetration test, soya rejimi va pilot, real provayder sinovi, legacy dvigatelni olib tashlash.
