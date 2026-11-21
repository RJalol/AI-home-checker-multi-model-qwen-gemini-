# Qwen + Gemini asosidagi AI HW Checker — reja (yangilangan 2026-11-20, benchmark + bitta uy ishi ko'p o'quvchi)

Asos: 2026-11-19/20 suhbati va eski repo (`RJalol/AI-home-checker-multi-model`, PR #2, #3, #4; ular endi shu repoga birlashtirilgan). Hammasi taxmin, hali o'lchanmagan.

## 0. Qaror (2026-11-20)
- Daromadga bog'lash hozir ko'rib chiqilmaydi. Maqsad: texnik jihatdan arzon va sifatli tizim.
- Dinamik kvota (byudjet ÷ o'quvchilar) **olib tashlandi**: u o'quvchi ko'paysa sifatni tushiradi.
- Test (yopiq javobli topshiriqlar) tekshiruvi HW checker'da emas, alohida funksiya.
- **Benchmark natijasi (egasining o'zi o'tkazgan):** Qwen3.7 Plus (o'rta daraja) va Gemini 3.6 Flash (qiyin foto) yaxshi ishladi, shuning uchun ikkalasi default qilindi. Qaysi thinking darajasi bilan sinalgani noma'lum; defaultlar: Qwen3.7 Plus low, Gemini 3.6 Flash high.
- Grok va DeepSeek hozircha qo'shilmaydi (Grok arzon modellari yo'q; DeepSeek vision tajribaviy, tejam ~5–10%).
- Xarajat o'quvchi soniga chiziqli o'sadi; uni sifatni tushirmasdan bir tekshiruv narxini kamaytirish bilan boshqaramiz.
- Byudjet to'xtatgichi faqat himoya (ogohlantirish) (4-bo'lim).
- Ish oqimi: o'quvchi bir marta yuboradi → natija o'qituvchiga → o'qituvchi tasdiqlaydi/tahrirlaydi → o'quvchiga va baholash jadvaliga (CRM) tushadi. "Bir marta" qoidasi va baholash jadvali CRM'da; HW checker faqat `final_score` yuboradi.
- Bitta uy ishini ko'p o'quvchi yuborsa: assignment (uy ishi) darajasida umumiy topshiriq/kalit va prompt keshi (9-bo'lim).

## 1. Maqsad
- Provayderlar: Qwen (router, oson, o'rta), Gemini (qiyin foto, audio, zaxira o'rta), GLM (qiyin matn), Claude Haiku (faqat tekshiruv), GPT-5.4 mini (zaxira).
- Jev 1.13 hozircha qo'shilmadi (8-bo'lim).

## 2. Pipeline (default)

| Bosqich | Model | Izoh |
|---|---|---|
| Router | qwen3.7-flash, off | faylni (foto) o'zi ko'radi |
| Oson | qwen3.5-flash, off | |
| O'rta | **qwen3.7-plus, low** | benchmarkdan keyin; Qwen kaliti bo'lmasa `medium_alt` |
| O'rta (zaxira) | gemini-3.5-flash-lite, low (`medium_alt`) | Qwen kaliti yo'q yoki xato bo'lganda |
| Qiyin matn | glm-5.2, high (faqat matn) | |
| Qiyin foto | **gemini-3.6-flash, high** | benchmarkdan keyin; qayta tekshirilmaydi |
| Audio | gemini-3.6-flash, low | |
| Tekshiruv | claude-haiku-5-5; keyin gemini-3.5-flash-lite → qwen3.7-plus | har doim baholovchidan boshqa provayder |
| Zaxira | gpt-5.4-mini | |

Eski holatga qaytish faqat env bilan (kodsiz): `HWCHECKER_HARD_VISION_PROVIDER=claude` + `_MODEL=claude-sonnet-5` + `_THINKING=default`; `HWCHECKER_MEDIUM_PROVIDER=gemini` + `_MODEL=gemini-3.5-flash-lite`.

## 3. Sifatni tushirmaydigan tejam usullari
1. **Chiqish tokeni:** qat'iy JSON, qisqa fikr-mulohaza (prompt o'zgarishi — benchmark bilan). `HWCHECKER_MAX_OUTPUT_TOKENS` ixtiyoriy himoya.
2. **To'g'ri yo'naltirish:** fan/topshiriq turi bo'yicha benchmark isbotlagan eng arzon yetarli model.
3. **Prompt/kontekst keshlash:** umumiy topshiriq+kalit prefiksi; haqiqiy tejam `tokens.cached` bilan o'lchanadi.
4. **Audio:** jimlikni kesish (VAD), 60–90 s limit.
5. **Foto:** `MAX_IMAGE_SIDE_PX=1568` (allaqachon).
6. **Thinking:** easy off, medium low, qiyinda high.
7. **Tekshiruv chegarasi:** ishonch < 0.70 yoki ball chegaradan ≤ 2 (allaqachon).
8. **Kesh:** model almashganda eski natija berilmaydi.
9. Har o'zgarish: "o'zgartirish → benchmark → qabul".

## 3a. Sifat himoyasi
- Benchmark to'plamida (8 fan × 5 o'quvchi ishi) o'rtacha ball og'ishi va katta xatolar soni oshmasa — qabul.
- Noto'g'ri yo'naltirish ulushi (qiyin → oson) alohida o'lchanadi.
- Ishlab chiqarishda ~5% topshiriq soya rejimda kuchli model bilan qayta baholanadi (drift nazorati).

## 4. Himoya chegarasi (sifatni tushirmaydi)
- `HWCHECKER_DAILY_BUDGET_USD` / `HWCHECKER_MONTHLY_BUDGET_USD`: 80% va 100% da ogohlantirish (log + Sentry), hech narsani to'xtatmaydi. Byudjetni "g'ayritabiiy" deb hisoblagan darajaga qo'ying (masalan kutilgan xarajatning 150%).
- Tashkilot bo'yicha kunlik/daqiqalik topshiriq chegarasi bor (`ORG_MAX_SUBMISSIONS_PER_*`); `HWCHECKER_RATE_LIMIT_PER_MINUTE` provayder darajasida.
- Provayder konsollarida alohida byudjet va ogohlantirishlar.

## 5. Taxminiy oylik xarajat (150 000 topshiriq, "Asosiy"; ro'yxat narxi, o'lchanmagan)

| Qism | Oldingi | Yangi |
|---|---|---|
| Qwen oson | ~59 | ~59 |
| O'rta | ~169 (Gemini Flash-Lite) | ~110 (Qwen3.7 Plus) |
| Audio (eski engine) | ~78 | ~78 |
| Qiyin matn (GLM) | ~44 | ~44 |
| Qiyin foto | ~194 (Sonnet) | ~83 (Gemini 3.6 Flash); 2027-01-01 dan ~166 |
| Tekshiruv (Haiku) | ~25 | ~25 |
| Router | ~7 | ~7 |
| **Jami** | **$575–670** | **~$405–500** (2027-01-01 dan ~$560–660) |

Eslatma: `qwen3.7-plus` narxi repoda "not re-checked (list price carries a promo)" deb belgilangan; promo tugasa o'rta darajadagi tejam kamayishi mumkin. Tekshiruvni faqat Qwen + Gemini bilan qilsak (Haiku'siz) ~$100 atrofida (+~$65–75), shuning uchun Haiku kaliti qoldirildi. Umumiy prefiks keshi tejami bu jadvalga kiritilmagan.

## 6. Xavflar
- Router xatosi (qiyin → oson) eng xavfli; benchmarkda o'lchanadi.
- Gemini 3.6 Flash 2027-01-01 dan ikki baravar qimmatlashadi (qiyin foto ~$83 → ~$166).
- Gemini ishlamasa qiyin foto va audio to'xtaydi; zaxira (gpt-5.4-mini, Qwen3.7 Plus) bor, audio uchun yo'q.
- Audio Gemini bahosi boshqa provayderdan tekshirilmaydi (Qwen/Claude eshita olmaydi).
- Qiyin foto natijasi (Gemini) qayta tekshirilmaydi: Haiku kuchsizroq bo'lgani uchun uni bekor qilishi mumkin.
- Benchmark kichik to'plamda; real trafikda qayta tekshiruv ulushi va token sarfi o'lchanishi kerak.
- Prompt keshi provayderlar tomonidan kafolatlanmagan; tejam `tokens.cached` bilan tasdiqlanmaguncha taxmin.

## 7. Keyingi qadamlar
1. CRM'dan `external_assignment_id` (+ ixtiyoriy `task_text`, `answer_key`) yuborishni yoqish.
2. Qwen3.7 Plus narxini Alibaba Cloud sahifasidan tekshirish.
3. Real trafikda xarajat jurnalidan haqiqiy token sarfi, `tokens.cached` ulushi va qayta tekshiruv ulushini o'lchash, so'ng narx jadvalini yangilash.
4. Gemini Flex (50%) va Qwen Batch (50%) — real `cached` raqamlaridan keyin qaror (9-bo'lim).
5. Jev: faqat matnli topshiriqlar yoki alohida qayta tekshiruv darvozasi sifatida, benchmark va maxfiylik shartlaridan keyin.
6. DeepSeek: faqat matnli topshiriqlar uchun nomzod sifatida benchmark qilish (ixtiyoriy).

## 8. Migratsiya (2026-11-20)
**O'zgargan:** o'rta daraja Qwen3.7 Plus (Gemini Flash-Lite `medium_alt` sifatida qoldi), qiyin foto Gemini 3.6 Flash, boshqa provayderdan tekshiruv tanlovi (Haiku → Gemini → Qwen), kesh kaliti model sozlamasini hisobga oladi, `HWCHECKER_MAX_OUTPUT_TOKENS`, byudjet ogohlantirishi (`infra/budget.py`), benchmark nomzodlari yangilandi.

**Jev nega qo'shilmadi:** CRM yuklamalarida savol matni yo'q, router fotoni o'zi ko'radi; Jev faqat matn qabul qiladi va hamkorlar uni tajribaviy deb belgilagan.

**Tekshirilmagan:** testlar soxta SDK'lar bilan (API kalitlari yo'q); real model nomlari, token sarfi va qayta tekshiruv ulushi o'lchanmagan.

## 9. Bitta uy ishi, ko'p o'quvchi (2026-11-20)
**Nega tejaladi:** natija avval o'qituvchiga boradi, o'quvchi kutmaydi; bir uy ishining topshiriq matni va javoblar kaliti barcha o'quvchilarda bir xil.

**Amalga oshirildi:** `assignments` jadvali va `submissions.assignment_id`; `POST /submissions` `external_assignment_id`, `task_text`, `answer_key` (har biri ≤ 20 000 belgi) qabul qiladi, bir marta saqlanadi, kalit API javobida qaytmaydi; prompt'da topshiriq/kalit o'quvchi faylidan OLDIN, hamma o'quvchida bir xil (kesh prefiksi); `tokens.cached` (Gemini `cached_content_token_count`, OpenAI-protokol `cached_tokens`); `init_db()` mavjud bazaga ustun qo'shadi.

**Provayder shartlari (rasmiy hujjatlardan, 2026-11-20):**
- Gemini: implicit kesh Gemini 2.5+ da avtomatik, tejam kafolatlanmagan; Batch va Flex 50% arzon. Flex hujjatda faqat Interactions API'da (`service_tier='flex'`), bizning kod `generate_content` ishlatadi, shuning uchun qo'shilmadi. Flex kutish 10 daqiqa+ bo'lishi mumkin, sig'im to'lsa o'zi Standard'ga o'tmaydi.
- Qwen: kesh yaratish 125%, o'qish 10%; Batch 50%, lekin batch va kesh birga qo'llanmaydi, batch ichida prefiks keshi ishlamaydi.

**Tejam taxmini (o'lchanmagan):** umumiy prefiks katta bo'lsa ayrim bosqichlar narxi ~30–50% tushishi mumkin; aniq raqam `tokens.cached` dan keyin.
