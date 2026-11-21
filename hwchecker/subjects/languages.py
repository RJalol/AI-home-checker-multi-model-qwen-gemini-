"""
hwchecker/subjects/languages.py
Language SubjectProfiles: English, Russian, Uzbek, Turkish -- each with
Writing/Reading/Listening SkillProfiles (full text-pipeline support).

Speaking is deliberately NOT implemented here, per spec section 1:
"Speaking (gapirish) MVP'ga kirmaydi ... Speaking uchun kod yozmang,
faqat kengaytirish nuqtasini (extension point) qoldiring." See
SPEAKING_EXTENSION_POINT below for where/how to add it later -- adding
it should not require changing anything outside this function call.
"""

from .base import RubricCriterion, SkillProfile, SubjectProfile

_LANGUAGE_SYSTEM_PROMPT_TEMPLATE = """You are a strict, fair {subject_id} language examiner.

Grading paradigm: {grading_paradigm}

Rubric criteria:
{criteria_block}

Error taxonomy (use one of these exact values for "error_type" when applicable, or null if none apply):
{error_taxonomy}

# SCORING DISCIPLINE
- Grade inflation is a serious failure. List every real grammar/vocabulary/content error before deciding a score.
- A response with any real, substantive error cannot score 90 or above.
- Judge demonstrated language accuracy and task achievement, not confidence or length.
- When genuinely uncertain, choose the stricter classification and lower "confidence" accordingly.

# OUTPUT FORMAT
Return ONLY this JSON object, nothing else -- no markdown fences, no commentary:
{{
  "classification": "CORRECT | INCORRECT | PARTIALLY_CORRECT | INCORRECT_WITH_MISCONCEPTION",
  "score": 0-100,
  "error_type": "string from the error taxonomy above, or null",
  "error_location": "string describing precisely where the error occurs, or null",
  "feedback": "educational, encouraging-in-TONE explanation -- tone never softens the score",
  "sub_scores": {{"content": 0-100, "grammar": 0-100, "vocabulary": 0-100, "coherence": 0-100}},
  "confidence": 0.0-1.0
}}
"""

# ---------------------------------------------------------------------------
# SPEAKING EXTENSION POINT (not implemented in MVP)
# ---------------------------------------------------------------------------
# To add Speaking later, for a given language:
#   1. Add a SkillProfile("speaking", input_modality="audio", requires_asr=True,
#      pronunciation_api_available=<bool>) to that language's `skills` dict below.
#   2. Route it to an audio-capable provider (Gemini natively understands audio;
#      see this project's existing local_checker.py / cloud_checker.py for the
#      ASR/transcription pattern other providers need instead).
#   3. Nothing else in router/, providers/, or pipeline/ needs to change --
#      SkillProfile.input_modality == "audio" is already a first-class concept
#      in the data model, this MVP just never populates it.
# No Speaking code is written here on purpose, per spec section 1/11.
# ---------------------------------------------------------------------------


def _make_language_profile(language_id: str, *, languagetool_supported: bool) -> SubjectProfile:
    reading_writing_criteria = [
        RubricCriterion("content_task_achievement", 30, "The response addresses the task/prompt fully"),
        RubricCriterion("grammar_accuracy", 30, "Grammar, verb tenses, and sentence structure are correct"),
        RubricCriterion("vocabulary", 20, "Vocabulary range and word choice are appropriate for the task"),
        RubricCriterion("coherence", 20, "Ideas are organized clearly with appropriate cohesion"),
    ]

    error_taxonomy = [
        "Grammar Error", "Vocabulary Error", "Spelling Error", "Punctuation Error",
        "Coherence Issue", "Task Achievement Issue", "Comprehension Error", "Missing Detail",
    ]

    # LanguageTool has first-class support for English/Russian and partial
    # support for Turkish; not meaningfully supported for Uzbek (spec
    # section 3 notes this explicitly) -- languagetool_supported controls
    # whether checkers/languagetool_check.py is wired in for this language,
    # done by subjects/registry.py to avoid a circular import.
    skills = {
        "writing": SkillProfile(
            name="writing", input_modality="text", requires_asr=False,
            pronunciation_api_available=False, deterministic_checker=None,
        ),
        "reading": SkillProfile(
            name="reading", input_modality="text", requires_asr=False,
            pronunciation_api_available=False, deterministic_checker=None,
        ),
        "listening": SkillProfile(
            name="listening", input_modality="text", requires_asr=False,  # the completed answer sheet is text; the source audio was heard separately
            pronunciation_api_available=False, deterministic_checker=None,
        ),
        # "speaking" intentionally omitted -- see SPEAKING EXTENSION POINT above.
    }

    return SubjectProfile(
        subject_id=language_id,
        grading_paradigm="mixed",
        topic_weights={"writing": 0.4, "reading": 0.35, "listening": 0.25},
        rubric_criteria=reading_writing_criteria,
        error_taxonomy=error_taxonomy,
        system_prompt_template=_LANGUAGE_SYSTEM_PROMPT_TEMPLATE,
        deterministic_checker=None,
        skills=skills,
    )


ENGLISH_PROFILE = _make_language_profile("english", languagetool_supported=True)
RUSSIAN_PROFILE = _make_language_profile("russian", languagetool_supported=True)
TURKISH_PROFILE = _make_language_profile("turkish", languagetool_supported=True)   # partial LanguageTool support
UZBEK_PROFILE = _make_language_profile("uzbek", languagetool_supported=False)      # no LanguageTool support -- LLM-only grammar checking
