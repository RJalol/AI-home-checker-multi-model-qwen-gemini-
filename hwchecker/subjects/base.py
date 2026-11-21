"""
hwchecker/subjects/base.py
The subject-agnostic data model every subject profile is built from,
exactly per spec section 7. Adding a new subject means adding ONE new
SubjectProfile instance to subjects/registry.py's SUBJECT_REGISTRY --
nothing in router/, providers/, or pipeline/ should ever need to change.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from hwchecker.safety import UNTRUSTED_INPUT_BLOCK

# Shared by every subject's system_prompt_template unless a subject needs
# something genuinely different (languages.py customizes this further
# per skill). Encodes the classification taxonomy and JSON schema from
# spec section 8, plus a strict scoring-discipline policy so grade
# inflation doesn't happen just because the feedback tone is encouraging.
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a strict, fair grading assistant for {subject_id}.

Grading paradigm: {grading_paradigm}

Rubric criteria:
{criteria_block}

Error taxonomy (use one of these exact values for "error_type" when applicable, or null if none apply):
{error_taxonomy}

# SCORING DISCIPLINE
- Grade inflation is a serious failure. List every real error before deciding a score.
- A response with any real, substantive error cannot score 90 or above.
- A confident or fluent-sounding response is not automatically a good one -- judge only
  demonstrated correctness against the rubric above.
- When genuinely uncertain between two classifications, choose the stricter one and lower
  "confidence" accordingly rather than picking the more generous option to seem certain.
- "confidence" reflects how sure YOU are of your own classification/score (used to decide
  whether this result gets a second opinion) -- it is not about the student's certainty.

# OUTPUT FORMAT
Return ONLY this JSON object, nothing else -- no markdown fences, no commentary:
{{
  "classification": "CORRECT | INCORRECT | PARTIALLY_CORRECT | INCORRECT_WITH_MISCONCEPTION",
  "score": 0-100,
  "error_type": "string from the error taxonomy above, or null",
  "error_location": "string describing precisely where the error occurs, or null",
  "feedback": "educational, encouraging-in-TONE explanation of what's right/wrong and how to improve -- tone never softens the score",
  "sub_scores": {{}},
  "confidence": 0.0-1.0
}}
"""


# Appended to every subject's prompt AFTER the template has been .format()-ed (so the braces
# below need no escaping). Real homework photos hold several questions; a single overall verdict
# is not enough for a teacher or a student, so the model also returns a per-question breakdown.
# The keys match what the CRM / report generator already read from the older engine's output.
PER_QUESTION_BLOCK = """# PER-QUESTION BREAKDOWN
The submission may contain several questions or tasks. Detect every one of them and grade each separately.
In addition to the fields above, the JSON object must also contain these two keys:
  "questions": [
    {
      "question_number": 1,
      "question": "short restatement of the question / task",
      "student_answer": "what the student wrote, short",
      "correct_answer": "the correct answer or model solution, short",
      "score": 0-100,
      "mistakes": ["each precise mistake, empty list if none"],
      "error_categories": ["values from the error taxonomy, empty list if none"],
      "analysis": "what is right / wrong in this question and how to fix it",
      "suggestions": ["one concrete tip"]
    }
  ],
  "summary": {"strengths": ["..."], "weaknesses": ["..."], "topics_to_review": ["..."], "recommendations": ["..."]}
Rules:
- The top-level "score" is the overall score for the whole submission (weigh the questions by how much they matter) and the top-level "feedback" is a short overall comment. If the file holds just one question, "questions" has exactly one item.
- Never invent a question that is not in the file.
- The file may be a phone photo. If part of it is blurry, cropped, upside down or illegible, say so in that question's "analysis", do not guess what it says, and lower the top-level "confidence" (0.3 or less when most of it cannot be read).
"""


@dataclass
class RubricCriterion:
    name: str
    max_points: float
    description: str


@dataclass
class SkillProfile:
    """Only populated for language subjects (see subjects/languages.py)."""

    name: str                     # "reading" | "writing" | "listening" | "speaking"
    input_modality: str           # "text" | "audio"
    requires_asr: bool = False
    pronunciation_api_available: bool = False
    deterministic_checker: Optional[Callable] = None


@dataclass
class SubjectProfile:
    subject_id: str
    grading_paradigm: str          # "objective" | "rubric" | "mixed"
    topic_weights: dict            # dict[str, float] -- used by router/difficulty.py
    rubric_criteria: list          # list[RubricCriterion]
    error_taxonomy: list           # list[str]
    deterministic_checker: Optional[Callable] = None
    system_prompt_template: str = ""
    skills: Optional[dict] = None  # dict[str, SkillProfile] -- only for language subjects

    def build_rubric(self, homework=None) -> dict:
        """
        Assembles the rubric dict passed to AIProvider.grade_homework().
        `homework` is accepted (and can be used by subclasses of this
        method / more elaborate profiles) for rubrics that depend on the
        specific assignment, but the default implementation just exposes
        the static profile data -- most subjects don't need per-homework
        rubric variation beyond what topic_weights/rubric_criteria
        already capture.
        """
        return {
            "subject_id": self.subject_id,
            "grading_paradigm": self.grading_paradigm,
            "criteria": [
                {"name": c.name, "max_points": c.max_points, "description": c.description}
                for c in self.rubric_criteria
            ],
            "error_taxonomy": list(self.error_taxonomy),
        }

    def build_system_prompt(self, skill: Optional[str] = None, *, per_question: bool = True, language: Optional[str] = None) -> str:
        """
        Renders system_prompt_template with this profile's data baked
        in. For language subjects, pass `skill` ("reading"/"writing"/...)
        to pull in that skill's SkillProfile-specific framing too.

        On top of the template it appends the per-question breakdown rules
        (`per_question=False` leaves them out, e.g. for benchmark runs that
        only need the verdict) and the "which language to answer in" block
        (`language` overrides the FEEDBACK_LANGUAGE setting, see language.py).
        """
        criteria_block = "\n".join(f"- {c.name} ({c.max_points} pts): {c.description}" for c in self.rubric_criteria)
        error_block = ", ".join(self.error_taxonomy)

        skill_note = ""
        if skill and self.skills and skill in self.skills:
            sp = self.skills[skill]
            skill_note = (
                f"\n\nThis is a {skill.upper()} task (input modality: {sp.input_modality})."
                + (" Audio input requires ASR/transcription before grading." if sp.requires_asr else "")
            )

        prompt = self.system_prompt_template.format(
            subject_id=self.subject_id,
            grading_paradigm=self.grading_paradigm,
            criteria_block=criteria_block,
            error_taxonomy=error_block,
        ) + skill_note

        from hwchecker.language import language_instruction

        if per_question:
            prompt += "\n\n" + PER_QUESTION_BLOCK
        return prompt + "\n\n" + UNTRUSTED_INPUT_BLOCK + "\n\n" + language_instruction(language)
