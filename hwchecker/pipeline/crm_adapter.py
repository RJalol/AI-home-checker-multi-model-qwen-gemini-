"""
hwchecker/pipeline/crm_adapter.py
Maps the tiered router's output onto the shape the rest of the application already reads.

Kept apart from worker_tasks.py (which pulls in the database and the queue) so it can be
imported and tested on its own.
"""


# A headline score further than this many points from the mean of the per-question scores is
# flagged for the teacher (the model's holistic score is kept -- it may weigh questions unevenly).
SCORE_MISMATCH_TOLERANCE = 15.0
LOW_CONFIDENCE_THRESHOLD = 0.5
# Loose bounds: only blatant contradictions between the label and the number are flagged.
_CLASSIFICATION_SCORE_RANGE = {
    "CORRECT": (70.0, 100.0),
    "PARTIALLY_CORRECT": (15.0, 95.0),
    "INCORRECT": (0.0, 60.0),
    "INCORRECT_WITH_MISCONCEPTION": (0.0, 60.0),
}


def compute_attention_reasons(overall: float, classification: str, confidence: float,
                              question_scores: list, verification: dict) -> list:
    """Machine-readable reasons a teacher should look at this result first (empty = none)."""
    reasons = []
    numeric = [q for q in question_scores if isinstance(q, (int, float)) and not isinstance(q, bool)]
    if len(numeric) >= 2:
        mean = sum(numeric) / len(numeric)
        if abs(mean - overall) > SCORE_MISMATCH_TOLERANCE:
            reasons.append("overall_score_differs_from_question_scores")
    low, high = _CLASSIFICATION_SCORE_RANGE.get(classification, (0.0, 100.0))
    if not (low <= overall <= high):
        reasons.append("classification_conflicts_with_score")
    if isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if verification:
        if verification.get("classification_changed"):
            reasons.append("second_opinion_changed_classification")
        elif verification.get("outcome") == "verifier_overrode":
            reasons.append("second_opinion_disagreed")
    return reasons


def adapt_router_result(schema_result: dict) -> dict:
    """
    Converts the hwchecker router's output (classification/score/max_score/error_type/
    ... plus the optional per-question "questions" / "summary") into the shape db.py,
    the report generator and the CRM already consume (overall_score, grade, questions[],
    summary{}) -- ai_result is stored as JSONB either way, but the teacher UI and
    report_generator.py read these top-level names.

    * When the model returned a per-question breakdown, every question becomes its own
      entry. If it did not, the whole verdict is wrapped as a single question.
    * "grade" is the readable band label ("Excellent", "Good", ...) the older engine
      always produced -- it used to be set to the raw classification ("PARTIALLY_CORRECT").
      The classification itself is kept under "classification".
    """
    from config import grade_for_score

    overall = schema_result["score"]
    feedback = schema_result.get("feedback") or ""
    error_type = schema_result.get("error_type")
    error_location = schema_result.get("error_location")

    questions = []
    for index, q in enumerate(schema_result.get("questions") or [], start=1):
        questions.append({
            "question_number": q.get("question_number", index),
            "question": q.get("question", ""),
            "student_answer": q.get("student_answer", ""),
            "expected_solution": q.get("expected_solution") or q.get("correct_answer", ""),
            "analysis": q.get("analysis", ""),
            "mistakes": list(q.get("mistakes") or []),
            "error_categories": list(q.get("error_categories") or []),
            "correct_answer": q.get("correct_answer", ""),
            "suggestions": list(q.get("suggestions") or []),
            "difficulty": q.get("difficulty", ""),
            "score": q.get("score", overall),
        })

    if not questions:
        questions = [{
            "question_number": 1,
            "question": "",
            "student_answer": "",
            "expected_solution": "",
            "analysis": feedback,
            "mistakes": [error_location] if error_location else [],
            "error_categories": [error_type] if error_type else [],
            "correct_answer": "",
            "suggestions": [],
            "difficulty": "",
            "score": overall,
        }]

    summary = schema_result.get("summary") or {}
    adapted = {
        "overall_score": overall,
        "grade": grade_for_score(overall)[0],
        "classification": schema_result["classification"],
        "feedback": feedback,
        "questions": questions,
        "summary": {
            "strengths": list(summary.get("strengths") or []),
            "weaknesses": list(summary.get("weaknesses") or []),
            "topics_to_review": list(summary.get("topics_to_review") or []),
            "recommendations": list(summary.get("recommendations") or []),
        },
        "provider": schema_result["provider"],
        "confidence": schema_result["confidence"],
    }
    if schema_result.get("cached"):
        adapted["cached"] = True
    verification = schema_result.get("verification") or {}
    if verification:
        adapted["verification"] = verification
    # Raw model scores only (not the fallback "whole verdict as one question" entry, which
    # simply repeats the overall score).
    model_question_scores = [
        q.get("score") for q in (schema_result.get("questions") or []) if isinstance(q, dict)
    ]
    reasons = compute_attention_reasons(
        overall, schema_result["classification"], schema_result["confidence"], model_question_scores, verification
    )
    adapted["needs_attention"] = bool(reasons)
    adapted["attention_reasons"] = reasons
    return adapted
