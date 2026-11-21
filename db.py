"""
db.py
PostgreSQL-backed data layer for the CRM review workflow: organizations
(tenants), teachers, students, and submissions (the review queue).

This is the single source of truth for "has a teacher approved this
grade yet." Unlike training_logger.py's JSONL files (fine for building a
fine-tuning dataset) or Redis (fine for transient job/cache state),
review-queue submissions need proper multi-tenant isolation, concurrent-
safe updates, and queryability (a teacher's pending queue, a CRM's
approved-since-X poll) that only a real relational database gives you
at this scale.

Run once before first use:
    python -c "from db import init_db; init_db()"
"""

import enum
import hashlib
import secrets
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class SubmissionStatus(str, enum.Enum):
    PENDING_AI = "pending_ai"          # queued, AI grading not finished yet
    GRADING_FAILED = "grading_failed"  # AI grading errored out
    PENDING_REVIEW = "pending_review"  # AI graded it, waiting on a teacher
    APPROVED = "approved"              # teacher approved (as-is or edited) -- final


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_uuid(value) -> bool:
    """True if `value` is a well-formed UUID string. Ids come from the URL path; handing a
    malformed one to a UUID column makes PostgreSQL raise (-> HTTP 500) instead of "not found"."""
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


class InvalidTransition(Exception):
    """A submission is not in a status from which the requested change is allowed
    (e.g. approving something already approved, or reviewing before the AI finished)."""

    def __init__(self, submission_id: str, current_status: str, wanted: str):
        super().__init__(f"submission {submission_id} is '{current_status}', cannot {wanted}")
        self.submission_id = submission_id
        self.current_status = current_status
        self.wanted = wanted


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Organization(Base):
    """One tenant -- one school / training center / CRM account."""

    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    api_key_hash = Column(String(64), nullable=False, unique=True)
    webhook_url = Column(String(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)


class Teacher(Base):
    __tablename__ = "teachers"
    __table_args__ = (
        UniqueConstraint("organization_id", "external_teacher_id", name="uq_teacher_org_external"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    organization_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    external_teacher_id = Column(String(255), nullable=False)  # the CRM's own id for this teacher
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)


class Student(Base):
    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint("organization_id", "external_student_id", name="uq_student_org_external"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    organization_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    external_student_id = Column(String(255), nullable=False)  # the CRM's own id for this student
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)


class Assignment(Base):
    """One homework task given to many students. Holds the task text and the teacher's answer
    key ONCE, so every student's submission shares the same prompt prefix (provider-side prompt
    caching) instead of each submission carrying its own copy."""

    __tablename__ = "assignments"
    __table_args__ = (
        UniqueConstraint("organization_id", "external_assignment_id", name="uq_assignment_org_external"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    organization_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    external_assignment_id = Column(String(255), nullable=False)  # the CRM's own id for this homework
    task_text = Column(Text, nullable=True)
    answer_key = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)


class Submission(Base):
    """One homework file, from AI grading through teacher review to final approval."""

    __tablename__ = "submissions"
    __table_args__ = (
        UniqueConstraint("organization_id", "external_submission_id", name="uq_submission_org_external"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    organization_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    teacher_id = Column(UUID(as_uuid=False), ForeignKey("teachers.id"), nullable=True)
    student_id = Column(UUID(as_uuid=False), ForeignKey("students.id"), nullable=True)
    assignment_id = Column(UUID(as_uuid=False), ForeignKey("assignments.id"), nullable=True, index=True)
    # The CRM's own id for this submission, if it wants idempotent resubmits
    # (submitting the same external_submission_id twice won't create a duplicate).
    external_submission_id = Column(String(255), nullable=True)

    subject_key = Column(String(64), nullable=False, default="math")
    custom_subject_name = Column(String(255), nullable=True)
    language_key = Column(String(32), nullable=True)
    skill_key = Column(String(32), nullable=True)
    model_name = Column(String(128), nullable=True)
    backend = Column(String(32), nullable=True)
    extra_instructions = Column(Text, nullable=True)

    input_file_path = Column(String(1024), nullable=True)

    status = Column(String(32), nullable=False, default=SubmissionStatus.PENDING_AI.value)
    ai_result = Column(JSONB, nullable=True)          # what the AI produced, untouched
    reviewed_result = Column(JSONB, nullable=True)    # non-null only if the teacher edited it
    reviewer_note = Column(Text, nullable=True)

    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cached_tokens = Column(Integer, default=0)

    webhook_delivered = Column(Boolean, default=False)
    webhook_attempts = Column(Integer, default=0)
    last_webhook_error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_now)
    graded_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)

    def final_result(self) -> Optional[dict]:
        """The authoritative result: the teacher's edit if there is one,
        otherwise the AI's original (implicitly approved as-is)."""
        return self.reviewed_result or self.ai_result

    def to_dict(self, include_internal: bool = False) -> dict:
        """include_internal=True adds server-side details (the file path on THIS host) that
        in-process callers (workers) need but API responses must not leak."""
        data = {
            "id": self.id,
            "organization_id": self.organization_id,
            "teacher_id": self.teacher_id,
            "student_id": self.student_id,
            "assignment_id": self.assignment_id,
            "external_submission_id": self.external_submission_id,
            "subject_key": self.subject_key,
            "custom_subject_name": self.custom_subject_name,
            "language_key": self.language_key,
            "skill_key": self.skill_key,
            "model_name": self.model_name,
            "backend": self.backend,
            "extra_instructions": self.extra_instructions,
            "status": self.status,
            "ai_result": self.ai_result,
            "reviewed_result": self.reviewed_result,
            "final_result": self.final_result(),
            "reviewer_note": self.reviewer_note,
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
            },
            "webhook_delivered": self.webhook_delivered,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "graded_at": self.graded_at.isoformat() if self.graded_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }
        if include_internal:
            data["input_file_path"] = self.input_file_path
            data["assignment"] = self._assignment_context()
        return data

    def _assignment_context(self) -> Optional[dict]:
        """The shared task text / answer key for the prompt (workers only: the key is not
        echoed in API responses)."""
        if not self.assignment_id:
            return None
        with session_scope() as db:
            a = db.get(Assignment, self.assignment_id)
            if a is None:
                return None
            return {"task_text": a.task_text or "", "answer_key": a.answer_key or ""}


class WebhookDelivery(Base):
    """Audit trail of webhook delivery attempts, for debugging a CRM
    integration that isn't receiving approved results."""

    __tablename__ = "webhook_deliveries"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    submission_id = Column(UUID(as_uuid=False), ForeignKey("submissions.id"), nullable=False)
    attempted_at = Column(DateTime(timezone=True), default=_now)
    status_code = Column(Integer, nullable=True)
    success = Column(Boolean, default=False)
    error = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Engine / session management
# ---------------------------------------------------------------------------
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope():
    """Usage: with session_scope() as db: db.query(...)"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Creates all tables if they don't already exist. Safe to call repeatedly.
    For production schema *changes* over time, use Alembic migrations instead
    of relying on this (see README)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    # create_all never alters an existing table: add the one new column an older database lacks.
    if engine.dialect.name == "postgresql":
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS assignment_id UUID REFERENCES assignments(id)"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_submissions_assignment_id ON submissions (assignment_id)"))


# ---------------------------------------------------------------------------
# Organization (tenant) management
# ---------------------------------------------------------------------------
def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return "hwck_" + secrets.token_urlsafe(32)


def create_organization(name: str, webhook_url: Optional[str] = None) -> tuple:
    """
    Returns (organization_id, raw_api_key). raw_api_key is shown ONCE --
    only its hash is stored, so save it somewhere safe immediately; it
    cannot be recovered later (generate a new one instead if lost).
    """
    raw_key = generate_api_key()
    with session_scope() as db:
        org = Organization(name=name, api_key_hash=hash_api_key(raw_key), webhook_url=webhook_url)
        db.add(org)
        db.flush()
        org_id = org.id
    return org_id, raw_key


def get_organization_by_api_key(raw_key: str) -> Optional[dict]:
    key_hash = hash_api_key(raw_key)
    with session_scope() as db:
        org = db.query(Organization).filter_by(api_key_hash=key_hash).first()
        if not org:
            return None
        return {"id": org.id, "name": org.name, "webhook_url": org.webhook_url}


def get_organization(organization_id: str) -> Optional[dict]:
    with session_scope() as db:
        org = db.query(Organization).filter_by(id=organization_id).first()
        if not org:
            return None
        return {"id": org.id, "name": org.name, "webhook_url": org.webhook_url}


def get_or_create_teacher(organization_id: str, external_teacher_id: str, name: Optional[str] = None) -> str:
    with session_scope() as db:
        teacher = db.query(Teacher).filter_by(
            organization_id=organization_id, external_teacher_id=external_teacher_id
        ).first()
        if teacher:
            return teacher.id
        teacher = Teacher(organization_id=organization_id, external_teacher_id=external_teacher_id, name=name)
        db.add(teacher)
        db.flush()
        return teacher.id


def get_or_create_student(organization_id: str, external_student_id: str, name: Optional[str] = None) -> str:
    with session_scope() as db:
        student = db.query(Student).filter_by(
            organization_id=organization_id, external_student_id=external_student_id
        ).first()
        if student:
            return student.id
        student = Student(organization_id=organization_id, external_student_id=external_student_id, name=name)
        db.add(student)
        db.flush()
        return student.id


def get_or_create_assignment(
    organization_id: str, external_assignment_id: str, task_text: Optional[str] = None, answer_key: Optional[str] = None
) -> str:
    """Returns the assignment id, creating it on first sight. task_text / answer_key, when given,
    are stored (a later non-empty value replaces an earlier one, so the teacher can fix the key)."""
    with session_scope() as db:
        a = db.query(Assignment).filter_by(
            organization_id=organization_id, external_assignment_id=external_assignment_id
        ).first()
        if a is None:
            a = Assignment(organization_id=organization_id, external_assignment_id=external_assignment_id)
            db.add(a)
        if task_text:
            a.task_text = task_text
        if answer_key:
            a.answer_key = answer_key
        db.flush()
        return a.id


# ---------------------------------------------------------------------------
# Submissions (the review queue)
# ---------------------------------------------------------------------------
def create_submission(
    organization_id: str,
    teacher_external_id: Optional[str] = None,
    teacher_name: Optional[str] = None,
    student_external_id: Optional[str] = None,
    student_name: Optional[str] = None,
    external_submission_id: Optional[str] = None,
    subject_key: str = "math",
    custom_subject_name: str = "",
    language_key: str = "",
    skill_key: str = "",
    model_name: str = "",
    backend: str = "",
    extra_instructions: str = "",
    input_file_path: str = "",
    assignment_id: Optional[str] = None,
) -> str:
    """
    Creates a new submission row (status=pending_ai) and returns its id.
    If external_submission_id is given and already exists for this
    organization, returns the EXISTING submission's id instead of
    creating a duplicate (idempotent resubmission).
    """
    if external_submission_id:
        with session_scope() as db:
            existing = db.query(Submission).filter_by(
                organization_id=organization_id, external_submission_id=external_submission_id
            ).first()
            if existing:
                return existing.id

    teacher_id = (
        get_or_create_teacher(organization_id, teacher_external_id, teacher_name)
        if teacher_external_id else None
    )
    student_id = (
        get_or_create_student(organization_id, student_external_id, student_name)
        if student_external_id else None
    )

    try:
        with session_scope() as db:
            sub = Submission(
                organization_id=organization_id,
                teacher_id=teacher_id,
                student_id=student_id,
                assignment_id=assignment_id,
                external_submission_id=external_submission_id,
                subject_key=subject_key,
                custom_subject_name=custom_subject_name,
                language_key=language_key,
                skill_key=skill_key,
                model_name=model_name,
                backend=backend,
                extra_instructions=extra_instructions,
                input_file_path=input_file_path,
                status=SubmissionStatus.PENDING_AI.value,
            )
            db.add(sub)
            db.flush()
            return sub.id
    except IntegrityError:
        # Two simultaneous requests with the same external_submission_id: the unique
        # constraint let exactly one through -- return that row (idempotent), not a 500.
        if not external_submission_id:
            raise
        with session_scope() as db:
            existing = db.query(Submission).filter_by(
                organization_id=organization_id, external_submission_id=external_submission_id
            ).first()
            if existing:
                return existing.id
        raise


def mark_submission_graded(
    submission_id: str, ai_result: dict, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0
) -> bool:
    """
    Stores the AI result and moves the submission to pending_review -- but ONLY if it is still
    waiting for the AI (pending_ai) or previously failed (grading_failed). A late or duplicate
    grading job must never overwrite a result a teacher is already looking at, or re-open an
    approved submission. Returns True if the result was stored, False if it was ignored.
    """
    with session_scope() as db:
        sub = db.query(Submission).filter_by(id=submission_id).with_for_update().first()
        if not sub:
            return False
        if sub.status not in (SubmissionStatus.PENDING_AI.value, SubmissionStatus.GRADING_FAILED.value):
            return False
        sub.ai_result = ai_result
        sub.status = SubmissionStatus.PENDING_REVIEW.value
        sub.graded_at = _now()
        sub.input_tokens = input_tokens
        sub.output_tokens = output_tokens
        sub.cached_tokens = cached_tokens
        return True


def mark_submission_failed(submission_id: str, error: str) -> bool:
    """Marks a still-pending_ai submission as grading_failed. Ignored (False) if it has
    meanwhile been graded, reviewed or approved by another job."""
    with session_scope() as db:
        sub = db.query(Submission).filter_by(id=submission_id).with_for_update().first()
        if not sub or sub.status != SubmissionStatus.PENDING_AI.value:
            return False
        sub.status = SubmissionStatus.GRADING_FAILED.value
        sub.reviewer_note = f"Grading failed: {error}"
        return True


def reset_for_regrade(submission_id: str, organization_id: str) -> Optional[dict]:
    """
    Puts a grading_failed or pending_review submission back to pending_ai so it can be graded
    again (e.g. after a provider outage or a prompt/model fix). Approved submissions are final
    and cannot be regraded. Returns the updated submission, None if it does not exist for this
    organization; raises InvalidTransition otherwise.
    """
    if not _is_uuid(submission_id):
        return None
    with session_scope() as db:
        sub = db.query(Submission).filter_by(id=submission_id, organization_id=organization_id).with_for_update().first()
        if not sub:
            return None
        if sub.status not in (SubmissionStatus.GRADING_FAILED.value, SubmissionStatus.PENDING_REVIEW.value):
            raise InvalidTransition(submission_id, sub.status, "regrade")
        sub.status = SubmissionStatus.PENDING_AI.value
        if sub.reviewer_note and sub.reviewer_note.startswith("Grading failed:"):
            sub.reviewer_note = None
        db.flush()
        return sub.to_dict()


def list_stuck_pending_ai(older_than_seconds: int, limit: int = 500) -> list:
    """(id, age_seconds) of submissions still pending_ai after `older_than_seconds` -- the
    candidates for the reconciler (reconcile.py). Oldest first."""
    cutoff = _now() - timedelta(seconds=older_than_seconds)
    with session_scope() as db:
        rows = (
            db.query(Submission.id, Submission.created_at)
            .filter(Submission.status == SubmissionStatus.PENDING_AI.value, Submission.created_at < cutoff)
            .order_by(Submission.created_at.asc())
            .limit(limit)
            .all()
        )
        now = _now()
        return [(r.id, (now - r.created_at).total_seconds()) for r in rows]


def get_submission(
    submission_id: str, organization_id: Optional[str] = None, include_internal: bool = False
) -> Optional[dict]:
    if not _is_uuid(submission_id):
        return None
    with session_scope() as db:
        q = db.query(Submission).filter_by(id=submission_id)
        if organization_id:
            q = q.filter_by(organization_id=organization_id)
        sub = q.first()
        return sub.to_dict(include_internal=include_internal) if sub else None


def list_submissions(
    organization_id: str,
    status: Optional[str] = None,
    teacher_external_id: Optional[str] = None,
    student_external_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> list:
    with session_scope() as db:
        q = db.query(Submission).filter_by(organization_id=organization_id)
        if status:
            q = q.filter_by(status=status)
        if teacher_external_id:
            q = q.join(Teacher, Submission.teacher_id == Teacher.id).filter(
                Teacher.external_teacher_id == teacher_external_id
            )
        if student_external_id:
            q = q.join(Student, Submission.student_id == Student.id).filter(
                Student.external_student_id == student_external_id
            )
        if since:
            date_field = Submission.approved_at if status == SubmissionStatus.APPROVED.value else Submission.created_at
            q = q.filter(date_field >= since)
        q = q.order_by(Submission.created_at.desc()).limit(limit).offset(offset)
        return [s.to_dict() for s in q.all()]


def approve_submission(
    submission_id: str,
    organization_id: str,
    corrected_result: Optional[dict] = None,
    reviewer_note: Optional[str] = None,
) -> Optional[dict]:
    """
    Approves a submission -- either as-is (corrected_result=None, the AI's
    result stands) or with teacher edits (corrected_result= the corrected
    JSON). Either way, status becomes "approved" and this is what a CRM
    should treat as final. Scoped to organization_id so one tenant can't
    approve another tenant's submission.

    Raises InvalidTransition unless the submission is currently pending_review (the check and
    the update happen under one row lock, so concurrent approvals cannot both succeed).
    """
    if not _is_uuid(submission_id):
        return None
    with session_scope() as db:
        # Row lock: of N simultaneous approvals exactly one sees pending_review; the others
        # block until it commits, then see "approved" and are refused (-> HTTP 409).
        sub = (
            db.query(Submission)
            .filter_by(id=submission_id, organization_id=organization_id)
            .with_for_update()
            .first()
        )
        if not sub:
            return None
        if sub.status != SubmissionStatus.PENDING_REVIEW.value:
            raise InvalidTransition(submission_id, sub.status, "approve")
        if corrected_result is not None:
            sub.reviewed_result = corrected_result
        sub.status = SubmissionStatus.APPROVED.value
        sub.reviewed_at = _now()
        sub.approved_at = _now()
        if reviewer_note:
            sub.reviewer_note = reviewer_note
        db.flush()
        return sub.to_dict()


def record_webhook_attempt(submission_id: str, success: bool, status_code: Optional[int] = None, error: Optional[str] = None) -> None:
    with session_scope() as db:
        sub = db.get(Submission, submission_id)
        if sub:
            sub.webhook_attempts = (sub.webhook_attempts or 0) + 1
            if success:
                sub.webhook_delivered = True
                sub.last_webhook_error = None
            else:
                sub.last_webhook_error = error

        delivery = WebhookDelivery(
            submission_id=submission_id, status_code=status_code, success=success, error=error
        )
        db.add(delivery)
