"""
migrations.py
Versioned schema migrations for databases created by an OLDER version of this code.

`Base.metadata.create_all()` (db.init_db) creates missing TABLES but never alters an existing one,
so every new column on an existing table needs a migration here. Each migration has a number and
runs exactly once: the applied numbers are kept in the `schema_migrations` table. The statements
are written to be idempotent (IF NOT EXISTS), so a fresh database -- where create_all already made
the column -- passes through them harmlessly.

To add a migration: append a (number, name, [sql, ...]) tuple to MIGRATIONS. Never edit or
renumber one that has been released.

Only PostgreSQL is supported (that is what the application runs on); on any other dialect
(a unit-test SQLite engine) nothing is done.
"""

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)

MIGRATIONS = [
    (1, "submissions.assignment_id", [
        "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS assignment_id UUID REFERENCES assignments(id)",
        "CREATE INDEX IF NOT EXISTS ix_submissions_assignment_id ON submissions (assignment_id)",
    ]),
    (2, "submissions.regrade_count", [
        "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS regrade_count INTEGER NOT NULL DEFAULT 0",
    ]),
]


def apply_migrations(engine) -> list:
    """Applies the migrations that have not run yet; returns the numbers it applied."""
    if engine.dialect.name != "postgresql":
        return []
    applied_now = []
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, name VARCHAR(128) NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        ))
        # One migrator at a time (several API/worker processes may start together).
        conn.execute(text("SELECT pg_advisory_xact_lock(7283641)"))
        done = {row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))}
        for version, name, statements in sorted(MIGRATIONS):
            if version in done:
                continue
            for statement in statements:
                conn.execute(text(statement))
            conn.execute(
                text("INSERT INTO schema_migrations (version, name) VALUES (:v, :n)"), {"v": version, "n": name}
            )
            applied_now.append(version)
            logger.info("applied schema migration %s (%s)", version, name)
    return applied_now


def current_version(engine) -> int:
    """The highest applied migration number (0 if none or not PostgreSQL)."""
    if engine.dialect.name != "postgresql":
        return 0
    with engine.connect() as conn:
        try:
            return int(conn.execute(text("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")).scalar() or 0)
        except Exception:
            return 0
