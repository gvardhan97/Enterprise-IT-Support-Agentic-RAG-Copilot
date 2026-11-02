import json
import sqlite3
from datetime import datetime

import pytest

from app.services import audit


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.db"
    monkeypatch.setattr(audit.settings, "audit_db_path", str(path))
    return path


def rows(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT created_at, question, source_used, trace_json FROM query_audit ORDER BY id").fetchall()
    finally:
        con.close()


def test_init_db_is_safe_to_run_twice(db_path):
    audit.init_db()
    audit.write_audit("q1", "private_kb", [])
    audit.init_db()

    assert len(rows(db_path)) == 1


def test_write_audit_stores_question_source_and_trace(db_path):
    audit.init_db()

    audit.write_audit("How do I reset MFA?", "web_search", ["Router → KB", "Web fallback → Tavily search"])

    created_at, question, source, trace_json = rows(db_path)[0]
    assert question == "How do I reset MFA?"
    assert source == "web_search"
    assert json.loads(trace_json) == ["Router → KB", "Web fallback → Tavily search"]
    assert datetime.fromisoformat(created_at).tzinfo is not None


def test_question_text_is_stored_as_data_not_sql(db_path):
    audit.init_db()
    question = "x'); DROP TABLE query_audit; --"

    audit.write_audit(question, "direct", [])

    assert rows(db_path)[0][1] == question
