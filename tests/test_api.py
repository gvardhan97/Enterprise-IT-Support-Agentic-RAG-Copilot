import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.services import audit, metrics

ADMIN = {"x-admin-key": "test-admin-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.settings, "audit_db_path", str(tmp_path / "audit.db"))
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path / "uploads"))
    audit.init_db()
    metrics.reset()
    return TestClient(app)


def fake_ask(source="private_kb"):
    def ask(question):
        return {
            "answer": f"answer to {question}",
            "source_used": source,
            "trace": ["Router → KB"],
            "citations": [{"title": "vpn.md", "url": "", "type": "private_kb"}],
            "current_query": question + " (rewritten)",
        }
    return ask


def test_home_page_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health(client):
    assert client.get("/api/health").json() == {
        "status": "ok",
        "service": "Enterprise IT Support Agentic RAG Copilot",
    }


def test_chat_returns_answer_and_writes_audit(client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "ask", fake_ask())

    response = client.post("/api/chat", json={"question": "How do I use the VPN?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "answer to How do I use the VPN?",
        "source_used": "private_kb",
        "trace": ["Router → KB"],
        "citations": [{"title": "vpn.md", "url": "", "type": "private_kb"}],
        "rewritten_query": "How do I use the VPN? (rewritten)",
    }
    con = sqlite3.connect(tmp_path / "audit.db")
    assert con.execute("SELECT question, source_used FROM query_audit").fetchall() == [
        ("How do I use the VPN?", "private_kb")
    ]
    con.close()


@pytest.mark.parametrize("body", [{}, {"question": "x"}, {"question": "x" * 3001}])
def test_chat_rejects_bad_questions(client, body):
    assert client.post("/api/chat", json=body).status_code == 422


def test_chat_failure_returns_500_and_counts_error(client, monkeypatch):
    def broken(question):
        raise RuntimeError("Pinecone timed out")
    monkeypatch.setattr(routes, "ask", broken)

    response = client.post("/api/chat", json={"question": "vpn?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Pinecone timed out"
    assert client.get("/api/metrics").json()["errors"] == 1


def test_metrics_track_each_chat(client, monkeypatch):
    for source in ["private_kb", "web_search"]:
        monkeypatch.setattr(routes, "ask", fake_ask(source))
        client.post("/api/chat", json={"question": "vpn?"})

    snap = client.get("/api/metrics").json()

    assert snap["requests"] == 2
    assert snap["source_used"] == {"private_kb": 1, "web_search": 1}
    assert snap["web_fallback_rate"] == 0.5
    assert snap["latency_ms"]["max"] >= 0


@pytest.mark.parametrize("headers", [{}, {"x-admin-key": "wrong"}])
def test_ingest_needs_admin_key(client, headers):
    response = client.post("/api/ingest", files={"file": ("kb.txt", b"text")}, headers=headers)

    assert response.status_code == 401


def test_ingest_rejects_unsupported_files(client):
    response = client.post("/api/ingest", files={"file": ("run.exe", b"MZ")}, headers=ADMIN)

    assert response.status_code == 400
    assert response.json()["detail"] == "Supported: .docx, .md, .pdf, .txt"


def test_ingest_indexes_document(client, monkeypatch, tmp_path):
    indexed = []
    monkeypatch.setattr(routes, "add_documents", lambda chunks: indexed.extend(chunks) or ["id"] * len(chunks))

    response = client.post("/api/ingest", files={"file": ("wifi.md", b"Join the CORP network.")}, headers=ADMIN)

    assert response.json() == {"message": "Document indexed", "file": "wifi.md", "chunks": 1, "ids_created": 1}
    assert (tmp_path / "uploads" / "wifi.md").read_bytes() == b"Join the CORP network."
    assert indexed[0].page_content == "Join the CORP network."


def test_ingest_cannot_write_outside_upload_dir(client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "add_documents", lambda chunks: [])

    response = client.post("/api/ingest", files={"file": ("../../evil.md", b"x")}, headers=ADMIN)

    assert response.status_code == 200
    assert (tmp_path / "uploads" / "evil.md").exists()
    assert not (tmp_path.parent / "evil.md").exists()
