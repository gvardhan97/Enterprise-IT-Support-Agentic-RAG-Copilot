import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

# Settings are read once, at import time, so point them at throwaway paths and
# blank keys before any app module loads. No test ever talks to OpenAI,
# Pinecone or Tavily.
_tmp = Path(tempfile.mkdtemp(prefix="rag-tests-"))
os.environ.update({
    "AUDIT_DB_PATH": str(_tmp / "audit.db"),
    "UPLOAD_DIR": str(_tmp / "uploads"),
    "ADMIN_API_KEY": "test-admin-key",
    "OPENAI_API_KEY": "",
    "PINECONE_API_KEY": "",
    "TAVILY_API_KEY": "",
})

from app.rag import workflow  # noqa: E402
from app.rag.state import EvidenceGrade, RouteDecision  # noqa: E402


class FakeLLM:
    """Stands in for ChatOpenAI. Grades are handed out in order, one per call."""

    def __init__(self, route="kb", kb_grades=("good",), web_grades=("good",)):
        self.route = route
        self.kb_grades = list(kb_grades)
        self.web_grades = list(web_grades)
        self.prompts = []

    def with_structured_output(self, schema, method=None):
        def invoke(prompt):
            self.prompts.append(prompt)
            if schema is RouteDecision:
                return RouteDecision(route=self.route)
            grades = self.kb_grades if "Private company KB evidence" in prompt else self.web_grades
            return EvidenceGrade(grade=grades.pop(0))
        return SimpleNamespace(invoke=invoke)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if prompt.lstrip().startswith("Rewrite this IT support question"):
            return SimpleNamespace(content="  vpn setup windows rewritten  ")
        return SimpleNamespace(content="fake answer")


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs


class FakeWebSearch:
    def __init__(self, result):
        self.result = result
        self.queries = []

    def invoke(self, payload):
        self.queries.append(payload["query"])
        return self.result


KB_DOCS = [
    Document(page_content="Open GlobalProtect and sign in.", metadata={"source": "data/sample_kb/vpn.md"}),
    Document(page_content="Use your SSO password.", metadata={"source": "data/sample_kb/vpn.md"}),
    Document(page_content="MFA is required off-site.", metadata={"source": "data/sample_kb/mfa_policy.md"}),
]

WEB_RESULT = {
    "answer": "Teams is degraded in EU.",
    "results": [
        {"title": "Teams status", "url": "https://status.example.com/teams", "content": "Degraded"},
        {"title": "", "url": "https://example.com/blog", "content": "Workaround"},
    ],
}


@pytest.fixture
def fake_agent(monkeypatch):
    """Wire fakes into the LangGraph nodes; returns them so tests can inspect calls."""
    def install(llm, docs=KB_DOCS, web_result=WEB_RESULT):
        retriever = FakeRetriever(docs)
        web = FakeWebSearch(web_result)
        monkeypatch.setattr(workflow, "llm", lambda: llm)
        monkeypatch.setattr(workflow, "get_retriever", lambda: retriever)
        monkeypatch.setattr(workflow, "web_search_tool", lambda: web)
        return SimpleNamespace(llm=llm, retriever=retriever, web=web)
    return install
