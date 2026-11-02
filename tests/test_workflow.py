import pytest

from app.rag import workflow
from conftest import FakeLLM


def test_greeting_skips_retrieval(fake_agent):
    agent = fake_agent(FakeLLM(route="direct"))

    result = workflow.ask("hi there")

    assert result["source_used"] == "direct"
    assert result["answer"] == "fake answer"
    assert result["trace"] == ["Router → DIRECT", "Direct response → no retrieval"]
    assert agent.retriever.queries == []
    assert agent.web.queries == []


def test_good_kb_evidence_answers_from_kb_without_web(fake_agent):
    agent = fake_agent(FakeLLM(kb_grades=["good"]))

    result = workflow.ask("How do I connect to the VPN?")

    assert result["source_used"] == "private_kb"
    assert result["trace"] == [
        "Router → KB",
        "Private KB retrieval → 3 chunks",
        "KB evidence grade → GOOD",
        "Answer generation → PRIVATE KB",
    ]
    assert agent.web.queries == []


def test_kb_citations_are_deduplicated_by_source(fake_agent):
    fake_agent(FakeLLM(kb_grades=["good"]))

    result = workflow.ask("How do I connect to the VPN?")

    assert result["citations"] == [
        {"title": "vpn.md", "url": "", "type": "private_kb"},
        {"title": "mfa_policy.md", "url": "", "type": "private_kb"},
    ]


def test_kb_answer_prompt_carries_every_chunk(fake_agent):
    agent = fake_agent(FakeLLM(kb_grades=["good"]))

    workflow.ask("How do I connect to the VPN?")

    answer_prompt = agent.llm.prompts[-1]
    assert "Answer ONLY from the private company KB" in answer_prompt
    for doc in agent.retriever.docs:
        assert doc.page_content in answer_prompt


def test_weak_kb_falls_back_to_web(fake_agent):
    agent = fake_agent(FakeLLM(kb_grades=["weak"], web_grades=["good"]))

    result = workflow.ask("Latest Microsoft Teams outage guidance?")

    assert result["source_used"] == "web_search"
    assert agent.web.queries == ["Latest Microsoft Teams outage guidance?"]
    assert result["trace"][-3:] == [
        "Web fallback → Tavily search",
        "Web evidence grade → GOOD",
        "Answer generation → WEB SEARCH",
    ]
    # A web result with no title is cited by its URL.
    assert result["citations"] == [
        {"title": "Teams status", "url": "https://status.example.com/teams", "type": "web"},
        {"title": "https://example.com/blog", "url": "https://example.com/blog", "type": "web"},
    ]


def test_weak_evidence_everywhere_rewrites_once_then_gives_up(fake_agent):
    agent = fake_agent(FakeLLM(kb_grades=["weak", "weak"], web_grades=["weak", "weak"]))

    result = workflow.ask("vpn broken")

    assert result["source_used"] == "insufficient_evidence"
    assert "contact the IT help desk" in result["answer"]
    # Second pass uses the rewritten query, with whitespace stripped.
    assert agent.retriever.queries == ["vpn broken", "vpn setup windows rewritten"]
    assert agent.web.queries == ["vpn broken", "vpn setup windows rewritten"]
    assert result["retry_count"] == 1
    assert result["trace"][-1] == "Stopped → insufficient reliable evidence"


def test_rewritten_query_can_recover_from_kb(fake_agent):
    agent = fake_agent(FakeLLM(kb_grades=["weak", "good"], web_grades=["weak"]))

    result = workflow.ask("vpn broken")

    assert result["source_used"] == "private_kb"
    assert result["current_query"] == "vpn setup windows rewritten"
    assert agent.web.queries == ["vpn broken"]


def test_zero_retries_stops_after_first_weak_web_grade(fake_agent, monkeypatch):
    monkeypatch.setattr(workflow.settings, "max_retries", 0)
    agent = fake_agent(FakeLLM(kb_grades=["weak"], web_grades=["weak"]))

    result = workflow.ask("vpn broken")

    assert result["source_used"] == "insufficient_evidence"
    assert agent.retriever.queries == ["vpn broken"]


@pytest.mark.parametrize("web_grade, retry_count, expected", [
    ("good", 0, "generate_from_web"),
    ("good", 5, "generate_from_web"),
    ("weak", 0, "rewrite_query"),
    ("weak", 1, "insufficient"),
])
def test_after_web_routing(web_grade, retry_count, expected):
    assert workflow.after_web({"web_grade": web_grade, "retry_count": retry_count}) == expected


@pytest.mark.parametrize("kb_grade, expected", [("good", "generate_from_kb"), ("weak", "search_web")])
def test_after_kb_routing(kb_grade, expected):
    assert workflow.after_kb({"kb_grade": kb_grade}) == expected


def test_search_web_handles_plain_text_result(fake_agent):
    fake_agent(FakeLLM(), web_result="raw text from search")

    update = workflow.search_web({"current_query": "q", "trace": []})

    assert update["web_results"] == "raw text from search"
    assert update["citations"] == []


def test_missing_api_keys_fail_loudly(monkeypatch):
    monkeypatch.setattr(workflow, "_llm", None)
    monkeypatch.setattr(workflow, "_web_search", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        workflow.llm()
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        workflow.web_search_tool()
