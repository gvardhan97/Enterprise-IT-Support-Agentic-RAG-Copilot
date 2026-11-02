import pytest

from app.services import metrics


@pytest.fixture(autouse=True)
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


def test_empty_snapshot():
    assert metrics.snapshot() == {
        "requests": 0,
        "errors": 0,
        "error_rate": 0.0,
        "source_used": {},
        "web_fallback_rate": 0.0,
        "latency_ms": {"p50": 0.0, "p95": 0.0, "max": 0.0},
    }


def test_counts_sources_errors_and_rates():
    for source in ["private_kb", "private_kb", "web_search", "direct"]:
        metrics.record_chat(source, 100)
    metrics.record_error()

    snap = metrics.snapshot()

    assert snap["requests"] == 5
    assert snap["errors"] == 1
    assert snap["error_rate"] == 0.2
    assert snap["source_used"] == {"private_kb": 2, "web_search": 1, "direct": 1}
    assert snap["web_fallback_rate"] == 0.25


def test_latency_percentiles():
    for ms in range(1, 101):
        metrics.record_chat("private_kb", float(ms))

    assert metrics.snapshot()["latency_ms"] == {"p50": 51.0, "p95": 95.0, "max": 100.0}


def test_only_recent_latencies_are_kept():
    metrics.record_chat("private_kb", 99999)
    for _ in range(1000):
        metrics.record_chat("private_kb", 10)

    snap = metrics.snapshot()
    assert snap["requests"] == 1001
    assert snap["latency_ms"]["max"] == 10


def test_only_errors_has_no_fallback_rate():
    metrics.record_error()

    assert metrics.snapshot()["web_fallback_rate"] == 0.0
