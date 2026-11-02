import threading
from collections import Counter, deque

# In-memory chat telemetry. Resets when the server restarts.
_lock = threading.Lock()
_latencies_ms = deque(maxlen=1000)
_sources = Counter()
_requests = 0
_errors = 0


def record_chat(source_used: str, latency_ms: float) -> None:
    global _requests
    with _lock:
        _requests += 1
        _sources[source_used] += 1
        _latencies_ms.append(latency_ms)


def record_error() -> None:
    global _requests, _errors
    with _lock:
        _requests += 1
        _errors += 1


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return round(ordered[index], 1)


def snapshot() -> dict:
    with _lock:
        latencies = list(_latencies_ms)
        answered = _requests - _errors
        return {
            "requests": _requests,
            "errors": _errors,
            "error_rate": round(_errors / _requests, 3) if _requests else 0.0,
            "source_used": dict(_sources),
            "web_fallback_rate": round(_sources["web_search"] / answered, 3) if answered else 0.0,
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "max": round(max(latencies), 1) if latencies else 0.0,
            },
        }


def reset() -> None:
    global _requests, _errors
    with _lock:
        _latencies_ms.clear()
        _sources.clear()
        _requests = 0
        _errors = 0
