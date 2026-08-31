"""
pipeline/llm_telemetry.py

Per-call latency and per-upstream-serving-latency sampling — split out of `llm_client.py`
(2026-08-16). `llm_client.py` re-exports everything here so existing
`llm_client.latency_distribution()`-style call sites keep working unchanged.
"""

from __future__ import annotations

import threading

# ── per-CALL latency samples (hedging decision input) ───────────────────────────
# Keeps every call's timing (`LLMResponse.elapsed_s`) to answer whether hedging can work: a large
# within-run spread means a duplicate call can land somewhere healthier (hedging pays); a small
# spread means the provider is uniformly slow (a duplicate lands on the same degraded provider,
# doubling spend for nothing). `StageTimer` can't distinguish these — it samples per atom, not per
# call. Kept per-ROLE since `vision` and `content_quality` may have different tail behaviour.
_LATENCY: dict[str, list[float]] = {}
# Same samples, keyed by the UPSTREAM that served them. Kept apart from `_LATENCY` (not nested)
# so `latency_distribution()`'s existing shape doesn't change. This is what makes `provider.sort`
# auditable — which upstream OpenRouter picks varies run to run, and a routing regression would
# otherwise be invisible.
_UPSTREAM_LATENCY: dict[str, list[float]] = {}
_LATENCY_LOCK = threading.Lock()


def _record_latency(role: str, elapsed: float, upstream: str | None = None) -> None:
    with _LATENCY_LOCK:
        _LATENCY.setdefault(role, []).append(float(elapsed))
        if upstream:                       # optional — 2-arg callers (tests) stay valid
            _UPSTREAM_LATENCY.setdefault(upstream, []).append(float(elapsed))


def _summarize(xs: list[float]) -> dict:
    """{count, p50, p95, max, spread} for one sorted sample list."""
    def _p(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        i = (len(xs) - 1) * (p / 100.0)
        lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    p50, p95 = _p(50), _p(95)
    return {"count": len(xs), "p50": round(p50, 3), "p95": round(p95, 3),
            "max": round(xs[-1], 3), "spread": round(p95 / p50, 2) if p50 else None}


def latency_marker() -> dict:
    """Bookmark the sample lists so a later summary can report one run, not the process lifetime.

    `_LATENCY`/`_UPSTREAM_LATENCY` are append-only module globals, so in a long-lived process
    (the MCP server) a second run would otherwise fold the first run's calls into its own
    distribution. Percentile summaries can't be subtracted, so the marker records per-key sample
    COUNTS and the distributions slice from there — the raw samples diff cleanly.

    Not `reset_latency()` at run start: that would discard a concurrently running caller's
    samples, which assumes runs are sequential — an assumption about the host the client-agnostic
    invariant forbids. `reset_latency()` stays for tests.

    Roles and upstreams are namespaced rather than flattened into one dict, since role names
    (from settings.yaml) and upstream names (from the provider) could otherwise collide on a
    word and silently offset the wrong list."""
    with _LATENCY_LOCK:
        return {"roles": {r: len(xs) for r, xs in _LATENCY.items()},
                "upstreams": {u: len(xs) for u, xs in _UPSTREAM_LATENCY.items()}}


def _sliced(samples: dict[str, list[float]], offsets: dict) -> dict[str, list[float]]:
    """Sorted samples per key, dropping everything recorded before the marker. A key absent from
    the marker did not exist at mark time, so all of its samples belong to this run → offset 0.
    Caller holds `_LATENCY_LOCK`."""
    out = {}
    for key, xs in samples.items():
        tail = xs[offsets.get(key, 0):]
        if tail:
            out[key] = sorted(tail)
    return out


def latency_distribution(since: dict | None = None) -> dict[str, dict]:
    """Per-role per-CALL latency: {count, p50, p95, max, spread}. `spread` = p95/p50 — a big
    spread means the tail varies call to call, a spread near 1.0 means every call is equally slow.
    This view cannot tell routing from anything else — that's what `upstream_distribution()` is for.

    `since` is a `latency_marker()` taken at run start; the result then covers that run only.
    `since=None` reports the process lifetime — today's behaviour, unchanged."""
    with _LATENCY_LOCK:
        snap = _sliced(_LATENCY, (since or {}).get("roles", {}))
    return {role: _summarize(xs) for role, xs in snap.items()}


def upstream_distribution(since: dict | None = None) -> dict[str, dict]:
    """Per-UPSTREAM per-CALL latency — who actually served this run's calls, and how fast.

    The audit trail for `provider.sort: "throughput"`: the chosen upstream legitimately changes
    between runs, so a routing regression has no other symptom. A concentration on one upstream is
    the healthy shape; a denied upstream appearing at all, or an `"unknown"` bucket, is the alarm.

    `since` bounds it to one run — see `latency_marker`. Without it a stale upstream would stay in
    the summary long after it stopped serving."""
    with _LATENCY_LOCK:
        snap = _sliced(_UPSTREAM_LATENCY, (since or {}).get("upstreams", {}))
    return {up: _summarize(xs) for up, xs in snap.items()}


def reset_latency() -> None:
    """Drop collected samples (tests / a fresh measurement window)."""
    with _LATENCY_LOCK:
        _LATENCY.clear()
        _UPSTREAM_LATENCY.clear()   # both, or a "fresh window" silently keeps stale upstreams
