"""
pipeline/llm_telemetry.py

Per-call latency samples, by role and by the upstream that served them. `llm_client` re-exports
these names.
"""

from __future__ import annotations

import threading

# ── per-call latency samples, per role ─────────────────────────────────────────
_LATENCY: dict[str, list[float]] = {}
# Same samples, keyed by the upstream that served them, so `provider.sort` routing is auditable.
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

    The samples are append-only, so the marker records per-key sample counts and the
    distributions slice from there. Used instead of `reset_latency()`, which would discard a
    concurrent caller's samples."""
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
    """Per-role per-call latency: {count, p50, p95, max, spread}, with `spread` = p95/p50.

    `since` is a `latency_marker()` taken at run start, bounding the result to that run;
    `since=None` reports the process lifetime."""
    with _LATENCY_LOCK:
        snap = _sliced(_LATENCY, (since or {}).get("roles", {}))
    return {role: _summarize(xs) for role, xs in snap.items()}


def upstream_distribution(since: dict | None = None) -> dict[str, dict]:
    """Per-upstream per-call latency — who served this run's calls, and how fast.

    A denied upstream appearing at all, or an `"unknown"` bucket, is the alarm. `since` bounds it
    to one run (see `latency_marker`)."""
    with _LATENCY_LOCK:
        snap = _sliced(_UPSTREAM_LATENCY, (since or {}).get("upstreams", {}))
    return {up: _summarize(xs) for up, xs in snap.items()}


def reset_latency() -> None:
    """Drop collected samples (tests / a fresh measurement window)."""
    with _LATENCY_LOCK:
        _LATENCY.clear()
        _UPSTREAM_LATENCY.clear()   # both, or a "fresh window" silently keeps stale upstreams
