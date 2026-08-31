"""
pipeline/llm_spend.py

Pricing, cost accounting, rail attribution, and the api_stats.json file — split out of
`llm_client.py` (2026-08-16) because "route a call to a backend" and "account for what it cost"
are unrelated concerns that happened to share a file. `llm_client.py` re-exports everything here
(see its own compat section) so the ~45 existing `llm_client.spend_today()`-style call sites keep
working unchanged.

Every dollar this repo spends passes through `_bump_stats` (LLM calls, via `llm_client.call`) or
`record_external_cost` (non-LLM providers, e.g. the hosted embedder) — so this module is the ONE place
spend is recorded, read back, and flushed to disk.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opyt_core.paths import opyt_path

from pipeline.ingestion.utils import log

# ── Pricing table — the FALLBACK, not the source of truth ────────────────────
# (input_per_M_usd, output_per_M_usd). Updated quarterly. `PRICING_TABLE_VERSION` below is the
# ONLY home for the version — it is stamped into api_stats.json so a recorded figure can be read
# back against the rates that produced it. A `llm_backends.pricing_table_version` key mirrored it
# in settings.yaml until 2026-08-28, read by nothing and kept in sync by a comment; both had gone
# two months stale together. Do not re-add it: a version that describes THIS table belongs beside
# THIS table.
#
# For OpenRouter this table is a fallback only: `call()` prefers the charge the response reports
# (see `_reported_cost`), because OpenRouter picks the serving upstream per request and upstreams
# don't share a rate card — this table is a lower bound, not an estimate.
#
# ⚠️ A row here can go stale SILENTLY, and did: `llama-3.3-70b` sat at (0.13, 0.40) from 2026-06
# until 2026-08-28, when OpenRouter's catalog said (0.71, 0.71) — 5.5x low on input, on the model
# six of nine roles use. It cost nothing, because the reported charge wins on every normal call;
# it would have cost a BYOK user, whose calls report no charge and so land HERE. Re-verify against
# https://openrouter.ai/api/v1/models when you bump the version, and never hand-estimate a row.
PRICING_TABLE_VERSION = "2026-08"

# Every key is vendor-prefixed, because every key is an OpenRouter slug — there is no direct
# vendor API on any path. Claude models are reachable, as `anthropic/claude-*`; a bare
# `claude-sonnet-*` row would name a model id that nothing here can route.
_PRICING: dict[str, tuple[float, float]] = {
    # OpenRouter (cheapest routes, approximate) — the only provider now.
    # Rows without a dated note were read from OpenRouter's catalog on 2026-08-28.
    "meta-llama/llama-3.3-70b-instruct": (0.71, 0.71),
    "meta-llama/llama-3.1-8b-instruct":  (0.02, 0.05),
    "openai/gpt-4o-mini":                (0.15, 0.60),
    "google/gemini-2.5-flash":           (0.30, 2.50),  # multimodal vision role; OpenRouter, verified 2026-06
    "anthropic/claude-sonnet-5":         (2.00, 10.00),  # frontier_reader role; verified 2026-08-09
    "deepseek/deepseek-v4-flash":        (0.087, 0.174),   # frontier_reader role (settings.yaml)
    # Model OVERRIDES on the `vision` role (pipeline/model_routing.py). `call()` records under the
    # override, not the role's model, so these are the keys the OCR cascade's spend looks up.
    "google/gemma-3-27b-it":             (0.08, 0.45),
    "amazon/nova-lite-v1":               (0.06, 0.24),     # declared OCR fallback
}

# Models this table was asked for and does not have — logged ONCE each. A miss returns $0, and a
# silent $0 is not a harmless estimate: `rail_budget_exhausted` reads this meter, so an unpriced
# model on the reported-cost-absent path gives a rail a ceiling that can never bind.
_UNPRICED_LOGGED: set[str] = set()


# ── Cost accounting ──────────────────────────────────────────────────────────

_STATS_LOCK = threading.Lock()


# ── which RAIL is spending ───────────────────────────────────────────────────
# Every dollar passes through `_bump_stats` or `record_external_cost`, so labelling those two
# gives total attribution with no call-site churn. The label lets each rail's daily ceiling
# govern only that rail instead of sharing one pool

UNATTRIBUTED = "unattributed"
_CURRENT_RAIL = UNATTRIBUTED      # a module GLOBAL, deliberately — see `rail()`


def current_rail() -> str:
    """The rail label spend is presently attributed to. `UNATTRIBUTED` outside any rail scope."""
    return _CURRENT_RAIL


@contextmanager
def rail(name: str):
    """Label every dollar recorded inside this scope as belonging to `name`.

    A module global, not a ContextVar: `ThreadPoolExecutor` doesn't propagate context to workers,
    and the hottest spend paths fan out through one, so a ContextVar would read as unset there and
    silently misattribute. Nesting restores the previous label (not unattributed) in a `finally`,
    so an inner scope or a raising rail can't smear its label over the rest of an outer run."""
    global _CURRENT_RAIL
    prev = _CURRENT_RAIL
    _CURRENT_RAIL = name or UNATTRIBUTED
    try:
        yield
    finally:
        _CURRENT_RAIL = prev


def _rail_key(when: str) -> str:
    """The `by_rail` bucket key: `"{rail}|{date}"` — flat and composite (matching every other
    bucket) rather than nested `by_rail[rail][date]`, so `_bump_stats` needs no rail-specific
    path and an old api_stats.json with no `by_rail` still loads with no migration."""
    return f"{_CURRENT_RAIL}|{when}"


def _fresh_stats() -> dict[str, Any]:
    """The empty stats shape — ONE definition.

    The test seam (`_override_stats_file_for_tests`) used to hand-copy this literal, so adding a
    bucket to production silently drifted from the test copy and `_bump_stats` KeyError'd on
    whichever tests reset it. A second copy of a contract is the bug; a factory is the fix."""
    return {
        "lifetime": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "by_model": {},
        "by_role": {},
        "by_date": {},
        "by_provider": {},   # NON-LLM external spend (the hosted embedder) — no tokens, request-counted
        # Which UPSTREAM actually served each call — `by_provider`/`by_model` can't answer this,
        # since `provider.sort` routes one model id to many upstreams per request at different
        # price/speed/quality.
        "by_upstream": {},
        # Which rail caused each dollar, keyed `"{rail}|{date}"` — see `_rail_key`.
        "by_rail": {},
    }


_STATS: dict[str, Any] = _fresh_stats()

_STATS_FILE_OVERRIDE: Path | None = None
_STATS_LOADED = False


def _stats_file() -> Path:
    if _STATS_FILE_OVERRIDE is not None:
        return _STATS_FILE_OVERRIDE
    # Canonical, machine-wide location alongside opyt.db / note_state.pkl. Lives
    # outside any worktree so the writer (pipeline) and reader (Tauri app, which
    # runs from a different worktree / the installed bundle) agree on one path.
    return opyt_path("api_stats.json")


def _load_stats_once() -> None:
    """Lazily load existing api_stats.json so re-runs accumulate."""
    global _STATS_LOADED, _STATS
    if _STATS_LOADED:
        return
    p = _stats_file()
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            for key in _STATS:
                if key in existing:
                    _STATS[key] = existing[key]
        except (OSError, json.JSONDecodeError):
            log(f"[llm_spend] could not load existing {p}, starting fresh")
    _STATS_LOADED = True


def _serving_upstream(provider: str, raw: dict | None) -> str:
    """WHO actually served this call — the OpenRouter upstream when there is one, else the
    configured provider.

    Never returns None. An unattributable call is bucketed as `"unknown"` rather than dropped,
    because a silently-absent sample is indistinguishable from no call at all — and "we cannot
    see it" is the condition this bucket exists to make visible."""
    if provider == "openrouter":
        return str((raw or {}).get("provider") or "unknown")
    return provider or "unknown"


def _bump_stats(role: str, model: str, in_tok: int, out_tok: int, cost: float,
                upstream: str | None = None) -> None:
    _load_stats_once()
    today = datetime.now(timezone.utc).date().isoformat()
    with _STATS_LOCK:
        life = _STATS["lifetime"]
        life["calls"] += 1
        life["input_tokens"] += in_tok
        life["output_tokens"] += out_tok
        life["cost_usd"] = round(life["cost_usd"] + cost, 6)

        buckets = [("by_model", model), ("by_role", role), ("by_date", today),
                   ("by_rail", _rail_key(today))]
        if upstream:                       # optional so existing 5-arg callers/tests stay valid
            buckets.append(("by_upstream", upstream))
        for bucket_name, key in buckets:
            b = _STATS[bucket_name].setdefault(
                key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            )
            b["calls"] += 1
            b["input_tokens"] += in_tok
            b["output_tokens"] += out_tok
            b["cost_usd"] = round(b["cost_usd"] + cost, 6)


def flush_stats() -> Path:
    """Write accumulated stats to disk. Safe to call multiple times.

    Atomic-write-via-rename: build the complete file under a sibling temp name, then
    os.replace() it into place, so a concurrent reader (the Tauri spend widget) never sees a
    truncated file.

    Never overwrites a file this process has not read: `_STATS_LOADED` False means the on-disk
    history was never merged into `_STATS`, so writing would replace a recorded history with
    nothing. An absent file is not
    this case — `_load_stats_once` marks itself loaded either way, so a fresh install still
    writes its first one."""
    p = _stats_file()
    if not _STATS_LOADED:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with _STATS_LOCK:
        snapshot = json.dumps({"pricing_version": PRICING_TABLE_VERSION, **_STATS}, indent=2)
    tmp = p.parent / (p.name + ".tmp")
    tmp.write_text(snapshot)
    os.replace(tmp, p)
    return p


atexit.register(flush_stats)


def cost_for(model: str, in_tok: int, out_tok: int) -> float:
    """Estimate from the static rate card. For OpenRouter prefer `_reported_cost` — see the
    _PRICING header for why a per-model table cannot be right once routing selects per request."""
    price = _PRICING.get(model)
    if price is None:
        if model not in _UNPRICED_LOGGED:
            _UNPRICED_LOGGED.add(model)
            log(f"[spend] no price row for {model!r} and the provider reported no charge — "
                f"this call records as $0. Add it to _PRICING from OpenRouter's catalog.")
        return 0.0
    pin, pout = price
    return (in_tok * pin + out_tok * pout) / 1_000_000


def _reported_cost(raw: dict | None) -> float | None:
    """What the provider says it actually CHARGED, or None to fall back to the price table.

    OpenRouter returns `usage.cost` in USD (verified against its own rate card, see
    applies. A zero is treated as UNREPORTED, not free — under BYOK, OpenRouter charges nothing
    while the real money is spent upstream, so 0.0 would silently under-report."""
    try:
        cost = float(((raw or {}).get("usage") or {}).get("cost") or 0.0)
    except (TypeError, ValueError, AttributeError):
        return None
    return cost or None


def spend_total() -> float:
    """The cumulative $ spend recorded in api_stats.json — LLM (OpenRouter) AND external
    providers (the hosted embedder) folded into one lifetime total. The figure the radar
    paid-sweep cap enforces against. A fresh install with no stats file reads 0.0."""
    _load_stats_once()
    with _STATS_LOCK:
        return float(_STATS["lifetime"]["cost_usd"])


def spend_today() -> float:
    """TODAY's recorded $ spend (UTC day), LLM + external providers, from the SAME in-memory
    `_STATS` the writers bump — NOT from re-reading api_stats.json, because `flush_stats` runs at
    `atexit` and a long-lived process's on-disk file is stale by exactly the spend it just made.
    A daily ceiling that read the file would be blind to the runaway it exists to stop.

    It is the RECORDED figure, not an invoice. Everything in it is now metered per call by a
    provider that reports its own cost, so the two agree; it carried a flat per-REQUEST estimate
    for twitterapi.io until that provider was removed on 2026-08-30. If an estimated provider is
    ever added back, conservative-HIGH is the right direction for a ceiling."""
    _load_stats_once()
    today = datetime.now(timezone.utc).date().isoformat()
    with _STATS_LOCK:
        return float((_STATS["by_date"].get(today) or {}).get("cost_usd", 0.0))


def spend_today_for_rail(rail_name: str) -> float:
    """TODAY's recorded $ spend (UTC day) for ONE rail — the figure that rail's daily ceiling
    reads. A SIBLING of `spend_today()`, not a replacement — that one keeps its total-spend
    meaning since it has readers beyond the gates. Same in-memory-`_STATS` reasoning applies.
    A rail that has spent nothing today (or an unknown rail name) reads 0.0 — no missing meter,
    only an empty one."""
    _load_stats_once()
    today = datetime.now(timezone.utc).date().isoformat()
    with _STATS_LOCK:
        row = _STATS["by_rail"].get(f"{rail_name}|{today}") or {}
        return float(row.get("cost_usd", 0.0))


def spend_today_by_rail() -> dict[str, float]:
    """TODAY's recorded spend for EVERY rail, merged across processes — the REPORTING reader.
    Never gate on this; a gate uses `spend_today_for_rail(RAIL)` instead, because a gate runs
    inside the spending process (in-memory `_STATS` is the only honest meter there) while this
    runs in the long-lived MCP server, which isn't the process that spent. So it merges the
    on-disk file (carries background rails' flushed spend) with in-memory `_STATS` (carries
    in-process, unflushed spend) and takes the larger per rail
    Fail-safe: an absent or unreadable file contributes nothing."""
    today = datetime.now(timezone.utc).date().isoformat()
    suffix = f"|{today}"

    def _today_rows(buckets: Any) -> dict[str, float]:
        if not isinstance(buckets, dict):
            return {}
        return {k[: -len(suffix)]: float((v or {}).get("cost_usd", 0.0))
                for k, v in buckets.items() if isinstance(k, str) and k.endswith(suffix)}

    on_disk: dict[str, float] = {}
    try:
        on_disk = _today_rows(json.loads(_stats_file().read_text()).get("by_rail"))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass

    _load_stats_once()
    with _STATS_LOCK:
        in_memory = _today_rows(_STATS["by_rail"])

    return {rail_name: max(on_disk.get(rail_name, 0.0), in_memory.get(rail_name, 0.0))
            for rail_name in set(on_disk) | set(in_memory)}


def record_external_cost(provider: str, cost_usd: float, *, requests: int = 1) -> None:
    """Attribute NON-LLM paid spend (e.g. the hosted embedder) into the SAME
    api_stats.json the LLM router writes, so a single cumulative $ total (`lifetime.cost_usd`)
    covers every paid provider — the figure the radar paid-sweep $-cap reads. Providers billed
    per REQUEST (no tokens) get a request count + cost under a `by_provider` bucket; the cost
    also folds into the lifetime total + today's by_date row so existing spend readers stay
    honest. Shares the stats lock + atexit flush with the LLM path (no separate file to race).
    Fail-safe: a non-positive cost is a no-op."""
    if cost_usd <= 0:
        return
    _load_stats_once()
    today = datetime.now(timezone.utc).date().isoformat()
    with _STATS_LOCK:
        _STATS["lifetime"]["cost_usd"] = round(_STATS["lifetime"]["cost_usd"] + cost_usd, 6)
        prov = _STATS["by_provider"].setdefault(provider, {"requests": 0, "cost_usd": 0.0})
        prov["requests"] += requests
        prov["cost_usd"] = round(prov["cost_usd"] + cost_usd, 6)
        day = _STATS["by_date"].setdefault(
            today, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        day["cost_usd"] = round(day["cost_usd"] + cost_usd, 6)
        # Same row shape as by_date, cost-only since an external provider bills per REQUEST.
        # `_CURRENT_RAIL` needs no lock (atomic under the GIL) — `_STATS_LOCK` is already held.
        r = _STATS["by_rail"].setdefault(
            _rail_key(today), {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        r["cost_usd"] = round(r["cost_usd"] + cost_usd, 6)


# ── Test seam ────────────────────────────────────────────────────────────────

def _override_stats_file_for_tests(path: Path | None) -> None:
    """Tests can redirect api_stats.json to a temp dir."""
    global _STATS_FILE_OVERRIDE, _STATS_LOADED, _STATS
    _STATS_FILE_OVERRIDE = path
    _STATS_LOADED = False
    _STATS = _fresh_stats()      # one shape, so a new bucket can't drift from the test seam
