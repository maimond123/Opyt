"""
opyt_core/config.py

Resolves config/settings.yaml — LLM role models, the `embeddings:` block, taxonomy path, cookie
browser, repo root.

Resolution order: explicit $OPYT_CONFIG, then user-local ~/.opyt/settings.yaml (written by
first-run bootstrap), then the repo's packaged example (config/settings.example.yaml). The
author's real config/settings.yaml is gitignored and never ships.

Load-bearing reader: `pipeline/model_routing.py`, which resolves LLM role models and the
embedder's model id from here. Losing this file un-configures the embedder.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml

# Re-exported from paths.py (the sandbox seam) so callers keep using config.opyt_home() with one
# source of truth for the OPYT data home (~/.opyt, $OPYT_HOME-overridable).
from .paths import opyt_home

REPO_ROOT = Path(__file__).resolve().parent.parent
# Shipped template (generic placeholders); fallback when neither $OPYT_CONFIG nor
# ~/.opyt/settings.yaml exists.
_REPO_CONFIG = REPO_ROOT / "config" / "settings.example.yaml"


def config_path() -> Path:
    """The ACTIVE settings.yaml. First existing wins: $OPYT_CONFIG → user-local
    ~/.opyt/settings.yaml (written by bootstrap) → repo/packaged default."""
    if env := os.environ.get("OPYT_CONFIG"):
        return Path(env).expanduser()
    user = opyt_home() / "settings.yaml"
    if user.exists():
        return user
    return _REPO_CONFIG


def settings() -> dict:
    with open(config_path()) as f:
        return yaml.safe_load(f) or {}


# ── OpenRouter upstream deny-list ────────────────────────────────────────────────────────
# OpenRouter is a broker: the same model id is served by multiple upstream providers, chosen per
# request, and they are not interchangeable — DeepInfra and Cloudflare were measured
# reproducibly broken (slow, and silently empty-but-billed, respectively). Denied via `ignore`,
# not `order` (an `order` list with fallbacks can still land on a denied upstream). This is the
# default in code because settings.yaml is untracked and would not ship; override with
# `openrouter.deny_upstreams` (including `[]` to disable).
#
# MEASURED. DeepInfra, slow: 2026-07-22 qwen3-embedding-8b 71–93 s on 5/5 calls; 2026-07-29
# llama-3.3-70b 8.4–23.7 s on 10/10, while every other upstream answered the identical prompt
# under 2.9 s. Cloudflare, silently empty: 2026-07-29, 8/8 calls returned
# `finish_reason: "tool_calls"` with `content == ""` while BILLING 70 completion tokens — the
# batch then degrades to keep-all, so ungraded boilerplate enters the KB, billed, with no error
# raised anywhere. `ignore` and NOT `order`: the 2026-07-22 embed fix used `order` alone and its
# comment claimed DeepInfra was "excluded"; it was not.
_DENY_UPSTREAMS = ("DeepInfra", "Cloudflare")

# ── Latency-sorted upstream selection ────────────────────────────────────────────────────
# The deny-list rules out what's known-broken but by default OpenRouter still picks the cheapest
# upstream, which produces a long latency tail. Sorting fixes the tail; WHICH sort is a measured
# trade. Default in code; override with `openrouter.sort` ("throughput"/"price" are the other
# policies, null disables).
#
# "latency", not "throughput", since 2026-09-15 — re-measured when the classify roles moved to
# gpt-oss-120b. Three arms x 6 calls interleaved, triage-size and content_quality-size payloads:
# sort:"price" p50 14.60 s on the big payload (the cheap hosts crawl exactly where 63–97% of
# ingest wall-clock lives) · sort:"throughput" p50 1.19 s at ~5.5x price-arm cost, spread 13.7x ·
# sort:"latency" p50 1.45 s at ~2.6x price-arm cost, spread 1.4x. Latency keeps ~95% of the
# speed at half throughput's cost, with the tightest tail of the three — batches finish
# together instead of waiting on one straggler. See
# docs/plans/2026-09-15-cheaper-classify-models-and-latency-sort.md for the full table.
#
# HISTORY (kept because it explains why a sort exists at all): the original three-arm A/B on
# 2026-07-31, llama-3.3-70b — baseline p50 4.28 s / spread 10.83x · sort:"throughput" p50 0.79 s /
# spread 1.63x · ignore:[DeepInfra] p50 2.28 s / spread 5.39x. Banning DeepInfra alone just moved
# the tail to AkashML; slowness was a property of not selecting on speed at all. Quality cleared
# first: 4 upstreams x 3 rounds x 2 pages, cross-upstream disagreement 1-in-17 and 1-in-26
# against within-upstream ~0.
_SORT_DEFAULT = "latency"


def _resolve_routing() -> dict:
    """Read the OpenRouter routing policy from settings.yaml in one pass.

    Fail-safe: an unreadable or absent config keeps the built-in defaults instead of dropping
    them."""
    try:
        cfg = settings().get("openrouter") or {}
    except Exception:
        cfg = {}
    # `in` rather than truthiness — an explicit empty list / null is a real choice (disable), not a
    # missing key, and must not fall through to the default.
    if "deny_upstreams" in cfg:
        deny = [str(p) for p in (cfg.get("deny_upstreams") or [])]
    else:
        deny = list(_DENY_UPSTREAMS)
    if "sort" in cfg:
        raw = cfg.get("sort")
        sort = str(raw) if raw else None
    else:
        sort = _SORT_DEFAULT
    return {"deny": deny, "sort": sort}


# Resolved once per process, not per request — `merge_provider_routing` runs in the hot path of
# every LLM call and embed batch. A routing edit takes effect on the next run, same contract as
# every other settings.yaml value.
#
# Per-request re-reads did not merely cost I/O, they broke a test: resolving through `settings()`
# each time meant an open() + yaml.safe_load() per request, and THAT I/O RELEASES THE GIL
# mid-batch, reshuffling the completion order of `embed`'s concurrent slices. It turned a latent
# order-dependence in `tests/kb/test_embed.py` into a 4-in-5 failure.
_routing_cache: dict | None = None
_routing_lock = threading.Lock()


def _routing_policy() -> dict:
    global _routing_cache
    if _routing_cache is None:
        with _routing_lock:
            if _routing_cache is None:               # re-check: another thread may have filled it
                _routing_cache = _resolve_routing()
    return _routing_cache


def openrouter_deny_upstreams() -> list[str]:
    """Upstream providers OpenRouter must never route to. settings.yaml
    `openrouter.deny_upstreams` overrides the built-in default (an explicit `[]` disables it).

    Memoized (see above). Returns a COPY so a caller mutating the result cannot poison the cache."""
    return list(_routing_policy()["deny"])


def merge_provider_routing(base: dict | None = None) -> dict:
    """Fold the routing policy (deny-list + throughput sort) into an OpenRouter `provider`
    preferences block, preserving whatever the caller already set. Both OpenRouter surfaces (chat
    in `llm_client`, embeddings in `kb.embed`) route through this so there is one policy, not two
    that drift.

    Merges rather than overwrites because both callers already set something: the chat path sets
    `require_parameters` for JSON-mode roles and the embed path sets an `order`. A fresh dict
    would drop either — dropping `require_parameters` un-guarantees JSON mode and quietly
    reintroduces parse failures.

    The sort YIELDS to an explicit `order`, and that is not politeness about precedence
    (OpenRouter's own rule for order+sort together is undocumented). An `order` states a
    preference the sort has no view on: the embed path's `order: [Nebius, SiliconFlow]` picks
    Nebius because SiliconFlow serves fp8 — a lossier numeric format — at 4x the price, so
    letting throughput override it would write quantized vectors into an index of
    full-precision ones and degrade similarity with no error anywhere. NOT contradicted by
    `allow_fallbacks: True` already being able to reach SiliconFlow: a fallback fires only when
    Nebius is DOWN, where the alternative is no vector at all, while a sort would pick it while
    Nebius is healthy. Same provider, opposite trade."""
    out = dict(base or {})
    policy = _routing_policy()
    if deny := policy["deny"]:
        existing = list(out.get("ignore") or [])
        out["ignore"] = existing + [p for p in deny if p not in existing]
    if (sort := policy["sort"]) and "sort" not in out and not out.get("order"):
        out["sort"] = sort
    return out


# Inert keys in settings.yaml that must not be stripped: `credible_people.profiles` holds 28
# hand-curated X handles, and the live roster is the `oracles` table. Re-measured 2026-09-06:
# THREE of the 28 have an oracles row, not the 8 this said, so the key is the only record of the
# other 25. MEASURE IT THROUGH A JOIN, not against `oracles.canonical_id`: an X canonical id is
# `x:user:<numeric>`, so a handle never matches one directly and a naive re-count returns 0.
# `oracle_sources.source_key` for the `x` pairs and `entities.profile.handle` on the canonical
# entity both give 3, independently.
#
# "Unread by code" was the previous wording and it was never true. `bootstrap.py:37-39` reads the
# key and BLANKS it when writing a fresh user's settings.yaml, so the author's list is not
# inherited — the one operation this sentence said nothing does. `50408d48` said so three hours
# after the sentence was written and did not come back to it. Nothing SELECTS these handles as a
# roster, which is the claim that was meant; say that instead.


# Where an owner publishes when they have not said otherwise. A DEFAULT and not a constant:
# `settings.yaml`'s `service_url` still wins, which is what keeps a self-hosted service and the
# test suite possible. It exists because sharing must not begin with editing a config file — the
# address of the one hosted service is not a decision anybody wants to make.
DEFAULT_SERVICE_URL = "https://api.useopyt.com"


def service_url() -> str:
    """The service this install publishes its export to — `settings.yaml`'s `service_url`, or
    `DEFAULT_SERVICE_URL`.

    CONFIG, NOT A CREDENTIAL, and the split is deliberate: the address of the host an owner
    publishes to is not secret and belongs beside the rest of their settings, while the token that
    proves they may publish there lives in ~/.opyt/.env like every other credential.

    ⚠️A READER READS THIS TOO, which is easy to miss because the name says publishes.
    `share_tools.accept` uses it for a bare pasted code, which names no issuing service. A full
    `useopyt.com` invite maps to `DEFAULT_SERVICE_URL`; every other explicit host redeems at its
    own origin. So this value decides where bare-code redemption goes and what URL lands in that
    peer row.

    Fail-safe: an unreadable config is the default, never an error. Publishing is now something a
    person asks for in chat, so there is nowhere to report "your config file is malformed" that
    would not be worse than publishing to the address they meant anyway."""
    try:
        val = settings().get("service_url")
    except Exception:
        return DEFAULT_SERVICE_URL
    if isinstance(val, str) and val.strip():
        return val.strip()
    return DEFAULT_SERVICE_URL


def cookie_browser() -> str | None:
    """Which browser to read local-session cookies from (X/Claude/Substack scrapes).
    $OPYT_BROWSER overrides settings.yaml `cookies.browser`; 'auto'/unset → None, which
    means auto-detect (try every installed browser in priority order). Best-effort:
    never raises — a missing/broken config degrades to auto."""
    val = os.environ.get("OPYT_BROWSER")
    if not val:
        try:
            val = (settings().get("cookies") or {}).get("browser")
        except Exception:
            val = None
    if isinstance(val, str) and val.strip() and val.strip().lower() != "auto":
        return val.strip().lower()
    return None


def cookie_profile() -> str | None:
    """Optional generic browser profile from settings.yaml `cookies.profile`.

    Unset lets a generic cookie reader auto-pick a lone logged-in profile. X does not read
    this setting: it uses its OPYT-managed profile. Best-effort: never raises.
    """
    try:
        val = (settings().get("cookies") or {}).get("profile")
    except Exception:
        val = None
    return val.strip() if isinstance(val, str) and val.strip() else None
