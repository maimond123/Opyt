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
# `openrouter.deny_upstreams` (including `[]` to disable). Full measurements:
_DENY_UPSTREAMS = ("DeepInfra", "Cloudflare")

# ── Throughput-sorted upstream selection ─────────────────────────────────────────────────
# The deny-list rules out what's known-broken but by default OpenRouter still picks the cheapest
# upstream, which produces a long latency tail. Sorting by "throughput" measured p50 4.28s->0.79s
# at ~4.5x the $/M cost — worth it, cleared for quality first. Default in code; override with
# `openrouter.sort` ("latency"/"price" are the other policies, null disables). Full measurements
# and quality check:
_SORT_DEFAULT = "throughput"


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
# every other settings.yaml value. Why per-request re-reads broke a test:
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


def _reset_routing_cache_for_tests() -> None:
    """Drop the memo so a test can monkeypatch `settings` and see the effect."""
    global _routing_cache
    _routing_cache = None


def merge_provider_routing(base: dict | None = None) -> dict:
    """Fold the routing policy (deny-list + throughput sort) into an OpenRouter `provider`
    preferences block, preserving whatever the caller already set. Both OpenRouter surfaces (chat
    in `llm_client`, embeddings in `kb.embed`) route through this so there is one policy, not two
    that drift.

    Merges rather than overwrites: an existing `ignore` is unioned with the deny-list, and the
    sort yields to an explicit `order` from the caller. Full rationale (why merging vs. replacing
    matters here):"""
    out = dict(base or {})
    policy = _routing_policy()
    if deny := policy["deny"]:
        existing = list(out.get("ignore") or [])
        out["ignore"] = existing + [p for p in deny if p not in existing]
    if (sort := policy["sort"]) and "sort" not in out and not out.get("order"):
        out["sort"] = sort
    return out


# Inert keys in settings.yaml that must not be stripped: `credible_people.profiles` holds 28
# hand-curated X handles, unread by code (the roster now lives in the `oracles` table), but only
# 8 of the 28 exist there — this key is the only record of the other 20.


def service_url() -> str | None:
    """The service `opyt-push` uploads this install's export to (settings.yaml `service_url`).

    CONFIG, NOT A CREDENTIAL, and the split is deliberate: the address of the host an owner
    publishes to is not secret and belongs beside the rest of their settings, while the token that
    proves they may publish there lives in ~/.opyt/.env like every other credential. Nothing
    reads this but `push`; a READER never sets it, because their peer row already carries the
    full URL the grant code resolved to.

    Fail-safe: an unreadable config is a missing key, and `push` says which key is missing and
    which file to put it in."""
    try:
        val = settings().get("service_url")
    except Exception:
        return None
    return val.strip() if isinstance(val, str) and val.strip() else None


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
    """Which browser PROFILE to read local-session cookies from. $X_CHROME_PROFILE overrides
    settings.yaml `cookies.profile`; unset means auto-pick the lone candidate. `onboard` writes
    the user's pick here. Best-effort: never raises. History:
"""
    val = os.environ.get("X_CHROME_PROFILE")
    if not val:
        try:
            val = (settings().get("cookies") or {}).get("profile")
        except Exception:
            val = None
    return val.strip() if isinstance(val, str) and val.strip() else None
