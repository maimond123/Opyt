"""
opyt_core/config.py

Resolves config/settings.yaml — LLM role models, taxonomy path, cookie browser, repo root.

Resolution order: explicit $OPYT_CONFIG, then user-local ~/.opyt/settings.yaml (written by
first-run bootstrap), then the repo's packaged example (config/settings.example.yaml).

Load-bearing reader: `pipeline/model_routing.py`, which resolves LLM role models from here.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml

from .paths import opyt_home

REPO_ROOT = Path(__file__).resolve().parent.parent
# Shipped template (generic placeholders).
_REPO_CONFIG = REPO_ROOT / "config" / "settings.example.yaml"


def config_path() -> Path:
    """The active settings.yaml, in the resolution order above."""
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
# Upstreams measured broken (DeepInfra slow, Cloudflare silently empty but billed). Denied via
# `ignore`, not `order`, which can still fall back onto them. Override with
# `openrouter.deny_upstreams` (`[]` disables).
_DENY_UPSTREAMS = ("DeepInfra", "Cloudflare")

# ── Latency-sorted upstream selection ────────────────────────────────────────────────────
# By default OpenRouter picks the cheapest upstream, which has a long latency tail. "latency"
# keeps most of throughput's speed at about half its cost, with the tightest tail. Override with
# `openrouter.sort` ("throughput"/"price"; null disables).
_SORT_DEFAULT = "latency"


def _resolve_routing() -> dict:
    """Read the OpenRouter routing policy from settings.yaml in one pass.

    Fail-safe: an unreadable or absent config keeps the built-in defaults instead of dropping
    them."""
    try:
        cfg = settings().get("openrouter") or {}
    except Exception:
        cfg = {}
    # `in`, not truthiness: an explicit empty list / null means disable.
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


# Resolved once per process: `merge_provider_routing` is on the hot path of every LLM call. A
# routing edit takes effect on the next run.
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

    Returns a copy so a caller mutating the result cannot poison the cache."""
    return list(_routing_policy()["deny"])


def merge_provider_routing(base: dict | None = None) -> dict:
    """Fold the routing policy (deny-list + sort) into an OpenRouter `provider` preferences
    block, preserving whatever the caller already set (chat's `require_parameters`, an explicit
    `order`). Every OpenRouter call routes through this.

    The sort yields to an explicit `order`: a caller that pins upstreams in order means it."""
    out = dict(base or {})
    policy = _routing_policy()
    if deny := policy["deny"]:
        existing = list(out.get("ignore") or [])
        out["ignore"] = existing + [p for p in deny if p not in existing]
    if (sort := policy["sort"]) and "sort" not in out and not out.get("order"):
        out["sort"] = sort
    return out

