"""
pipeline/model_routing.py — the ONE home for model slugs + the provider-availability preflight.

"Is this model served by a provider I allow" is a fact about OpenRouter's live catalog crossed
with the active deny-list, so it can't be caught statically. The invariant is enforced in three
layers, two of them here:

  1. AST guard (.guards.py `bare-model-slug`) — a model slug may only be a literal HERE, so the
     registry below is complete by construction and preflight cannot be blind to a new override.
  2. Preflight (this module) — before a rail spends anything, every registered model is checked
     for surviving providers under the active deny-list (rail_runtime.models_unroutable). A dead
     model with no live fallback blocks the rail; fragile/unknown warn in the rail log and on the
     `oracle` screen's `model_routing` notice. Fails LOUD, before the spend.
  3. Runtime (llm_client.ModelUnroutableError) — a "no providers" 404 is TERMINAL, not transient:
     never retried, never counted against the shared breaker.

Selection criterion is provider redundancy first, price second. See
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from opyt_core.config import openrouter_deny_upstreams, settings
from opyt_core.paths import opyt_home

# ── the registry: EVERY code-level model override lives here ──────────────────────────────────
# Role models (the `vision`/`content_quality`/... entries in settings.yaml) are config, not
# literals, and are enumerated separately by `registered_models()` so preflight covers both.
#
# OCR_MODEL: served by four independent orgs (Parasail, Nebius, Novita, Phala) under the deny-list.
OCR_MODEL = "google/gemma-3-27b-it"
CHART_MODEL = "google/gemini-2.5-flash"

# The atom-KB embedder's code default (overridable via `embeddings:` in settings.yaml). Registered
# here because it is the most load-bearing model in the system: an unroutable embedder blocks
# every atom, and the subspace invariant means it can't be swapped without a full re-embed. Only
# two orgs survive the deny-list (Nebius, SiliconFlow) — see doc for the measurement.
EMBED_MODEL = "qwen/qwen3-embedding-8b"

# Declared FALLBACKS, in order. `resolve_ocr_model` walks these and picks the first with any
# surviving org; the substitution is always REPORTED, never inferred. `nova-lite-v1` is
# Amazon-Bedrock-only, hence a fallback and not the primary. See doc for why other candidates (qwen3-vl-8b, gemini-2.5-flash-lite,
# llama-4-scout) are deliberately excluded.
OCR_FALLBACKS = (OCR_MODEL, "amazon/nova-lite-v1")

MIN_ORGS = 2          # below this a model is one withdrawal from dead — warn, don't fail
_CACHE_TTL = 86400    # 24h; the cache key includes the deny-list, so editing it re-checks at once
_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


def registered_models() -> dict[str, str]:
    """Every model this install may call → why it is called. Code overrides (above) PLUS the role
    models from the active settings.yaml, because a role model can go unroutable exactly the same
    way and is just as load-bearing."""
    out = {OCR_MODEL: "OCR (code override)", CHART_MODEL: "chart read (code override)",
           EMBED_MODEL: "atom-KB embeddings"}
    for m in OCR_FALLBACKS:
        out.setdefault(m, "OCR fallback")
    # `settings` is imported at module level on purpose, so a wrong name fails loudly at import
    # time instead of being swallowed by a broad except below. See doc for the incident.
    try:
        cfg = settings() or {}
    except (OSError, ValueError):     # missing / unparseable settings.yaml — degrade, never block
        return out
    roles = ((cfg.get("llm_backends") or {}).get("roles") or {})
    for role, spec in (roles or {}).items():
        if (spec or {}).get("provider") == "openrouter" and (spec or {}).get("model"):
            out.setdefault(str(spec["model"]), f"role:{role}")
    return out


def _org_of(provider_name: str) -> str:
    """A provider display name → the ORGANISATION behind it.

    Counting endpoints overstates redundancy (e.g. `gemini-2.5-flash-lite` reports five endpoints
    from one company). Heuristic: first word, lowered ('Google AI Studio'→google). Can only
    under-count redundancy, which fails toward warning — the safe direction."""
    return (provider_name or "").strip().split()[0].lower() if provider_name else ""


def _cache_path():
    return opyt_home() / "model_routing_cache.json"


def _cache_key(model: str, deny: list[str]) -> str:
    # The deny-list is part of the key so a deny-list edit invalidates stale cached answers.
    return f"{model}|{','.join(sorted(deny))}"


def _cache_read(key: str):
    try:
        blob = json.loads(_cache_path().read_text())
        hit = blob.get(key)
        if hit and (time.time() - hit["at"]) < _CACHE_TTL:
            return hit["orgs"]
    except Exception:
        pass
    return None


def _cache_write(key: str, orgs: list[str]) -> None:
    try:
        p = _cache_path()
        blob = {}
        if p.exists():
            try:
                blob = json.loads(p.read_text())
            except Exception:
                blob = {}
        blob[key] = {"at": time.time(), "orgs": orgs}
        p.write_text(json.dumps(blob, indent=2))
    except Exception:
        pass          # a cache that cannot be written must never break a run


def surviving_orgs(model: str, deny: list[str] | None = None, *,
                   api_key: str | None = None, use_cache: bool = True,
                   fetch: bool = True) -> list[str] | None:
    """Distinct ORGS serving `model` after the deny-list is applied.

    Returns None when availability could not be determined (no key, network error, unparseable
    response). None is distinct from `[]`: `[]` means nothing can serve this model (block), None
    means unknown (warn, proceed) — collapsing them would let a flaky network abort onboarding.

    `fetch=False` answers from the cache alone (miss → None). For interactive surfaces like the
    `oracle` screen, which must never pay a catalog round-trip inside a tool call — the rail
    preflight is what populates the cache."""
    import os
    if deny is None:
        deny = openrouter_deny_upstreams()
    key = _cache_key(model, deny)
    if use_cache:
        cached = _cache_read(key)
        if cached is not None:
            return cached
    if not fetch:
        return None
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        req = urllib.request.Request(_ENDPOINTS_URL.format(model=model),
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        eps = (data.get("data") or {}).get("endpoints") or []
    except Exception:
        return None
    denyset = {d.strip().lower() for d in deny}
    orgs = sorted({_org_of(e.get("provider_name")) for e in eps
                   if e.get("provider_name") and e["provider_name"].strip().lower() not in denyset
                   and _org_of(e.get("provider_name")) not in denyset})
    _cache_write(key, orgs)
    return orgs


def preflight(models: dict[str, str] | None = None, *, deny: list[str] | None = None,
              fetch: bool = True) -> dict:
    """Check every registered model BEFORE a run spends anything.

    Returns {"ok": bool, "dead": [...], "fragile": [...], "unknown": [...], "checked": {...}}.
    `dead` (zero surviving orgs) flips `ok` — that stage cannot function. `fragile` (one org) and
    `unknown` warn but never block. One excusable dead case: an OCR-family model while
    `resolve_ocr_model` still finds a survivor — the cascade substitutes and reports, so flipping
    `ok` for it would block every rail on a failure the fallback chain exists to absorb.
    """
    if deny is None:
        deny = openrouter_deny_upstreams()
    models = models or registered_models()
    dead, fragile, unknown, checked = [], [], [], {}
    for model, why in models.items():
        orgs = surviving_orgs(model, deny, fetch=fetch)
        checked[model] = orgs
        if orgs is None:
            unknown.append((model, why))
        elif len(orgs) == 0:
            dead.append((model, why))
        elif len(orgs) < MIN_ORGS:
            fragile.append((model, why, orgs))
    blocking = [d for d in dead if d[0] not in OCR_FALLBACKS]
    if len(blocking) < len(dead) and resolve_ocr_model(deny=deny, fetch=fetch)[0] is None:
        blocking = dead          # the whole OCR chain is dead — nothing excusable about it
    return {"ok": not blocking, "dead": dead, "fragile": fragile,
            "unknown": unknown, "checked": checked, "deny": list(deny)}


def resolve_ocr_model(*, deny: list[str] | None = None,
                      fetch: bool = True) -> tuple[str | None, str]:
    """The OCR model to actually use, walking `OCR_FALLBACKS` in order.

    Returns (model, reason). `model` is None only when every declared candidate is definitively
    dead. An `unknown` verdict (network/no key) resolves to the primary: preflight must never be
    the reason a run cannot start."""
    for i, m in enumerate(OCR_FALLBACKS):
        orgs = surviving_orgs(m, deny, fetch=fetch)
        if orgs is None:
            return m, f"{m} (availability unknown — proceeding with the primary)"
        if len(orgs) >= 1:
            if i == 0:
                return m, f"{m} ({len(orgs)} orgs)"
            # A substitution is always REPORTED. Never inferred, never silent.
            return m, f"FALLBACK {m} ({len(orgs)} orgs) — primary {OCR_FALLBACKS[0]} is unroutable"
    return None, (f"no declared OCR model is routable under deny={deny} — "
                  f"tried {list(OCR_FALLBACKS)}")


def format_report(rep: dict) -> str:
    """Human-readable preflight summary — what a CLI/onboarding surface prints."""
    lines = [f"[preflight] deny-list: {rep['deny'] or '(none)'}"]
    for model, why in rep["dead"]:
        lines.append(f"  DEAD     {model}  ({why}) — NO provider survives the deny-list")
    for model, why, orgs in rep["fragile"]:
        lines.append(f"  FRAGILE  {model}  ({why}) — only {orgs}; one withdrawal from dead")
    for model, why in rep["unknown"]:
        lines.append(f"  unknown  {model}  ({why}) — could not verify (no key / network)")
    if rep["ok"] and not rep["fragile"] and not rep["unknown"]:
        lines.append(f"  all {len(rep['checked'])} registered models OK")
    return "\n".join(lines)
