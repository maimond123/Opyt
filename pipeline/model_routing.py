"""
pipeline/model_routing.py — the one home for model slugs, plus the provider-availability preflight.

"Is this model served by a provider I allow" is a fact about OpenRouter's live catalog crossed
with the active deny-list, so it can't be caught statically. The invariant is enforced in three
layers, two of them here:

  1. AST guard (.guards.py `bare-model-slug`) — a model slug may only be a literal here, so the
     registry below is complete by construction.
  2. Preflight (this module) — before a rail spends anything, every registered model is checked
     for surviving providers under the active deny-list (`models_unroutable`). A dead model with
     no live fallback blocks the rail; fragile/unknown warn.
  3. Runtime (llm_client.ModelUnroutableError) — a "no providers" 404 is terminal: never retried,
     never counted against the shared breaker.

Selection criterion is provider redundancy first, price second.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from opyt_core.config import openrouter_deny_upstreams, settings
from opyt_core.paths import opyt_home
from pipeline.credentials import model_access
from pipeline.ingestion import utils

# ── the registry: every code-level model override lives here ──────────────────────────────────
# Role models (the `vision` entry in settings.yaml) are config, not literals, and are enumerated
# separately by `registered_models()` so preflight covers both.
#
# CHART_MODEL: the one image model — reads a chart or answers NOT_CHART. Served by twelve
# independent orgs at the time of writing (DeepInfra, CoreWeave, Venice, Chutes, Crusoe, Friendli, Novita,
# Parasail, Io Net, SambaNova, ModelRun, SiliconFlow), so no fallback is declared.
CHART_MODEL = "google/gemma-4-31b-it"

# The roles the code calls, one per `llm_backends.roles` entry it reads. A role left in an older
# settings.yaml after the code stopped calling it is not here, so it is never preflighted.
ROLES = ("vision",)

MIN_ORGS = 2          # below this a model is one withdrawal from dead — warn, don't fail
_CACHE_TTL = 86400    # 24h; the cache key includes the deny-list, so editing it re-checks at once
_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


def registered_models() -> dict[str, str]:
    """Every model this install may call → why it is called: code overrides plus the role models
    from the active settings.yaml. Only roles the code calls (`ROLES`) count, so a
    role deleted from the code but left in an older settings.yaml is never preflighted."""
    out = {CHART_MODEL: "chart read (code override)"}
    # `settings` is imported at module level on purpose, so a wrong name fails at import time
    # instead of being swallowed by the except below.
    try:
        cfg = settings() or {}
    except (OSError, ValueError):     # missing / unparseable settings.yaml — degrade, never block
        return out
    roles = ((cfg.get("llm_backends") or {}).get("roles") or {})
    for role, spec in (roles or {}).items():
        if role not in ROLES:
            continue
        if (spec or {}).get("provider") == "openrouter" and (spec or {}).get("model"):
            out.setdefault(str(spec["model"]), f"role:{role}")
    return out


def _org_of(provider_name: str) -> str:
    """A provider display name → the organisation behind it: first word, lowered
    ('Google AI Studio'→google). Counting orgs, not endpoints, avoids overstating redundancy."""
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
    """Distinct orgs serving `model` after the deny-list is applied.

    Returns None when availability could not be determined (no key, network error, unparseable
    response). `[]` means nothing can serve this model (block); None means unknown (warn, proceed).

    `fetch=False` answers from the cache alone (miss → None), for tool calls that must not pay a
    catalog round-trip; the rail preflight populates the cache."""
    if deny is None:
        deny = openrouter_deny_upstreams()
    key = _cache_key(model, deny)
    if use_cache:
        cached = _cache_read(key)
        if cached is not None:
            return cached
    if not fetch:
        return None
    if not api_key:
        from pipeline.credentials import get_credential
        api_key = get_credential("openrouter")
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
    """Check every registered model before a run spends anything.

    Returns {"ok": bool, "dead": [...], "fragile": [...], "unknown": [...], "checked": {...}}.
    `dead` (zero surviving orgs) flips `ok`; `fragile` (one org) and `unknown` never block.
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
    return {"ok": not dead, "dead": dead, "fragile": fragile,
            "unknown": unknown, "checked": checked, "deny": list(deny)}


def routing_notice() -> dict | None:
    """What the oracle screen says about model routability, or None when nothing is wrong.

    Cache-only (`fetch=False`), so a tool call never pays catalog round-trips. `unknown` is
    suppressed: here it is cache-miss noise."""
    rep = preflight(fetch=False)
    notice = {}
    if rep["dead"]:
        notice["dead"] = [f"{m} ({why}) — NO provider survives the deny-list"
                          for m, why in rep["dead"]]
    if rep["fragile"]:
        notice["fragile"] = [f"{m} ({why}) — only {orgs}; one withdrawal from dead"
                             for m, why, orgs in rep["fragile"]]
    return notice or None


def chart_model_unroutable(*, deny: list[str] | None = None, fetch: bool = True) -> str | None:
    """Why `CHART_MODEL` cannot be served under the deny-list, or None when it can. An `unknown`
    verdict (network / no key) is None too: preflight must never be the reason a run cannot start."""
    if deny is None:
        deny = openrouter_deny_upstreams()
    if surviving_orgs(CHART_MODEL, deny, fetch=fetch) == []:
        return f"{CHART_MODEL} has no provider under deny={deny}"
    return None


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


def models_unroutable(rail: str) -> str | None:
    """The preflight at a job's spend boundary — the reason this pass must skip, or None to
    proceed. Asked in the job's own process after its keys load (the catalog fetch needs the key).

    Blocks only on `ok: False`; `fragile` and `unknown` log and proceed, and a preflight that
    throws proceeds. A free install (no key) spends nothing, so there is nothing to check."""
    if not model_access():
        return None
    try:
        rep = preflight()
        if rep["dead"] or rep["fragile"] or rep["unknown"]:
            utils.log(f"[{rail}] {format_report(rep)}")
        if not rep["ok"]:
            dead = ", ".join(f"{m} ({why})" for m, why in rep["dead"])
            return (f"unroutable under the active deny-list: {dead}. No provider survives and "
                    f"no fallback is declared — every call would 404. Edit the OpenRouter "
                    f"deny-list (or the model) and re-run; the `oracle` screen's "
                    f"`model_routing` notice shows what survives.")
        return None
    except Exception:
        return None
