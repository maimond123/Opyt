"""
pipeline/llm_providers.py

Provider discovery and credential liveness validation. `llm_client.py` re-exports everything
here so existing `llm_client.validate_provider(...)` call sites keep working.

Routes through `llm_client._BACKENDS` / settings.yaml so no vendor name is hardcoded — validation
stays provider-neutral. `llm_client` is imported lazily inside each function: `llm_client.py`
imports this module back for its re-exports, so a top-level import here would deadlock the cycle.
"""

from __future__ import annotations

from pipeline import llm_spend
from pipeline.ingestion.utils import load_yaml_config


def known_providers() -> set[str]:
    """Providers the backend can actually call (the _BACKENDS dispatch keys)."""
    from pipeline import llm_client
    return set(llm_client._BACKENDS)


def default_provider() -> str:
    """The fallback provider for roles that don't name one (settings.yaml)."""
    cfg = load_yaml_config().get("llm_backends") or {}
    return cfg.get("default_provider", "openrouter")


def configured_providers() -> set[str]:
    """Every distinct provider referenced across llm_backends.roles (each role
    defaulting to default_provider) — the set a credential check must cover."""
    cfg = load_yaml_config().get("llm_backends") or {}
    default = cfg.get("default_provider", "openrouter")
    roles = cfg.get("roles") or {}
    provs = {(r.get("provider") or default) for r in roles.values()}
    return provs or {default}


def _cheapest_configured_model(provider: str) -> str | None:
    """Cheapest model (by output price) among the roles that use `provider`, so a
    liveness ping bills the least. None if no configured role uses this provider."""
    cfg = load_yaml_config().get("llm_backends") or {}
    default = cfg.get("default_provider", "openrouter")
    roles = cfg.get("roles") or {}
    models = {
        r["model"] for r in roles.values()
        if (r.get("provider") or default) == provider and r.get("model")
    }
    if not models:
        return None
    # Unknown-priced models sort last (inf), so a priced one always wins the tie.
    return min(models, key=lambda m: llm_spend._PRICING.get(m, (0.0, float("inf")))[1])


def validate_provider(provider: str, key: str | None = None, *,
                      model: str | None = None) -> tuple[bool, str]:
    """Liveness-check a provider's credential with a 1-token ping through its own backend.

    If `key` is given it's passed straight to the backend as a parameter, never written to
    os.environ, so it can be validated before being persisted. key=None validates whatever
    the env already holds. Returns (ok, human_message)."""
    from pipeline import llm_client
    try:
        backend = llm_client._get_backend(provider)  # ValueError on unknown provider
    except ValueError as e:
        return False, str(e)
    model = model or _cheapest_configured_model(provider)
    if not model:
        return False, f"no configured model for provider {provider!r} to validate against"

    # Stack-local param, not a process-global os.environ swap — avoids racing a concurrent call().
    extra = {"api_key": key} if key is not None else {}
    try:
        backend(model, "ping", "ping", 1, **extra)   # raises on auth/network failure
        return True, f"{provider} key is valid"
    except Exception as e:
        return False, f"{provider} key validation failed: {e}"


def validate_provider_status(provider: str, key: str | None = None, *,
                             model: str | None = None) -> tuple[bool, str, int | None]:
    """`validate_provider` plus the HTTP status, when the backend surfaced one.

    401 and 402 need opposite remedies (new key vs. funding), so callers must not collapse
    them. Status is sniffed out of the message string since the backend has no typed status
    field yet; `None` means undetermined and callers must name both possibilities.
    """
    ok, msg = validate_provider(provider, key, model=model)
    if ok:
        return True, msg, None
    status = None
    for code in (401, 402, 403, 429):
        if str(code) in msg:
            status = code
            break
    return False, msg, status
