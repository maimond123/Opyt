"""
pipeline/llm_client.py

Single facade for every LLM call in the codebase. Each caller asks for a
**role** (`note_classify`, `vision`, ...) and the client
looks up the configured provider + model in settings.yaml. This keeps backend
choice out of the call sites and in one place.

Only one backend exists (OpenRouter), but the provider indirection (`_BACKENDS`,
`_PROVIDER_ENV`, `_get_backend`) stays so adding the next provider is a registration,
not a rewrite. Do not collapse it into a hardcoded OpenRouter call.

`call` is synchronous, and that is the whole surface. An `acall` async wrapper existed
until 2026-08-28 on the stated grounds that "the existing pipeline uses both"; it never
had a caller. Concurrency here is per-caller (a thread pool plus its own semaphore), not
an event loop, so a coroutine had nothing to attach to.

Cloudflare on Groq/OpenRouter blocks Python's default UA, so we always send a
browser-shaped UA to avoid 1010 errors.

Latency telemetry (`pipeline/llm_telemetry.py`) and provider discovery
(`pipeline/llm_providers.py`) live in separate modules. Local provider-charge estimates were
deliberately removed: Opyt retains response and token metadata.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from opyt_core import credentials_registry
from opyt_core.config import merge_provider_routing

from pipeline import llm_telemetry
from pipeline.ingestion.utils import load_yaml_config

# Cloudflare 1010 fix — both Groq + some OpenRouter routes need this.
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Network endpoints
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Provider → env var holding its API key, so preflight (presence) and validate_provider
# (liveness) agree on where a provider's key lives. Derived from `opyt_core.credentials_registry`
# (keyed on each row's `provider` field) — register a provider there, not here.
_PROVIDER_ENV: dict[str, str] = credentials_registry.PROVIDER_ENV


# ── Public response type ─────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    raw: dict | None = field(default=None, repr=False)


# ── Config + role resolution ─────────────────────────────────────────────────


def _load_role_config(role: str) -> dict:
    """Read the role's provider/model/max_tokens from settings.yaml."""
    cfg = load_yaml_config().get("llm_backends") or {}
    roles = cfg.get("roles") or {}
    if role not in roles:
        raise ValueError(
            f"llm_client: role {role!r} not declared under llm_backends.roles "
            f"in settings.yaml. Add it before calling."
        )
    role_cfg = dict(roles[role])  # copy so caller can override safely
    role_cfg.setdefault("provider", cfg.get("default_provider", "openrouter"))
    role_cfg.setdefault("max_tokens", 4096)
    return role_cfg


# ── Backends ─────────────────────────────────────────────────────────────────


class _BackendError(Exception):
    """A backend call failed. `status` carries the HTTP code when known (None for a
    config-class failure like a missing key), so an AIMD gate can act on a 429 specifically
    instead of treating every failure as backpressure and halving on a missing API key.
    Mirrors `pipeline.kb.embed.EmbedError.status`, which the embed gate already reads."""

    def __init__(self, *args, status: int | None = None):
        super().__init__(*args)
        self.status = status


class ModelUnroutableError(_BackendError):
    """No provider can serve this model — the request is IMPOSSIBLE, not merely failing, and
    retrying it will never succeed. Excluded from the circuit breaker (`ignore=`) so one
    unroutable role can't trip the shared breaker for every other role. Callers should treat it
    as terminal for the whole run, not retry per item. See `pipeline/model_routing.py` for the
    preflight that catches this before any request.
    """


# The marker OpenRouter returns when the provider filter leaves no candidates. Matched on the
# response BODY because the status code alone (404) cannot distinguish "no such model" from
# "no permitted provider for this model" — and only the latter is a config error we can name.
_UNROUTABLE_MARKERS = ("all providers have been ignored", "no allowed providers",
                       "no providers available")


def _http_json(req: urllib.request.Request, timeout: float = 180.0) -> tuple[dict, float]:
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        low = body.lower()
        if any(m in low for m in _UNROUTABLE_MARKERS):
            # Terminal, not transient — see ModelUnroutableError. Raised as its own type so the
            # breaker can ignore it and the caller can disable the stage instead of per-item retry.
            raise ModelUnroutableError(f"HTTP {e.code}: {body}", status=e.code) from None
        raise _BackendError(f"HTTP {e.code}: {body}", status=e.code) from None
    return data, time.time() - t0


_or_breaker = None
def _openrouter_breaker():
    """Lazy module-level breaker: an OpenRouter outage trips once and every caller
    (across sessions — state persists) fails fast instead of repeating futile retries."""
    global _or_breaker
    if _or_breaker is None:
        from pipeline.circuit_breaker import CircuitBreaker
        _or_breaker = CircuitBreaker("openrouter")
    return _or_breaker


def _call_openrouter_sync(model: str, system: str, user: str, max_tokens: int,
                          response_format: str | None = None,
                          images: list[str] | None = None,
                          frequency_penalty: float | None = None,
                          api_key: str | None = None) -> tuple[str, int, int, float, dict]:
    # api_key: explicit override (validate_provider) so a key under test never has to be
    # written into the shared os.environ. Falls back to the env for normal calls.
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise _BackendError("OPENROUTER_API_KEY not set")
    # Multimodal: OpenRouter is OpenAI-compatible — image blocks use {"type":"image_url"}.
    if images:
        user_content: Any = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in images
        ]
    else:
        user_content = user
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    prefs: dict = {}
    if response_format == "json_object":
        # require_parameters: only route to a provider that honors JSON mode.
        body["response_format"] = {"type": "json_object"}
        prefs["require_parameters"] = True
    # Upstream routing (deny broken upstreams, rank rest by latency — see opyt_core.config)
    # applies to every chat role, not just JSON ones, since it's a shared transport concern.
    # Lives here, not in `call()`, because provider routing is an OpenRouter-specific concept.
    prefs = merge_provider_routing(prefs)
    if prefs:
        body["provider"] = prefs
    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _BROWSER_UA,
            "HTTP-Referer": "https://github.com/maimond123/Opyt",
            "X-Title": "opyt",
        },
    )
    # Breaker: a sustained OpenRouter failure trips and fails fast for a cooldown. `ignore=` keeps
    # a terminal routing failure (retrying can't help) off the shared breaker.
    data, elapsed = _openrouter_breaker().call(lambda: _http_json(req),
                                               ignore=(ModelUnroutableError,))
    # content can be null (refusal, content filter, empty / tool-only completion); coerce to "".
    choices = data.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    return text, in_tok, out_tok, elapsed, data


_BACKENDS = {
    "openrouter": _call_openrouter_sync,
}


def _get_backend(provider: str):
    """Indirection so tests can monkey-patch."""
    if provider not in _BACKENDS:
        raise ValueError(f"unknown provider: {provider!r}")
    return _BACKENDS[provider]


def _serving_upstream(provider: str, raw: dict | None) -> str:
    """The upstream that served a call, for latency telemetry only."""
    if provider == "openrouter":
        return str((raw or {}).get("provider") or "unknown")
    return provider or "unknown"


# ── Public API ───────────────────────────────────────────────────────────────


def call(role: str, *, system: str, user: str, model: str | None = None,
         max_tokens: int | None = None, images: list[str] | None = None,
         frequency_penalty: float | None = None) -> LLMResponse:
    """The LLM call. Synchronous, and the only surface.

    `images` (image URLs) routes through the provider's multimodal schema — used
    by the `vision` role. `frequency_penalty` discourages the model from repeating
    itself — the fix for a model looping on a dense image (see ocr_cascade.py's OCR
    call). Like `response_format`, both are threaded to the backend ONLY when
    supplied, so 4-arg test fakes stay valid."""
    role_cfg = _load_role_config(role)
    if model:
        role_cfg["model"] = model
    if max_tokens:
        role_cfg["max_tokens"] = max_tokens

    provider = role_cfg["provider"]
    backend = _get_backend(provider)
    # Pass response_format / images / frequency_penalty ONLY when present, so the
    # test-fake backends (4-arg) and every existing role are untouched — only
    # opted-in calls thread them.
    rf = role_cfg.get("response_format")
    extra = {"response_format": rf} if rf else {}
    if images:
        extra["images"] = images
    if frequency_penalty is not None:
        extra["frequency_penalty"] = frequency_penalty
    text, in_tok, out_tok, elapsed, raw = backend(
        role_cfg["model"], system, user, role_cfg["max_tokens"], **extra
    )
    upstream = _serving_upstream(provider, raw)
    llm_telemetry._record_latency(role, elapsed, upstream=upstream)
    return LLMResponse(
        text=text.strip(),
        model=role_cfg["model"],
        provider=provider,
        input_tokens=in_tok,
        output_tokens=out_tok,
        raw=raw,
    )


def preflight(role: str) -> str | None:
    """Verify the role's provider has its API key set in the environment.

    Returns None if ready, or a human-readable reason string if not. Lets
    callers fail-fast before starting batched work (e.g. extract_all)."""
    cfg = _load_role_config(role)
    provider = cfg["provider"]
    env_var = _PROVIDER_ENV.get(provider)
    if not env_var:
        return f"unknown provider {provider!r} for role {role!r}"
    if not os.environ.get(env_var):
        return f"{env_var} not set (required for role {role!r} → {provider})"
    return None


# ── Test seam ────────────────────────────────────────────────────────────────

def _set_backend_for_tests(provider: str, fn) -> None:
    """Tests can substitute a fake backend instead of network."""
    _BACKENDS[provider] = fn


# ── Backward-compatible telemetry re-export ──────────────────────────────────
# `llm_telemetry` owns this code; `__getattr__` keeps existing telemetry readers live without
# giving `llm_client` a second writable state store. Provider discovery is imported directly by
# its callers so the two modules do not form an import cycle.
_REEXPORTS = (llm_telemetry,)


def __getattr__(name: str):
    for module in _REEXPORTS:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
