"""
pipeline/llm_client.py — the one entry for every LLM call.

Each caller asks for a role (`vision`, ...) and the client looks up the configured provider and
model in settings.yaml, keeping backend choice out of the call sites. Only OpenRouter exists, but
the provider indirection (`_BACKENDS`, `_PROVIDER_ENV`, `_get_backend`) is kept so the next
provider is a registration, not a rewrite. `call` is synchronous.
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

# Cloudflare blocks Python's default UA on some OpenRouter routes (error 1010).
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Provider → env var holding its API key. Register a provider in `opyt_core.credentials_registry`.
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
    """A backend call failed. `status` carries the HTTP code when known (None for a config
    failure like a missing key), so an AIMD gate halves only on a 429."""

    def __init__(self, *args, status: int | None = None):
        super().__init__(*args)
        self.status = status


class ModelUnroutableError(_BackendError):
    """No provider can serve this model; retrying never succeeds. Terminal for the whole run,
    and kept off the circuit breaker so one unroutable role can't trip it for every other role.
    """


# OpenRouter's body text when the provider filter leaves no candidates; the 404 status alone
# cannot tell this from "no such model".
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
            raise ModelUnroutableError(f"HTTP {e.code}: {body}", status=e.code) from None
        raise _BackendError(f"HTTP {e.code}: {body}", status=e.code) from None
    return data, time.time() - t0


_or_breaker = None
def _openrouter_breaker():
    """Lazy module-level breaker shared by every OpenRouter call; its state persists across sessions."""
    global _or_breaker
    if _or_breaker is None:
        from pipeline.circuit_breaker import CircuitBreaker
        _or_breaker = CircuitBreaker("openrouter")
    return _or_breaker


def _call_openrouter_sync(model: str, system: str, user: str, max_tokens: int,
                          response_format: str | None = None,
                          images: list[str] | None = None,
                          frequency_penalty: float | None = None
                          ) -> tuple[str, int, int, float, dict]:
    # The error string below is pinned by .guards.py; keep it verbatim.
    from pipeline.credentials import get_credential
    api_key = get_credential("openrouter")
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
    # Upstream routing (see opyt_core.config) applies to every chat role; it lives here because
    # provider routing is OpenRouter-specific.
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
    """The LLM call.

    `images` (image URLs) use the provider's multimodal schema; `frequency_penalty` stops a model
    looping on a dense image. Like `response_format`, both reach the backend only when supplied,
    so 4-arg test fakes stay valid."""
    role_cfg = _load_role_config(role)
    if model:
        role_cfg["model"] = model
    if max_tokens:
        role_cfg["max_tokens"] = max_tokens

    provider = role_cfg["provider"]
    backend = _get_backend(provider)
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
    """Verify the role's provider has its API key resolvable.

    Returns None if ready, or a human-readable reason. Resolves through `get_credential`, not
    the env alone, so a key in the home's .env passes. With no key every role is refused: a free
    install makes no model call.
    """
    cfg = _load_role_config(role)
    provider = cfg["provider"]
    env_var = _PROVIDER_ENV.get(provider)
    if not env_var:
        return f"unknown provider {provider!r} for role {role!r}"
    if not os.environ.get(env_var):
        from pipeline.credentials import SERVICES, get_credential
        service = next((s for s, v in SERVICES.items() if v == env_var), None)
        if service is None or not get_credential(service):
            return f"{env_var} not set (required for role {role!r} → {provider})"
    return None


# ── Test seam ────────────────────────────────────────────────────────────────

def _set_backend_for_tests(provider: str, fn) -> None:
    """Tests can substitute a fake backend instead of network."""
    _BACKENDS[provider] = fn


# ── Telemetry re-export ──────────────────────────────────────────────────────
# `llm_telemetry` owns this state; `__getattr__` keeps `llm_client.<telemetry name>` readers live.
_REEXPORTS = (llm_telemetry,)


def __getattr__(name: str):
    for module in _REEXPORTS:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
