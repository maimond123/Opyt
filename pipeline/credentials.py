"""
pipeline/credentials.py
Credential storage and validation for OPYT pipeline.

SECURITY: API keys should NEVER be passed through conversation text,
command arguments, or logged to stdout. Keys are stored in ~/.opyt/.env
and read from there. The agent should tell users to edit the file
directly — never ask them to paste keys into the chat.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from opyt_core.credentials_registry import REGISTRY
from opyt_core.credentials_registry import SERVICES as SERVICES   # derived: {service: env_var}
from opyt_core.paths import opyt_path


# SERVICES is derived from `opyt_core.credentials_registry.REGISTRY` — add a credential by adding
# a row there, never a key to this map. Never add a REGISTRY entry whose `validate_credential`
# falls through to "No validation available" — a credential path that accepts input and ignores it
# is worse than one that doesn't exist.
#
# X needs no key AT ALL, and there is no way to supply one. EVERY X read — your bookmarks, an
# Oracle's timeline, a single post, a profile — runs on your own logged-in x.com session through
# x.com's internal GraphQL API. `TWITTERAPI_KEY` was a real credential here until 2026-08-30, for
# an unrelated paid third party used to pull OTHER people's profiles; it is gone, and the guard
# `retired-twitterapi-provider` stops it coming back.

def _env_file_path(env_path: Path | None = None) -> Path:
    """Resolve .env file location."""
    if env_path:
        return Path(env_path)
    # Check standard locations
    candidates = [
        opyt_path(".env"),
        Path(__file__).parent.parent / ".env",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Default to <OPYT_HOME>/.env for new users
    return opyt_path(".env")


S2_USER_AGENT = "opyt/1.0"


def s2_headers() -> dict[str, str]:
    """Headers for ANY Semantic Scholar request — the one place the S2 key is attached, shared
    across all six S2 call sites so a key added once works everywhere.

    Unauthenticated S2 rate-limits aggressively across every anonymous caller; a key makes the
    limit per-key instead of shared. Absent key -> plain headers, so requests still work, just
    throttled (fail-safe). Returns a plain dict for compatibility with both `requests` and
    `urllib.request`.
    """
    headers = {"User-Agent": S2_USER_AGENT}
    key = get_credential("semanticscholar")
    if key:
        headers["x-api-key"] = key        # S2's scheme — NOT Bearer
    return headers


def _write_env_atomic(path: Path, lines: list[str]) -> None:
    """Write .env via temp-file + os.replace so a crash can't leave a half-written
    (truncated) file. os.replace is atomic within a filesystem, and chmod-ing the
    temp before the rename means the secret is never world-readable, even briefly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.chmod(0o600)
    os.replace(tmp, path)


def store_credentials(updates: dict[str, str], env_path: Path | None = None) -> Path:
    """Write SEVERAL credentials to the .env in ONE atomic replace.

    Use for credential SETS that must land together, e.g. X's access+refresh pair — X invalidates
    the old refresh token on every refresh, so persisting only half the pair bricks the next run.
    `updates` maps service name (see SERVICES) -> value; also updates os.environ.
    """
    unknown = [s for s in updates if s not in SERVICES]
    if unknown:
        raise ValueError(f"Unknown service(s): {unknown}. Valid: {list(SERVICES.keys())}")

    path = _env_file_path(env_path)
    lines = path.read_text().splitlines() if path.exists() else []

    for service, value in updates.items():
        env_var = SERVICES[service]
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={value}"
                break
        else:
            lines.append(f"{env_var}={value}")

    _write_env_atomic(path, lines)

    # Reflect into the live process only after the on-disk write succeeds.
    for service, value in updates.items():
        os.environ[SERVICES[service]] = value

    return path


def validate_credential(service: str, key: str) -> tuple[bool, str]:
    """Per-service validation. Returns (success, message).

    - LLM providers (openrouter/anthropic/…): liveness ping via the configured
      backend (llm_client.validate_provider) — provider-agnostic, no vendor hardcoded
      and no direct Anthropic SDK. Routes through settings.yaml like every other call.
    - github: GET /user (free for authenticated requests).
    - semanticscholar: a real fetch of one known paper (free), because a bad S2 key does not
      error — S2 ignores it and silently drops you back to the shared anonymous pool. A format
      check would "pass" a dead key and the user would never learn why they still get 429s.
    """
    from pipeline.llm_client import known_providers, validate_provider
    if service in known_providers():
        return validate_provider(service, key)
    if service == "github":
        return _validate_github(key)
    if service == "semanticscholar":
        return _validate_s2(key)
    if service == "opyt_service":
        return _validate_opyt_service_token(key)
    return True, f"No validation available for {service}"


def _validate_opyt_service_token(key: str) -> tuple[bool, str]:
    """Format check only — there is no endpoint to ask.

    Unlike every other row here, this credential does not name a third party with a fixed URL:
    the service that issued it is whichever host the owner publishes to, and its address is not
    in this file. Reaching one would need a base URL this function is never given, so the honest
    check is the shape and the honest message says so."""
    if not key or len(key.strip()) < 20:
        return False, "Token is too short to be a service token"
    if any(c.isspace() for c in key):
        return False, "Token contains whitespace"
    return True, "Format looks valid (no endpoint to test against — the host is yours to name)"


def _validate_github(key: str) -> tuple[bool, str]:
    """GET /user — free for authenticated requests."""
    try:
        import requests
        resp = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            username = resp.json().get("login", "unknown")
            return True, f"GitHub token valid (user: {username})"
        else:
            return False, f"GitHub API returned {resp.status_code}"
    except Exception as e:
        return False, f"GitHub validation failed: {e}"


def _validate_s2(key: str) -> tuple[bool, str]:
    """Fetch one known paper with the key attached. Free — S2 charges nothing.

    A 429 is NOT reported as invalid: it means the key was not honored on THIS request, which an
    unauthenticated burst also produces, so it cannot distinguish a bad key from a busy moment.
    Saying "invalid" there would send the user to regenerate a perfectly good key.
    """
    try:
        import requests
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762",
            params={"fields": "title"},
            headers={"User-Agent": S2_USER_AGENT, "x-api-key": key},
            timeout=15,
        )
        if resp.status_code == 200:
            return True, "Semantic Scholar key accepted"
        if resp.status_code in (401, 403):
            return False, f"Semantic Scholar rejected the key ({resp.status_code})"
        if resp.status_code == 429:
            return True, ("rate-limited right now, so the key could not be confirmed — "
                          "this does NOT mean it is invalid; try again in a minute")
        return False, f"Semantic Scholar returned {resp.status_code}"
    except Exception as e:
        return False, f"Semantic Scholar validation failed: {e}"


def get_credential(service: str) -> str | None:
    """Read credential from env vars → ~/.opyt/.env → repo .env."""
    if service not in SERVICES:
        raise ValueError(f"Unknown service: {service}. Valid: {list(SERVICES.keys())}")

    env_var = SERVICES[service]

    # Check environment first
    val = os.getenv(env_var)
    if val:
        return val

    # Check .env files
    for candidate in [
        opyt_path(".env"),
        Path(__file__).parent.parent / ".env",
    ]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None
