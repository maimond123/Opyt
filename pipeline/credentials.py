"""
pipeline/credentials.py — credential lookup for the OPYT pipeline.

Security: keys live in ~/.opyt/.env and are read from there; they never pass through
conversation text or command arguments, and are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

from opyt_core.credentials_registry import SERVICES as SERVICES   # derived: {service: env_var}
from opyt_core.paths import opyt_path


# Add a credential by adding a row to `opyt_core.credentials_registry.REGISTRY`, not to SERVICES.
# X needs no key: every X read runs on the user's own logged-in x.com session.

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
    """Headers for any Semantic Scholar request — the one place the S2 key is attached.
    Without a key, requests still work, just on the shared anonymous rate limit."""
    headers = {"User-Agent": S2_USER_AGENT}
    key = get_credential("semanticscholar")
    if key:
        headers["x-api-key"] = key        # S2's scheme, not Bearer
    return headers


def openalex_headers() -> dict[str, str]:
    """Headers for any OpenAlex request — the one place the OpenAlex key is attached. Without a
    key, requests still work, on the keyless daily budget, which is counted per IP."""
    key = get_credential("openalex")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _user_key(env_var: str) -> str | None:
    """A key the user owns: ~/.opyt/.env → env var → repo .env. Re-read from disk on every call
    so a mid-session connect takes effect without a restart.

    The home file comes before the environment because `opyt_core` copies it into the environment
    at import: a process started before a connect holds the old value there, and reading the
    environment first served that stale copy for the life of the process (the worker
    kept a spent starter key for hours after the user's own key was stored)."""
    home = _read_env_line(opyt_path(".env"), env_var)
    if home:
        return home
    val = os.getenv(env_var)
    if val:
        return val
    return _read_env_line(Path(__file__).parent.parent / ".env", env_var)


def _read_env_line(path: Path, env_var: str) -> str | None:
    """`env_var`'s value in one .env file, or None when the file or the line is absent."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{env_var}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def get_credential(service: str) -> str | None:
    """Read credential from ~/.opyt/.env → env vars → repo .env (see `_user_key`)."""
    if service not in SERVICES:
        raise ValueError(f"Unknown service: {service}. Valid: {list(SERVICES.keys())}")
    return _user_key(SERVICES[service])


def model_access() -> bool:
    """Does this install hold an OpenRouter key? Without one Opyt runs free: no
    image reading, no model call of any kind. Re-read per call, so a key connected
    mid-session counts on the next one."""
    return get_credential("openrouter") is not None
