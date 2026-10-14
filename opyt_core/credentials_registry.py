"""
opyt_core/credentials_registry.py — the list of credentials OPYT knows how to use.

One row per credential; adding a key is adding a row here, and nothing else. `KNOWN`,
`SERVICES` and `PROVIDER_ENV` are derived views at the bottom, not copies.

Imports nothing but `dataclasses`: `opyt_core/keys.py` depends on it and must stay cheap.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Credential:
    """One credential, stated once. `tier` is `required` or `optional`, deliberately no third value."""

    env: str                        # the identity: the environment variable name
    service: str                    # the short alias callers use: get_credential("github")
    provider: str | None = None     # LLM providers only; None for everything else
    tier: str = "optional"          # required | optional — decides missing-key severity
    purpose: str = ""               # one line: what stops working without it
    signup_url: str = ""            # a user with a missing key needs the link, not just the name

    @property
    def required(self) -> bool:
        return self.tier == "required"


# Order is the order a user sees in `opyt-keys --list`: required first, then optional.
REGISTRY: tuple[Credential, ...] = (
    Credential(
        env="OPENROUTER_API_KEY",
        service="openrouter",
        provider="openrouter",
        tier="optional",
        purpose="runs image reading (charts in posts) during ingest. Without it Opyt runs "
                "free and reads no images; search is by keyword either way",
        signup_url="https://openrouter.ai/keys",
    ),
    Credential(
        env="GITHUB_TOKEN",
        service="github",
        tier="optional",
        # Two budgets: core REST reads and `/search/repositories`. The rail is paced for the
        # anonymous search limit, so this key is throughput, never capability.
        purpose="triples GitHub search from 10 to 30 requests/min and raises other GitHub reads "
                "from 60 to 5000 req/hr; without it everything still works, just slower. A token "
                "with NO scopes ticked is enough, since OPYT only reads public repos",
        signup_url="https://github.com/settings/tokens",
    ),
    Credential(
        env="OPYT_SERVICE_TOKEN",
        service="opyt_service",
        tier="optional",
        # Only for the person publishing a knowledge base; a reader's token lives in their peers
        # registry, not `.env`.
        purpose="uploads your export to the service that hosts it for other people to query; "
                "without it your knowledge base stays on your own disk and only readers you "
                "hand the file to can read it",
        # The token comes from whoever runs the service, so the link is to standing one up.
        signup_url="https://github.com/maimond123/Opyt",
    ),
    Credential(
        env="OPENALEX_API_KEY",
        service="openalex",
        tier="optional",
        # OpenAlex meters a daily budget per IP; a key is counted on its own and gets ten times
        # the keyless one. A key OpenAlex does not know fails every call with 401, so
        # `onboard(start='openalex')` checks it before storing.
        purpose="raises OpenAlex's daily budget tenfold; without it paper lookups share the "
                "keyless budget and pause once it is spent, until the next day",
        signup_url="https://openalex.org/settings/api",
    ),
    Credential(
        env="S2_API_KEY",
        service="semanticscholar",
        tier="optional",
        purpose="unauthenticated Semantic Scholar is about 1 req/sec SHARED across every "
                "anonymous caller, so paper ingest gets 429'd and skips papers until a later run",
        signup_url="https://www.semanticscholar.org/product/api#api-key-form",
    ),
)


# ── Derived views ─────────────────────────────────────────────────────────────────

KNOWN: tuple[str, ...] = tuple(c.env for c in REGISTRY)
SERVICES: dict[str, str] = {c.service: c.env for c in REGISTRY}
PROVIDER_ENV: dict[str, str] = {c.provider: c.env for c in REGISTRY if c.provider}


def by_service(service: str) -> Credential | None:
    """The row for a service alias, or None if the alias is unknown."""
    return next((c for c in REGISTRY if c.service == service), None)


