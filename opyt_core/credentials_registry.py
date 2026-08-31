"""
opyt_core/credentials_registry.py — THE list of credentials OPYT knows how to use.

One row per credential. Adding a key is adding a row here, and nothing else. This replaced five
hand-maintained lists of "which env var holds which credential" that could silently disagree;
`KNOWN`, `SERVICES`, and `PROVIDER_ENV`
are derived views at the bottom of this file, not copies. (The other two derived surfaces — the
`.env` template writer and the `check_prerequisites` preflight — were deleted caller-less on
2026-08-28/29; the rows outlived every hand list.)

Validators deliberately do not live here: `validate_credential` does real network calls, and this
module must import nothing but `dataclasses` since `opyt_core/keys.py` depends on it and must
stay cheap and import-free. Dispatch stays in `pipeline/credentials.py`, keyed on `service`;
`test_every_registry_service_has_a_validator` pins that every row has a branch there.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Credential:
    """One credential, stated once.

    `tier` is `required` or `optional` — deliberately no third value. Do not add a `dormant` tier;
    see doc for why one was rejected.
    """

    env: str                        # the identity: the environment variable name
    service: str                    # the short alias callers use: get_credential("github")
    provider: str | None = None     # LLM providers only; None for everything else
    tier: str = "optional"          # required | optional — decides missing-key severity
    purpose: str = ""               # one line: what stops working without it
    signup_url: str = ""            # a user with a MISSING key needs the link, not just the name

    @property
    def required(self) -> bool:
        return self.tier == "required"


# Order is the order a user sees: required first, then optional. It drives BOTH the `.env`
# scaffold's line order and `opyt-keys --list`, which previously disagreed with each other.
REGISTRY: tuple[Credential, ...] = (
    Credential(
        env="OPENROUTER_API_KEY",
        service="openrouter",
        provider="openrouter",
        tier="required",
        purpose="runs every LLM role (classification, extraction, vision) and the hosted "
                "embedder, so hybrid search cannot be built or queried without it",
        signup_url="https://openrouter.ai/keys",
    ),
    Credential(
        env="GITHUB_TOKEN",
        service="github",
        tier="optional",
        purpose="raises the GitHub rate limit from 60 to 5000 req/hr; a token with NO scopes "
                "ticked is enough, since OPYT only reads public repos",
        signup_url="https://github.com/settings/tokens",
    ),
    Credential(
        env="OPYT_SERVICE_TOKEN",
        service="opyt_service",
        tier="optional",
        # The OWNER half of foreign KB reads. A READER needs nothing here: they redeem a one-time
        # grant code, and the reader token that comes back is written into their peers registry,
        # not into `.env`. This row is only for the person PUBLISHING a knowledge base.
        purpose="uploads your export to the service that hosts it for other people to query; "
                "without it your knowledge base stays on your own disk and only readers you "
                "hand the file to can read it",
        # The one row whose issuer is not a third party: the token comes from whoever RUNS the
        # service, which for a self-hosted one is the owner themselves. So the link is to the
        # instructions for standing one up, because that is what a person holding no token needs.
        signup_url="https://github.com/maimond123/Opyt",
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


# ── Derived views — the four former registries, now computed ─────────────────────
# Each of these WAS a hand-maintained literal somewhere else. They are exported so the old call
# sites keep their names and shapes, which is what made this collapse behavior-preserving.

KNOWN: tuple[str, ...] = tuple(c.env for c in REGISTRY)
SERVICES: dict[str, str] = {c.service: c.env for c in REGISTRY}
PROVIDER_ENV: dict[str, str] = {c.provider: c.env for c in REGISTRY if c.provider}


def by_service(service: str) -> Credential | None:
    """The row for a service alias, or None if the alias is unknown."""
    return next((c for c in REGISTRY if c.service == service), None)


