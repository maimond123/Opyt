"""
pipeline/kb/ingest_github.py — GitHub repos → artifact atoms.

One repo→atom mapping (`_repo_atom`), for one repo somebody referenced (`github_atom_from_url`,
entry_mode='author_referenced') and for the helpers the GitHub archive walk uses
(`oracles/sources/archive/github.py`): `github_owner_repo`, `_github_owner`, `_pushed_at`.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from functools import partial

from pipeline.platforms.source_classify import GITHUB_RESERVED

from . import derive, schema
from .ingest_common import (BASIS_OBSERVED, BODY_ABSENT, BODY_COMPLETE, body_fields,
                            changed_hash, store_atom)


# ── Single repo from a URL ──────────────────────────────────────────────────────────────

_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/([A-Za-z0-9._-]+)",
    re.I)


def github_owner_repo(url: str) -> tuple[str, str] | None:
    """(owner, repo) from a GitHub repo url, or None if it isn't one — a bare profile, a reserved
    first segment, or a different host (gist.github.com). Owner case is not authoritative; the atom
    keys on the API's canonical login."""
    m = _GITHUB_REPO_RE.match((url or "").strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo or owner.lower() in GITHUB_RESERVED:
        return None
    return owner, repo


# The account half of a github url; a url that is also a repository resolves to the repository
# (`github_owner_repo` first).
_GITHUB_OWNER = re.compile(r"github\.com/([^/?#]+)", re.I)


def _github_owner(url: str | None) -> str | None:
    """The owner in a github.com url, or None when the first segment is one of github.com's own
    routes (`github.com/sponsors/alice` is a common bio-link shape)."""
    m = _GITHUB_OWNER.search(url or "")
    if not m:
        return None
    owner = m.group(1)
    return None if owner.lower() in GITHUB_RESERVED else owner


def _fetch_repo(owner: str, name: str) -> dict | None:
    """One repo dict via GET /repos/{owner}/{name}. None on 404 / any failure (fail-safe)."""
    from pipeline.platforms.github.api import GitHubRateLimited, gh_get
    try:
        return gh_get(f"https://api.github.com/repos/{owner}/{name}")
    except GitHubRateLimited:
        # One repo with no coverage claim riding on it, so swallowing is safe and keeps a Hopper
        # deposit from raising through MCP; the sweep below deliberately does not swallow.
        return None


def _repo_atom(repo: dict, *, atom_id: str, raw_hash: str,
               entry_mode: str, readme: str | None = None) -> tuple[dict, str]:
    """Build the artifact-atom dict + its author-entity id for one repo; the caller owns `atom_id`
    so both entry paths agree on identity. The README is the body: None → `BODY_ABSENT` (also when
    the README fetch failed, which `body_state` does not yet distinguish)."""
    meta = derive.derive_github(repo)
    who_id = meta["who_id"]
    atom = {
        "atom_id": atom_id,
        "source_type": "github",
        "what_kind": "artifact",
        "who_id": who_id,
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],   # 'push' — not a publish date
        "about_entities": meta["about_entities"],
        "source_url": repo.get("html_url"),
        "raw_hash": raw_hash,
        "description": meta["description"],
        "payload": {
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            # `code_language` not `language`: this is a programming language ("C++"), while other
            # sources use `content_lang` for a natural language ("en") — keeps the two out of the
            # same expression index.
            "code_language": repo.get("language"),
            "license": ((repo.get("license") or {}) or {}).get("spdx_id"),  # best-effort
            # Repo topics are the author's own labels, so they use the cross-source `source_tags`
            # name, slugged by `_slugs` to match every other source's tag filter.
            "source_tags": meta["source_tags"],
            **body_fields(BODY_COMPLETE if readme else BODY_ABSENT, BASIS_OBSERVED),
        },
        "entry_mode": entry_mode,
    }
    return atom, who_id


def github_atom_from_url(conn: sqlite3.Connection, url: str, *,
                         entry_mode: str = "author_referenced",
                         seen: dict | None = None, on_written=None,
                         prefetched: dict | None = None, isolate: bool = True,
                         store=store_atom) -> str | None:
    """Fetch one github repo by URL → an artifact atom, idempotent by content hash.

    Returns the canonical `github:{api-owner}/{api-name}` whenever the atom exists after the call,
    or None if the url isn't a repo or the fetch/write failed; with `isolate`, never raises.
    `seen` is `{atom_id: raw_hash}` (absent → loaded). `on_written(atom_id)` fires once the atom is
    durable. `prefetched={"repo", "readme"}` skips both API calls. `store` is the write
    (`store_atom`); a caller that lands the atom with its own state later passes one that holds it."""
    from pipeline.ingestion.sources.github import _repo_to_markdown
    from pipeline.platforms.github.api import fetch_readme
    from pipeline.ingestion.utils import log

    owner_repo = github_owner_repo(url)
    if owner_repo is None:
        return None
    try:
        repo = (prefetched or {}).get("repo") or _fetch_repo(*owner_repo)
        if not repo or not repo.get("name"):
            return None
        owner = (repo.get("owner") or {}).get("login", "") or owner_repo[0]
        name = repo.get("name", "")
        atom_id = f"github:{owner}/{name}"          # canonical case — dedup key across both entries

        if seen is None:
            seen = schema.load_hashes(conn, "github")
        # Key presence, not truthiness: a prefetched `readme` of None means the repo has none.
        readme = prefetched["readme"] if prefetched and "readme" in prefetched \
            else fetch_readme(owner, name)
        md = _repo_to_markdown(repo, readme, author=f"@{owner}", author_name=owner)
        raw_hash = changed_hash(atom_id, md, seen)
        if raw_hash is None:                      # unchanged snapshot → already present, no rewrite
            return atom_id

        atom, who_id = _repo_atom(repo, atom_id=atom_id, raw_hash=raw_hash,
                                  entry_mode=entry_mode, readme=readme)
        schema.upsert_entity(conn, who_id, name=owner)
        store(conn, atom, md, on_written=on_written and partial(on_written, atom_id))
        seen[atom_id] = raw_hash
        return atom_id
    except Exception as e:                            # foreground callers receive a failed result
        if not isolate:
            raise
        log(f"[footprint] github atom from {url} skipped (fetch/write failed): {e}")
        return None


# A fork's atom would carry the upstream's README under this person's who_id, so forks are
# dropped (and counted), not ingested.


def _pushed_at(repo: dict) -> datetime | None:
    """This repo's `pushed_at` as an aware datetime, or None when it is absent or unparseable.

    Read from the list response, so it costs no call. None means neither window gate fires, so an
    undated repo is processed, never dropped."""
    raw = (repo.get("pushed_at") or "").strip()
    if not raw:
        return None
    try:
        pushed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return pushed if pushed.tzinfo else pushed.replace(tzinfo=timezone.utc)
