"""
pipeline/kb/ingest_github.py — GitHub repos → ARTIFACT atoms (direct-to-atom, free source).

A repo is an artifact its owner authored, so `entry_mode="oracle-footprint"` (the same mode the
X and Substack footprint sweeps write) and `what_kind="artifact"`. Reuses
the existing GitHub client + repo renderer wholesale; the atom-KB layer adds the routing card
and the chunk embeddings.

NOTE: this is `pipeline.kb.ingest_github` — distinct from `pipeline.ingestion.sources.github`
(the Layer-1 fetch/render helpers it borrows from). Same source, two consumers, no shared state.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from . import derive, schema
from .embed import assert_model
from .ingest_common import (BASIS_OBSERVED, BODY_ABSENT, BODY_COMPLETE, AtomSink, body_fields,
                            snapshot_and_hash, submit_atom)


def _fetch_handle_repos(handle: str) -> list[dict]:
    """Repos for a handle, trying user then org. A user handle 404s the org endpoint and
    vice-versa; whichever returns rows wins."""
    from pipeline.ingestion.sources.github import _fetch_repos, _fetch_org_repos
    repos = _fetch_repos(handle)
    if not repos:
        repos = _fetch_org_repos(handle)
    return repos or []


# ── Single-repo-from-URL (the footprint link-dispatch twin of the handle crawl) ─────────
# A repo link an Oracle *references* in a tweet → its OWN artifact atom. Same repo→atom mapping
# as the handle crawl below, a different entry: entry_mode='author_referenced' (the Oracle
# pointed at it) instead of 'oracle-footprint' (we swept a tracked handle's own archive). Both
# are in HUMAN_ATTESTED; the distinction they carry is authorship, not reachability.

_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/([A-Za-z0-9._-]+)",
    re.I)
# github.com's own product/route first-segments — NOT user/org owners. Anything else that slips
# through 404s the fetch.
_GH_RESERVED = {"features", "about", "pricing", "marketplace", "sponsors", "topics", "collections",
                "trending", "explore", "settings", "notifications", "orgs", "apps", "login", "join",
                "search", "new", "organizations", "account", "site", "security", "enterprise",
                "readme", "contact", "pulls", "issues", "codespaces", "dashboard"}


def _github_owner_repo(url: str) -> tuple[str, str] | None:
    """(owner, repo) from a GitHub REPO url, or None if it isn't one — a bare profile, a reserved
    first segment, or a different host (gist.github.com). Strips a trailing '.git' and any deeper
    path. Owner case is NOT authoritative here; the atom keys on the API's canonical login."""
    m = _GITHUB_REPO_RE.match((url or "").strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo or owner.lower() in _GH_RESERVED:
        return None
    return owner, repo


def _fetch_repo(owner: str, name: str) -> dict | None:
    """One repo dict via GET /repos/{owner}/{name}. None on 404 / any failure (fail-safe)."""
    from pipeline.ingestion.sources.github import _gh_get
    resp = _gh_get(f"https://api.github.com/repos/{owner}/{name}")
    return resp.json() if resp else None


def _seed_owner_identity(conn: sqlite3.Connection, handle: str) -> str | None:
    """Store the handle's DECLARED website on its entity, once per handle, before the repo loop.
    Returns the entity id when a link landed, else None. Never raises.

    Without this, a GitHub footprint can orphan from the Oracle's canonical entity: `resolve.py`'s
    self-platforms don't include `github:`, so cross-platform merge depends on this outbound link
    landing first.
    merge mechanics.

    Order is load-bearing: `upsert_entity` COALESCEs `identity_links`, so this must run before any
    bare per-repo upsert or the link can never land.

    Costs one GitHub request per HANDLE (not per repo); reuses the same client the rest of this
    adapter borrows."""
    from pipeline.ingestion.sources.github import _fetch_user
    try:
        prof = _fetch_user(handle) or {}
    except Exception:                    # fail-safe: an identity link is never worth the crawl
        return None
    owner = prof.get("login") or handle
    # GitHub's profile "website" field is `blog`. Absent/blank is the common case and writes
    # nothing — the per-repo upsert still creates the entity, just without a link to merge on.
    site = (prof.get("blog") or "").strip()
    if not site:
        return None
    entity_id = f"github:{owner}"
    schema.upsert_entity(conn, entity_id, name=owner, identity_links=[site])
    return entity_id


def _repo_atom(repo: dict, *, atom_id: str, raw_ref: str, raw_hash: str,
               entry_mode: str, readme: str | None = None) -> tuple[dict, str]:
    """Build the artifact-atom dict + its author-entity id for one repo. Shared by `sync_github`
    (entry_mode='oracle-footprint') and `github_atom_from_url` (entry_mode='author_referenced');
    the caller owns `atom_id` so the two paths can't disagree on identity.

    `readme` is the fetched README — a repo's README IS its body, so a README-less repo is a real
    ABSENT atom. It is also None when the fetch FAILED; `body_state` doesn't yet distinguish the
    two — a known fail-safe gap."""
    meta = derive.derive_github(repo)
    who_id = meta["who_id"]
    atom = {
        "atom_id": atom_id,
        "source_type": "github",
        "what_kind": "artifact",
        "who_id": who_id,
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],   # 'push' — NOT a publish date
        "about_entities": meta["about_entities"],
        "source_url": repo.get("html_url"),
        "raw_ref": raw_ref,
        "raw_hash": raw_hash,
        "description": meta["description"],
        "payload": {
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            # `code_language` not `language`: this is a PROGRAMMING language ("C++"), while other
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


def github_atom_from_url(conn: sqlite3.Connection, embedder, url: str, *,
                         entry_mode: str = "author_referenced",
                         seen: dict | None = None, sink=None, on_written=None,
                         prefetched: dict | None = None) -> str | None:
    """Fetch ONE github repo by URL → an artifact atom, idempotent by content hash. Returns the
    CANONICAL `atom_id` (`github:{api-owner}/{api-name}`) whenever the atom exists after the call
    (freshly minted or already present/unchanged), or None if the url isn't a repo or the fetch/
    embed failed — the caller must NOT vouch to a missing atom. Never raises.

    Keys on the API's canonical owner login, not the URL's casing, so a footprint reference dedups
    against the tracked-handle crawl instead of minting a twin. `seen` is a caller-threaded
    `{atom_id: raw_hash}` for batch dedup; absent → loaded from the DB.

    `sink` + `on_written(atom_id)`: join a caller's batch instead of paying an own embed round-trip.
    With a sink the atom is NOT durable when this returns — `on_written` is the only landed signal.

    `prefetched={"repo": ..., "readme": ...}` skips BOTH GitHub API calls for a caller running many
    URLs across a pool; without it the two round-trips happen serially wherever this is called."""
    from pipeline.ingestion.sources.github import _fetch_readme, _repo_to_markdown
    from pipeline.ingestion.utils import log

    owner_repo = _github_owner_repo(url)
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
        # `readme` is legitimately None (a repo with no README), so a prefetched payload must be
        # detected by key presence — `.get("readme") or _fetch_readme(...)` would re-fetch every
        # README-less repo and quietly reintroduce the round-trip this parameter exists to remove.
        readme = prefetched["readme"] if prefetched and "readme" in prefetched \
            else _fetch_readme(owner, name)
        md = _repo_to_markdown(repo, readme, author=f"@{owner}", author_name=owner)
        decided = snapshot_and_hash("github", atom_id, md, seen)
        if decided is None:                          # unchanged snapshot → already present, no re-embed
            return atom_id
        raw_ref, raw_hash = decided

        assert_model(conn, embedder)                 # guard the store's embedding identity BEFORE spend
        atom, who_id = _repo_atom(repo, atom_id=atom_id, raw_ref=raw_ref, raw_hash=raw_hash,
                                  entry_mode=entry_mode, readme=readme)
        schema.upsert_entity(conn, who_id, name=owner)
        submit_atom(conn, embedder, sink, atom=atom, snapshot_text=md, on_written=on_written)
        seen[atom_id] = raw_hash
        return atom_id
    except Exception as e:                            # fetch/embed/write failure → SKIP (no vouch target)
        log(f"[footprint] github atom from {url} skipped (fetch/embed failed): {e}")
        return None


# ── A fork is an EDGE, not an atom ────────────────────────────────────────────────────
# A fork's atom would carry the upstream's README under this person's who_id — false attribution.
# The fork filter drops the ATOM, so an upstream project's README can never outrank what an Oracle
# actually wrote. It used to also record the ACT as a `forked` edge, at the cost of one extra API
# call per fork (the repo-LIST endpoint omits `parent`/`source`). That edge was never read, and the
# `edges` table was deleted 2026-08-23 — so the call went with it.


def _pushed_before(repo: dict, since) -> bool:
    """Was this repo last PUSHED before `since`? The incrementality gate for a refresh crawl.

    `pushed_at` comes from the repo-LIST response already in hand, so this costs no extra call and
    saves the README + fork-edge calls that follow. Fail-safe: a missing/unparseable date returns
    False (repo is processed), never silently dropping a repo."""
    if since is None:
        return False
    raw = (repo.get("pushed_at") or "").strip()
    if not raw:
        return False
    try:
        pushed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if pushed.tzinfo is None:
        pushed = pushed.replace(tzinfo=timezone.utc)
    ref = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    return pushed < ref


def sync_github(conn: sqlite3.Connection, embedder, *, handles: list[str],
                min_stars: int = 0, include_forks: bool = False,
                since: datetime | None = None) -> dict:
    """Ingest each handle's repos as artifact atoms. Idempotent by snapshot hash. Returns a summary.

    Cross-platform identity is `resolve.py`'s job and it reads only `identity_links`, which
    `_seed_owner_identity` already stores. A `same_entity` parameter used to attest the same fact
    a second time as an edge; it went with the `edges` table on 2026-08-23.

    `since` is the refresh seam: skip any repo whose `pushed_at` predates it, before the README
    fetch and the fork upstream lookup. An EFFICIENCY fix, not a correctness one — `GITHUB_TOKEN`
    already lifts the rate limit well past the binding constraint (wall clock / lock hold time).
    Onboarding passes None.

    TWO filters answer different questions:

      • `include_forks=False` (default) — "is this THEIR work?" A fork's atom carries the
        upstream's README under this person's `who_id`, so keeping it is misattribution. Forks are
        NOT dropped entirely: the ACT is always recorded as a `forked` edge; the flag only gates
        the ATOM.

      • `min_stars` — "is this work notable?" A lagging popularity proxy (a brand-new repo has 0
        stars however good it is).

    Default `min_stars=0`: a fixed star floor makes both a false-positive and a false-negative
    error on real data (see doc). Authorship has an exact answer, so it's the default filter;
    notability stays opt-in for a caller that wants a highlight reel."""
    from pipeline.ingestion.sources.github import _fetch_readme, _repo_to_markdown

    assert_model(conn, embedder)  # fail-fast on model drift before any spend
    seen = schema.load_hashes(conn, "github")

    # Batch the embed across repos (ARC-1 Job A): one shared sink pools many repos' chunks per
    # flush so the process-wide embed gate works a WIDE flush (8-way) instead of a per-repo call
    # re-serializing it. Fetch stays serial — a GitHub-API crawl, no scrape-concurrency change.
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, flush_chunks=8 * bs)
    counts = {"added": 0}            # bumped in on_written — a DURABLE-write count, not a submit count
    submitted = skipped = forked = stale = 0

    def _mark() -> None:             # fires AFTER the atom's row commits (never on a poison-skip)
        counts["added"] += 1

    for handle in handles:
        # FIRST, before any repo upsert: the owner's declared website, which is what lets
        # `resolve.resolve_entities` fold this GitHub footprint into the Oracle's canonical entity
        # instead of stranding it. See `_seed_owner_identity` for why the order is load-bearing.
        _seed_owner_identity(conn, handle)
        repos = _fetch_handle_repos(handle)
        for repo in repos:
            # Stale first — before the fork branch, so an untouched fork also skips its upstream
            # GET. Both of the two calls this loop makes per repo happen below this line.
            if _pushed_before(repo, since):
                stale += 1
                continue
            # Forks first — misattribution, not a quality filter: a fork's atom carries the
            # upstream's README/description stamped with this person's who_id. Checked before
            # `_fetch_readme` so a skipped fork costs no README call.
            if repo.get("fork"):
                if not include_forks:
                    forked += 1       # counts forks EXCLUDED, not forks seen — see the return dict
                    continue
            if int(repo.get("stargazers_count", 0)) < min_stars:
                continue
            owner = (repo.get("owner") or {}).get("login", "") or handle
            name = repo.get("name", "")
            if not name:
                continue
            atom_id = f"github:{owner}/{name}"

            readme = _fetch_readme(owner, name)
            md = _repo_to_markdown(repo, readme, author=f"@{owner}", author_name=owner)

            decided = snapshot_and_hash("github", atom_id, md, seen)
            if decided is None:
                skipped += 1
                continue
            raw_ref, raw_hash = decided

            atom, who_id = _repo_atom(repo, atom_id=atom_id, raw_ref=raw_ref, raw_hash=raw_hash,
                                      # NOT user-saved (curation) — a tracked handle's own archive.
                                      entry_mode="oracle-footprint", readme=readme)
            schema.upsert_entity(conn, who_id, name=owner)

            seen[atom_id] = raw_hash          # within-run dedup: mark on DECISION (in-memory, rebuilt
            submitted += 1                    # from the DB each run) — a duplicate owner across two
            sink.submit(atom, md, on_written=_mark)   # handles can't re-embed the repo
    sink.close()

    # added < submitted when the sink isolates a poison-chunk repo (skip-and-continue, not abort).
    # `forked` counts repos excluded as ATOMS (the edge is still written either way). `stale` is
    # kept separate from `skipped`: stale means never fetched, skipped means fetched-unchanged —
    return {"source": "github", "added": counts["added"], "skipped": skipped,
            "forked": forked, "stale": stale, "failed": submitted - counts["added"],
            "total": schema.count_atoms(conn, "github")}
