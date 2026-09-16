"""
pipeline/kb/ingest_github.py — GitHub repos → ARTIFACT atoms (direct-to-atom, free source).

A repo is an artifact its owner authored, so `what_kind="artifact"`. Reuses the existing GitHub
client + repo renderer wholesale; the atom-KB layer adds the routing card and the chunk
embeddings.

Three entries, one repo→atom mapping (`_repo_atom`):

  sync_github(handles=…)      — a tracked account's whole archive, entry_mode='oracle-footprint'
  github_atom_from_url(url)   — ONE repo somebody referenced, entry_mode='author_referenced'
  sync_github_source(url)     — the ROUTER over both, for a caller holding a discovered source
                                url and no idea which shape it is

NOTE: this is `pipeline.kb.ingest_github` — distinct from `pipeline.ingestion.sources.github`
(the Layer-1 fetch/render helpers it borrows from). Same source, two consumers, no shared state.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from pipeline.ingestion.source_classify import GITHUB_RESERVED

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
    if not repo or owner.lower() in GITHUB_RESERVED:
        return None
    return owner, repo


# The ACCOUNT half of a github url. `_github_owner_repo` above answers "is this one repository";
# this answers "whose page is this", and a url that is both resolves to the repository — see
# `sync_github_source`.
_GITHUB_OWNER = re.compile(r"github\.com/([^/?#]+)", re.I)


def _github_owner(url: str | None) -> str | None:
    """The owner in a github.com url, or None when the first segment is one of github.com's own
    routes. The reserved check belongs in BOTH url tests, not just the repository one: without it
    `github.com/sponsors/alice` — a common bio-link shape — swept the handle `sponsors`, so the
    person's real archive was never crawled and three API calls were spent proving `sponsors` has
    no repos."""
    m = _GITHUB_OWNER.search(url or "")
    if not m:
        return None
    owner = m.group(1)
    return None if owner.lower() in GITHUB_RESERVED else owner


def _fetch_repo(owner: str, name: str) -> dict | None:
    """One repo dict via GET /repos/{owner}/{name}. None on 404 / any failure (fail-safe)."""
    from pipeline.ingestion.sources.github import GitHubRateLimited, _gh_get
    try:
        resp = _gh_get(f"https://api.github.com/repos/{owner}/{name}")
    except GitHubRateLimited:
        # ONE repo, and no coverage claim rides on it — the callers (`github_atom_from_url`,
        # `ingest_x_footprint._prefetch_one_artifact`) already treat None as "could not fetch"
        # and neither stamps a window. Swallowing here keeps a Hopper deposit from raising
        # through the MCP surface; the SWEEP below deliberately does not swallow.
        return None
    return resp.json() if resp else None


def _seed_owner_identity(conn: sqlite3.Connection, handle: str) -> str | None:
    """Store the handle's DECLARED website on its entity, once per handle, before the repo loop.
    Returns the entity id when a link landed, else None. Never raises.

    Without this, a GitHub footprint can orphan from the Oracle's canonical entity: `resolve.py`'s
    self-platforms don't include `github:`, so cross-platform merge depends on this outbound link
    landing first.

    Order is load-bearing: `upsert_entity` COALESCEs `identity_links`, so this must run before any
    bare per-repo upsert or the link can never land.

    Costs one GitHub request per HANDLE (not per repo); reuses the same client the rest of this
    adapter borrows."""
    from pipeline.ingestion.sources.github import GitHubRateLimited, _fetch_user
    try:
        prof = _fetch_user(handle) or {}
    except GitHubRateLimited:
        raise                            # the crawl must not report success — see `sync_github`
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


# ── The per-run repo ceiling ──────────────────────────────────────────────────────────
# Anonymous core REST is 60 requests/hour, PER IP, and a sweep spends roughly one call per
# surviving repo plus one per handle. Measured against the live store 2026-09-06: of 123 GitHub
# owners, 115 hold a single link-routed repo (anonymous is ample for those), one holds 27, and
# @VictorTaelin holds 189 — ~191 calls, 3.2x the entire hourly budget.
#
# So a sweep is BOUNDED and RESUMABLE rather than complete: it takes this many repos, reports the
# frontier it actually reached, and `oracle_refresh.backfill_pair` comes back next cycle with
# `before=` set to that frontier. 50 is about one hour's budget with room for the handle's profile
# call and the list pages.
#
# A capped run is NOT blocked, and the vocabulary matters (`d7dbcfcf`): a host that stopped us
# verified nothing, while a run that stopped itself knows exactly how far it got — so it reports a
# frontier and its caller stamps the TTL.
#
# ⚠️ The resume rests on the LIST ORDER, which is set two modules away: `_fetch_repos` and
# `_fetch_org_repos` both ask for `sort=pushed&direction=desc`. Newest-first is what makes "the
# first N repos" and "everything after the frontier" the same set. Served in another order the
# frontier is still honest — it is the oldest repo actually fetched — but the batches interleave
# and a resume leaves holes.
REPOS_PER_RUN = 50


# ── A fork is an EDGE, not an atom ────────────────────────────────────────────────────
# A fork's atom would carry the upstream's README under this person's who_id — false attribution.
# The fork filter drops the ATOM, so an upstream project's README can never outrank what an Oracle
# actually wrote. It used to also record the ACT as a `forked` edge, at the cost of one extra API
# call per fork (the repo-LIST endpoint omits `parent`/`source`). That edge was never read, and the
# `edges` table was deleted 2026-08-23 — so the call went with it.


def _pushed_at(repo: dict) -> datetime | None:
    """This repo's `pushed_at` as an aware datetime, or None when it is absent or unparseable.

    Comes from the repo-LIST response already in hand, so reading it costs no extra call — which
    is what lets both window gates below skip a repo before its README GET, and what lets the
    sweep report the frontier it reached. Fail-safe: None means neither gate fires and the repo is
    PROCESSED, so an undated repo is never silently dropped."""
    raw = (repo.get("pushed_at") or "").strip()
    if not raw:
        return None
    try:
        pushed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return pushed if pushed.tzinfo else pushed.replace(tzinfo=timezone.utc)


def _aware(ts: datetime | None) -> datetime | None:
    """A caller-supplied bound as an aware datetime. Naive input is read as UTC, matching the
    `pushed_at` values it is compared against."""
    if ts is None:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def sync_github(conn: sqlite3.Connection, embedder, *, handles: list[str],
                min_stars: int = 0, include_forks: bool = False,
                since: datetime | None = None, before: datetime | None = None) -> dict:
    """Ingest each handle's repos as artifact atoms. Idempotent by snapshot hash. Returns a summary.

    Cross-platform identity is `resolve.py`'s job and it reads only `identity_links`, which
    `_seed_owner_identity` already stores. A `same_entity` parameter used to attest the same fact
    a second time as an edge; it went with the `edges` table on 2026-08-23.

    THREE window knobs, all reading `pushed_at` out of the list response so they cost no call:

      • `since` — the FORWARD seam. Skip any repo pushed before it. A refresh pass derives it
        from the cursor, so the run costs one call per repo touched since the last pull.

      • `before` — the BACKWARD seam. Skip any repo pushed at or after it. A resume pass sets it
        to the coverage frontier the previous run reported, so the covered prefix is skipped for
        free instead of re-fetched. X has no equivalent (its pagination carries no upper bound);
        GitHub does, because the whole repo list arrives before any README is fetched.

      • `REPOS_PER_RUN` — the per-run ceiling. Not a parameter: no caller has a reason to want
        a different one, since 60 req/hr is a physical bound and not a preference, and even a
        user asking for a whole archive is better served by three resumable cycles than by one
        run the host cuts off mid-way. See that constant.

    A crawl the host refuses returns BLOCKED rather than a short list — see `GitHubRateLimited`.
    A crawl that spends its own ceiling does NOT: it reports `covered_from` and `capped`, and the
    next cycle continues from there.

    TWO filters answer different questions:

      • `include_forks=False` (default) — "is this THEIR work?" A fork's atom carries the
        upstream's README under this person's `who_id`, so keeping it is misattribution. An
        excluded fork is COUNTED (`forked` in the summary) and nothing else: the `forked` edge that
        used to record the act went with the `edges` table on 2026-08-23, and so did the extra API
        call it needed. See the fork note above `REPOS_PER_RUN`.

      • `min_stars` — "is this work notable?" A lagging popularity proxy (a brand-new repo has 0
        stars however good it is).

    Default `min_stars=0`: a fixed star floor makes both a false-positive and a false-negative
    error on real data (see doc). Authorship has an exact answer, so it's the default filter;
    notability stays opt-in for a caller that wants a highlight reel."""
    from pipeline.ingestion.sources.github import (GitHubRateLimited, _fetch_readme,
                                                   _repo_to_markdown)

    assert_model(conn, embedder)  # fail-fast on model drift before any spend
    seen = schema.load_hashes(conn, "github")

    # Batch the embed across repos (ARC-1 Job A): one shared sink pools many repos' chunks per
    # flush so the process-wide embed gate works a WIDE flush (8-way) instead of a per-repo call
    # re-serializing it. Fetch stays serial — a GitHub-API crawl, no scrape-concurrency change.
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, flush_chunks=8 * bs)
    counts = {"added": 0}            # bumped in on_written — a DURABLE-write count, not a submit count
    submitted = skipped = forked = stale = covered = capped = fetched = 0
    since, before = _aware(since), _aware(before)
    # Two frontiers, and the return dict explains why both are needed. `oldest_fetched` is the
    # oldest repo this run actually spent a call on; `oldest_listed` is the oldest repo the list
    # response showed, fetched or not.
    oldest_fetched: datetime | None = None
    oldest_listed: datetime | None = None

    def _mark() -> None:             # fires AFTER the atom's row commits (never on a poison-skip)
        counts["added"] += 1

    # The crawl is the one GitHub path that CLAIMS COVERAGE, so it is the one that may not
    # swallow a refusal. `_fetch_repo` swallows (one repo, nothing stamped); this does not.
    blocked: str | None = None
    swept = 0
    try:
        for handle in handles:
            if fetched >= REPOS_PER_RUN:
                # The ceiling went in the previous handle. Leaving `swept` behind is what puts
                # this handle in `undetermined`, so a multi-handle run says how many archives it
                # never opened rather than reporting a clean finish.
                break
            # FIRST, before any repo upsert: the owner's declared website, which is what lets
            # `resolve.resolve_entities` fold this GitHub footprint into the Oracle's canonical entity
            # instead of stranding it. See `_seed_owner_identity` for why the order is load-bearing.
            _seed_owner_identity(conn, handle)
            repos = _fetch_handle_repos(handle)
            for repo in repos:
                pushed = _pushed_at(repo)
                if pushed is not None and (oldest_listed is None or pushed < oldest_listed):
                    oldest_listed = pushed
                # Stale first — before the fork branch, so an untouched fork also skips its upstream
                # GET. Both of the two calls this loop makes per repo happen below this line.
                if since is not None and pushed is not None and pushed < since:
                    stale += 1
                    continue
                # Already inside the coverage frontier a resume pass is walking back from. Skipping
                # it here is the whole point of `before`: re-fetching the covered prefix is exactly
                # the spend `REPOS_PER_RUN` exists to bound.
                if before is not None and pushed is not None and pushed >= before:
                    covered += 1
                    continue
                # Forks first — misattribution, not a quality filter: a fork's atom carries the
                # upstream's README/description stamped with this person's who_id. Checked before
                # `_fetch_readme` so a skipped fork costs no README call.
                if repo.get("fork") and not include_forks:
                    forked += 1           # counts forks EXCLUDED, not forks seen — see the return dict
                    continue
                if int(repo.get("stargazers_count", 0)) < min_stars:
                    continue
                owner = (repo.get("owner") or {}).get("login", "") or handle
                name = repo.get("name", "")
                if not name:
                    continue
                # The ceiling sits HERE — after every gate that costs nothing, immediately before
                # the first call this repo would spend. What is left is COUNTED and not fetched, so
                # the summary can say how much of the archive a resume still owes.
                if fetched >= REPOS_PER_RUN:
                    capped += 1
                    continue
                atom_id = f"github:{owner}/{name}"

                readme = _fetch_readme(owner, name)
                # Count the call, not the atom: an unchanged repo below still spent this fetch and
                # is still covered, so both the ceiling and the frontier move here.
                fetched += 1
                if pushed is not None and (oldest_fetched is None or pushed < oldest_fetched):
                    oldest_fetched = pushed
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
            swept += 1
    except GitHubRateLimited as e:
        # DRAIN, never discard: repos already fetched still land below, matching
        # `run_concurrent`'s source-error contract. Stopping costs nothing a retry would
        # not also pay — the limit is per-IP, so every later call fails identically.
        blocked = str(e)

    sink.close()

    # Did the sweep read the whole list, down to its floor? Only then is the FLOOR its reach.
    # A capped run stopped early by our own choice and a blocked one by the host's, and neither
    # reached past the oldest repo it actually fetched.
    to_the_floor = not capped and blocked is None
    # The frontier this run reached, which `oracle_refresh._pull_pair` stamps as `covered_from`
    # and a resume pass hands back as `before`.
    #
    # The floor arm is what keeps a resume from livelocking. A capped run's reach is the oldest
    # repo it fetched; report that same value for an UNCAPPED run and every repo between it and
    # the floor becomes a permanent gap — the next run would find nothing there and the frontier
    # would never move again. An uncapped run looked at everything down to `since` (or, with no
    # `since`, to the end of the archive), and looking is what coverage means.
    #
    # `covered_from` is a per-HANDLE claim. A multi-handle run reports the min across the handles
    # it swept, which is only a frontier when `handles` holds one entry — the shape `_pull_pair`
    # always calls with.
    reached = (since or oldest_listed) if to_the_floor else oldest_fetched

    # added < submitted when the sink isolates a poison-chunk repo (skip-and-continue, not abort).
    # `forked` counts repos excluded as ATOMS — the only record a fork leaves, since the `forked`
    # edge went with the `edges` table. Three counters keep three reasons a repo was never
    # fetched apart, because each has a different remedy: `stale` is behind the forward window
    # (nothing owed), `covered` is behind the backward frontier (already held), `capped` is owed
    # to the next cycle. `skipped` is none of them — it means fetched-and-unchanged.
    out = {"source": "github", "added": counts["added"], "skipped": skipped,
           "forked": forked, "stale": stale, "failed": submitted - counts["added"],
           "total": schema.count_atoms(conn, "github")}
    if covered:
        out["covered"] = covered
    if capped:
        out["capped"] = capped
    if reached is not None:
        out["covered_from"] = reached.isoformat()
    if blocked:
        # `error` + `undetermined` is `classify_run`'s BLOCKED, and BLOCKED is the only outcome
        # that stops `_pull_pair` advancing the cursor, widening `covered_from` and stamping the
        # TTL. Without both keys the run reads as ERROR, which is a caller fault needing a human
        # — a rate limit is neither.
        out["error"] = blocked
    # Handles whose archive was NOT fully walked, so a run that reached three of five reports 2
    # rather than a bare flag. Set for a CAPPED run too, and deliberately without `error`:
    # `classify_run` demotes only on `error`, so this stays an INGESTED run that says what it
    # still owes.
    unswept = len(handles) - swept
    if unswept:
        out["undetermined"] = unswept
    return out


def sync_github_source(conn: sqlite3.Connection, embedder, url: str, *, min_stars: int = 0,
                       since: datetime | None = None,
                       before: datetime | None = None) -> dict | None:
    """ONE discovered GitHub source url → atoms, routed on the url's SHAPE.

    A two-segment url (`github.com/acme/memory`) names one REPOSITORY and mints that repo's own
    atom under `entry_mode='author_referenced'` — the person POINTED at it, they did not
    necessarily write it. Anything else carrying an owner (`github.com/alice`) names an ACCOUNT,
    and its archive is swept under 'oracle-footprint'. The repository test runs FIRST because
    every repository url also contains an owner; asking "who owns this" first is exactly the bug
    this function exists to remove.

    NOT the MCP `hopper` tool for the repository half: Hopper stamps 'user-saved', which asserts
    the USER personally saved the link, and that mode feeds Frontier's query generation. A link
    in someone's bio is that person's claim, not his.

    Returns a `sync_github`-shaped summary (`added` / `error`, read by
    `ingest_common.classify_run`), or None when the url names neither a repository nor an
    account — a real answer, left to each caller's own vocabulary to report.

    `since` and `before` reach the ACCOUNT half only. A repository url names one repo, so a
    window has nothing to select over and `mint_artifact` takes none.

    ONE home for this routing on purpose. `discover_profile._classify_url` tags every github.com
    link `github` and drops `source_classify`'s profile-vs-artifact verdict, so every rail
    consuming a source dict has to re-derive it — and the two that did had already written the
    rule differently by 2026-09-05: `onboard_footprint` took the LAST url segment (crawling an
    account named after the repo) and `expand._route_source` took the FIRST (sweeping the owner's
    whole archive). Only the onboarding one could fire: the refresh rail's own caller synthesizes
    `github.com/{owner}` from an entity id, and `oracle_refresh_state.pair_from_member` refuses an
    `owner/name` id outright. The second copy was therefore a latent divergence, not a live
    second bug — and consolidating is what stops it becoming one when an input shape widens.
    """
    from . import link_router

    if _github_owner_repo(url):
        res = link_router.mint_artifact(conn, embedder, url, "github",
                                        entry_mode="author_referenced")
        if res["status"] == "failed":
            # `mint_artifact` collapses a dead link, a failed fetch and a failed embed into one
            # status, so this cannot say which — it says what it knows: no atom exists.
            return {"added": 0,
                    "error": f"no atom minted for {url} (dead link, or the fetch/embed failed)"}
        # `present` means the repo was already in the store and `mint_artifact` recorded the
        # attestation without a network call. A real route, and zero added.
        return {"added": 1 if res["status"] == "minted" else 0, "atom_id": res["atom_id"]}

    owner = _github_owner(url)
    if not owner:
        return None
    return sync_github(conn, embedder, handles=[owner], min_stars=min_stars, since=since,
                       before=before)
