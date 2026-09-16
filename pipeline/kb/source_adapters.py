"""
pipeline/kb/source_adapters.py

The registry for footprint source types a confirmed person's identity can route to — substack
and blog today. `gate_and_sync_website` owns the "gate this website's single-authorship, then hand
it to its sync_* adapter" shape that was duplicated in `expand.py` and `onboard_footprint.py`
before this file existed; both now call it, so the gate-then-sync order lives in exactly one
place and a caller cannot forget the gate by editing its own copy. (A third copy sat in the
`run_ingest.py` ingest CLI, deleted 2026-09-08 with every other hand-run entry point.)

GitHub is deliberately absent from `WEBSITE_ADAPTERS`: `sync_github` attributes to the attested
repo owner, never the person, so there is no authorship to gate, and an unknown key raises
`KeyError` rather than silently skipping the gate. Other source-type-shaped registries
(`link_router.py`, `discover_profile.py`'s `seed_type`) use different vocabularies and do not
share this one. See `docs/plans/2026-08-16-refactoring-execution-progress.md` step 9 for the
measurement that scoping is based on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from . import eligibility, ingest_blog, ingest_substack
from .eligibility import GateDecision


@dataclass(frozen=True)
class WebsiteAdapter:
    """One website source type's ingest callable. The registry KEY is the source type — carrying
    it on the value too gave the same fact two homes that had to be kept in agreement, and nothing
    ever read the copy."""

    # Normalized signature: (conn, embedder, url, *, author_name, since, limit, handle) -> summary
    sync: Callable[..., dict]


def _substack_sync(conn, embedder, url: str, *, author_name: str | None = None,
                   since: datetime | None = None, limit: int = 0,
                   handle: str | None = None) -> dict:
    return ingest_substack.sync_substack_footprint(
        conn, embedder, publication_url=url, handle=handle,
        author_name=author_name, since=since, limit=limit)


def _blog_sync(conn, embedder, url: str, *, author_name: str | None = None,
               since: datetime | None = None, limit: int = 0,
               handle: str | None = None) -> dict:
    return ingest_blog.sync_blog_footprint(
        conn, embedder, blog_url=url, handle=handle,
        author_name=author_name, since=since, limit=limit)


# Both adapters attribute a whole site to `who_id = the person` by inference, so both must pass
# the single-author eligibility gate first — a multi-author/org site would otherwise launder its
# other authors onto one trusted person. GitHub is not a key here at all (see module docstring).
WEBSITE_ADAPTERS: dict[str, WebsiteAdapter] = {
    "substack": WebsiteAdapter(_substack_sync),
    "blog": WebsiteAdapter(_blog_sync),
}


def gate_and_sync_website(conn, embedder, source_type: str, url: str, *,
                          author_name: str | None = None, since: datetime | None = None,
                          limit: int = 0, force: bool = False,
                          handle: str | None = None) -> tuple[GateDecision, dict | None]:
    """Gate ONE website source, then run its adapter — the shape duplicated near-identically in
    `expand.py` and `onboard_footprint.py` before this existed, and the ONLY production path to a
    website adapter now that both route through it.

    Returns `(decision, summary)`. `summary` is None whenever `decision.decision != "ingest"` —
    the caller decides what a refusal MEANS for it (return a skip record, print + maybe record an
    affiliation, mark needs-review) because that bookkeeping genuinely differs per caller; this
    function only owns the identical gate-then-sync core, not what surrounds it.

    `source_type` must be a `WEBSITE_ADAPTERS` key — raises `KeyError` otherwise, deliberately (see
    the module docstring on why GitHub must not be reachable through this seam at all)."""
    adapter = WEBSITE_ADAPTERS[source_type]
    decision = eligibility.gate(conn, url, expected_author=author_name, force=force)
    if decision.decision != "ingest":
        return decision, None
    summary = adapter.sync(conn, embedder, url, author_name=author_name,
                           since=since, limit=limit, handle=handle)
    return decision, summary
