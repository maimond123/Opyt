"""
pipeline/kb/source_adapters.py

The registry for footprint source types a confirmed person's identity can route to — substack
and blog today. Collapses the "gate this website's single-authorship, then hand it to its
sync_* adapter" shape that was duplicated in `expand.py`, `run_ingest.py`, and
`onboard_footprint.py` before this file existed.

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
    source_type: str
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
    "substack": WebsiteAdapter("substack", _substack_sync),
    "blog": WebsiteAdapter("blog", _blog_sync),
}


def gate_and_sync_website(conn, embedder, source_type: str, url: str, *,
                          author_name: str | None = None, since: datetime | None = None,
                          limit: int = 0, force: bool = False,
                          handle: str | None = None) -> tuple[GateDecision, dict | None]:
    """Gate ONE website source, then run its adapter — the shape duplicated near-identically in
    `expand.py`, `run_ingest.py`, and `onboard_footprint.py` before this existed.

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
