"""Saved Substack posts, saved as full-body user-saved atoms with their `save` presence.

`SavedSubstackPosts` saves one post per unit from the work the Saved Posts listing offered
(`saved_items`), preparing it through `fetch_body` and `prepare`.

What the stored body claims is the point of this module, because `export` ships `body_state` to
shared KBs:
  pending   the body fetch was refused (a Cloudflare block): owed, retried
  partial   a paid post read without an accepted session: Substack's teaser, known to be short
  complete  the body arrived, over a session the host accepted where the post is paid
  absent    there was no body to fetch
"""

from __future__ import annotations

from typing import NamedTuple
from urllib.parse import urlsplit

from pipeline.curation import saved_items, saving, signals, substack_render, substack_walks
from pipeline.ingestion.utils import SyncAuthError, log
from pipeline.kb import derive
from pipeline.kb.ingest_common import (BASIS_OBSERVED, BASIS_STATED, BODY_ABSENT,
                                      BODY_COMPLETE, BODY_PARTIAL, BODY_PENDING, body_fields,
                                      changed_hash, store_atom)
from pipeline.kb.vision import enrich_markdown_images
from pipeline.kb.writes import write_tx
from pipeline.platforms import sessions
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.platforms.substack import publication as substack_publication
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import Outages
from pipeline.worker.quota_loop.source import YOURS, Changed, Outcome, Unit

PLATFORM = substack_walks.PLATFORM


class Body(NamedTuple):
    md: str          # the cleaned body; "" when none arrived
    got: bool        # a non-empty body arrived
    blocked: bool    # the fetch was refused: the body is owed, not absent


def fetch_body(source, rec: dict) -> Body:
    """One post's full body over the saved list's transport (doing). A refused fetch is
    `blocked`, never an empty post; any other per-post failure keeps the stub."""
    base, slug = rec.get("publication_url") or "", rec.get("slug") or ""
    if not (base and slug):
        return Body("", False, False)
    try:
        full = source.full_post(base, slug)
        md = substack_render.clean_body_html((full or {}).get("body_html") or "")
        return Body(md, bool(md.strip()), False)
    except substack_contracts.SubstackFetchError as e:
        # A block, not an empty post: the stub is stored `pending` so it is fetched again rather
        # than freezing a stale block.
        log(f"[curation] substack full-body BLOCKED for {slug!r} (stub kept, RETRYABLE): {e}")
        return Body("", False, True)
    except SyncAuthError:
        raise                          # no session: nothing about this post is known
    except Exception as e:             # per-post failure keeps the stub, never aborts the run
        log(f"[curation] substack full-body fetch failed for {slug!r}: {e}")
        return Body("", False, False)


def body_state(body: Body, preview_only: bool) -> tuple[str, str]:
    """What a saved post's body is: `(body_state, body_basis)` for its payload (deciding).

    | Body fetch | Paid teaser (`preview_only`) | Result               |
    |------------|------------------------------|----------------------|
    | blocked    | —                            | `pending`, observed  |
    | got a body | yes                          | `partial`, stated    |
    | got a body | no                           | `complete`, stated   |
    | no body    | —                            | `absent`, observed   |
    """
    if body.blocked:
        return BODY_PENDING, BASIS_OBSERVED
    if preview_only:
        return BODY_PARTIAL, BASIS_STATED
    if body.got:
        return BODY_COMPLETE, BASIS_STATED
    return BODY_ABSENT, BASIS_OBSERVED


class Prepared(NamedTuple):
    atom: dict
    md: str
    meta: dict


def prepare(rec: dict, body: Body, authenticated: bool, seen: dict,
            img_cache: dict) -> Prepared | None:
    """The atom a saved post comes to, or None when its snapshot is unchanged (a `pending` stub
    blocked again). `authenticated`: the body fetch carried a session the host accepted.

    `paywalled` is only Substack's audience flag (`audience == "only_paid"`); a missing body is
    `body_state`'s fact. A paid post fetched with no accepted session is Substack's teaser, so it
    is `partial`, never `complete`. Judged per publication, since only `*.substack.com` honors the
    session the local transport sends everywhere."""
    atom_id = f"substack:{rec['id']}"
    paywalled = bool(substack_publication.is_paywalled(rec))
    preview_only = body.got and paywalled and not authenticated
    md = substack_render.saved_atom_markdown(rec, body.md if body.got else "")
    # Describe inline images before hashing, so the `*Image:* …` text is inside the hashed +
    # indexed text. No-op on the stub path (no body, no refs).
    if body.got:
        md, _ = enrich_markdown_images(md, img_cache, entry_mode="user-saved",
                                       context=rec.get("title") or "")
    raw_hash = changed_hash(atom_id, md, seen)
    if raw_hash is None:
        return None
    meta = derive.derive_substack(rec)
    atom = {
        "atom_id": atom_id,
        "source_type": "substack",
        "what_kind": "opinion",
        "who_id": meta["who_id"],
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": rec.get("url"),
        "raw_hash": raw_hash,
        "description": meta["description"],
        # All four body states are live here; `paywalled` is orthogonal to every one of them —
        # it is Substack's own audience flag, not a statement about the body.
        "payload": {"word_count": rec.get("wordcount", 0), "paywalled": paywalled,
                    **body_fields(*body_state(body, preview_only))},
        "entry_mode": "user-saved",
    }
    return Prepared(atom, md, meta)


# ── One saved post per unit ─────────────────────────────────────────────────────

def _stored(conn, atom_id: str) -> tuple[str, str | None] | None:
    """`(raw_hash, body_state)` of a stored atom, or None."""
    row = conn.execute("SELECT raw_hash, json_extract(payload, '$.body_state') FROM atoms "
                       "WHERE atom_id=?", (atom_id,)).fetchone()
    return (row[0], row[1]) if row else None


def _refuse(conn, saved: saved_items.Saved, atom_id: str, stored, lease_lost) -> None:
    """Every refusal before a saved post's write, inside its transaction."""
    saving.require_lease(lease_lost)
    if _stored(conn, atom_id) != stored:
        raise Changed(f"{atom_id} changed while its body was fetched")
    saved_items.require_current(conn, saved)


def save_one(conn, source, saved: saved_items.Saved, img_cache: dict, lease_lost) -> str:
    """Fetch and save one saved post: its atom, its author's `save` presence and its done-state.
    A stored post is done unless its body is owed. A refused body is stored as a `pending` stub
    and the post stays owed. Returns what the post came to."""
    rec, atom_id = saved.data, f"substack:{saved.item}"
    stored = _stored(conn, atom_id)
    if stored is not None and stored[1] != BODY_PENDING:
        with write_tx(conn, site="saved_substack.done"):
            _refuse(conn, saved, atom_id, stored, lease_lost)
            saved_items.settle(conn, saved)
        return "present"
    body = fetch_body(source, rec)
    authenticated = source.authenticated_body(rec.get("publication_url") or "")
    # Only a stub blocked again may skip as unchanged. Any other fetch of a `pending` stub
    # settles it, even when its markdown is the same stub, because its `body_state` changes.
    seen = {atom_id: stored[0]} if stored and body.blocked else {}
    prepared = prepare(rec, body, authenticated, seen, img_cache)
    if prepared is None:
        return "body_pending"           # blocked again: the same stub stays
    with write_tx(conn, site="saved_substack.atom"):
        _refuse(conn, saved, atom_id, stored, lease_lost)
        store_atom(conn, prepared.atom, prepared.md)
        if not body.blocked:
            saved_items.settle(conn, saved)
    signals.assert_saved_author(conn, signals.author(prepared.meta), PLATFORM)
    return "body_pending" if body.blocked else "saved"


def _host(saved: saved_items.Saved) -> str | None:
    """A saved post's publication host, where the web pool admits its body fetch."""
    return urlsplit(saved.data.get("publication_url") or "").hostname or None


class SavedSubstackPosts:
    """One saved Substack post per unit, from the work the Saved Posts listing offered, in the
    web pool on the publication's own host. It never reads the list, so a refused body never
    costs another listing request."""
    name, klass, floor = "saved_substack_posts", YOURS, 0

    def __init__(self, quota: str, outages: Outages) -> None:
        self.quota = quota
        self.outages = outages

    def _owed(self, conn, now) -> list[Unit]:
        if substack_walks.paused_until(self.outages, now) is not None:
            return []
        return saving.owed_units(conn, PLATFORM, "saved-substack", now, host=_host)

    def candidates(self, conn, now):
        owed = self._owed(conn, now)
        return owed if owed and substack_walks.connected() else []

    def next_unit(self, conn, now):
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit, ctx):
        if (PLATFORM, "saved") not in ctx.cache:
            ctx.cache[(PLATFORM, "saved")] = sessions.saved_source()
        with saving.image_cache() as img_cache:
            try:
                return save_one(conn, ctx.cache[(PLATFORM, "saved")], unit.payload, img_cache,
                                ctx.lease_lost)
            except SyncAuthError as exc:
                ctx.cache.pop((PLATFORM, "saved"), None)
                raise substack_walks.not_signed_in(exc) from exc

    def record(self, conn, unit, fetched):
        if fetched == "body_pending":
            # The host refused the body after the fetch's own retries: a block, which waits as an
            # account and counts toward the shared Substack outage.
            return Outcome("body_pending", failure=retries.ACCOUNT, outage=True, answered=False,
                           detail="the post's body was refused; its stub is stored pending")
        return Outcome(fetched)

    def next_due(self, conn, now):
        due = saving.next_due(conn, PLATFORM)
        paused = substack_walks.paused_until(self.outages, now)
        return max(due, paused) if due is not None and paused is not None else due
