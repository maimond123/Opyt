"""Publish a prepared thread, and complete its work, only while its input atom and Oracle still
exist unchanged and the lease holds: the atom and its completion commit together."""

from typing import Callable

from pipeline.kb.ingest_common import store_atom
from pipeline.kb.writes import write_tx
from pipeline.worker.quota_loop.source import Changed
from pipeline.oracles.sources.threads import state


def _refuse_lost(ctx) -> None:
    if ctx.lease_lost():
        raise Changed("thread quota lease lost")


def publish(conn, ctx, work: state.Work, atom: dict, markdown: str,
            finish: Callable[[state.Work], None]) -> None:
    """Write the thread atom, then `finish` the written atom's work, in one transaction."""
    with write_tx(conn, site="threads.atom"):
        current = state.require_current(conn, work)
        _refuse_lost(ctx)
        if "links_owed" in atom["payload"]:
            links = list(atom["payload"]["links_owed"])
            for entry in current.payload.get("links_owed") or []:
                if entry not in links:
                    links.append(entry)
            atom["payload"]["links_owed"] = links
        atom["payload"] = {**current.payload, **atom["payload"]}
        store_atom(conn, atom, markdown)
        finish(state.read(conn, work.atom_id))


def complete(conn, ctx, work: state.Work, finish: Callable[[state.Work], None]) -> None:
    """`finish` a thread whose atom needs no write (gone, or unchanged), while the lease holds."""
    with write_tx(conn, site="threads.complete"):
        _refuse_lost(ctx)
        finish(work)
