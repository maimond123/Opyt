"""Publish linked content while its referring posts still exist; models run outside transactions.
A source fetches its targets first, holding each atom, then writes them and completes their links
in one transaction."""

from pipeline.kb.ingest_common import store_atom
from pipeline.kb.schema import stored_atom_id
from pipeline.worker.quota_loop.source import Changed
from pipeline.oracles.sources.references import state


class Held(list):
    """The atoms a fetch made, held for `write`: `(atom, text, on_written)`. Called as an
    adapter's `store`, it holds instead of writing."""

    def __call__(self, conn, atom, text, *, on_written=None) -> None:
        self.append((atom, text, on_written))


def write(conn, ctx, held: Held, links_for) -> None:
    """Inside the caller's `write_tx`: the held targets, while the lease holds and each still
    has a referring post that owes it. A target another lane saved or promoted meanwhile is left
    as that lane wrote it."""
    if ctx.lease_lost():
        raise Changed("references quota lease lost")
    for atom, _text, _cb in held:
        if not any(state.current(conn, link) for link in links_for(atom["atom_id"])):
            raise Changed(f"no referring post still owes {atom['atom_id']}")
    for atom, text, cb in held:
        if not stored_atom_id(conn, atom["atom_id"]):
            store_atom(conn, atom, text, on_written=cb)
