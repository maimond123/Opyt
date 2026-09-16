"""The singular, local-only removal surface. Consequences first, then explicit confirmation."""
from __future__ import annotations

import sqlite3


def register_forget_tools(mcp) -> None:
    @mcp.tool()
    def forget(atom_id: str | None = None, oracle: str | None = None,
               confirm: bool = False) -> dict:
        """REMOVE one local atom, or STOP TRACKING one Oracle. Specify exactly one target.

        `atom_id` deletes that atom, its chunks, search entries, snapshot and references.
        `oracle` ends one person's subscription and keeps their existing atoms. Use a canonical
        id, locally known handle, URL or exact roster name; ambiguous names require an id.
        There is no bulk deletion and no foreign `kb` target.

        TWO-PHASE: first call with confirm=False. Show the human identity (description, author,
        date and source URL for an atom), then read every `consent` sentence to the user.
        Call confirm=True with the SAME target only after the user approves those consequences.
        Never treat approval to stop tracking a person as approval to delete their atoms.

        A bookmarked X post can return on an hourly pass until unbookmarked on x.com; a tracked
        author's footprint can return on a due refresh. Read these warnings before confirming.
        Shared copies lose the atom after a successful push with reader demand, not immediately.

        Returns status=preview, forgotten, not_found, ambiguous, busy, or error, with scope and
        identity. A busy Oracle refresh changes nothing; retry after the current pass finishes.
        """
        from opyt_core.paths import opyt_db
        from pipeline.credentials import get_credential
        from pipeline.kb import forget as removal, oracles, schema

        if (atom_id is None) == (oracle is None):
            return {"status": "error", "message": "Specify exactly one of atom_id or oracle."}
        target = atom_id if atom_id is not None else oracle
        if not target.strip():
            return {"status": "error", "message": "The target must not be empty."}
        if not opyt_db().exists():
            return {"status": "not_found", "scope": "atom" if atom_id is not None else "oracle"}
        conn = schema.connect(read_only=not confirm)
        try:
            if atom_id is not None:
                return removal.atom(conn, atom_id, confirm=confirm,
                                    shared=bool(get_credential("opyt_service")))
            return oracles.forget(conn, oracle.strip(), confirm=confirm)
        except (OSError, sqlite3.Error, ValueError) as exc:
            return {"status": "error", "message": str(exc)}
        finally:
            conn.close()
