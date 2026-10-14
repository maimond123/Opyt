"""The singular, local-only removal surface. Consequences first, then explicit confirmation."""
from __future__ import annotations

from pipeline.forget import request


def register_forget_tools(mcp) -> None:
    @mcp.tool(annotations={"title": "Forget an item, Oracle or source", "readOnlyHint": False,
                           "destructiveHint": True, "openWorldHint": False})
    def forget(atom_id: str | None = None, atom_ids: list[str] | None = None,
               oracle: str | None = None, source: str | None = None,
               delete_atoms: bool = False, confirm: bool = False) -> dict:
        """REMOVE local atoms, STOP TRACKING one Oracle, or DROP ONE SOURCE of an Oracle.
        Specify exactly one of atom_id, atom_ids, oracle or source.

        `atom_id` deletes that atom, its text, search entry and references.
        `atom_ids` does the same for several atoms in one preview and one confirm: when the
        user wants more than one atom gone, pass them all in one list rather than one call per
        atom. An unknown id refuses the whole list and nothing is removed.
        `oracle` ends one person's subscription and keeps their existing atoms. Use a canonical
        id, locally known handle, URL or exact roster name; ambiguous names require an id.
        `source` is one of an Oracle's other places, by URL: a Substack, blog or GitHub Opyt
        found and added for them. Opyt stops pulling it and never adds it back, even if their
        profile still links it; their other sources keep going. When the same source belongs to
        two Oracles, also pass `oracle` to choose. The profile they were added by is not a
        source: to drop that, forget the Oracle.
        `delete_atoms=True` (with `oracle` or `source`) also deletes the atoms Opyt pulled from
        that person's sources, or from that one source; atoms the user saved themselves stay.
        There is no foreign `kb` target.

        TWO-PHASE: first call with confirm=False. Show the human identity (description, author,
        date and source URL of each atom), then read every `consent` sentence to the user.
        Call confirm=True with the SAME target and the SAME `delete_atoms` only after the user
        approves those consequences. Never treat approval to stop tracking a person as approval
        to delete their atoms: set `delete_atoms` only when the user asked for their atoms gone.

        A bookmarked X post can return on an hourly pass until unbookmarked on x.com; a tracked
        author's footprint can return on a due refresh. Read these warnings before confirming.
        Shared copies lose the atom after a successful push with reader demand, not immediately.

        Returns status=preview, forgotten, not_found, ambiguous, or error, with scope and
        identity (for `atom_ids`, an `atoms` list, and `unknown` when not_found); with
        `delete_atoms`, `atoms_deleted` (and `atoms_not_deleted` with `errors` if any could not
        be removed).
        """
        return request.forget(atom_id=atom_id, atom_ids=atom_ids, oracle=oracle, source=source,
                              delete_atoms=delete_atoms, confirm=confirm)
