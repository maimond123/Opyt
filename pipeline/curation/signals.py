"""One person's signal, written with their entity row.

Presence may come from partial work; a count only from a complete walk. The writers stay in
`schema` and commit one person at a time, so they cannot join a caller's `write_tx`. Every write here is idempotent instead, and a caller orders
them before the state that says they happened, so a crash in between repeats them harmlessly.
"""

import sqlite3
from dataclasses import dataclass

from pipeline.kb import schema


@dataclass(frozen=True)
class Person:
    entity_id: str
    name: str | None = None
    site: str | None = None
    profile: dict | None = None


@dataclass(frozen=True)
class Tally:
    """One person's total in a complete walk, and the context the signal carries."""
    person: Person
    count: int
    extra: dict | None = None


def _entity(conn: sqlite3.Connection, person: Person) -> None:
    schema.upsert_entity(conn, person.entity_id, name=person.name,
                         identity_links=[person.site] if person.site else None,
                         profile=person.profile)


def assert_person(conn: sqlite3.Connection, person: Person, signal_type: str,
                  platform: str) -> None:
    """One person's entity row and their signal asserted, whether or not they hold it already:
    the entity's name and links refresh on every call."""
    _entity(conn, person)
    schema.ensure_signal(conn, person.entity_id, signal_type, platform)


def author(meta: dict) -> Person:
    """The person a Substack post's derived meta (`derive.derive_substack`) names as its
    author. X's saved posts name theirs through `x_collectors.bookmark_author`."""
    return Person(meta["who_id"], name=meta.get("who_name"), site=meta.get("who_site") or None)


def assert_saved_author(conn: sqlite3.Connection, person: Person, platform: str) -> None:
    """A saved post's author, asserted with `save` presence once the post's own transaction has
    committed. Presence, not a count: the saved listing's complete walk is the counting
    authority. After the commit, because these writers commit by themselves, so a refused or
    unsaved post leaves no author behind; the listing's next walk repairs a crash in between."""
    assert_person(conn, person, "save", platform)


def oracle_ids(conn: sqlite3.Connection) -> set[str]:
    """The current Oracles' heads, read once per pass for `is_oracle`."""
    return {r["head"] for r in conn.execute(schema.ROSTER_HEADS_SQL)}


def is_oracle(conn: sqlite3.Connection, entity_id: str, oracles: set[str]) -> bool:
    """Is this person already an Oracle, by their own id or the canonical one it resolves to? A
    signal drawn from Oracles' work skips them: a person is not their own candidate."""
    return entity_id in oracles or schema.current_canonical(conn, entity_id) in oracles


def presence(conn: sqlite3.Connection, people: list[Person], signal_type: str,
             platform: str) -> int:
    """Assert the signal for each person who lacks it, and say how many were written. A person
    who already holds it is only read, so re-asserting a whole list takes no write lock."""
    missing = [p for p in people if conn.execute(
        "SELECT 1 FROM curation_signals WHERE entity_id=? AND signal_type=? AND platform=?",
        (p.entity_id, signal_type, platform)).fetchone() is None]
    for person in missing:
        assert_person(conn, person, signal_type, platform)
    return len(missing)


def counts(conn: sqlite3.Connection, tallies: list[Tally], signal_type: str,
           platform: str) -> None:
    """Replace each person's count with a complete walk's total, confirming the signal."""
    for tally in tallies:
        _entity(conn, tally.person)
        schema.set_signal(conn, tally.person.entity_id, signal_type, platform,
                          count=tally.count, extra=tally.extra)


def stored(conn: sqlite3.Connection, signal_type: str, platform: str) -> int:
    """How many people the store holds this signal for: what actually landed, beside what a
    collector says it saw."""
    return conn.execute(
        "SELECT COUNT(*) FROM curation_signals WHERE signal_type=? AND platform=?",
        (signal_type, platform)).fetchone()[0]
