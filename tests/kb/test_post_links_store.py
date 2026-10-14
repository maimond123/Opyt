"""discovery.post_links.own_post_candidates — read from the store, for one Oracle's cluster.

Only the Oracle's own `oracle-footprint` X posts count: a post they saved (`user-saved`) or a post
by someone else is not theirs. A post with no stored body is skipped, not raised.
"""
from __future__ import annotations

import pytest

from pipeline.kb import resolve, schema
from pipeline.kb.ingest_common import snapshot_hash, store_atom
from pipeline.oracles.discovery.post_links import own_post_candidates

X_ID = "x:user:1761533439566639105"
OTHER = "x:user:42"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _post(atom_id: str, author: str, body: str) -> str:
    return f'---\nauthor: "@{author}"\n---\n\n{body}\n'


def _atom(conn, atom_id, who_id, text, entry_mode="oracle-footprint") -> None:
    raw_hash = snapshot_hash(text)
    store_atom(conn, dict(
        atom_id=atom_id, source_type="x", what_kind="opinion", who_id=who_id,
        when_ts="2026-10-01", when_precision="day", about_entities=[],
        source_url=f"https://x.com/i/status/{atom_id}", raw_hash=raw_hash,
        description=atom_id, payload={}, entry_mode=entry_mode,
    ), text)


def test_counts_only_the_oracles_own_posts(conn):
    schema.upsert_entity(conn, X_ID, name="Alex Imas", profile={"handle": "alexolegimas"})
    schema.upsert_entity(conn, OTHER, name="Someone", profile={"handle": "someone"})
    resolve.resolve_entities(conn)
    _atom(conn, "xprofile:1", X_ID,
          _post("1", "alexolegimas", "new essay https://aleximas.substack.com/p/a"))
    _atom(conn, "xprofile:2", X_ID,
          _post("2", "alexolegimas", "again https://open.substack.com/pub/aleximas/p/b"))
    _atom(conn, "xprofile:3", X_ID,
          _post("3", "alexolegimas", "saved https://saved.dev/x"), entry_mode="user-saved")
    _atom(conn, "xprofile:4", OTHER,
          _post("4", "someone", "mine https://someone.dev"))

    got = own_post_candidates(conn, X_ID)
    assert [(c.key, c.post_count) for c in got] == [("aleximas.substack.com", 2)]


def test_a_post_with_no_body_is_skipped(conn):
    schema.upsert_entity(conn, X_ID, name="Alex Imas", profile={"handle": "alexolegimas"})
    resolve.resolve_entities(conn)
    _atom(conn, "xprofile:1", X_ID,
                _post("1", "alexolegimas", "https://aleximas.substack.com/p/a"))
    _atom(conn, "xprofile:2", X_ID, _post("2", "alexolegimas", "https://rhys.dev"))
    conn.execute("DELETE FROM bodies WHERE atom_id = 'xprofile:1'")

    assert [c.key for c in own_post_candidates(conn, X_ID)] == ["rhys.dev"]


def test_an_oracle_with_no_x_has_no_candidates(conn):
    schema.upsert_entity(conn, "blog:rhys.dev", name="Rhys", identity_links=["https://rhys.dev"])
    resolve.resolve_entities(conn)
    assert own_post_candidates(conn, "blog:rhys.dev") == []
