"""A free install (no key) ingests and searches.

Every atom lands with its body and its keyword index, and nothing on the ingest path asks for a
model: search is keyword-only for everyone.
"""
from __future__ import annotations

import pytest

from pipeline import model_routing
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.retrieval import engine


@pytest.fixture()
def free(kb_home, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return kb_home


def _mk_atom(aid: str) -> dict:
    return dict(atom_id=aid, source_type="x", what_kind="opinion", who_id="x:u",
                when_ts="2024-01-01", when_precision="day",
                about_entities=[], source_url="u", raw_hash="h",
                description="d", payload={}, entry_mode="user-saved")


def test_a_free_install_writes_every_atom_and_finds_it_by_keyword(free):
    conn = schema.connect()
    written: list[str] = []
    for aid, word in (("x:a", "rollups"), ("x:b", "sequencers")):
        store_atom(conn, _mk_atom(aid), f"a post about {word} " * 20,
                   on_written=lambda a=aid: written.append(a))

    assert sorted(written) == ["x:a", "x:b"]                # both durable, both counted done
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 2
    run = engine.search_atoms(conn, "rollups")
    assert [h.atom_id for h in run.hits] == ["x:a"]
    conn.close()


def test_the_rail_preflight_skips_on_a_free_install(free, monkeypatch):
    def boom(**_kw):
        raise AssertionError("a free install has no model spend to preflight")
    monkeypatch.setattr(model_routing, "preflight", boom)
    assert model_routing.models_unroutable("oracle_refresh") is None
