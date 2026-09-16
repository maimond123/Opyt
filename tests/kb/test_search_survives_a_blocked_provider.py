"""`search` must answer, not raise, when the model provider refuses the query.

⚠️ THE HOLE THIS CLOSES. Measured 2026-09-15 alongside the allowance failure: with the key
spent, `atom_semantic_search` asked OpenRouter to embed the query, the exception travelled out
through `run_kb_search` — which has a `finally` and no `except` — and the most-used tool in the
product answered with a traceback, over a store holding 1,004 atoms the keyword arm would have
found perfectly well. A tool that dies says "this product is broken"; a tool that returns
keyword-ranked hits and one sentence saying why says what is actually true.
"""
from __future__ import annotations

import pytest

from opyt_core import kb as kb_entry
from pipeline.circuit_breaker import CircuitBreaker
from pipeline.kb import allowance_notice as an
from pipeline.kb import schema
from pipeline.kb.embed import EmbedError
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot
from pipeline.kb.retrieve import search_atoms


class RefusesTheQuery:
    """Embeds documents, refuses queries — the exact asymmetry a spent key produces. The store
    was built while the allowance was alive; only the query comes after it ran out."""

    def __init__(self, inner):
        self._inner = inner
        self.dim = inner.dim
        self.provider = inner.provider
        self.model = inner.model
        self.query_instruction = inner.query_instruction

    def embed(self, texts, *, role: str = "document"):
        if role == "query":
            raise EmbedError('HTTP 403: {"error":{"message":"Key limit exceeded (total limit)"}}',
                             retryable=False)
        return self._inner.embed(texts, role=role)


def _store(conn, emb):
    for atom_id, text in (("github:a/agentkit", "an autonomous agent framework with tools"),
                          ("github:b/agents", "an agent framework library"),
                          ("x:1", "thoughts on rollup and proof systems")):
        raw_ref, raw_hash = write_snapshot("github", atom_id, text)
        store_atom(conn, emb, atom=dict(
            atom_id=atom_id, source_type="github", what_kind="artifact",
            who_id="github:a", when_ts="2024-05-01", when_precision="day", about_entities=[],
            source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
            description=f"{atom_id} card", payload={}, entry_mode="user-saved",
        ), snapshot_text=text)


def test_hybrid_loses_one_arm_and_still_answers(kb_home, fake_embedder):
    """The measured case. Half the search is gone; the half that needs no provider still ranks."""
    conn = schema.connect()
    _store(conn, fake_embedder)

    run = search_atoms(conn, "agent framework", RefusesTheQuery(fake_embedder), k=8)

    assert run.hits, "the keyword arm needs no provider and must still return the store's atoms"
    assert run.effective_mode == "bm25", "`effective_mode` has always named the arms that RAN"
    assert "Key limit exceeded" in run.vector_arm_error
    conn.close()


def test_an_explicitly_semantic_search_falls_back_rather_than_returning_nothing(
        kb_home, fake_embedder):
    """`mode="semantic"` asked for the arm that is gone. Answering with the other one is what
    `_query_embedder`'s own degrade already does for a foreign store, so both routes to a missing
    vector arm behave the same way — and an empty list would read as "your store has nothing"."""
    conn = schema.connect()
    _store(conn, fake_embedder)

    run = search_atoms(conn, "agent framework", RefusesTheQuery(fake_embedder),
                       mode="semantic", k=8)

    assert run.hits
    assert run.effective_mode == "bm25"
    assert run.vector_arm_error is not None
    conn.close()


def test_a_store_that_disagrees_with_itself_still_raises(kb_home, fake_embedder, monkeypatch):
    """⚠️ THE NARROWNESS IS THE POINT. `QueryVectorError` wraps the query embed and nothing else.
    The other way the vector arm fails is a width disagreement between the stored vectors and the
    store's own `kb_meta` — real corruption — and degrading THAT to keyword search would hide a
    bug behind a slightly thinner answer. A provider the user can pay is not a corrupt store."""
    conn = schema.connect()
    _store(conn, fake_embedder)

    def boom(_conn):
        raise ValueError("kb_meta says 768, the blobs say 1024")

    monkeypatch.setattr("pipeline.kb.embed.stored_dtype", boom)
    with pytest.raises(ValueError):
        search_atoms(conn, "agent framework", fake_embedder, k=8)
    conn.close()


def test_the_answer_says_which_half_ran(kb_home, fake_embedder):
    """A reader handed BM25 scores who thinks they got hybrid ones will misread a conceptual
    question that found nothing as "the store does not have it"."""
    conn = schema.connect()
    _store(conn, fake_embedder)
    conn.close()

    out = kb_entry.run_kb_search("agent framework", k=8,
                                 embedder=RefusesTheQuery(fake_embedder))

    codes = [n.get("code") for n in out["notices"]]
    assert "vector_arm_unavailable" in codes
    assert out["trace"]["ran"] == "bm25"


def test_the_notice_names_the_cause_and_the_remedy(kb_home, fake_embedder, monkeypatch):
    """The two notices ride together and say different things: one tells the reader what they
    GOT (keyword ranking), the other why and what ends it."""
    from mcp_server import atoms_tools

    monkeypatch.setattr(an, "_CACHE", None)
    monkeypatch.setattr("opyt_core.readiness.openrouter",
                        lambda: {"state": "trial_over", "message": "The starter allowance is up."})
    breaker = CircuitBreaker("openrouter")
    for _ in range(breaker.threshold):
        breaker.record_failure("HTTP 403: Key limit exceeded (total limit)")

    out = {"notices": [{"code": "vector_arm_unavailable", "message": "half of it ran"}]}
    atoms_tools._attach_allowance_notice(out)

    blocked = [n for n in out["notices"] if n.get("code") == "model_provider_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["next_call"] == "onboard(start='openrouter')"


def test_a_healthy_search_carries_no_such_notice(kb_home, fake_embedder, monkeypatch):
    """The gate is the DEGRADE, not the call. A search that kept both arms must stay clean —
    otherwise the field becomes furniture and the reader learns to skip it."""
    from mcp_server import atoms_tools

    monkeypatch.setattr(an, "_CACHE", None)
    out = {"notices": []}
    atoms_tools._attach_allowance_notice(out)
    assert out["notices"] == []


def test_the_raw_transport_blob_never_reaches_the_copy(kb_home, fake_embedder):
    """A host reading `vector_arm_unavailable` out lands whatever is interpolated into it in
    front of a person. A `SubspaceError` reads as a sentence and belongs there; a refused
    provider arrives as `EmbedError: HTTP 403: {"error":{"message":...}}` and does not. The cause
    still travels on `reason` for anyone debugging."""
    conn = schema.connect()
    _store(conn, fake_embedder)
    conn.close()

    out = kb_entry.run_kb_search("agent framework", k=8,
                                 embedder=RefusesTheQuery(fake_embedder))
    notice = next(n for n in out["notices"] if n["code"] == "vector_arm_unavailable")

    assert "403" not in notice["message"] and "{" not in notice["message"]
    assert "Key limit exceeded" in notice["reason"]


def test_the_keyword_only_caveat_is_asked_for_not_merely_offered(kb_home, monkeypatch):
    """`vector_arm_unavailable` carries no `host_note`, so without this the reader whose
    conceptual question now finds nothing concludes their library does not have it."""
    from mcp_server import atoms_tools

    monkeypatch.setattr(an, "_CACHE", None)
    monkeypatch.setattr("opyt_core.readiness.openrouter",
                        lambda: {"state": "trial_over", "message": "The allowance is up."})
    breaker = CircuitBreaker("openrouter")
    for _ in range(breaker.threshold):
        breaker.record_failure("HTTP 403: Key limit exceeded (total limit)")

    out = {"notices": [{"code": "vector_arm_unavailable", "message": "half of it ran"}]}
    atoms_tools._attach_allowance_notice(out)

    note = next(n for n in out["notices"] if n["code"] == "model_provider_blocked")["host_note"]
    assert "only the keyword half of the search ran" in note
    assert "do not use the words 'bm25'" in note.lower()
