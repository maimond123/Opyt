"""The `SourceAdapter` registry (step 9 of the refactor-composability audit) — pins that BOTH
website adapters actually gate, and that GitHub cannot reach this seam at all.

`eligibility.gate` and the two `sync_*` functions are monkeypatched (no network, no DB writes):
this file is about the ROUTING contract (does a refusal skip the sync? does the right adapter get
called with the right normalized kwargs?), not about re-proving the gate's own classify logic —
that is `test_eligibility.py`'s job."""
from __future__ import annotations

import pytest

from pipeline.kb import eligibility, ingest_blog, ingest_substack, source_adapters


@pytest.fixture()
def conn(kb_home, tmp_path):
    from pipeline.kb import schema
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _fake_gate(monkeypatch, decision: str, *, reason: str = "test"):
    calls: list[dict] = []

    def fake(conn, source_url, *, expected_author=None, force=False, **kw):
        calls.append({"url": source_url, "expected_author": expected_author, "force": force})
        return eligibility.GateDecision(decision, reason)

    monkeypatch.setattr(eligibility, "gate", fake)
    return calls


# ── structural: GitHub cannot reach this seam ─────────────────────────────────────


def test_github_is_not_a_website_adapter():
    """Not a flag — a STRUCTURAL exclusion. `sync_github` attributes to the attested repo owner,
    never the person, so there is nothing for a gate to protect; the registry must never make it
    LOOK like GitHub could be gated by making it a key that merely happens to skip the gate."""
    assert "github" not in source_adapters.WEBSITE_ADAPTERS
    assert set(source_adapters.WEBSITE_ADAPTERS) == {"substack", "blog"}


def test_routing_github_through_the_registry_raises(conn):
    with pytest.raises(KeyError):
        source_adapters.gate_and_sync_website(conn, object(), "github", "https://github.com/x/y")


# ── a refusal skips the sync, for BOTH website adapters ───────────────────────────


@pytest.mark.parametrize("source_type", ["substack", "blog"])
@pytest.mark.parametrize("decision", ["skip", "needs-review"])
def test_a_non_ingest_decision_never_reaches_the_adapter(conn, monkeypatch, source_type, decision):
    _fake_gate(monkeypatch, decision, reason="because")
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))

    got_decision, summary = source_adapters.gate_and_sync_website(
        conn, object(), source_type, "https://example.com", author_name="Carol")

    assert summary is None
    assert got_decision.decision == decision


# ── an "ingest" decision reaches the RIGHT adapter, normalized ────────────────────


def test_ingest_calls_substack_with_publication_url(conn, monkeypatch):
    _fake_gate(monkeypatch, "ingest")
    seen = {}
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda conn, embedder, **kw: seen.update(kw) or {"added": 1})
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong adapter")))

    decision, summary = source_adapters.gate_and_sync_website(
        conn, object(), "substack", "https://carol.substack.com",
        author_name="Carol", limit=5, handle="carol")

    assert decision.decision == "ingest"
    assert summary == {"added": 1}
    assert seen["publication_url"] == "https://carol.substack.com"
    assert seen["handle"] == "carol"
    assert seen["author_name"] == "Carol"
    assert seen["limit"] == 5


def test_ingest_calls_blog_with_blog_url(conn, monkeypatch):
    _fake_gate(monkeypatch, "ingest")
    seen = {}
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: seen.update(kw) or {"added": 1})
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong adapter")))

    decision, summary = source_adapters.gate_and_sync_website(
        conn, object(), "blog", "https://simonwillison.net", author_name="Simon", limit=10)

    assert decision.decision == "ingest"
    assert summary == {"added": 1}
    assert seen["blog_url"] == "https://simonwillison.net"
    assert seen["author_name"] == "Simon"
    assert seen["limit"] == 10


# ── force threads through to the gate, exactly like the pre-registry call sites ───


def test_force_reaches_the_gate_call(conn, monkeypatch):
    calls = _fake_gate(monkeypatch, "ingest")
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint", lambda conn, embedder, **kw: {})

    source_adapters.gate_and_sync_website(
        conn, object(), "blog", "https://x.example", force=True)

    assert calls[0]["force"] is True
