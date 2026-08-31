"""Stage-5 footprint-eligibility — the single-author gate. Fetch + LLM are monkeypatched (no
network, no key), so the classify/cache logic and the 4-way gate decision are proven offline.

The load-bearing property under test is the FLIPPED degrade direction vs screen.py: an unsure or
failed classify goes to needs-review (NOT auto-ingest), and a transient failure is NEVER cached —
one hiccup must not poison a source forever."""
from __future__ import annotations

import pytest

from pipeline.kb import eligibility, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _mock_fetch(monkeypatch, text="Hi, I'm Carol and this is my personal blog.", *, counter=None):
    """Replace the home/about fetch with a canned string (None = a fetch miss)."""
    def fake(url):
        if counter is not None:
            counter["n"] += 1
        return text
    monkeypatch.setattr(eligibility, "_fetch_home_text", fake)


def _mock_llm(monkeypatch, verdict_json, *, counter=None):
    """Wire preflight → ready and call → a canned verdict body."""
    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)

    def fake_call(role, *, system, user, **kw):
        if counter is not None:
            counter["n"] += 1
        return type("R", (), {"text": verdict_json})()

    monkeypatch.setattr(llm_client, "call", fake_call)


# ── the 4-way decision ───────────────────────────────────────────────────────────

def test_single_author_is_eligible_and_cached(conn, monkeypatch):
    _mock_fetch(monkeypatch)
    _mock_llm(monkeypatch, '{"authorship":"single","author_name":"Carol","confidence":"high"}')

    d = eligibility.gate(conn, "https://carol.example.com")
    assert d.decision == "ingest"
    # verdict cached under the person-independent bare-host key
    row = schema.get_authorship(conn, "carol.example.com")
    assert row["authorship"] == "single" and row["author_name"] == "Carol"


def test_multi_author_is_skipped_and_no_atom_stored(conn, monkeypatch):
    _mock_fetch(monkeypatch, "The Anthropic team writes about AI safety, alignment, and policy.")
    _mock_llm(monkeypatch, '{"authorship":"multi","author_name":null,"confidence":"high"}')

    d = eligibility.gate(conn, "https://anthropic.com")
    assert d.decision == "skip"
    assert schema.count_atoms(conn, "blog") == 0          # the adapter never ran
    assert schema.get_authorship(conn, "anthropic.com")["authorship"] == "multi"


def test_llm_failure_degrades_to_needs_review_and_is_not_cached(conn, monkeypatch):
    _mock_fetch(monkeypatch)
    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)

    def boom(*a, **k):
        raise RuntimeError("breaker open")

    monkeypatch.setattr(llm_client, "call", boom)

    d = eligibility.gate(conn, "https://carol.example.com")
    assert d.decision == "needs-review" and d.verdict.authorship == "unknown"
    # transient failure must NOT be cached — a re-run can still succeed
    assert schema.get_authorship(conn, "carol.example.com") is None


def test_missing_key_degrades_closed(conn, monkeypatch):
    _mock_fetch(monkeypatch)
    from pipeline import llm_client
    monkeypatch.setattr(llm_client, "preflight", lambda role: "OPENROUTER_API_KEY not set")

    d = eligibility.gate(conn, "https://carol.example.com")
    assert d.decision == "needs-review" and d.verdict.authorship == "unknown"
    assert schema.get_authorship(conn, "carol.example.com") is None


def test_fetch_failure_degrades_to_needs_review(conn, monkeypatch):
    _mock_fetch(monkeypatch, text=None)                   # no fetchable home text
    d = eligibility.gate(conn, "https://carol.example.com")
    assert d.decision == "needs-review" and d.verdict.authorship == "unknown"
    assert schema.get_authorship(conn, "carol.example.com") is None


def test_single_author_mismatch_is_needs_review(conn, monkeypatch):
    _mock_fetch(monkeypatch, "This is Dave Squatter's personal blog about woodworking.")
    _mock_llm(monkeypatch, '{"authorship":"single","author_name":"Dave Squatter"}')

    # We KNOW the Oracle is Carol; the site is single-authored by someone else → don't launder it.
    d = eligibility.gate(conn, "https://carol.example.com", expected_author="Carol Writer")
    assert d.decision == "needs-review" and "mismatch" in d.reason


def test_single_author_name_match_ingests(conn, monkeypatch):
    _mock_fetch(monkeypatch)
    _mock_llm(monkeypatch, '{"authorship":"single","author_name":"Carol Writer"}')
    # "Carol" ⊆ "Carol Writer" (token-subset) → a match, not a mismatch.
    d = eligibility.gate(conn, "https://carol.example.com", expected_author="Carol")
    assert d.decision == "ingest"


# ── cache + force ────────────────────────────────────────────────────────────────

def test_cache_hit_skips_fetch_and_classify(conn, monkeypatch):
    fetches, calls = {"n": 0}, {"n": 0}
    _mock_fetch(monkeypatch, counter=fetches)
    _mock_llm(monkeypatch, '{"authorship":"single","author_name":"Carol"}', counter=calls)

    eligibility.gate(conn, "https://carol.example.com")
    assert fetches["n"] == 1 and calls["n"] == 1

    # A DIFFERENT path on the SAME host hits the cache (host-keyed) → no re-fetch, no re-classify.
    d2 = eligibility.gate(conn, "https://carol.example.com/2026/07/a-post")
    assert d2.decision == "ingest" and d2.verdict.cached is True
    assert fetches["n"] == 1 and calls["n"] == 1


def test_force_bypasses_gate_without_classify(conn, monkeypatch):
    fetches = {"n": 0}
    _mock_fetch(monkeypatch, counter=fetches)               # must never be called under --force

    d = eligibility.gate(conn, "https://whatever.example.com", force=True)
    assert d.decision == "ingest" and d.verdict is None
    assert fetches["n"] == 0                                # pure override: no fetch, no LLM spend


# ── helper units ─────────────────────────────────────────────────────────────────

def test_name_matcher_lenient_but_flags_clear_disagreement():
    assert eligibility._name_matches("Simon Willison", "Simon Willison")
    assert eligibility._name_matches("Simon", "Simon Willison")            # token subset
    assert eligibility._name_matches("Willison", "Simon Willison")         # shared long token
    assert eligibility._name_matches("Carol", None)                        # missing → no contradiction
    assert not eligibility._name_matches("Carol Writer", "Dave Squatter")  # clear disagreement


def test_parser_tolerates_fenced_json_and_rejects_bad_kind():
    v = eligibility._parse_verdict('```json\n{"authorship":"single","author_name":"X"}\n```')
    assert v["authorship"] == "single" and v["author_name"] == "X"
    assert eligibility._parse_verdict("no json here") is None
    assert eligibility._parse_verdict('{"authorship":"weird"}') is None    # invalid kind → unknown
