"""scholar_probe — the paper sample over the SCHOLAR candidate list.

Offline: the adapter's own `_get` is monkeypatched with module-level byte literals (the dominant
pattern for a transport-owning adapter), and the breaker is injected through the constructor
rather than patched.
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import probe_store, schema
from pipeline.kb import frontier_sources as fs
from pipeline.kb import scholar_probe as sp


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


class _PassThrough:
    """A breaker that never trips — the adapter's transport is what these prove, not its backoff."""
    def call(self, fn):
        return fn()

    def allow(self):
        return True


_WORKS = json.dumps({"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/a",
     "title": "Directed  Evolution of Enzymes", "publication_date": "2026-08-01",
     "abstract_inverted_index": {"We": [0], "evolve": [1], "enzymes": [2]},
     "primary_location": {"source": {"display_name": "Nature"},
                          "landing_page_url": "https://nature.com/a"},
     "type": "article", "cited_by_count": 12},
    {"id": "https://openalex.org/W2", "title": "A Second Work", "publication_date": "2026-07-01",
     "abstract_inverted_index": None, "primary_location": {}, "type": "preprint"},
    {"id": "https://openalex.org/W3", "title": "", "abstract_inverted_index": None,
     "primary_location": {}},                       # no title AND no abstract → nothing to read
]}).encode()

_AUTHOR = json.dumps({
    "id": "https://openalex.org/A5043841592", "display_name": "Frances H. Arnold",
    "works_count": 928, "cited_by_count": 120000, "summary_stats": {"h_index": 150},
    "last_known_institutions": [{"display_name": "California Institute of Technology"}],
    # Present in the real response and DELIBERATELY unread — see `describe`.
    "topics": [{"display_name": "Atmospheric chemistry and aerosols"}],
}).encode()

_GROUP_BY = json.dumps({"group_by": [{"key": "2026", "count": 14}, {"key": "2025", "count": 14},
                                     {"key": "2024", "count": 30},
                                     {"key": "bad", "count": 1}]}).encode()


def _adapter(monkeypatch, body, seen=None):
    def _fake_get(url, *, headers=None):
        if seen is not None:
            seen["url"] = url
        return body(url) if callable(body) else body
    monkeypatch.setattr(fs, "_get", _fake_get)
    return fs.OpenAlexWorksAdapter(breaker=_PassThrough())


def _scholar_candidate(conn, eid="openalex:A5043841592", *, count=3):
    schema.upsert_entity(conn, eid, name="F. Arnold")
    schema.set_signal(conn, eid, "save", "openalex", count=count)
    conn.commit()


# ── the adapter ──────────────────────────────────────────────────────────────────

def test_recent_works_sorts_by_date_not_relevance(monkeypatch):
    """The opposite of the Frontier works adapter, on purpose. That one searches TERMS, so a work
    competes on match quality; here the filter already names exactly one person, so there is no
    match quality to rank on and recency is the only ordering that means anything."""
    seen = {}
    a = _adapter(monkeypatch, _WORKS, seen)

    works = a.works("A5043841592", limit=25)

    assert "filter=author.id%3AA5043841592" in seen["url"]
    assert "sort=publication_date%3Adesc" in seen["url"]
    assert "per-page=25" in seen["url"]
    assert len(works) == 3


def test_year_counts_returns_the_whole_distribution_in_one_call(monkeypatch):
    """This is what lets the lookback question carry real numbers instead of blind presets. A
    `group_by` returns every year bucket, not a page of them."""
    seen = {}
    a = _adapter(monkeypatch, _GROUP_BY, seen)

    assert a.year_counts("A5043841592") == {2026: 14, 2025: 14, 2024: 30}
    assert "group_by=publication_year" in seen["url"]


def test_the_openalex_breaker_is_shared_with_the_frontier_works_adapter():
    """One host, one budget. Two paths against `api.openalex.org` with independent breakers would
    hammer it, and the anonymous allowance is per IP and resets only at midnight UTC — so a burned
    day stays burned. `CircuitBreaker` is keyed by this string in a persisted table."""
    assert fs.OpenAlexWorksAdapter.breaker_host == fs.OpenAlexAdapter.breaker_host


# ── rendering ────────────────────────────────────────────────────────────────────

def test_a_work_becomes_a_partial_observed_probe_atom():
    """PARTIAL and OBSERVED, never COMPLETE: the abstract is all the works response carries and no
    PDF is fetched on this path. Claiming COMPLETE would say we read the paper."""
    atoms = sp.render_works(json.loads(_WORKS)["results"],
                            who_id="openalex:A5043841592", name="F. Arnold")

    assert [a["atom_id"] for a in atoms] == ["oaprobe:W1", "oaprobe:W2"]   # W3 carried nothing
    first = atoms[0]
    assert first["source_type"] == "paper"
    assert first["who_id"] == "openalex:A5043841592"
    assert first["payload"]["body_state"] == "partial"
    assert first["payload"]["body_basis"] == "observed"
    assert first["description"] == "F. Arnold · Directed Evolution of Enzymes · Nature"
    assert "We evolve enzymes" in first["_markdown"]


# ── the description ──────────────────────────────────────────────────────────────

def test_the_description_leads_with_the_users_own_saved_titles():
    """A user reading a paper usually does not know its author, so their OWN evidence comes first
    — the titles they chose to save — then stated facts about the person."""
    out = sp.describe(json.loads(_AUTHOR), saved=["Directed Evolution of Enzymes"],
                      recent=["Squidly", "EZSolver"])

    assert out.startswith("you saved: Directed Evolution of Enzymes")
    assert "California Institute of Technology · 928 works" in out
    assert "h-index 150" in out
    assert "recent: Squidly; EZSolver" in out


def test_the_description_never_uses_openalex_topics_or_affiliations():
    """Measured 2026-09-08: Frances Arnold's top OpenAlex topic comes back "Atmospheric chemistry
    and aerosols" — she won a Nobel for the directed evolution of enzymes — and her affiliation
    history lists Pasadena City College. Those derived fields would misdescribe real people."""
    out = sp.describe(json.loads(_AUTHOR), saved=[], recent=[])

    assert "Atmospheric" not in out
    assert "aerosols" not in out


def test_a_missing_author_record_still_describes_from_saved_titles():
    """Fail-safe. The profile call is the half OPYT can lose; the user's own saved titles are the
    half that matters most, and they need no network at all."""
    assert sp.describe(None, saved=["A Paper They Saved"], recent=[]) == \
        "you saved: A Paper They Saved"


def _saved(conn, i, title, authors):
    schema.upsert_atom(conn, {
        "atom_id": f"paper:arXiv:{i}", "source_type": "paper", "what_kind": "artifact",
        # The PLACEHOLDER, which is what 67 of 82 live paper atoms actually carry: Semantic
        # Scholar never resolved the work, so there is no `scholar:` id for the first author.
        "who_id": f"paper-authors:paper:arXiv:{i}", "when_ts": f"2026-0{9 - i}-01",
        "when_precision": "day", "about_entities": [], "source_url": "", "raw_ref": "",
        "raw_hash": str(i), "entry_mode": "user-saved", "payload": {"authors": authors},
        "description": f"Some Lead · {title} · Nature · 2026"})
    conn.commit()


def test_saved_titles_finds_papers_a_person_CO_wrote_not_only_ones_they_led(conn):
    """The defect this replaces, measured on the live store: `saved_titles` filtered on
    `atoms.who_id`, which names the FIRST author and nothing else — and 67 of 82 paper atoms do
    not even have that, carrying the `paper-authors:{paper_id}` placeholder instead. It returned
    [] for all 44 real candidates, emptying the most useful half of every card.

    Authorship is what `payload.authors` says, which is the question
    `paper_authors.papers_by_author` answers."""
    from pipeline.kb import paper_authors

    _saved(conn, 1, "Newest Paper", [{"name": "Lead", "openalex_id": "A1"},
                                     {"name": "F. Arnold", "openalex_id": "A5043841592"}])
    _saved(conn, 2, "Older Paper", [{"name": "F. Arnold", "openalex_id": "A5043841592"}])
    _saved(conn, 3, "Not Theirs", [{"name": "Someone Else", "openalex_id": "A9"}])

    mine = paper_authors.papers_by_author(conn)["openalex:A5043841592"]

    assert sp.saved_titles(conn, mine) == ["Newest Paper", "Older Paper"]


def test_no_saved_papers_reads_nothing_rather_than_everything(conn):
    """An empty id list must not fall through to an unfiltered query."""
    _saved(conn, 1, "Someone Else's", [{"name": "Other", "openalex_id": "A9"}])

    assert sp.saved_titles(conn, []) == []


# ── the queue ────────────────────────────────────────────────────────────────────

def test_only_scholar_candidates_with_an_openalex_id_are_queued(conn):
    """A `scholar:`-only candidate carries a Semantic Scholar author id, which OpenAlex cannot
    take. ABSENT from this queue, not failed — the same way `candidate_probe` treats a candidate
    with no X identity. Stage 6's ORCID merge is what gives them an `openalex:` member."""
    _scholar_candidate(conn, "openalex:A5043841592")
    _scholar_candidate(conn, "scholar:2081297")
    schema.upsert_entity(conn, "x:user:1", name="Someone")
    schema.set_signal(conn, "x:user:1", "follow", "x", count=1)
    conn.commit()

    queued = [c["who_id"] for c in sp.candidate_queue(conn)]

    assert queued == ["openalex:A5043841592"]


def test_a_confirmed_oracle_is_never_probed(conn):
    """Their real footprint is already in `atoms`. Probing them would duplicate trusted content
    into the untrusted candidate store."""
    _scholar_candidate(conn)
    schema.upsert_oracle(conn, "openalex:A5043841592", name="F. Arnold")
    conn.commit()

    assert sp.candidate_queue(conn) == []


# ── the run ──────────────────────────────────────────────────────────────────────

def test_a_probed_scholar_lands_atoms_in_the_candidate_store_not_the_kb(conn, fake_embedder,
                                                                       monkeypatch):
    """The trust boundary. `retrieve.py` searches every row in `atoms` with no filter, so a
    candidate row there would be indistinguishable from Oracle knowledge."""
    _scholar_candidate(conn)
    adapter = _adapter(monkeypatch, lambda url: _AUTHOR if "/authors/" in url else _WORKS)

    out = sp.probe_scholars(conn, fake_embedder, adapter=adapter, pace_seconds=0)

    assert out["by_status"][probe_store.STATUS_OK] == 1
    assert out["atoms"] == 2
    assert probe_store.count_probe_atoms(conn, "openalex:A5043841592") == 2
    assert conn.execute("SELECT count(*) FROM atoms").fetchone()[0] == 0


def test_a_second_run_re_embeds_nothing(conn, fake_embedder, monkeypatch):
    """Idempotent by content hash: an unchanged work is skipped BEFORE the embed, so a re-pull of
    a researcher who published nothing new costs no vectors."""
    _scholar_candidate(conn)
    adapter = _adapter(monkeypatch, lambda url: _AUTHOR if "/authors/" in url else _WORKS)

    sp.probe_scholars(conn, fake_embedder, adapter=adapter, pace_seconds=0)
    again = sp.probe_scholars(conn, fake_embedder, adapter=adapter, ttl_days=0, pace_seconds=0)

    assert again["atoms"] == 0
    assert again["by_status"][probe_store.STATUS_OK] == 1
    assert probe_store.count_probe_atoms(conn, "openalex:A5043841592") == 2


def test_an_author_with_no_works_records_a_durable_empty(conn, fake_embedder, monkeypatch):
    """A FACT about the candidate, recorded so it is never re-fetched every run and never read as
    "no field"."""
    _scholar_candidate(conn)
    adapter = _adapter(monkeypatch, b'{"results": []}')

    out = sp.probe_scholars(conn, fake_embedder, adapter=adapter, pace_seconds=0)

    assert out["by_status"][probe_store.STATUS_EMPTY] == 1
    assert probe_store.pull_states(conn)["openalex:A5043841592"]["status"] == \
        probe_store.STATUS_EMPTY


def test_an_open_breaker_stops_the_run_without_marking_anyone_failed(conn, fake_embedder,
                                                                     monkeypatch):
    """Host-wide, not per-candidate. Every remaining candidate would fail identically, so marking
    them failed would record an observation nobody made."""
    _scholar_candidate(conn)

    class _Open(fs.OpenAlexWorksAdapter):
        def available(self):
            return False

    out = sp.probe_scholars(conn, fake_embedder, adapter=_Open(), pace_seconds=0)

    assert out["stopped"] == "breaker" and out["requests"] == 0
    assert probe_store.pull_states(conn) == {}


def test_the_description_is_cached_on_the_entity_for_the_card(conn, fake_embedder, monkeypatch):
    _scholar_candidate(conn)
    adapter = _adapter(monkeypatch, lambda url: _AUTHOR if "/authors/" in url else _WORKS)

    sp.probe_scholars(conn, fake_embedder, adapter=adapter, pace_seconds=0)

    from pipeline.kb import screen
    card = next(c for c in screen.build_screen(conn)["candidates"]
                if c["canonical_id"] == "openalex:A5043841592")
    assert "California Institute of Technology" in card["description"]


def test_an_untitled_paper_is_not_offered_as_a_title(conn):
    """`Untitled` is `derive_paper`'s placeholder for metadata that carried no title — 2 of 11
    saved papers on the live store. "you saved: Untitled" spends the card's most useful line
    saying we do not know."""
    from pipeline.kb import paper_authors

    _saved(conn, 1, "Untitled", [{"name": "F. Arnold", "openalex_id": "A5043841592"}])
    _saved(conn, 2, "A Real Title", [{"name": "F. Arnold", "openalex_id": "A5043841592"}])

    mine = paper_authors.papers_by_author(conn)["openalex:A5043841592"]

    assert sp.saved_titles(conn, mine) == ["A Real Title"]
