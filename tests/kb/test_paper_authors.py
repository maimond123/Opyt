"""curation.papers — the authors of user-saved papers, as screenable people.

Fully offline by nature, not by monkeypatch: this module reads atom rows and writes entities and
signals. There is no network, no LLM and no embed to stub.
"""
from __future__ import annotations

import pytest

from pipeline.curation import papers as pa
from pipeline.kb import schema
from pipeline.oracles import roster


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _paper(conn, atom_id, authors, *, entry_mode="user-saved"):
    schema.upsert_atom(conn, {
        "atom_id": atom_id, "source_type": "paper", "what_kind": "artifact",
        "who_id": "paper-authors:" + atom_id, "when_ts": "2026-01-01", "when_precision": "day",
        "about_entities": [], "source_url": "", "raw_hash": atom_id,
        "description": "", "entry_mode": entry_mode, "payload": {"authors": authors}})
    conn.commit()


def _sig(conn, entity_id, platform):
    return conn.execute(
        "SELECT count FROM curation_signals WHERE entity_id=? AND signal_type='save' "
        "AND platform=?", (entity_id, platform)).fetchone()


# ── the tally ────────────────────────────────────────────────────────────────────

def test_an_author_across_two_saved_papers_outcounts_a_one_off_coauthor(conn):
    """The count is the whole ranking signal. OpenAlex has no follow primitive, so authorship of
    a paper the user chose to save is the only evidence there is, and how often is what separates
    a person the user follows the work of from someone who happened to share a byline once."""
    _paper(conn, "paper:arXiv:1", [{"name": "Repeat Author", "scholar_id": "111"},
                                   {"name": "One Off", "scholar_id": "222"}])
    _paper(conn, "paper:arXiv:2", [{"name": "Repeat Author", "scholar_id": "111"}])

    out = pa.sync_paper_author_signals(conn)

    assert _sig(conn, "scholar:111", "scholar")["count"] == 2
    assert _sig(conn, "scholar:222", "scholar")["count"] == 1
    assert out["papers"] == 2 and out["authors"] == 2 and out["multi_paper_authors"] == 1


def test_a_second_run_leaves_the_counts_unchanged(conn):
    """The regression this feature is most likely to reintroduce: summing a re-read total into a
    total inflates the count on every run. This producer re-walks the whole saved set every run, so it must `set_signal`."""
    _paper(conn, "paper:arXiv:1", [{"name": "A", "scholar_id": "111"}])
    _paper(conn, "paper:arXiv:2", [{"name": "A", "scholar_id": "111"}])

    pa.sync_paper_author_signals(conn)
    pa.sync_paper_author_signals(conn)
    pa.sync_paper_author_signals(conn)

    assert _sig(conn, "scholar:111", "scholar")["count"] == 2


def test_only_the_papers_the_user_chose_produce_signals(conn):
    """An admitted frontier artifact is something OPYT found, not something the user chose, and
    only a choice is a vouch. `entry_mode` is allow-listed, never deny-listed, so a mode added
    later cannot leak in by default."""
    _paper(conn, "paper:arXiv:1", [{"name": "Chosen", "scholar_id": "111"}])
    _paper(conn, "paper:arXiv:2", [{"name": "Crawled", "scholar_id": "222"}],
           entry_mode="frontier")
    _paper(conn, "paper:arXiv:3", [{"name": "Referenced", "scholar_id": "333"}],
           entry_mode="author_referenced")

    out = pa.sync_paper_author_signals(conn)

    assert _sig(conn, "scholar:111", "scholar")["count"] == 1
    assert _sig(conn, "scholar:222", "scholar") is None
    assert _sig(conn, "scholar:333", "scholar") is None
    assert out["papers"] == 1


# ── which registry issues the entity id ──────────────────────────────────────────

def test_a_scholar_id_wins_over_an_openalex_one_for_the_same_author(conn):
    """Not a quality judgement. `derive_paper` mints `who_id = scholar:{first_author_id}` and
    `atomize_paper` upserts THAT id as an entity, so minting `openalex:…` for the same person
    would stand a second entity beside it carrying half the signal."""
    _paper(conn, "paper:arXiv:1", [{"name": "Both", "scholar_id": "111",
                                    "openalex_id": "A5043841592"}])

    pa.sync_paper_author_signals(conn)

    assert _sig(conn, "scholar:111", "scholar")["count"] == 1
    assert _sig(conn, "openalex:A5043841592", "openalex") is None


def test_an_openalex_only_author_gets_an_openalex_entity(conn):
    _paper(conn, "paper:arXiv:1", [{"name": "OA Only", "openalex_id": "A5043841592"}])

    pa.sync_paper_author_signals(conn)

    assert _sig(conn, "openalex:A5043841592", "openalex")["count"] == 1


def test_an_author_with_no_registry_id_is_skipped_and_counted(conn):
    """A name is not an identity — "Frances Arnold" matches 16 distinct OpenAlex people. And a
    candidate minted from a name could not be acted on: confirming a scholar
    Oracle pulls their back catalogue by author id, so there would be nothing to pull.

    Reported, never silent. The drop is a decision, and a decision the user cannot see is
    indistinguishable from a bug."""
    _paper(conn, "paper:arXiv:1", [{"name": "Has One", "scholar_id": "111"},
                                   {"name": "Nameless Contributor"},
                                   {"name": "Also Id-less"}])

    out = pa.sync_paper_author_signals(conn)

    assert out["authors"] == 1
    assert out["authors_without_a_registry_id"] == 2


# ── the merge key ────────────────────────────────────────────────────────────────

def test_the_orcid_is_the_only_identity_link_written(conn):
    """`resolve` unions on `identity_links`, so what goes in this column decides who merges with
    whom. An ORCID is safe because nobody can claim another person's; an institution or lab URL
    is not, because two researchers in one department share it."""
    _paper(conn, "paper:arXiv:1", [{"name": "F. Arnold", "openalex_id": "A5043841592",
                                    "orcid": "0000-0002-4027-364X"}])

    pa.sync_paper_author_signals(conn)

    links = conn.execute("SELECT identity_links FROM entities WHERE entity_id=?",
                         ("openalex:A5043841592",)).fetchone()[0]
    assert links == '["https://orcid.org/0000-0002-4027-364X"]'


def test_one_papers_missing_orcid_does_not_erase_anothers(conn):
    """Metadata coverage is uneven — 73% of 256 measured authorships carry an ORCID — so the same
    person arrives with one on some papers and without on others. The merge key is worth more
    than which paper happened to supply it."""
    _paper(conn, "paper:arXiv:1", [{"name": "F. Arnold", "openalex_id": "A5043841592"}])
    _paper(conn, "paper:arXiv:2", [{"name": "F. Arnold", "openalex_id": "A5043841592",
                                    "orcid": "0000-0002-4027-364X"}])
    _paper(conn, "paper:arXiv:3", [{"name": "F. Arnold", "openalex_id": "A5043841592"}])

    pa.sync_paper_author_signals(conn)

    links = conn.execute("SELECT identity_links FROM entities WHERE entity_id=?",
                         ("openalex:A5043841592",)).fetchone()[0]
    assert links == '["https://orcid.org/0000-0002-4027-364X"]'


def test_an_empty_store_reports_zero_rather_than_failing(conn):
    assert pa.sync_paper_author_signals(conn) == {
        "source": "paper-authors", "papers": 0, "authors": 0, "multi_paper_authors": 0,
        "authors_without_a_registry_id": 0}


# ── coauthors: the one producer whose input grows with the roster ────────────────

def _oracle_paper(conn, atom_id, authors):
    _paper(conn, atom_id, authors, entry_mode="oracle-footprint")


def test_a_repeat_collaborator_is_signalled_and_a_one_off_is_not(conn):
    """`min_papers=2` is the bound on the one producer that scales with the Oracle roster rather
    than with what the user did. It is a different decision from refusing to cap saved-paper
    authors: there every author is backed by a user action, here none is — the evidence is
    entirely inferred from a byline. One shared byline is noise; two is a working relationship."""
    _oracle_paper(conn, "paper:arXiv:1", [{"name": "Regular", "openalex_id": "A1"},
                                          {"name": "One Off", "openalex_id": "A2"}])
    _oracle_paper(conn, "paper:arXiv:2", [{"name": "Regular", "openalex_id": "A1"}])

    out = pa.sync_coauthor_signals(conn)

    assert conn.execute("SELECT count FROM curation_signals WHERE entity_id='openalex:A1' "
                        "AND signal_type='coauthor'").fetchone()[0] == 2
    assert not conn.execute("SELECT 1 FROM curation_signals WHERE entity_id='openalex:A2'"
                            ).fetchone()
    assert out["signalled"] == 1 and out["below_min_papers"] == 1


def test_a_confirmed_oracle_is_not_their_own_candidate(conn):
    _oracle_paper(conn, "paper:arXiv:1", [{"name": "The Oracle", "openalex_id": "A1"}])
    _oracle_paper(conn, "paper:arXiv:2", [{"name": "The Oracle", "openalex_id": "A1"}])
    roster.upsert_oracle(conn, "openalex:A1", name="The Oracle")
    conn.commit()

    out = pa.sync_coauthor_signals(conn)

    assert out["signalled"] == 0
    assert not conn.execute("SELECT 1 FROM curation_signals").fetchone()


def test_a_coauthor_signal_never_touches_the_saved_paper_tally(conn):
    """Two modes, two signal types, one walk. A person who is both an author the user saved AND
    an Oracle's collaborator carries TWO distinct signals — which is what corroboration means —
    not one signal counted twice."""
    _paper(conn, "paper:arXiv:1", [{"name": "Both", "openalex_id": "A1"}])
    _oracle_paper(conn, "paper:arXiv:2", [{"name": "Both", "openalex_id": "A1"}])
    _oracle_paper(conn, "paper:arXiv:3", [{"name": "Both", "openalex_id": "A1"}])

    pa.sync_paper_author_signals(conn)
    pa.sync_coauthor_signals(conn)

    rows = dict(conn.execute("SELECT signal_type, count FROM curation_signals "
                             "WHERE entity_id='openalex:A1'"))
    assert rows == {"save": 1, "coauthor": 2}


def test_a_second_coauthor_run_leaves_the_counts_unchanged(conn):
    _oracle_paper(conn, "paper:arXiv:1", [{"name": "Regular", "openalex_id": "A1"}])
    _oracle_paper(conn, "paper:arXiv:2", [{"name": "Regular", "openalex_id": "A1"}])

    pa.sync_coauthor_signals(conn)
    pa.sync_coauthor_signals(conn)

    assert conn.execute("SELECT count FROM curation_signals WHERE entity_id='openalex:A1'"
                        ).fetchone()[0] == 2


def test_a_coauthor_only_candidate_lands_in_the_papers_group_unselected(conn):
    """Signals only, never auto-Oracles — the same discipline that disabled the second-degree
    follow scout. The user promotes them or nobody does."""
    from pipeline.oracles import screen

    for i in (1, 2):
        _oracle_paper(conn, f"paper:arXiv:{i}", [{"name": "Collaborator", "openalex_id": "A1"}])
    pa.sync_coauthor_signals(conn)

    papers = next(g for g in screen.build_screen(conn)["groups"] if g.get("source") == "papers")
    card = next(c for c in papers["candidates"] if c["canonical_id"] == "openalex:A1")

    assert card["reason"] == "co-wrote 2 papers with one of your Oracles"
    assert roster.on_roster(conn, "openalex:A1") is False


def test_the_report_names_the_backfill_when_atoms_predate_the_author_list(conn):
    """The detector for a hand-run backfill: a maintenance pass whose trigger nothing can detect is a
    pass nobody runs. Papers are immutable under Policy B, so an atom
    written before the author list landed is never revisited and its authors reach nothing."""
    _paper(conn, "paper:arXiv:1", [{"name": "Has One", "openalex_id": "A1"}])
    _paper(conn, "paper:arXiv:2", [])                        # predates the author list

    out = pa.sync_paper_author_signals(conn)

    assert out["needs_backfill"] == 1
    assert "backfill_paper_authors" in out["remedy"]


def test_a_fully_backfilled_store_says_nothing_about_it(conn):
    """A remedy that appears when nothing is wrong is noise, and noise is how a real one gets
    ignored."""
    _paper(conn, "paper:arXiv:1", [{"name": "Has One", "openalex_id": "A1"}])

    assert "needs_backfill" not in pa.sync_paper_author_signals(conn)


def test_a_coauthor_signal_names_the_oracle_they_wrote_with(conn):
    for n in (1, 2):
        _oracle_paper(conn, f"paper:arXiv:{n}", [{"name": "The Oracle", "openalex_id": "A1"},
                                                 {"name": "Colleague", "openalex_id": "A2"}])
    _oracle_paper(conn, "paper:arXiv:3", [{"name": "Other", "openalex_id": "A3"},
                                          {"name": "Colleague", "openalex_id": "A2"}])
    roster.upsert_oracle(conn, "openalex:A1", name="The Oracle")
    conn.commit()

    pa.sync_coauthor_signals(conn)

    extra = conn.execute("SELECT extra FROM curation_signals WHERE entity_id='openalex:A2'"
                         ).fetchone()[0]
    assert '"with": ["openalex:A1"]' in extra
    assert pa.coauthors_of(conn, "openalex:A1") == [
        {"name": "Colleague", "canonical_id": "openalex:A2", "papers": 3}]
    assert pa.coauthors_of(conn, "openalex:A9") == []
