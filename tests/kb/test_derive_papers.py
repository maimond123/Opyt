"""derive_paper + the canonical paper-id rule. Pure, offline (no network)."""
from __future__ import annotations

from pipeline.kb import derive
from pipeline.kb import ingest_papers as ip

_PAPER = {
    "paperId": "arXiv:2401.00001",
    "title": "Scaling\nAutonomous Agents",
    "abstract": "We study how autonomous agents compose tools into larger systems.",
    "authors": [
        {"authorId": "111", "name": "Alice Researcher"},
        {"authorId": "222", "name": "Bob Coauthor"},
        {"authorId": None, "name": "Nameless Contributor"},
    ],
    "year": 2024,
    "publicationDate": "2024-01-15",
    "venue": "NeurIPS",
    "citationCount": 42,
    "url": "https://www.semanticscholar.org/paper/abc",
    "externalIds": {"ArXiv": "2401.00001", "DOI": "10.1234/abcd"},
}


def test_derive_paper_fields():
    m = derive.derive_paper(_PAPER)
    assert m["who_id"] == "scholar:111"            # the FIRST author — NOT the Oracle
    assert m["who_name"] == "Alice Researcher"
    assert m["when_ts"] == "2024-01-15" and m["when_precision"] == "day"
    assert m["about_entities"] == []
    # Mechanical description: newline flattened, structural fields only.
    assert "\n" not in m["description"]
    assert "Alice Researcher" in m["description"]
    assert "Scaling Autonomous Agents" in m["description"]
    assert "NeurIPS" in m["description"] and "2024" in m["description"]


def test_derive_paper_year_precision_is_honest():
    # No publicationDate → year precision, not a fake Jan-1 day.
    p = {k: v for k, v in _PAPER.items() if k != "publicationDate"}
    m = derive.derive_paper(p)
    assert m["when_ts"] == "2024-01-01" and m["when_precision"] == "year"


def test_derive_paper_undated_is_empty():
    p = {"paperId": "url:x", "title": "T", "authors": []}
    m = derive.derive_paper(p)
    assert m["when_ts"] == "" and m["when_precision"] == ""


def test_derive_paper_no_author_falls_back_to_paper_authors():
    # A raw hosted PDF with no metadata → a stable per-paper placeholder id, never a crash.
    p = {"paperId": "url:example.com/x.pdf", "title": "Anon", "authors": [], "year": 2023}
    m = derive.derive_paper(p)
    assert m["who_id"] == "paper-authors:url:example.com/x.pdf"
    assert m["who_name"] == "paper"


def test_canonical_paper_id_rule():
    # paperId wins when present.
    assert ip.paper_atom_id({"paperId": "abc123"}) == "paper:abc123"
    # else derive from external ids: arXiv (version-stripped) > DOI (lowercased).
    assert ip.paper_atom_id({"externalIds": {"ArXiv": "2401.00001v3"}}) == "paper:arXiv:2401.00001"
    assert ip.paper_atom_id({"externalIds": {"DOI": "10.5/AbCd"}}) == "paper:DOI:10.5/abcd"
    # unidentifiable → None (caller returns None, never mints paper:None).
    assert ip.paper_atom_id({}) is None
