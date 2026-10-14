"""`verdict.judge` — the decision table, one
case per row, drawn from a dry run where it can be. Plain values: no network, no
store."""

import pytest

from pipeline.oracles.discovery.backlink import Facts
from pipeline.oracles.discovery.verdict import Candidate, judge

ROOT = "x.com/aleximas"
IDS = frozenset({ROOT, "aleximas.com"})          # the root and what its bio declares
KNOWN = frozenset({ROOT})


def _one(c, facts=None, *, known=KNOWN, ids=IDS):
    (v,) = judge([c], {c.key: facts} if facts else {}, oracle_ids=ids, known=known)
    return v


def test_linked_from_their_posts_and_linking_back_is_sure():
    """Alex Imas's Substack: in his posts, and its author profile names his X."""
    c = Candidate("aleximas.substack.com", "https://aleximas.substack.com", post_count=41)
    v = _one(c, Facts(c.key, True, frozenset({ROOT}), None, True))
    assert v.verdict == "sure" and v.source_type == "substack"
    assert v.reasons == ("in 41 of their posts", "links back to their X")


def test_a_site_that_links_someone_else_drops():
    """macromostly links @econberger, not Imas: it is someone he cites."""
    c = Candidate("macromostly.substack.com", "https://macromostly.substack.com", post_count=3)
    assert _one(c, Facts(c.key, True, frozenset({"x.com/econberger"}), None, True)).verdict == "drop"


def test_a_redirect_to_a_known_source_merges_silently():
    """herbiebradley.substack.com now serves pathwaysai.org, his root."""
    c = Candidate("herbiebradley.substack.com", "https://herbiebradley.substack.com", post_count=3)
    v = _one(c, Facts(c.key, True, frozenset({"x.com/herbiebradley"}), "pathwaysai.org", True),
             known=frozenset({"x.com/herbiebradley", "pathwaysai.org"}),
             ids=frozenset({"x.com/herbiebradley", "pathwaysai.org"}))
    assert v.verdict == "merge" and v.reasons == ()


def test_theirs_but_nothing_to_pull_is_also():
    """A GitHub account with no public repositories: theirs, and nothing to pull."""
    c = Candidate("github.com/someone", "https://github.com/someone", post_count=2)
    v = _one(c, Facts(c.key, True, frozenset({ROOT}), None, False))
    assert v.verdict == "also" and v.reasons == ("github.com",)


def test_a_site_whose_publishing_is_unknown_is_sure_when_it_links_back():
    """executor.sh: Rhys's product, linked in 24 posts, linking back. A home page shows no feed
    tag either way (chrisbarber.co, a personal site, carries none), so it is added; forget removes
    it if it turns out to be a product."""
    c = Candidate("executor.sh", "https://executor.sh", post_count=24)
    v = _one(c, Facts(c.key, True, frozenset({"x.com/rhyssullivan"}), None, None),
             ids=frozenset({"x.com/rhyssullivan"}), known=frozenset({"x.com/rhyssullivan"}))
    assert v.verdict == "sure"


def test_a_handle_inside_a_post_never_counts():
    c = Candidate("x.com/someoneelse", "https://x.com/someoneelse", post_count=9)
    assert _one(c).verdict == "drop"


def test_an_x_a_profile_of_theirs_declares_is_sure():
    c = Candidate("x.com/gergelyorosz", "https://x.com/GergelyOrosz",
                  declared="on their Substack profile")
    v = _one(c)
    assert v.verdict == "sure" and v.source_type == "x"
    assert v.reasons == ("on their Substack profile",)


def test_a_research_profile_a_profile_of_theirs_declares_is_also():
    c = Candidate("scholar.google.com/citations?user=P8EMNvkAAAAJ",
                  "https://scholar.google.com/citations?user=P8EMNvkAAAAJ",
                  declared="on their site")
    v = _one(c)
    assert v.verdict == "also" and v.reasons == ("scholar.google.com",)


def test_a_research_profile_only_in_posts_drops():
    c = Candidate("arxiv.org/a/roberts_d_3.html", "https://arxiv.org/a/roberts_d_3.html",
                  post_count=2)
    assert _one(c).verdict == "drop"


def test_a_declared_orcid_is_sure():
    c = Candidate("orcid.org/0000-0002-4027-364X", "https://orcid.org/0000-0002-4027-364X",
                  declared="on their site")
    v = _one(c)
    assert v.verdict == "sure" and v.reasons == ("their ORCID", "on their site")


def test_a_declared_site_is_sure_even_unread():
    """aleximas.com is in his bio; its page could not be read (JavaScript-built)."""
    c = Candidate("aleximas.com", "https://aleximas.com", post_count=4,
                  declared="in their X bio")
    v = _one(c)
    assert v.verdict == "sure" and v.reasons == ("in their X bio", "in 4 of their posts")


def test_a_declared_org_is_also():
    c = Candidate("github.com/bowang-lab", "https://github.com/orgs/bowang-lab",
                  declared="in their X bio")
    assert _one(c, Facts(c.key, True, frozenset(), None, True)).verdict == "also"


@pytest.mark.parametrize("posts", [1, 6])
def test_a_site_that_could_not_be_read_is_dropped_however_often_it_is_linked(posts):
    """Ruled 2026-10-14: OPYT could not pull it, so it is never offered. wsj.com was asked about
    in all five onboarding runs; cell.com six times from one person."""
    c = Candidate("cell.com", "https://www.cell.com/cell/fulltext/x", post_count=posts)
    assert _one(c, Facts(c.key)).verdict == "drop"


def test_one_hop_needs_its_own_link_back():
    """herbiebradley.com is listed on the Pathways author profile. Listed and linking back is
    sure; listed alone is not."""
    c = Candidate("herbiebradley.com", "https://herbiebradley.com", via="pathwaysai.org")
    sure = _one(c, Facts(c.key, True, frozenset({ROOT}), None, None))
    assert sure.verdict == "sure" and sure.reasons == ("listed on pathwaysai.org",
                                                       "links back to their X")
    assert _one(c, Facts(c.key, True, frozenset(), None, None)).verdict == "drop"


def test_known_sources_get_no_verdict_and_the_order_is_sure_first():
    cands = [Candidate("b.example", "https://b.example", post_count=1),
             Candidate("a.example", "https://a.example", post_count=5),
             Candidate(ROOT, "https://x.com/aleximas", declared="on their Substack profile"),
             Candidate("c.example", "https://c.example", post_count=9)]
    facts = {"a.example": Facts("a.example", True, frozenset({ROOT}), None, True),
             "b.example": Facts("b.example", True, frozenset({ROOT}), None, True)}
    got = judge(cands, facts, oracle_ids=IDS, known=KNOWN)
    assert [(v.key, v.verdict) for v in got] == [
        ("a.example", "sure"), ("b.example", "sure"), ("c.example", "drop")]


def test_a_site_declared_on_another_profile_of_theirs_carries_that_profiles_reason():
    """What their GitHub profile declares is theirs as what the root declares
    is, and says where it was declared."""
    c = Candidate("soren-tools.example", "https://soren-tools.example",
                  declared="on their GitHub profile")
    v = _one(c, Facts(c.key, True, frozenset(), None, True))
    assert v.verdict == "sure" and v.reasons == ("on their GitHub profile",)
