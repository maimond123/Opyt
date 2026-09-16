"""Hopper — the one deposit surface: any URL → the right adapter → one `user-saved` atom.

Offline. Every network seam (`_fetch_article`, `_fetch_one_tweet`, `classify_link_deep`, the
paper/github/substack adapters) is monkeypatched, so these prove the ROUTING and the INVARIANTS,
not the scrapes:

  • the wider `classify_reference` vocabulary, and — the load-bearing one — that widening it did
    NOT touch `classify_link`, whose None is a MEASURED decision inside the X footprint filter;
  • preview makes no fetch for non-X sources, writes nothing, and answers already-present;
  • every dumped atom is stamped `entry_mode='user-saved'`;
  • a dump creates an ENTITY but never an ORACLE, and never an ORACLE row for a bare host;
  • a repeat dump is a no-op, and every failure path writes nothing at all.
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import hopper, ingest_blog, ingest_x, link_router, schema
from pipeline.kb.embed import ensure_kb_meta


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _no_deep_probe(no_deep_probe):
    """Autouse in THIS file only: nearly every test here hoppers `_ARTICLE_URL`, and the article
    fallback pays a bounded page read to check its own guess. The stub itself lives in conftest
    (`no_deep_probe`) because a second file needs it; the tests that are ABOUT the probe re-patch
    the same symbol to say otherwise."""


_ARTICLE_URL = "https://www.theverge.com/2026/8/1/some-cool-article"
_ARTICLE_ATOM = "blog:theverge.com/2026/8/1/some-cool-article"
_BODY = (
    "The interesting engineering question in agent design is how you decompose a task into steps "
    "the agent can actually execute reliably, and what you do when one of them fails halfway. "
    "This piece walks through retries, timeouts and idempotency in a real deployment."
)


def _article(url=_ARTICLE_URL, title="Some Cool Article", date="2026-08-01", author="Jane Doe"):
    return {"url": url, "title": title, "date": date, "author": author, "content": _BODY}


def _atom_row(conn, atom_id):
    return conn.execute("SELECT source_type, who_id, entry_mode, description FROM atoms "
                        "WHERE atom_id=?", (atom_id,)).fetchone()


# ── routing: the wider vocabulary, and the sniffer it must not disturb ────────────────

@pytest.mark.parametrize("url,kind", [
    ("https://arxiv.org/abs/2401.00001", "paper"),
    ("https://doi.org/10.1000/xyz", "paper"),
    ("https://acme.com/whitepaper.pdf", "paper"),
    ("https://github.com/ggerganov/llama.cpp", "github"),
    ("https://simonw.substack.com/p/a-post", "substack"),
    ("https://x.com/karpathy/status/1234567890", "x"),
    ("https://twitter.com/karpathy/status/1234567890", "x"),
    ("https://mobile.x.com/karpathy/status/1234567890/photo/1", "x"),
])
def test_a_known_shape_is_sniffed_not_guessed(url, kind):
    """A url host is a FACT, so it outranks anything a host model asserts. `basis='sniffed'` is
    what says the answer was checked rather than believed."""
    assert link_router.classify_reference(url) == (kind, "sniffed")
    assert link_router.classify_reference(url, hint="article") == (kind, "sniffed"), \
        "a recognized host must beat the hint — the hint is a read, the host is a fact"


# ── the DOI in the path: what makes the paper test generalize past a host list ────────
#
# Measured 2026-09-09: ACS, JACS, Wiley, ACM, Taylor & Francis, SAGE, Springer and APS all print
# the DOI in their own url, and NONE of their hosts was in `_PAPER_HOSTS` — so a JACS paper routed
# to `article` and landed as `blog:pubs.acs.org/doi/10.1021/…`. The publisher list that would fix
# that is unbounded; the DOI's shape is not.


@pytest.mark.parametrize("url,doi", [
    ("https://pubs.acs.org/doi/10.1021/jacs.4c01234", "10.1021/jacs.4c01234"),
    ("https://onlinelibrary.wiley.com/doi/10.1002/anie.202401234", "10.1002/anie.202401234"),
    ("https://dl.acm.org/doi/10.1145/3442188.3445922", "10.1145/3442188.3445922"),
    # Springer and APS do NOT use a `/doi/` segment, which is why the test is the DOI and not
    # that segment — anchoring on `/doi/` would have missed both.
    ("https://link.springer.com/article/10.1007/s00214-024-03100-5", "10.1007/s00214-024-03100-5"),
    # Lowercased on the way in: DOIs are case-insensitive, so `PhysRevLett` and `physrevlett`
    # are one paper and must not become two immutable atoms.
    ("https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.123456",
     "10.1103/physrevlett.130.123456"),
])
def test_a_doi_in_the_path_is_a_paper_on_any_host(url, doi):
    """Routed AND parsed. Both halves are required and they live in different modules: the router
    decides the url is a paper, and only then does `_parse_paper_url` ever see it."""
    from pipeline.kb import ingest_papers

    assert link_router.classify_reference(url) == ("paper", "sniffed")
    assert ingest_papers.paper_from_url(url, enrich=False)["paperId"] == f"DOI:{doi}"


def test_every_url_form_of_one_paper_still_deduplicates():
    """The contract the generalization could most easily have broken. `_parse_paper_url` promises
    every link form of one paper mints ONE atom, and papers are immutable — a split identity can
    never be merged back."""
    from pipeline.kb import ingest_papers
    ids = {ingest_papers.paper_from_url(u, enrich=False)["paperId"] for u in (
        "https://pubs.acs.org/doi/10.1021/jacs.4c01234",
        "https://pubs.acs.org/doi/abs/10.1021/jacs.4c01234",
        "https://doi.org/10.1021/jacs.4c01234",
        "https://dx.doi.org/10.1021/jacs.4c01234",
        "https://pubs.acs.org/doi/10.1021/jacs.4c01234?ref=article",
    )}
    assert ids == {"DOI:10.1021/jacs.4c01234"}


def test_an_arxiv_doi_still_collapses_onto_the_preprint():
    """The generalized pattern now matches the arXiv DOI on any host too, so the collapse that
    stops one preprint minting two atoms has to survive it."""
    from pipeline.kb import ingest_papers
    for u in ("https://doi.org/10.48550/arXiv.2302.13971",
              "https://arxiv.org/abs/2302.13971"):
        assert ingest_papers.paper_from_url(u, enrich=False)["paperId"] == "arXiv:2302.13971"


def test_an_ordinary_url_is_not_mistaken_for_a_doi():
    """The false-positive direction. `10.` plus four-to-nine digits plus a slash is the guard; a
    version number, a date path or a plain slug must not trip it."""
    for u in (_ARTICLE_URL, "https://example.com/v10.2/release-notes",
              "https://example.com/2024/10.15/some-post", "https://example.com/pricing"):
        assert link_router.classify_link(u) is None, u


# ── tier 3: a DOI the page PRINTS but does not declare ────────────────────────────────
#
# The probe already reads 64KB of the head looking for `citation_doi`; scanning the same bytes for
# a printed DOI costs nothing extra and covers every publisher that renders a page without
# citation tags. It stays on the right side of this module's one line: read an identifier the
# source DECLARES, never infer one. Both inference routes measured badly on 2026-09-09 — Crossref
# title search resolved 1 of 3, returning a Faculty Opinions review of AlphaFold and a different
# paper called "Is Attention All You Need?", and a wrong paper atom is immutable.


@pytest.mark.parametrize("markup,doi", [
    (b'<p>DOI: 10.1039/d3sc01260c</p>', "10.1039/d3sc01260c"),
    (b'<a href="https://doi.org/10.1145/3442188.3445922">link</a>', "10.1145/3442188.3445922"),
    (b'cite as doi.org/10.1021/jacs.4c01234.', "10.1021/jacs.4c01234"),
])
def test_a_printed_doi_is_read_from_the_page(markup, doi):
    from pipeline.kb.link_router import _PRINTED_DOI_RE
    m = _PRINTED_DOI_RE.search(markup)
    assert m and m.group(1).decode().rstrip(".,;)") == doi


@pytest.mark.parametrize("markup", [
    b'var buildId = 10.1234/abcdef;',          # a bare number pair in a script payload
    b'{"analytics":"10.5555/session"}',        # json that happens to look DOI-shaped
    b'<p>no identifier here at all</p>',
])
def test_a_bare_number_is_not_read_as_a_doi(markup):
    """This runs on pages that reached the ARTICLE fallback — mostly NOT papers — so the required
    `doi.org/` or `doi:` prefix is what stops 64KB of markup producing a false paper."""
    from pipeline.kb.link_router import _PRINTED_DOI_RE
    assert _PRINTED_DOI_RE.search(markup) is None


def test_the_declared_tag_outranks_the_printed_one():
    """A `citation_doi` is the publisher naming THIS page's paper; a printed DOI may be a citation
    of somebody else's. Order is the whole guard, so it is asserted rather than assumed."""
    from pipeline.kb import link_router as lr
    body = (b'<meta name="citation_doi" content="10.1038/mine"/>'
            b'<p>as shown in DOI: 10.1234/somebody-elses</p>')
    assert lr._CITATION_DOI_RE.search(body).group(1) == b"10.1038/mine"
    assert lr._PRINTED_DOI_RE.search(body).group(1) == b"10.1234/somebody-elses"


def test_a_bare_link_falls_back_to_article():
    """The headline use case: 'I found a cool article'. Nothing matches, so it routes to the
    catch-all rather than refusing — the fetch + content gate are the quality guard, not the URL."""
    assert link_router.classify_reference(_ARTICLE_URL) == ("article", "fallback")


def test_the_hint_is_consulted_between_the_sniff_and_the_fallback():
    """Ordered AFTER the sniff and BEFORE the fallback: an article catch-all that fired first would
    make the hint dead code. An unrecognized hint is ignored rather than honored (fail-safe)."""
    lab = "https://lab.example.edu/pubs/attention.html"
    assert link_router.classify_reference(lab, hint="paper") == ("paper", "hint")
    assert link_router.classify_reference(lab, hint="nonsense") == ("article", "fallback")


def test_the_hint_earns_its_keep_on_exactly_one_kind():
    """Measured 2026-08-13, and the obvious guesses are WRONG — which is why this is pinned.

    Three adapters are host-anchored on the same hosts the sniffer checks, so a hint routes to an
    adapter that refuses the URL: hint='paper' on a lab's html page parses to no paper id at all.
    The ONE case where a hint changes the outcome is a Substack on a CUSTOM DOMAIN — invisible to
    any classifier without a fetch (`ingest_substack.py:63` says so), but accepted by the adapter.

    It is not cosmetic. Routed as an article the post keys `blog:{host}/p/{slug}`; routed as a
    Substack it keys `substack:{post_id}` — the SAME id the bookmark and footprint paths write. Get
    it wrong and the same post lives as two atoms that can never dedupe.

    If this test ever fails because a hinted 'paper'/'github'/'x' started resolving, that is GOOD
    news — an adapter widened. Update the tool docstring, which currently tells the host model not
    to bother."""
    from pipeline.kb.ingest_substack import _SUBSTACK_POST_RE

    custom = "https://newsletter.pragmaticengineer.com/p/the-scoop"
    assert link_router.classify_reference(custom) == ("article", "fallback"), \
        "the sniffer cannot see a custom-domain Substack — that is the gap"
    assert link_router.classify_reference(custom, hint="substack") == ("substack", "hint")
    assert _SUBSTACK_POST_RE.match(custom), "and the adapter DOES accept it — so the hint is live"

    # The other three: hinting them buys nothing, because each adapter checks the host itself.
    from pipeline.kb import ingest_github, ingest_papers
    lab = "https://lab.example.edu/pubs/attention.html"
    assert ingest_papers.paper_from_url(lab, enrich=False) is None
    assert ingest_github._github_owner_repo(lab) is None
    assert link_router.parse_tweet_id(lab) is None


@pytest.mark.parametrize("ref", ["", "   ", "how do agents retry", "/Users/me/paper.pdf",
                                 "arxiv.org/abs/2401.00001"])
def test_a_non_url_is_unroutable_never_guessed(ref):
    """The ONLY genuine unroutable: something with no host to fetch. A scheme-less string is
    included on purpose — it is a plausible paste, and inferring `https://` for it would be the
    guess this surface exists to refuse."""
    assert link_router.classify_reference(ref) == (None, "none")


def test_widening_the_vocabulary_did_not_widen_the_footprint_sniffer():
    """THE regression guard for the split. `classify_link`'s None is read as a DECISION by the X
    footprint substance filter: a post whose only link is bare carries no retrievable substance and
    faces the 200-char naked bar (measured, David 2026-07-20). If someone ever 'simplifies' the two
    classifiers into one, an x.com or news link starts answering non-None, `_dispatchable_link`
    flips to True, and that measurement silently reverses with no test failing anywhere else."""
    from pipeline.kb import ingest_x_footprint as fp

    for bare in ("https://x.com/karpathy/status/1234567890", _ARTICLE_URL,
                 "https://openai.com/careers"):
        assert link_router.classify_link(bare) is None
        assert fp._classify_link(bare) is None, "the footprint re-export must stay the ARTIFACT set"
        assert fp._dispatchable_link([bare]) is False


def test_predicted_atom_id_is_offline_and_admits_what_it_cannot_know():
    """The free already-present check. Substack returns None because the atom keys on the post's
    numeric id, which only the fetch knows — an honest None, not a failure."""
    assert link_router.predicted_atom_id("https://x.com/k/status/99", "x") == "x:99"
    assert link_router.predicted_atom_id(_ARTICLE_URL, "article") == _ARTICLE_ATOM
    assert link_router.predicted_atom_id("https://github.com/a/b", "github") == "github:a/b"
    assert link_router.predicted_atom_id("https://s.substack.com/p/x", "substack") is None


# ── preview: free by contract ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,hint,kind", [
    ("https://arxiv.org/abs/2504.13171", None, "paper"),
    ("https://github.com/letta-ai/letta", None, "github"),
    ("https://newsletter.example.com/p/a-post", "substack", "substack"),
])
def test_preview_makes_no_fetch_on_a_source_the_caller_can_read(conn, monkeypatch, url, hint, kind):
    """A reference that ROUTES on a fact — a known host, a DOI in the path, a hint — is free. The
    caller has already read the page, so preview must not read it again. Every network seam is
    booby-trapped, including the X one, to prove the X path does not leak into them.

    `article` left this set on 2026-09-09: it is the one route that is a GUESS, and it now pays
    one bounded probe to check the guess. See the test below."""
    from pipeline.ingestion.sources import blog as src
    def _boom(*a, **k):
        raise AssertionError("preview must not fetch a source the caller can read")
    monkeypatch.setattr(src, "_fetch_article", _boom)
    monkeypatch.setattr(ingest_x, "peek_tweet", _boom)
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet", _boom)
    monkeypatch.setattr(link_router, "classify_link_deep", _boom)

    out = hopper.save(conn, None, url, kind_hint=hint)   # embedder is None — nothing may embed
    assert out["status"] == "preview" and out["kind"] == kind
    assert out["already_present"] is False
    assert out["entry_mode"] == "user-saved"
    assert "description" not in out, "no title/description for a kind the caller already read"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


# ── the article fallback is a guess, and it gets checked ──────────────────────────────
#
# Measured 2026-09-09: `https://www.nature.com/articles/s41586-021-03819-2` — AlphaFold — landed as
# `blog:nature.com/articles/…` with `who_id = blog:nature.com`, `what_kind = opinion`, and the
# author list scraped into the title. It never deduped against the same paper saved by DOI, and
# none of its 34 authors reached `sync_paper_author_signals`. `classify_link_deep` had been able to
# read that page's `citation_doi` the whole time; only `ingest_x_footprint` ever called it, so the
# SAME url routed correctly when an Oracle tweeted it and wrongly when a user pasted it.


def test_a_page_that_declares_a_doi_is_a_paper_not_an_article(conn, monkeypatch):
    """The probe REWRITES the reference to the doi.org form it found, which is what makes `save`
    mint the paper instead of the landing page it was handed."""
    monkeypatch.setattr(link_router, "classify_link_deep",
                        lambda u: ("paper", "https://doi.org/10.1038/s41586-021-03819-2", None))

    out = hopper.save(conn, None, "https://www.nature.com/articles/s41586-021-03819-2")

    assert out["kind"] == "paper"
    assert out["reference"] == "https://doi.org/10.1038/s41586-021-03819-2"
    assert out["atom_id"] == "paper:DOI:10.1038/s41586-021-03819-2"


def test_a_page_that_declares_nothing_stays_an_article(conn):
    """The probe must not widen the paper route. A blog post is still a blog post, and the whole
    reason `classify_reference` has an article catch-all is that most links are one. Rides the
    autouse None, which is the same answer a real fetch of a blog post gives."""
    out = hopper.save(conn, None, _ARTICLE_URL)

    assert out["kind"] == "article" and out["atom_id"] == _ARTICLE_ATOM


def test_the_probe_never_runs_when_the_url_already_answered(conn, monkeypatch):
    """LAST RESORT, and the word is load-bearing: a fetch on every preview would make the free
    already-present check cost a round trip per link, which is exactly what the tool description
    tells the host to rely on when it is deciding among ten search results."""
    calls = []
    monkeypatch.setattr(link_router, "classify_link_deep", lambda u: calls.append(u) or None)

    hopper.save(conn, None, "https://pubs.acs.org/doi/10.1021/jacs.4c01234")
    hopper.save(conn, None, "https://arxiv.org/abs/2504.13171")
    hopper.save(conn, None, "https://github.com/letta-ai/letta")

    assert calls == []


def test_previewing_an_x_post_reads_it_because_the_caller_cannot(conn, monkeypatch):
    """David 2026-08-13: the "the host model has already read the page" premise holds for four
    kinds and FAILS for X — x.com serves a JS shell to unauthenticated fetchers, so a model holding
    a status link knows the url and nothing else.

    That breaks the preview's ONE job. Four atom ids are self-describing, so a wrong link is
    visible on sight; `x:2086520133909168332` is unverifiable by a human. OPYT can read what the
    caller cannot, so the preview reads the post and shows it. `description` is `derive_x`'s string,
    i.e. LITERALLY what the atom
    will carry, so what you approve is what gets stored."""
    monkeypatch.setattr(ingest_x, "peek_tweet", lambda tid: dict(_TWEET))
    out = hopper.save(conn, None, _TWEET_URL)

    assert out["status"] == "preview" and out["kind"] == "x"
    assert "@karpathy" in out["description"] and "agent design" in out["description"]
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_an_unreadable_x_post_is_flagged_before_the_user_says_yes(conn, monkeypatch):
    """The other half of the same win. A deleted / protected post, or a missing twitterapi.io key,
    used to be discovered on the CONFIRM — after the user had already approved a save that could
    not happen. Now the preview says so."""
    monkeypatch.setattr(ingest_x, "peek_tweet", lambda tid: None)
    out = hopper.save(conn, None, _TWEET_URL)

    assert "description" not in out
    assert "deleted, protected or suspended" in out["unreadable"]
    assert "would most likely fail" in out["unreadable"]


def test_a_present_x_post_is_not_re_read_to_preview_it(conn, monkeypatch):
    """Already have it → nothing to verify, so nothing to read verifying it. The presence check
    must short-circuit ahead of the card, which consumes a request against a rate bucket."""
    schema.upsert_atom(conn, {"atom_id": "x:1750000000000000000", "source_type": "x"})
    monkeypatch.setattr(ingest_x, "peek_tweet",
                        lambda tid: pytest.fail("must not re-read a present post"))
    out = hopper.save(conn, None, _TWEET_URL)
    assert out["already_present"] is True and "description" not in out


def test_a_confirmed_x_save_does_not_read_the_post_twice(conn, fake_embedder, monkeypatch):
    """`save(confirm=True)` runs the preview internally to route and dedup. Without
    `enrich=not confirm` it would read the tweet for the card and then again for the ingest.

    It must not happen: the second read draws on a 500/15-min rate bucket and
    can return a DIFFERENT tweet from the one previewed if it changed in between."""
    peeks: list = []
    monkeypatch.setattr(ingest_x, "peek_tweet",
                        lambda tid: (peeks.append(tid), dict(_TWEET))[1])
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet",
                        lambda tid, profile=None: (dict(_TWEET), None))
    hopper.save(conn, fake_embedder, _TWEET_URL, confirm=True)
    assert peeks == [], "a confirm must not also pay for a preview card"


def test_a_model_mismatch_stops_a_direct_x_save_before_any_work(conn, fake_embedder, monkeypatch):
    """The direct path shares the bulk writer's pre-spend identity boundary.

    A late guard used to archive a raw snapshot and pay for an embedding before `_write_atom`
    rejected the vector subspace. Neither the X fetch nor the embed call belongs after a known
    mismatch.
    """
    ensure_kb_meta(conn, "model-a", fake_embedder.dim, fake_embedder.provider)
    fake_embedder.model = "model-b"
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet",
                        lambda *_: pytest.fail("model drift must stop before fetching"))
    monkeypatch.setattr(fake_embedder, "embed",
                        lambda *_a, **_kw: pytest.fail("model drift must stop before embedding"))

    assert ingest_x.x_atom_from_url(conn, fake_embedder, _TWEET_URL) == ("failed", None)
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_preview_answers_already_present_for_free(conn):
    schema.upsert_atom(conn, {"atom_id": _ARTICLE_ATOM, "source_type": "blog"})
    out = hopper.save(conn, None, _ARTICLE_URL)
    assert out["already_present"] is True
    assert "no-op" in out["note"]


def test_an_unroutable_reference_reports_and_writes_nothing(conn):
    out = hopper.save(conn, None, "the thing Karpathy said about agents", confirm=True)
    assert out["status"] == "unroutable" and out["kind"] is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


# ── the article path: the headline use case ───────────────────────────────────────────

def _patch_article(monkeypatch, article=None, calls=None):
    from pipeline.ingestion.sources import blog as src
    def _fetch(url):
        if calls is not None:
            calls.append(url)
        return article
    monkeypatch.setattr(src, "_fetch_article", _fetch)
    # The date cascade's one live network arm — a dumped article turns rung-2 recovery ON, so
    # without this stub a test whose article carries no date would hit a real feed.
    monkeypatch.setattr(ingest_blog, "_feed_date_map", lambda base: {})


def test_a_dumped_article_lands_as_a_user_saved_atom(conn, fake_embedder, monkeypatch):
    _patch_article(monkeypatch, _article())
    out = hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)

    assert out["status"] == "saved" and out["atom_id"] == _ARTICLE_ATOM
    src_type, who_id, entry_mode, desc = _atom_row(conn, _ARTICLE_ATOM)
    # source_type stays "blog": the field describes the EXTRACTION SHAPE, not the publisher's
    # identity. A dumped news article is a blog atom.
    assert src_type == "blog"
    assert entry_mode == "user-saved", "a hand-dump means the same thing a bookmark means"
    assert who_id == "blog:theverge.com", "attribution is host-bound, never oracle-bound"
    assert "Jane Doe" in desc, "trafilatura's byline reaches the atom description"


def test_the_entity_a_dump_creates_is_bare(conn, fake_embedder, monkeypatch):
    """Hopper may create an ENTITY (atomizing anything does — it is the substrate resolution runs
    on). It must not assert anything ABOUT that entity from unvetted input. Each null below is
    load-bearing: `kind` because a host is not a human; `name` because upsert COALESCEs and the
    first dumped byline would become theverge.com's permanent name; `identity_links` because the
    self-link is what merges a canonical, and seeding it could fold an org site into a real person.
    """
    _patch_article(monkeypatch, _article())
    hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)

    row = conn.execute("SELECT name, identity_links FROM entities WHERE entity_id=?",
                       ("blog:theverge.com",)).fetchone()
    assert row is not None, "the entity must exist so the atom's who_id resolves"
    assert tuple(row) == (None, None)


def test_a_dump_onto_a_tracked_blog_joins_the_existing_person(conn, fake_embedder, monkeypatch):
    """The other side of bareness. When the host IS already tracked, `upsert_entity`'s COALESCE
    keeps every field, so the dumped article attaches to that person instead of orphaning."""
    schema.upsert_entity(conn, "blog:theverge.com", name="Jane Doe", identity_links=["https://www.theverge.com"])
    _patch_article(monkeypatch, _article())
    hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)

    name, links = conn.execute(
        "SELECT name, identity_links FROM entities WHERE entity_id=?",
        ("blog:theverge.com",)).fetchone()
    assert name == "Jane Doe"
    assert json.loads(links) == ["https://www.theverge.com"]


def test_a_repeat_dump_costs_no_second_fetch(conn, fake_embedder, monkeypatch):
    """Policy B, applied to the paste-it-twice case: the presence check runs BEFORE the fetch, so
    handing over the same link again is a no-op rather than a second page load and embed."""
    calls: list[str] = []
    _patch_article(monkeypatch, _article(), calls=calls)
    assert hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)["status"] == "saved"
    again = hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)

    assert again["status"] == "already_present" and again["atom_id"] == _ARTICLE_ATOM
    assert calls == [_ARTICLE_URL], f"the article was fetched {len(calls)}x for two dumps"


def test_a_blocked_page_stores_nothing(conn, fake_embedder, monkeypatch):
    """Fail-safe: a Cloudflare shell is never stored as a real post, and the status separates 'the
    host stopped us' (retryable) from 'this page has no article on it'."""
    _patch_article(monkeypatch, None)                 # the fetch itself failed → UNDETERMINED
    out = hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)

    assert out["status"] == "blocked" and out["atom_id"] is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_a_model_mismatch_stops_a_direct_article_save_before_any_work(conn, fake_embedder,
                                                                        monkeypatch):
    """The Blog hand-dump must reject model drift before fetching or embedding, like its crawl."""
    ensure_kb_meta(conn, "model-a", fake_embedder.dim, fake_embedder.provider)
    fake_embedder.model = "model-b"
    monkeypatch.setattr("pipeline.ingestion.sources.blog._fetch_article",
                        lambda *_: pytest.fail("model drift must stop before fetching"))
    monkeypatch.setattr(fake_embedder, "embed",
                        lambda *_a, **_kw: pytest.fail("model drift must stop before embedding"))

    assert ingest_blog.article_atom_from_url(conn, fake_embedder, _ARTICLE_URL) == ("failed", None)
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_the_article_adapter_refuses_to_claim_authorship(conn, fake_embedder):
    """The structural stand-in for `eligibility.gate`, which this path deliberately does NOT run
    (it would SKIP the multi-author news sites that ARE the headline use case). The crawl may
    attribute a site to a person; a URL-host dump may not."""
    with pytest.raises(ValueError, match="oracle-footprint"):
        ingest_blog.article_atom_from_url(conn, fake_embedder, _ARTICLE_URL,
                                          entry_mode="oracle-footprint")


# ── the X one-post path ───────────────────────────────────────────────────────────────

_TWEET_URL = "https://x.com/karpathy/status/1750000000000000000"
_TWEET = {
    "id": "1750000000000000000",
    "url": _TWEET_URL,
    "text": "The interesting part of agent design is what happens when a step fails halfway.",
    "createdAt": "Tue Jan 23 12:00:00 +0000 2024",
    "author": {"userName": "karpathy", "name": "Andrej Karpathy", "id": "33836629"},
    "entities": {}, "likeCount": 900, "replyCount": 12,
}


def test_a_dumped_tweet_lands_on_the_bookmark_key(conn, fake_embedder, monkeypatch):
    """Keys on `x:{tweet_id}` — the SAME id the bookmark walk uses, so dumping a post you also
    bookmark collapses to ONE atom rather than a twin. (The opposite of the footprint adapter's
    `xprofile:` split, and for the opposite reason: that path renders by a different route, this
    one renders exactly what the bookmark walk renders.)"""
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet",
                        lambda tid, profile=None: (dict(_TWEET), None))
    out = hopper.save(conn, fake_embedder, _TWEET_URL, confirm=True)

    assert out["status"] == "saved" and out["atom_id"] == "x:1750000000000000000"
    src_type, who_id, entry_mode, _ = _atom_row(conn, "x:1750000000000000000")
    assert (src_type, who_id, entry_mode) == ("x", "x:user:33836629", "user-saved")


def test_a_dumped_tweet_records_the_save_signal(conn, fake_embedder, monkeypatch):
    """A hand-dumped post IS a save, so it writes the author entity and the `save` curation signal
    exactly as a bookmark does — that feeds Stage-4 candidate ranking. It does NOT make an Oracle."""
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet",
                        lambda tid, profile=None: (dict(_TWEET), None))
    hopper.save(conn, fake_embedder, _TWEET_URL, confirm=True)

    # A tweet carries its author's identity, unlike a bare blog host where only the domain is
    # known — so this entity arrives NAMED. (It asserted `kind == "person"` until that column was
    # deleted 2026-08-23; the name is the part that was ever real.)
    name = conn.execute("SELECT name FROM entities WHERE entity_id=?",
                        ("x:user:33836629",)).fetchone()[0]
    assert name
    sig = conn.execute("SELECT signal_type, count FROM curation_signals WHERE entity_id=?",
                       ("x:user:33836629",)).fetchone()
    assert tuple(sig) == ("save", 1)


def test_a_tweet_we_cannot_fetch_stores_nothing(conn, fake_embedder, monkeypatch):
    """Deleted / protected / no key and no conversation — all one outcome: SKIP. No atom, no
    entity, no signal, nothing marked processed."""
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet", lambda tid, profile=None: (None, None))
    out = hopper.save(conn, fake_embedder, _TWEET_URL, confirm=True)

    assert out["status"] == "failed" and out["atom_id"] is None
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0


# ── the artifact kinds, and the invariant that spans all five ─────────────────────────

def test_a_dumped_paper_is_stamped_user_saved_not_author_referenced(conn, fake_embedder,
                                                                    monkeypatch):
    """`author_referenced` means a TRACKED PERSON pointed at it. Nobody pointed at this — David
    handed it over — so the mode has to be the one that records that."""
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2504.13171", "url": "https://arxiv.org/abs/2504.13171",
             "externalIds": {"ArXiv": "2504.13171"}, "title": "Sleep-time Compute",
             "abstract": "Beyond inference scaling at test-time.", "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "full body text")

    out = hopper.save(conn, fake_embedder, "https://arxiv.org/abs/2504.13171", confirm=True)
    assert out["status"] == "saved" and out["atom_id"] == "paper:arXiv:2504.13171"
    assert _atom_row(conn, "paper:arXiv:2504.13171")[2] == "user-saved"


def test_a_present_paper_costs_no_fetch(conn, fake_embedder, monkeypatch):
    """The offline id makes the already-present answer free for papers too."""
    schema.upsert_atom(conn, {"atom_id": "paper:arXiv:2504.13171", "source_type": "paper"})
    from pipeline.kb import ingest_papers as ip
    monkeypatch.setattr(ip, "atomize_paper",
                        lambda *a, **k: pytest.fail("a present paper must not be re-minted"))

    out = hopper.save(conn, fake_embedder, "https://arxiv.org/abs/2504.13171", confirm=True)
    assert out["status"] == "already_present"


def test_a_repeat_substack_save_reports_present_not_saved(conn, fake_embedder, monkeypatch):
    """LIVE-CAUGHT 2026-08-13, and offline tests missed it because they never saved twice.

    Substack is the ONE kind with no url-derived id, so `mint_artifact`'s presence pre-check
    cannot run and the adapter collapses mint-and-present into a single return value. It also
    writes `seen[atom_id]` on a mint (`ingest_substack.py:461`), so asking the ledger afterwards
    answers "yes" either way. Without a snapshot taken BEFORE the call, a repeat save reported
    "saved" for an atom it had not touched — the tool telling the user it did work it did not do,
    which is the exact silent wrongness the preview/confirm shape exists to prevent."""
    from pipeline.kb import ingest_substack as sub
    url = "https://newsletter.example.com/p/a-post"

    def _fake(conn_, emb, u, *, entry_mode="author_referenced", seen=None, img_cache=None):
        aid = "substack:12345"
        if seen is not None and aid in seen:      # policy B — present, no re-render
            return aid
        schema.upsert_atom(conn_, {"atom_id": aid, "source_type": "substack",
                                   "entry_mode": entry_mode})
        if seen is not None:
            seen[aid] = "hash-1"                  # the adapter's own post-mint ledger write
        return aid
    monkeypatch.setattr(sub, "substack_atom_from_url", _fake)

    first = hopper.save(conn, fake_embedder, url, kind_hint="substack", confirm=True)
    second = hopper.save(conn, fake_embedder, url, kind_hint="substack", confirm=True)
    assert first["status"] == "saved"
    assert second["status"] == "already_present", "a repeat must not claim it saved anything"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


def test_a_body_without_metadata_is_reported_not_silently_called_saved(conn, fake_embedder,
                                                                       monkeypatch):
    """LIVE-OBSERVED 2026-08-13 on arXiv 2504.13171: the PDF resolved to 55 chunks of real full
    text while the Semantic Scholar lookup came back `undetermined` (throttled), so the paper
    landed as "Untitled" with a placeholder author and no date — and papers are immutable under
    Policy B, so it can never be repaired.

    Hopper cannot fix that (the repair belongs in `ingest_papers` and touches an invariant). What
    it must not do is answer a bare "saved". A user told only that would discover months later
    that the atom cannot say who wrote the thing or when."""
    from pipeline.kb import ingest_papers as ip
    thin = {"paperId": "arXiv:2401.99999", "url": "https://arxiv.org/abs/2401.99999",
            "externalIds": {"ArXiv": "2401.99999"}, "authors": []}     # no title, no date
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(thin))
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "a real full body, many pages of it")

    out = hopper.save(conn, fake_embedder, "https://arxiv.org/abs/2401.99999", confirm=True)
    assert out["status"] == "saved", "the body IS worth keeping — this is a degraded success"
    assert "warning" in out and "no title, no date" in out["warning"]
    assert "will NOT repair it" in out["warning"]


def test_a_healthy_atom_carries_no_warning(conn, fake_embedder, monkeypatch):
    """The other half: the warning must stay rare, or it becomes noise the host model learns to
    skip. An article with a title and a date says nothing."""
    _patch_article(monkeypatch, _article())
    out = hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)
    assert out["status"] == "saved" and "warning" not in out


def test_hopper_never_creates_an_oracle(conn, fake_embedder, monkeypatch):
    """THE invariant. `add_oracle` is the human gate on who becomes a tracked, trusted person;
    growing the roster from a pasted link would route around it. Dumps may create entities — that
    is what ingesting anything does — but the `oracles` table must be untouched by all five kinds.
    """
    _patch_article(monkeypatch, _article())
    monkeypatch.setattr(ingest_x, "_fetch_one_tweet",
                        lambda tid, profile=None: (dict(_TWEET), None))
    hopper.save(conn, fake_embedder, _ARTICLE_URL, confirm=True)
    hopper.save(conn, fake_embedder, _TWEET_URL, confirm=True)

    assert conn.execute("SELECT COUNT(*) FROM oracles").fetchone()[0] == 0
