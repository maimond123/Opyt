"""Card fill: a homepage, product, job or event
link is never fetched as a document; its page's title and description are read once, kept by URL,
and written into every post that links it in X's own card format, so "which tools are my Oracles
recommending?" finds the post by what it links."""
from __future__ import annotations

import pytest

from pipeline.ingestion.x_render import tweet_to_markdown
from pipeline.kb import link_cards, schema
from pipeline.kb.link_cards import Card
from pipeline.oracles import roster
from pipeline.oracles.sources.references.cards import Cards
from pipeline.platforms.web import page_head
from pipeline.platforms.web.page_head import Head
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.serial import Serial
from tests.kb.test_ingest_x_footprint import _LONG, _patch_fetch, _payload, _raw, _sync, _version

_HOME = "https://jevjobs.ai/"
_JEV = Card("Use Jev to discover jobs", "Jev finds the roles that fit you, from every job board.")


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture()
def cards_loop(conn):
    roster.upsert_oracle(conn, "x:user:99", name="Carol")    # a forgotten Oracle owes nothing
    conn.commit()
    health = Outages(PLATFORM_PAUSES_S)
    loop = QuotaLoop("web", [Cards(health)], executor=Serial(), outages=health,
                     connect=schema.connect)
    loop.ctx.cache.update(session={})
    return loop


def _reads(monkeypatch, heads: dict[str, Head | None]) -> list[str]:
    """Stub the page read: each URL's head, in order of asking."""
    calls = []

    def read_head(url):
        calls.append(url)
        return heads.get(url)
    monkeypatch.setattr(page_head, "read_head", read_head)
    return calls


def _body(conn, atom_id):
    return schema.read_body(conn, atom_id)


# ── what the card says, decided from the page ─────────────────────────────────

@pytest.mark.parametrize("url, head, want", [
    ("https://vercel.com/careers", Head("Careers – Vercel", "Join us."), True),       # thin
    ("https://vercel.com/careers", Head("Careers", "Careers"), True),                 # = title
    ("https://vercel.com/careers", Head("Careers – Vercel", "Build the web with us, wherever "
                                     "you are, on the team that ships Next.js."), False),
    ("https://vercel.com/", Head("Vercel", ""), False),                    # already the homepage
    ("https://jobs.lever.co/acme/1", Head("Engineer", ""), False),        # hosted: page speaks
])
def test_when_the_homepage_speaks_for_the_page(url, head, want):
    assert link_cards.wants_homepage(url, head) is want


def test_a_page_with_no_title_is_named_by_its_host_and_a_silent_one_is_empty():
    assert link_cards.card_of("https://www.rhys.dev/", Head("", "Builds things")) == Card(
        "rhys.dev", "Builds things")
    assert link_cards.card_of("https://rhys.dev/", Head()) == Card("", "")
    assert link_cards.card_of("https://rhys.dev/", Head("Rhys Sullivan", "Rhys Sullivan")) == Card(
        "Rhys Sullivan", "")                                       # says its name once


def test_a_thin_page_reads_its_homepage_and_a_failed_page_reads_nothing(monkeypatch):
    calls = _reads(monkeypatch, {
        "https://vercel.com/careers": Head("Careers – Vercel", "Join us."),
        "https://vercel.com/": Head("Vercel", "The autonomous stack for every app and agent.")})
    assert link_cards.read("https://vercel.com/careers") == Card(
        "Vercel", "The autonomous stack for every app and agent.")
    assert calls == ["https://vercel.com/careers", "https://vercel.com/"]

    _reads(monkeypatch, {})
    assert link_cards.read("https://vercel.com/careers") is None


def test_a_hosted_page_keeps_its_own_card(monkeypatch):
    calls = _reads(monkeypatch, {"https://jobs.lever.co/acme/1": Head("Engineer — London", "")})
    assert link_cards.read("https://jobs.lever.co/acme/1") == Card("Engineer — London", "")
    assert calls == ["https://jobs.lever.co/acme/1"]


# ── the post shows it: a render with the card, and a stored post filled, agree ──

def test_filling_a_stored_post_gives_what_a_render_with_the_card_gives():
    tweet = {"id": "1", "text": "wtf.... https://t.co/x", "author": {"userName": "jj", "id": "5"},
             "entities": {"urls": [
                 {"url": "https://t.co/x", "expanded_url": _HOME, "display_url": "jevjobs.ai"},
                 {"url": "https://t.co/y", "expanded_url": "https://acme.com/", "display_url":
                  "acme.com"}]}}
    plain = tweet_to_markdown(dict(tweet), source="x-bookmark")
    attached = dict(tweet)
    link_cards.attach(attached, {_HOME: _JEV})

    filled = link_cards.fill_markdown(plain, _HOME, _JEV)

    assert filled == tweet_to_markdown(attached, source="x-bookmark")
    assert f"> **[Use Jev to discover jobs]({_HOME})**\n> Jev finds" in filled
    assert "- [acme.com](https://acme.com/)" in filled                    # its neighbour untouched
    assert link_cards.fill_markdown(plain, _HOME, Card("", "")) == plain


# ── the worker: read once, write into every post that owes it ──────────────────

def test_an_oracles_short_post_gets_its_card_and_a_rewalk_keeps_it(conn, cards_loop, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("44", text="wtf....", link=_HOME)])
    _sync(conn, handle="carol")
    assert [e["url"] for e in _payload(conn, "xprofile:44")["links_owed"]] == [_HOME]
    calls = _reads(monkeypatch, {_HOME: Head(_JEV.title, _JEV.description)})

    assert cards_loop.step(conn).result == "filled"

    assert calls == [_HOME]
    assert f"> **[Use Jev to discover jobs]({_HOME})**" in _body(conn, "xprofile:44")
    assert _payload(conn, "xprofile:44")["links_owed"] == []
    assert link_cards.stored(conn, _HOME) == _JEV
    assert conn.execute("SELECT count(*) FROM bodies WHERE atom_id='xprofile:44' "
                        "AND text LIKE '%Jev finds the roles%'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM bodies_fts WHERE bodies_fts MATCH 'roles'"
                        ).fetchone()[0] == 1                  # and the card is searchable
    version = _version(conn, "xprofile:44")

    again = _sync(conn, handle="carol")       # the render attaches the card

    assert again["skipped"] == 1 and _version(conn, "xprofile:44") == version


def test_a_card_already_read_is_written_without_reading_the_page(conn, cards_loop, monkeypatch):
    with conn:
        link_cards.save(conn, _HOME, _JEV)
    _patch_fetch(monkeypatch, [_raw("45", text="also this", link=_HOME)])
    calls = _reads(monkeypatch, {})
    _sync(conn, handle="carol")                # rendered with the card already

    assert cards_loop.step(conn).result == "final"            # nothing left to fill

    assert calls == [] and f"[Use Jev to discover jobs]({_HOME})" in _body(conn, "xprofile:45")
    assert _payload(conn, "xprofile:45")["links_owed"] == []


def test_a_page_that_cannot_be_read_now_writes_nothing_and_stays_owed(conn, cards_loop, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("46", text="this", link=_HOME)])
    _sync(conn, handle="carol")
    version = _version(conn, "xprofile:46")
    _reads(monkeypatch, {_HOME: None})

    step = cards_loop.step(conn)

    assert step.result == "blip"
    assert _version(conn, "xprofile:46") == version and link_cards.stored(conn, _HOME) is None
    assert [e["url"] for e in _payload(conn, "xprofile:46")["links_owed"]] == [_HOME]


def test_a_long_post_with_an_x_card_owes_no_card(conn, monkeypatch):
    post = {**_raw("47", text=_LONG, link=_HOME), "card": {"binding_values": [
        {"key": "title", "value": {"string_value": "Jev"}}]}}
    _patch_fetch(monkeypatch, [post])
    _sync(conn, handle="carol")
    assert _payload(conn, "xprofile:47")["links_owed"] == []
