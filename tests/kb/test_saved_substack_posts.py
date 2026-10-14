"""SavedSubstackPosts saves one offered post per unit: its atom and its done-state land together,
its body state tells the truth, and a refused body or a refused write leaves the post owed."""

import json
from datetime import datetime, timezone

import pytest

from pipeline.curation import saved_items, substack_render, substack_saved, substack_walks
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.platforms import sessions
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.source import Changed, Context

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
BODY = ("<h2>Intro</h2><p>Autonomous agents compose small tools into larger systems, and the "
        "framework matters.</p><p>See <a href='https://arxiv.org/abs/1234'>the paper</a>.</p>")


def rec(pid=99, *, audience="everyone", pub="https://carol.substack.com"):
    return {"id": pid, "title": "Scaling Laws", "subtitle": "a deep dive",
            "post_date": "2026-05-01T00:00:00Z", "author_handle": "carol", "author_name": "Carol",
            "publication_name": "Carol Writes", "publication_url": pub,
            "wordcount": 900, "audience": audience, "slug": f"post-{pid}",
            "url": f"{pub}/p/post-{pid}", "preview": "a short preview line"}


class Transport:
    """`saved_source()`: the bodies, and whether the host accepted the session."""

    def __init__(self, body=BODY, authenticated=True):
        self.body, self.authenticated, self.asked = body, authenticated, []

    def authenticated_body(self, base):
        return self.authenticated

    def full_post(self, base, slug):
        self.asked.append(slug)
        if isinstance(self.body, Exception):
            raise self.body
        return {"body_html": self.body} if self.body else None


@pytest.fixture()
def transport(monkeypatch):
    t = Transport()
    monkeypatch.setattr(sessions, "saved_source", lambda profile=None: t)
    monkeypatch.setattr(connections, "_connected_substack", lambda: True)
    return t


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    saved_items.init_schema(c)
    yield c
    c.close()


def offer(conn, *recs, account="42"):
    saved_items.offer(conn, "substack", account,
                      [(str(r["id"]), saved_items.revision(r), r) for r in recs], NOW)


def source(outages=None):
    return substack_saved.SavedSubstackPosts("web", outages or Outages(PLATFORM_PAUSES_S))


def run(conn, src=None):
    """One unit: the source's outcome, or None when nothing is owed."""
    src = src or source()
    unit = src.next_unit(conn, NOW)
    if unit is None:
        return None
    return src.record(conn, unit, src.fetch(conn, unit, Context("web")))


def atom(conn, atom_id="substack:99"):
    row = conn.execute("SELECT entry_mode, payload FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()
    return (row[0], json.loads(row[1])) if row else None


def test_a_saved_post_lands_with_its_body_its_save_and_its_done_state(conn, transport):
    offer(conn, rec())

    outcome = run(conn)

    assert (outcome.status, outcome.failure) == ("saved", None)
    mode, payload = atom(conn)
    assert mode == "user-saved" and payload["body_state"] == "complete"
    assert saved_items.owed(conn, "substack") == []
    assert conn.execute("SELECT count FROM curation_signals WHERE entity_id='substack:carol' "
                        "AND signal_type='save' AND platform='substack'").fetchone()[0] == 1
    assert run(conn) is None


def test_the_unit_runs_on_the_publications_host(conn, transport):
    offer(conn, rec(pub="https://letters.example.com"))
    unit = source().next_unit(conn, NOW)

    assert unit.host == "letters.example.com"
    assert unit.item.platform == "substack"


def test_a_paid_post_read_without_an_accepted_session_is_partial(conn, transport):
    transport.authenticated = False
    offer(conn, rec(audience="only_paid"))

    assert run(conn).status == "saved"
    assert atom(conn)[1]["body_state"] == "partial"
    assert atom(conn)[1]["paywalled"] is True


def test_a_paid_post_over_an_accepted_session_is_complete(conn, transport):
    offer(conn, rec(audience="only_paid"))

    run(conn)
    assert atom(conn)[1]["body_state"] == "complete"


def test_a_refused_body_is_stored_pending_and_the_post_stays_owed(conn, transport):
    """A Cloudflare block is not an empty post. The stub is searchable meanwhile; the post is
    charged as a block, which waits as an account and counts toward the Substack outage."""
    transport.body = substack_contracts.SubstackFetchError("HTTP 403")
    offer(conn, rec())

    outcome = run(conn)

    assert (outcome.status, outcome.failure, outcome.outage) == \
        ("body_pending", retries.ACCOUNT, True)
    assert atom(conn)[1]["body_state"] == "pending"
    assert [s.item for s in saved_items.owed(conn, "substack")] == ["99"]


def test_a_pending_post_whose_body_arrives_later_becomes_complete_and_done(conn, transport):
    transport.body = substack_contracts.SubstackFetchError("HTTP 403")
    offer(conn, rec())
    run(conn)

    transport.body = BODY
    assert run(conn).status == "saved"
    assert atom(conn)[1]["body_state"] == "complete"
    assert saved_items.owed(conn, "substack") == []


def test_a_pending_post_whose_refetch_fails_plainly_is_absent_not_an_outage(conn, transport):
    """Once blocked, a post whose next fetch fails for an ordinary reason (deleted, no
    slug, a reset) has no body to fetch. Its stub's markdown is unchanged, but it is not blocked
    again, so it must not count toward the shared Substack outage or stay owed forever."""
    transport.body = substack_contracts.SubstackFetchError("HTTP 403")
    offer(conn, rec())
    run(conn)

    transport.body = RuntimeError("TLS reset, not a block")
    outcome = run(conn)

    assert (outcome.status, outcome.failure, outcome.outage) == ("saved", None, False)
    assert atom(conn)[1]["body_state"] == "absent"
    assert saved_items.owed(conn, "substack") == []


def test_a_post_with_no_body_is_absent_and_done(conn, transport):
    transport.body = None
    offer(conn, rec())

    assert run(conn).status == "saved"
    assert atom(conn)[1]["body_state"] == "absent"
    assert saved_items.owed(conn, "substack") == []


def test_a_stored_post_is_done_without_another_request(conn, transport):
    offer(conn, rec())
    run(conn)
    offer(conn, dict(rec(), title="Scaling Laws, edited"))     # a new revision of a stored post

    assert run(conn).status == "present"
    assert transport.asked == ["post-99"]
    assert saved_items.owed(conn, "substack") == []


def test_a_newer_offer_during_the_fetch_refuses_the_write(conn, transport,
                                                          monkeypatch):
    offer(conn, rec())
    src = source()
    unit = src.next_unit(conn, NOW)
    fetch = transport.full_post

    def offered_meanwhile(base, slug):
        offer(conn, dict(rec(), title="changed"))
        return fetch(base, slug)

    transport.full_post = offered_meanwhile
    with pytest.raises(Changed):
        src.fetch(conn, unit, Context("web"))
    assert atom(conn) is None


def test_a_missing_session_never_pauses_the_web_pool(conn, transport):
    transport.body = SyncAuthError("no managed profile")
    offer(conn, rec())
    src = source()
    unit = src.next_unit(conn, NOW)

    with pytest.raises(substack_walks.NotSignedIn) as raised:
        src.fetch(conn, unit, Context("web"))
    assert retries.kind_of(raised.value) == retries.BLIP
    assert atom(conn) is None


def test_a_paused_substack_offers_no_saved_post(conn, transport):
    outages = Outages(PLATFORM_PAUSES_S)
    until = outages.pause("substack", NOW, 1.0)
    offer(conn, rec())

    assert source(outages).next_unit(conn, NOW) is None
    assert source(outages).next_unit(conn, until) is not None


def test_without_a_session_no_saved_post_is_offered(conn, transport, monkeypatch):
    monkeypatch.setattr(connections, "_connected_substack", lambda: False)
    offer(conn, rec())

    assert source().next_unit(conn, NOW) is None


# ── whole-list import ─────────────────

IMG = "https://substackcdn.com/image/fetch/$s_!abc/https%3A//example.com/chart.png"
IMG_BODY = (f'<p>Revenue grew sharply.</p><figure><img src="{IMG}" alt="revenue chart">'
            f'</figure><p>See <a href="https://arxiv.org/abs/1234">the paper</a>.</p>')


def test_saved_post_images_survive_extraction():
    """`clean_body_html` must not run either extractor with images disabled, or every chart in a
    saved post is deleted before anything could describe it. Fails if either flag flips."""
    md = substack_render.clean_body_html(IMG_BODY)
    assert md.count("![") == 1 and IMG in md


def test_saved_post_image_becomes_searchable_body_text(conn, transport, monkeypatch):
    """The description must be inside the hashed + indexed body, or it never reaches the index
    — same ordering as both footprint adapters."""
    from pipeline import chart_read
    monkeypatch.setattr(chart_read, "read_image",
                        lambda url, context="": chart_read.MediaRead(
                            "a bar chart of quarterly revenue", "chart"))
    transport.body = IMG_BODY
    offer(conn, rec())

    assert run(conn).status == "saved"
    assert "*Image:* a bar chart of quarterly revenue" in schema.read_body(conn, "substack:99")
    assert conn.execute("SELECT count(*) FROM bodies_fts WHERE bodies_fts MATCH 'quarterly'"
                        ).fetchone()[0] == 1


def test_a_blocked_body_is_not_recorded_as_paywalled(conn, transport):
    """`paywalled` must not be `only_paid OR not got_body`, which would store `paywalled=True` for
    every block. The two facts are different, `body_state` carries the second, and `export` ships
    `paywalled` into shared KBs."""
    transport.body = substack_contracts.SubstackFetchError("cloudflare challenge")
    offer(conn, rec())
    run(conn)

    payload = atom(conn)[1]
    assert payload["paywalled"] is False and payload["body_state"] == "pending"
    assert payload["body_basis"] == "observed"
    assert schema.load_body_pending(conn, "substack") == {"substack:99"}


def test_a_real_paywall_is_still_recorded_as_paywalled(conn, transport):
    transport.body = None
    offer(conn, rec(audience="only_paid"))
    run(conn)

    assert atom(conn)[1]["paywalled"] is True


@pytest.mark.parametrize("body, preview_only, state, basis", [
    (substack_saved.Body("", False, True), False, "pending", "observed"),
    (substack_saved.Body("text", True, False), True, "partial", "stated"),
    (substack_saved.Body("text", True, False), False, "complete", "stated"),
    (substack_saved.Body("", False, False), False, "absent", "observed"),
])
def test_body_state_table(body, preview_only, state, basis):
    assert substack_saved.body_state(body, preview_only) == (state, basis)


def test_a_refused_write_leaves_no_author_behind(conn, transport):
    """The author's entity and `save` presence land only with the post's atom."""
    offer(conn, rec())
    src = source()
    unit = src.next_unit(conn, NOW)
    fetch = transport.full_post

    def offered_meanwhile(base, slug):
        offer(conn, dict(rec(), title="changed"))
        return fetch(base, slug)

    transport.full_post = offered_meanwhile
    with pytest.raises(Changed):
        src.fetch(conn, unit, Context("web"))
    assert atom(conn) is None
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0
