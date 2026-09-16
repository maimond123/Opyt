"""ingest_curation — the Step-2 adapter (T3-T8). Offline: harvested fetch logic is
monkeypatched, so these prove the WIRING (entity+signal stamping, full-body atom build,
stub fallback, failure isolation), not the live scrape."""
from __future__ import annotations

import json

import pytest

from pipeline.kb import ingest_curation as ic
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _sig(conn, entity_id, signal_type, platform="x"):
    return conn.execute(
        "SELECT count, extra FROM curation_signals WHERE entity_id=? AND signal_type=? "
        "AND platform=?", (entity_id, signal_type, platform)).fetchone()


def test_curation_cli_rejects_x_profile(kb_home):
    with pytest.raises(SystemExit) as exited:
        ic._cli(["--x-profile", "Default"])

    assert exited.value.code == 2


def test_x_collector_uses_the_managed_session_without_profile(conn, monkeypatch):
    spec = next(spec for spec in ic.COLLECTOR_SPECS if spec.collector == "x_following")
    monkeypatch.setattr(ic, "sync_following_signals",
                        lambda conn: {"source": "x-following", "following": 0})

    assert ic.run_collector(conn, spec) == {"source": "x-following", "following": 0}


# ── people-only stampers ─────────────────────────────────────────────────────────

def test_stamp_x_person_writes_entity_signal_and_identity_link(conn):
    cand = {"user_id": "33836629", "display_name": "Elon", "site": "https://tesla.com"}
    ic._stamp_x_person(conn, cand, "like", count=4)
    ent = conn.execute("SELECT name, identity_links FROM entities WHERE entity_id='x:user:33836629'").fetchone()
    assert ent["name"] == "Elon"
    assert json.loads(ent["identity_links"]) == ["https://tesla.com"]
    assert _sig(conn, "x:user:33836629", "like")["count"] == 4


def test_following_stamper_wiring(conn, monkeypatch):
    from pipeline.ingestion import x_graphql_core as core
    monkeypatch.setattr(core, "read_x_cookies", lambda: {"twid": "u=1"})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: "1")
    monkeypatch.setattr(core, "auth_headers", lambda cookies, referer: {})
    monkeypatch.setattr(core, "fetch_following", lambda c, h, v: [
        {"user_id": "2", "display_name": "A", "site": "https://a.com"},
        {"user_id": "3", "display_name": "B", "site": ""},
    ])
    out = ic.sync_following_signals(conn)
    assert out == {"source": "x-following", "following": 2}
    assert _sig(conn, "x:user:2", "follow")["count"] == 1
    assert _sig(conn, "x:user:3", "follow")["count"] == 1


def test_following_skips_without_viewer_id(conn, monkeypatch):
    from pipeline.ingestion import x_graphql_core as core
    monkeypatch.setattr(core, "read_x_cookies", lambda: {})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: None)
    assert ic.sync_following_signals(conn) == {"source": "x-following", "skipped": "no_viewer_id"}


# ── one X person, three signals, ONE rest_id (the join invariant, at the DB level) ──

# ── the bookmark SIGNAL collector (2026-09-13) ──────────────────────────────────
#
# ⚠️ WHY IT EXISTS. `save` was reachable only through the content arm, which runs on the
# `bookmark_catchup` rail, which only the resident worker launches. With no worker the row queued
# at consent was never claimed, so the signal never landed at all — the screen scored every
# candidate on one signal, nothing cleared CORROBORATION_MIN, and the user was told "each of these
# showed up once. Nothing is pre-selected." This walk is five requests and no per-item call, so it
# belongs with the other free collectors in the blocking setup pass; bodies and threads do not.

def _bookmark_walk(monkeypatch, items, *, viewer="1", raises=None):
    from pipeline.ingestion import x_graphql as xg
    from pipeline.ingestion import x_graphql_core as core

    monkeypatch.setattr(core, "read_x_cookies", lambda: {"twid": f"u={viewer}"})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: viewer)

    def walk(limit=0):
        for it in items:
            yield it
        if raises is not None:
            raise raises

    monkeypatch.setattr(xg, "iterate_bookmarks", walk)


def _bm(tid, uid, name="A", handle="a", site=None):
    return {"id": tid, "author": {"id": uid, "userName": handle, "name": name, "site": site}}


def test_the_bookmark_walk_counts_saves_per_author(conn, monkeypatch):
    _bookmark_walk(monkeypatch, [_bm("1", "5"), _bm("2", "5"), _bm("3", "6", handle="b")])

    out = ic.sync_bookmark_signals(conn)

    assert (out["bookmarks"], out["candidates"]) == (3, 2)
    assert _sig(conn, "x:user:5", "save")[0] == 2      # the count the screen renders as "12x"
    assert _sig(conn, "x:user:6", "save")[0] == 1
    assert "undetermined" not in out


def test_your_own_bookmarked_post_does_not_make_you_a_candidate(conn, monkeypatch):
    _bookmark_walk(monkeypatch, [_bm("1", "1"), _bm("2", "5")], viewer="1")

    out = ic.sync_bookmark_signals(conn)

    assert out["candidates"] == 1 and out["bookmarks"] == 1
    assert _sig(conn, "x:user:1", "save") is None


def test_the_bookmark_walk_skips_without_viewer_id(conn, monkeypatch):
    _bookmark_walk(monkeypatch, [_bm("1", "5")], viewer=None)
    assert ic.sync_bookmark_signals(conn) == {
        "source": "x-bookmark-signals", "skipped": "no_viewer_id"}


def test_a_full_walk_REPLACES_the_count_so_an_unsaved_post_is_reflected(conn, monkeypatch):
    """`set_signal`, like its four peers: this hands in a person's whole aggregate, not a delta,
    so summing would inflate on every re-run. Unbookmarking two of three must read as one."""
    _bookmark_walk(monkeypatch, [_bm("1", "5"), _bm("2", "5"), _bm("3", "5")])
    ic.sync_bookmark_signals(conn)
    assert _sig(conn, "x:user:5", "save")[0] == 3

    _bookmark_walk(monkeypatch, [_bm("1", "5")])
    ic.sync_bookmark_signals(conn)

    assert _sig(conn, "x:user:5", "save")[0] == 1


def test_an_interrupted_walk_keeps_what_it_saw_but_claims_no_total(conn, monkeypatch):
    """⚠️ THE ONE WAY THIS COULD DESTROY INFORMATION. A partial aggregate written with
    `set_signal` REPLACES a correct count with a smaller one. The pages that answered are still
    real — same rule the bookmark walk's own skip paths follow — so they are kept as PRESENCE."""
    from pipeline.ingestion import x_graphql_core as core

    _bookmark_walk(monkeypatch, [_bm(str(i), "5") for i in range(4)])
    ic.sync_bookmark_signals(conn)
    assert _sig(conn, "x:user:5", "save")[0] == 4

    _bookmark_walk(monkeypatch, [_bm("1", "5"), _bm("9", "7", handle="c")],
                   raises=core.XRateLimited("window spent"))
    out = ic.sync_bookmark_signals(conn)

    assert out["undetermined"] == 1                    # `classify_run`'s BLOCKED, not a clean zero
    assert _sig(conn, "x:user:5", "save")[0] == 4      # NOT lowered to 1
    assert _sig(conn, "x:user:7", "save")[0] == 1      # but the new person still landed


def test_the_content_arm_never_inflates_the_count_the_walk_owns(conn, monkeypatch):
    """`sync_bookmarks` stamps once per atom against a corpus re-walked forever. It used
    `add_signal`, which was right while it was the only writer and double-counts the moment the
    full-set walk sets a true aggregate. Presence is all it may assert now."""
    from pipeline.kb import schema as sch

    _bookmark_walk(monkeypatch, [_bm("1", "5"), _bm("2", "5")])
    ic.sync_bookmark_signals(conn)
    assert _sig(conn, "x:user:5", "save")[0] == 2

    sch.ensure_signal(conn, "x:user:5", "save", "x")   # what the content arm now does, per atom
    sch.ensure_signal(conn, "x:user:5", "save", "x")

    assert _sig(conn, "x:user:5", "save")[0] == 2


def test_multi_signal_person_unifies_on_one_rest_id(conn):
    cand = {"user_id": "777", "display_name": "P", "site": "https://p.com"}
    ic._stamp_x_person(conn, cand, "like", count=2)
    ic._stamp_x_person(conn, cand, "follow")
    ic._stamp_x_person(conn, cand, "list", count=1, extra={"list_names": ["AI"]})
    rows = conn.execute("SELECT signal_type FROM curation_signals WHERE entity_id='x:user:777'").fetchall()
    assert {r["signal_type"] for r in rows} == {"like", "follow", "list"}
    assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id='x:user:777'").fetchone()[0] == 1


# ── the follow list: a REFUSAL is not an empty graph ─────────────────────────────

def test_a_refused_follow_list_records_a_SKIP_not_a_successful_zero(conn, monkeypatch):
    """MEASURED 2026-09-08: substack.com 403'd every reader route for over half an hour, and this
    collector reported `follows: 0`. `_stamp_run` reads the `skipped` key to decide the
    clock's status, so with none the pass was stored as `found=0, status=ok` — a truthful-looking
    record that the account follows nobody.

    Downstream that is not cosmetic. `onboard_state._curation_ok` counts an `ok` run as the
    curation phase being DONE, so a fresh install advanced to `done` with zero candidates and told
    the user their sign-in had gone stale and to reconnect — false, and unfixable by reconnecting.
    The status word is the whole fix: `refused` keeps the phase honest, and the rail retries on its
    own cadence."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    class _Refused:
        def follows(self):
            raise sub.SubstackListingError("reader page refused: HTTP 403")

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Refused())

    out = ic.sync_substack_follows(conn)
    assert out["skipped"] == "refused"
    assert "follows" not in out              # never a count — a count would be read as a fact

    ic._stamp_run(conn, ic.SPEC_BY_COLLECTOR["substack_follows"], out)
    row = curation_state.get_run(conn, "substack_follows")
    assert row.last_status == "refused" and not row.ok
    assert row.found is None                 # counts ride only on a success


def test_an_empty_follow_list_is_still_a_successful_zero(conn, monkeypatch):
    """The other half, and it has to keep working: an account that genuinely follows nobody is a
    SUCCESSFUL walk that found nobody. Collapsing this into the refusal case would trade one
    misreport for its mirror image."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    class _Empty:
        def follows(self):
            return []

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Empty())

    out = ic.sync_substack_follows(conn)
    assert out["follows"] == 0 and "skipped" not in out

    ic._stamp_run(conn, ic.SPEC_BY_COLLECTOR["substack_follows"], out)
    row = curation_state.get_run(conn, "substack_follows")
    assert row.ok and row.found == 0


# ── a skip has to say WHY, not just that it skipped ──────────────────────────────

def test_a_refused_run_records_the_reason_and_not_just_the_word_refused(conn, monkeypatch):
    """MEASURED 2026-09-15 on the hosted box: `mcp_child.log` carried "substack follow-list
    REFUSED ... : hosted follow-list request was refused" while that home's `collector_runs` row
    carried `last_status='refused', last_detail=NULL`. `_stamp_run` read the result's `skipped`
    key and never its `detail`, so the only copy of the reason lived in a log that nothing
    downstream reads — and a diagnosis of that home could report the refusal but not its cause.

    Three collectors return `{"skipped": "refused", "detail": ...}` and all three lost it the same
    way, so the fix belongs in the one writer rather than three call sites."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    class _Refused:
        def follows(self):
            raise sub.SubstackListingError("hosted follow-list request was refused")

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Refused())

    out = ic.sync_substack_follows(conn)
    ic._stamp_run(conn, ic.SPEC_BY_COLLECTOR["substack_follows"], out)

    row = curation_state.get_run(conn, "substack_follows")
    assert row.last_status == "refused"
    assert "hosted follow-list request was refused" in (row.last_detail or "")


def test_a_skip_whose_status_is_its_whole_reason_attaches_no_detail(conn):
    """The mirror half. `no_viewer_id` is fully described by its own status word — there is no
    sentence underneath it — so the four X collectors that return it must not start recording an
    empty or invented one. This is what keeps the new read a PASS-THROUGH of what the collector
    said rather than a second place that manufactures detail."""
    from pipeline.kb import curation_state

    ic._stamp_run(conn, ic.SPEC_BY_COLLECTOR["x_following"],
                  {"source": "x-following", "skipped": "no_viewer_id"})

    row = curation_state.get_run(conn, "x_following")
    assert row.last_status == "no_viewer_id"
    assert not row.last_detail


def test_a_later_success_clears_the_detail_a_refusal_left(conn, monkeypatch):
    """A STICKY detail is the failure this whole area keeps producing — §7 of the contention
    handoff is about a sticky `last_error` making a coin flip look chronic. `last_detail` is about
    the LAST outcome, so a run that succeeds must blank the previous refusal's sentence rather
    than leave it standing next to `status='ok'`."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    class _Refused:
        def follows(self):
            raise sub.SubstackListingError("reader page refused: HTTP 403")

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Refused())
    spec = ic.SPEC_BY_COLLECTOR["substack_follows"]
    ic._stamp_run(conn, spec, ic.sync_substack_follows(conn))
    assert curation_state.get_run(conn, "substack_follows").last_detail

    class _Healthy:
        def follows(self):
            return []

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Healthy())
    ic._stamp_run(conn, spec, ic.sync_substack_follows(conn))

    row = curation_state.get_run(conn, "substack_follows")
    assert row.ok and not row.last_detail


# ── the subscription list: a DIFFERENT graph, and the first claim about money ─────

_SUBS_PAGE = {
    "subscriptions": [
        {"id": 11, "publication_id": 101, "membership_state": "free_signup",
         "is_favorite": False, "is_founding": False, "visibility": "private"},
        {"id": 12, "publication_id": 102, "membership_state": "subscribed",
         "is_favorite": True, "is_founding": False, "visibility": "private"},
        # A subscription whose publication is missing from the join — no URL, so no entity key.
        {"id": 13, "publication_id": 999, "membership_state": "free_signup"},
    ],
    "publications": [
        {"id": 101, "name": "Free Pub", "subdomain": "freepub", "custom_domain": None,
         "author_id": 5, "has_plans": False},
        {"id": 102, "name": "Paid Pub", "subdomain": "paidpub",
         "custom_domain": "letters.example.com", "author_id": 6, "has_plans": True},
    ],
    "publicationUsers": [],
    "publicationsWithPledges": [],
}


def test_the_parser_joins_a_subscription_to_its_publication(kb_home):
    """The payload is two parallel arrays; the join is `subscriptions[].publication_id` ->
    `publications[].id` (measured 36/36). A subscription with no publication in the join has no
    URL, and the URL is the entity key — so it is dropped rather than keyed on `substack:unknown`.

    A custom domain wins over the subdomain, which is not an edge case: 20 of 36 measured
    subscriptions are on one."""
    from pipeline.ingestion.sources import substack as sub

    assert sub.parse_subscription_page(_SUBS_PAGE) == [
        {"id": 11, "name": "Free Pub", "url": "https://freepub.substack.com",
         "membership_state": "free_signup", "is_favorite": False},
        {"id": 12, "name": "Paid Pub", "url": "https://letters.example.com",
         "membership_state": "subscribed", "is_favorite": True},
    ]


def test_a_wrong_cursor_parameter_stops_the_walk_instead_of_looping(kb_home):
    """The `cursor=` parameter NAME is inferred — no measured response has ever carried a
    `nextCursor`, so the walk has never actually paged. This is the failure that inference buys:
    the server ignores an unknown parameter and re-serves page 1 forever.

    Dedupe makes it terminate. Every record on the second page is already `seen`, so
    `new_this_page` is 0 and the walk stops LOUD with what it has, rather than looping to the
    page cap or silently under-fetching."""
    from pipeline.ingestion.sources import substack as sub

    page = dict(_SUBS_PAGE, nextCursor="always-the-same")

    class _Stuck:
        def __init__(self):
            self.calls = 0

        def page(self, cursor):
            self.calls += 1
            return page

    source = _Stuck()
    assert len(sub.fetch_subscription_list(source)) == 2
    assert source.calls == 2          # page 1, then the repeat that proves the cursor is inert


def test_a_refused_subscription_list_records_a_SKIP_not_a_successful_zero(conn, monkeypatch):
    """The THIRD rail to need this rule. `_stamp_run` reads the `skipped` key to set the clock's
    status, so without it a Cloudflare 403 stores `found=0, status=ok` — indistinguishable from an
    account that subscribes to nothing, and `onboard` reads that as the curation phase being
    done."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    monkeypatch.setattr(sub, "subscription_list_source", lambda profile=None: object())
    monkeypatch.setattr(sub, "fetch_subscription_list", lambda source, **kw: (_ for _ in ()).throw(
        sub.SubstackListingError("subscription listing refused on page 1")))

    out = ic.sync_substack_subscriptions(conn)
    assert out["skipped"] == "refused"
    assert "subscriptions" not in out

    ic._stamp_run(conn, ic.SPEC_BY_COLLECTOR["substack_subscriptions"], out)
    row = curation_state.get_run(conn, "substack_subscriptions")
    assert row.last_status == "refused" and not row.ok and row.found is None


def test_the_collector_lands_the_paid_flag_the_follow_read_never_could(conn, monkeypatch):
    """`is_paid` was `None` on every Substack signal ever written, because the only writer read the
    FOLLOW endpoint and that payload carries no payment field. `membership_state` supplies it, and
    `screen.reflect()` already had the branch — so this is the first code path in OPYT that makes a
    claim about the user's money. The raw state rides along so the inferred `subscribed` -> True
    arm can be audited against a real row without another request."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import screen

    monkeypatch.setattr(sub, "subscription_list_source", lambda profile=None: object())
    monkeypatch.setattr(sub, "fetch_subscription_list",
                        lambda source, **kw: sub.parse_subscription_page(_SUBS_PAGE))

    assert ic.sync_substack_subscriptions(conn)["subscriptions"] == 2

    free = json.loads(_sig(conn, "substack:freepub", "subscribe", "substack")["extra"])
    paid = json.loads(_sig(conn, "substack:letters.example.com", "subscribe",
                           "substack")["extra"])
    assert free["is_paid"] is False and free["membership_state"] == "free_signup"
    assert paid["is_paid"] is True and paid["is_favorite"] is True

    def cand(extra):
        return screen.Candidate(canonical_id="substack:letters.example.com", signals=[
            {"signal_type": "subscribe", "platform": "substack", "count": 1, "extra": extra}])

    assert screen.reflect(cand({"is_paid": True})) == "you subscribe (paid)"
    assert screen.reflect(cand({"is_paid": None})) == "you subscribe"


def test_an_unrecognised_membership_state_never_invents_a_payment(kb_home):
    """Three-way, and the None arm is why. `unsubscribed` is neither a live paid nor a live free
    relationship, and a value Substack adds tomorrow is unknown — both must fall to the phrasing
    that claims nothing. Only `subscribed` may print "(paid)"."""
    assert ic._is_paid("subscribed") is True
    assert ic._is_paid("free_signup") is False
    assert ic._is_paid("unsubscribed") is None
    assert ic._is_paid("") is None
    assert ic._is_paid("some_state_substack_adds_later") is None


# ── the ONE-SHOT rename, and why it must never re-run ────────────────────────────

def _seed_follow_era_store(conn):
    """A store as the pre-2026-09-08 collector left it: follow rows stamped `subscribe`, and the
    clock row under the retired collector key."""
    from pipeline.kb import curation_state

    for eid in ("substack:followed-a", "substack:followed-b"):
        schema.upsert_entity(conn, eid, name=eid)
        schema.set_signal(conn, eid, "subscribe", "substack", extra={"is_paid": None})
    curation_state.record_run(conn, "substack_subs", status="ok", found=2, stored_after=2)


def test_the_migration_renames_the_follow_signals_and_their_clock_row(conn, monkeypatch):
    """`screen.reflect()` told the user "you subscribe" about people they merely followed, because
    `subscriber-lists?lists=following` is the follow graph. The rename is the fix, and it carries
    the clock row with it so the new collector does not inherit a history it never made."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    _seed_follow_era_store(conn)

    class _Empty:
        def follows(self):
            return []

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Empty())
    ic.sync_substack_follows(conn)

    assert _sig(conn, "substack:followed-a", "subscribe", "substack") is None
    assert _sig(conn, "substack:followed-a", "follow", "substack") is not None
    assert curation_state.get_run(conn, "substack_subs") is None
    assert curation_state.get_run(conn, "substack_follows") is not None


def test_the_migration_never_eats_a_real_subscription(conn, monkeypatch):
    """THE reason this is a one-shot and not a connect-hook migration. From the subscription
    collector onward `('substack','subscribe')` is a REAL subscription, so a rename that re-ran
    would silently convert every one of them into a follow, on every connect, forever.

    The marker is the clock row the migration itself creates: once `substack_follows` exists the
    guard never opens again, even though an older build on the primary checkout can still recreate
    a `substack_subs` row."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    _seed_follow_era_store(conn)

    class _Empty:
        def follows(self):
            return []

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Empty())
    ic.sync_substack_follows(conn)                     # migrates

    schema.upsert_entity(conn, "substack:really-subscribed", name="Paid")
    schema.set_signal(conn, "substack:really-subscribed", "subscribe", "substack",
                      extra={"is_paid": True, "membership_state": "subscribed"})
    # An older build re-creates the retired clock row mid-window.
    curation_state.record_run(conn, "substack_subs", status="ok", found=2, stored_after=2)

    ic.sync_substack_follows(conn)                     # must NOT migrate again

    assert _sig(conn, "substack:really-subscribed", "subscribe", "substack") is not None
    assert _sig(conn, "substack:really-subscribed", "follow", "substack") is None


def test_the_migration_is_a_no_op_on_a_store_that_never_ran_the_follow_collector(conn,
                                                                                 monkeypatch):
    """A fresh install has nothing to rename. It must also not leave a half-done migration — no
    clock row is invented, so the collector's own first run creates it."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import curation_state

    class _Empty:
        def follows(self):
            return []

    monkeypatch.setattr(sub, "follow_source", lambda profile=None: _Empty())
    ic._migrate_substack_follow_signals(conn)

    assert curation_state.get_run(conn, "substack_follows") is None
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0


# ── Substack saved → full-body atom + save signal ────────────────────────────────

_REC = {
    "id": 99, "title": "Scaling Laws", "subtitle": "a deep dive",
    "post_date": "2026-05-01T00:00:00Z", "author_handle": "carol", "author_name": "Carol",
    "publication_name": "Carol Writes", "publication_url": "https://carol.substack.com",
    "wordcount": 900, "audience": "everyone", "slug": "scaling-laws",
    "url": "https://carol.substack.com/p/scaling-laws", "preview": "a short preview line",
}
_BODY_HTML = ("<h2>Intro</h2><p>Autonomous agents compose small tools into larger systems, "
              "and the framework matters.</p><p>See <a href='https://arxiv.org/abs/1234'>the paper</a>.</p>")


class _FakeSaved:
    """A `saved_source()` stand-in — the seam the collector actually uses.

    Patched here rather than at `read_substack_cookies`, because that reader is no longer on the
    collector's path: `saved_source()` is the one place the transport is chosen, and a test that
    stubs the layer beneath it would keep passing after the collector stopped going through the
    seam at all.
    """

    def __init__(self, recs, *, full_body=None, authenticated=True):
        self.recs, self.full_body = recs, full_body
        self._authenticated = authenticated

    def authenticated_body(self, base):
        return self._authenticated

    def full_post(self, base, slug):
        return {"body_html": self.full_body} if self.full_body else None


def _patch_substack(monkeypatch, *, full_body=_BODY_HTML, paywalled=False, recs=None,
                    authenticated=True):
    from pipeline.ingestion.sources import substack as sub
    source = _FakeSaved(recs if recs is not None else [dict(_REC)],
                        full_body=full_body, authenticated=authenticated)
    monkeypatch.setattr(sub, "saved_source", lambda profile=None: source)
    monkeypatch.setattr(sub, "fetch_saved_posts",
                        lambda src: sub.SavedPosts([dict(r) for r in src.recs], True))
    monkeypatch.setattr(sub, "_is_paywalled", lambda rec: paywalled)
    return source


# ── the saved-posts SIGNAL collector (2026-09-13) ───────────────────────────────
#
# ⚠️ WHY IT EXISTS, and why it matters more here than the X twin it mirrors. Substack has three
# signal types, one UNREADABLE by construction (liking is a forward write with no reverse index),
# and the other two are near-disjoint graphs — 36 subscriptions against 20 follows overlapping by
# ZERO on the live store, 2026-09-13. So a `save` is the only realistic second signal anyone on
# Substack has, and it reached the store only through the content arm, which runs on the
# `substack_saved_catchup` rail, which only the resident worker launches. 56 Substack entities in
# the live store, not one corroborated.

def _saved_walk(monkeypatch, recs, *, complete=True):
    """Patch the two seams the signal collector uses: the transport picker and the walk."""
    from pipeline.ingestion.sources import substack as sub
    source = _FakeSaved([dict(r) for r in recs])
    monkeypatch.setattr(sub, "saved_source", lambda profile=None: source)
    monkeypatch.setattr(sub, "fetch_saved_posts",
                        lambda src: sub.SavedPosts([dict(r) for r in src.recs], complete))
    return source


def _saved(pid, *, handle="carol", pub="https://carol.substack.com"):
    return {**_REC, "id": pid, "slug": f"p{pid}", "author_handle": handle,
            "publication_url": pub, "url": f"{pub}/p/p{pid}"}


def test_the_saved_walk_counts_saves_per_publication(conn, monkeypatch):
    _saved_walk(monkeypatch, [_saved(1), _saved(2), _saved(3, handle="dave",
                                                          pub="https://dave.substack.com")])

    out = ic.sync_substack_saved_signals(conn)

    assert (out["saved_posts"], out["candidates"]) == (3, 2)
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 2
    assert _sig(conn, "substack:dave", "save", "substack")["count"] == 1
    assert "undetermined" not in out


def test_a_saved_publication_merges_with_the_subscription_that_keys_on_its_url(conn, monkeypatch):
    """⚠️ THE KEYING SPLIT, and the reason this collector stores the publication URL. The saved
    list carries a HANDLE and the two account reads do not, so a publication the user both saves
    from and subscribes to mints TWO ids — and two candidates carrying one signal each are both
    below the corroboration bar, filtered out before a human sees them. Stage-3 closes it on the
    publication URL, which only works if this side writes one."""
    from pipeline.ingestion.sources import substack as sub

    pub = "https://carol.substack.com"
    monkeypatch.setattr(sub, "subscription_list_source", lambda profile=None: object())
    monkeypatch.setattr(sub, "fetch_subscription_list", lambda source, **kw: [
        {"id": 1, "name": "Carol Writes", "url": pub, "membership_state": "free_signup",
         "is_favorite": False}])
    ic.sync_substack_subscriptions(conn)                       # keys on the SUBDOMAIN
    _saved_walk(monkeypatch, [_saved(1, handle="carolwrites")])  # keys on the HANDLE

    ic.sync_substack_saved_signals(conn)
    ic.resolve_after_pull(conn)

    subdomain_id, handle_id = _canonicals(conn, "substack:carol", "substack:carolwrites")
    assert subdomain_id and subdomain_id == handle_id


def test_a_full_saved_walk_REPLACES_the_count_so_unsaving_is_reflected(conn, monkeypatch):
    """`set_signal`, like its peers: a completed walk hands in the whole aggregate, not a delta."""
    _saved_walk(monkeypatch, [_saved(1), _saved(2), _saved(3)])
    ic.sync_substack_saved_signals(conn)
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 3

    _saved_walk(monkeypatch, [_saved(1)])
    ic.sync_substack_saved_signals(conn)

    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 1


def test_a_truncated_saved_walk_keeps_what_it_saw_but_claims_no_total(conn, monkeypatch):
    """⚠️ THE ONE WAY THIS COULD DESTROY INFORMATION, and the reason `fetch_saved_posts` reports
    completeness at all. Three of its exits truncate and return NORMALLY, so a partial aggregate
    written with `set_signal` would REPLACE a correct count with a smaller one. The pages that
    answered are still real, so they land as PRESENCE."""
    _saved_walk(monkeypatch, [_saved(i) for i in range(4)])
    ic.sync_substack_saved_signals(conn)
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 4

    _saved_walk(monkeypatch, [_saved(1), _saved(9, handle="dave",
                                                pub="https://dave.substack.com")],
                complete=False)
    out = ic.sync_substack_saved_signals(conn)

    assert out["undetermined"] == 1              # `classify_run`'s BLOCKED, not a clean zero
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 4   # NOT lowered to 1
    assert _sig(conn, "substack:dave", "save", "substack")["count"] == 1    # new person landed


def test_a_refused_saved_list_reports_a_skip_not_an_empty_walk(conn, monkeypatch):
    """The fourth collector to need this rule. `_stamp_run` derives the clock's status from
    `skipped`, so without it a Cloudflare 403 records a SUCCESSFUL walk that found nobody — and
    `found=0, status=ok` is what `screen` and `onboard` both read as fact."""
    from pipeline.ingestion.sources import substack as sub

    _saved_walk(monkeypatch, [_saved(1)])

    def _refused(src):
        raise sub.SubstackListingError("saved-posts listing refused on page 1")

    monkeypatch.setattr(sub, "fetch_saved_posts", _refused)

    out = ic.sync_substack_saved_signals(conn)

    assert out["skipped"] == "refused"
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0


def test_the_saved_content_arm_never_inflates_the_count_the_walk_owns(
        conn, fake_embedder, monkeypatch):
    """`sync_substack_saved` stamps once per atom against a corpus re-walked forever. It used
    `add_signal`, which was right while it was the only writer and double-counts the moment the
    full-set walk sets a true aggregate. Presence is all it may assert now."""
    recs = [_saved(1), _saved(2)]
    _saved_walk(monkeypatch, recs)
    ic.sync_substack_saved_signals(conn)
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 2

    _patch_substack(monkeypatch, recs=recs)          # the content arm, over the same two posts
    ic.sync_substack_saved(conn, fake_embedder)

    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 2


def test_a_subscription_and_a_save_corroborate_one_publication(conn, monkeypatch):
    """THE ACCEPTANCE SHAPE. `screen.CORROBORATION_MIN` wants 2 distinct (signal_type, platform)
    pairs, and on the live store 0 of 36 Substack entities had them — subscribe and follow
    overlap by zero, and likes cannot be read. This is the pair that clears it."""
    from pipeline.ingestion.sources import substack as sub

    monkeypatch.setattr(sub, "fetch_subscription_list", lambda source, **kw: [
        {"id": 1, "name": "Carol Writes", "url": "https://carol.substack.com",
         "membership_state": "free_signup", "is_favorite": False}])
    monkeypatch.setattr(sub, "subscription_list_source", lambda profile=None: object())
    ic.sync_substack_subscriptions(conn)
    _saved_walk(monkeypatch, [_saved(1, handle="")])   # no handle → the subdomain id, as subs key
    ic.sync_substack_saved_signals(conn)

    kinds = conn.execute("SELECT signal_type FROM curation_signals WHERE entity_id='substack:carol'"
                         ).fetchall()
    assert {r["signal_type"] for r in kinds} == {"subscribe", "save"}


def test_saved_post_builds_full_body_atom(conn, fake_embedder, monkeypatch):
    _patch_substack(monkeypatch)
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 1 and out["stub_fallback"] == 0

    atom = conn.execute("SELECT * FROM atoms WHERE atom_id='substack:99'").fetchone()
    assert atom["source_type"] == "substack" and atom["entry_mode"] == "user-saved"
    assert atom["who_id"] == "substack:carol"
    assert json.loads(atom["payload"]) == {"word_count": 900, "paywalled": False,
                                           "body_state": "complete",
                                           "body_basis": "stated"}

    # Chunks carry the REAL body (not the stub preview), and are clean (no YAML/HTML chrome).
    chunks = conn.execute("SELECT text FROM chunks WHERE atom_id='substack:99'").fetchall()
    body = " ".join(c["text"] for c in chunks)
    assert "Autonomous agents compose small tools" in body
    assert "source: substack" not in body and "<p>" not in body

    # Authorship rides `who_id`, and the save signal lands on the author id.
    who = conn.execute("SELECT who_id FROM atoms WHERE atom_id='substack:99'").fetchone()["who_id"]
    assert who == "substack:carol"
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 1


def test_saved_post_falls_back_to_stub_when_paywalled(conn, fake_embedder, monkeypatch):
    _patch_substack(monkeypatch, full_body=None, paywalled=True)
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 1 and out["stub_fallback"] == 1
    atom = conn.execute("SELECT payload FROM atoms WHERE atom_id='substack:99'").fetchone()
    assert json.loads(atom["payload"])["paywalled"] is True
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='substack:99'").fetchall())
    assert "a short preview line" in body       # the stub preview is the fallback surface


def test_saved_post_idempotent_skips_before_refetch(conn, fake_embedder, monkeypatch):
    source = _patch_substack(monkeypatch)
    ic.sync_substack_saved(conn, fake_embedder)
    # Second run: the full-body fetch must NOT be called again (immutable saved artifact).

    def _boom(*a, **k):
        raise AssertionError("full-body re-fetched for an already-ingested saved post")

    monkeypatch.setattr(source, "full_post", lambda base, slug: _boom(base, slug, {}))
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 0 and out["skipped"] == 1


def test_the_collector_picks_its_transport_through_the_one_seam(conn, fake_embedder, monkeypatch):
    """`saved_source()` is where local-vs-hosted is decided. Reaching for `read_substack_cookies`
    here would put that choice in two places and hand a hosted child the transport the
    2026-09-06 session-token boundary forbids — which is why that reader RAISES there."""
    from pipeline.ingestion.sources import substack as sub

    _patch_substack(monkeypatch)
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: pytest.fail(
        "the collector reached for the cookie reader instead of the transport seam"))

    assert ic.sync_substack_saved(conn, fake_embedder)["added"] == 1


def test_a_paid_post_over_an_UNAUTHENTICATED_transport_is_recorded_as_partial(
        conn, fake_embedder, monkeypatch):
    """A hosted home fetches bodies from the publication's PUBLIC endpoint, so a paid post comes
    back as Substack's teaser. We know that before reading a byte — the audience says paid and
    the transport carries no session — so `complete` there would ship a false claim to any KB
    this store is shared with. `partial` is the state `atoms_tools` documents as a paywall
    teaser."""
    _patch_substack(monkeypatch, paywalled=True, authenticated=False)

    ic.sync_substack_saved(conn, fake_embedder)

    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='substack:99'").fetchone()["payload"])
    assert payload["body_state"] == "partial" and payload["body_basis"] == "stated"
    assert payload["paywalled"] is True


def test_a_paid_post_over_the_users_OWN_session_is_left_alone(conn, fake_embedder, monkeypatch):
    """The local transport carries the user's Substack session, so a paid post they subscribe to
    comes back whole. Whether they subscribe is not knowable from anything the run holds, and
    guessing it from body length is a heuristic nobody has measured — so this path is unchanged.
    `paywalled` still records Substack's audience flag, which is a different fact."""
    _patch_substack(monkeypatch, paywalled=True, authenticated=True)

    ic.sync_substack_saved(conn, fake_embedder)

    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='substack:99'").fetchone()["payload"])
    assert payload["body_state"] == "complete"
    assert payload["paywalled"] is True


def test_a_refused_saved_list_is_a_BLOCKED_run_not_a_quiet_one(conn, fake_embedder, monkeypatch):
    """MEASURED LIVE 2026-09-07: `/api/v1/reader/saved` refused four consecutive attempts, and the
    pass reported `status: ok, added: 0` — indistinguishable from a week with nothing new saved,
    on a rail that therefore never looks again.

    `blocked`, not `error`, because Cloudflare lifts on its own; `error` is for a dead session
    that needs a person. `classify_run` reads `undetermined` for exactly that split."""
    from pipeline.ingestion.sources import substack as sub
    from pipeline.kb import ingest_common

    source = _patch_substack(monkeypatch)

    def _refused(src):
        raise sub.SubstackListingError("saved-posts listing refused on page 1")

    monkeypatch.setattr(sub, "fetch_saved_posts", _refused)

    out = ic.sync_substack_saved(conn, fake_embedder)

    assert out["error"] and out["added"] == 0
    assert ingest_common.classify_run(out) == ingest_common.RUN_BLOCKED


# ── Saved posts keep their images, and the VLM makes them searchable ──────────────

_IMG = "https://substackcdn.com/image/fetch/$s_!abc/https%3A//example.com/chart.png"
_IMG_BODY = (f'<p>Revenue grew sharply.</p><figure><img src="{_IMG}" alt="revenue chart">'
             f'</figure><p>See <a href="https://arxiv.org/abs/1234">the paper</a>.</p>')


def test_saved_post_images_survive_extraction(monkeypatch):
    """`_clean_body_html` ran BOTH extractors with images explicitly disabled, so every chart in a
    saved post was deleted before anything could describe it. Fails if either flag flips back."""
    md = ic._clean_body_html(_IMG_BODY)
    assert md.count("![") == 1 and _IMG in md


def test_saved_post_image_becomes_searchable_chunk_text(conn, fake_embedder, monkeypatch):
    """The description must be inside the HASHED + chunked surface, or it never reaches the index —
    same ordering as both footprint adapters."""
    _patch_substack(monkeypatch, full_body=_IMG_BODY)
    from pipeline import ocr_cascade
    monkeypatch.setattr(ocr_cascade, "read_image",
                        lambda url, context="": ocr_cascade.MediaRead(
                            "a bar chart of quarterly revenue", "chart", True))

    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 1
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id='substack:99'"))
    assert "*Image:* a bar chart of quarterly revenue" in body


# Two tests lived here that pinned the shared reference extractor's filters — an `![alt](url)`
# image never becoming a bogus `references` edge, and Substack TOC anchors / self-links being
# dropped. Both went with `outbound_links` and the `edges` table on 2026-08-23; nothing read
# those edges. The image half that still matters — that a chart's OCR text reaches the body — is
# `test_ocr_text_reaches_the_body` above.


# ── A BLOCKED body writes a RETRYABLE stub (a temporary failure must not become a hole) ──


def _patch_blocking(monkeypatch, *, fail_times):
    """Saved-list + a full-body fetch that is BLOCKED for the first `fail_times` calls, then
    succeeds. Returns the call counter so a test can prove a re-fetch really happened."""
    from pipeline.ingestion.sources import substack as sub
    calls = {"n": 0}
    source = _patch_substack(monkeypatch)

    def _full(base, slug):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise sub.SubstackFetchError("cloudflare challenge")
        return {"body_html": _BODY_HTML}

    monkeypatch.setattr(source, "full_post", _full)
    return calls


def _chunk_text(conn, atom_id="substack:99"):
    return " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id=?", (atom_id,)))


def test_blocked_body_writes_retryable_stub_and_counts_undetermined(conn, fake_embedder,
                                                                    monkeypatch):
    _patch_blocking(monkeypatch, fail_times=99)
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 1                 # the RECORD is kept — a saved post has value w/o body
    assert out["stub_fallback"] == 1
    assert out["undetermined"] == 1          # ... and the block is READABLE, not folded into stubs
    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='substack:99'").fetchone()["payload"])
    assert payload["body_state"] == "pending"    # retryable — we were STOPPED
    assert payload["body_basis"] == "observed"
    assert schema.load_body_pending(conn, "substack") == {"substack:99"}


def test_a_blocked_body_is_not_recorded_as_paywalled(conn, fake_embedder, monkeypatch):
    """⚠️ FIXED 2026-09-04. `paywalled` was `only_paid OR not got_body`, so EVERY block and every
    network error stored `paywalled=True`. The two facts are different and `body_state` already
    carries the second one — and `export` ships `paywalled` into shared KBs, where a reader has no
    way to tell a real paywall from a fetch we lost."""
    _patch_blocking(monkeypatch, fail_times=99)          # `_is_paywalled` stubbed False inside
    ic.sync_substack_saved(conn, fake_embedder)
    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='substack:99'").fetchone()["payload"])
    assert payload["paywalled"] is False          # Substack never said paid-subscribers-only
    assert payload["body_state"] == "pending"     # …and the missing body is still recorded


def test_a_real_paywall_is_still_recorded_as_paywalled(conn, fake_embedder, monkeypatch):
    """The half that must not regress: `audience == 'only_paid'` is what the flag is FOR."""
    _patch_substack(monkeypatch, full_body=None, paywalled=True)
    ic.sync_substack_saved(conn, fake_embedder)
    row = conn.execute("SELECT payload FROM atoms LIMIT 1").fetchone()
    assert json.loads(row["payload"])["paywalled"] is True


def test_blocked_stub_upgrades_in_place_when_the_block_clears(conn, fake_embedder, monkeypatch):
    """The whole point of the retryable stub: a Cloudflare block that cleared must not leave a
    permanent hole in content the user explicitly SAVED."""
    calls = _patch_blocking(monkeypatch, fail_times=1)

    first = ic.sync_substack_saved(conn, fake_embedder)
    assert first["undetermined"] == 1
    assert "Autonomous agents compose small tools" not in _chunk_text(conn)   # stub only

    second = ic.sync_substack_saved(conn, fake_embedder)                     # block has cleared
    assert calls["n"] == 2                                                   # it really re-fetched
    assert second["undetermined"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM atoms").fetchone()["c"] == 1  # UPGRADED, not doubled
    assert "Autonomous agents compose small tools" in _chunk_text(conn)       # real body indexed
    assert "a short preview line" not in _chunk_text(conn)                    # stub chunks replaced
    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='substack:99'").fetchone()["payload"])
    assert payload["body_state"] == "complete"                                # flag self-cleared
    assert schema.load_body_pending(conn, "substack") == set()


def test_bodyless_stub_is_never_retried(conn, fake_embedder, monkeypatch):
    """The gate that keeps the retry cheap. A post with genuinely NO body (podcast, link post)
    will never gain one, so it stays permanent and costs nothing — only a BLOCK earns a retry.
    Fails if `body_state='pending'` is set on every stub instead of only on UNDETERMINED."""
    source = _patch_substack(monkeypatch, full_body=None, paywalled=True)
    first = ic.sync_substack_saved(conn, fake_embedder)
    assert first["stub_fallback"] == 1 and first["undetermined"] == 0
    assert schema.load_body_pending(conn, "substack") == set()

    def _boom(*a, **k):
        raise AssertionError("re-fetched a stub that was EMPTY, not blocked")

    monkeypatch.setattr(source, "full_post", lambda base, slug: _boom(base, slug, {}))
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 0 and out["skipped"] == 1


def test_still_blocked_retry_stays_pending(conn, fake_embedder, monkeypatch):
    """Blocked again on the retry: the stub renders identically so there is nothing to rewrite,
    and the stored flag must survive that no-op or the post silently stops being retried."""
    _patch_blocking(monkeypatch, fail_times=99)
    ic.sync_substack_saved(conn, fake_embedder)
    second = ic.sync_substack_saved(conn, fake_embedder)
    assert second["undetermined"] == 1       # counted again — we were stopped again
    assert second["added"] == 0              # nothing new to write
    assert schema.load_body_pending(conn, "substack") == {"substack:99"}   # STILL retryable


# ── ARC-1 Job A: batch the embed across saved posts; `save` signal rides on_written ──

def test_saved_batches_embed_and_signals_match_added(conn, recording_embedder, monkeypatch):
    # Four saved posts (same author) pool into ONE flush = ONE embed call. The `save` signal rides
    # on_written, so it is written only for an atom that DURABLY landed, never for one that failed
    # to embed. Proves both the batching win and the durable-signal invariant.
    #
    # It asserts PRESENCE, not 4: since 2026-09-13 this arm writes `ensure_signal`, because
    # `sync_substack_saved_signals` — the full-set walk — owns the count, and summing 1 per atom
    # into the row the screen renders as "saved N post(s)" would double it.
    from pipeline.ingestion.sources import substack as sub
    recs = [{**_REC, "id": 100 + i, "slug": f"s{i}",
             "url": f"https://carol.substack.com/p/s{i}"} for i in range(4)]
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "fetch_saved_posts",
                        lambda src: sub.SavedPosts([dict(r) for r in recs], True))
    monkeypatch.setattr(sub, "_fetch_full_post",
                        lambda base, slug, cookies: {"body_html": _BODY_HTML})
    monkeypatch.setattr(sub, "_is_paywalled", lambda rec: False)

    out = ic.sync_substack_saved(conn, recording_embedder)
    assert out["added"] == 4 and out["failed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 4
    assert len(recording_embedder.calls) == 1                      # four saved posts, ONE flush
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 1   # presence, not a count


# ── orchestration: failure isolation ─────────────────────────────────────────────

def test_curation_pull_isolates_one_sources_failure(conn, fake_embedder, monkeypatch):
    # bookmarks + all X signal pulls fail (no session); substack subs/saved fail (no cookie).
    # curation_pull must capture each error stub and still return, never raise.
    import pipeline.kb.ingest_x as ingest_x
    monkeypatch.setattr(ingest_x, "sync_bookmarks",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dead session")))
    for name in ("sync_lists_signals", "sync_following_signals", "sync_likes_signals",
                 "sync_substack_follows", "sync_substack_subscriptions",
                 "sync_bookmark_signals", "sync_substack_saved_signals"):
        monkeypatch.setattr(ic, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ic, "sync_substack_saved", lambda *a, **k: {"source": "substack-saved", "added": 0})

    out = ic.curation_pull(conn, fake_embedder)
    assert "error" in out["x_bookmarks"]
    assert "error" in out["x_lists"] and "error" in out["x_following"]
    assert out["substack_saved"] == {"source": "substack-saved", "added": 0}


# ── 2a: the adapter reports WHERE its time went (it was the last one that didn't) ──

def test_saved_run_reports_stage_timings(conn, fake_embedder, monkeypatch):
    """Curation sits off the live-tested footprint path, so it kept getting skipped by passes that
    measured their way to a decision. Without these keys, "should curation be parallelized too?"
    can only be answered by guessing — which is how ARC-1 Step 3 survived three documents."""
    _patch_substack(monkeypatch, full_body=_IMG_BODY)
    # The autouse `ocr` fake (tests/kb/conftest.py) already covers the image seam.
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 1
    assert {"stage_seconds", "stage_latency", "llm_call_latency", "llm_upstreams"} <= set(out)
    # The three stages this adapter actually spends in, plus the sink's own.
    assert {"list_fetch", "body_fetch", "vlm"} <= set(out["stage_seconds"])
    # Per-ENTRY samples, not just totals: a fat tail is invisible in a mean, and the distribution
    # is what would size a worker pool.
    assert out["stage_latency"]["body_fetch"]["count"] == 1


def test_stage_timings_do_not_count_a_skipped_post(conn, fake_embedder, monkeypatch):
    """Policy B skips an already-ingested post BEFORE the paid fetch, so a re-run must record no
    `body_fetch` entry at all — otherwise the timings would report work that never happened."""
    _patch_substack(monkeypatch)
    ic.sync_substack_saved(conn, fake_embedder)
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["skipped"] == 1 and out["added"] == 0
    assert "body_fetch" not in out["stage_seconds"]
    assert "list_fetch" in out["stage_seconds"]      # the listing still happened


# ── reconcile_saved_signals — the derived `save` arm of the candidate list ───────
#
# The stampers fire once per atom, on the run that ingests it. These prove the reconcile
# repairs what that write-once property can lose, WITHOUT the two ways a naive fix breaks:
# inflating `count` (add_signal SUMS) and inventing nameless candidates (no entities row).

def _saved_atom(conn, *, atom_id, who_id, source_type, entry_mode="user-saved"):
    schema.upsert_atom(conn, {
        "atom_id": atom_id, "source_type": source_type, "what_kind": "opinion",
        "who_id": who_id, "when_ts": "2026-08-01T00:00:00+00:00", "when_precision": "day",
        "about_entities": None, "source_url": None,
        "raw_ref": None, "raw_hash": "h", "description": None, "payload": None,
        "entry_mode": entry_mode, "basis": "observed", "body_state": "complete",
    })


def test_reconcile_stamps_a_save_for_an_unsignalled_bookmark_author(conn):
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    assert _sig(conn, "x:user:7", "save") is None          # the drift this repairs

    out = ic.reconcile_saved_signals(conn)

    assert out["inserted"] == {"x": 1}
    assert _sig(conn, "x:user:7", "save")["count"] == 1
    assert out["signal_bearing_entities"] == 1


def test_reconcile_covers_substack_saves_on_the_same_pass(conn):
    schema.upsert_entity(conn, "substack:acme", name="Acme")
    _saved_atom(conn, atom_id="substack:9", who_id="substack:acme", source_type="substack")

    out = ic.reconcile_saved_signals(conn)

    assert out["inserted"] == {"substack": 1}
    assert _sig(conn, "substack:acme", "save", platform="substack")["count"] == 1


def test_reconcile_is_idempotent_and_never_inflates_count(conn):
    """The whole reason this is insert-if-absent and not `add_signal`: it runs on every read."""
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    _saved_atom(conn, atom_id="x:2", who_id="x:user:7", source_type="x")   # 2 atoms, 1 author

    first = ic.reconcile_saved_signals(conn)
    for _ in range(5):
        again = ic.reconcile_saved_signals(conn)
        assert again["inserted"] == {}                     # nothing left to do

    assert first["inserted"] == {"x": 1}                   # DISTINCT: one row for two atoms
    assert _sig(conn, "x:user:7", "save")["count"] == 1     # never accumulated


def test_reconcile_leaves_a_stamped_signal_and_its_count_alone(conn):
    """A real like/save history must survive the reconcile untouched — it repairs absence only."""
    ic._stamp_x_person(conn, {"user_id": "7", "display_name": "Ada"}, "save", count=4)
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")

    out = ic.reconcile_saved_signals(conn)

    assert out["inserted"] == {}
    assert _sig(conn, "x:user:7", "save")["count"] == 4


def test_reconcile_reports_an_orphan_author_instead_of_inventing_one(conn):
    """No entities row → a signal would make a NAMELESS candidate. Report, never fabricate."""
    _saved_atom(conn, atom_id="x:1", who_id="x:user:404", source_type="x")

    out = ic.reconcile_saved_signals(conn)

    assert out["inserted"] == {}
    assert out["orphans"] == {"x": 1}
    assert _sig(conn, "x:user:404", "save") is None


def test_reconcile_ignores_atoms_the_user_did_not_save(conn):
    """`entry_mode` is the curation act. Oracle footprint is corpus, not a curation signal —
    stamping it would make every account an Oracle ever quoted into a candidate."""
    schema.upsert_entity(conn, "x:user:8", name="Bob")
    _saved_atom(conn, atom_id="x:3", who_id="x:user:8", source_type="x",
                entry_mode="oracle-footprint")

    out = ic.reconcile_saved_signals(conn)

    assert out["inserted"] == {}
    assert _sig(conn, "x:user:8", "save") is None


# ── the LIST clock: curation_pull writes `collector_runs` ───────────────────────
#
# Before this, nothing recorded when any collector last ran. The system could say whether a
# candidate's CONTENT was stale and not whether the candidate LIST was — so someone you followed
# yesterday stayed invisible until a human hand-ran this module. These prove the pull now stamps
# that clock, and that the two timestamps stay separated under every outcome.

def _patch_collector_fetches(monkeypatch):
    """Stub the FETCH layer under all five people-only collectors, leaving the collectors
    themselves REAL. That is what makes `found` vs `stored_after` meaningful here: `found` comes
    out of the collector's own return dict and `stored_after` counts rows it really wrote."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_likes, x_lists
    from pipeline.ingestion.sources import substack as sub

    monkeypatch.setattr(core, "read_x_cookies", lambda: {"twid": "u=1"})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: "1")
    monkeypatch.setattr(core, "auth_headers", lambda cookies, referer: {})
    monkeypatch.setattr(core, "fetch_following", lambda c, h, v: [
        {"user_id": "2", "display_name": "A"}, {"user_id": "3", "display_name": "B"}])

    monkeypatch.setattr(x_lists, "fetch_owned_lists", lambda c, h, v: [{"id": "L1", "name": "AI"}])
    monkeypatch.setattr(x_lists, "fetch_list_members", lambda lid, c, h: [])
    monkeypatch.setattr(x_lists, "aggregate_members", lambda owned, by_list, vid: [
        {"user_id": "2", "display_name": "A", "list_names": ["AI"]}])

    monkeypatch.setattr(x_likes, "fetch_liked_authors", lambda vid, c, h: [{}])
    monkeypatch.setattr(x_likes, "aggregate_authors", lambda authors, vid: [
        {"user_id": "4", "display_name": "C", "liked_count": 3}])

    # The bookmark LIST, which is all the signal collector reads — two saves by one author, so
    # `found` (bookmarks walked) and `stored_after` (people) are deliberately different numbers.
    # The viewer's own bookmark is in here to prove it is dropped.
    from pipeline.ingestion import x_graphql as xg
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0: iter([
        {"id": "10", "author": {"id": "5", "userName": "d", "name": "D"}},
        {"id": "11", "author": {"id": "5", "userName": "d", "name": "D"}},
        {"id": "12", "author": {"id": "1", "userName": "me", "name": "Me"}},
    ]))

    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "own_user_id", lambda cookies: 7)
    monkeypatch.setattr(sub, "fetch_follows", lambda cookies, uid=None: [
        {"name": "Acme", "url": "https://acme.substack.com"}])
    monkeypatch.setattr(sub, "fetch_subscription_list", lambda source, **kw: [
        {"id": 1, "name": "Beta", "url": "https://beta.substack.com",
         "membership_state": "subscribed", "is_favorite": False}])
    # The saved LIST, which is all the Substack signal collector reads — two posts by one
    # publication, so `found` (people) and the walk's own `saved_posts` are different numbers,
    # exactly as they are for the X bookmark walk above.
    monkeypatch.setattr(sub, "saved_source", lambda profile=None: object())
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda src: sub.SavedPosts(
        [{**_REC, "id": 1, "slug": "p1"}, {**_REC, "id": 2, "slug": "p2"}], True))


def _patch_content_arms(monkeypatch):
    """The two CONTENT sources are not on the list clock, so these tests only need them silent."""
    import pipeline.kb.ingest_x as ingest_x
    monkeypatch.setattr(ingest_x, "sync_bookmarks", lambda *a, **k: {"source": "x", "added": 0})
    monkeypatch.setattr(ic, "sync_substack_saved",
                        lambda *a, **k: {"source": "substack-saved", "added": 0})


def test_a_successful_pull_stamps_every_collector(conn, fake_embedder, monkeypatch):
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)

    ic.curation_pull(conn, fake_embedder)

    rows = {r.collector: r for r in cs.list_runs(conn)}
    assert set(rows) == set(ic.COLLECTORS)
    for row in rows.values():
        assert row.last_status == "ok"
        assert row.last_ok_at == row.last_attempt_at        # a success stamps BOTH marks
        assert cs.status_summary(conn, (row.collector,))["collectors"][0]["stale"] is False
    # `found` is what each collector SAID it saw; `stored_after` is what the store now holds.
    assert (rows["x_following"].found, rows["x_following"].stored_after) == (2, 2)
    assert (rows["x_lists"].found, rows["x_lists"].stored_after) == (1, 1)
    assert (rows["x_likes"].found, rows["x_likes"].stored_after) == (1, 1)
    assert (rows["substack_follows"].found, rows["substack_follows"].stored_after) == (1, 1)
    assert (rows["substack_subscriptions"].found,
            rows["substack_subscriptions"].stored_after) == (1, 1)
    # PEOPLE, like every other row here — two bookmarks by one author fold to one candidate, and
    # the viewer's own bookmark (also in the fixture) is dropped before counting. The walk's own
    # item count rides the summary as `bookmarks`, not on this clock, because a collapse in
    # `found` has to mean "fewer people saved", the same thing it means for the other four.
    assert (rows["x_bookmark_signals"].found,
            rows["x_bookmark_signals"].stored_after) == (1, 1)


def test_the_content_arms_get_no_row_on_this_clock(conn, fake_embedder, monkeypatch):
    """The CONTENT arms are deliberately not tracked here — a row for either would invite
    `curation_catchup` to re-run a paid content pipeline unattended.

    `x_bookmark_signals` IS on this clock and is not a counter-example: it walks the same list and
    lands the same `save` signal, but reads no body, calls no model, and writes no atom — which is
    the property this clock selects for. Substack saved has no such sibling yet, so its signal
    still reaches the store only through its content arm."""
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)

    ic.curation_pull(conn, fake_embedder)

    assert {r.collector for r in cs.list_runs(conn)} == set(ic.COLLECTORS)
    assert cs.get_run(conn, "x_bookmarks") is None
    assert cs.get_run(conn, "substack_saved") is None


def test_a_raising_collector_stamps_error_without_advancing_last_ok(conn, fake_embedder,
                                                                   monkeypatch):
    """THE split, exercised end-to-end. If the failure advanced `last_ok_at`, one dead X session
    would report the candidate list as freshly seen for a full staleness window."""
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)
    ic.curation_pull(conn, fake_embedder)
    good = cs.get_run(conn, "x_following")
    assert good.last_status == "ok"

    monkeypatch.setattr(ic, "sync_following_signals",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dead session")))
    out = ic.curation_pull(conn, fake_embedder)

    assert "error" in out["x_following"]                     # the pull still reports it
    row = cs.get_run(conn, "x_following")
    assert row.last_status == "error"
    assert "dead session" in row.last_detail
    assert row.last_ok_at == good.last_ok_at                 # ...and the list did NOT get younger
    assert row.last_attempt_at >= row.last_ok_at             # we DID try
    assert (row.found, row.stored_after) == (2, 2)           # last good reading survives the error


def test_a_collectors_own_skip_reason_is_recorded_not_flattened(conn, fake_embedder, monkeypatch):
    """`{"skipped": "no_viewer_id"}` is not an error and it is not a success. Recording it as `ok`
    would make a dead X session look like an empty follow list — the two need different fixes."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)
    monkeypatch.setattr(core, "viewer_id", lambda cookies: None)

    ic.curation_pull(conn, fake_embedder)

    row = cs.get_run(conn, "x_lists")
    assert row.last_status == "no_viewer_id"
    assert row.last_ok_at is None and row.last_attempt_at is not None
    assert cs.status_summary(conn, ("x_lists",))["collectors"][0]["stale"] is True
    # Substack has its own session, so it is unaffected — failure isolation reaches the clock too.
    assert cs.get_run(conn, "substack_follows").last_status == "ok"


def test_every_spec_names_a_key_its_collector_actually_returns(conn, monkeypatch):
    """The test that pays for itself. The collectors disagree about their own return shape
    (`candidates` / `following` / `follows` / `subscriptions`), so `CollectorSpec` records the name
    rather than normalising five working functions. This turns "the spec drifted from the collector" from a
    silently-NULL `found` column into a red test."""
    _patch_collector_fetches(monkeypatch)
    for spec in ic.COLLECTOR_SPECS:
        res = ic.run_collector(conn, spec)
        assert spec.found_key in res, f"{spec.collector} never returns {spec.found_key!r}"
        assert isinstance(res[spec.found_key], int)


# ── the pull ends in resolution: one person, not two candidates ─────────────────
#
# `screen.rank_candidates` groups on COALESCE(canonical_id, entity_id) and the pre-tick bar is ≥2
# DISTINCT signals. Unresolved, someone you follow on X whose bio links a Substack you subscribe to
# is two rows carrying one signal each — filtered out before a human ever sees them. Nothing errors;
# the candidate simply never appears. That is why the pull, not the SCREEN, has to close this.

def _patch_one_person_on_two_platforms(monkeypatch, *, site="https://acme.substack.com"):
    """The cross-platform merge shape, minimally: ONE human, an X follow and a Substack
    subscription, joined only by the X bio site pointing at the publication home."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_likes, x_lists
    from pipeline.ingestion.sources import substack as sub

    monkeypatch.setattr(core, "read_x_cookies", lambda: {"twid": "u=1"})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: "1")
    monkeypatch.setattr(core, "auth_headers", lambda cookies, referer: {})
    monkeypatch.setattr(core, "fetch_following", lambda c, h, v: [
        {"user_id": "2", "display_name": "Acme Author", "site": site}])
    monkeypatch.setattr(x_lists, "fetch_owned_lists", lambda c, h, v: [])
    monkeypatch.setattr(x_lists, "fetch_list_members", lambda lid, c, h: [])
    monkeypatch.setattr(x_lists, "aggregate_members", lambda owned, by_list, vid: [])
    monkeypatch.setattr(x_likes, "fetch_liked_authors", lambda vid, c, h: [])
    monkeypatch.setattr(x_likes, "aggregate_authors", lambda authors, vid: [])
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "own_user_id", lambda cookies: 7)
    monkeypatch.setattr(sub, "fetch_follows", lambda cookies, uid=None: [
        {"name": "Acme", "url": site}])
    monkeypatch.setattr(sub, "fetch_subscription_list", lambda source, **kw: [
        {"id": 1, "name": "Acme", "url": site, "membership_state": "free_signup",
         "is_favorite": False}])
    # This person saved nothing — but the walk must still be stubbed, or the pull's saved-signal
    # collector goes to the live reader endpoint and spends its 20s of retry backoff.
    monkeypatch.setattr(sub, "saved_source", lambda profile=None: object())
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda src: sub.SavedPosts([], True))


def _canonicals(conn, *entity_ids):
    return [conn.execute("SELECT canonical_id FROM entities WHERE entity_id=?",
                         (eid,)).fetchone()["canonical_id"] for eid in entity_ids]


def test_a_full_pull_ends_resolved_so_the_two_rows_are_one_candidate(conn, fake_embedder,
                                                                     monkeypatch):
    _patch_one_person_on_two_platforms(monkeypatch)
    _patch_content_arms(monkeypatch)

    out = ic.curation_pull(conn, fake_embedder)

    x_canon, sub_canon = _canonicals(conn, "x:user:2", "substack:acme")
    assert x_canon and x_canon == sub_canon, "the pull left the same human as two candidates"
    assert out["resolve"]["duplicate_rows_collapsed"] == 1
    assert out["resolve"]["cross_platform"] == 1


def test_a_resolve_failure_never_sinks_a_pull_that_landed_data(conn, fake_embedder, monkeypatch):
    """Fail-safe, same direction as the broken-clock test below: every signal is committed before
    resolution runs, so a resolve blowing up must degrade to an unmerged store — never lose the
    pull's report."""
    from pipeline.kb import resolve

    _patch_one_person_on_two_platforms(monkeypatch)
    _patch_content_arms(monkeypatch)
    monkeypatch.setattr(resolve, "resolve_entities",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    out = ic.curation_pull(conn, fake_embedder)

    assert "db locked" in out["resolve"]["error"]
    assert _sig(conn, "x:user:2", "follow")["count"] == 1      # the signal landed regardless


def test_a_broken_clock_never_sinks_a_pull_that_landed_data(conn, fake_embedder, monkeypatch):
    """Fail-safe, in the load-bearing direction: the state table is bookkeeping ABOUT the pull, so
    a write failure there must not lose a pull that actually wrote signals."""
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)
    monkeypatch.setattr(cs, "record_run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))

    out = ic.curation_pull(conn, fake_embedder)

    assert out["x_following"] == {"source": "x-following", "following": 2}
    assert _sig(conn, "x:user:2", "follow")["count"] == 1     # the signal landed regardless


def test_a_second_pull_does_not_inflate_the_counts(conn, fake_embedder, monkeypatch):
    """THE regression this rail made urgent. `curation_pull` was hand-run, so summing a full-set
    re-read into itself was a slow leak nobody watched. `curation_catchup` runs it ~4x a day
    forever. Measured live on 2026-08-12: two runs seven seconds apart took `follow/x` from 468 to
    886 and `like/x` max from 15 to 30 — nobody liked 15 tweets in seven seconds."""
    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)

    ic.curation_pull(conn, fake_embedder)
    first = {(r["entity_id"], r["signal_type"]): r["count"]
             for r in conn.execute("SELECT entity_id, signal_type, count FROM curation_signals")}
    for _ in range(3):
        ic.curation_pull(conn, fake_embedder)
    again = {(r["entity_id"], r["signal_type"]): r["count"]
             for r in conn.execute("SELECT entity_id, signal_type, count FROM curation_signals")}

    assert first == again
    assert first[("x:user:4", "like")] == 3          # the aggregate the collector reported, once


def test_a_changed_aggregate_still_lands_on_a_re_pull(conn, fake_embedder, monkeypatch):
    """Idempotent is not frozen. When you really do like more of someone's posts, the new total
    must replace the old one — including downward, when you unlike."""
    from pipeline.ingestion import x_likes

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)
    ic.curation_pull(conn, fake_embedder)
    assert _sig(conn, "x:user:4", "like")["count"] == 3

    monkeypatch.setattr(x_likes, "aggregate_authors", lambda authors, vid: [
        {"user_id": "4", "display_name": "C", "liked_count": 9}])
    ic.curation_pull(conn, fake_embedder)
    assert _sig(conn, "x:user:4", "like")["count"] == 9


def test_the_save_arm_keeps_summing_across_pulls(conn, fake_embedder, monkeypatch):
    """The boundary. `save` is stamped ONCE per atom by the content arms, which is a real event
    stream — converting it to replacement would be the same mistake in the other direction."""
    schema.upsert_entity(conn, "x:user:9", name="Ada")
    schema.add_signal(conn, "x:user:9", "save", "x")
    schema.add_signal(conn, "x:user:9", "save", "x")
    assert _sig(conn, "x:user:9", "save")["count"] == 2


def test_reconciled_author_becomes_a_ranked_candidate(conn):
    """The consumer's view: the reconcile's job is not a row, it is a candidate."""
    from pipeline.kb import screen

    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    assert screen.rank_candidates(conn) == []

    ic.reconcile_saved_signals(conn)

    cands = screen.rank_candidates(conn)
    assert [c.name for c in cands] == ["Ada"]
