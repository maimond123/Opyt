"""What's new: `latest` and the notice that rides own-store searches.

The decisions (`provenance`, `grouping`) are tested with plain values; the flow on a temporary
store, with the clock handed in.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server import build
from pipeline.arrivals import fresh, grouping, notice, provenance, show, window
from pipeline.frontier import candidates, substack_candidates, x_candidates
from pipeline.frontier import queries as fq
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.timeparse import utc_iso
from pipeline.topics import store

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


# ── provenance: one row of its table each ─────────────────────────────────────────
def _atom(entry_mode, who_id="x:user:1", payload=None):
    return {"entry_mode": entry_mode, "who_id": who_id, "payload": payload or {}}


def test_what_the_user_saved_says_so():
    assert provenance.why(_atom("user-saved"), {}) == "you saved it"


def test_an_oracles_post_names_the_oracle():
    assert provenance.why(_atom("oracle-footprint"), {"x:user:1": "Alex Imas"}) == \
        "your Oracle Alex Imas posted it"


def test_a_linked_atom_names_each_referrer_once_falling_back_to_the_handle():
    refs = [{"by": "x:user:1", "post": "xprofile:9"}, {"by": "x:user:1", "post": "xprofile:10"},
            {"by": "x:user:2", "handle": "deanwball", "post": "xprofile:11"}]
    atom = _atom("author_referenced", who_id="paper:1", payload={"referenced_by": refs})
    assert provenance.why(atom, {"x:user:1": "Alex Imas"}) == \
        "Alex Imas and @deanwball linked it"


def test_a_linked_atom_with_no_referrer_on_record_still_says_an_oracle_linked_it():
    assert provenance.why(_atom("author_referenced"), {}) == "one of your Oracles linked it"


@pytest.mark.parametrize("n, tail", [(4, "and 1 other"), (5, "and 2 others")])
def test_many_referrers_are_counted_past_three(n, tail):
    refs = [{"by": f"x:user:{i}", "handle": f"h{i}"} for i in range(n)]
    assert provenance.why(_atom("author_referenced", payload={"referenced_by": refs}), {}) \
        == f"@h0, @h1, @h2 {tail} linked it"


# ── grouping: plain values ────────────────────────────────────────────────────────
TOPIC_A, TOPIC_B, TOPIC_C = {"topic_id": "a"}, {"topic_id": "b"}, {"topic_id": "c"}


def test_groups_follow_the_topics_then_the_unfiled_and_an_atom_can_sit_in_two():
    new = ["n1", "n2", "n3", "n4"]
    out = grouping.arrange(new, [(TOPIC_A, {"n3", "n1"}), (TOPIC_B, set()), (TOPIC_C, {"n3"})],
                           unfiled=True)
    assert out == [{"topic": TOPIC_A, "atom_ids": ["n1", "n3"]},
                   {"topic": TOPIC_C, "atom_ids": ["n3"]},
                   {"topic": None, "atom_ids": ["n2", "n4"]}]


def test_a_scoped_look_has_no_unfiled_group():
    assert grouping.arrange(["n1", "n2"], [(TOPIC_A, {"n1"})], unfiled=False) == [
        {"topic": TOPIC_A, "atom_ids": ["n1"]}]


def test_the_limit_is_handed_to_the_groups_in_turn():
    """One group took all 50 lines and the 11,829 items no group matched got
    none."""
    grouped = [{"topic": TOPIC_A, "atom_ids": ["n1", "n2", "n5"]},
               {"topic": None, "atom_ids": ["n3", "n4"]}]
    assert [g["listed"] for g in grouping.cut(grouped, 3)] == [["n1", "n2"], ["n3"]]
    assert [g["listed"] for g in grouping.cut(grouped, 1)] == [["n1"], []]
    assert [g["listed"] for g in grouping.cut(grouped, 9)] == [["n1", "n2", "n5"], ["n3", "n4"]]


# ── fresh: new means recently published ───────────────────────────────────────────
def _row(when_ts, arrived="2026-10-12 05:00:00", entry_mode="oracle-footprint", who_id="x:1",
         linked_on=()):
    return {"atom_id": "a", "entry_mode": entry_mode, "who_id": who_id, "when_ts": when_ts,
            "arrived": arrived, "linked_on": list(linked_on)}


@pytest.mark.parametrize("row,new", [
    (_row("2026-10-11"), True),
    (_row("2026-10-05"), True),                      # published a week before it was pulled
    (_row("2026-10-04"), False),
    (_row("2013-04-02"), False),                     # an Oracle's archive
    (_row(None, linked_on=["2026-10-10", "2025-01-01"]), False),   # the earliest link decides
    (_row(None, linked_on=["2026-10-10"]), True),    # an undated page linked this week
    (_row(None), True),                              # no date at all: nothing says it is old
])
def test_an_arrival_is_new_when_it_was_published_within_a_week_of_arriving(row, new):
    assert fresh.is_new(row) is new


def test_a_cluster_is_named_by_its_x_account_else_any_name_else_its_address():
    named = fresh.oracle_names([
        ("blog:stratechery.com", "blog:stratechery.com", "Ben Thompson's blog"),
        ("x:user:40273", "blog:stratechery.com", "Ben Thompson"),
        ("substack:aleximas", "blog:aleximas.com", "Ghosts of Electricity"),
        ("blog:antoinebuteau.com", "blog:antoinebuteau.com", "")])
    assert named == {"blog:stratechery.com": ("blog:stratechery.com", "Ben Thompson"),
                     "x:user:40273": ("blog:stratechery.com", "Ben Thompson"),
                     "substack:aleximas": ("blog:aleximas.com", "Ghosts of Electricity"),
                     "blog:antoinebuteau.com": ("blog:antoinebuteau.com", "antoinebuteau.com")}


def test_older_arrivals_are_counted_one_line_per_oracle_and_per_kind():
    who = {"x:user:40273": ("c1", "Ben Thompson"), "blog:stratechery.com": ("c1", "Ben Thompson")}
    rows = [_row("2013-04-02", who_id="x:user:40273"), _row("2026-01-02", who_id="blog:stratechery.com"),
            _row("2021-06-01", entry_mode="user-saved", who_id="x:user:9"),
            _row(None, entry_mode="author_referenced", who_id="blog:a.com", linked_on=["2024-03-01"])]
    assert fresh.older_lines(rows, who) == ["Ben Thompson: 2 older posts, 2013–2026",
                                            "Posts you saved earlier: 1, 2021",
                                            "Pages your Oracles linked in older posts: 1, 2024"]


# ── the flow, on a temporary store ────────────────────────────────────────────────
def _store(conn, atom_id, text, *, arrived, entry_mode="user-saved", who="x:user:1",
           payload=None, published=None):
    raw_hash = snapshot_hash(text)
    store_atom(conn, dict(
        atom_id=atom_id, source_type="x", what_kind="opinion", who_id=who,
        when_ts=published or arrived[:10],
        when_precision="day", about_entities=[], source_url=f"https://x.com/i/{atom_id}",
        raw_hash=raw_hash, description=f"{atom_id} card", payload=payload or {},
        entry_mode=entry_mode), text)
    conn.execute("UPDATE atoms SET first_seen = ? WHERE atom_id = ?", (arrived, atom_id))
    conn.commit()


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    _store(c, "x:1", "agent payments settle on stablecoins",
           arrived="2026-10-10 09:00:00")
    _store(c, "x:2", "agent payments with cards and x402",
           arrived="2026-10-09 09:00:00", entry_mode="oracle-footprint", who="x:user:2")
    _store(c, "x:3", "a rollup proof benchmark", arrived="2026-10-08 09:00:00")
    _store(c, "x:old", "agent payments last month", arrived="2026-09-20 09:00:00")
    schema.upsert_entity(c, "x:user:2", name="Alex Imas")
    c.commit()
    yield c
    c.close()


def _topic(conn, name, words, *, people=(), now="2026-10-01T00:00:00+00:00"):
    topic_id = store.new_topic(conn, name=name, words=list(words), people=list(people), now=now)
    conn.commit()
    return topic_id


def _watch(conn, topic_id, name, finds):
    """A watch of the topic, and its finds `(candidate_id, title, status)`."""
    qid = fq.query_id_for(fq.normalize(name))
    store.add_watches(conn, topic_id, [{"name": name, "text": name, "sources": ["arxiv"],
                                        "context": [], "query_id": qid}],
                      now="2026-10-01T00:00:00+00:00")
    for cid, title, status in finds:
        conn.execute("INSERT INTO frontier_candidates (candidate_id, source, kind, title, url, "
                     "published, summary, status, first_seen_at, last_seen_at) VALUES "
                     "(?, 'arxiv', 'paper', ?, ?, '2026-10-09', 'an abstract', ?, ?, ?)",
                     (cid, title, f"https://arxiv.org/abs/{cid}", status, cid, cid))
        conn.execute("INSERT INTO frontier_candidate_queries VALUES (?, ?, 'a')", (cid, qid))
    conn.commit()


def _ids(group):
    return [a["atom_id"] for a in group["atoms"]]


def test_a_first_look_covers_the_last_week_newest_first(conn):
    out = show.show(conn, topic_id=None, limit=50, now=NOW)
    assert out["since"] == "2026-10-04T12:00:00+00:00"
    assert out["new"] == 3 and out["topics"] == []
    assert _ids(out["unfiled"]) == ["x:1", "x:2", "x:3"]


def test_posts_are_grouped_by_the_topics_whose_words_or_people_match_then_unfiled(conn):
    payments = _topic(conn, "Agent payments", ["agent payments"], now="2026-10-01T00:00:00+00:00")
    imas = _topic(conn, "Imas", [], people=[{"name": "Alex Imas", "who_ids": ["x:user:2"]}],
                  now="2026-10-02T00:00:00+00:00")
    _topic(conn, "Nothing new", ["zk email"], now="2026-10-03T00:00:00+00:00")
    out = show.show(conn, topic_id=None, limit=50, now=NOW)       # topics oldest first
    assert [(t["topic_id"], _ids(t["posts"])) for t in out["topics"]] == [
        (payments, ["x:1", "x:2"]), (imas, ["x:2"])]
    assert out["topics"][0]["name"] == "Agent payments" and _ids(out["unfiled"]) == ["x:3"]
    assert out["new"] == 3 and out["next"] == show.CHECK


def test_x_and_substack_finds_read_as_who_said_it_and_what_it_says(conn):
    """Their adapters' candidates as the host checks them:
    the author or publication labels the link, as the atom's link is labelled once admitted, and
    the summary is what the post says, where it names the watch."""
    proteins = _topic(conn, "Proteins", [])
    _watch(conn, proteins, "Boltz-2", [])
    qid = fq.query_id_for(fq.normalize("Boltz-2"))
    post = x_candidates.from_post({
        "id": "100", "author": {"name": "Fixture Lab", "userName": "fixturelab"},
        "url": "https://x.com/fixturelab/status/100", "text": "I ran Boltz-2 &amp; it works",
        "createdAt": "Fri Oct 09 13:55:36 +0000 2026", "likeCount": 4})
    essay = substack_candidates.from_post({
        "id": "9", "title": "Folding week", "subtitle": "What changed",
        "description": "What changed", "url": "https://bio.substack.com/p/folding",
        "published": "2026-10-10T09:00:00.000Z", "paid": False, "publication": "Bio Weekly",
        "author": "A. Writer"}, "…we tried Boltz-2 on kinases…")
    candidates.stage(conn, [post, essay], qid, now="2026-10-11T00:00:00+00:00")
    conn.commit()
    [topic] = show.show(conn, topic_id=None, limit=50, now=NOW)["topics"]
    assert [(f["kind"], f["title"], f["cite"], f["summary"]) for f in topic["finds"]] == [
        ("substack", "Bio Weekly", "[Bio Weekly · 2026-10-10](https://bio.substack.com/p/folding)",
         "Folding week — What changed\n…we tried Boltz-2 on kinases…"),
        ("x", "Fixture Lab (@fixturelab)",
         "[Fixture Lab (@fixturelab) · 2026-10-09](https://x.com/fixturelab/status/100)",
         "I ran Boltz-2 & it works")]


def test_a_topics_waiting_finds_are_handed_over_for_the_check(conn):
    """Finds wait outside the library until the host checks them: never marked seen by a look,
    and a checked one is not handed over again."""
    rails = _topic(conn, "Agent payments", [])
    _watch(conn, rails, "x402", [("arxiv:1", "Settling over x402", "new"),
                                 ("arxiv:2", "x402 at scale", "passed"),
                                 ("arxiv:3", "x402 receipts", "kept")])
    out = show.show(conn, topic_id=None, limit=50, now=NOW)
    [topic] = out["topics"]
    assert topic["finds"] == [{"candidate_id": "arxiv:1", "kind": "paper",
                               "title": "Settling over x402", "when": "2026-10-09",
                               "cite": "[Settling over x402 · 2026-10-09](https://arxiv.org/abs/arxiv:1)",
                               "summary": "an abstract", "found_by": "x402"}]
    assert "posts" not in topic and out["next"] == show.CHECK
    again = show.show(conn, topic_id=None, limit=50, now=NOW)
    assert [f["candidate_id"] for f in again["topics"][0]["finds"]] == ["arxiv:1"]


def test_a_busy_topic_hands_over_its_finds_in_turns(conn, monkeypatch):
    monkeypatch.setattr(show, "FINDS", 2)
    rails = _topic(conn, "Agent payments", [])
    _watch(conn, rails, "x402", [(f"arxiv:{i}", f"x402 paper {i}", "new") for i in range(3)])
    [topic] = show.show(conn, topic_id=None, limit=50, now=NOW)["topics"]
    assert len(topic["finds"]) == 2 and topic["finds_more"].startswith("1 more finds wait")


def test_a_kept_find_is_not_an_arrival(conn):
    """The host showed it when it checked it, so its admission is not news a second time."""
    _store(conn, "paper:1", "agent payments protocols",
           arrived="2026-10-10 10:00:00", entry_mode="frontier", who="paper-authors:1")
    out = show.show(conn, topic_id=None, limit=50, now=NOW)
    assert "paper:1" not in _ids(out["unfiled"])


def test_each_atom_carries_its_link_and_why_and_no_offer_to_remove_anything(conn):
    """The host must not report pages as bad to a user, and a removal offer would invite it to."""
    out = show.show(conn, topic_id=None, limit=50, now=NOW)
    card = next(a for a in out["unfiled"]["atoms"] if a["atom_id"] == "x:2")
    assert card == {"atom_id": "x:2", "who": "Alex Imas", "when": "2026-10-09",
                    "cite": "[Alex Imas · 2026-10-09](https://x.com/i/x:2)",
                    "description": "x:2 card", "why": "your Oracle Alex Imas posted it"}
    assert "remove" not in out


def test_only_an_unscoped_look_marks_what_arrived_seen(conn):
    rails = _topic(conn, "Agent payments", ["agent payments"])
    scoped = show.show(conn, topic_id=rails, limit=50, now=NOW)
    assert [t["topic_id"] for t in scoped["topics"]] == [rails] and "unfiled" not in scoped
    assert window.start(conn, NOW) == "2026-10-04T12:00:00+00:00"   # nothing marked

    show.show(conn, topic_id=None, limit=50, now=NOW)
    assert window.start(conn, NOW) == utc_iso(NOW)
    again = show.show(conn, topic_id=None, limit=50, now=NOW)
    assert (again["topics"], again["note"]) == (
        [], f"Nothing new was published since {utc_iso(NOW)}.")

    _store(conn, "x:4", "agent payments later", arrived="2026-10-11 13:00:00")
    later = show.show(conn, topic_id=rails, limit=50, now=NOW)
    assert _ids(later["topics"][0]["posts"]) == ["x:4"]


def test_a_cut_list_says_how_much_it_left_out(conn):
    out = show.show(conn, topic_id=None, limit=2, now=NOW)
    unfiled = out["unfiled"]
    assert (unfiled["new"], len(unfiled["atoms"]), unfiled["more"]) == (3, 2, 1)
    assert "will not be listed again" in out["more"]


def test_an_unknown_topic_is_refused_and_marks_nothing(conn):
    out = show.show(conn, topic_id="t_nope", limit=50, now=NOW)
    assert out["status"] == "error" and "t_nope" in out["message"]
    assert conn.execute("SELECT COUNT(*) FROM arrivals_seen").fetchone()[0] == 0


def test_a_promoted_atom_arrives_when_it_was_promoted_and_an_old_one_is_only_counted(conn):
    """An old post the user saves today is not news to them: it is counted, not listed."""
    conn.execute("UPDATE atoms SET promoted_at = '2026-10-11 08:00:00' WHERE atom_id = 'x:old'")
    conn.commit()
    out = show.show(conn, topic_id=None, limit=50, now=NOW)
    assert "x:old" not in _ids(out["unfiled"])
    assert out["older"] == ["Posts you saved earlier: 1, 2026"]


def test_an_archive_landing_is_counted_and_only_what_was_published_lately_is_listed(conn):
    for i in range(3):
        _store(conn, f"x:arch{i}", "an old essay", arrived="2026-10-10 10:00:00",
               entry_mode="oracle-footprint", who="x:user:2", published=f"201{i}-05-01")
    out = show.show(conn, topic_id=None, limit=50, now=NOW)
    assert _ids(out["unfiled"]) == ["x:1", "x:2", "x:3"]
    assert out["older"] == ["Alex Imas: 3 older posts, 2010–2012"]
    assert notice.notice(conn, NOW) is None                 # the look marked everything seen


# ── the notice ────────────────────────────────────────────────────────────────────
def test_the_latest_notice_counts_what_arrived_until_the_user_looks(conn):
    assert notice.notice(conn, NOW) == {
        "arrived": 3, "finds": {}, "call": "latest()",
        "message": "3 new items arrived in the user's library since they last looked. latest() "
                   "shows them by topic, with why each is here."}
    window.mark_seen(conn, NOW)
    assert notice.notice(conn, NOW) is None


def test_the_notice_names_the_topics_with_finds_until_they_are_checked(conn):
    window.mark_seen(conn, NOW)
    rails = _topic(conn, "Agent payments", [])
    _watch(conn, rails, "x402", [("arxiv:1", "x402", "new"), ("arxiv:2", "x402", "new")])
    said = notice.notice(conn, NOW)
    assert said["finds"] == {"Agent payments": 2}
    assert said["message"] == ("new finds wait to be checked on Agent payments (2). latest() "
                               "shows them by topic, with why each is here.")
    conn.execute("UPDATE frontier_candidates SET status = 'passed'")
    conn.commit()
    assert notice.notice(conn, NOW) is None


# ── the tool ──────────────────────────────────────────────────────────────────────
def test_latest_is_declared_read_only_and_not_destructive(kb_home):
    [tool] = [t for t in asyncio.run(build.server().list_tools()) if t.name == "latest"]
    declared = tool.annotations.model_dump(by_alias=True)
    assert (declared["readOnlyHint"], declared["destructiveHint"]) == (True, False)
    assert set(tool.parameters["properties"]) == {"topic_id", "limit"}
