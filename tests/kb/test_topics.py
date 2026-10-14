"""Topics: the lists a host writes, the flows that follow, change
and stop a topic, the preview, the host's check, and which of the user's people's posts belong.

The decisions (`lists`) with plain values; the flows on a temporary store; the preview against
fake sources, never the network.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server import build
from pipeline.frontier import candidates, chain
from pipeline.frontier import queries as fq
from pipeline.frontier.candidates import Candidate
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.topics import checked, create, lists, posts, preview, store

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
SOURCES = ["arxiv", "github", "openalex", "x", "substack"]


@pytest.fixture()
def woken(monkeypatch):
    """What the flows woke the scrapers process for, in order."""
    out: list[str] = []
    monkeypatch.setattr(chain, "queries_added", lambda: out.append("queries") or True)
    monkeypatch.setattr(chain, "finds_kept", lambda: out.append("kept") or True)
    return out


@pytest.fixture()
def conn(kb_home, woken):
    c = schema.connect()
    yield c
    c.close()


def _post(conn, atom_id, text, *, who="x:user:1", when="2026-10-10",
          entry_mode="oracle-footprint"):
    raw_hash = snapshot_hash(text)
    store_atom(conn, dict(
        atom_id=atom_id, source_type="x", what_kind="opinion", who_id=who, when_ts=when,
        when_precision="day", about_entities=[], source_url=f"https://x.com/i/{atom_id}",
        raw_hash=raw_hash, description=f"{atom_id} card", payload={},
        entry_mode=entry_mode), text)
    conn.commit()


# ── the lists, as the host sends them ──────────────────────────────────────────────
def test_each_spelling_of_a_watch_is_its_own_query_under_the_watchs_name():
    rows, problems = lists.watch_rows(
        [{"name": "Boltz-2", "spellings": ["Boltz2", "boltz-2 ", "--"], "sources": ["github"]},
         "x402"], SOURCES)
    assert problems == []
    assert [(r["name"], r["text"], r["sources"]) for r in rows] == [
        ("Boltz-2", "Boltz-2", ["github"]), ("Boltz-2", "Boltz2", ["github"]),
        ("x402", "x402", SOURCES)]                          # a bare name searches every source
    assert rows[1]["query_id"] == fq.query_id_for("boltz2")


@pytest.mark.parametrize("watch, says", [
    ({"name": "AI drug discovery clinical trial success rates in practice"}, "describes"),
    ({"name": "x402", "sources": ["twitter"]}, "unknown sources twitter"),
    ({"name": "  "}, "needs a name"),
])
def test_a_watch_that_does_not_name_one_thing_on_known_sources_is_refused(watch, says):
    assert says in " ".join(lists.watch_rows([watch], SOURCES)[1])


def test_words_are_deduplicated_and_each_must_hold_a_word():
    assert lists.phrases(["drug discovery", "drug discovery", "wet lab"]) == (
        ["drug discovery", "wet lab"], [])
    assert lists.phrases(["!!"])[1] == ["'!!' holds no word to match."]


def test_people_resolve_against_the_store_or_are_refused(conn):
    schema.upsert_entity(conn, "x:user:7", name="Soren Larson",
                         profile={"handle": "hypersoren"})
    conn.commit()
    assert lists.persons(conn, ["@hypersoren"]) == (
        [{"name": "Soren Larson", "who_ids": ["x:user:7"]}], [])
    assert "Nobody in the store" in lists.persons(conn, ["@nobody"])[1][0]


# ── the flows ──────────────────────────────────────────────────────────────────────
def test_a_refused_topic_writes_nothing(conn, woken):
    out = create.create(conn, name="", watches=[{"name": "x402", "sources": ["twitter"]}],
                        words=None, people=None)
    assert out["status"] == "refused" and "needs a name" in out["reason"]
    assert conn.execute("SELECT COUNT(*) FROM user_topics").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM frontier_queries").fetchone()[0] == 0
    assert woken == []


def test_a_new_topic_runs_its_watches_and_asks_about_a_daily_update(conn, woken):
    out = create.create(conn, name="Agent payments",
                        watches=[{"name": "x402", "spellings": ["x-402"]}],
                        words=["agent payments"], people=None)
    assert out["status"] == "ok" and out["next"] == create.DAILY
    assert out["watches"] == [{"name": "x402", "spellings": ["x402", "x-402"], "sources": SOURCES}]
    assert (out["words"], out["people"], out["waiting"]) == (["agent payments"], [], 0)
    assert {q["text"] for q in fq.active_queries(conn)} == {"x402", "x-402"}
    assert woken == ["queries"]


def test_an_edit_adds_and_drops_watches_and_replaces_the_words(conn):
    topic_id = create.create(conn, name="Agent payments", watches=["x402", "AP2"],
                             words=["agent payments"], people=None)["topic_id"]
    out = create.edit(conn, topic_id=topic_id, watches=["MPP"], drop_watches=["AP2"],
                      words=["stablecoin payments"])
    assert [w["name"] for w in out["watches"]] == ["x402", "MPP"]
    assert out["words"] == ["stablecoin payments"]
    assert {q["text"] for q in fq.active_queries(conn)} == {"x402", "MPP"}
    only_watches = create.edit(conn, topic_id=topic_id, watches=["AP2"])
    assert [w["name"] for w in only_watches["watches"]] == ["x402", "MPP", "AP2"]
    refused = create.edit(conn, topic_id=topic_id, drop_watches=["nope"])
    assert refused["status"] == "refused" and "nope" in refused["reason"]


def test_a_dropped_topic_stops_its_watches_unless_another_topic_owns_them(conn):
    first = create.create(conn, name="A", watches=["x402"], words=None, people=None,
                          now="2026-10-12T10:00:00+00:00")["topic_id"]
    second = create.create(conn, name="B", watches=["x402", "AP2"], words=None, people=None,
                           now="2026-10-12T11:00:00+00:00")["topic_id"]
    assert create.drop(conn, topic_id=second)["dropped"] == "B"
    assert {q["text"] for q in fq.active_queries(conn)} == {"x402"}
    assert [t["topic_id"] for t in create.listing(conn)["topics"]] == [first]
    assert create.drop(conn, topic_id=second)["status"] == "refused"


# ── the host's check ───────────────────────────────────────────────────────────────
def _find(conn, cid, status="new"):
    conn.execute("INSERT INTO frontier_candidates (candidate_id, source, kind, title, url, status, "
                 "first_seen_at, last_seen_at) VALUES (?, 'arxiv', 'paper', 't', 'u', ?, ?, ?)",
                 (cid, status, cid, cid))
    conn.commit()


def test_the_check_keeps_and_passes_waiting_finds_and_wakes_admission_for_kept_ones(conn, woken):
    for cid in ("arxiv:1", "arxiv:2"):
        _find(conn, cid)
    out = checked.run(conn, keep=["arxiv:1"], passed=["arxiv:2"])
    assert (out["kept"], out["passed"]) == (1, 1) and woken == ["kept"]
    assert {r[0]: r[1] for r in conn.execute(
        "SELECT candidate_id, status FROM frontier_candidates")} == {"arxiv:1": "kept",
                                                                     "arxiv:2": "passed"}
    assert checked.run(conn, keep=[], passed=["arxiv:3"])["passed"] == 0
    assert checked.run(conn, keep=None, passed=None)["status"] == "refused"
    assert woken == ["kept"]


# ── the user's people's posts ──────────────────────────────────────────────────────
def test_a_topic_matches_posts_holding_its_words_or_written_by_its_people(conn):
    _post(conn, "x:1", "AI drug discovery is a bottleneck trade")
    _post(conn, "x:2", "Instacart joined Muse today", who="x:user:9")
    _post(conn, "x:3", "a thread about rollups")
    topic = {"words": ["drug discovery"], "people": [{"name": "C", "who_ids": ["x:user:9"]}]}
    assert posts.matching(conn, topic, {"x:1", "x:2", "x:3"}) == {"x:1", "x:2"}
    assert posts.matching(conn, topic, {"x:3"}) == set()
    assert posts.recent(conn, topic, NOW) == 2


# ── the preview ────────────────────────────────────────────────────────────────────
class _Source:
    def __init__(self, slug, titles=(), error=None):
        self.slug, self.host, self.min_interval_s = slug, slug, 0.0
        self._titles, self._error, self.asked = list(titles), error, []

    def search(self, query, *, since, limit, now, context=()):
        self.asked.append((query, since))
        if self._error:
            raise self._error
        return [Candidate(candidate_id=f"{self.slug}:{i}", source=self.slug, kind="paper",
                          title=t, url=f"https://example/{i}") for i, t in enumerate(self._titles)]


def test_the_preview_counts_what_each_watch_would_have_found_and_writes_nothing(
        conn, woken):
    _post(conn, "x:1", "AI drug discovery is a bottleneck trade")
    registry = {"arxiv": _Source("arxiv", ["Boltz-2 on kinases", "A docking survey"]),
                "openalex": _Source("openalex", error=RuntimeError("down"))}
    before = list(conn.iterdump())
    out = preview.run(conn, name="AI drug discovery", now=NOW, registry=registry,
                      watches=[{"name": "Boltz-2"}], words=["drug discovery"], people=None)
    assert out["watches"] == [{"name": "Boltz-2", "spellings": ["Boltz-2"],
                               "by_source": {"arxiv": 1, "openalex": "failed"},
                               "found_last_month": 1,
                               "examples": [{"title": "Boltz-2 on kinases", "when": None,
                                             "cite": "[Boltz-2 on kinases](https://example/0)",
                                             "snippet": ""}]}]
    assert (out["posts_last_month"], out["posts_per_day"]) == (1, 0.0)
    assert registry["arxiv"].asked[0][1] == datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert list(conn.iterdump()) == before and woken == []


def test_a_preview_past_its_budget_says_what_it_did_not_check(conn, monkeypatch):
    monkeypatch.setattr(preview, "BUDGET_S", -1.0)
    out = preview.run(conn, name="A", now=NOW, registry={"arxiv": _Source("arxiv", ["x402"])},
                      watches=["x402"], words=None, people=None)
    assert out["watches"][0]["by_source"] == {"arxiv": "unchecked"}


# ── the tool ───────────────────────────────────────────────────────────────────────
def test_watch_carries_the_measured_rules_and_is_not_read_only(kb_home):
    [tool] = [t for t in asyncio.run(build.server().list_tools()) if t.name == "watch"]
    assert lists.RULES in tool.description
    declared = tool.annotations.model_dump(by_alias=True)
    assert (declared["readOnlyHint"], declared["destructiveHint"]) == (False, False)
