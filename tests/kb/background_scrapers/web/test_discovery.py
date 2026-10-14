"""Discovery (`sources/web/discovery.py`): its two stages per Oracle, root and links, and what
`register` records from their verdicts."""
from __future__ import annotations

from datetime import timedelta

import pytest

from pipeline.kb import schema
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop import retries
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import asks
from pipeline.oracles.found import list_found
from pipeline.oracles import windows
from pipeline.oracles import review_items
from pipeline.oracles.sources.web import discovery

from .._support import NOW, days

CID = "substack:soren"


def _v(verdict, url, stype="blog", key=None, reasons=("in 3 of their posts",)):
    return {"key": key or url.split("//", 1)[-1], "url": url, "source_type": stype,
            "verdict": verdict, "reasons": list(reasons), "post_count": 3}


ROOT = {"display_name": "Soren", "root_id": "soren.substack.com",
        "verdicts": [_v("sure", "https://x.com/Hyper_Soren", "x", "x.com/hyper_soren")]}
LINKS = {"display_name": "Soren", "root_id": "soren.substack.com", "verdicts": [
    _v("sure", "https://soren.blog"),
    _v("sure", "https://github.com/soren", "github"),
    _v("also", "https://executor.example"),
    _v("merge", "https://old.example"),
    _v("drop", "https://someone-else.example"),
    _v("sure", "https://held.example"),
]}


@pytest.fixture()
def oracle(conn):
    schema.upsert_entity(conn, CID, name="Soren", identity_links=["https://soren.substack.com"])
    roster.upsert_oracle(conn, CID, name="Soren")
    asks.record_ask(conn, CID, x_from=days(90), web_from=windows.ALL_TIME, now=NOW)
    return CID


@pytest.fixture()
def found(monkeypatch):
    """Both stages answering with `ROOT` / `LINKS`, and X connected."""
    seen = []

    def root(seed, *, seed_type, **_kw):
        seen.append(("root", seed, seed_type))
        return dict(ROOT)

    def links(seed, *, seed_type, **_kw):
        seen.append(("links", seed, seed_type))
        return dict(LINKS)
    monkeypatch.setattr("pipeline.oracles.discovery.profile.discover_root", root)
    monkeypatch.setattr("pipeline.oracles.discovery.profile.discover_profile", links)
    monkeypatch.setattr("pipeline.platforms.connections._connected_x", lambda: True)
    return seen


def _hold(conn, url, status="dismissed"):
    conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                 "source_key, status) VALUES (?, 'blog', ?, ?, ?)",
                 (CID, url, review_items.source_key(url), status))
    conn.commit()


def _x_reached(conn):
    """The X pair has its window: newest page in, history back past the ask."""
    conn.execute("UPDATE oracle_sources SET last_pulled_at=?, covered_from=? WHERE source_type='x'",
                 (NOW.isoformat(), days(400).isoformat()))
    conn.commit()


def _items(conn):
    return sorted((r["source_url"], r["kind"], r["status"])
                  for r in conn.execute("SELECT * FROM oracle_review_items"))


# ── the stages ────────────────────────────────────────────────────────────────────

def test_the_root_stage_is_owed_at_the_ask_and_registers_their_x(conn, web_loop, oracle, found):
    d = discovery.Discovery()
    unit = d.next_unit(conn, NOW)
    assert unit.key == f"{CID}|discovery|root"
    assert web_loop.run_unit(conn, d, unit).result == "found"

    (x,) = [r for r in registry.list_sources(conn) if r.source_type == "x"]
    assert (x.canonical_id, x.source_key) == (CID, "hyper_soren")      # case folded
    assert _items(conn) == [("https://x.com/Hyper_Soren", "source", "added")]
    ask = asks.ask_for(conn, CID)
    assert ask.root_probed_at and ask.discovered_at is None
    assert list_found(conn) == []


def test_the_links_stage_waits_for_x_to_reach_its_window(conn, web_loop, oracle, found):
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert d.next_unit(conn, NOW) is None                  # X has not pulled its window yet

    _x_reached(conn)
    unit = d.next_unit(conn, NOW)
    assert unit.key == f"{CID}|discovery|links"
    assert web_loop.run_unit(conn, d, unit).result == "found"
    assert found[-1] == ("links", "soren", "substack")
    ask = asks.ask_for(conn, CID)
    assert ask.discovered_at and not ask.rediscover
    assert d.next_unit(conn, NOW) is None


def test_a_pair_that_gave_up_counts_as_finished(conn, web_loop, oracle, found):
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    conn.execute("UPDATE oracle_sources SET failures=? WHERE source_type='x'",
                 (retries.GIVE_UP_AFTER,))
    conn.commit()
    assert d.next_unit(conn, NOW).key == f"{CID}|discovery|links"


def test_the_links_stage_records_every_verdict_and_owes_only_the_sure_ones(conn, web_loop,
                                                                           oracle, found):
    _hold(conn, "https://held.example")
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))

    assert _items(conn) == [
        ("https://executor.example", "also", "added"),
        ("https://github.com/soren", "source", "added"),
        ("https://held.example", "source", "dismissed"),      # held: never written again
        ("https://old.example", "source", "added"),            # a merge: recorded, not pulled
        ("https://soren.blog", "source", "added"),
        ("https://x.com/Hyper_Soren", "source", "added"),
    ]
    assert [(f.source_type, f.url, f.pulled_at) for f in list_found(conn)] == [
        ("blog", "https://soren.blog", None), ("github", "https://github.com/soren", None)]
    assert list_found(conn, CID)[0].source()["metadata"]["reasons"] == ["in 3 of their posts"]


def test_a_forced_rediscovery_runs_both_stages_and_leaves_recorded_sources_alone(
        conn, web_loop, oracle, found):
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    conn.execute("UPDATE oracle_review_items SET status='dismissed' "
                 "WHERE source_url='https://soren.blog'")
    conn.commit()

    asks.record_ask(conn, CID, rediscover=True, now=NOW)
    assert d.next_unit(conn, NOW).key == f"{CID}|discovery|root"
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert d.next_unit(conn, NOW).key == f"{CID}|discovery|links"
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert ("https://soren.blog", "source", "dismissed") in _items(conn)
    assert not asks.ask_for(conn, CID).rediscover


def test_an_oracle_forgotten_while_discovery_fetched_gets_nothing_registered(conn, oracle,
                                                                             found):
    got = discovery.find_sources(conn, {"canonical_id": CID, "members": [{"entity_id": CID}]},
                                 discovery.LINKS)
    conn.execute("DELETE FROM oracles")
    conn.commit()
    assert discovery.register(conn, CID, got, x_connected=True, now=NOW) == "forgotten"
    assert list_found(conn) == [] and registry.list_sources(conn) == []
    assert _items(conn) == []
    assert asks.ask_for(conn, CID).discovered_at is None


def test_a_root_that_does_not_resolve_is_retried_not_stamped(conn, web_loop, oracle, monkeypatch,
                                                            no_jitter):
    """An X probe that hit a spent window comes back empty. Stamping it would lose the Oracle's
    sources for good."""
    monkeypatch.setattr("pipeline.oracles.discovery.profile.discover_root",
                        lambda *a, **k: {"display_name": "", "verdicts": []})
    d = discovery.Discovery()
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "blip"
    ask = asks.ask_for(conn, CID)
    assert ask.root_probed_at is None and ask.discovery_failures == 1
    assert "did not resolve" in ask.discovery_error
    retry = parse_ts(ask.discovery_retry_at)
    assert retry == NOW + timedelta(minutes=1) or retry > NOW
    assert d.next_unit(conn, NOW) is None and d.next_due(conn, NOW) == retry

    conn.execute("UPDATE oracle_asks SET discovery_failures=?, discovery_retry_at=NULL",
                 (retries.GIVE_UP_AFTER,))
    conn.commit()
    assert d.next_unit(conn, NOW + timedelta(days=30)) is None
    asks.record_ask(conn, CID, now=NOW)                    # the user asks again
    assert d.next_unit(conn, NOW) is not None


# ── plan: what register writes (pure) ─────────────────────────────────────────────

def test_plan_waits_on_x_until_it_is_connected_and_on_an_orcid_until_it_resolves():
    verdicts = [_v("sure", "https://x.com/Carol", "x"),
                _v("sure", "https://orcid.org/0000-0002-1825-0097", "orcid", "orcid.org/0000"),
                _v("sure", "https://orcid.org/0000-0001-5109-3700", "orcid", "orcid.org/0001")]
    todo = discovery.plan(verdicts, set(), x_connected=False,
                          openalex_ids={"orcid.org/0001": "A42"})
    assert todo["x_handles"] == [] and todo["openalex"] == ["A42"]
    assert [r["url"] for r in todo["rows"]] == ["https://orcid.org/0000-0001-5109-3700"]
    assert todo["first_pulls"] == []

    todo = discovery.plan(verdicts[:1], set(), x_connected=True, openalex_ids={})
    assert todo["x_handles"] == ["carol"]


def test_plan_never_writes_a_held_key_or_a_drop():
    verdicts = [_v("sure", "https://a.example"), _v("drop", "https://b.example")]
    held = {("blog", review_items.source_key("https://a.example"))}
    todo = discovery.plan(verdicts, held, x_connected=True, openalex_ids={})
    assert todo == {"rows": [], "x_handles": [], "openalex": [], "first_pulls": []}


# ── stage_owed (pure) ─────────────────────────────────────────────────────────────

def _ask(**kw):
    base = dict(canonical_id=CID, x_from=None, web_from=None, scholar_from=None,
                scholar_topics=None, asked_at=NOW.isoformat(), position=1, discovered_at=None,
                rediscover=False, root_probed_at=None)
    return asks.Ask(**{**base, **kw})


@pytest.mark.parametrize("ask, rows, owed", [
    (_ask(), [], "root"),
    (_ask(root_probed_at="t"), [], "links"),                                 # nothing to wait for
    (_ask(root_probed_at="t", discovered_at="t"), [], None),
    (_ask(root_probed_at="t", discovered_at="t", rediscover=True), [], "links"),
    (_ask(root_probed_at="t"), [registry.SourceRow(CID, "x", "soren")], None),  # X owes history
])
def test_stage_owed(ask, rows, owed):
    assert discovery.stage_owed(ask, rows, NOW) == owed


# ── what an added source takes over (step 5) ──────────────────────────────────────

def _atom(conn, atom_id, who_id, entry_mode, source_type="blog"):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, entry_mode, raw_hash) "
                 "VALUES (?, ?, ?, ?, 'h')", (atom_id, source_type, who_id, entry_mode))
    conn.commit()


def _mode(conn, atom_id):
    return conn.execute("SELECT entry_mode FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()[0]


def _head(conn, eid):
    return schema.current_canonical(conn, eid)


def test_a_sure_source_takes_over_what_was_stored_from_it_and_joins_the_oracle(conn, web_loop,
                                                                               oracle, found):
    schema.upsert_entity(conn, "blog:soren.blog", identity_links=["https://soren.blog"])
    _atom(conn, "https://soren.blog/p/one", "blog:soren.blog", "author_referenced")
    _atom(conn, "https://soren.blog/p/two", "blog:soren.blog", "user-saved")
    _atom(conn, "https://someone-else.example/p", "blog:someone-else.example",
          "author_referenced")
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))

    assert _mode(conn, "https://soren.blog/p/one") == "oracle-footprint"
    assert _mode(conn, "https://soren.blog/p/two") == "user-saved"         # the user's stays theirs
    assert _mode(conn, "https://someone-else.example/p") == "author_referenced"   # dropped
    assert _head(conn, "blog:soren.blog") == _head(conn, CID)


@pytest.mark.parametrize("entity_first", [True, False])
def test_a_sure_github_account_joins_the_oracle(conn, web_loop, oracle, found, entity_first):
    """The links stage is sure of github.com/soren and owes it a first pull. The account joins
    the Oracle once its entity exists: a repo saved before, or its first pull's first repo."""
    from pipeline.kb import resolve
    if entity_first:
        schema.upsert_entity(conn, "github:soren", name="soren")
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert ("github", "https://github.com/soren") in {(f.source_type, f.url)
                                                      for f in list_found(conn)}
    if not entity_first:
        schema.upsert_entity(conn, "github:soren", name="soren")    # the first pull's first repo
        resolve.resolve_entities(conn)                              # the first pull's finish
    assert _head(conn, "github:soren") == _head(conn, CID)


def test_a_dismissed_source_leaves_the_oracle_again(conn, web_loop, oracle, found):
    from pipeline.kb import resolve
    schema.upsert_entity(conn, "blog:soren.blog", identity_links=["https://soren.blog"])
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert _head(conn, "blog:soren.blog") == _head(conn, CID)

    conn.execute("UPDATE oracle_review_items SET status='dismissed' "
                 "WHERE source_url='https://soren.blog'")
    conn.commit()
    resolve.resolve_entities(conn)
    assert _head(conn, "blog:soren.blog") != _head(conn, CID)


def test_a_declared_orcid_becomes_their_papers_and_joins_the_oracle(conn, web_loop, oracle,
                                                                     found, monkeypatch):
    orcid = "https://orcid.org/0000-0002-4027-364X"
    LINKS_WITH_ORCID = {**LINKS, "verdicts": [
        _v("sure", orcid, "orcid", "orcid.org/0000-0002-4027-364X", reasons=("their ORCID",))]}
    monkeypatch.setattr("pipeline.oracles.discovery.profile.discover_profile",
                        lambda *a, **k: dict(LINKS_WITH_ORCID))
    monkeypatch.setattr("pipeline.platforms.papers.openalex.OpenAlexClient.author_by_orcid",
                        lambda self, o: {"id": "https://openalex.org/A5043841592"})
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))

    assert ("openalex", "A5043841592") in {(r.source_type, r.source_key)
                                           for r in registry.list_sources(conn)}
    assert _head(conn, "openalex:A5043841592") == _head(conn, CID)


def _first_x_page(conn, loop, monkeypatch, post: str) -> None:
    """The X walk's first page on the pair the root stage registered: what the page writer
    stores (the account with X's spelling and a bio that links only their site, and one post),
    then the walk's own record step, which folds the account into its Oracle."""
    from pipeline.kb.ingest_common import store_atom
    from pipeline.kb.ingest_common import snapshot_hash
    from pipeline.oracles.sources.timeline import sources

    from .._support import page

    def first_page(conn_, kit, **kw):
        schema.upsert_entity(conn_, "x:user:99", name="Soren", identity_links=["https://soren.dev"],
                             profile={"handle": "Hyper_Soren"})
        text = f'---\nauthor: "@Hyper_Soren"\n---\n\n{post}\n'
        raw_hash = snapshot_hash(text)
        store_atom(conn_, dict(
            atom_id="x:500", source_type="x", what_kind="opinion", who_id="x:user:99",
            when_ts="2026-10-01", when_precision="day", about_entities=[],
            source_url="https://x.com/i/status/500", raw_hash=raw_hash,
            description="x:500", payload={}, entry_mode="oracle-footprint"), text)
        return page("500", days(5))
    monkeypatch.setattr("pipeline.oracles.sources.timeline.pages.walk_page", first_page)
    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))


def test_a_declared_x_joins_the_oracle_and_the_links_stage_reads_its_posts(
        conn, web_loop, loop, ctx, oracle, found, monkeypatch):
    """A real case: a Substack-rooted Oracle whose profile declares
    their X, and an X bio that links only their site, never the Substack. The root profile's
    word is what makes the account theirs, so once the X walk's first page writes it, it is a
    member of the Oracle, and the links stage starts from it and reads its posts."""
    from pipeline.oracles.discovery import post_links

    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))           # root: their X becomes a pair
    _first_x_page(conn, loop, monkeypatch, "new essay https://soren.dev/p/one")
    assert _head(conn, "x:user:99") == _head(conn, CID)

    read = []

    def links(seed, *, seed_type, conn=None, oracle_id=None, **_kw):
        read.append((seed, seed_type, [c.key for c in
                                       post_links.own_post_candidates(conn, oracle_id)]))
        return dict(LINKS)
    monkeypatch.setattr("pipeline.oracles.discovery.profile.discover_profile", links)
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert read == [("Hyper_Soren", "x", ["soren.dev"])]


def test_the_oracles_own_substack_linked_from_its_x_posts_is_not_found_again(
        conn, web_loop, loop, ctx, oracle, monkeypatch):
    """Once their X joins, the links stage roots at it, and their posts link the Substack the
    Oracle was confirmed under. That Substack is already theirs: it gets no row, no first pull,
    and the archives line is not said again. The real `discover_profile`, network stubbed."""
    from pipeline.oracles import told
    from pipeline.oracles.discovery.backlink import Facts

    probed = []
    monkeypatch.setattr("pipeline.oracles.discovery.profile.discover_root",
                        lambda *a, **k: dict(ROOT))
    monkeypatch.setattr("pipeline.platforms.connections._connected_x", lambda: True)
    monkeypatch.setattr("pipeline.oracles.discovery.probes.probe_twitter_bio",
                        lambda handle: probed.append(handle) or ({"display_name": "Soren"}, []))
    monkeypatch.setattr("pipeline.oracles.discovery.backlink.substack_profile",
                        lambda url: (_substack_profile(), url))
    monkeypatch.setattr("pipeline.oracles.discovery.backlink.facts_for",
                        lambda key, url: Facts(key, reached=True, has_feed=True,
                                               links_to=frozenset({"x.com/hyper_soren"})))
    told.init_schema(conn)
    conn.execute("INSERT INTO milestones_shown VALUES (?, 'archives', ?)", (CID, NOW.isoformat()))
    conn.commit()

    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))           # root: their X becomes a pair
    _first_x_page(conn, loop, monkeypatch, "new essay https://soren.substack.com/p/one")
    _x_reached(conn)
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"

    assert probed == ["Hyper_Soren"]                                # the links stage rooted at X
    assert _items(conn) == [("https://x.com/Hyper_Soren", "source", "added")]
    assert list_found(conn) == []
    assert conn.execute("SELECT 1 FROM milestones_shown WHERE milestone='archives'").fetchone()


# ── the links stage reads every profile page of theirs ─────────────────

def _substack_profile(*links):
    """Soren's Substack author profile: their X, connected, and the sites `links` lists."""
    return {"name": "Soren", "handle": "soren", "primaryPublication": {"subdomain": "soren"},
            "twitterAccount": {"screen_name": "Hyper_Soren", "is_connected_account": True},
            "userLinks": [{"url": u} for u in links]}


GITHUB_USER = {"login": "soren", "name": "Soren", "blog": "soren-tools.example",
               "twitter_username": "Hyper_Soren", "public_repos": 3}


@pytest.fixture()
def pages(monkeypatch):
    """Every profile read stubbed at its network call and recorded in `reads`: the X bio
    declares `x_bio`, the Substack profile and the GitHub user answer `substack` / `github` (an
    exception is raised), and a candidate's back-link read finds a site that publishes and links
    nowhere. X is connected."""
    from types import SimpleNamespace
    from pipeline.oracles.discovery import shaping
    from pipeline.oracles.discovery.backlink import Facts

    p = SimpleNamespace(x_bio=[], substack=_substack_profile(), github=GITHUB_USER, reads=[])

    def x_bio(handle):
        p.reads.append(("x", handle))
        return ({"display_name": "Soren", "bio": "", "website": ""},
                [shaping.DiscoveredSource("blog", u) for u in p.x_bio])

    def substack(url):
        p.reads.append(("substack", url))
        return (p.substack, url) if p.substack else (None, None)

    def github(owner):
        p.reads.append(("github", owner))
        if isinstance(p.github, Exception):
            raise p.github
        return p.github

    def facts(key, url):
        p.reads.append(("backlink", key))
        return Facts(key, reached=True, has_feed=True)
    monkeypatch.setattr("pipeline.oracles.discovery.probes.probe_twitter_bio", x_bio)
    monkeypatch.setattr("pipeline.oracles.discovery.backlink.substack_profile", substack)
    monkeypatch.setattr("pipeline.platforms.github.api.fetch_user", github)
    monkeypatch.setattr("pipeline.oracles.discovery.backlink.facts_for", facts)
    monkeypatch.setattr("pipeline.platforms.connections._connected_x", lambda: True)
    return p


def _reasons(conn, url):
    import json
    row = conn.execute("SELECT reasons FROM oracle_review_items WHERE source_url=?",
                       (url,)).fetchone()
    return json.loads(row[0]) if row else None


def _github_member(conn):
    """Their GitHub account, added for them by an earlier pass, is a member of the Oracle."""
    from pipeline.kb import resolve
    url = "https://github.com/soren"
    schema.upsert_entity(conn, "github:soren", name="soren")
    conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                 "source_key, status) VALUES (?, 'github', ?, ?, 'added')",
                 (CID, url, review_items.source_key(url)))
    conn.commit()
    resolve.resolve_entities(conn)
    assert _head(conn, "github:soren") == _head(conn, CID)


def _to_links_stage(conn, web_loop, loop, monkeypatch, pages, post="no links"):
    """The root stage on their Substack (its profile declares their X, which becomes a pair),
    then the X walk's first page, which makes the account a member. Clears `pages.reads`."""
    d = discovery.Discovery()
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"
    _first_x_page(conn, loop, monkeypatch, post)
    assert _head(conn, "x:user:99") == _head(conn, CID)
    _x_reached(conn)
    pages.reads.clear()
    return d


def test_a_site_only_their_substack_profile_declares_is_theirs(
        conn, web_loop, loop, ctx, oracle, pages, monkeypatch):
    """Alex Imas's shape: confirmed under their Substack, their X joined, so the links stage
    roots at X. Neither their X bio nor their posts name a site their Substack profile lists. It
    is declared on a profile of theirs, so it gets the verdict a root-declared site gets, with
    its own reason."""
    d = _to_links_stage(conn, web_loop, loop, monkeypatch, pages)
    pages.substack = _substack_profile("https://soren-essays.example")
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"

    assert pages.reads[:2] == [("x", "Hyper_Soren"),                     # the root stays X
                               ("substack", "https://soren.substack.com")]
    assert ("https://soren-essays.example", "source", "added") in _items(conn)
    assert _reasons(conn, "https://soren-essays.example") == ["on their Substack profile"]
    assert ("blog", "https://soren-essays.example") in {(f.source_type, f.url)
                                                        for f in list_found(conn)}
    assert asks.ask_for(conn, CID).discovered_at


def test_a_site_their_github_profile_declares_is_theirs(
        conn, web_loop, loop, ctx, oracle, pages, monkeypatch):
    """The GitHub account's website field names a site nothing else of theirs does. Its
    `twitter_username` is their X, already theirs: no second X."""
    d = _to_links_stage(conn, web_loop, loop, monkeypatch, pages)
    _github_member(conn)
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"

    assert ("github", "soren") in pages.reads
    assert ("https://soren-tools.example", "source", "added") in _items(conn)
    assert _reasons(conn, "https://soren-tools.example") == ["on their GitHub profile"]
    assert [r.source_key.lower() for r in registry.list_sources(conn)
            if r.source_type == "x"] == ["hyper_soren"]                    # one X, theirs


def test_a_blog_members_home_page_declares_nothing(
        conn, web_loop, loop, ctx, oracle, pages, monkeypatch):
    """Their site is a member (their X bio links it), and its home page links a friend's GitHub
    and their employer. A home page is not a profile: it is never read for declarations, so
    neither is added as theirs."""
    fetched = []

    class _Home:
        status_code, url = 200, "https://soren.dev"
        text = ('<title>Soren</title><a href="https://github.com/soren-friend">a friend</a>'
                '<a href="https://employer.example">work</a>')

    monkeypatch.setattr("pipeline.oracles.discovery.probes.host_request",
                        lambda _get, url, **kw: fetched.append(url) or _Home())
    schema.upsert_entity(conn, "blog:soren.dev", identity_links=["https://soren.dev"])
    d = _to_links_stage(conn, web_loop, loop, monkeypatch, pages)
    assert _head(conn, "blog:soren.dev") == _head(conn, CID)
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"

    assert fetched == []
    assert {u for u, _k, _s in _items(conn)} == {"https://x.com/Hyper_Soren"}


def test_a_profile_that_could_not_be_read_leaves_the_links_stage_to_retry(
        conn, web_loop, loop, ctx, oracle, pages, monkeypatch, no_jitter):
    """GitHub refuses with its rate limit. Stamping the stage would lose what that profile
    declares for good, so it waits as a blip, before any candidate is read, and the next try
    after the limit lifts records it."""
    from pipeline.platforms.github.api import GitHubRateLimited
    d = _to_links_stage(conn, web_loop, loop, monkeypatch, pages)
    _github_member(conn)
    pages.github = GitHubRateLimited("GitHub rate limit reached")
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "blip"

    ask = asks.ask_for(conn, CID)
    assert ask.discovered_at is None and ask.discovery_failures == 1
    assert "github:soren" in ask.discovery_error
    assert not [r for r in pages.reads if r[0] == "backlink"]
    assert {u for u, _k, _s in _items(conn)} == {"https://x.com/Hyper_Soren",
                                                 "https://github.com/soren"}

    pages.github = GITHUB_USER
    retry = parse_ts(ask.discovery_retry_at)
    assert web_loop.run_unit(conn, d, d.next_unit(conn, retry)).result == "found"
    assert asks.ask_for(conn, CID).discovered_at
    assert ("https://soren-tools.example", "source", "added") in _items(conn)


def test_an_oracle_with_only_an_x_account_reads_only_its_x(conn, web_loop, pages):
    """Nothing changes for an X-only Oracle: one profile, its bio, read at each stage."""
    cid = "x:user:7"
    schema.upsert_entity(conn, cid, name="Solo", profile={"handle": "solo"})
    roster.upsert_oracle(conn, cid, name="Solo")
    asks.record_ask(conn, cid, x_from=days(90), web_from=windows.ALL_TIME, now=NOW)
    pages.x_bio = ["https://solo.example"]
    d = discovery.Discovery()
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"
    _x_reached(conn)
    assert web_loop.run_unit(conn, d, d.next_unit(conn, NOW)).result == "found"

    assert [r for r in pages.reads if r[0] != "backlink"] == [("x", "solo"), ("x", "solo")]
    assert _items(conn) == [("https://solo.example", "source", "added")]
    assert _reasons(conn, "https://solo.example") == ["in their X bio"]


def test_new_sources_after_archives_were_told_make_archives_said_again(conn, web_loop, oracle,
                                                                       found):
    from pipeline.oracles import told
    told.init_schema(conn)
    conn.execute("INSERT INTO milestones_shown VALUES (?, 'archives', ?)", (CID, NOW.isoformat()))
    conn.commit()
    d = discovery.Discovery()
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    _x_reached(conn)
    web_loop.run_unit(conn, d, d.next_unit(conn, NOW))
    assert conn.execute("SELECT 1 FROM milestones_shown WHERE milestone='archives'").fetchone() \
        is None


def test_a_taken_over_post_is_present_to_its_first_pull(conn):
    """The walk keys a Substack post as `substack:{id}`, the same id the references worker
    wrote, so it finds the post present and writes nothing new."""
    from pipeline.oracles import coverage
    from pipeline.oracles.sources.archive.substack import Substack
    _atom(conn, "substack:42", "substack:soren", "author_referenced", source_type="substack")
    before = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    assert coverage.take_over(conn, "substack", "https://soren.substack.com") == 1
    assert Substack().document(conn, {"url": "https://soren.substack.com"}, {"id": 42}, None,
                               None) == "present"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == before
    assert _mode(conn, "substack:42") == "oracle-footprint"
