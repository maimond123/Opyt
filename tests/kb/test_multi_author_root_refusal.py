"""A multi-author site cannot be an Oracle ROOT — the refusal, and the two ways past it.

An Oracle is a PERSON whose judgement the user borrows. `eligibility.gate` already refuses to
attribute a team's writing to anybody, and `expand._route_source` calls it with no `force` — so a
`multi` site SKIPS at every ingest, forever. Confirming one as a root therefore wrote an `oracles`
row that could never produce an atom, and `seed_from_entities` then registered an `oracle_sources`
pair the refresh loop re-walked and re-skipped on every TTL. The zombie is what these tests pin
out of existence.

Two doors reach `upsert_oracle` with a website root — `oracle(action='confirm')` and
`add_oracle(confirm=True)` — and both are covered here, because the venue fix on this same loop
(2026-09-08) exists precisely because one of them had been fixed and the other had not.

Offline: the authorship cache is SEEDED rather than the classifier stubbed, so the tests drive the
real `classify_authorship` cache-hit path and never reach a fetch or an LLM.
"""
from __future__ import annotations

import pytest

from pipeline.kb import eligibility, oracle_refresh_state as st
from pipeline.kb import oracles, schema

_TEAM = "https://www.anthropic.com"
_SOLO = "https://simonwillison.net"


@pytest.fixture()
def conn(kb_home, tmp_path, no_venue):
    # `no_venue`: rooting a site URL asks OpenAlex whether the host is a research venue before
    # falling through to the blog branch. None of these is one. `solo_site` is deliberately NOT
    # taken — these tests seed the authorship cache themselves, which is the path under test.
    c = st.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _seed_verdict(conn, url: str, authorship: str, author_name: str | None = None) -> None:
    """Prime the authorship cache the way a previous classify would have left it."""
    schema.put_authorship(conn, eligibility._site_key(url), authorship, author_name)


# ── the refusal itself ──────────────────────────────────────────────────────────

def test_a_team_site_is_refused_and_writes_no_oracle(conn):
    """The whole point. An `oracles` row for a site the ingest gate skips forever is a row that
    can never produce an atom, so the mint is where it has to be stopped."""
    _seed_verdict(conn, _TEAM, "multi")

    out = oracles.confirm(conn, add_handles=[_TEAM])

    assert out["confirmed"] == []
    assert len(out["refused"]) == 1
    assert out["refused"][0]["handle"] == _TEAM
    assert schema.list_oracles(conn) == []


def test_a_solo_blog_is_still_confirmed(conn):
    """The check must not cost a personal blog its root — that is the primary `onboard('blog')`
    path, and refusing it would take the whole feature down with the fix."""
    _seed_verdict(conn, _SOLO, "single", "Simon Willison")

    out = oracles.confirm(conn, add_handles=[_SOLO])

    assert out["refused"] == []
    assert len(out["confirmed"]) == 1
    assert len(schema.list_oracles(conn)) == 1


def test_an_unclassifiable_site_is_minted_not_refused(conn, monkeypatch):
    """DEGRADES OPEN, the opposite direction to `eligibility.gate`, and the costs are why. There,
    ingesting an unclassifiable site launders a team onto a person. Here, refusing one costs a
    real person their root over a home page that happened not to fetch — and the ingest gate is
    still downstream to catch it if the site really is a team's."""
    monkeypatch.setattr(eligibility, "_fetch_home_text", lambda url: None)

    out = oracles.confirm(conn, add_handles=["https://someones-blog.example"])

    assert out["refused"] == []
    assert len(schema.list_oracles(conn)) == 1


def test_the_refusal_names_what_to_do_instead(conn):
    """A refusal with no alternative reads as "OPYT cannot help you", which is false: the user CAN
    keep individual posts. The reason string is the only place that reaches them, so the two
    escape routes are part of the contract, not phrasing."""
    _seed_verdict(conn, _TEAM, "multi")

    reason = oracles.confirm(conn, add_handles=[_TEAM])["refused"][0]["reason"]

    assert "hopper" in reason
    assert "Oracle" in reason


# ── the account test ────────────────────────────────────────────────────────────

def test_a_person_whose_cluster_head_is_their_employers_site_is_not_refused(conn):
    """The false positive this would otherwise ship, and it is not hypothetical:
    `ingest_x_footprint` writes the X profile's website field into `identity_links`, `resolve`
    merges on those links, and `blog:` sorts below `x:user:` — so an ordinary employee whose X
    profile lists their employer ends up in a cluster headed by the employer's site. Judging the
    HEAD would refuse the person; judging "is there an account here" does not."""
    _seed_verdict(conn, _TEAM, "multi")
    schema.upsert_entity(conn, "blog:www.anthropic.com", name="Anthropic",
                         identity_links=[_TEAM])
    schema.upsert_entity(conn, "x:user:12345", name="An Employee",
                         identity_links=[_TEAM], profile={"handle": "employee"})
    conn.commit()
    from pipeline.kb import resolve
    resolve.resolve_entities(conn)
    cid = schema.get_entity(conn, "x:user:12345")["canonical_id"]

    out = oracles.confirm(conn, canonical_ids=[cid])

    assert out["refused"] == []
    assert len(out["confirmed"]) == 1


# ── the screened door ───────────────────────────────────────────────────────────

def test_a_screened_pick_gets_the_same_check(conn):
    """`add_handles` is not the only way in. The Substack collectors mint `substack:` entities for
    whatever the user follows, and a multi-author publication is a perfectly ordinary thing to
    follow — so the ranked list can offer one."""
    pub = "https://every.substack.com"
    _seed_verdict(conn, pub, "multi")
    schema.upsert_entity(conn, "substack:every.substack.com", name="Every",
                         identity_links=[pub])
    conn.commit()

    out = oracles.confirm(conn, canonical_ids=["substack:every.substack.com"])

    assert out["confirmed"] == []
    assert len(out["refused"]) == 1
    assert schema.list_oracles(conn) == []


# ── no zombie pair ──────────────────────────────────────────────────────────────

def test_a_refused_root_leaves_the_refresh_loop_nothing_to_walk(conn):
    """The cost the refusal actually removes. `seed_from_entities` registers a pair for every
    confirmed Oracle, and `oracle_refresh` re-walks every registered pair — so a minted-then-
    skipped root is not one wasted call, it is one every 336 hours forever."""
    _seed_verdict(conn, _TEAM, "multi")
    oracles.confirm(conn, add_handles=[_TEAM])

    st.seed_from_entities(conn)

    assert st.list_sources(conn) == []


# ── the other door: `add_oracle(confirm=True)` ──────────────────────────────────

def test_add_oracle_refuses_the_same_site(conn):
    """The second door onto `upsert_oracle`. Fixing one and not the other is the exact gap the
    venue two-door fix closed on this same loop, so both are pinned together."""
    _seed_verdict(conn, _TEAM, "multi")

    out = oracles.add_oracle(conn, None, _TEAM, confirm=True)

    assert out.get("refused") == [_TEAM]
    assert schema.list_oracles(conn) == []


def test_a_second_attempt_does_not_walk_past_the_check(conn):
    """The hole a check on the `else` branch alone would leave. The first refused attempt leaves
    behind the `blog:{host}` entity `_resolve_handle` already minted, so the next call matches the
    local roster and takes the OTHER branch — which is why the check sits below both."""
    _seed_verdict(conn, _TEAM, "multi")
    oracles.add_oracle(conn, None, _TEAM, confirm=True)

    assert oracles._match_local_roster(conn, _TEAM) is not None      # the branch flipped
    out = oracles.add_oracle(conn, None, _TEAM, confirm=True)

    assert out.get("refused") == [_TEAM]
    assert schema.list_oracles(conn) == []
