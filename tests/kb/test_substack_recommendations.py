"""curation.recommended — an Oracle's published endorsements, as screenable candidates.

The network is the only thing stubbed. Everything else is real: real entities, real signals, the
real entity-key derivation, and the real `screen` read that ranks what this writes. The payload
fixture carries field names with synthetic values — no real publication is named in the repo.
"""
from __future__ import annotations

import pytest
from tests.kb.recommendation_support import run_pass

from pipeline.kb import schema
from pipeline.curation import recommended as recs
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.platforms.substack import publication as substack_publication
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import ranking
from pipeline.oracles import screen


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    registry.init_state_schema(c)
    yield c
    c.close()


def _oracle(conn, canonical_id, publication_url, *, name="An Oracle"):
    """A confirmed Oracle with a registered `substack` pair — what the collector reads."""
    eid = publication_url.replace("https://", "substack:")
    schema.upsert_entity(conn, canonical_id, name=name, identity_links=[publication_url])
    roster.upsert_oracle(conn, canonical_id, name=name)
    registry.upsert_source(conn, registry.SourceRow(canonical_id=canonical_id, source_type="substack",
                                        source_key=publication_url, status="trusted"))
    conn.commit()
    return eid


def _rec_item(pub_id, *, subdomain, custom_domain=None, name="A Pub", bio="writes things"):
    """One item in the bare-array response, in the measured shape."""
    return {
        "id": 5000 + pub_id,
        "recommended_publication_id": pub_id,
        "recommending_publication_id": 1,
        "description": None,
        "blurb_active": False,
        "recommendedPublication": {
            "id": pub_id, "name": name, "subdomain": subdomain,
            "custom_domain": custom_domain, "author_id": 900 + pub_id,
            "payments_state": "disabled",
            "author": {"id": 900 + pub_id, "handle": f"h{pub_id}", "name": "An Author",
                       "bio": bio},
        },
    }


def _stub_network(monkeypatch, by_url: dict, *, raises: dict | None = None):
    """Map publication URL -> its recommendation array. `raises` maps a URL to an exception."""
    raises = raises or {}
    monkeypatch.setattr(substack_publication, "fetch_publication_id",
                        lambda url: (_ for _ in ()).throw(raises[url]) if url in raises
                        else abs(hash(url)) % 100000)
    ids = {abs(hash(u)) % 100000: v for u, v in by_url.items()}
    monkeypatch.setattr(substack_publication, "fetch_recommendations",
                        lambda pub_id: substack_publication.parse_recommendations(ids.get(pub_id, [])))


# ── the parser ───────────────────────────────────────────────────────────────────

def test_the_parser_keys_on_the_publication_url_not_the_author_handle(kb_home):
    """The recommendations payload carries an author `handle` that the account-level reads do not,
    and the parser deliberately drops it.

    Keying on the handle would mint `substack:h101` for a publication the user's subscription list
    already keyed as `substack:pub-a` — two candidates carrying one signal each, which is exactly
    the split that keeps a person below the >=2-signal bar. A custom domain keys on its HOST for
    the same reason: that is what the subscription read stores."""
    out = substack_publication.parse_recommendations([
        _rec_item(101, subdomain="pub-a"),
        _rec_item(102, subdomain="pub-b", custom_domain="letters.example.com"),
    ])
    assert [r["url"] for r in out] == ["https://pub-a.substack.com",
                                       "https://letters.example.com"]
    assert all("handle" not in r for r in out)


def test_an_item_with_no_publication_url_is_dropped(kb_home):
    """The URL is the entity key and there is no second way to derive one, so a publication with
    neither a subdomain nor a custom domain is skipped rather than keyed on `substack:unknown`."""
    assert substack_publication.parse_recommendations([
        {"recommendedPublication": {"id": 1, "name": "No Host"}},
        {"recommendedPublication": "not-a-dict"},
        "not-a-dict",
    ]) == []


# ── the signal ───────────────────────────────────────────────────────────────────

def test_two_oracles_recommending_the_same_publication_count_as_two(conn, monkeypatch):
    """`count` is how many Oracles recommend this publication — the corroboration this signal has
    instead of a `min_papers` floor. A publication both Oracles name is stronger evidence than one
    only a single Oracle names, and the count is what carries that to the rank."""
    _oracle(conn, "x:user:1", "https://one.example.com")
    _oracle(conn, "x:user:2", "https://two.example.com")
    shared = _rec_item(300, subdomain="shared")
    _stub_network(monkeypatch, {
        "https://one.example.com": [shared, _rec_item(301, subdomain="only-one")],
        "https://two.example.com": [shared],
    })

    out = run_pass(conn)

    assert out["signalled"] == 2
    counts = dict(conn.execute(
        "SELECT entity_id, count FROM curation_signals WHERE signal_type='recommended'"))
    assert counts == {"substack:shared": 2, "substack:only-one": 1}


def test_the_extra_names_which_oracles_recommended_them(conn, monkeypatch):
    """`count` and `extra.oracles` are two projections of one list. The count ranks; the list is
    the audit trail for a signal the user never created, and they are written in one call so they
    cannot disagree."""
    import json
    _oracle(conn, "x:user:1", "https://one.example.com")
    _stub_network(monkeypatch, {"https://one.example.com": [_rec_item(300, subdomain="shared")]})

    run_pass(conn)

    extra = conn.execute("SELECT extra FROM curation_signals WHERE entity_id='substack:shared'"
                         ).fetchone()[0]
    assert json.loads(extra)["oracles"] == ["x:user:1"]


def test_an_oracle_recommending_another_oracle_creates_no_candidate(conn, monkeypatch):
    """A confirmed Oracle is past the screen — re-proposing them as a candidate is noise, and
    proposing a person to themselves is nonsense. Same exclusion, same shape, as
    `sync_coauthor_signals`."""
    _oracle(conn, "x:user:1", "https://one.example.com")
    # The second Oracle's canonical id IS the entity id this recommendation derives.
    _oracle(conn, "substack:two", "https://two.substack.com")
    _stub_network(monkeypatch, {
        "https://one.example.com": [_rec_item(400, subdomain="two")],
        "https://two.substack.com": [],
    })

    out = run_pass(conn)

    assert out["already_oracles"] == 1
    assert out["signalled"] == 0
    assert conn.execute("SELECT COUNT(*) FROM curation_signals "
                        "WHERE signal_type='recommended'").fetchone()[0] == 0


def test_one_refused_oracle_does_not_cost_the_others_their_signals(conn, monkeypatch):
    """Per-Oracle isolation is what stands in for a fan-out bound. A Cloudflare 403 mid-pass must
    be a delay, not a hole: the Oracles already read are written, the refused one is reported so
    "read nobody" is distinguishable from "nobody recommends anybody", and the next pass re-reads
    everyone because `set_signal` is a full-set write."""
    _oracle(conn, "x:user:1", "https://one.example.com")
    _oracle(conn, "x:user:2", "https://two.example.com")
    _stub_network(monkeypatch,
                  {"https://one.example.com": [_rec_item(300, subdomain="shared")]},
                  raises={"https://two.example.com":
                          substack_contracts.SubstackPublicReadError("HTTP 403")})

    out = run_pass(conn)

    assert (out["oracles_read"], out["skipped_oracles"], out["signalled"]) == (1, 1, 1)


# ── the tier ─────────────────────────────────────────────────────────────────────

def test_a_recommendation_never_ranks_as_an_endorsement(conn, monkeypatch):
    """The user did nothing. `_ENDORSEMENT` is the tier for acts the user performed, so a
    recommendation must not earn the primary rank key however many Oracles carry it — and
    `reflect()` must say whose act it was, so the candidate is never mistaken for a choice the
    user made."""
    _oracle(conn, "x:user:1", "https://one.example.com")
    _stub_network(monkeypatch, {"https://one.example.com": [_rec_item(300, subdomain="shared")]})
    run_pass(conn)

    cand = next(c for c in ranking.rank_candidates(conn)
                if c.canonical_id == "substack:shared")
    assert cand.has_endorsement is False
    assert screen.reflect(cand.signals) == "recommended by one of your Oracles"


def test_the_reflect_phrase_names_the_number_of_oracles(kb_home):
    """Two Oracles is a materially different claim from one, and the phrase says which."""
    cand = ranking.Candidate(canonical_id="substack:shared", signals=[
        {"signal_type": "recommended", "platform": "substack", "count": 3, "extra": {}}])
    assert screen.reflect(cand.signals) == "recommended by 3 of your Oracles"


# ── the rail wiring ──────────────────────────────────────────────────────────────

def test_the_pass_is_skipped_until_its_own_floor_elapses(conn, monkeypatch):
    """The clock is load-bearing, not hygiene: an ungated read would go out on every turn of the
    scraper against the host whose 403 window rate-limits every other Substack read OPYT makes."""
    from pipeline.oracles.sources.web import recommendations as web_recs

    _oracle(conn, "x:user:1", "https://one.example.com")
    _stub_network(monkeypatch, {"https://one.example.com": [_rec_item(300, subdomain="shared")]})

    assert run_pass(conn)["status"] == "ok"
    assert run_pass(conn)["status"] == "not_due"


def test_a_pass_that_could_not_read_every_oracle_does_not_stamp_a_good_walk(conn, monkeypatch):
    """`last_ok_at` is what a later reader takes as "this walk saw the whole list". A pass that
    skipped an Oracle saw a partial one, so it advances the attempt clock (the floor counts
    failures too) without claiming success."""
    from pipeline.curation import clock
    from pipeline.oracles.sources.web import recommendations as web_recs

    _oracle(conn, "x:user:1", "https://one.example.com")
    _stub_network(monkeypatch, {},
                  raises={"https://one.example.com":
                          substack_contracts.SubstackPublicReadError("HTTP 403")})

    assert run_pass(conn)["status"] == "partial"
    row = clock.get_run(conn, recs.COLLECTOR)
    assert row.last_ok_at is None and row.last_attempt_at is not None
