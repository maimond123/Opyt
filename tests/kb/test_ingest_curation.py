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
    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {"twid": "u=1"})
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
    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: None)
    assert ic.sync_following_signals(conn) == {"source": "x-following", "skipped": "no_viewer_id"}


# ── one X person, three signals, ONE rest_id (the join invariant, at the DB level) ──

def test_multi_signal_person_unifies_on_one_rest_id(conn):
    cand = {"user_id": "777", "display_name": "P", "site": "https://p.com"}
    ic._stamp_x_person(conn, cand, "like", count=2)
    ic._stamp_x_person(conn, cand, "follow")
    ic._stamp_x_person(conn, cand, "list", count=1, extra={"list_names": ["AI"]})
    rows = conn.execute("SELECT signal_type FROM curation_signals WHERE entity_id='x:user:777'").fetchall()
    assert {r["signal_type"] for r in rows} == {"like", "follow", "list"}
    assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id='x:user:777'").fetchone()[0] == 1


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


def _patch_substack(monkeypatch, *, full_body=_BODY_HTML, paywalled=False):
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda cookies: [dict(_REC)])
    monkeypatch.setattr(sub, "_fetch_full_post",
                        lambda base, slug, cookies: {"body_html": full_body} if full_body else None)
    monkeypatch.setattr(sub, "_is_paywalled", lambda rec: paywalled)


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
    _patch_substack(monkeypatch)
    ic.sync_substack_saved(conn, fake_embedder)
    # Second run: the full-body fetch must NOT be called again (immutable saved artifact).
    from pipeline.ingestion.sources import substack as sub

    def _boom(*a, **k):
        raise AssertionError("full-body re-fetched for an already-ingested saved post")

    monkeypatch.setattr(sub, "_fetch_full_post", _boom)
    out = ic.sync_substack_saved(conn, fake_embedder)
    assert out["added"] == 0 and out["skipped"] == 1


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
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda cookies: [dict(_REC)])
    monkeypatch.setattr(sub, "_is_paywalled", lambda rec: False)

    def _full(base, slug, cookies):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise sub.SubstackFetchError("cloudflare challenge")
        return {"body_html": _BODY_HTML}

    monkeypatch.setattr(sub, "_fetch_full_post", _full)
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
    _patch_substack(monkeypatch, full_body=None, paywalled=True)
    first = ic.sync_substack_saved(conn, fake_embedder)
    assert first["stub_fallback"] == 1 and first["undetermined"] == 0
    assert schema.load_body_pending(conn, "substack") == set()

    from pipeline.ingestion.sources import substack as sub

    def _boom(*a, **k):
        raise AssertionError("re-fetched a stub that was EMPTY, not blocked")

    monkeypatch.setattr(sub, "_fetch_full_post", _boom)
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
    # Four saved posts (same author) pool into ONE flush = ONE embed call. The `save` signal moved
    # into on_written, so it fires exactly once per DURABLY-written atom — count == added, never for
    # an atom that failed to embed. Proves both the batching win and the durable-signal invariant.
    from pipeline.ingestion.sources import substack as sub
    recs = [{**_REC, "id": 100 + i, "slug": f"s{i}",
             "url": f"https://carol.substack.com/p/s{i}"} for i in range(4)]
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda cookies: [dict(r) for r in recs])
    monkeypatch.setattr(sub, "_fetch_full_post",
                        lambda base, slug, cookies: {"body_html": _BODY_HTML})
    monkeypatch.setattr(sub, "_is_paywalled", lambda rec: False)

    out = ic.sync_substack_saved(conn, recording_embedder)
    assert out["added"] == 4 and out["failed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 4
    assert len(recording_embedder.calls) == 1                      # four saved posts, ONE flush
    assert _sig(conn, "substack:carol", "save", "substack")["count"] == 4   # one save per durable atom


# ── orchestration: failure isolation ─────────────────────────────────────────────

def test_curation_pull_isolates_one_sources_failure(conn, fake_embedder, monkeypatch):
    # bookmarks + all X signal pulls fail (no session); substack subs/saved fail (no cookie).
    # curation_pull must capture each error stub and still return, never raise.
    import pipeline.kb.ingest_x as ingest_x
    monkeypatch.setattr(ingest_x, "sync_bookmarks",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dead session")))
    for name in ("sync_lists_signals", "sync_following_signals", "sync_likes_signals",
                 "sync_substack_subs"):
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
    """Stub the FETCH layer under all four people-only collectors, leaving the collectors
    themselves REAL. That is what makes `found` vs `stored_after` meaningful here: `found` comes
    out of the collector's own return dict and `stored_after` counts rows it really wrote."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_likes, x_lists
    from pipeline.ingestion.sources import substack as sub

    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {"twid": "u=1"})
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

    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "own_user_id", lambda cookies: 7)
    monkeypatch.setattr(sub, "fetch_subscriptions", lambda cookies, uid: [
        {"name": "Acme", "url": "https://acme.substack.com"}])


def _patch_content_arms(monkeypatch):
    """The two CONTENT sources are not on the list clock, so these tests only need them silent."""
    import pipeline.kb.ingest_x as ingest_x
    monkeypatch.setattr(ingest_x, "sync_bookmarks", lambda *a, **k: {"source": "x", "added": 0})
    monkeypatch.setattr(ic, "sync_substack_saved",
                        lambda *a, **k: {"source": "substack-saved", "added": 0})


def test_a_successful_pull_stamps_all_four_collectors(conn, fake_embedder, monkeypatch):
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)

    ic.curation_pull(conn, fake_embedder)

    rows = {r.collector: r for r in cs.list_runs(conn)}
    assert set(rows) == set(ic.COLLECTORS)
    for row in rows.values():
        assert row.last_status == "ok"
        assert row.last_ok_at == row.last_attempt_at        # a success stamps BOTH marks
        assert cs.is_stale(row) is False
    # `found` is what each collector SAID it saw; `stored_after` is what the store now holds.
    assert (rows["x_following"].found, rows["x_following"].stored_after) == (2, 2)
    assert (rows["x_lists"].found, rows["x_lists"].stored_after) == (1, 1)
    assert (rows["x_likes"].found, rows["x_likes"].stored_after) == (1, 1)
    assert (rows["substack_subs"].found, rows["substack_subs"].stored_after) == (1, 1)


def test_the_content_arms_get_no_row_on_this_clock(conn, fake_embedder, monkeypatch):
    """Bookmarks and Substack saved are deliberately NOT tracked here — bookmarks have their own
    rail and their signal is re-derivable, and saved needs a signals-only mode first. A row for
    either would invite `curation_catchup` to re-run a paid content pipeline unattended."""
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
    assert cs.is_stale(row) is True
    # Substack has its own session, so it is unaffected — failure isolation reaches the clock too.
    assert cs.get_run(conn, "substack_subs").last_status == "ok"


def test_a_tiered_stop_records_the_collectors_it_skipped(conn, fake_embedder, monkeypatch):
    """ONLY THE ORCHESTRATOR KNOWS. A skipped collector never runs, so it cannot record its own
    skip — and with no row it is indistinguishable from one that has never existed."""
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)

    out = ic.curation_pull(conn, fake_embedder, tiered=True, sufficient_at=1)

    assert out["tiered_stopped_after"] == "tier1"
    assert "x_following" not in out and "x_likes" not in out      # they really did not run
    for name in ("x_following", "x_likes"):
        row = cs.get_run(conn, name)
        assert row is not None, f"{name} is indistinguishable from never-ran"
        assert row.last_status == "skipped_tier"
        assert row.last_ok_at is None                             # the list keeps ageing
        assert "tier1" in row.last_detail


def test_a_tier2_stop_records_only_likes_as_skipped(conn, fake_embedder, monkeypatch):
    from pipeline.kb import curation_state as cs

    _patch_collector_fetches(monkeypatch)
    _patch_content_arms(monkeypatch)
    # Tier 1 yields 2 entities, Tier 2 adds a third → the ladder clears only after following.
    out = ic.curation_pull(conn, fake_embedder, tiered=True, sufficient_at=3)

    assert out["tiered_stopped_after"] == "tier2"
    assert cs.get_run(conn, "x_following").last_status == "ok"
    assert cs.get_run(conn, "x_likes").last_status == "skipped_tier"


def test_every_spec_names_a_key_its_collector_actually_returns(conn, monkeypatch):
    """The test that pays for itself. The four collectors disagree about their own return shape
    (`candidates` / `following` / `subscriptions`), so `CollectorSpec` records the name rather than
    normalising four working functions. This turns "the spec drifted from the collector" from a
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

    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {"twid": "u=1"})
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
    monkeypatch.setattr(sub, "fetch_subscriptions", lambda cookies, uid: [
        {"name": "Acme", "url": site}])


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


def test_a_tiered_stop_still_resolves(conn, fake_embedder, monkeypatch):
    """The tiered ladder returns from `_done` early, and a thin store is exactly where an unmerged
    duplicate is most likely to cost someone their pre-tick. All three exits go through `_done`, so
    proving the earliest one covers the placement."""
    _patch_one_person_on_two_platforms(monkeypatch)
    _patch_content_arms(monkeypatch)

    out = ic.curation_pull(conn, fake_embedder, tiered=True, sufficient_at=1)

    assert out["tiered_stopped_after"] == "tier1"
    assert "x_following" not in out                      # Tier 2 really did not run
    assert out["resolve"]["total_entities"] >= 1


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
