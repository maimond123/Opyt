"""Archive progress survives turns/restarts; coverage never gets ahead of content."""
import sqlite3
from datetime import timedelta

import pytest

from pipeline.kb import schema
from pipeline.timeparse import utc_now
from pipeline.oracles import registry
from pipeline.oracles import asks
from pipeline.oracles import found
from pipeline.oracles import windows
from pipeline.oracles.sources.archive.adapter import Page
from pipeline.oracles.sources.archive import run
from pipeline.oracles.sources.archive import state as walks
from pipeline.oracles.sources.web.archives import Archives
from pipeline.oracles.sources.web.archives import WebNewPosts
from pipeline.oracles.sources.web.discovery import LINKS, Discovery
from pipeline.oracles.sources import loops
from pipeline.oracles.sources import pair_rows
from pipeline.worker.quota_loop import retries

from .test_archives import CID, _pair, _found, oracle

URL = "https://a.substack.com"


@pytest.fixture()
def archive(conn, oracle):
    _pair(conn, "substack", URL)
    loop = loops.web_loop()
    source = next(s for s in loop.sources if isinstance(s, Archives))
    assert loop.run_unit(conn, source, source.next_unit(conn, utc_now())).result == "started"
    return loop, source, run.ADAPTERS["substack"]


def step(conn, archive):
    loop, source, _ = archive
    unit = source.next_unit(conn, utc_now())
    assert unit is not None
    return loop.run_unit(conn, source, unit)


def writer(conn, spec, data, ctx, hold):
    aid = f"substack:{data['id']}"
    hold({"atom_id": aid, "source_type": "substack", "what_kind": "opinion",
                 "who_id": spec["entity"], "entry_mode": "oracle-footprint",
                 "raw_hash": aid, "payload": {}}, "A substantive saved document about this subject.")
    return aid


def test_page_and_documents_resume_and_only_then_stamp(conn, archive, monkeypatch):
    calls = []
    adapter = archive[2]

    def page(spec, cursor, ctx):
        calls.append(cursor["offset"])
        return Page([("1", {"id": 1}), ("2", {"id": 2})], [{"offset": 2}]) if cursor["offset"] == 0 else Page()

    monkeypatch.setattr(adapter, "page", page)
    monkeypatch.setattr(adapter, "document", writer)
    assert step(conn, archive).result == "listed"
    assert registry.list_sources(conn)[0].last_pulled_at is None
    assert step(conn, archive).result == "saved"
    # Reconstruct the source and executor: only durable units are consulted.
    loop = loops.web_loop()
    archive = (loop, next(s for s in loop.sources if isinstance(s, Archives)), adapter)
    assert step(conn, archive).result == "saved"
    assert step(conn, archive).result == "listed"
    assert registry.list_sources(conn)[0].last_pulled_at is None
    assert step(conn, archive).result == "ingested"
    assert registry.list_sources(conn)[0].last_pulled_at
    assert calls == [0, 2]
    assert archive[1].next_unit(conn, utc_now()) is None


def test_failed_later_listing_keeps_cursor_and_coverage_owed(conn, archive, monkeypatch):
    adapter = archive[2]
    monkeypatch.setattr(adapter, "page", lambda s, c, ctx: Page(cursors=[{"offset": 1}]))
    step(conn, archive)
    seen = []

    def fail(s, cursor, ctx):
        seen.append(cursor)
        raise TimeoutError("later page unavailable")

    monkeypatch.setattr(adapter, "page", fail)
    assert step(conn, archive).result == "blip"
    assert registry.list_sources(conn)[0].last_pulled_at is None
    assert archive[1].next_unit(conn, utc_now()) is None
    asks.record_ask(conn, CID, web_from=windows.ALL_TIME)
    monkeypatch.setattr(adapter, "page", lambda s, c, ctx: seen.append(c) or Page())
    assert step(conn, archive).result == "listed"
    assert seen == [{"offset": 1}, {"offset": 1}]


def test_a_document_that_did_not_land_stays_pending_without_repeating_listing(conn, archive,
                                                                              monkeypatch):
    """An adapter that names an atom but held none to write: no atom is present after the write,
    so the task is unsaved and charged, the listing is not read again, and no coverage is
    claimed."""
    monkeypatch.setattr(archive[2], "page", lambda *a: Page([("9", {"id": 9})]))
    monkeypatch.setattr(archive[2], "document", lambda conn, spec, data, ctx, hold: "substack:9")
    step(conn, archive)
    assert step(conn, archive).result == "unsaved"
    row = conn.execute("SELECT * FROM archive_units WHERE done=0").fetchone()
    assert row["kind"] == "document" and row["failures"] == 1
    assert schema.count_atoms(conn, "substack") == 0
    assert registry.list_sources(conn)[0].last_pulled_at is None


def test_forget_during_document_prevents_publication_and_completion(conn, archive, monkeypatch):
    monkeypatch.setattr(archive[2], "page", lambda *a: Page([("9", {"id": 9})]))
    step(conn, archive)

    def forgotten(conn, *args):
        conn.execute("DELETE FROM oracles")
        conn.commit()
        return writer(conn, *args)

    monkeypatch.setattr(archive[2], "document", forgotten)
    assert step(conn, archive).result in ("forgotten", "changed")
    assert schema.count_atoms(conn, "substack") == 0
    assert not conn.execute("SELECT 1 FROM archive_walks WHERE completed=1").fetchone()


def test_a_document_and_its_task_completion_commit_together(conn, archive, monkeypatch):
    """A document's atom says the task is done as surely as its `done` flag does: a completion
    that fails takes the atom back, and the task stays owed."""
    monkeypatch.setattr(archive[2], "page", lambda *a: Page([("9", {"id": 9})]))
    monkeypatch.setattr(archive[2], "document", writer)
    step(conn, archive)

    def refused(*_a, **_kw):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(walks, "record_document", refused)
    step(conn, archive)

    assert schema.count_atoms(conn, "substack") == 0
    assert conn.execute("SELECT kind FROM archive_units WHERE done=0").fetchone()["kind"] == "document"


def test_first_pull_does_not_stamp_at_start(conn, oracle):
    _found(conn, URL)
    loop = loops.web_loop()
    source = Archives()
    assert loop.run_unit(conn, source, source.next_unit(conn, utc_now())).result == "started"
    assert found.list_found(conn)[0].pulled_at is None
    assert conn.execute("SELECT 1 FROM archive_walks WHERE completed=0").fetchone()


def test_progress_reports_unsaved_archive_units_even_when_parent_has_no_failure(conn, archive, monkeypatch):
    from pipeline.oracles import milestones
    monkeypatch.setattr(archive[2], "page", lambda *a: Page([("1", {"id": 1})]))
    step(conn, archive)
    conn.execute("UPDATE archive_units SET status='unsaved',failures=4,last_error='archive document did not save' WHERE done=0")
    conn.commit()
    monkeypatch.setattr(milestones.liveness, "repair_or_report_cached", lambda: {"ok": True})
    pick = milestones.for_picks(conn)[0]
    assert pick["archives"] is False
    assert pick["stuck"] == {"cause": "unsaved", "detail": "archive document did not save"}


def test_first_github_archive_ignores_existing_atom_cursor(conn, oracle, monkeypatch):
    _pair(conn, "github", "alice", cursor_ts="2026-09-01T00:00:00Z")
    source = Archives()
    loop = loops.web_loop()
    assert loop.run_unit(conn, source, source.next_unit(conn, utc_now())).result == "started"
    walk = walks.active(conn, "pair|x:user:1|github|alice")
    assert walk.spec["since"] is None


def test_linked_repository_keeps_referenced_provenance_without_reading_owner_profile(conn, oracle, monkeypatch):
    _found(conn, "https://github.com/alice/repo", source_type="github")
    def no_profile(*args):
        pytest.fail("a reference to one repo must not crawl its owner's profile")
    monkeypatch.setattr("pipeline.platforms.github.api.fetch_user", no_profile)
    loop = loops.web_loop()
    source = Archives()
    assert loop.run_unit(conn, source, source.next_unit(conn, utc_now())).result == "started"
    walk = walks.active(conn, "found|x:user:1|github|https://github.com/alice/repo")
    assert walk.spec["entry_mode"] == "author_referenced"


def test_reserved_github_path_is_skipped_before_any_profile_or_archive_request(conn, oracle, monkeypatch):
    _found(conn, "https://github.com/sponsors/alice", source_type="github")
    monkeypatch.setattr("pipeline.platforms.github.api.fetch_user",
                        lambda *a: pytest.fail("reserved route spent a request"))
    loop = loops.web_loop()
    source = Archives()
    assert loop.run_unit(conn, source, source.next_unit(conn, utc_now())).result == "skipped"
    assert not conn.execute("SELECT 1 FROM archive_walks").fetchone()


def test_a_post_given_up_on_lets_discovery_go_on_but_claims_no_coverage(conn, archive, monkeypatch):
    """Seen live: Substack refused one listed post, and discovery's links stage waits
    on the archive's first pull. Given up on, that post must not hold discovery back for good, and
    the walk still claims no coverage it does not have."""
    monkeypatch.setattr(archive[2], "page", lambda s, c, ctx: Page([("1", {"id": 1})]))
    assert step(conn, archive).result == "listed"
    conn.execute("UPDATE oracle_asks SET root_probed_at='then' WHERE canonical_id=?", (CID,))
    conn.commit()
    row = registry.list_sources(conn)[0]
    assert not walks.exhausted(conn, pair_rows.unit_key(row))
    assert Discovery()._owed(conn, utc_now()) == []

    conn.execute("UPDATE archive_units SET failures=? WHERE done=0", (retries.GIVE_UP_AFTER,))
    conn.commit()
    assert walks.exhausted(conn, pair_rows.unit_key(row))
    assert [stage for stage, _ask in Discovery()._owed(conn, utc_now())] == [LINKS]
    assert registry.list_sources(conn)[0].last_pulled_at is None
    assert archive[1].next_unit(conn, utc_now()) is None


def test_a_found_github_source_walk_brings_it_in_under_the_id_github_spells(conn, oracle,
                                                                           monkeypatch):
    """The walk writes the repos under the one id GitHub spells
    (`github:Karpathy`), and that account joins the Oracle it was found for."""
    import base64
    from .test_archive_adapters import Response

    from pipeline.oracles import review_items
    with conn:                                  # what discovery's register writes for a sure one
        review_items.write_found(conn, CID, [{"source_type": "github", "status": "added",
                                              "url": "https://github.com/karpathy"}])
        conn.execute("INSERT INTO found_sources (canonical_id, source_type, url, metadata, "
                     "found_at) VALUES (?, 'github', 'https://github.com/karpathy', '{}', ?)",
                     (CID, utc_now().isoformat()))

    repo = {"full_name": "Karpathy/nanoGPT", "name": "nanoGPT", "owner": {"login": "Karpathy"},
            "html_url": "https://github.com/Karpathy/nanoGPT", "fork": False,
            "pushed_at": "2026-10-01T00:00:00Z", "description": "The simplest GPT"}
    readme = {"encoding": "base64", "content": base64.b64encode(b"# nanoGPT").decode()}

    def answers(path, params=None, **kw):
        if path == "/users/karpathy/repos":
            return Response([repo] if params["page"] == 1 else [])
        return Response(readme) if path.endswith("/readme") else Response({}, 404)
    monkeypatch.setattr("pipeline.oracles.sources.archive.github.get", answers)
    loop = loops.web_loop()
    source = next(s for s in loop.sources if isinstance(s, Archives))
    for _ in range(10):
        if (unit := source.next_unit(conn, utc_now())) is None:
            break
        loop.run_unit(conn, source, unit)

    assert {r[0] for r in conn.execute("SELECT who_id FROM atoms")} == {"github:Karpathy"}
    assert {r[0] for r in conn.execute(
        "SELECT entity_id FROM entities WHERE entity_id LIKE 'github:%'")} == {"github:Karpathy"}
    assert schema.current_canonical(conn, "github:Karpathy") == schema.current_canonical(conn, CID)
