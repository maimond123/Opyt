"""Frontier stage 3 — the admissions source (`pipeline/frontier/admitting.py`), proven offline
against stubbed mint helpers.

The contract, in the order it matters:

  • AN ALREADY-PRESENT ARTIFACT MATERIALIZES ON ITS FIRST ADMISSION, without a fetch. This is the
    whole reason the stage exists. `atomize_paper` returns None for a paper it already has, so a
    driver that reads the return value retries that candidate FOREVER — Policy B guarantees the
    next try returns None too. Presence, not the return value, is the question.
  • ADMISSION NEVER REWRITES PROVENANCE. Repo dedup is by content hash and the star count is inside
    the hash, so one new star would re-write an existing atom with stage 3's own entry_mode. An
    `author_referenced` atom relabelled `frontier` leaves HUMAN_ATTESTED — the default search
    scope — so the user's own corpus would shrink every day stage 3 ran.
  • THE RETRIES ARE BOUNDED BY CONSTRUCTION. The retryable-vs-terminal rule is a guess (no
    distribution exists yet), so correctness cannot rest on it. The attempt cap makes `new → new`
    forever impossible whatever the rule gets wrong, and a failed candidate waits an hour.
  • A FAILURE WRITES NOTHING. No atom, no status change beyond the attempt counter.
  • THE HOUR'S ALLOWANCE BOUNDS SPEND. A backlog drains oldest-first, at most
    `ADMISSIONS_PER_HOUR` an hour, and `next_due` says when the hour frees.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.frontier import admit, admitting, allowance, candidates
from pipeline.frontier.admitting import Admissions
from pipeline.kb import schema
from pipeline.kb import ingest_papers as ip
from pipeline.platforms.papers import lookup as paper_lookup
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.source import Changed, Context

_SEEN = "2026-08-10T00:00:00+00:00"


_T0 = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _source(now) -> Admissions:
    return Admissions(Outages(PLATFORM_PAUSES_S), clock=lambda: now)


def _ctx():
    return Context("web")


@pytest.fixture()
def admit_all(conn):
    """Every admission the source offers, run as the web pool runs them, each call an hour and a
    minute after the last, so a failed candidate's wait has passed. What each came to."""
    clock = {"now": _T0}

    def run():
        clock["now"] += timedelta(hours=1, minutes=1)
        now = clock["now"]
        src = _source(now)
        return [src.record(conn, unit, src.fetch(conn, unit, _ctx())).status
                for unit in src.candidates(conn, now)]
    return run


_KIND_OF = {"arxiv": "paper", "openalex": "paper", "github": "repo"}


def _candidate(conn, cid, source, url, *, first_seen=_SEEN, kind=None, payload="{}", summary="s"):
    conn.execute(
        "INSERT INTO frontier_candidates (candidate_id, source, kind, title, url, published, "
        "summary, payload, status, first_seen_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?,'kept',?,?)",
        (cid, source, kind or _KIND_OF.get(source, source), f"t {cid}", url, "2026-08-09",
         summary, payload, first_seen, first_seen))
    conn.commit()
    return cid


def _row(conn, cid):
    return conn.execute(
        "SELECT status, attempts, last_error, atom_id FROM frontier_candidates "
        "WHERE candidate_id=?", (cid,)).fetchone()


def _seed_atom(conn, atom_id, entry_mode):
    schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": atom_id.split(":")[0],
                              "entry_mode": entry_mode, "raw_hash": "seed"})
    conn.commit()


def _never(*a, **kw):                       # an ingest that must not happen
    raise AssertionError("the mint helper was called for an already-present artifact")


# ── Hazard A: the inverted dedup contract ───────────────────────────────────────
def test_an_already_present_paper_materializes_on_the_first_pass(conn, monkeypatch, admit_all):
    """THE REGRESSION THIS MODULE EXISTS FOR. `atomize_paper` returns None for a paper it already
    holds, and Policy B skips before the fetch — so that None never becomes an id, on any future
    run. Reading it as failure is an infinite nightly retry, not a slow success."""
    _seed_atom(conn, "paper:arXiv:2501.00001", "author_referenced")
    cid = _candidate(conn, "arxiv:2501.00001", "arxiv", "https://arxiv.org/abs/2501.00001")
    monkeypatch.setattr(admit, "mint", _never)

    out = admit_all()

    assert _row(conn, cid)["status"] == "materialized"
    assert _row(conn, cid)["attempts"] == 0          # a success never increments
    assert out == ["materialized"]

# ── Hazard B: a content refresh must not rewrite provenance ─────────────────────
def test_admitting_a_present_repo_does_not_rewrite_its_entry_mode(conn, monkeypatch, admit_all):
    """`author_referenced` is in HUMAN_ATTESTED and the frontier mode deliberately is not, so a
    relabel here silently SHRINKS stage 1's only input — every night stage 3 runs. Mirrors
    tests/kb/test_ingest_substack.py::test_atom_from_url_policy_b_never_clobbers one rail over."""
    _seed_atom(conn, "github:owner/name", "author_referenced")
    cid = _candidate(conn, "repo:owner/name", "github", "https://github.com/owner/name")
    monkeypatch.setattr(admit, "mint", _never)

    admit_all()

    assert _row(conn, cid)["status"] == "materialized"
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='github:owner/name'"
                        ).fetchone()["entry_mode"] == "author_referenced"


def test_a_repo_present_under_different_casing_is_still_recognised(conn, monkeypatch, admit_all):
    """The pre-check parses the URL offline and cannot know the API's canonical `owner.login`, so
    an exact-case check would miss a stored `github:ggerganov/…` when the candidate URL says
    `Ggerganov` — and the miss lands in the rewrite path with the ingest already under way."""
    _seed_atom(conn, "github:ggerganov/llama.cpp", "oracle-footprint")
    cid = _candidate(conn, "repo:Ggerganov/llama.cpp", "github",
                     "https://github.com/Ggerganov/llama.cpp")
    monkeypatch.setattr(admit, "mint", _never)

    admit_all()

    assert _row(conn, cid)["status"] == "materialized"
    assert _row(conn, cid)["atom_id"] == "github:ggerganov/llama.cpp"   # the id as stored
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='github:ggerganov/llama.cpp'"
                        ).fetchone()["entry_mode"] == "oracle-footprint"


def test_the_frontier_mode_is_never_human_attested():
    """The one edit that silently breaks anti-narrowing, asserted directly rather than trusted to
    a comment. The guard rule in .guards.py carries the WHY at commit time; this fails the suite."""
    assert admit.ENTRY_MODE not in schema.HUMAN_ATTESTED


# ── The happy path ──────────────────────────────────────────────────────────────
def test_a_fresh_artifact_is_admitted_as_the_frontier_mode(conn, monkeypatch, admit_all):
    """Approval is NOT discovery: tagging an admitted candidate `user-saved` would present a
    crawler find as something the user saved. Approval lives in `status`; `entry_mode` says how it
    was FOUND."""
    cid = _candidate(conn, "arxiv:2501.00003", "arxiv", "https://arxiv.org/abs/2501.00003")

    def _mint(conn_, row):
        _seed_atom(conn_, "paper:arXiv:2501.00003", admit.ENTRY_MODE)
        return None                                  # helpers report nothing useful — D1 decides
    monkeypatch.setattr(admit, "mint", _mint)

    admit_all()

    assert _row(conn, cid)["status"] == "materialized"
    assert _row(conn, cid)["attempts"] == 0
    assert _row(conn, cid)["atom_id"] == "paper:arXiv:2501.00003"   # which watch found it
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='paper:arXiv:2501.00003'"
                        ).fetchone()["entry_mode"] == "frontier"


# ── Failure handling ────────────────────────────────────────────────────────────
def test_an_underivable_id_is_terminal_on_the_first_attempt(conn, monkeypatch, admit_all):
    """`no_atom_id` needs no network to establish and cannot change on a retry, so it is one of the
    few failures that is unambiguously terminal."""
    monkeypatch.setattr(admit, "mint", _never)
    cid = _candidate(conn, "repo:not/a/repo", "github", "https://github.com/onlyowner")

    admit_all()

    row = _row(conn, cid)
    assert (row["status"], row["last_error"]) == ("rejected", "no_atom_id")


def test_a_failing_candidate_is_bounded_by_the_attempt_cap(conn, monkeypatch, admit_all):
    """The retryable/terminal rule is a GUESS — this is why being wrong is survivable. A candidate
    that never converges is forced terminal by the counter, not by the classification."""
    monkeypatch.setattr(admitting, "ADMIT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(admit, "mint", lambda *a, **kw: None)      # declines, writes nothing
    cid = _candidate(conn, "arxiv:2501.00004", "arxiv", "https://arxiv.org/abs/2501.00004")

    seen = []
    for _ in range(5):                                             # more passes than the cap
        admit_all()
        seen.append((_row(conn, cid)["status"], _row(conn, cid)["attempts"]))

    assert seen[0] == ("kept", 1)
    assert seen[1] == ("kept", 2)
    assert seen[2] == ("rejected", 3)                              # forced at exactly the cap
    assert seen[3] == seen[4] == ("rejected", 3)                   # never a 4th attempt
    assert _row(conn, cid)["last_error"] == "ingest_declined"


def test_a_raising_ingester_leaves_no_partial_state(conn, monkeypatch, admit_all):
    """The repo invariant: a failed external call SKIPS. No atom, nothing marked done — only the
    counter moves, so a later run re-attempts from a clean slate."""
    def _boom(*a, **kw):
        raise TimeoutError("upstream hung")
    monkeypatch.setattr(admit, "mint", _boom)
    cid = _candidate(conn, "arxiv:2501.00005", "arxiv", "https://arxiv.org/abs/2501.00005")

    admit_all()

    row = _row(conn, cid)
    assert (row["status"], row["attempts"]) == ("kept", 1)
    assert row["last_error"] == "error_timeouterror"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_a_locked_store_never_charges_an_attempt(conn, monkeypatch, admit_all):
    """A `database is locked` that escapes the mint's write deadline is the store's
    contention, not the candidate's fault. Charged as a failed attempt, five contended passes would
    permanently reject a good candidate; owed instead, it is simply tried again."""
    def _locked(*a, **kw):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(admit, "mint", _locked)
    cid = _candidate(conn, "arxiv:2501.00006", "arxiv", "https://arxiv.org/abs/2501.00006")

    for _ in range(admitting.ADMIT_MAX_ATTEMPTS):
        admit_all()

    row = _row(conn, cid)
    assert (row["status"], row["attempts"], row["last_error"]) == ("kept", 0, "locked")


# ── The hour's allowance ────────────────────────────────────────────────────────
def test_an_hour_never_admits_more_than_its_allowance(conn, monkeypatch):
    """Materializing fetches full text and writes it — the rail's first real RAM and network
    exposure. A backlog drains oldest-first over several hours rather than in one."""
    monkeypatch.setattr(allowance, "ADMISSIONS_PER_HOUR", 2)
    monkeypatch.setattr(admit, "mint", lambda *a, **kw: None)
    for i in range(5):
        _candidate(conn, f"arxiv:2501.1000{i}", "arxiv", f"https://arxiv.org/abs/2501.1000{i}",
                   first_seen=f"2026-08-0{i + 1}T00:00:00+00:00")
    src = _source(_T0)

    offered = src.candidates(conn, _T0)
    assert [u.payload["candidate_id"] for u in offered] == ["arxiv:2501.10000",
                                                           "arxiv:2501.10001"]   # oldest first
    for unit in offered:
        src.fetch(conn, unit, _ctx())
    assert src.candidates(conn, _T0) == []
    assert _row(conn, "arxiv:2501.10004")["attempts"] == 0        # newest untouched


def test_a_spent_hour_says_when_it_frees(conn, monkeypatch):
    """A held-back backlog reads exactly like an empty one unless something says when it moves."""
    monkeypatch.setattr(allowance, "ADMISSIONS_PER_HOUR", 1)
    monkeypatch.setattr(admit, "mint", lambda *a, **kw: None)
    for i in range(2):
        _candidate(conn, f"arxiv:2501.2000{i}", "arxiv", f"https://arxiv.org/abs/2501.2000{i}")
    src = _source(_T0)
    src.fetch(conn, src.candidates(conn, _T0)[0], _ctx())

    assert src.candidates(conn, _T0) == []
    assert src.next_due(conn, _T0) == _T0 + allowance.HOUR


def test_a_failed_candidate_waits_an_hour(conn, monkeypatch):
    monkeypatch.setattr(admit, "mint", lambda *a, **kw: None)
    cid = _candidate(conn, "arxiv:2501.40001", "arxiv", "https://arxiv.org/abs/2501.40001")
    src = _source(_T0)
    src.fetch(conn, src.candidates(conn, _T0)[0], _ctx())

    assert _row(conn, cid)["status"] == "kept"
    assert src.candidates(conn, _T0 + timedelta(minutes=59)) == []
    assert src.next_due(conn, _T0) == _T0 + allowance.HOUR
    assert len(src.candidates(conn, _T0 + allowance.HOUR)) == 1


def test_nothing_owed_is_never_due(conn):
    assert _source(_T0).next_due(conn, _T0) is None


# ── The web pool ────────────────────────────────────────────────────────────────
def test_admissions_run_one_at_a_time_per_service(conn):
    _candidate(conn, "arxiv:2501.50001", "arxiv", "https://arxiv.org/abs/2501.50001")
    _candidate(conn, "repo:o/n", "github", "https://github.com/o/n")
    _candidate(conn, "x:1", "x", "https://x.com/u/status/1")
    _candidate(conn, "substack:2", "substack", "https://www.forever.ai/p/two")
    units = {u.payload["candidate_id"]: u for u in _source(_T0).candidates(conn, _T0)}
    assert units["arxiv:2501.50001"].host == "api.semanticscholar.org"
    assert units["repo:o/n"].host == "api.github.com"
    assert (units["x:1"].host, units["substack:2"].host) == ("x.com", "substack.com")


def test_x_and_substack_finds_are_admitted_by_their_own_minters_as_the_frontier_mode(
        conn, monkeypatch, admit_all):
    """Keyed on the atom's own id by their adapters (a Substack url does not carry the post id),
    and minted by the paths hopper and the Oracle links use, never as `user-saved`."""
    _candidate(conn, "x:100", "x", "https://x.com/u/status/100")
    _candidate(conn, "substack:219807469", "substack", "https://www.forever.ai/p/rentosertib")
    minted = []

    def _x(conn_, url, *, entry_mode):
        minted.append(("x", url, entry_mode))
        _seed_atom(conn_, "x:100", entry_mode)
        return "saved", "x:100"

    def _substack(conn_, url, *, entry_mode):
        minted.append(("substack", url, entry_mode))
        _seed_atom(conn_, "substack:219807469", entry_mode)
        return "substack:219807469"

    monkeypatch.setattr(admit.ingest_x, "x_atom_from_url", _x)
    monkeypatch.setattr(admit.ingest_substack, "substack_atom_from_url", _substack)
    assert admit_all() == ["materialized", "materialized"]
    assert sorted(minted) == [("substack", "https://www.forever.ai/p/rentosertib", "frontier"),
                              ("x", "https://x.com/u/status/100", "frontier")]
    assert _row(conn, "substack:219807469")["atom_id"] == "substack:219807469"


def test_an_x_post_the_user_already_saved_is_not_minted_again(conn, monkeypatch, admit_all):
    _seed_atom(conn, "x:200", "user-saved")
    cid = _candidate(conn, "x:200", "x", "https://x.com/u/status/200")
    monkeypatch.setattr(admit, "mint", _never)
    assert admit_all() == ["materialized"]
    assert _row(conn, cid)["atom_id"] == "x:200"
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='x:200'"
                        ).fetchone()["entry_mode"] == "user-saved"


def test_a_paused_platform_offers_no_admission(conn):
    """The web pool's outage pause on a platform holds frontier's admissions too."""
    _candidate(conn, "repo:o/n", "github", "https://github.com/o/n")
    _candidate(conn, "arxiv:2501.50002", "arxiv", "https://arxiv.org/abs/2501.50002")
    src = _source(_T0)
    src.outages.pause("github", _T0, 1.0)
    assert [u.payload["candidate_id"] for u in src.candidates(conn, _T0)] == ["arxiv:2501.50002"]


def test_a_lost_lease_writes_no_status(conn, monkeypatch):
    monkeypatch.setattr(admit, "mint", lambda *a, **kw: None)
    cid = _candidate(conn, "arxiv:2501.60001", "arxiv", "https://arxiv.org/abs/2501.60001")
    src = _source(_T0)
    unit = src.candidates(conn, _T0)[0]
    with pytest.raises(Changed):
        src.fetch(conn, unit, Context("web", lease_lost=lambda: True))
    assert tuple(_row(conn, cid)) == ("kept", 0, None, None)


# ── Requeue ─────────────────────────────────────────────────────────────────────
def test_requeue_rejected_round_trips(conn, monkeypatch, admit_all):
    """Every part of the retry rule is provisional, so acting on the measurement has to be
    possible. Without the counter reset the requeued row re-rejects at once and the flag is
    decorative."""
    monkeypatch.setattr(admitting, "ADMIT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(admit, "mint", lambda *a, **kw: None)
    cid = _candidate(conn, "arxiv:2501.30001", "arxiv", "https://arxiv.org/abs/2501.30001")
    for _ in range(2):
        admit_all()
    assert _row(conn, cid)["status"] == "rejected"

    assert candidates.requeue_rejected(conn) == 1
    row = _row(conn, cid)
    assert (row["status"], row["attempts"], row["last_error"]) == ("kept", 0, None)

    def _mint(conn_, row):
        _seed_atom(conn_, "paper:arXiv:2501.30001", admit.ENTRY_MODE)
    monkeypatch.setattr(admit, "mint", _mint)
    admit_all()
    assert _row(conn, cid)["status"] == "materialized"


def test_requeue_can_be_narrowed_to_one_reason(conn, monkeypatch, admit_all):
    """The point of the reason slug: fix the rule for ONE failure mode, requeue exactly the rows it
    got wrong, leave the genuinely dead ones dead."""
    monkeypatch.setattr(admit, "mint", _never)
    dead = _candidate(conn, "repo:bad/url", "github", "https://github.com/onlyowner")
    admit_all()
    conn.execute("UPDATE frontier_candidates SET status='rejected', last_error='blocked_metadata' "
                 "WHERE candidate_id=?", ("repo:bad/url",))
    other = _candidate(conn, "repo:also/bad", "github", "https://github.com/onlyowner2")
    admit_all()

    assert candidates.requeue_rejected(conn, last_error="blocked_metadata") == 1
    assert _row(conn, dead)["status"] == "kept"
    assert _row(conn, other)["status"] == "rejected"          # a different reason, left alone


# ── The finder/minter split ─────────────────────────────────────────────────────
def test_a_new_paper_finder_needs_no_arm_of_its_own(conn, monkeypatch, admit_all):
    """WHAT THE SPLIT BOUGHT, asserted as behaviour rather than as shape. `openalex` appears
    nowhere in stage 3 — it dispatches on `kind`, so a paper from any finder reaches the paper
    minter. Keyed on `source`, this candidate would fall through to `no_atom_id` and every future
    paper source would need a third arm doing byte-identically what the arxiv arm already does."""
    _seed_atom(conn, "paper:arXiv:2608.09055", "author_referenced")
    cid = _candidate(conn, "openalex:W7202193881", "openalex",
                     "https://doi.org/10.48550/arxiv.2608.09055")
    monkeypatch.setattr(admit, "mint", _never)

    admit_all()

    assert _row(conn, cid)["status"] == "materialized"


def test_a_candidate_with_no_kind_is_rejected_rather_than_guessed(conn, monkeypatch, admit_all):
    """The safety behind having NO backfill. Every pre-existing row is `materialized` and stage 3
    only reads `status='kept'`, so a NULL `kind` is never reached — and if one ever were, the
    fail-closed answer must be a rejection, never a guess from `source` that reintroduces the
    coupling the split removed."""
    monkeypatch.setattr(admit, "mint", _never)
    cid = _candidate(conn, "arxiv:2501.99999", "arxiv", "https://arxiv.org/abs/2501.99999")
    conn.execute("UPDATE frontier_candidates SET kind = NULL WHERE candidate_id=?", (cid,))
    conn.commit()

    admit_all()

    row = _row(conn, cid)
    assert (row["status"], row["last_error"]) == ("rejected", "no_atom_id")


def test_the_minter_is_handed_the_metadata_stage_two_already_collected(conn, monkeypatch, admit_all):
    """The measured hazard. Semantic Scholar resolved 1 of 15 OpenAlex DOIs; the
    rest 404, which is `FETCH_ABSENT` — and `atomize_paper` does NOT skip that. Without the
    finder's own title and abstract each 404 would freeze a contentless atom into the store
    forever, because papers are immutable under Policy B and no later run revisits one."""
    cid = _candidate(conn, "openalex:W7164878913", "openalex",
                     "https://doi.org/10.5281/zenodo.20719927",
                     summary="A study of verifiable disclosure.",
                     payload=json.dumps({"authors": [{"name": "A. Person"}]}))
    conn.execute("UPDATE frontier_candidates SET title=? WHERE candidate_id=?",
                 ("Earned Trust", cid))
    conn.commit()

    seen = {}
    monkeypatch.setattr(paper_lookup, "fetch_s2_paper",
                        lambda lookup: (None, paper_lookup.ABSENT))
    monkeypatch.setattr(ip, "atomize_paper",
                        lambda conn_, paper, **kw: seen.update(paper) or
                        _seed_atom(conn_, "paper:DOI:10.5281/zenodo.20719927", admit.ENTRY_MODE))

    admit_all()

    assert _row(conn, cid)["status"] == "materialized"
    assert seen["title"] == "Earned Trust"
    assert seen["abstract"] == "A study of verifiable disclosure."
    # Name only, and the `list[str]` payload above is the older shape — this doubles as
    # the proof that rows staged before the author id was carried still read, since a candidate
    # payload is frozen at stage-2 write time.
    assert seen["authors"] == [{"name": "A. Person"}]


def test_the_openalex_author_id_reaches_the_minter_but_never_as_a_scholar_id(conn, monkeypatch, admit_all):
    """The whole point of carrying `author.id`: without it the only route back to a PERSON is a
    `display_name` search, which returned 16 people for "Frances Arnold" — one with
    928 works and one with 2.

    It rides under `openalexId`, NEVER `authorId`. `derive_paper` reads `authorId` to mint
    `who_id = scholar:{id}`, so an OpenAlex id there would assert a Semantic Scholar identity that
    does not exist; `who_id` must still fall to the honest `paper-authors:{id}` placeholder."""
    cid = _candidate(conn, "openalex:W7164878913", "openalex",
                     "https://doi.org/10.5281/zenodo.20719927",
                     summary="A study of verifiable disclosure.",
                     payload=json.dumps({"authors": [
                         {"name": "F. Arnold", "openalex_id": "A5043841592",
                          "orcid": "0000-0002-4027-364X", "position": "middle"},
                         {"name": "No Id Coauthor"}]}))
    conn.execute("UPDATE frontier_candidates SET title=? WHERE candidate_id=?",
                 ("Earned Trust", cid))
    conn.commit()

    seen = {}
    monkeypatch.setattr(paper_lookup, "fetch_s2_paper",
                        lambda lookup: (None, paper_lookup.ABSENT))
    monkeypatch.setattr(ip, "atomize_paper",
                        lambda conn_, paper, **kw: seen.update(paper) or
                        _seed_atom(conn_, "paper:DOI:10.5281/zenodo.20719927", admit.ENTRY_MODE))

    admit_all()

    assert seen["authors"] == [
        {"name": "F. Arnold", "openalexId": "A5043841592",
         "orcid": "0000-0002-4027-364X", "position": "middle"},
        {"name": "No Id Coauthor"}]
    assert not any("authorId" in a for a in seen["authors"])
    assert ip.derive.derive_paper(seen)["who_id"].startswith("paper-authors:")


def test_non_object_payload_has_no_known_metadata_fields():
    row = {"payload": "[]", "title": "Paper", "summary": "Abstract", "published": "2026-10-05"}
    assert admit.known_metadata(row)["authors"] == []


# ── the host's check comes before the library ────
def test_a_find_nobody_has_checked_is_never_admitted(conn, admit_all):
    """A find waits as `new` until the host checks it; only a kept one reaches the library."""
    cid = _candidate(conn, "arxiv:2508.00001", "arxiv", "https://arxiv.org/abs/2508.00001")
    conn.execute("UPDATE frontier_candidates SET status='new' WHERE candidate_id=?", (cid,))
    conn.commit()
    assert admit_all() == []
    assert _row(conn, cid)["status"] == "new"


def test_the_check_moves_only_waiting_finds_and_says_how_many(conn):
    for cid in ("arxiv:a", "arxiv:b", "arxiv:c", "arxiv:d"):
        _candidate(conn, cid, "arxiv", f"https://arxiv.org/abs/{cid}")
    conn.execute("UPDATE frontier_candidates SET status='new' WHERE candidate_id != 'arxiv:d'")
    out = candidates.check(conn, kept=["arxiv:a", "arxiv:d", "arxiv:a"], passed=["arxiv:b"])
    conn.commit()
    assert out == {"kept": 1, "passed": 1}          # arxiv:d was already kept; a repeat counts once
    assert [_row(conn, c)["status"] for c in ("arxiv:a", "arxiv:b", "arxiv:c", "arxiv:d")] == [
        "kept", "passed", "new", "kept"]
    assert candidates.check(conn, kept=[], passed=["arxiv:a"]) == {"kept": 0, "passed": 0}


def test_waiting_lists_a_querys_unchecked_finds_with_what_found_them(conn):
    from pipeline.frontier import queries as fq
    fq.add(conn, "OpenFold3", ["arxiv"], now="2026-08-01T00:00:00Z")
    qid = fq.query_id_for("openfold3")
    for cid, status in (("arxiv:a", "new"), ("arxiv:b", "passed")):
        _candidate(conn, cid, "arxiv", f"https://arxiv.org/abs/{cid}")
        conn.execute("UPDATE frontier_candidates SET status=? WHERE candidate_id=?", (status, cid))
        conn.execute("INSERT INTO frontier_candidate_queries VALUES (?, ?, 'a')", (cid, qid))
    conn.commit()
    assert [(r["candidate_id"], r["found_by"]) for r in candidates.waiting(conn, [qid])] == [
        ("arxiv:a", "OpenFold3")]
