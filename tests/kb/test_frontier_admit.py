"""Frontier stage 3 — ADMIT, proven offline against stubbed mint helpers.

The contract, in the order it matters:

  • AN ALREADY-PRESENT ARTIFACT MATERIALIZES ON THE FIRST PASS, without a fetch. This is the whole
    reason the module exists. `atomize_paper` returns None for a paper it already has, so a driver
    that reads the return value retries that candidate nightly FOREVER — Policy B guarantees the
    next run returns None too. Presence, not the return value, is the question.
  • ADMISSION NEVER REWRITES PROVENANCE. Repo dedup is by content hash and the star count is inside
    the hash, so one new star would re-write an existing atom with stage 3's own entry_mode. An
    `author_referenced` atom relabelled `frontier` leaves HUMAN_ATTESTED — which is stage 1's only
    input — so the generator's corpus would shrink every night stage 3 ran.
  • THE LOOP IS BOUNDED BY CONSTRUCTION. The retryable-vs-terminal rule is a guess (no distribution
    exists yet), so correctness cannot rest on it. The attempt cap makes `new → new` forever
    impossible whatever the rule gets wrong.
  • A FAILURE WRITES NOTHING. No atom, no status change beyond the attempt counter.
"""
from __future__ import annotations

import pytest

from pipeline.kb import frontier_admit as fa
from pipeline.kb import schema

_SEEN = "2026-08-10T00:00:00+00:00"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


_KIND_OF = {"arxiv": "paper", "openalex": "paper", "github": "repo"}


def _candidate(conn, cid, source, url, *, first_seen=_SEEN, kind=None, payload="{}", summary="s"):
    conn.execute(
        "INSERT INTO frontier_candidates (candidate_id, source, kind, title, url, published, "
        "summary, payload, status, first_seen_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?,'new',?,?)",
        (cid, source, kind or _KIND_OF.get(source, source), f"t {cid}", url, "2026-08-09",
         summary, payload, first_seen, first_seen))
    conn.commit()
    return cid


def _row(conn, cid):
    return conn.execute(
        "SELECT status, attempts, last_error FROM frontier_candidates WHERE candidate_id=?",
        (cid,)).fetchone()


def _seed_atom(conn, atom_id, entry_mode):
    schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": atom_id.split(":")[0],
                              "entry_mode": entry_mode, "raw_hash": "seed"})
    conn.commit()


def _never(*a, **kw):                       # an ingest that must not happen
    raise AssertionError("the mint helper was called for an already-present artifact")


# ── Hazard A: the inverted dedup contract ───────────────────────────────────────
def test_an_already_present_paper_materializes_on_the_first_pass(conn, monkeypatch):
    """THE REGRESSION THIS MODULE EXISTS FOR. `atomize_paper` returns None for a paper it already
    holds, and Policy B skips before the fetch — so that None never becomes an id, on any future
    run. Reading it as failure is an infinite nightly retry, not a slow success."""
    _seed_atom(conn, "paper:arXiv:2501.00001", "author_referenced")
    cid = _candidate(conn, "arxiv:2501.00001", "arxiv", "https://arxiv.org/abs/2501.00001")
    monkeypatch.setattr(fa, "_ingest", _never)

    out = fa.run_frontier_admit(conn, embedder=object())

    assert _row(conn, cid)["status"] == "materialized"
    assert _row(conn, cid)["attempts"] == 0          # a success never increments
    assert out["materialized"] == 1


def test_the_embedder_is_never_built_for_an_already_present_artifact(conn, monkeypatch):
    """'Already present costs one SELECT' is only true if it also costs no embedder. Constructing
    one needs an API key and a live network path, so a machine with neither must still drain a
    backlog of artifacts it already has."""
    _seed_atom(conn, "paper:arXiv:2501.00002", "user-saved")
    _candidate(conn, "arxiv:2501.00002", "arxiv", "https://arxiv.org/abs/2501.00002")
    monkeypatch.setattr(fa, "_ingest", _never)
    from pipeline.kb import embed
    monkeypatch.setattr(embed, "get_kb_embedder",
                        lambda **kw: pytest.fail("built an embedder for a present artifact"))

    assert fa.run_frontier_admit(conn, embedder=None)["materialized"] == 1


# ── Hazard B: a content refresh must not rewrite provenance ─────────────────────
def test_admitting_a_present_repo_does_not_rewrite_its_entry_mode(conn, monkeypatch):
    """`author_referenced` is in HUMAN_ATTESTED and the frontier mode deliberately is not, so a
    relabel here silently SHRINKS stage 1's only input — every night stage 3 runs. Mirrors
    tests/kb/test_ingest_substack.py::test_atom_from_url_policy_b_never_clobbers one rail over."""
    _seed_atom(conn, "github:owner/name", "author_referenced")
    cid = _candidate(conn, "repo:owner/name", "github", "https://github.com/owner/name")
    monkeypatch.setattr(fa, "_ingest", _never)

    fa.run_frontier_admit(conn, embedder=object())

    assert _row(conn, cid)["status"] == "materialized"
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='github:owner/name'"
                        ).fetchone()["entry_mode"] == "author_referenced"


def test_a_repo_present_under_different_casing_is_still_recognised(conn, monkeypatch):
    """The pre-check parses the URL offline and cannot know the API's canonical `owner.login`, so
    an exact-case check would miss a stored `github:ggerganov/…` when the candidate URL says
    `Ggerganov` — and the miss lands in the rewrite path with the ingest already under way."""
    _seed_atom(conn, "github:ggerganov/llama.cpp", "oracle-footprint")
    cid = _candidate(conn, "repo:Ggerganov/llama.cpp", "github",
                     "https://github.com/Ggerganov/llama.cpp")
    monkeypatch.setattr(fa, "_ingest", _never)

    fa.run_frontier_admit(conn, embedder=object())

    assert _row(conn, cid)["status"] == "materialized"
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='github:ggerganov/llama.cpp'"
                        ).fetchone()["entry_mode"] == "oracle-footprint"


def test_the_frontier_mode_is_never_human_attested():
    """The one edit that silently breaks anti-narrowing, asserted directly rather than trusted to
    a comment. The guard rule in .guards.py carries the WHY at commit time; this fails the suite."""
    assert fa.ENTRY_MODE not in schema.HUMAN_ATTESTED


# ── The happy path ──────────────────────────────────────────────────────────────
def test_a_fresh_artifact_is_admitted_as_the_frontier_mode(conn, monkeypatch):
    """Approval is NOT discovery: tagging an admitted candidate `user-saved` would feed it straight
    back into the generator. Approval lives in `status`; `entry_mode` says how it was FOUND."""
    cid = _candidate(conn, "arxiv:2501.00003", "arxiv", "https://arxiv.org/abs/2501.00003")

    def _mint(conn_, embedder, row):
        _seed_atom(conn_, "paper:arXiv:2501.00003", fa.ENTRY_MODE)
        return None                                  # helpers report nothing useful — D1 decides
    monkeypatch.setattr(fa, "_ingest", _mint)

    fa.run_frontier_admit(conn, embedder=object())

    assert _row(conn, cid)["status"] == "materialized"
    assert _row(conn, cid)["attempts"] == 0
    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='paper:arXiv:2501.00003'"
                        ).fetchone()["entry_mode"] == "frontier"


# ── Failure handling ────────────────────────────────────────────────────────────
def test_an_underivable_id_is_terminal_on_the_first_attempt(conn, monkeypatch):
    """`no_atom_id` needs no network to establish and cannot change on a retry, so it is one of the
    few failures that is unambiguously terminal."""
    monkeypatch.setattr(fa, "_ingest", _never)
    cid = _candidate(conn, "repo:not/a/repo", "github", "https://github.com/onlyowner")

    fa.run_frontier_admit(conn, embedder=object())

    row = _row(conn, cid)
    assert (row["status"], row["last_error"]) == ("rejected", "no_atom_id")


def test_a_failing_candidate_is_bounded_by_the_attempt_cap(conn, monkeypatch):
    """The retryable/terminal rule is a GUESS — this is why being wrong is survivable. A candidate
    that never converges is forced terminal by the counter, not by the classification."""
    monkeypatch.setattr(fa, "ADMIT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(fa, "_ingest", lambda *a, **kw: None)      # declines, writes nothing
    cid = _candidate(conn, "arxiv:2501.00004", "arxiv", "https://arxiv.org/abs/2501.00004")

    seen = []
    for _ in range(5):                                             # more passes than the cap
        fa.run_frontier_admit(conn, embedder=object())
        seen.append((_row(conn, cid)["status"], _row(conn, cid)["attempts"]))

    assert seen[0] == ("new", 1)
    assert seen[1] == ("new", 2)
    assert seen[2] == ("rejected", 3)                              # forced at exactly the cap
    assert seen[3] == seen[4] == ("rejected", 3)                   # never a 4th attempt
    assert _row(conn, cid)["last_error"] == "ingest_declined"


def test_a_raising_ingester_leaves_no_partial_state(conn, monkeypatch):
    """The repo invariant: a failed external call SKIPS. No atom, nothing marked done — only the
    counter moves, so a later run re-attempts from a clean slate."""
    def _boom(*a, **kw):
        raise TimeoutError("upstream hung")
    monkeypatch.setattr(fa, "_ingest", _boom)
    cid = _candidate(conn, "arxiv:2501.00005", "arxiv", "https://arxiv.org/abs/2501.00005")

    fa.run_frontier_admit(conn, embedder=object())

    row = _row(conn, cid)
    assert (row["status"], row["attempts"]) == ("new", 1)
    assert row["last_error"] == "error_timeouterror"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


# ── The per-run cap ─────────────────────────────────────────────────────────────
def test_the_run_never_exceeds_its_admission_cap(conn, monkeypatch):
    """Materializing fetches full text and embeds it — the rail's first real RAM and spend
    exposure. A backlog drains oldest-first over several passes rather than in one."""
    monkeypatch.setattr(fa, "_ingest", lambda *a, **kw: None)
    for i in range(5):
        _candidate(conn, f"arxiv:2501.1000{i}", "arxiv", f"https://arxiv.org/abs/2501.1000{i}",
                   first_seen=f"2026-08-0{i + 1}T00:00:00+00:00")

    out = fa.run_frontier_admit(conn, embedder=object(), limit=2)

    assert out["considered"] == 2 and out["backlog"] == 5
    assert _row(conn, "arxiv:2501.10000")["attempts"] == 1        # oldest first
    assert _row(conn, "arxiv:2501.10004")["attempts"] == 0        # newest untouched


def test_a_bound_cap_is_logged_rather_than_silent(conn, monkeypatch):
    """A run that quietly dropped two thirds of its work reads exactly like one that covered
    everything."""
    monkeypatch.setattr(fa, "_ingest", lambda *a, **kw: None)
    lines: list[str] = []
    monkeypatch.setattr(fa, "log", lines.append)
    for i in range(3):
        _candidate(conn, f"arxiv:2501.2000{i}", "arxiv", f"https://arxiv.org/abs/2501.2000{i}")

    fa.run_frontier_admit(conn, embedder=object(), limit=1)

    assert any("admission cap bound" in ln for ln in lines)


# ── Requeue ─────────────────────────────────────────────────────────────────────
def test_requeue_rejected_round_trips(conn, monkeypatch):
    """Every part of the retry rule is provisional, so acting on the measurement has to be
    possible. Without the counter reset the requeued row re-rejects at once and the flag is
    decorative."""
    monkeypatch.setattr(fa, "ADMIT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(fa, "_ingest", lambda *a, **kw: None)
    cid = _candidate(conn, "arxiv:2501.30001", "arxiv", "https://arxiv.org/abs/2501.30001")
    for _ in range(2):
        fa.run_frontier_admit(conn, embedder=object())
    assert _row(conn, cid)["status"] == "rejected"

    assert fa.requeue_rejected(conn) == 1
    row = _row(conn, cid)
    assert (row["status"], row["attempts"], row["last_error"]) == ("new", 0, None)

    def _mint(conn_, embedder, row):
        _seed_atom(conn_, "paper:arXiv:2501.30001", fa.ENTRY_MODE)
    monkeypatch.setattr(fa, "_ingest", _mint)
    fa.run_frontier_admit(conn, embedder=object())
    assert _row(conn, cid)["status"] == "materialized"


def test_requeue_can_be_narrowed_to_one_reason(conn, monkeypatch):
    """The point of the reason slug: fix the rule for ONE failure mode, requeue exactly the rows it
    got wrong, leave the genuinely dead ones dead."""
    monkeypatch.setattr(fa, "_ingest", _never)
    dead = _candidate(conn, "repo:bad/url", "github", "https://github.com/onlyowner")
    fa.run_frontier_admit(conn, embedder=object())
    conn.execute("UPDATE frontier_candidates SET status='rejected', last_error='blocked_metadata' "
                 "WHERE candidate_id=?", ("repo:bad/url",))
    other = _candidate(conn, "repo:also/bad", "github", "https://github.com/onlyowner2")
    fa.run_frontier_admit(conn, embedder=object())

    assert fa.requeue_rejected(conn, last_error="blocked_metadata") == 1
    assert _row(conn, dead)["status"] == "new"
    assert _row(conn, other)["status"] == "rejected"          # a different reason, left alone


# ── The spawn ───────────────────────────────────────────────────────────────────
@pytest.fixture()
def spawn_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPYT_NO_FRONTIER_ADMIT", raising=False)
    monkeypatch.setenv("OPYT_FRONTIER_ADMIT_STAMP", str(tmp_path / "stamp"))
    monkeypatch.setenv("OPYT_FRONTIER_ADMIT_LOG", str(tmp_path / "admit.log"))
    calls: list = []
    monkeypatch.setattr(fa.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    return calls


def test_the_spawn_is_detached_and_never_writes_to_stdout(spawn_env):
    """The server's stdout IS the JSON-RPC channel, so an inherited handle corrupts the protocol."""
    from pathlib import Path
    assert fa.spawn_frontier_admit() is True
    cmd, kw = spawn_env[0]
    assert cmd[1:] == ["-m", "pipeline.kb.frontier_admit", "--once"]
    assert kw["stdout"] is kw["stderr"] and kw["stdout"] is not None
    assert kw["stdin"] == fa.subprocess.DEVNULL
    assert kw["start_new_session"] is True
    assert (Path(kw["cwd"]) / "pipeline" / "kb").is_dir()


def test_the_kill_switch_stops_stage_three_without_touching_stage_two(spawn_env, monkeypatch):
    """Each rail owns its switch. Stage 2 fails on a flaky upstream index and stage 3 on a PDF
    fetch or an embed, so either must be disableable alone."""
    monkeypatch.setenv("OPYT_NO_FRONTIER_ADMIT", "1")
    assert fa.spawn_frontier_admit(force=True) is False
    assert spawn_env == []


def test_the_coalesce_window_stops_every_session_firing_a_pass(spawn_env):
    assert fa.spawn_frontier_admit() is True
    assert fa.spawn_frontier_admit() is False
    assert fa.spawn_frontier_admit(force=True) is True
    assert len(spawn_env) == 2


def test_a_broken_spawn_is_swallowed_rather_than_raised(spawn_env, monkeypatch):
    """It is called from the MCP server's startup path — a hiccup must never stop it serving."""
    def _boom(*a, **kw):
        raise OSError("no fork for you")
    monkeypatch.setattr(fa.subprocess, "Popen", _boom)
    assert fa.spawn_frontier_admit(force=True) is False


def test_the_server_wires_the_spawner():
    """Stage 3 in its OWN try/except beside the others — never a tail of stage 2."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "mcp_server" / "server.py").read_text()
    assert "spawn_frontier_admit" in src


# ── The finder/minter split ─────────────────────────────────────────────────────
def test_a_new_paper_finder_needs_no_arm_of_its_own(conn, monkeypatch):
    """WHAT THE SPLIT BOUGHT, asserted as behaviour rather than as shape. `openalex` appears
    nowhere in stage 3 — it dispatches on `kind`, so a paper from any finder reaches the paper
    minter. Keyed on `source`, this candidate would fall through to `no_atom_id` and every future
    paper source would need a third arm doing byte-identically what the arxiv arm already does."""
    _seed_atom(conn, "paper:arXiv:2608.09055", "author_referenced")
    cid = _candidate(conn, "openalex:W7202193881", "openalex",
                     "https://doi.org/10.48550/arxiv.2608.09055")
    monkeypatch.setattr(fa, "_ingest", _never)

    fa.run_frontier_admit(conn, embedder=object())

    assert _row(conn, cid)["status"] == "materialized"


def test_a_candidate_with_no_kind_is_rejected_rather_than_guessed(conn, monkeypatch):
    """The safety behind having NO backfill. Every pre-existing row is `materialized` and stage 3
    only reads `status='new'`, so a NULL `kind` is never reached — and if one ever were, the
    fail-closed answer must be a rejection, never a guess from `source` that reintroduces the
    coupling the split removed."""
    monkeypatch.setattr(fa, "_ingest", _never)
    cid = _candidate(conn, "arxiv:2501.99999", "arxiv", "https://arxiv.org/abs/2501.99999")
    conn.execute("UPDATE frontier_candidates SET kind = NULL WHERE candidate_id=?", (cid,))
    conn.commit()

    fa.run_frontier_admit(conn, embedder=object())

    row = _row(conn, cid)
    assert (row["status"], row["last_error"]) == ("rejected", "no_atom_id")


def test_the_minter_is_handed_the_metadata_stage_two_already_collected(conn, monkeypatch):
    """The measured hazard. Semantic Scholar resolved 1 of 15 OpenAlex DOIs on 2026-08-26; the
    rest 404, which is `FETCH_ABSENT` — and `atomize_paper` does NOT skip that. Without the
    finder's own title and abstract each 404 would freeze a contentless atom into the store
    forever, because papers are immutable under Policy B and no later run revisits one."""
    import json as _json
    cid = _candidate(conn, "openalex:W7164878913", "openalex",
                     "https://doi.org/10.5281/zenodo.20719927",
                     summary="A study of verifiable disclosure.",
                     payload=_json.dumps({"authors": ["A. Person"]}))
    conn.execute("UPDATE frontier_candidates SET title=? WHERE candidate_id=?",
                 ("Earned Trust", cid))
    conn.commit()

    seen = {}
    from pipeline.kb import ingest_papers as ip
    monkeypatch.setattr(ip, "_fetch_s2_paper",
                        lambda lookup: (None, __import__(
                            "pipeline.kb.ingest_common", fromlist=["x"]).FETCH_ABSENT))
    monkeypatch.setattr(ip, "atomize_paper",
                        lambda conn_, emb, paper, **kw: seen.update(paper) or
                        _seed_atom(conn_, "paper:DOI:10.5281/zenodo.20719927", fa.ENTRY_MODE))

    fa.run_frontier_admit(conn, embedder=object())

    assert _row(conn, cid)["status"] == "materialized"
    assert seen["title"] == "Earned Trust"
    assert seen["abstract"] == "A study of verifiable disclosure."
    # Name only. An OpenAlex author id is not a Semantic Scholar one, and `derive_paper` reads
    # `authorId` to mint `who_id = scholar:{id}` — supplying one would assert an identity that
    # does not exist, so the honest `paper-authors:{id}` placeholder is the right outcome.
    assert seen["authors"] == [{"name": "A. Person"}]
