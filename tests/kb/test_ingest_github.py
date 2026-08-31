"""pipeline.kb.ingest_github — the atom-KB GitHub adapter.

Two entries share ONE repo→atom mapping (`_repo_atom`):
  • sync_github(handles=…)     — a tracked handle's whole archive, entry_mode='oracle-footprint'.
  • github_atom_from_url(url)  — ONE repo an Oracle *referenced*, entry_mode='author_referenced'.

These cover the URL parser and the single-repo entry offline (mocked fetch + fake embedder). The
key invariant is CANONICAL identity: the atom keys on the API's owner login, not the URL's casing,
so a footprint reference dedups against the handle crawl instead of minting a twin. sync_github's
live pagination is proven against the ingestion crawler it reuses.
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import ingest_github as gh
from pipeline.kb import schema


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/ggerganov/llama.cpp", ("ggerganov", "llama.cpp")),
    ("https://github.com/Ggerganov/llama.cpp/blob/master/README.md", ("Ggerganov", "llama.cpp")),
    ("https://www.github.com/openai/whisper.git", ("openai", "whisper")),
    ("http://github.com/a/b?tab=readme", ("a", "b")),
    ("https://github.com/features/copilot", None),   # reserved product page, not an owner
    ("https://github.com/karpathy", None),           # bare profile (one segment)
    ("https://gist.github.com/x/abc", None),         # gist is a different host
    ("https://arxiv.org/abs/2401.1", None),          # not github at all
])
def test_github_owner_repo_parse(url, expected):
    assert gh._github_owner_repo(url) == expected


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _no_owner_profile(monkeypatch):
    """`sync_github` fetches the owner's profile once per handle (`_seed_owner_identity`). Default
    it to 'no profile' so every OTHER test keeps its existing shape and never reaches a real socket;
    the identity tests below override it."""
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh_ing, "_fetch_user", lambda login: None)


def _repo(owner="ggerganov", name="llama.cpp"):
    return {"name": name, "owner": {"login": owner}, "language": "C++",
            "stargazers_count": 60000, "forks_count": 8000, "description": "LLM inference in C/C++",
            "topics": ["llm"], "pushed_at": "2026-01-05T00:00:00Z", "license": {"spdx_id": "MIT"},
            "html_url": f"https://github.com/{owner}/{name}"}


def _patch_repo(monkeypatch, repo):
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: repo)
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: "# readme\nbody")


def test_github_atom_from_url_mints_referenced_artifact(conn, fake_embedder, monkeypatch):
    _patch_repo(monkeypatch, _repo())
    atom_id = gh.github_atom_from_url(conn, fake_embedder,
                                      "https://github.com/ggerganov/llama.cpp")
    assert atom_id == "github:ggerganov/llama.cpp"
    row = conn.execute("SELECT who_id, what_kind, entry_mode, source_type FROM atoms "
                       "WHERE atom_id=?", (atom_id,)).fetchone()
    assert row["who_id"] == "github:ggerganov" and row["what_kind"] == "artifact"
    assert row["entry_mode"] == "author_referenced" and row["source_type"] == "github"


def test_github_atom_from_url_canonical_case(conn, fake_embedder, monkeypatch):
    # URL casing differs from the API's canonical owner login; the atom keys on the API's, so a
    # footprint reference dedups against the tracked-handle crawl instead of minting a twin.
    _patch_repo(monkeypatch, _repo(owner="ggerganov"))
    atom_id = gh.github_atom_from_url(conn, fake_embedder,
                                      "https://github.com/GGERGANOV/llama.cpp")
    assert atom_id == "github:ggerganov/llama.cpp"


def test_github_atom_from_url_idempotent(conn, fake_embedder, monkeypatch):
    _patch_repo(monkeypatch, _repo())
    a = gh.github_atom_from_url(conn, fake_embedder, "https://github.com/ggerganov/llama.cpp")
    b = gh.github_atom_from_url(conn, fake_embedder, "https://github.com/ggerganov/llama.cpp")
    assert a == b == "github:ggerganov/llama.cpp"                       # unchanged snapshot → same id
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (a,)).fetchone()[0] == 1


def test_github_atom_from_url_not_a_repo(conn, fake_embedder):
    assert gh.github_atom_from_url(conn, fake_embedder, "https://github.com/features/x") is None


def test_github_atom_from_url_fetch_failure_returns_none(conn, fake_embedder, monkeypatch):
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: None)    # 404 / network failure
    assert gh.github_atom_from_url(conn, fake_embedder,
                                   "https://github.com/ghost/missing") is None


# ── ARC-1 Job A: sync_github batches the embed across a handle's repos ─────────────

def _readme(marker: str, n: int = 40) -> str:
    """A README long enough to chunk, distinct per repo so vectors don't collapse."""
    return f"# {marker}\n\n" + " ".join(f"{marker}{i:04d}" for i in range(n))


def test_sync_github_batches_embed_across_repos(conn, recording_embedder, monkeypatch):
    # Four repos of one handle. Under the batching sink they pool into ONE flush = ONE embed call;
    # the old per-repo store_atom paid four. Proves Job A without changing fetch (serial GitHub API).
    from pipeline.ingestion.sources import github as gh_ing
    repos = [_repo(owner="acme", name=f"proj{i}") for i in range(4)]
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: repos)
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))

    out = gh.sync_github(conn, recording_embedder, handles=["acme"])
    assert out["added"] == 4 and out["failed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 4
    assert len(recording_embedder.calls) == 1                      # four repos, ONE flush


def test_the_two_entries_stamp_different_modes(conn, recording_embedder, monkeypatch):
    """The two entries share `_repo_atom` and MUST NOT collapse onto one mode.

    A handle sweep is a tracked person's own archive — 'oracle-footprint', the same mode the X and
    Substack footprint sweeps write, and inside HUMAN_ATTESTED. A repo that person merely POINTED
    at is 'author_referenced'. Both are reachable; only the authorship claim differs. This pins the
    2026-08-25 rename: the sweep used to stamp 'crawled', which sat outside HUMAN_ATTESTED and made
    a confirmed Oracle's repos unreachable to every sitting.
    """
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: [_repo(owner="acme", name="own")])
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))
    gh.sync_github(conn, recording_embedder, handles=["acme"])

    _patch_repo(monkeypatch, _repo(owner="other", name="pointed"))
    gh.github_atom_from_url(conn, recording_embedder, "https://github.com/other/pointed")

    modes = dict(conn.execute("SELECT atom_id, entry_mode FROM atoms"))
    assert modes["github:acme/own"] == "oracle-footprint"
    assert modes["github:other/pointed"] == "author_referenced"
    assert "crawled" not in set(modes.values())


def test_sync_github_poison_repo_isolated_not_aborted(conn, recording_embedder, monkeypatch):
    # Behavior CHANGE from the old per-atom store_atom: an embed failure mid-crawl used to RAISE and
    # lose every later repo. Under the sink the bad repo is isolated and the crawl finishes — the two
    # good repos still land, `failed` surfaces the one that didn't.
    recording_embedder.poison = "ZZPOISON"
    from pipeline.ingestion.sources import github as gh_ing
    repos = [_repo(owner="acme", name="good0"), _repo(owner="acme", name="bad"),
             _repo(owner="acme", name="good1")]
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: repos)
    monkeypatch.setattr(gh_ing, "_fetch_readme",
                        lambda owner, name: ("# bad\n\n" + "ZZPOISON " * 20) if name == "bad"
                        else _readme(name))

    out = gh.sync_github(conn, recording_embedder, handles=["acme"])
    ids = {r[0] for r in conn.execute("SELECT atom_id FROM atoms").fetchall()}
    assert ids == {"github:acme/good0", "github:acme/good1"}       # crawl finished past the bad repo
    assert out["added"] == 2 and out["failed"] == 1


# ── Forks: a MISATTRIBUTION filter, not a quality one ─────────────────────────
# A fork's atom is built from the UPSTREAM's README + description and stamped with the forker's
# `who_id`. Live on @willccbb (2026-08-03): `github:willccbb/vllm` carried "A high-throughput and
# memory-efficient inference and serving engine for LLMs" attributed to Will Brown, who did not
# write vLLM — a false fact in a KB whose premise is "credible people's content". 14 of 26 repos
# (54%) on that profile were forks.
#
# The ATOM is dropped; the ACT is kept as an edge (`_record_fork_edge`) — "he forked vLLM" is true,
# "he wrote vLLM" is not, and they used to ride on the same object.

def _fork(owner="acme", name="upstream-thing", stars=0):
    r = _repo(owner=owner, name=name)
    r.update({"fork": True, "stargazers_count": stars})
    return r


def test_forks_are_excluded_from_a_handle_crawl(conn, recording_embedder, monkeypatch):
    from pipeline.ingestion.sources import github as gh_ing
    repos = [_repo(owner="acme", name="mine"), _fork(owner="acme", name="theirs")]
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: repos)
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))

    out = gh.sync_github(conn, recording_embedder, handles=["acme"])
    ids = {r[0] for r in conn.execute("SELECT atom_id FROM atoms").fetchall()}
    assert ids == {"github:acme/mine"}
    assert out["forked"] == 1              # reported, so a dropped count is never read as a regression


def test_fork_skipped_before_the_readme_fetch(conn, recording_embedder, monkeypatch):
    """Ordering matters for cost: a skipped fork must not spend a README call."""
    from pipeline.ingestion.sources import github as gh_ing
    fetched: list[str] = []
    monkeypatch.setattr(gh, "_fetch_handle_repos",
                        lambda handle: [_fork(owner="acme", name="theirs")])
    monkeypatch.setattr(gh_ing, "_fetch_readme",
                        lambda owner, name: fetched.append(name) or _readme(name))

    gh.sync_github(conn, recording_embedder, handles=["acme"])
    assert fetched == []


def test_include_forks_opts_back_in(conn, recording_embedder, monkeypatch):
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh, "_fetch_handle_repos",
                        lambda handle: [_fork(owner="acme", name="theirs")])
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))

    out = gh.sync_github(conn, recording_embedder, handles=["acme"], include_forks=True)
    # `forked` counts EXCLUSIONS, so it stays 0 here — one repo is never in both buckets.
    assert out["added"] == 1 and out["forked"] == 0


def test_star_floor_would_make_both_errors_the_fork_filter_avoids(conn, recording_embedder,
                                                                 monkeypatch):
    """Why the default is `min_stars=0` + fork-filter rather than a star threshold.

    Real shape from @willccbb: `trl` is a FORK with 19 stars — a floor of 5 KEEPS someone else's
    code. `nbabench-data` is HIS OWN with 4 stars — the same floor DROPS his work. Stars measure
    popularity x age; authorship is the question with an exact answer."""
    from pipeline.ingestion.sources import github as gh_ing
    repos = [_fork(owner="acme", name="trl", stars=19),
             {**_repo(owner="acme", name="nbabench-data"), "stargazers_count": 4, "fork": False}]
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: repos)
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))

    ids = {r[0] for r in conn.execute("SELECT atom_id FROM atoms").fetchall()}
    gh.sync_github(conn, recording_embedder, handles=["acme"])
    ids = {r[0] for r in conn.execute("SELECT atom_id FROM atoms").fetchall()}
    assert ids == {"github:acme/nbabench-data"}          # his own kept, the popular fork dropped


# ── A fork costs NO extra API call ────────────────────────────────────────────
#
# The act used to be recorded as a `forked` edge, which cost one single-repo GET per fork (the
# repo-LIST endpoint omits `parent`/`source`). Nothing ever read that edge, and the `edges` table
# was deleted 2026-08-23 — so the call went too. What survives is the part that mattered: the
# fork's ATOM is still dropped, so an upstream README is never attributed to the forker.

def test_a_fork_costs_no_upstream_lookup(conn, recording_embedder, monkeypatch):
    """No `_fetch_repo` patch, deliberately: the fork branch must reach no network at all, and
    tests/conftest.py's live-socket guard turns a regression into a loud failure rather than a
    silent per-fork API bill."""
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh, "_fetch_handle_repos",
                        lambda handle: [_fork(owner="acme", name="vllm")])
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))

    out = gh.sync_github(conn, recording_embedder, handles=["acme"])
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0   # no misattribution
    assert out["forked"] == 1


# ── The owner's declared website: what stops a GitHub footprint ORPHANING ─────
# Measured on @willccbb (rerun8): 17 GitHub atoms and all 14 `forked` edges hung off
# `github:willccbb`, whose `canonical_id` was itself, while the Oracle's canonical was
# `blog:willcb.com`. An identity-scoped "what has this person been working on" missed the whole
# GitHub footprint — while an `edges` row asserted, attested, that it was him.
#
# A CAPTURE gap, not a resolver gap: all three `upsert_entity` calls passed only name+kind, so
# `identity_links` stayed empty. And structural — `resolve._SELF_PLATFORMS` excludes github, so its
# links are `attests`-only, X's are too, and `attests ∩ attests` never merges. Storing GitHub's
# OUTBOUND website is what supplies the `self ∩ attests` intersection the resolver already fires on.

def _profile(login="acme", blog="https://acme.dev"):
    return {"login": login, "blog": blog}


def _patch_owner_profile(monkeypatch, prof):
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh_ing, "_fetch_user", lambda login: prof)


def _crawl(conn, embedder, monkeypatch, repos=None):
    from pipeline.ingestion.sources import github as gh_ing
    repos = repos if repos is not None else [_repo(owner="acme", name="mine")]
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: repos)
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))
    return gh.sync_github(conn, embedder, handles=["acme"])


def _links(conn, entity_id):
    row = conn.execute("SELECT identity_links FROM entities WHERE entity_id=?",
                       (entity_id,)).fetchone()
    return json.loads(row["identity_links"]) if row and row["identity_links"] else []


def test_a_declared_website_is_stored_on_the_owner_entity(conn, recording_embedder, monkeypatch):
    _patch_owner_profile(monkeypatch, _profile(blog="https://acme.dev"))
    _crawl(conn, recording_embedder, monkeypatch)
    assert _links(conn, "github:acme") == ["https://acme.dev"]


def test_github_shares_a_canonical_id_with_the_blog_it_declares(conn, recording_embedder,
                                                                monkeypatch):
    """THE test — the one that pins the whole point. Storing the link is only worth doing if the
    resolver then folds the GitHub footprint into the same person as the blog."""
    from pipeline.kb import resolve
    schema.upsert_entity(conn, "blog:acme.dev", name="Acme", identity_links=["https://acme.dev"])
    _patch_owner_profile(monkeypatch, _profile(blog="https://acme.dev"))
    _crawl(conn, recording_embedder, monkeypatch)

    resolve.resolve_entities(conn)
    canon = dict(conn.execute("SELECT entity_id, canonical_id FROM entities").fetchall())
    assert canon["github:acme"] == canon["blog:acme.dev"]        # one person, not two
    # And it merged for the RIGHT reason: the blog's `self` met github's `attests`.
    assert canon["github:acme"] == "blog:acme.dev"


def test_no_declared_website_writes_no_links_and_the_crawl_is_unaffected(conn, recording_embedder,
                                                                        monkeypatch):
    """The common case. A profile with a blank website must cost nothing and change nothing —
    the per-repo upsert still creates the entity, just with no link to merge on."""
    _patch_owner_profile(monkeypatch, _profile(blog=""))
    out = _crawl(conn, recording_embedder, monkeypatch)
    assert out["added"] == 1
    assert _links(conn, "github:acme") == []


def test_a_failed_profile_fetch_does_not_break_the_crawl(conn, recording_embedder, monkeypatch):
    """404 / rate limit / network on the profile lookup → no link, and every repo still lands.
    Same fail-safe posture as `_fetch_repo`: an identity link is never worth the crawl."""
    _patch_owner_profile(monkeypatch, None)
    out = _crawl(conn, recording_embedder, monkeypatch)
    assert out["added"] == 1 and _links(conn, "github:acme") == []


def test_a_raising_profile_fetch_is_swallowed(conn, recording_embedder, monkeypatch):
    """`_gh_get` swallows its own failures, but the seam must not depend on that staying true."""
    from pipeline.ingestion.sources import github as gh_ing

    def _boom(login):
        raise RuntimeError("github exploded")

    monkeypatch.setattr(gh_ing, "_fetch_user", _boom)
    assert _crawl(conn, recording_embedder, monkeypatch)["added"] == 1


def test_the_per_repo_upserts_do_not_clobber_the_stored_link(conn, recording_embedder, monkeypatch):
    """ORDER IS LOAD-BEARING. `upsert_entity` COALESCEs `identity_links`, so the bare per-repo
    upserts (name+kind only) leave a stored link alone — but only because it landed FIRST. Four
    repos, so the upsert runs four more times after the seed."""
    _patch_owner_profile(monkeypatch, _profile(blog="https://acme.dev"))
    repos = [_repo(owner="acme", name=f"proj{i}") for i in range(4)]
    _crawl(conn, recording_embedder, monkeypatch, repos=repos)
    assert _links(conn, "github:acme") == ["https://acme.dev"]


def test_the_api_login_wins_over_the_crawl_handle(conn, recording_embedder, monkeypatch):
    """GitHub handles are case-insensitive; the entity id is not. The seed must key on the API's
    canonical `login` or it seeds `github:ACME` and the repo loop's `github:acme` stays linkless."""
    from pipeline.ingestion.sources import github as gh_ing
    _patch_owner_profile(monkeypatch, _profile(login="acme", blog="https://acme.dev"))
    monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: [_repo(owner="acme",
                                                                        name="mine")])
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))
    gh.sync_github(conn, recording_embedder, handles=["ACME"])          # crawled by a cased handle
    assert _links(conn, "github:acme") == ["https://acme.dev"]


def test_the_profile_is_fetched_once_per_handle_not_per_repo(conn, recording_embedder, monkeypatch):
    """The cost claim: one extra GitHub request per HANDLE. Unauthenticated GitHub is 60 req/hr,
    so a per-repo fetch would burn the budget on a 26-repo profile."""
    from pipeline.ingestion.sources import github as gh_ing
    seen: list[str] = []
    monkeypatch.setattr(gh_ing, "_fetch_user",
                        lambda login: seen.append(login) or _profile(blog="https://acme.dev"))
    _crawl(conn, recording_embedder, monkeypatch,
           repos=[_repo(owner="acme", name=f"proj{i}") for i in range(4)])
    assert seen == ["acme"]


def test_forks_add_no_chunks(kb_home, tmp_path, recording_embedder, monkeypatch):
    """The test that pins the whole rationale. A fork must not touch ranking: retrieval scores
    CHUNKS, and only atoms make chunks. If a fork ever added one, an upstream project's
    professionally-written README would compete with what the Oracle actually said."""
    from pipeline.ingestion.sources import github as gh_ing
    monkeypatch.setattr(gh_ing, "_fetch_readme", lambda owner, name: _readme(name))

    def _chunks(db_name: str, repos: list) -> int:
        monkeypatch.setattr(gh, "_fetch_handle_repos", lambda handle: repos)
        c = schema.connect(tmp_path / db_name)
        gh.sync_github(c, recording_embedder, handles=["acme"])
        n = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        c.close()
        return n

    own = [_repo(owner="acme", name="mine")]
    assert _chunks("no_fork.db", own) == _chunks("with_fork.db",
                                                 own + [_fork(owner="acme", name="theirs")]) > 0
