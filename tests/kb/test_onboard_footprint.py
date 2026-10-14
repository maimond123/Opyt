"""onboard_footprint — the router from a confirmed Oracle's found sources to their first pull.

Each source routes exactly once: a blog, Substack or GitHub straight to its walk (no authorship
gate: a wrong source is forgotten after the fact), scholar/orcid not-built.
`begin` is a recorder here (no walk, no network). GitHub's repo-vs-account split is the walk's,
covered in `test_resumable_archives.py`.
"""

from __future__ import annotations

import pytest

from pipeline.kb import resolve, schema
from pipeline.oracles.sources.first_pull import onboard_footprint


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    schema.upsert_entity(c, "x:user:7", name="Simon")   # the Oracle
    yield c
    c.close()


@pytest.fixture()
def begun(monkeypatch):
    """A `begin` that records each walk it is asked to start."""
    calls = []
    monkeypatch.setattr(resolve, "resolve_entities", lambda conn: None)

    def begin(conn, url, **kw):
        calls.append({"url": url, **kw})
        return {"archive_started": True}
    begin.calls = calls
    return begin


def _never(*a, **k):
    raise AssertionError("no walk may begin here")


def _src(source_type, url, *, handle=None):
    return {"source_type": source_type, "url": url,
            "metadata": {"handle": handle} if handle else {}}


# ── blog / substack / github → its walk ──────────────────────────────────────

def test_a_blog_begins_its_walk_with_its_handle(conn, begun):
    src = _src("blog", "https://simonwillison.net", handle="simonwillison")
    out = onboard_footprint(conn, "x:user:7", [src], begin=begun, author_name="Simon")

    assert out["ingested"] == 1
    assert out["results"][0]["stats"] == {"archive_started": True}
    assert begun.calls == [{"url": "https://simonwillison.net", "author_name": "Simon",
                            "since": None, "handle": "simonwillison"}]


def test_a_substack_begins_its_walk(conn, begun):
    src = _src("substack", "https://carol.substack.com", handle="carol")
    out = onboard_footprint(conn, "x:user:7", [src], begin=begun, author_name="Carol")

    assert out["ingested"] == 1
    assert begun.calls[0]["url"] == "https://carol.substack.com"
    assert begun.calls[0]["handle"] == "carol"


def test_github_begins_its_walk(conn, begun):
    out = onboard_footprint(conn, "x:user:7", [_src("github", "https://github.com/simonw")],
                            begin=begun)
    assert out["ingested"] == 1 and begun.calls[0]["url"] == "https://github.com/simonw"


def test_a_github_url_naming_nothing_is_skipped(conn):
    out = onboard_footprint(conn, "x:user:7", [_src("github", "https://github.com")],
                            begin=lambda *a, **k: None)
    assert out["ingested"] == 0 and out["results"][0]["action"] == "skipped"


# ── boundaries ─────────────────────────────────────────────────────────────────

def test_a_source_that_raises_is_an_error_and_the_rest_go_on(conn, begun):
    def begin(conn, url, **kw):
        if "slow" in url:
            raise TimeoutError("read timed out")
        return begun(conn, url, **kw)
    out = onboard_footprint(conn, "x:user:7", [_src("blog", "https://slow.dev"),
                                               _src("github", "https://github.com/simonw")],
                            begin=begin)
    assert out["errors"] == 1 and out["ingested"] == 1


def test_isolate_false_lets_the_exception_out(conn):
    def begin(conn, url, **kw):
        raise TimeoutError("read timed out")
    with pytest.raises(TimeoutError):
        onboard_footprint(conn, "x:user:7", [_src("blog", "https://slow.dev")], begin=begin,
                          isolate=False)


def test_scholar_is_not_built(conn):
    """`not-built`, not `deferred`. `deferred` means bounded work something will resume — an
    `oracle(action='ingest')` X pull that hit a rate window, a budget-capped Frontier pair — and
    nothing resumes scholar/orcid. One word, one meaning, or the user-facing copy promises a
    continuation no rail performs."""
    src = _src("scholar", "https://scholar.google.com/citations?user=abc", handle="abc")
    out = onboard_footprint(conn, "x:user:7", [src], begin=_never)
    assert out["not_built"] == 1
    assert out["results"][0]["action"] == "not-built"


def test_missing_oracle_id_is_an_error(conn):
    out = onboard_footprint(conn, "", [_src("github", "https://github.com/x")], begin=_never)
    assert "error" in out
