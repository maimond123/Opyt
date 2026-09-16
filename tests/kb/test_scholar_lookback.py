"""The THIRD lookback window — the paper corpus — and the count-first ask it enables.

Offline: `year_counts` is monkeypatched at the adapter, so no request leaves the process.
"""
from __future__ import annotations

from datetime import date

import pytest

from pipeline.kb import expand, oracles, schema
from pipeline.kb import frontier_sources as fs


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _scholar_oracle(conn, cid="openalex:A5043841592"):
    schema.upsert_entity(conn, cid, name="F. Arnold")
    schema.upsert_oracle(conn, cid, name="F. Arnold")
    conn.commit()
    return cid


# ── the window ───────────────────────────────────────────────────────────────────

def test_the_scholar_selector_follows_the_web_shape_not_the_x_one():
    """A published corpus is durable, like a blog archive and unlike a timeline, so a short
    default would truncate a body of work rather than trim a stream. No ceiling either — what
    bounds this pull is embed spend per work, and the pull is abstract-only."""
    assert expand.SCHOLAR_LOOKBACK_PRESETS["all"] is None       # 'all' is a real preset
    assert "6mo" not in expand.SCHOLAR_LOOKBACK_PRESETS         # no short default to fall into


def test_an_unknown_scholar_preset_widens_and_an_unknown_x_preset_refuses():
    """The asymmetry is deliberate. On this selector None means "no lower bound", so a
    fallthrough WIDENS. On the X selector None means the adapter's own 183-day default, so a
    fallthrough there would silently turn a request for a narrow window into the widest pull."""
    assert oracles._scholar_since("nonsense") is None

    with pytest.raises(ValueError):
        oracles._x_since("nonsense")


def test_the_report_names_three_windows_and_never_collapses_them():
    """One datetime can be right for at most one of an ephemeral stream, a durable web archive
    and a published corpus."""
    rep = oracles._lookback_report(None, None, None)

    assert rep["scholar"] == "whole corpus"
    assert rep["web"] == "full archive"
    assert "6-month default" in rep["x"]
    assert {"x_since", "web_since", "scholar_since"} <= set(rep)


# ── count-first ──────────────────────────────────────────────────────────────────

def test_the_counts_turn_the_presets_into_real_numbers(conn, monkeypatch):
    """The thing neither the `x` nor the `web` selector can do. One free `group_by` returns the
    whole distribution, so the question reads "928 papers; the last 2 years is 28"."""
    cid = _scholar_oracle(conn)
    y = date.today().year
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "year_counts",
                        lambda self, aid, *, topics=None: {y: 10, y - 1: 18, y - 3: 74, y - 8: 826})

    counts = oracles.scholar_year_counts(conn, cid)

    assert counts["total"] == 928
    assert counts["by_window"]["2yr"] == 28          # this year + last
    assert counts["by_window"]["5yr"] == 102         # + the one 3 years back
    assert counts["by_window"]["10yr"] == 928
    assert counts["openalex_id"] == "A5043841592"


def test_a_failed_count_degrades_to_the_blind_presets(conn, monkeypatch):
    """FAIL-SAFE, and this is the flip condition the plan names: a count is what makes the
    question better, never what makes it possible."""
    cid = _scholar_oracle(conn)

    def _boom(self, aid, *, topics=None):
        raise RuntimeError("openalex is down")
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "year_counts", _boom)

    assert oracles.scholar_year_counts(conn, cid) is None
    # And the report still answers, with presets and no numbers.
    rep = oracles._lookback_report(None, None, None, None)
    assert "scholar_ask" not in rep and rep["scholar"] == "whole corpus"


def test_a_person_with_no_openalex_id_is_never_counted(conn, monkeypatch):
    """No network call at all — there is nothing to ask about."""
    schema.upsert_entity(conn, "x:user:1", name="Someone")
    schema.upsert_oracle(conn, "x:user:1", name="Someone")
    conn.commit()
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "year_counts",
                        lambda self, aid, *, topics=None: pytest.fail("must not be called"))

    assert oracles.scholar_year_counts(conn, "x:user:1") is None


def test_the_ask_carries_the_numbers_when_they_are_known(conn, monkeypatch):
    cid = _scholar_oracle(conn)
    y = date.today().year
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "year_counts",
                        lambda self, aid, *, topics=None: {y: 10, y - 1: 18, y - 8: 900})

    rep = oracles._lookback_report(None, None, None, oracles.scholar_year_counts(conn, cid))

    assert "928 papers" in rep["scholar_ask"]
    assert "2yr is 28" in rep["scholar_ask"]


def test_the_ingest_path_finds_the_author_id_without_a_network_call(conn, monkeypatch):
    """A missing or failed COUNT must not also cost the pull — the id is on disk either way."""
    cid = _scholar_oracle(conn)
    monkeypatch.setattr(fs.OpenAlexWorksAdapter, "year_counts",
                        lambda self, aid, *, topics=None: pytest.fail("the ingest path must not count"))

    assert oracles._openalex_id(conn, cid) == "A5043841592"
