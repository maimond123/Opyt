"""Shared fixtures for the atom-KB tests.

`kb_home` sandboxes the WHOLE store (its one DB file) under a tmp dir via
`$OPYT_HOME`, so a test never touches the real `~/.opyt`.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def kb_home(tmp_path, monkeypatch):
    """Point $OPYT_HOME at a tmp dir so schema.connect()/opyt_db() sandbox."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _x_meter_is_per_test():
    """The X request budget must not leak ACROSS tests.

    `x_rate._RATE_STATE` is a module global and the file behind it is per-`$OPYT_HOME`
    (sandboxed per test), so the dict is the one thing that survives — and since the backfill
    divides the bucket by what the meter says, a test that leaves it spent silently defers the
    next test's pairs. Cleared on both sides so the order tests run in cannot matter."""
    from pipeline.platforms.x import rate as x_rate
    x_rate._RATE_STATE.clear()
    yield
    x_rate._RATE_STATE.clear()


@pytest.fixture(autouse=True)
def ocr(monkeypatch):
    """Fake `chart_read.read_image` — the image seam.

    Autouse, so no atom-KB test can reach a real image model by omission. Requesting `ocr` in a
    signature yields this same instance, so a test that wants to assert on the reads just names it;
    a test that merely happens to ingest a post containing an image gets a deterministic fake for
    free instead of a live call. That asymmetry is deliberate: the failure being prevented is a
    test that never mentions images at all silently paying for them.

    The X tweet-media path and the long-form markdown path both run the OCR cascade, so patching
    anything else intercepts nothing and the call falls through to real HTTP (which is why
    `tests/conftest.py` blocks live `llm_client.call` — a fall-through must fail by name, not hang).

    `ocr.calls` is every URL read, in dispatch order. `ocr.respond(fn)` swaps the behavior, where
    `fn(url, context) -> MediaRead | None` and None means the read FAILED (the poison-value rule:
    caller must not cache it). The default returns document-substance text echoing the URL."""
    from pipeline import chart_read

    class _Ocr:
        def __init__(self):
            self.calls: list[str] = []
            self._fn = lambda url, context: chart_read.MediaRead(f"desc({url})", "chart")

        def respond(self, fn):
            self._fn = fn
            return self

        def _read(self, url, *, context=""):
            self.calls.append(url)
            return self._fn(url, context)

    spy = _Ocr()
    monkeypatch.setattr(chart_read, "read_image", spy._read)
    return spy



@pytest.fixture()
def no_venue(monkeypatch):
    """This site is not a research venue — the answer OpenAlex gives for 14 of 15 real blog hosts.

    Requested explicitly, never autouse. Rooting a site URL asks OpenAlex once whether the host
    resolves to a single source (`identify.openalex_root`), so a test that roots one either stubs
    that answer or trips the live-network guard. Making it automatic would hide the call from the
    very tests that exercise the blog path, and a test that WANTS the venue path simply does not
    ask for this."""
    from pipeline.platforms.papers import openalex
    monkeypatch.setattr(openalex.OpenAlexClient, "sources_by_name", lambda self, name: [])


@pytest.fixture()
def no_deep_probe(monkeypatch):
    """This page declares no DOI — the answer a real fetch of a blog post gives.

    `hopper.preview` pays one bounded page read whenever a reference would fall through to the
    `article` catch-all, to check whether the page declares `citation_doi` and is really a paper.
    Any test that hoppers an article url reaches a live socket without this. Named rather than
    autouse, same contract as `no_venue`: a test that is ABOUT the probe asks for
    something else instead.
    """
    from pipeline.kb import link_router
    monkeypatch.setattr(link_router, "classify_link_deep", lambda url: None)



@pytest.fixture(autouse=True)
def _openalex_pacing_off(monkeypatch):
    """Courtesy pacing is a LIVE-NETWORK property, and this suite has no network — `tests/conftest`
    fails any test that reaches a real socket. Left on, `lookup._openalex_read` sleeps its
    declared second per read and the paper tests alone spent ~6s asleep proving nothing. arXiv's
    3 seconds would cost more.

    The pacing itself is covered directly, in `test_ingest_papers.py`, against the clock rather
    than against the wall.
    """
    from pipeline.platforms.papers import openalex
    from pipeline.platforms.papers import arxiv
    from pipeline.platforms.papers import lookup as paper_lookup
    monkeypatch.setattr(openalex.OpenAlexClient, "min_interval_s", 0.0)
    monkeypatch.setattr(arxiv.ArxivClient, "min_interval_s", 0.0)
    monkeypatch.setattr(paper_lookup._OPENALEX_PACE, "next_at", 0.0)
    monkeypatch.setattr(paper_lookup._ARXIV_PACE, "next_at", 0.0)
