"""Repo-wide test guards.

WHY THIS EXISTS (2026-08-02). Migrating the X/long-form image paths off `describe_image` onto the
OCR cascade broke the seam nine test files were faking. Those tests patched
`pipeline.processing.describe_images.describe_image` (both since deleted — the cascade lives at
`pipeline.ocr_cascade` now); the cascade calls `pipeline.llm_client.call`,
so the patches stopped intercepting and the tests fell through to REAL HTTP against fake URLs
(`https://pbs/0.jpg`) — where they HUNG rather than failed. A hang is the worst possible signal: it
looks like a slow suite, it can spend real money, and it names neither the test nor the reason.

So a unit test that reaches `llm_client.call` now fails LOUDLY and says what to patch. This is the
same rule the pipeline itself follows — a failure must be visible at the moment it happens, never
degrade into something indistinguishable from working (CLAUDE.md fail-safe; cf. the 2026-08-01 OCR
outage, silent for weeks because a missing transcript looks exactly like a post with no images).

Tests that genuinely want the network mark themselves:

    @pytest.mark.live_llm          # + the existing skipif-on-no-key guard

The guard sits at `urlopen` — the ACTUAL network boundary — and not at `llm_client.call`. That
choice matters and the first draft got it wrong: blocking `call` broke 35 tests that patch a LOWER
seam (the backend, or `urlopen` itself) and then legitimately exercise `call` end-to-end. A guard
that blocks the layer under test is not a guard, it is a second bug. At `urlopen` the rule is
simply: **whoever fakes the network wins.** A test that patches `urlopen` does so inside its own
body, after this autouse fixture, so its patch takes precedence; only a test that reaches a REAL
socket trips. Both backends (anthropic and openrouter) funnel through `_http_json` → `urlopen`, so
one seam covers every provider.

TWO seams, not one (2026-08-02). The first version guarded ONLY `urlopen`, and the pipeline also
speaks `requests` — `ingest_papers._download_pdf`, the scrapers, the GitHub API. A new test calling
the real `atomize_paper` sailed straight through and spent 44 SECONDS pulling a PDF for a fake arXiv
id. Exactly the failure this file exists to prevent, one transport over: a guard that covers most of
the boundary reads as a guard that covers the boundary. `Session.request` is the chokepoint every
`requests.get/post/Session` call funnels through, so patching it there keeps the same
whoever-fakes-the-network-wins rule for tests that stub `requests.get` in their own body.
"""
from __future__ import annotations

import atexit
import urllib.request

import pytest


@pytest.fixture(scope="session", autouse=True)
def _stats_file_never_resolves_to_the_real_store(tmp_path_factory):
    """A test process must never write ~/.opyt/api_stats.json. It used to, and it wiped it.

    THE BUG (found 2026-08-16, reproduced on main): one `pytest` run took a $42.00 lifetime spend
    figure to $0.00. Three pre-existing pieces composed into it, none wrong alone —
    `atexit.register(flush_stats)` at import; the per-test `isolated_stats` fixtures resetting BOTH
    the path override to None AND `_STATS` to `_fresh_stats()` at teardown; and `flush_stats` then
    writing that empty `_STATS` to whatever `_stats_file()` resolved to, which after teardown is
    the REAL path. Nothing failed and nothing was logged. The only evidence was a spend history
    that had quietly become zero — the same shape as every other bug this suite hunts.

    ⚠️ WHY NOT REDIRECT `$OPYT_HOME` FOR THE SESSION, which would sandbox everything at once:
    MEASURED, IT BREAKS THE SUITE. `config_path()` resolves $OPYT_CONFIG -> $OPYT_HOME/settings.yaml
    -> repo default, the repo has no settings.yaml, and the packaged default does not declare the
    roles the tests use. Moving the home moves the CONFIG, and role resolution dies. So this
    redirects the ONE file that gets written, and nothing else.

    Two layers, because they fail independently. `_stats_file` is rebound so the "no override"
    state resolves into a session sandbox instead of the real store — this survives a per-test
    fixture resetting the override to None, which is the exact hole. And the atexit flush is
    unregistered so no write happens at process exit at all.

    ⚠️ `_stats_file` is deliberately NOT restored. `atexit` handlers run AFTER session teardown,
    so restoring it here would re-arm the very write this exists to prevent. Nothing runs after a
    session fixture's teardown except process exit, so leaving it rebound costs nothing.

    ⚠️ THIS PATCHES `llm_spend`, NOT `llm_client` (the pricing/accounting/stats-file code moved to
    `pipeline/llm_spend.py` in the step-7 module split, 2026-08-16). `llm_client.flush_stats` /
    `llm_client._stats_file` still resolve — `llm_client` re-exports them — but a re-export can only
    forward READS live; it cannot make a WRITE land in the other module's namespace. `_load_stats_once`
    and `flush_stats` both live in `llm_spend.py` and call `_stats_file()` there by bare name, so
    patching `llm_client._stats_file` would silently stop mattering: this fixture exists specifically
    to prevent the failure mode this file's docstring describes, so it patches the real owner.
    """
    from pipeline import llm_spend

    sandbox = tmp_path_factory.mktemp("api-stats") / "api_stats.json"

    def _sandboxed_stats_file():
        # The per-test override still wins, so `isolated_stats` keeps working unchanged; only the
        # UNSET case is redirected, and that is the case that used to hit the real store.
        return llm_spend._STATS_FILE_OVERRIDE or sandbox

    llm_spend._stats_file = _sandboxed_stats_file
    atexit.unregister(llm_spend.flush_stats)
    yield


@pytest.fixture(autouse=True)
def _model_routing_cache_never_resolves_to_the_real_store(tmp_path_factory, monkeypatch):
    """`model_routing.surviving_orgs` answers from `$OPYT_HOME/model_routing_cache.json` before it
    fetches — with no override that is the REAL user cache, so an offline test's verdict would
    depend on whatever the live deny-list looked like the last time a rail ran. Same shape as
    `_stats_file` above, reads instead of writes. Function-scoped and fresh per test: successful
    fake-endpoint fetches WRITE the cache unconditionally, and a session-shared file would let one
    test's write become another test's cached verdict. The network guard below still covers the
    fetch a miss falls through to."""
    from pipeline import model_routing
    p = tmp_path_factory.mktemp("mr-cache") / "model_routing_cache.json"
    monkeypatch.setattr(model_routing, "_cache_path", lambda: p)


@pytest.fixture(autouse=True)
def _refusal_markers_never_reach_the_real_store(tmp_path_factory, monkeypatch):
    """A test must never stamp `~/.opyt/<rail>_budget_refused`. It did, on the first run after the
    marker was introduced (2026-08-30): three markers appeared in the real store, one per rail
    whose gate a test happened to cross.

    That is not a stray write, it is a FALSE PAUSE — `rail_budgets.paused_today()` reads exactly
    these files, so a `pytest` run would have made the next real `search` announce that three rails
    were paused when none of them were. The same class of harm as the `api_stats.json` wipe above:
    invisible, and it corrupts the state a notice reads rather than failing anything.

    Same shape as that fixture, and same reason for the shape — redirect the ONE path that gets
    written, never `$OPYT_HOME`, which is measured to break the suite. Function-scoped and fresh
    per test: mtime is the whole record, so one test's stamp must not read as another's pause.

    Patching `rail_runtime.refusal_marker` covers BOTH sides at once, the writer in
    `rail_budget_exhausted` and the reader in `rail_budgets._refused_today`, because both reach it
    by module attribute rather than a from-import. Any test wanting a real marker writes it under
    its own `kb_home` and patches nothing — those still work, because they patch nothing here."""
    from pipeline.kb import rail_runtime
    sandbox = tmp_path_factory.mktemp("refusal-markers")
    monkeypatch.setattr(rail_runtime, "refusal_marker",
                        lambda rail: sandbox / f"{rail}_budget_refused")


@pytest.fixture(autouse=True)
def _ocr_resolution_pinned(monkeypatch):
    """`read_image` resolves its model through the network-backed catalog on first use — a seam
    the nine files faking `llm_client.call` never patch, so unpinned it trips the network guard
    below. Pin the primary per test; a test exercising resolution calls
    `_reset_stage_for_tests()` (which clears the pin) and patches
    `model_routing.resolve_ocr_model` itself."""
    from pipeline import ocr_cascade
    monkeypatch.setattr(ocr_cascade, "_RESOLVED", ocr_cascade.OCR_MODEL)


@pytest.fixture(autouse=True)
def _rail_preflight_pinned_open(monkeypatch):
    """Every rail runs `models_unroutable` before spending, and its preflight is a network-backed
    catalog check — a seam the rail tests (which fake locks, budgets, and ingesters) never patch.
    Pinned OPEN per test. The rails bind the name at import (`from ... import models_unroutable`),
    so the pin must land on each rail module, not on `rail_runtime`; `tests/kb/test_rail_gate.py`
    reaches the real function through `rail_runtime`, which stays unpinned for exactly that
    reason."""
    from pipeline.kb import (bookmark_catchup, curation_catchup, frontier_admit,
                             frontier_execute, oracle_refresh, probe_catchup,
                             sitting_scheduler)
    for mod in (bookmark_catchup, curation_catchup, frontier_admit, frontier_execute,
                oracle_refresh, probe_catchup, sitting_scheduler):
        monkeypatch.setattr(mod, "models_unroutable", lambda rail: None)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_llm: test intentionally issues a real (paid) network call")
    config.addinivalue_line(
        "markers", "real_gate: drive the real content_gate (opts out of the keep-all stub)")
    config.addinivalue_line(
        "markers", "real_triage: drive the real url triage (opts out of the approve-all stub)")
    config.addinivalue_line(
        "markers", "loopback: test talks to a server IT STARTED on 127.0.0.1/::1 (never the net)")


def _is_loopback(url: str) -> bool:
    """Is this URL a request to this machine, on this interface, and nowhere else?"""
    from urllib.parse import urlparse
    host = (urlparse(str(url)).hostname or "").lower()
    return host in ("127.0.0.1", "::1", "localhost")


@pytest.fixture(autouse=True)
def _block_live_network(request, monkeypatch):
    if request.node.get_closest_marker("live_llm"):
        return                                  # opted in — leave the real socket alone

    # ⚠️ `loopback` IS NOT A SECOND `live_llm`. `live_llm` means "this call is real and PAID", and
    # it lifts the guard entirely. `loopback` means "this test started a server on 127.0.0.1 and is
    # talking to it" — the credential channel in opyt_core/local_auth.py is exactly that, and
    # stubbing its socket would leave the security properties (loopback-only bind, nonce, one-shot)
    # untested. So the exemption is SCOPED, not blanket: a `loopback` test that reaches a routable
    # address still fails, which is the property that matters. Marking it `live_llm` instead would
    # have been a lie in the marker name AND would have hidden it from `-m "not live_llm"` runs.
    loopback_ok = request.node.get_closest_marker("loopback") is not None
    _real_urlopen = urllib.request.urlopen

    def _blocked(req, *a, **kw):
        url = getattr(req, "full_url", req)
        if loopback_ok and _is_loopback(url):
            return _real_urlopen(req, *a, **kw)
        # Name the CALLER. `pytrace=False` (below) suppresses the traceback — which is what keeps
        # the failure readable — so without this the message says a live call happened but not from
        # where, and every diagnosis becomes a manual grep. The pipeline frames are the only ones
        # that matter; urllib/llm_client plumbing is noise.
        import traceback
        _plumbing = ("llm_client.py", "circuit_breaker.py", "concurrency.py")
        frames = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                  for f in traceback.extract_stack()
                  if "/pipeline/" in f.filename
                  and not f.filename.endswith(_plumbing)][-4:]
        where = "\n  via " + "\n  via ".join(reversed(frames)) if frames else ""
        # `pytest.fail` raises an OutcomeException — a BaseException, NOT an Exception. That is
        # deliberate and load-bearing. The pipeline is full of fail-safe `except Exception` handlers
        # (a vision failure skips the image, a triage failure approves-all, a breaker counts the
        # error), and an `AssertionError` here got LAUNDERED by exactly those: the first draft of
        # this guard was swallowed by `url_triage`'s catch-all into a silent "approve-all gray"
        # degradation, and its errors also tripped the shared `openrouter` breaker, which then
        # fail-fasted an unrelated live test. A guard whose whole purpose is to make a hidden call
        # VISIBLE must not be catchable by the same handlers that hide things.
        pytest.fail(
            f"LIVE NETWORK CALL from a unit test — {url}{where}\n"
            f"Something reached a real socket through an unpatched seam. Patch the collaborator "
            f"the code ACTUALLY calls (e.g. `ocr_cascade.read_image` for image reads — see the "
            f"`ocr` fixture in tests/kb/conftest.py), or mark the test @pytest.mark.live_llm if "
            f"the call is intended.", pytrace=False)

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    try:
        import requests
    except ImportError:                             # requests is optional at runtime — stay fail-safe
        return
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda self, method, url, *a, **kw: _blocked(url))
