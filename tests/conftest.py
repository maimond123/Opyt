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

import subprocess
import urllib.request

import pytest


def reset_atoms_session() -> None:
    """Reset atom-tool process state between tests without a production-only seam."""
    from mcp_server import atoms_tools

    atoms_tools._SEARCHES = 0
    atoms_tools._FRONTIER_NOTICED = False
    atoms_tools._THIN_OFFERED = False
    atoms_tools._OPENED.clear()
    atoms_tools._RECENT.clear()


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
    """A rail that can call a model runs `models_unroutable` before spending, and its preflight is
    a network-backed catalog check — a seam the rail tests (which fake locks, budgets, and
    ingesters) never patch. Pinned OPEN per test. The rails bind the name at import
    (`from ... import models_unroutable`), so the pin must land on each rail module, not on
    `rail_runtime`; `tests/kb/test_rail_gate.py` reaches the real function through `rail_runtime`,
    which stays unpinned for exactly that reason.

    `frontier_execute` and `curation_catchup` are absent because neither calls a model at all.
    `frontier_execute`'s preflight came off with its ceiling on 2026-09-04; `curation_catchup`'s
    came off on 2026-09-05, once its collectors were traced and none reached an LLM. Both stay
    LABELLED rails, so a future paid change puts them back on this list —
    `test_a_free_rail_is_labelled_but_gets_NO_ceiling` is what fails if one grows a preflight."""
    from pipeline.kb import (bookmark_catchup, frontier_admit, oracle_refresh, probe_catchup,
                             sitting_scheduler, substack_saved_catchup)
    for mod in (bookmark_catchup, frontier_admit, oracle_refresh, probe_catchup,
                sitting_scheduler, substack_saved_catchup):
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


@pytest.fixture(autouse=True)
def _never_the_real_data_home(monkeypatch, tmp_path_factory):
    """No test may read or write the user's own `~/.opyt`.

    ⚠️ THE CLAIM WAS ALREADY WRITTEN DOWN AND WAS NOT TRUE. `tests/kb/conftest.py`'s header says
    its `$OPYT_HOME` fixture means "a test never touches the real `~/.opyt`" — but `kb_home` is
    opt-in, so it only held for tests that asked for it. Everything else resolved `opyt_home()`
    to the live store.

    Found on 2026-09-14 by looking: `pull_runs` and `pull_run_oracles` existed in the real
    `~/.opyt/opyt.db`, created by a suite run. DDL only and zero rows, so nothing was corrupted —
    but a test that can CREATE a table there can write one, and the next thing to reach for that
    store was a "completely new user" onboarding run, which would have found it pre-seeded.

    The reachable path was new and ordinary: `search` and `aggregate` now carry a completion
    notice, `_attach_pull_notice` opens the store to look for one, and no fixture stood between
    that and the user's home. Nothing about it was exotic, which is the point — this is the same
    class of guard as the network block and the LaunchAgent block: the side effect ESCAPES the
    test process, so it has to be stopped at the boundary rather than per-test.

    Session-scoped tmp dir per test, and a test that wants its own (`kb_home`, the service
    conftest's two homes) simply sets it again afterwards — a later `monkeypatch.setenv` wins."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path_factory.mktemp("opyt-home")))


@pytest.fixture(autouse=True)
def _pulls_run_inline(monkeypatch):
    """No test may leave a real Oracle pull running on a thread.

    Same class of guard as the network block and the LaunchAgent block above, and for the same
    reason: the side effect ESCAPES the test. Since `oracle(action='ingest')` stopped waiting for
    its own pull, `_ingest` spawns `_run_pull` on a daemon thread — which outlives the test that
    started it, wakes up after monkeypatching has been torn down, and then calls the REAL
    `_ingest_oracle` against the REAL network with whatever `$OPYT_HOME` the next test has set.

    Inline is also what makes the existing assertions meaningful. `_ingest` reports a FINISHED run
    when the record is already closed by the time it returns, which under this seam is always —
    so a test that drives `_ingest` still gets the full report and is testing the pull rather
    than the announcement. The handful of tests that are about the announcement itself replace
    this fixture with a no-op spawn.

    Same shape and the same one job as `onboard_tools._spawn`'s seam."""
    from mcp_server import oracle_tools
    monkeypatch.setattr(oracle_tools, "_spawn", lambda target: target() and None)


@pytest.fixture(autouse=True)
def _no_resident_service(monkeypatch, tmp_path_factory):
    """No test may touch the machine's LaunchAgents — the same class of guard as the network block
    above, for a side effect that ESCAPES the test process.

    ⚠️ IT LEAVES THE MACHINE CHANGED, WHICH IS WORSE THAN A HANG. `install_worker.install()` writes
    `~/Library/LaunchAgents/com.useopyt.worker.plist` and `launchctl bootstrap`s it — a real,
    resident, KeepAlive process on whoever ran `pytest`, outliving the suite, the checkout and the
    reboot. It became reachable from a unit test the moment consent started installing the worker
    (`onboard_tools._follow_consent_with_a_worker`): eight `test_onboard_tool` cases walked
    straight into it, and nothing about a green suite would have said so.

    GUARDED AT THE BOUNDARY, NOT AT THE FUNCTION, which is this file's own rule — "the guard sits
    at `urlopen`, the ACTUAL network boundary". Blocking `install()` itself would break the tests
    that legitimately exercise it, and a guard that blocks the layer under test is not a guard, it
    is a second bug. So: the plist path is redirected into a tmp dir and `launchctl` is replaced
    with a success that runs nothing. `install()` and `uninstall()` then execute their real logic
    on a real file that is not the user's, and whoever patches these in their own body still wins.

    `status()` is left ALONE on purpose. It only reads, every caller is fail-safe against it, and
    stubbing it would hide the one question the product must ask before it promises that work
    continues on its own.
    """
    from opyt_core import install_worker

    agents = tmp_path_factory.mktemp("launch-agents")
    monkeypatch.setattr(install_worker, "PLIST_PATH", agents / "com.useopyt.worker.plist")
    monkeypatch.setattr(install_worker, "_launchctl",
                        lambda *a: subprocess.CompletedProcess(list(a), 0, "", ""))
