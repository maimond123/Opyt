"""Repo-wide test guards.

Why this exists: migrating the X/long-form image paths off `describe_image` onto the
OCR cascade broke the seam nine test files were faking. Those tests patched
`pipeline.processing.describe_images.describe_image` (both since deleted — the cascade lives at
`pipeline.chart_read` now); the cascade calls `pipeline.llm_client.call`,
so the patches stopped intercepting and the tests fell through to REAL HTTP against fake URLs
(`https://pbs/0.jpg`) — where they HUNG rather than failed. A hang is the worst possible signal: it
looks like a slow suite, it can spend real money, and it names neither the test nor the reason.

So a unit test that reaches `llm_client.call` now fails LOUDLY and says what to patch. This is the
same rule the pipeline itself follows — a failure must be visible at the moment it happens, never
degrade into something indistinguishable from working (the fail-safe rule; cf. an earlier OCR
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

Two seams, not one. The first version guarded only `urlopen`, and the pipeline also
speaks `requests` — `ingest_papers._download_pdf`, the scrapers, the GitHub API. A new test calling
the real `atomize_paper` sailed straight through and spent 44 SECONDS pulling a PDF for a fake arXiv
id. Exactly the failure this file exists to prevent, one transport over: a guard that covers most of
the boundary reads as a guard that covers the boundary. `Session.request` is the chokepoint every
`requests.get/post/Session` call funnels through, so patching it there keeps the same
whoever-fakes-the-network-wins rule for tests that stub `requests.get` in their own body.
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request

import pytest


@pytest.fixture(autouse=True)
def _fresh_mcp_session():
    """Every test leaves no MCP session open: a test that opened a client session through
    `tests/mcp_session.py` has it closed here, so a test can hold one across calls the way a host
    holds one across a conversation."""
    yield
    if "tests.mcp_session" in sys.modules:
        sys.modules["tests.mcp_session"].close_all()


@pytest.fixture(autouse=True)
def _a_keyed_install(monkeypatch):
    """Every test runs as an install that holds an OpenRouter key, unless it removes it. A key is
    what turns models on (`credentials.model_access`), and `opyt_core` loads `~/.opyt/.env` at
    import, so without this a test's answer would depend on whether the developer running it has
    a key of their own. The value is fake: the network guard below stops any real call. A test of
    the free tier deletes it (`monkeypatch.delenv("OPENROUTER_API_KEY")`)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


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
def _chart_model_pinned(monkeypatch):
    """`read_image` checks its model against the network-backed catalog on first use — a seam
    the files faking `llm_client.call` never patch, so unpinned it trips the network guard
    below. Pin the model per test; a test exercising the check calls
    `_reset_stage_for_tests()` (which clears the pin) and patches
    `model_routing.chart_model_unroutable` itself."""
    from pipeline import chart_read
    monkeypatch.setattr(chart_read, "_RESOLVED", chart_read.CHART_MODEL)


@pytest.fixture(autouse=True)
def _rail_preflight_pinned_open(monkeypatch):
    """A rail that can call a model runs `models_unroutable` before spending, and its preflight is
    a network-backed catalog check — a seam the rail tests (which fake locks, budgets, and
    ingesters) never patch. Pinned OPEN per test. The rails bind the name at import
    (`from ... import models_unroutable`), so the pin must land on each rail module, not on
    `model_routing`; `tests/kb/test_rail_gate.py` reaches the real function through
    `model_routing`, which stays unpinned for exactly that reason.

    Frontier's searches and admissions run in the scrapers process (`background`) and
    publishing, under its gate."""
    from pipeline import background
    monkeypatch.setattr(background, "models_unroutable", lambda rail: None)


@pytest.fixture(autouse=True)
def _robots_txt_pinned_absent(monkeypatch):
    """Every web read asks its origin's robots.txt first (`robots.check`, from
    `host_access.request`): a network read the tests that fake a page never patch, so unpinned
    it trips the network guard below. Pinned to "this site has no robots.txt", with an empty
    cache per test; `tests/test_robots.py` patches `robots._read` itself."""
    from pipeline import robots
    monkeypatch.setattr(robots, "_read", lambda origin, user_agent: robots.ALLOW_ALL)
    monkeypatch.setattr(robots, "_CACHE", {})


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_llm: test intentionally issues a real (paid) network call")
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
            f"the code ACTUALLY calls (e.g. `chart_read.read_image` for image reads — see the "
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

    Found by looking: `pull_runs` and `pull_run_oracles` existed in the real
    `~/.opyt/opyt.db`, created by a suite run. DDL only and zero rows, so nothing was corrupted —
    but a test that can CREATE a table there can write one, and the next thing to reach for that
    store was a "completely new user" onboarding run, which would have found it pre-seeded.

    The reachable path was new and ordinary: `search` and `aggregate` now carry a completion
    notice, the rider flow (`mcp_server/riders.py`) opens the store to look for one, and no
    fixture stood between that and the user's home. Nothing about it was exotic, which is the
    point — this is the same class of guard as the network block and the LaunchAgent block: the
    side effect ESCAPES the test process, so it has to be stopped at the boundary rather than
    per-test.

    Session-scoped tmp dir per test, and a test that wants its own (`kb_home`, the service
    conftest's two homes) simply sets it again afterwards — a later `monkeypatch.setenv` wins."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path_factory.mktemp("opyt-home")))


@pytest.fixture(autouse=True)
def _no_resident_service(monkeypatch, tmp_path_factory):
    """No test may touch the machine's LaunchAgents — the same class of guard as the network block
    above, for a side effect that ESCAPES the test process.

    ⚠️ IT LEAVES THE MACHINE CHANGED, WHICH IS WORSE THAN A HANG. `install_worker.install()` writes
    `~/Library/LaunchAgents/com.useopyt.worker.plist` and `launchctl bootstrap`s it — a real,
    resident, KeepAlive process on whoever ran `pytest`, outliving the suite, the checkout and the
    reboot. It became reachable from a unit test the moment consent started installing the worker:
    eight `test_onboard_tool` cases walked straight into it, and nothing about a
    green suite would have said so. Every `onboard` call past the model-provider step reaches it
    now (`pipeline/onboarding/worker_check.ensure`).

    GUARDED AT THE BOUNDARY, NOT AT THE FUNCTION, which is this file's own rule — "the guard sits
    at `urlopen`, the ACTUAL network boundary". Blocking `install()` itself would break the tests
    that legitimately exercise it, and a guard that blocks the layer under test is not a guard, it
    is a second bug. So: the plist path is redirected into a tmp dir and `launchctl` is replaced
    with a success that runs nothing. `install()` and `uninstall()` then execute their real logic
    on a real file that is not the user's, and whoever patches these in their own body still wins.

    `status()` is left ALONE on purpose. It only reads, every caller is fail-safe against it, and
    stubbing it would hide the one question the product must ask before it promises that work
    continues on its own.

    And no test waits for a beat. `repair_or_report` gives a just-installed worker a few seconds
    to beat, and under the stubbed `launchctl` above nothing ever will — so every `onboard` call
    would sleep the full wait. A test about the wait sets its own patience.
    """
    from opyt_core import install_worker
    from pipeline.worker import liveness

    agents = tmp_path_factory.mktemp("launch-agents")
    monkeypatch.setattr(install_worker, "PLIST_PATH", agents / "com.useopyt.worker.plist")
    monkeypatch.setattr(install_worker, "_launchctl",
                        lambda *a: subprocess.CompletedProcess(list(a), 0, "", ""))
    monkeypatch.setattr(liveness, "REPORT_PATIENCE", 0.0)
    # A fresh cache per test: an answer cached under one test's stubs must not reach the next.
    monkeypatch.setattr(liveness, "_REPORT_CACHE", {})
