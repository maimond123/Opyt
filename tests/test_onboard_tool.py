"""
tests/test_onboard_tool.py

`onboard` — the thin orchestrator. The properties pinned here are the ones that make it safe
rather than the ones that make it work:

  • NO ARGUMENT MAY EVER CARRY A CREDENTIAL. Secrets travel over loopback; decisions travel
    over chat. A test asserts the signature, so adding an `api_key=` parameter fails the suite.
  • An UNFUNDED OpenRouter account BLOCKS (decision 9) rather than passing through — a store
    that builds and can never be queried is not worth building.
  • Semantic Scholar is never mentioned (decision 4): AI2 no longer approves third-party key
    requests, so a step telling a user to go get one cannot succeed.
"""

import inspect
import re
import time

import pytest

from opyt_core import readiness
from mcp_server import onboard_tools


class _MCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def onboard(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    # Enrichment spawns a REAL daemon thread that outlives the monkeypatching and reaches
    # `model_routing.preflight` — a live socket, from a unit test, after the stubs are gone.
    # Stubbed here rather than per-test so no onboarding test can start one by omission; the
    # tests that care about it assert on this return value.
    from pipeline.kb import enrichment
    monkeypatch.setattr(enrichment, "start_background",
                        lambda **kw: {"status": "running", "note": "stub"})
    m = _MCP()
    onboard_tools.register_onboard_tools(m)
    return m.tools["onboard"]


def test_registers_a_tool_named_onboard(onboard):
    assert onboard is not None


def test_no_argument_is_ever_a_credential(onboard):
    names = set(inspect.signature(onboard).parameters)
    assert not {n for n in names if "key" in n or "token" in n or "secret" in n}


def _at_missing_key(monkeypatch, *, trial_available: bool = False,
                    hosted: bool = False, trial_result: dict | None = None) -> list:
    """A fresh home whose one required key is absent. Returns the list both acquisitions append
    to, so a test can assert WHICH road was taken and that the other was not.

    `trial_available` defaults to FALSE — the pre-trial world — so every test written before the
    starter allowance keeps testing what it was written to test. A test that wants the shorter
    road asks for it.
    """
    calls = []
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "keys",
                                      "keys": {"openrouter": {"state": "missing", "message": ""},
                                               "ok": False},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools.openrouter_oauth, "acquire",
                        lambda **kw: calls.append("oauth") or {"status": "stored",
                                                               "message": "m"})
    monkeypatch.setattr(onboard_tools.trial, "available", lambda: trial_available)
    monkeypatch.setattr(onboard_tools.trial, "hosted_enabled", lambda: hosted)
    monkeypatch.setattr(onboard_tools.trial, "acquire",
                        lambda **kw: calls.append("trial") or (
                            trial_result or {"status": "stored", "message": "m"}))
    return calls


def test_the_first_call_explains_openrouter_and_opens_nothing(onboard, monkeypatch):
    """⚠️ THE DEFECT THIS SPLIT FIXES (2026-09-09, live on Claude Desktop). `acquire()` opens the
    browser and then blocks for up to five minutes, so the OpenRouter approval page arrived
    before the host had rendered one word — it could not have rendered one, because the tool had
    not returned. The user met a third-party consent screen with no idea what it was.

    The contract is that a bare `onboard()` performs NO side effect in the keys phase. The
    hostname assertion is part of it: the fix is telling the user where they are about to be
    sent, and a reword that drops the destination drops the fix.

    THE STARTER ALLOWANCE INHERITED THIS CONTRACT rather than being exempted from it — see the
    sibling test below. Both roads out of `missing` open a browser, so both must be announced a
    turn before they do."""
    calls = _at_missing_key(monkeypatch)

    out = onboard()

    assert calls == []
    assert out["status"] == "needs_openrouter"
    assert out["phase"] == "keys"
    assert "openrouter.ai" in out["message"]


def test_the_first_call_explains_the_trial_and_opens_nothing(onboard, monkeypatch):
    """The same contract on the shorter road. `trial.acquire` opens a tab and then blocks in the
    loopback capture exactly as `openrouter_oauth.acquire` does, so a bare `onboard()` must
    still perform no side effect — and must still say where the user is about to be sent, which
    for this road is a Google sign-in they did not ask for and have every reason to distrust."""
    calls = _at_missing_key(monkeypatch, trial_available=True)

    out = onboard()

    assert calls == []
    assert out["status"] == "needs_trial"
    assert out["phase"] == "keys"
    assert "google" in out["message"].lower()


def test_start_openrouter_runs_oauth_and_does_not_spend(onboard, monkeypatch):
    calls = _at_missing_key(monkeypatch)
    out = onboard(start="openrouter")
    # OpenRouter is the only key this phase can acquire, and `_phase_keys`' docstring says why a
    # second one cannot go here. Until 2026-09-05 this test also patched `key_paste.acquire` and
    # asserted it was NOT called; `key_paste` is deleted, so the "and alone" half is now
    # structural rather than something a test can catch.
    assert calls == ["oauth"]
    assert out["phase"] == "keys"


def test_an_unrecognized_start_word_opens_nothing(onboard, monkeypatch):
    """Refuse, never guess. Guessing here opens the very tab the split exists to announce."""
    calls = _at_missing_key(monkeypatch)
    out = onboard(start="openroutr")
    assert calls == []
    assert out["status"] == "error"


def test_unfunded_openrouter_blocks_before_curation(onboard, monkeypatch):
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "keys",
                                      "keys": {"openrouter": {"state": "unfunded",
                                                              "message": "add credit"},
                                               "ok": False},
                                      "sources": {}, "consent": {}, "curation": {}})
    out = onboard()
    assert out["status"] == "blocked"
    assert out["phase"] == "keys"
    assert out["openrouter"] == "unfunded"


def _at_broken_key(monkeypatch, state: str) -> list:
    """A home whose stored key is rejected (`dead`) or unverifiable (`unknown`). Returns the list
    `_run_openrouter` appends to, so a test can assert the remedy actually RAN."""
    calls = []
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "keys",
                                      "keys": {"openrouter": {"state": state,
                                                              "message": "rejected"},
                                               "ok": False},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools, "_run_openrouter",
                        lambda: calls.append("openrouter") or {"status": "waiting"})
    return calls


@pytest.mark.parametrize("state", ["dead", "unknown"])
def test_a_rejected_key_can_still_be_replaced(onboard, monkeypatch, state):
    """⚠️ THE REMEDY MUST BE REACHABLE. `readiness` answers a rejected key with "Call `onboard`
    again to approve a fresh one" and `allowance_notice` puts that call in `next_call` — and this
    branch used to refuse it and hand the same sentence back, with no way out. Measured
    2026-09-15 on a revoked key: the notice rendered, the user said yes, setup answered `blocked`.

    `unknown` is here with `dead` because the probe runs through the breaker the failures opened,
    so "could not verify" is the ordinary reading of a key that is failing, not a separate state.
    """
    calls = _at_broken_key(monkeypatch, state)
    out = onboard(start="openrouter")
    assert calls == ["openrouter"], f"{state} + explicit start must run the approval"
    assert out["status"] != "blocked"


@pytest.mark.parametrize("state", ["dead", "unknown"])
def test_a_rejected_key_still_blocks_the_passive_path(onboard, monkeypatch, state):
    """Only an explicit ask opens a browser tab. `onboard()` with no `start` must still refuse,
    for the reason the split exists: a tab that takes over the screen is never a side effect."""
    calls = _at_broken_key(monkeypatch, state)
    out = onboard()
    assert calls == []
    assert out["status"] == "blocked"


def test_an_unfunded_account_is_never_sent_to_approve_again(onboard, monkeypatch):
    """Money is the remedy, not another approval — a second key against the same empty balance is
    the loop `_NEXT_CALL` omits `unfunded` to avoid. Explicit `start` does NOT override this."""
    calls = _at_broken_key(monkeypatch, "unfunded")
    out = onboard(start="openrouter")
    assert calls == []
    assert out["status"] == "blocked"
    assert out["openrouter"] == "unfunded"


def test_the_semantic_scholar_KEY_is_never_advertised(onboard, monkeypatch):
    """AI2 no longer approves third-party key requests, so telling a user to get one sends them
    to a form that will reject them.

    The ban is on the CREDENTIAL, not on the company: the author page is a public URL that
    `oracles._scholar_root` accepts and `_unsupported_root` already recommends by name, and a
    27% share of researchers carry no ORCID (measured 2026-09-08 over 256 authorships), so
    banning the page as well would leave that share of the research root unreachable."""
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "done", "keys": {},
                                      "sources": {"x": False, "substack": False},
                                      "consent": {}, "curation": {"applicable": False}})
    blob = (str(onboard()) + (onboard.__doc__ or "")).lower()
    assert "s2_api_key" not in blob
    for phrase in ("semantic scholar key", "semantic scholar api", "semanticscholar.org/api"):
        assert phrase not in blob


# ── phase 1: a source ───────────────────────────────────────────────────────
#
# These helpers patch derive()'s INPUTS, never derive() itself, so the real phase logic runs.

def _keys_green(monkeypatch):
    monkeypatch.setattr(onboard_tools.onboard_state.readiness, "openrouter",
                        lambda: {"state": "ok", "message": ""})


def _at_sources(monkeypatch, *, connected: bool = False, substack: bool = False, **store):
    """Keys green, nothing connected and an empty store — so the sources phase is outstanding.

    ⚠️ BOTH session probes are stubbed, always. They scan `opyt_session_backends()`, and a test
    that fakes that list to exercise one probe otherwise sends the other one into the machine's
    REAL browser cookie store — where the developer running the suite is quite possibly logged
    into Substack, and the phase then advances for a reason the test never set up."""
    from pipeline.ingestion import x_graphql
    from pipeline.ingestion.sources import substack as sub

    _keys_green(monkeypatch)
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: connected)
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: substack)
    # The "done"-too-early gate verifies a connected Substack session is LIVE, not just present,
    # before it lets the consent/curation flow build on it. Default it to ready so a test that
    # sets up a connected session gets the phase it asked for; the gate's own tests set it False.
    monkeypatch.setattr(sub, "managed_substack_session_ready", lambda **kw: True)
    monkeypatch.setattr(onboard_tools.onboard_state, "_store_sources",
                        lambda: {**onboard_tools.onboard_state._NO_STORE_SOURCES, **store})
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending",
                        lambda live: sorted(live))   # nothing read yet


def test_nothing_connected_offers_every_root_not_just_x(onboard, monkeypatch):
    """⚠️ THE DEFECT THIS PHASE REPLACED. The predecessor asked one question — is X connected —
    with no bypass, so a user who reads blogs or watches research topics never got past it.

    The contract is that all four roots are reachable from this one response, and reachable as
    ANSWERS — every root is now a `source=` word, where three of them used to be prose pointing
    at a different tool. A host that reads `answers` must be able to pass whatever the user says
    without leaving this tool first.
    """
    _at_sources(monkeypatch)

    out = onboard()

    assert out["status"] == "needs_source"
    assert out["phase"] == "sources"
    assert set(out["answers"]) == {"x", "substack", "research", "blog", "skip"}
    msg = out["message"]
    for word in out["answers"]:
        assert f"onboard(source='{word}')" in msg


def test_every_source_word_has_a_handler(onboard):
    """`_SOURCE_WORDS` derives from two dicts plus `skip`, so a word can never be accepted with
    nothing behind it. Asserted rather than guarded: the alternative is a runtime check on a
    keyspace this module owns outright."""
    assert set(onboard_tools._SOURCE_WORDS) == (
        set(onboard_tools._LOGIN_URLS) | set(onboard_tools._NAMED_ROOTS) | {"skip"})
    assert not set(onboard_tools._LOGIN_URLS) & set(onboard_tools._NAMED_ROOTS)


@pytest.mark.parametrize("source,must_name", [("research", "orcid.org"),
                                              ("blog", "add_handles")])
def test_a_named_root_returns_the_call_to_make(onboard, monkeypatch, source, must_name):
    """A root with no account returns the QUESTION to put and the CALL to make with the answer.

    `oracle` and `sitting` own those writes; this phase points at them and never proxies them,
    so what a named root owes the caller is the exact invocation."""
    _at_sources(monkeypatch)

    out = onboard(source=source)

    assert out["status"] == "needs_name"
    assert out["source"] == source
    assert "oracle(action='confirm'" in out["message"]
    assert must_name in out["message"]


def test_no_onboarding_surface_creates_a_standing_watch(onboard, monkeypatch):
    """RULED 2026-09-12, third strike. The rule ("onboarding gets content, not commitments")
    was applied to the blind handoff, then to the ways-in list — and each time the research
    root's subject path kept its watch by exception, on the argument that the USER chose a
    subject. A live transcript showed what that choice really was: the host framed a binary,
    the user picked the word "subject" while thinking "here is what I am into", and nine
    interests became three commitments over an empty store. No exceptions survive: no
    onboarding surface names the watchlist call, any root, any branch."""
    _at_sources(monkeypatch)
    assert "sitting(action='watchlist'" not in onboard(source="research")["message"]
    assert "sitting(action='watchlist'" not in onboard(source="blog")["message"]
    for candidates in (0, 40):
        _all_done(monkeypatch, candidates=candidates)
        assert "sitting(action='watchlist'" not in onboard()["message"]


def test_a_named_root_assigns_the_lookup_to_the_host_not_the_user(onboard, monkeypatch):
    """⚠️ THE 2026-09-12 FINDING. This copy read "Have the user search '<name> ORCID' … and
    paste the URL", and a live host rendered it faithfully as "(I'll need their ORCID or
    OpenAlex profile link, not just a name)" — an identifier demanded from the one party who
    should never be asked for one. The adapter constraint is real — a bare name does not
    resolve — but the party doing the resolving is the host with a web search, never the user.
    The user's vocabulary is topics and names; everything after that is OPYT's job."""
    _at_sources(monkeypatch)

    for source in ("research", "blog"):
        msg = onboard(source=source)["message"].lower()
        assert "web-search" in msg
        assert "have the user search" not in msg
        assert "paste the url" not in msg


@pytest.mark.parametrize("source", ["research", "blog"])
def test_a_named_root_opens_no_browser(onboard, monkeypatch, source):
    """The dispatch splits on `_LOGIN_URLS` membership. A fallthrough would launch a sign-in
    window at a site the user never named — the same harm `_SOURCE_WORDS` refuses an unknown
    word for."""
    from pipeline.ingestion import guided_login

    _at_sources(monkeypatch)
    monkeypatch.setattr(guided_login, "start",
                        lambda url: pytest.fail(f"a named root must open nothing, got {url}"))

    assert onboard(source=source)["status"] == "needs_name"


@pytest.mark.parametrize("source", ["research", "blog"])
def test_a_named_root_records_nothing_and_leaves_the_question_open(
        onboard, monkeypatch, tmp_path, source):
    """Picking a KIND of source is not naming one. Until an `oracles` or `frontier_queries` row
    exists the question really is unanswered, so the phase must stay outstanding — writing the
    skip marker here would retire a question the user is still mid-way through answering."""
    _at_sources(monkeypatch)

    onboard(source=source)

    assert not (tmp_path / "onboard_sources_skipped").exists()
    assert onboard()["phase"] == "sources"


def test_source_x_opens_the_persistent_opyt_profile(onboard, monkeypatch):
    from pipeline.ingestion import browser_cookies as bc, guided_login

    started = []
    _at_sources(monkeypatch)
    monkeypatch.setattr(guided_login, "start",
                        lambda url: started.append(url) or bc.backend_for("chrome"))

    out = onboard(source="x")

    assert started == ["https://x.com/login"]
    assert out["status"] == "awaiting_login"


def test_an_unknown_source_word_is_refused_not_guessed(onboard, monkeypatch):
    """Guessing here would open a browser window at a site the user never named."""
    from pipeline.ingestion import guided_login

    _at_sources(monkeypatch)
    monkeypatch.setattr(guided_login, "start",
                        lambda url: pytest.fail("nothing may open for an unknown word"))

    out = onboard(source="twitter")

    assert out["status"] == "error"


def test_skip_records_the_question_was_put_and_moves_on(onboard, monkeypatch, tmp_path):
    """Same shape as the consent marker: every OTHER way out of this phase is a disk fact, and
    "not yet" leaves none — so without a marker the question repeats forever."""
    _at_sources(monkeypatch)

    out = onboard(source="skip")

    assert (tmp_path / "onboard_sources_skipped").exists()
    assert out["phase"] != "sources"


def test_a_named_blog_alone_gets_the_user_past_the_source_phase(onboard, monkeypatch):
    """No session, no candidates — one `oracles` row, which is all a blog reader ever has."""
    _at_sources(monkeypatch, oracles=True)
    assert onboard()["phase"] != "sources"


def test_an_existing_login_profile_says_how_to_resume(onboard, monkeypatch):
    """A profile on disk does not prove a window is open — OPYT cannot see one — so this branch
    must offer BOTH ways forward. The phrasing is not the contract; the two exits are."""
    from pipeline.ingestion import browser_cookies as bc

    _at_sources(monkeypatch)
    monkeypatch.setattr(bc, "opyt_session_backends", lambda: [bc.backend_for("chrome")])

    msg = onboard()["message"]

    assert "`onboard`" in msg and "onboard(source='x')" in msg


def test_source_x_reports_a_missing_launchable_browser(onboard, monkeypatch):
    from pipeline.ingestion import guided_login
    from pipeline.ingestion.utils import SyncAuthError

    _at_sources(monkeypatch)
    monkeypatch.setattr(guided_login, "start",
                        lambda url: (_ for _ in ()).throw(SyncAuthError("no launchable browser")))

    out = onboard(source="x")

    assert out["status"] == "needs_login"
    assert "launchable browser" in out["message"]


def test_source_substack_opens_the_substack_sign_in_page(onboard, monkeypatch):
    from pipeline.ingestion import browser_cookies as bc, guided_login

    started = []
    _at_sources(monkeypatch)
    monkeypatch.setattr(guided_login, "start",
                        lambda url: started.append(url) or bc.backend_for("chrome"))

    out = onboard(source="substack")

    assert started == ["https://substack.com/sign-in"]
    assert out["status"] == "awaiting_login"


def test_the_local_substack_copy_covers_both_sign_in_methods(onboard, monkeypatch):
    """⚠️ Locally the window opens on the user's OWN machine, so the request leaves their own
    address and the ordinary email path works — password and emailed code both. The warning is
    what makes either method land somewhere OPYT can read: a session finished in the user's
    normal browser is one this tool never sees, and nothing about the failure says so."""
    from pipeline.ingestion import browser_cookies as bc, guided_login

    _at_sources(monkeypatch)
    monkeypatch.setattr(guided_login, "start", lambda url: bc.backend_for("chrome"))

    msg = onboard(source="substack")["message"].lower()

    assert "password" in msg and "code" in msg
    assert "normal browser" in msg


def test_a_substack_session_alone_gets_the_user_past_the_source_phase(onboard, monkeypatch):
    _at_sources(monkeypatch, substack=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    assert onboard()["phase"] != "sources"


@pytest.mark.parametrize("source", ["x", "substack"])
def test_hosted_mints_a_sign_in_link_for_every_source(onboard, monkeypatch, source):
    """Hosted refused `substack` until 2026-09-07, because the login was built for one site.

    Now both sources take the same branch, and the site's own sign-in URL travels with the
    minted capability. A local window must never open on a hosted child either way.
    """
    from pipeline.ingestion import guided_login, hosted_browser

    minted = []
    _at_sources(monkeypatch)
    monkeypatch.setattr(hosted_browser, "enabled", lambda: True)
    monkeypatch.setattr(hosted_browser, "begin_login",
                        lambda site, url: minted.append((site, url)) or f"https://g/login/{site}/n")
    monkeypatch.setattr(guided_login, "start",
                        lambda url: pytest.fail("no local window may open on a hosted child"))

    out = onboard(source=source)

    assert out["status"] == "awaiting_login"
    assert minted == [(source, onboard_tools._LOGIN_URLS[source])]
    assert out["login_url"] == f"https://g/login/{source}/n"


def test_the_hosted_substack_copy_describes_the_same_desktop_as_x(onboard, monkeypatch):
    """Ruled 2026-09-15: Substack adopted X's desktop flow, and the copy describes that
    desktop rather than the deleted guided paste.

    It carries NO warning about the email's link (deleted 2026-09-16 with the page's own copy
    of it). The claim was that tapping the link spends the code's single sign-in, and that
    direction was never measured — only its converse was. A warning asserting a measurement
    nobody took is worse than silence, because the next reader treats it as settled.
    """
    from pipeline.ingestion import hosted_browser

    _at_sources(monkeypatch)
    monkeypatch.setattr(hosted_browser, "enabled", lambda: True)
    monkeypatch.setattr(hosted_browser, "begin_login", lambda site, url: "https://g/login/s/n")

    msg = onboard(source="substack")["message"].lower()

    assert "desktop" in msg and "type the code into that desktop" in msg
    assert "not to tap the link" not in msg and "stops working" not in msg
    assert "password" in msg
    # The paste flow is deleted; copy that guides one would describe a page that no longer
    # exists. And no cause was ever asserted for the old delivery silence that survived
    # retraction — the measured one is the debugger flag, and it is fixed, not advice.
    assert "paste" not in msg and "copy the sign-in link" not in msg
    assert "rate-limited" not in msg and "datacenter" not in msg


def test_the_hosted_x_copy_still_describes_the_desktop(onboard, monkeypatch):
    """X mails confirmation codes, not a bearer link, so it keeps the remote browser."""
    from pipeline.ingestion import hosted_browser

    _at_sources(monkeypatch)
    monkeypatch.setattr(hosted_browser, "enabled", lambda: True)
    monkeypatch.setattr(hosted_browser, "begin_login", lambda site, url: "https://g/login/x/n")

    msg = onboard(source="x")["message"].lower()

    assert "private browser desktop" in msg and "2fa" in msg
    assert "paste" not in msg


def test_a_managed_x_session_advances_past_sources(onboard, monkeypatch):
    _at_sources(monkeypatch, connected=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    assert onboard()["phase"] != "sources"


def test_signature_is_three_decisions_and_nothing_else(onboard):
    """`skip_github` sat here unread from 2026-08-15 to 2026-09-07; `connect_x` became one value
    of `source` rather than a second way to say the same thing.

    `start` joined on 2026-09-09 and passes both bars: `_phase_keys` reads it, and it opens no
    door that another argument already opens — it gates the OpenRouter approval, which nothing
    else can begin."""
    assert set(inspect.signature(onboard).parameters) == {"consent", "source", "start"}


# ── the cookie-scrape consent boundary ──────────────────────────────────────

def test_leaving_the_source_phase_WITHOUT_a_session_grants_no_scrape_consent(
        onboard, monkeypatch, tmp_path):
    """⚠️ THE 2026-08-20 COLD-START FINDING. The four curation collectors read a logged-in
    session, and this marker is what lets the background rail run them unattended. A user whose
    root is a named blog — or who answered `skip` — has authorized no session read at all, so
    passing this phase must not grant it."""
    _at_sources(monkeypatch, oracles=True)

    onboard()

    assert not (tmp_path / "curation_catchup_consent").exists()


def test_connecting_a_session_DOES_grant_the_scrape_consent(onboard, monkeypatch, tmp_path):
    _at_sources(monkeypatch, connected=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    onboard()

    assert (tmp_path / "curation_catchup_consent").exists()


def test_a_present_but_not_live_substack_session_holds_at_done(onboard, monkeypatch, tmp_path):
    """The "done"-too-early guard. `derive` marks Substack connected on cookie PRESENCE, but a
    just-signed-in session can be present without answering an authenticated request yet. Onboarding
    must HOLD — not build the consent question and the curation pull on it and report "nobody".
    Real failure, 2026-09-13: collectors skipped, 36 subscriptions sitting in the account."""
    _at_sources(monkeypatch, substack=True)
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "managed_substack_session_ready", lambda **kw: False)
    ran = []
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda: ran.append(1) or {"status": "ok", "ran": {}})

    out = onboard()

    assert out["status"] == "awaiting_login"
    assert "signed in yet" in out["message"]
    assert ran == []                                           # curation never ran on a dead session
    assert not (tmp_path / "curation_catchup_consent").exists()  # consent not granted on it either


def test_a_verifier_error_does_not_strand_a_user_who_did_sign_in(onboard, monkeypatch, tmp_path):
    """Fail-safe direction: if the readiness check itself throws, the flow must PROCEED, not trap a
    signed-in user behind a broken probe. The collector records a real skip if the session is
    actually dead — it never invents a connection."""
    _at_sources(monkeypatch, substack=True)
    from pipeline.ingestion.sources import substack as sub

    def _boom(**kw):
        raise RuntimeError("probe blew up")
    monkeypatch.setattr(sub, "managed_substack_session_ready", _boom)
    # Takes the platforms argument `_spawn` passes it; a 0-arg stub still passed this test but
    # blew up inside the background thread, which pytest surfaces only as a warning.
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda platforms=None: {"status": "stubbed", "ran": {}})

    out = onboard()

    assert out["status"] != "awaiting_login"
    assert (tmp_path / "curation_catchup_consent").exists()


# ── phase 2: consent ────────────────────────────────────────────────────────

def _at_consent(monkeypatch):
    """Keys green, a source connected, question not yet put."""
    _at_sources(monkeypatch, connected=True)
    # NOT stubbed any more: the two requests are a local sqlite write into `$OPYT_HOME`, so the
    # tests below read the real job rows instead of trusting a monkeypatched boolean.
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 0)
    # ⚠️ STUB THE CURATION RUN. Answering consent falls THROUGH into the curation phase in the
    # same call, so without this every consent test fires a real collector pass — ~9s each of
    # live cookie reads. `_at_curation` restores the real one.
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})


def test_no_consent_argument_asks_and_writes_nothing(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    out = onboard()
    assert out["status"] == "needs_consent"
    # The asymmetry is the whole point of the question, so the prompt must state both halves.
    assert "ONE-TIME" in out["message"] and "RECURRING" in out["message"]
    assert "cannot be turned off" in out["message"].lower()
    assert not (tmp_path / "onboard_consent_asked").exists()


def test_backlog_only_writes_one_marker(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    onboard(consent="backlog")
    assert (tmp_path / "bookmark_catchup_consent").exists()
    assert not (tmp_path / "oracle_refresh_consent").exists()
    assert (tmp_path / "onboard_consent_asked").exists()


def test_none_writes_only_the_asked_marker(onboard, monkeypatch, tmp_path):
    """⚠️ The reason the asked marker exists — 'no to both' must not be re-asked forever."""
    _at_consent(monkeypatch)
    onboard(consent="none")
    assert (tmp_path / "onboard_consent_asked").exists()
    assert not (tmp_path / "bookmark_catchup_consent").exists()


def _jobs(home) -> dict:
    """Every durable rail job the worker would find in this home, keyed by rail."""
    from pipeline.kb.rail_jobs import LOCAL_HOME_ID, RailJobStore
    return {j.rail: j for j in RailJobStore(home / "rail_jobs.db").list_jobs()
            if j.home_id == LOCAL_HOME_ID}


def test_backlog_consent_QUEUES_because_the_work_exists_now(onboard, monkeypatch, tmp_path):
    """The durable end of decision 13: "import my backlog NOW" becomes a due-now job row."""
    _at_consent(monkeypatch)

    out = onboard(consent="backlog")

    job = _jobs(tmp_path)["bookmark_catchup"]
    assert job.due_at <= time.time() and job.started_at is None
    assert out["consent_applied"]["queued"] == ["bookmark_catchup"]


def test_a_queued_rail_name_is_one_the_worker_can_actually_launch(onboard, monkeypatch,
                                                                  tmp_path):
    """A name the registry does not carry is a row no worker ever dispatches, and nothing says so.

    Nothing in production couples the two spellings — `rail_worker` imports `rail_jobs`, so the
    reverse direction cannot exist — which is why the agreement is asserted here, at the call
    site that chooses the name.
    """
    from pipeline.kb.rail_worker import RAILS

    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 8)

    onboard(consent="both")

    assert set(_jobs(tmp_path)) == {"bookmark_catchup", "oracle_refresh", "curation_catchup"}
    assert set(_jobs(tmp_path)) <= set(RAILS)


def test_consent_is_durable_BEFORE_the_job_row_exists(onboard, monkeypatch, tmp_path):
    """Order, and it is a real race: the worker can claim a due-now job within a second, and the
    child reads the marker for itself. Queue first and it exits reporting no consent."""
    from pipeline.kb import rail_jobs

    seen = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 8)
    monkeypatch.setattr(rail_jobs, "request_now", lambda rail, **kw: seen.append(
        (rail, (tmp_path / f"{rail}_consent").exists())) or True)

    onboard(consent="both")

    # `curation_catchup` comes FIRST since 2026-09-16: it is queued beside the walk it schedules
    # the repeat of, which is earlier in the call than `_apply_consent`. The property this test
    # exists for is unchanged and is why that move is safe — its marker is written by the
    # `grant_consent()` that precedes the walk, so the row still cannot exist before the consent
    # the child reads for itself.
    assert seen == [("curation_catchup", True), ("bookmark_catchup", True),
                    ("oracle_refresh", True)]


def test_the_call_that_STARTS_arm_a_is_the_call_that_queues_its_repeat(
        onboard, monkeypatch, tmp_path):
    """⚠️ §1 OF THE 2026-09-16 HANDOFF — a permanent gap, not a flaky race, measured on a clean
    install where `curation_catchup` had consent, had run, and had no rail row and no log.

    `onboard` runs in up to four calls, and the call that STARTS Arm A is the call that RETURNS
    the consent prompt. The queue used to sit after that early return, so it could only ever fire
    on the NEXT call — by which time `arm_a` is recomputed from a state the walk has already
    changed. `curation.pending` is "connected platforms whose own collectors have never
    succeeded", and one succeeding collector marks the whole platform read, so the condition was
    self-defeating: the rail was queued only while the walk had NOT worked. The walk takes ~8s and
    a human reading a consent prompt takes longer.

    This test is TWO CALLS for that reason. A one-call test — consent answered in the same call
    that spawns the walk — takes the other branch and passes against the broken code, which is
    exactly why the suite stayed green over a fresh install that never refreshed its candidate
    list again.
    """
    _at_consent(monkeypatch)
    walked: list[int] = []
    # Inline, so the ordering under test is the real one: by call two the walk HAS succeeded.
    monkeypatch.setattr(onboard_tools, "_spawn", lambda target: target())
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda platforms=None: walked.append(1) or {"status": "ok", "ran": {}})
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending",
                        lambda live: [] if walked else sorted(live))

    first = onboard()

    assert first["status"] == "needs_consent", "call one is the prompt — that is the whole shape"
    assert walked, "Arm A starts on the prompt call"
    assert "curation_catchup" in _jobs(tmp_path), (
        "the recurring half is queued by the call that STARTED the walk; deferring it to the "
        "consent call means it is never queued at all, since a successful walk empties `pending`"
    )

    onboard(consent="both")                       # the human turn, and `pending` is empty by now
    assert _jobs(tmp_path)["curation_catchup"].due_at <= time.time()


def test_a_two_call_onboarding_ends_with_every_rail_it_owes_the_user(
        onboard, monkeypatch, tmp_path):
    """⚠️ THE WHOLE END STATE, from the shape a real user takes — handoff step 5.

    The clean install of 2026-09-16 finished setup with TWO rows in `rail_jobs.db`, which is
    exactly what `_apply_consent` queues; every other rail was left to whatever else was supposed
    to queue it, and for two of them nothing did. Nothing asserted the SET, so nothing failed.

    This walks the real order — connect, prompt, human turn, answer — with the signal walk
    running inline so `curation.pending` empties between the calls exactly as it does in
    production, and pins what a fresh install must end up with. A rail that stops being queued
    fails here by name.
    """
    real_run_curation = onboard_tools._run_curation
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 8)

    from pipeline.kb import curation_catchup as cc
    walked: list[int] = []
    # The real Arm A entry point over a stubbed pass: the chain from a collector that ran to
    # `candidate_probe` is part of what this asserts, and stubbing `_run_curation` would cut it.
    monkeypatch.setattr(cc, "_run",
                        lambda **kw: walked.append(1) or {"status": "ok", "ran": {"x_lists": {}}})
    monkeypatch.setattr(onboard_tools, "_run_curation", real_run_curation)
    monkeypatch.setattr(onboard_tools, "_spawn", lambda target: target())
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending",
                        lambda live: [] if walked else sorted(live))

    assert onboard()["status"] == "needs_consent"      # call one: the prompt
    onboard(consent="both")                            # call two: the answer

    assert set(_jobs(tmp_path)) == {
        "curation_catchup",      # the recurring people-discovery pull — §1, never queued before
        "candidate_probe",       # chained off the walk that just ran — §2, never queued before
        "bookmark_catchup",      # the backlog the user just said yes to
        "oracle_refresh",        # the recurring half, against a roster that already exists
    }, "a fresh install must end setup owing the user every rail its consent promised"


def test_a_hosted_onboarding_queues_the_same_rails_into_the_shared_database(
        onboard, monkeypatch, tmp_path):
    """⚠️ §3 OF THE 2026-09-16 HANDOFF — the one context with no evidence behind it at all. No
    hosted run was exercised, so the claim that a hosted user was missing the same two rails as a
    local one was reasoning from code. It is reasoning that holds — nothing in this module
    branches on hosted vs local for WHETHER to queue — and this is it asserted instead.

    Hosted means two environment values and nothing else (`gateway/children._child_env`): the
    gateway's already-validated subject as the home id, and the operator's ONE shared control
    database, which a child inherits precisely so a product action can queue durable work. The
    rows must land THERE, under that subject — a row in the child's own home is one the single
    worker never opens.
    """
    from pipeline.kb import rail_jobs
    from pipeline.kb.rail_jobs import RailJobStore

    subject = "104729183746152938471"
    shared = tmp_path / "var" / "rail_jobs.db"
    monkeypatch.setenv(rail_jobs.WORKER_HOME_ID_ENV, subject)
    monkeypatch.setenv(rail_jobs.WORKER_DB_ENV, str(shared))
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 8)

    onboard(consent="both")

    assert {(j.home_id, j.rail) for j in RailJobStore(shared).list_jobs()} == {
        (subject, "curation_catchup"), (subject, "bookmark_catchup"), (subject, "oracle_refresh")}
    assert not (tmp_path / "rail_jobs.db").exists(), (
        "a hosted row in the child's own home is one the single worker never opens — the silent "
        "total failure `worker_db_path()` raises to prevent"
    )


def test_refresh_consent_does_NOT_queue_on_an_empty_roster(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 0)

    out = onboard(consent="refresh")

    # A child that finds zero pairs is a no-op dressed as an action.
    assert "oracle_refresh" not in _jobs(tmp_path)
    assert out["consent_applied"]["queued"] == []


def test_refresh_consent_DOES_queue_when_oracles_already_exist(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 8)

    onboard(consent="refresh")

    # Re-entry on a populated store: the work IS waiting.
    assert _jobs(tmp_path)["oracle_refresh"].due_at <= time.time()


def test_a_failed_queue_leaves_the_consent_written_and_says_it_is_not_queued(
        onboard, monkeypatch, tmp_path):
    """An operator's misconfigured worker database must not cost the user their answer."""
    from pipeline.kb import rail_jobs

    _at_consent(monkeypatch)
    monkeypatch.setattr(rail_jobs, "request_now", lambda rail, **kw: False)

    out = onboard(consent="backlog")

    assert (tmp_path / "bookmark_catchup_consent").exists()
    assert out["consent_applied"]["granted"] == ["backlog"]
    assert out["consent_applied"]["queued"] == []


def test_saying_no_on_re_entry_revokes(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    (tmp_path / "oracle_refresh_consent").touch()
    onboard(consent="backlog")
    assert not (tmp_path / "oracle_refresh_consent").exists()


def test_consenting_to_recurring_updates_installs_the_worker(onboard, monkeypatch, tmp_path):
    """⚠️ CONSENT AND THE WORKER ARE ONE DECISION. Until 2026-09-14 the user agreed to "keep your
    Oracles current — RECURRING, forever", a marker was written, a job row was queued, and nothing
    on the machine ever claimed it: `rail_worker` is the only thing that does and it was left as a
    command the user had to find and run. A fresh onboarding left `bookmark_catchup` and
    `substack_saved_catchup` queued at 15:58 and still unstarted an hour later."""
    from opyt_core import install_worker

    _at_consent(monkeypatch)
    monkeypatch.setattr(install_worker, "status",
                        lambda: {"supported": True, "installed": False, "loaded": False,
                                 "ran": False})
    installs = []
    monkeypatch.setattr(install_worker, "install",
                        lambda **kw: installs.append(True) or {"status": "INSTALLED"})

    out = onboard(consent="both")

    assert installs, "consenting to recurring refresh must install the resident worker"
    assert out["consent_applied"]["worker"]["status"] == "INSTALLED"
    assert "scheduled_updates" not in out          # nothing to warn about — it is running


def test_a_backlog_only_answer_installs_nothing(onboard, monkeypatch, tmp_path):
    """The one-time import is bounded, runs in-process, and does not outlive the session. Only
    the RECURRING half is a promise that needs a resident process to keep it."""
    from opyt_core import install_worker

    _at_consent(monkeypatch)
    monkeypatch.setattr(install_worker, "status",
                        lambda: {"supported": True, "installed": False, "loaded": False,
                                 "ran": False})
    monkeypatch.setattr(install_worker, "install",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("must not install")))
    removed = []
    monkeypatch.setattr(install_worker, "uninstall",
                        lambda **kw: removed.append(True) or {"status": "NOT_PRESENT"})

    onboard(consent="backlog")

    assert removed, "revoking refresh must also take the resident worker away"


def test_an_unkeepable_promise_is_flagged_to_the_host(onboard, monkeypatch, tmp_path):
    """The markers stand — the user's decision is theirs — but a host that is not told will go on
    saying "it updates on its own", which is the false claim `install_worker.status` was written
    to prevent. Absent when healthy, so the field is not noise."""
    from opyt_core import install_worker

    _at_consent(monkeypatch)
    monkeypatch.setattr(install_worker, "status",
                        lambda: {"supported": False, "installed": False, "loaded": False,
                                 "ran": False})

    out = onboard(consent="both")

    assert (tmp_path / "oracle_refresh_consent").exists()      # consent is NOT rolled back
    assert out["scheduled_updates"]["running"] is False
    assert "do NOT tell them things update automatically" in out["scheduled_updates"]["message"]


def test_a_hosted_home_is_not_told_its_schedule_is_dead(onboard, monkeypatch, tmp_path):
    """The other direction of the test above, and the one that was wrong on every remote run.

    A hosted child is a Linux process, so `install_worker.status()["supported"]` is False and the
    warning above fired — telling the host to say the library "updates when they ask for it" on
    the one home where a resident worker is already claiming the rows consent just queued. Under-
    promising is the rule when the probe cannot answer; this probe was answering a different
    question confidently.
    """
    from opyt_core import install_worker

    _at_consent(monkeypatch)
    monkeypatch.setenv("OPYT_WORKER_HOME_ID", "1078")
    monkeypatch.setenv("OPYT_WORKER_DB", str(tmp_path / "rail_jobs.db"))
    monkeypatch.setattr(install_worker, "status",
                        lambda: {"supported": False, "installed": False, "loaded": False,
                                 "ran": False})

    out = onboard(consent="both")

    assert out["consent_applied"]["worker"]["status"] == "RESIDENT"
    assert "scheduled_updates" not in out, "the schedule IS kept here — do not warn against it"


def test_an_unknown_consent_word_is_refused_not_guessed(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    out = onboard(consent="yes")
    assert out["status"] == "error"
    assert not (tmp_path / "onboard_consent_asked").exists()


def test_the_backlog_commitment_names_the_platform_that_is_connected(onboard, monkeypatch):
    """The prompt describes the import the user would actually get. A Substack-only user is asked
    about their saved posts, not about X bookmarks — the commitment is the ACT (content you saved
    yourself), and which lists it walks is the user's own connections."""
    _at_sources(monkeypatch, substack=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    msg = onboard()["message"]

    assert "Substack saved posts" in msg
    assert "X bookmarks" in msg          # only in the "if you connect that later" clause
    assert msg.index("Substack saved posts") < msg.index("X bookmarks")
    assert "RECURRING" in msg


def test_the_prompt_quotes_no_size_and_no_price_for_the_import(onboard, monkeypatch):
    """Both backlogs have been measured exactly once, on one account each — ~1,080 X bookmarks,
    THREE Substack saved posts. Either number in this prompt would sell a guess as a
    measurement. The honest statement is the shape, and it is the one the copy makes."""
    _at_consent(monkeypatch)
    msg = onboard()["message"]
    assert "how much you have saved" in msg
    assert "$" not in msg
    assert not re.search(r"\b\d{2,}\b", msg)      # no item counts, no dollar figures


def test_a_backlog_answer_grants_both_rails_and_queues_only_the_connected_one(
        onboard, monkeypatch, tmp_path):
    """One answer, one act, two markers — because the two rails have different request patterns
    and `bookmark_catchup.consented` forbids sharing one marker across loops.

    The unconnected platform's marker is still written, and the prompt says so: the consent phase
    is asked ONCE, so a marker withheld here could never be granted when that platform is
    connected later. Its rail is NOT queued, because a child that finds no session is a no-op
    dressed as an action whose recorded failure looks like a real one."""
    _at_sources(monkeypatch, substack=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    out = onboard(consent="backlog")

    assert (tmp_path / "substack_saved_catchup_consent").exists()
    assert (tmp_path / "bookmark_catchup_consent").exists()
    assert out["consent_applied"]["queued"] == ["substack_saved_catchup"]
    assert "bookmark_catchup" not in _jobs(tmp_path)


def test_a_platform_connected_after_the_question_is_offered_its_import(onboard, monkeypatch,
                                                                      tmp_path):
    """The consent phase is one-shot. Every user who answered before `substack_saved_catchup`
    existed answered a prompt naming X bookmarks alone, and reusing that answer for Substack
    would be consent obtained for a different question — so their marker is absent and no rail
    imports anything. The handoff is the only place left to say the import exists."""
    _at_sources(monkeypatch, substack=True)
    monkeypatch.setattr(onboard_tools.onboard_state, "_asked", lambda: True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: [])
    (tmp_path / "bookmark_catchup_consent").touch()

    msg = onboard()["message"]

    assert "your Substack saved posts" in msg
    assert "onboard(consent='backlog')" in msg


def test_the_offer_goes_quiet_once_the_marker_is_written(onboard, monkeypatch, tmp_path):
    """It reports an outstanding decision, not a feature — a user who already said yes (or was
    asked in a prompt that named the platform) must never see it again."""
    _at_sources(monkeypatch, substack=True)
    monkeypatch.setattr(onboard_tools.onboard_state, "_asked", lambda: True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: [])
    (tmp_path / "substack_saved_catchup_consent").touch()

    assert "onboard(consent='backlog')" not in onboard()["message"]


def test_the_prompt_states_both_cost_shapes(onboard, monkeypatch):
    """One-time-and-bounded vs recurring-forever are different commitments. Quoting only a
    per-item rate is useless — a user cannot apply one to an item count they do not know."""
    _at_consent(monkeypatch)
    msg = onboard()["message"].lower()
    assert "one-time" in msg or "one time" in msg
    assert "recurring" in msg or "every" in msg or "keeps" in msg


# ── curation is PER PLATFORM, because platforms are connected one at a time ─────
#
# ⚠️ THE DEFECT. The gate was `collectors and not _curation_ok()` — "has ANY collector anywhere
# ever succeeded" — so the first success on the first platform flipped the phase to `done` for
# good, and anything connected afterwards was never read. Measured 2026-09-13: David connected
# Substack, `substack_follows` succeeded, he then connected X, and `x_lists`/`x_following`/
# `x_likes` never ran once. `collector_runs` held only Substack rows while `x_lists` returned 6
# candidates the moment it was called by hand — a valid session, silently orphaned.
#
# It reads as a scoring fault, not a missing feature: `screen` ranks on distinct
# (signal_type, platform) pairs, so a platform nobody read cannot corroborate anyone and nothing
# pre-ticks. The user sees "each of these showed up once. Nothing is pre-selected."

def _record_ok(collector):
    from pipeline.kb import curation_state, schema
    conn = schema.connect()
    try:
        curation_state.record_run(conn, collector, status="ok")
    finally:
        conn.close()


def test_a_platform_connected_after_another_succeeded_is_still_read(onboard, monkeypatch):
    """Substack read, X connected later: the phase must come BACK to curation for X alone."""
    from pipeline.kb import onboard_state

    _keys_green(monkeypatch)
    monkeypatch.setattr(onboard_state, "_asked", lambda: True)
    monkeypatch.setattr(onboard_state, "_store_sources", lambda: onboard_state._NO_STORE_SOURCES)
    monkeypatch.setattr(onboard_state, "_connected_substack", lambda: True)
    monkeypatch.setattr(onboard_state, "_connected_x", lambda: False)

    _record_ok("substack_follows")
    assert onboard_state.derive()["phase"] == "done"          # Substack alone is finished

    monkeypatch.setattr(onboard_state, "_connected_x", lambda: True)
    state = onboard_state.derive()

    assert state["phase"] == "curation"                        # …and X reopens it
    assert state["curation"]["pending"] == ["x"]               # for X only
    assert state["curation"]["any_ok"] is True                 # the old gate still said "done"

    _record_ok("x_lists")
    assert onboard_state.derive()["phase"] == "done"           # one X collector settles X


def test_the_pull_is_scoped_to_the_platform_that_reopened_the_phase(onboard, monkeypatch):
    """`force=True` bypasses the per-collector floor, so an unscoped pass would re-walk the
    platform that already succeeded and spend its requests again for nothing."""
    seen = {}
    from pipeline.kb import curation_catchup as cc

    _at_curation(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: ["x"])
    monkeypatch.setattr(cc, "run_curation_catchup",
                        lambda **kw: seen.update(kw) or {"status": "ok", "ran": {}})
    onboard()

    assert seen["platforms"] == {"x"} and seen["force"] is True


def test_the_rail_itself_stays_unscoped(onboard, monkeypatch):
    """`platforms=None` is the rail's case: a scheduled refresh has no connect event to scope to,
    and narrowing it would freeze whichever platform the last `onboard` call did not name."""
    from pipeline.kb import curation_catchup as cc, ingest_curation

    ran = []
    monkeypatch.setattr(cc, "_platform_reachable", lambda p: True)
    monkeypatch.setattr(ingest_curation, "run_and_record",
                        lambda conn, spec: ran.append(spec.collector) or {"found": 0})
    cc.run_curation_catchup(force=True)

    assert set(ran) == set(ingest_curation.COLLECTORS)


# ── the backlog BODIES, in this process (2026-09-13) ────────────────────────────
#
# ⚠️ Consent queues `bookmark_catchup` through `request_now`, which only WRITES A ROW — the
# resident worker is the sole thing that ever claims it. On a from-source install there is no
# worker, so the row sat with `started_at` NULL forever and the import the user consented to never
# happened at all. This path does not depend on a process the user may not have.
#
# It runs AFTER the blocking signal pass and never instead of it: the screen is scored on signals,
# so those must be on disk before the candidate list is built, while nothing waits on a body.

def _saved(tools, platforms):
    """Start the arms and report them — the pair `_onboard` calls, driven as one step because the
    split exists for ORDER inside `_onboard`, not for the arms themselves. Nothing is joined any
    more (R7 overturned, 2026-09-14): the join is what the client kept killing."""
    return tools._report_saved_content(tools._start_saved_content(platforms))


def _bodies(monkeypatch, *, consented=True, substack=False):
    """Run the spawned pull SYNCHRONOUSLY — a real daemon thread would outlive the monkeypatching
    and hit the network after the stubs are gone. Returns the call log.

    Each platform's arm reads its OWN consent marker, so the two are set separately here for the
    reason `substack_saved_catchup` states: opting into one loop must never silently opt you into
    another with a different request pattern."""
    from pipeline.kb import bookmark_catchup, substack_saved_catchup
    calls = []
    monkeypatch.setattr(bookmark_catchup, "consented", lambda: consented)
    monkeypatch.setattr(substack_saved_catchup, "consented", lambda: substack)
    monkeypatch.setattr(onboard_tools, "_spawn", lambda target: calls.append(target) or target())
    return calls


def test_the_bodies_run_in_process_rather_than_waiting_on_a_worker(onboard, monkeypatch):
    from pipeline.kb import ingest_x

    pulled = []
    _bodies(monkeypatch)
    monkeypatch.setattr(ingest_x, "sync_bookmarks",
                        lambda conn, emb, **kw: pulled.append(conn) or {"added": 0})
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: ["x"])

    out = _saved(onboard_tools, {"x"})

    assert out["status"] == "importing"
    assert len(pulled) == 1
    # Its OWN connection: the caller's is mid-request and SQLite is not thread-shared.
    assert pulled[0] is not None
    # It reports the free half as ARRIVING — not complete, which nothing here knows any more —
    # and promises nothing about the metered half. "Enrichment" is the one name for that (R6),
    # and each post is searchable the moment it lands rather than when the import ends.
    assert "searchable the moment it lands" in out["message"]
    assert "Enrichment" in out["message"]


def test_the_bodies_do_not_run_for_a_platform_the_user_did_not_consent_to(onboard, monkeypatch):
    from pipeline.kb import ingest_x

    pulled = []
    _bodies(monkeypatch, consented=False)
    monkeypatch.setattr(ingest_x, "sync_bookmarks", lambda *a, **k: pulled.append(1))

    assert _saved(onboard_tools, {"x"}) is None
    assert pulled == []


def test_the_bodies_do_not_run_for_a_platform_that_was_not_pulled(onboard, monkeypatch):
    """Scoped like the signal pass it follows: connecting Substack must not walk X's bookmarks."""
    _bodies(monkeypatch)
    assert _saved(onboard_tools, {"substack"}) is None


def test_the_substack_bodies_run_in_process_too(onboard, monkeypatch):
    """The saved posts were the half that only a resident worker could ever import, and Substack
    is the platform that needs it most: subscribe and follow are near-disjoint graphs and likes
    are unreadable, so a `save` is the only realistic second signal anybody has there."""
    from pipeline.kb import ingest_curation

    pulled = []
    _bodies(monkeypatch, consented=False, substack=True)
    monkeypatch.setattr(ingest_curation, "sync_substack_saved",
                        lambda conn, emb, **kw: pulled.append(conn) or {"added": 0})

    out = _saved(onboard_tools, {"substack"})

    assert out["sources"] == ["substack-saved"]
    assert len(pulled) == 1 and pulled[0] is not None      # its OWN connection, per thread
    assert "no Enrichment pass behind it" in out["message"]


def test_each_platform_body_arm_is_gated_on_its_own_consent(onboard, monkeypatch):
    """Two markers, not one. A user who answered `backlog` to a prompt that named only X
    consented to X — `substack_saved_catchup` refuses an established-store fallback for the same
    reason, and this is the same rule one layer up."""
    from pipeline.kb import ingest_curation, ingest_x

    _bodies(monkeypatch, consented=True, substack=False)
    monkeypatch.setattr(ingest_x, "sync_bookmarks", lambda conn, emb, **kw: {"added": 0})
    monkeypatch.setattr(ingest_curation, "sync_substack_saved",
                        lambda *a, **k: pytest.fail("ran a Substack import nobody consented to"))

    out = _saved(onboard_tools, {"x", "substack"})

    assert out["sources"] == ["x-bookmarks"]


def test_a_body_pull_that_explodes_never_reaches_the_caller(onboard, monkeypatch):
    """Fail-safe: the curation pass already succeeded and the screen is already scored. A body
    pull is the last thing that may take the setup call down with it."""
    from pipeline.kb import ingest_x

    _bodies(monkeypatch)
    monkeypatch.setattr(ingest_x, "sync_bookmarks", _boom_bodies)

    assert _saved(onboard_tools, {"x"})["status"] == "importing"


def _boom_bodies(*a, **k):
    raise RuntimeError("x.com is down")


def test_arm_a_lands_before_arm_b_is_started(onboard, monkeypatch):
    """ORDER IS STILL THE POINT, and it is still kept — signals start before bodies do.

    What changed on 2026-09-14 is that NEITHER is waited for. Arm B's join is what got
    `onboard(consent='both')` cut off at the client's 60-second wall, so "setup is complete" now
    means the posts are IMPORTING. The ordering that genuinely needed the join — Enrichment
    starting against a finished corpus — moved onto the X arm's own thread, which is the only
    place that fact is known."""
    order = []
    _at_curation(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: ["x"])
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda p=None: order.append("signals") or {"status": "ok", "ran": {}})
    monkeypatch.setattr(onboard_tools, "_start_saved_content",
                        lambda p: order.append("bodies") or [(onboard_tools._body_arms()[0],
                                                              None)])

    out = onboard()

    assert order == ["signals", "bodies"]
    assert out["saved_content"]["status"] == "importing"
    assert "importing" in out["saved_content"]["message"]


# ── the connect-time split: Arm A before the question, Arm B after (2026-09-13) ──
#
# ⚠️ THERE IS NO "CONNECT MOMENT" TO HOOK LOCALLY. `guided_login.start` launches Chrome and
# returns immediately; detection is ONLY `has_managed_x_session()` polled inside
# `onboard_state.derive()`. So "fire at connect" means "fire on the next `onboard` call once
# `pending` is non-empty" — a phase-ORDERING change, not new plumbing.
#
# The two arms split on CONSENT, not on cost. Arm A needs no answer: connecting the session is
# already the consent for a cookie scrape, and `_onboard` grants curation consent on any live
# session. Arm B is the saved BODIES, which is exactly what the question is about, and on a fresh
# store `bookmark_catchup.consented()` has no marker and no atoms to infer one from.

def test_arm_a_runs_before_the_consent_question_is_even_put(onboard, monkeypatch):
    """The gain: the candidate list is already scored while the user reads the prompt, instead of
    after they answer it."""
    ran = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: ["x"])
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda p=None: ran.append(p) or {"status": "ok", "ran": {}})

    out = onboard()

    assert out["status"] == "needs_consent"          # the question is still put
    assert ran == [{"x"}]                            # …and the signal walk already happened
    # STARTED, not finished — Arm A stopped blocking the call that returns this prompt, because
    # its ~30s was being spent with the user watching a blank screen before being asked a
    # question. `screen` is the first surface that reads a candidate, and it is turns away.
    assert out["curation"]["status"] == "scoring"


def test_arm_b_does_not_run_before_the_user_has_answered(onboard, monkeypatch):
    """The saved BODIES are the thing being consented to. A pre-consent Arm B would import them
    and then ask permission."""
    from pipeline.kb import bookmark_catchup
    started = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(bookmark_catchup, "consented", lambda: True)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: ["x"])
    monkeypatch.setattr(onboard_tools, "_start_saved_content",
                        lambda p: started.append(p) or [])

    assert onboard()["status"] == "needs_consent"
    assert started == []


def test_answering_backlog_imports_every_connected_platforms_saved_posts(onboard, monkeypatch):
    """Arm A has already run on the PREVIOUS call by then, so `curation.pending` is empty — the
    platform set for Arm B cannot be derived from it. It comes from what is connected, on the call
    that granted the consent."""
    started = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: [])
    monkeypatch.setattr(onboard_tools, "_start_saved_content",
                        lambda p: started.append(p) or [])

    onboard(consent="backlog")

    assert started and "x" in started[0]


def test_a_plain_onboard_call_does_not_re_walk_the_saved_list(onboard, monkeypatch):
    """Nothing newly connected and no consent granted this call — so there is nothing to import,
    and re-walking 1,000 bookmarks to discover that is not free even when every atom fast-skips."""
    started = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_asked", lambda: True)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: [])
    monkeypatch.setattr(onboard_tools, "_start_saved_content",
                        lambda p: started.append(p) or [])

    onboard()

    assert started == [set()]


def test_enrichment_starts_on_the_x_arms_own_thread_once_the_bodies_land(onboard, monkeypatch):
    """⚠️ THE ORDERING THAT USED TO NEED THE JOIN. Enrichment walks the SAME corpus the arm just
    wrote, and starting it against a half-written one spends metered requests on bookmarks whose
    atoms do not exist yet. `onboard` guaranteed that by JOINING — which is exactly what the
    client kept killing. The constraint was never "the caller must wait"; it was "enrichment must
    start after the bodies land", and that is a fact the arm's thread knows and the caller does
    not."""
    from pipeline.kb import ingest_x
    order = []
    monkeypatch.setattr(ingest_x, "sync_bookmarks",
                        lambda conn, emb, **kw: order.append("bodies") or {"added": 3})
    monkeypatch.setattr(onboard_tools, "_start_enrichment",
                        lambda: order.append("enrichment") or {"status": "running"})

    onboard_tools._pull_x_bookmarks()

    assert order == ["bodies", "enrichment"]


def test_an_import_that_explodes_does_not_start_enrichment(onboard, monkeypatch):
    """An import that raised did not finish writing the corpus, so the pass that walks it has
    nothing correct to walk. Fail-safe in the right direction: skip, never spend."""
    from pipeline.kb import ingest_x
    started = []
    monkeypatch.setattr(ingest_x, "sync_bookmarks", lambda conn, emb, **kw: 1 / 0)
    monkeypatch.setattr(onboard_tools, "_start_enrichment", lambda: started.append(1))

    onboard_tools._pull_x_bookmarks()          # swallows, as always

    assert started == []


def test_substack_has_no_enrichment_analogue(onboard, monkeypatch):
    """§5 of the ruling: no rate meter exists there at all, so under R2 nothing there is ever
    background; the bodies are not in the list payload the way X's are; and an Oracle's Substack
    IS their archive, so there is no cheap breadth arm to defer anything from."""
    from pipeline.kb import substack_saved_catchup
    started = []
    monkeypatch.setattr(substack_saved_catchup, "sync_saved_posts",
                        lambda *a, **kw: {"added": 0}, raising=False)
    monkeypatch.setattr(onboard_tools, "_start_enrichment", lambda: started.append(1))

    onboard_tools._pull_substack_saved()

    assert started == []


# ── phase 3: the free curation pull ─────────────────────────────────────────

_REAL_RUN_CURATION = onboard_tools._run_curation


def _at_curation(monkeypatch):
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_asked", lambda: True)
    monkeypatch.setattr(onboard_tools, "_run_curation", _REAL_RUN_CURATION)


def test_phase3_calls_curation_catchup_not_curation_pull(onboard, monkeypatch):
    called = []
    _at_curation(monkeypatch)      # first: it restores the real _run_curation
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda *a: called.append("catchup") or {"status": "ok", "ran": {}})
    onboard()
    assert called == ["catchup"]


def test_phase3_forces_past_the_six_hour_floor_on_first_run(onboard, monkeypatch):
    """The floor throttles a BACKGROUND loop. A user who just asked to be set up is not that."""
    seen = {}
    from pipeline.kb import curation_catchup as cc
    monkeypatch.setattr(cc, "run_curation_catchup",
                        lambda **kw: seen.update(kw) or {"status": "ok", "ran": {}})
    _at_curation(monkeypatch)
    onboard()
    assert seen["force"] is True


def test_phase3_never_reaches_the_tiered_ladder(onboard, monkeypatch):
    """⚠️ `curation_pull(tiered=True)` reads the WHOLE STORE's signalled-entity count, so on an
    established store it clears after Tier 1 and permanently skips following and likes — the two
    collectors this exists to run. It would look like it was working."""
    from pipeline.kb import ingest_curation
    monkeypatch.setattr(ingest_curation, "curation_pull",
                        lambda *a, **kw: pytest.fail("curation_pull must never be called here"))
    from pipeline.kb import curation_catchup as cc
    monkeypatch.setattr(cc, "run_curation_catchup", lambda **kw: {"status": "ok", "ran": {}})
    _at_curation(monkeypatch)
    onboard()


# ── phase 4: the handoff ────────────────────────────────────────────────────

def _all_done(monkeypatch, candidates=0, **store):
    _at_curation(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_pending", lambda live: [])
    monkeypatch.setattr(onboard_tools, "_candidate_count", lambda: candidates)


def test_done_names_oracle_as_the_next_call(onboard, monkeypatch):
    _all_done(monkeypatch)
    out = onboard()
    assert out["phase"] == "done" and out["next_tool"] == "oracle"


def test_handoff_bounds_the_first_ingest(onboard, monkeypatch):
    """⚠️ `oracle(action='ingest')` is a synchronous foreground loop over EVERY confirmed Oracle
    with no pick cap and no time budget. Fixing that is out of scope, but `onboard` is what sends
    people into it — so the copy bounds the first pass. Only in the branch that HAS a screening
    list: with nothing to screen there is no first batch to bound."""
    _all_done(monkeypatch, candidates=40)
    msg = onboard()["message"]
    assert "3" in msg or "three" in msg.lower()


def test_done_reports_what_actually_landed(onboard, monkeypatch):
    _all_done(monkeypatch, candidates=42)
    assert onboard()["candidates"] == 42


def test_a_user_with_no_collector_is_not_told_a_collector_ran(onboard, monkeypatch, tmp_path):
    """⚠️ "The free collectors surfaced 0 people to screen" describes a pass that never happened.
    A blog or research-topic user has no platform session, so nothing collected — and the honest
    version of that sentence is the list of ways they can add something."""
    from pipeline.ingestion import x_graphql

    _all_done(monkeypatch)
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)
    monkeypatch.setattr(onboard_tools.onboard_state, "_store_sources",
                        lambda: {**onboard_tools.onboard_state._NO_STORE_SOURCES,
                                 "oracles": True})

    out = onboard()

    assert out["phase"] == "done"
    assert "collector" not in out["message"] or "no source" in out["message"]
    assert "add_handles" in out["message"] and "hopper" in out["message"]
    # RULED 2026-09-12: watches are not an onboarding move, so the ways-in list offers the
    # content routes only.
    assert "sitting(action='watchlist'" not in out["message"]


def _done_blind(monkeypatch):
    """Keys green, no session, nothing named, nothing watched — the user skipped through."""
    _at_sources(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_skipped", lambda: True)
    monkeypatch.setattr(onboard_tools.onboard_state, "_asked", lambda: True)
    monkeypatch.setattr(onboard_tools, "_candidate_count", lambda: 0)


def test_a_skipped_through_store_is_asked_a_question_not_handed_a_form(onboard, monkeypatch):
    """⚠️ A store with NO signal — nobody named, nothing watched, no session — used to get
    the same open-roots list as everyone else: the sources form, re-printed at the finish
    line. The honest move is a direct question, and the asking has rules: topics lead,
    because "what are you into" is easier to answer than "name five writers", and whichever
    way the user answers, the FINDING is the host's job."""
    _done_blind(monkeypatch)

    msg = onboard()["message"]

    assert msg.lower().index("what topics") < msg.lower().index("who do you already read")
    assert "add_handles" in msg
    assert "web-search" in msg.lower()
    assert "paste" not in msg.lower()
    # RULED 2026-09-12: onboarding creates NO standing watches. A watch is a commitment whose
    # wording the user should shape, and that conversation belongs after they have played with
    # the store — so the handoff fills the store (survey, saves, authors) and defers the watch.
    assert "sitting(action='watchlist'" not in msg


def test_a_user_who_named_something_is_not_asked_what_they_are_into(onboard, monkeypatch):
    """The direct question is for the store with no signal. A user who named a blog has
    already answered it, and asking again ignores what they said."""
    _done_blind(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_store_sources",
                        lambda: {**onboard_tools.onboard_state._NO_STORE_SOURCES,
                                 "oracles": True})

    msg = onboard()["message"]

    assert "What topics are you into" not in msg
    assert "hopper" in msg and "add_handles" in msg


def test_a_watchlist_only_store_is_still_asked_what_the_user_is_into(onboard, monkeypatch):
    """⚠️ THE 2026-09-12 TRANSCRIPT. A store carrying six watches and zero atoms got the
    generic open-roots list, because watchlist rows counted as "named something". The flow's
    question is "is there anything to PLAY with" — and watches stage an inbox, not a library.
    The opener stays truthful for this state: it credits the watches instead of claiming
    "nothing watched"."""
    _done_blind(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_store_sources",
                        lambda: {**onboard_tools.onboard_state._NO_STORE_SOURCES,
                                 "watchlist": True})

    msg = onboard()["message"]

    assert "What topics are you into" in msg
    assert "nothing watched" not in msg
    assert "staging candidates" in msg


def test_an_unreadable_atom_count_never_claims_an_empty_store(onboard, monkeypatch):
    """"Your store is empty" is a claim about the user's data; a broken read routes to the
    generic branch, which claims nothing."""
    _done_blind(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_atom_count", lambda: None)

    assert "What topics are you into" not in onboard()["message"]


def test_a_connected_X_user_is_still_told_the_other_roots_exist(onboard, monkeypatch):
    """⚠️ THE 2026-09-09 FINDING. `_phase_sources` promises the four roots are not exclusive —
    "answering one never closes the others" — and then `derive()` closes the question: one
    connected collector sets `sources.ok`, so the phase never returns. Only the no-collector
    branch of the handoff re-offered them, so a user who connected X and got a screening list
    was never told Substack existed. That is what made the second root easy to lose, not the
    host rendering the answers as a single-select."""
    _all_done(monkeypatch, candidates=40)   # X connected, candidates found

    msg = onboard()["message"]

    assert "source='substack'" in msg
    assert "source='x'" not in msg          # already connected; do not re-offer it


def test_the_ways_in_are_listed_from_one_place(onboard, monkeypatch):
    """All three done-branches print the same list, built by `_still_open`. Asserted as a
    property rather than by comparing copy: whatever a home has not connected is offered, and
    the naming paths are always offered, in every branch."""
    for candidates in (0, 40):
        _all_done(monkeypatch, candidates=candidates)
        msg = onboard()["message"]
        assert "add_handles" in msg and "hopper" in msg


def test_done_never_says_not_implemented(onboard, monkeypatch):
    _all_done(monkeypatch)
    assert onboard()["status"] != "not_implemented"


def test_the_real_server_registers_onboard():
    """⚠️ A PRESENCE REQUIREMENT, WHICH IS A TEST'S JOB — `.guards.py` catches retired patterns
    COMING BACK, not load-bearing wiring GOING AWAY. Registration is wrapped in a try/except so a
    stripped distribution still starts, which means a broken import degrades to a printed line
    and a silently missing tool. Three strings in the tree tell users to run `onboard`; if the
    registration ever falls out, those become promises to call a tool that is not there."""
    from mcp_server import server
    m = _MCP()
    server.register_onboard_tools(m) if hasattr(server, "register_onboard_tools") else None
    from mcp_server.onboard_tools import register_onboard_tools
    register_onboard_tools(m)
    assert "onboard" in m.tools

    src = (server.__file__ and open(server.__file__).read()) or ""
    assert "register_onboard_tools(mcp)" in src, "server.py no longer registers `onboard`"


def test_the_promises_other_modules_make_about_onboard_are_keepable():
    """Four strings in the tree told users to run `onboard` before it existed. They are now
    true — but "true" means the vocabulary MATCHES, not just that a tool with the right name
    exists. `oracle_refresh` promises opting in to the recurring half; `bookmark_catchup`
    promises the one-time backlog import. Both must be reachable through consent words this
    tool actually accepts, or the messages send users to an argument that errors."""
    from pipeline.kb import bookmark_catchup, oracle_refresh

    grants_refresh = {w for w, (_, r) in onboard_tools._CONSENT_WORDS.items() if r}
    grants_backlog = {w for w, (b, _) in onboard_tools._CONSENT_WORDS.items() if b}
    assert grants_refresh and grants_backlog

    assert "onboard" in open(oracle_refresh.__file__).read()
    assert "onboard" in open(bookmark_catchup.__file__).read()

    # The store_empty notice names `onboard` as the next tool a new user should call.
    from opyt_core import kb as kb_entry
    assert "onboard" in open(kb_entry.__file__).read()


def test_source_x_reconnects_a_session_that_is_already_connected(onboard, monkeypatch):
    """An X session can expire or belong to the wrong account after setup is finished.

    `graphql_get` tells the user to reconnect X when x.com rejects the hosted session, so the
    tool has to be able to carry that instruction out without the profile being deleted by hand.
    """
    from pipeline.ingestion import browser_cookies as bc, guided_login

    started = []
    _at_sources(monkeypatch, connected=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})
    monkeypatch.setattr(guided_login, "start",
                        lambda url: started.append(url) or bc.backend_for("chrome"))

    assert onboard()["phase"] != "sources"

    out = onboard(source="x")

    assert started == ["https://x.com/login"]
    assert out["status"] == "awaiting_login"


def test_a_consent_answer_is_never_dropped_in_favor_of_a_reconnect(onboard, monkeypatch):
    """Both arguments are decisions, and consent is the one that cannot be re-offered later.

    A caller that sends them together gets the consent recorded; the reconnect is one more
    call away, while a silently discarded consent answer is not recoverable by the user.
    """
    from pipeline.ingestion import guided_login

    started = []
    _at_sources(monkeypatch, connected=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})
    monkeypatch.setattr(guided_login, "start", lambda url: started.append(url))

    out = onboard(source="x", consent="none")

    assert started == []
    assert out["phase"] != "sources"


# ── The starter allowance ─────────────────────────────────────────────────────────────────────
# What these pin is the SHAPE of the detour, not the fact that it exists: the trial defers the
# OpenRouter approval and never replaces it, so every road still ends at the same door.

def _at_trial_over(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "keys",
                                      "keys": {"openrouter": {"state": "trial_over",
                                                              "message": "spent"},
                                               "ok": False},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools.openrouter_oauth, "acquire",
                        lambda **kw: calls.append("oauth") or {"status": "stored",
                                                               "message": "m"})
    monkeypatch.setattr(onboard_tools.trial, "acquire",
                        lambda **kw: calls.append("trial") or {"status": "stored"})
    return calls


def test_hosted_claims_the_allowance_inside_the_call_and_never_shows_the_step(onboard,
                                                                              monkeypatch):
    """THE WHOLE POINT OF THE HOSTED HALF. A hosted child is already talking to a gateway that
    authenticated this user before the child existed, so there is nobody left to ask — the
    allowance is claimed mid-call and the user is asked about SOURCES, never about a key.

    Pinned as "the phase is gone", not as "a mint happened": a version that mints and still
    returns a keys-phase prompt has paid the money and kept the friction."""
    calls = _at_missing_key(monkeypatch, trial_available=True, hosted=True)
    phases = iter(["keys", "sources"])
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": next(phases),
                                      "keys": {"openrouter": {"state": "missing", "message": ""},
                                               "ok": False},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools, "_phase_sources", lambda: {"phase": "sources"})

    out = onboard()

    assert calls == ["trial"]
    assert out["phase"] == "sources"


def test_a_gateway_that_will_not_mint_leaves_the_user_on_the_ordinary_path(onboard, monkeypatch):
    """FAIL-SAFE, and the invariant's exact shape: an optional input that is absent degrades to
    the old flow. A trial that cannot be claimed is a shortcut that is closed, not an outage —
    so the user meets the OpenRouter approval, which is where they would have been anyway."""
    calls = _at_missing_key(monkeypatch, trial_available=True,
                            trial_result={"status": "unavailable", "message": "no"})

    out = onboard(start="trial")

    assert calls == ["trial"]
    assert out["status"] == "needs_openrouter"      # a prompt, never an error
    assert "openrouter.ai" in out["message"]


def test_claiming_the_allowance_does_not_touch_the_oauth_flow(onboard, monkeypatch):
    calls = _at_missing_key(monkeypatch, trial_available=True)
    out = onboard(start="trial")
    assert calls == ["trial"]
    assert out["status"] == "ok" and out["step"] == "trial"


def test_a_spent_allowance_explains_before_it_opens_anything(onboard, monkeypatch):
    """The one moment OPYT asks anyone to pay. Same two-call discipline as every other step that
    takes the screen, and the cost sentence has to be here — this is the surface that sends them
    to a page with a card field on it."""
    calls = _at_trial_over(monkeypatch)

    out = onboard()

    assert calls == []
    assert out["status"] == "needs_openrouter"
    assert readiness.COST_NOTE in out["message"]


def test_a_spent_allowance_hands_over_to_the_ordinary_approval(onboard, monkeypatch):
    """The trial DEFERS the OpenRouter step; it does not replace it. So the end of the trial
    runs the same `acquire` a user with no trial would have run on their first call."""
    calls = _at_trial_over(monkeypatch)
    out = onboard(start="openrouter")
    assert calls == ["oauth"]
    assert out["phase"] == "keys"


def test_a_spent_allowance_cannot_be_claimed_again(onboard, monkeypatch):
    """Refuse, never guess — and here guessing would spend a round trip to be told no by a
    ledger that has already recorded this person."""
    calls = _at_trial_over(monkeypatch)
    out = onboard(start="trial")
    assert calls == []
    assert out["status"] == "error"


def test_the_minting_call_says_so_once_and_asks_nothing(onboard, monkeypatch):
    """Zero friction is the point, so the allowance is NOT a question — but a user who already
    has OpenRouter must be able to find the door. One sentence, carried on the call that minted,
    alongside whatever they are already being asked. A second call would cost the turn this
    whole design exists to remove."""
    calls = _at_missing_key(monkeypatch, trial_available=True, hosted=True)
    phases = iter(["keys", "sources"])
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": next(phases),
                                      "keys": {"openrouter": {"state": "missing", "message": ""},
                                               "ok": False},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools, "_phase_sources", lambda: {"phase": "sources"})

    out = onboard()

    assert calls == ["trial"]
    assert out["phase"] == "sources"              # still lands on the real next step
    note = out["trial"].lower()
    assert "own openrouter account" in note
    assert "once" in note and "never a question" in note


def test_a_user_with_their_own_key_can_switch_while_the_allowance_is_still_live(onboard,
                                                                                monkeypatch):
    """⚠️ WITHOUT THIS the only route to your own account is to spend somebody else's money
    first: a live trial means no phase is asking for a key, so `_phase_keys` never runs and
    `start` is ignored. Honored at ANY phase, the same rule `source` follows."""
    calls = []
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "sources",
                                      "keys": {"openrouter": {"state": "ok", "message": ""},
                                               "ok": True},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools.openrouter_oauth, "acquire",
                        lambda **kw: calls.append("oauth") or {"status": "stored",
                                                               "message": "m"})

    out = onboard(start="openrouter")

    assert calls == ["oauth"]
    assert out["step"] == "openrouter"


def test_a_working_install_cannot_claim_a_second_allowance(onboard, monkeypatch):
    """Refuse, never guess. Minting here would spend the operator's money to replace a key that
    already works."""
    calls = []
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "sources",
                                      "keys": {"openrouter": {"state": "ok", "message": ""},
                                               "ok": True},
                                      "sources": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools.trial, "acquire",
                        lambda **kw: calls.append("trial") or {"status": "stored"})

    out = onboard(start="trial")

    assert calls == []
    assert out["status"] == "error"


# ── Brevity is a property of these prompts, not a matter of taste ─────────────────────────────
# ⚠️ WHAT WENT WRONG, LIVE, 2026-09-11. `_trial_prompt` listed four bullets and a closing line.
# Claude expanded them into nine sentences covering model providers, passwords, and what happens
# when credits run out — shown to somebody who had just said "onboard me". Nothing was false and
# all of it was too much. These strings are instructions to a model that EXPANDS them, so what a
# user reads is a multiple of what is written; the only reliable brake is an explicit cap in the
# instruction itself. A terse model was already fine unaided, so the cap is what makes the copy
# independent of which model is driving.

@pytest.mark.parametrize("prompt,cap", [
    (onboard_tools._trial_prompt, "one sentence"),
    (onboard_tools._trial_over_prompt, "two sentences"),
    # Added 2026-09-11 after a live run: the trial was unavailable, this prompt ran as the
    # fallback, and being uncapped it produced exactly the paragraph the other two had just
    # been cleaned of. Capping two of three surfaces only moves the problem to the third.
    (onboard_tools._openrouter_prompt, "two sentences"),
])
def test_every_prompt_caps_what_the_model_may_say(prompt, cap):
    assert cap in prompt()["message"].lower()


def test_the_sign_in_step_does_not_read_like_a_promotion():
    """"Free starter allowance" attached to a sign-in button is the shape of a scam, and the
    user it scares off is exactly the non-technical one this step was shortened for. The step is
    a sign-in; money is not being asked for yet and must not be raised yet."""
    # Only the sentence the model is told to SAY. The rest of the string is instructions ABOUT
    # that sentence, and "do NOT mention credits" legitimately contains the word it forbids.
    message = onboard_tools._trial_prompt()["message"]
    said = message.split("SAY,", 1)[1].split("\n", 1)[0].lower()
    for word in ("free", "allowance", "trial", "credit", "offer"):
        assert word not in said, f"{word!r} is in what the user is told: {said!r}"


def test_the_sign_in_step_stays_short_enough_to_survive_a_verbose_model():
    """A budget, not a style rule. The instruction and what it produces are correlated, so the
    ceiling on one is the only lever on the other."""
    message = onboard_tools._trial_prompt()["message"]
    assert len(message) < 800, f"the ask has grown back to {len(message)} chars"


# ── The end of setup says what OPYT is FOR, not only what else to feed it ─────────────────────
# ⚠️ THE GAP. Every branch of `_handoff` returned `next_tool: "oracle"`, and its only list —
# `_still_open` — enumerates further ways to give OPYT something to read. So the moment a user
# finished setting up, the product described itself entirely in terms of input, and nine
# capabilities went unmentioned. `sitting` was the sharpest loss: reading a whole topic end to
# end in publication order is the thing people describe wanting, and nothing in onboarding said
# it existed.

def test_a_finished_setup_says_what_the_thing_is_for(onboard, monkeypatch):
    _all_done(monkeypatch, candidates=12)
    message = onboard()["message"]
    # The screening list is still the next step and must survive.
    assert "oracle" in message
    # ...but it is no longer the ONLY thing a finished user is told about: the message now says
    # what the collecting is FOR. It says it without naming a tool — see the test below.
    assert "reading a whole subject" in message


def test_the_handoff_names_no_tool_it_cannot_yet_justify(onboard, monkeypatch):
    """⚠️ WHAT REPLACED A FIVE-ITEM CAPABILITY LIST, 2026-09-12.

    That list was a placeholder whose own docstring promised to shrink "when the grounded one
    lands, rather than sit beside it". It landed — `opyt_core/suggest`, reached one call later
    from `oracle(action='ingest')` — and for one turn the two sat beside each other: a general
    list of tools here, a measured choice of directions immediately after. Two sessions writing
    the same message at two adjacent moments is exactly the duplication the brevity rule exists
    to prevent.

    NAMING A TOOL HERE IS GUESSING. `_candidate_count` counts people waiting to be SCREENED, and
    no atom exists until the first ingest finishes — so a recommendation made here is made
    against a corpus that does not exist. Recommending `sitting` to someone whose store will turn
    out to hold four atoms is the fixed-menu mistake, moved one turn earlier.
    """
    _all_done(monkeypatch, candidates=12)
    assert "reading a whole subject" in onboard()["message"].lower()

    # Asserted on the FRAGMENT, not the whole message, and the difference is real: `_still_open`
    # legitimately names `sitting(action='watchlist', ...)` as a way to give OPYT a subject to
    # follow. That is an input offer, which this phase is for. What must not appear is a
    # capability pitched at a store that does not exist yet.
    block = onboard_tools._what_opyt_can_do().lower()
    assert "name no tool" in block
    for tool in ("sitting(", "search(", "frontier(", "aggregate(", "share("):
        assert tool not in block


# ── When the shortcut turns out to be closed for THIS person ──────────────────────────────────
# ⚠️ LIVE, 2026-09-11. A returning user was told "sign in with Google, nothing to create, nothing
# to pay", said yes, and was answered with: "the free starter credit isn't available on your
# install, so what I said before about not paying anything no longer applies." An assistant
# retracting its own promise in public reads as an unreliable product — a worse outcome than the
# inconvenience being described. The old preface said "do not dwell on why", which invites the
# reconciliation it forbids: a model handed a contradiction will explain it, and being told not
# to only makes the explanation apologetic.

def test_a_returning_user_is_told_the_real_reason(onboard, monkeypatch):
    """`already_claimed` is specific and harmless — they have had theirs. "Not available on your
    install" is vague and faintly alarming, and the gateway had already said which it was."""
    _at_missing_key(monkeypatch, trial_available=True,
                    trial_result={"status": "unavailable", "reason": "already_claimed",
                                  "message": "no"})
    assert "already used" in onboard(start="trial")["message"].lower()


def test_the_fallback_never_retracts_what_was_already_said(onboard, monkeypatch):
    _at_missing_key(monkeypatch, trial_available=True,
                    trial_result={"status": "unavailable", "reason": "already_claimed",
                                  "message": "no"})
    message = onboard(start="trial")["message"].lower()
    assert "do not refer back" in message
    assert "do not apologise" in message


@pytest.mark.parametrize("reason", ["already_claimed", "daily_cap", "upstream", ""])
def test_the_fallback_never_reads_like_a_promotion(reason):
    """The promotional words were stripped from the ask and leaked back in through the fallback,
    which is the usual way such wording returns: by a path nobody re-read."""
    said = onboard_tools._unavailable_preface(reason).lower()
    for word in ("free", "allowance", "trial"):
        assert word not in said, f"{word!r} came back in the {reason or 'generic'} fallback"


# ── no call waits for a pull: onboard's half ───────────────────────────────────
# ⚠️ `onboard(consent='both')` WAS CUT OFF AT 61 SECONDS on 2026-09-14 (17:33:32), and the user's
# second substantive message about the product was about a timeout. It was waiting on Arm B's
# join over ~1,000 bookmarks. R7 is overturned: the promise is now "setup is done, your saved
# posts are importing" — weaker, and true.

def test_arm_b_is_started_and_never_joined(onboard, monkeypatch):
    """The join is the defect, not the import. A thread that is joined is a call that waits."""
    handles = []
    arm = onboard_tools._body_arms()[0]

    class _Watched:
        def join(self, *a, **kw):
            handles.append("joined")

    out = onboard_tools._report_saved_content([(arm, _Watched())])

    assert handles == []
    assert out["status"] == "importing"


def test_the_saved_content_copy_never_claims_the_import_finished(onboard):
    """"Imported" said about a thread still writing is the same class of false statement as "it
    timed out" said about a pull mid-flight — it just sounds friendlier."""
    out = onboard_tools._report_saved_content([(onboard_tools._body_arms()[0], None)])

    said = out["message"].lower()
    assert "importing now" in said
    for claim in ("are imported", "have been imported", "finished importing", "all done"):
        assert claim not in said
    assert "never that it has finished" in out["host_note"]


def test_nothing_started_reports_nothing(onboard):
    """A user who consented to neither backlog has no import to be told about, and inventing a
    sentence for one would be describing work that is not happening."""
    assert onboard_tools._report_saved_content([]) is None


def test_arm_a_is_started_and_the_consent_prompt_does_not_wait_for_it(onboard, monkeypatch):
    """~30s and ~25 requests, spent with the user watching a blank screen BEFORE the question
    they are waiting for. Started instead, it runs behind their reading time."""
    ran = []
    monkeypatch.setattr(onboard_tools, "_spawn", lambda target: ran.append("spawned"))
    monkeypatch.setattr(onboard_tools, "_run_curation",
                        lambda p=None: ran.append("ran") or {"status": "ok"})

    out = onboard_tools._start_arm_a({"x"})

    assert ran == ["spawned"]                 # spawned, and this call did not run it
    assert out["status"] == "scoring"
    assert "Do not wait for it" in out["host_note"]


def test_the_host_is_told_not_to_screen_in_the_same_turn(onboard, monkeypatch):
    """The screen is SCORED on these signals — a candidate list built before the walk lands is
    not a thin list, it is a WRONG one, with no symptom. The ordering holds with room to spare
    because `screen` is turns away; this is what keeps it that way."""
    monkeypatch.setattr(onboard_tools, "_spawn", lambda target: None)

    assert "do not call `screen` in this same turn" in \
        onboard_tools._start_arm_a({"x"})["host_note"]


def test_a_signal_walk_that_cannot_start_says_so_rather_than_claiming_it_did(onboard, monkeypatch):
    """Fail-safe, and in the honest direction: a candidate list nothing is building must not be
    described as being built."""
    monkeypatch.setattr(onboard_tools, "_spawn", lambda target: 1 / 0)

    assert onboard_tools._start_arm_a({"x"})["status"] == "not_started"


# ── the platform connected AFTER the consent answer ─────────────────────────────
#
# THE DEFECT THESE EXIST FOR, found 2026-09-14 in a real session: consent was answered at 22:47
# with only X live, Substack was connected at 23:03, the saved posts imported in-process — and
# `substack_saved_catchup` never got a job row, so the recurring sync never ran once. This is
# the ORDINARY path, not an edge case: `_phase_sources` asks for one root and lists the rest as
# "still open", and `_consent_prompt` is put before any of the others can be connected.
def test_connecting_substack_after_consent_queues_its_recurring_rail(onboard, monkeypatch,
                                                                     tmp_path):
    _at_curation(monkeypatch)
    _at_sources(monkeypatch, connected=True, substack=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})
    (tmp_path / "substack_saved_catchup_consent").touch()   # answered `both` in an earlier call
    (tmp_path / "bookmark_catchup_consent").touch()

    onboard()

    job = _jobs(tmp_path).get("substack_saved_catchup")
    assert job is not None and job.due_at <= time.time(), (
        "the saved posts are importing right now — with no row, they import exactly once and "
        "never again, which is the 2026-09-14 defect verbatim"
    )


def test_an_unconsented_platform_is_not_queued_by_connecting_it(onboard, monkeypatch, tmp_path):
    """Connecting a source is not consent to walk what it saved. The marker is the only yes."""
    _at_curation(monkeypatch)
    _at_sources(monkeypatch, connected=True, substack=True)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    onboard()

    assert "substack_saved_catchup" not in _jobs(tmp_path)


def test_substack_first_then_x_queues_every_rail(onboard, monkeypatch, tmp_path):
    """THE MIRROR ORDER. Every other test here walks X-first, because that is the order the
    sources prompt lists and the order every measured session happened to take — which is exactly
    why the reverse is worth asserting: the consent question is answered against whichever ONE
    platform is live at the time, and the other one connects afterwards. Whichever platform that
    is, its backlog rail has to be queued by the connect rather than by the answer.
    """
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 0)
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda *a: {"status": "stubbed", "ran": {}})

    _at_sources(monkeypatch, connected=False, substack=True)
    assert onboard()["status"] == "needs_consent"

    onboard(consent="both")
    assert "substack_saved_catchup" in _jobs(tmp_path)
    assert "bookmark_catchup" not in _jobs(tmp_path), "X is not connected — it has no backlog yet"

    _at_sources(monkeypatch, connected=True, substack=True)   # X arrives afterwards
    onboard()

    jobs = _jobs(tmp_path)
    assert {"substack_saved_catchup", "bookmark_catchup", "curation_catchup"} <= set(jobs)
    assert all(jobs[r].due_at <= time.time() + 1 for r in jobs)
