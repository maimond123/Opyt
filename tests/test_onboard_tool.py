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

import pytest

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
    m = _MCP()
    onboard_tools.register_onboard_tools(m)
    return m.tools["onboard"]


def test_registers_a_tool_named_onboard(onboard):
    assert onboard is not None


def test_no_argument_is_ever_a_credential(onboard):
    names = set(inspect.signature(onboard).parameters)
    assert not {n for n in names if "key" in n or "token" in n or "secret" in n}


def test_missing_openrouter_runs_oauth_and_does_not_spend(onboard, monkeypatch):
    calls = []
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "keys",
                                      "keys": {"openrouter": {"state": "missing", "message": ""},
                                               "ok": False},
                                      "browser": {}, "consent": {}, "curation": {}})
    monkeypatch.setattr(onboard_tools.openrouter_oauth, "acquire",
                        lambda **kw: calls.append("oauth") or {"status": "stored",
                                                               "message": "m"})
    monkeypatch.setattr(onboard_tools.key_paste, "acquire",
                        lambda *a, **kw: calls.append("paste") or {"status": "stored",
                                                                   "message": "m"})
    out = onboard()
    assert calls == ["oauth"]                 # ⚠️ OpenRouter FIRST, and alone
    assert out["phase"] == "keys"


def test_unfunded_openrouter_refuses_and_says_the_pull_was_free(onboard, monkeypatch):
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "keys",
                                      "keys": {"openrouter": {"state": "unfunded",
                                                              "message": "add credit"},
                                               "ok": False},
                                      "browser": {}, "consent": {}, "curation": {}})
    out = onboard()
    assert out["status"] == "blocked"
    assert "free" in out["message"].lower()
    assert "credit" in out["message"].lower()


def test_semantic_scholar_is_never_mentioned(onboard, monkeypatch):
    monkeypatch.setattr(onboard_tools.onboard_state, "derive",
                        lambda **kw: {"phase": "done", "keys": {}, "browser": {},
                                      "consent": {}, "curation": {}})
    blob = (str(onboard()) + (onboard.__doc__ or "")).lower()
    assert "semantic scholar" not in blob and "s2_api_key" not in blob


# ── phase 1: the browser session ────────────────────────────────────────────
#
# These helpers patch derive()'s INPUTS, never derive() itself, so the real phase logic runs.

def _keys_green(monkeypatch):
    monkeypatch.setattr(onboard_tools.onboard_state.readiness, "openrouter",
                        lambda: {"state": "ok", "message": ""})


def _at_browser(monkeypatch, profiles=(), blocked=None):
    _keys_green(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_x_profiles",
                        lambda: (list(profiles), blocked))
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_ok", lambda: False)


def test_browser_phase_warns_before_it_reads(onboard, monkeypatch):
    """⚠️ THE RE-ENTRY CASE. Keys are already done, so this call is the FIRST one — there is no
    previous payload to carry the pre-warn. It must warn and read NOTHING."""
    from pipeline.ingestion import browser_cookies as bc
    reads = []
    _keys_green(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_x_profiles",
                        lambda: reads.append(1) or ([], None))
    monkeypatch.setattr(onboard_tools, "_warned", lambda: False)
    # Pin the machine's shape: with a Keychain-gated backend installed, the warn must say so.
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("arc")])
    out = onboard()
    assert reads == []                                   # did not touch the browser
    assert "keychain" in out["message"].lower()


def test_the_warn_round_trip_survives_a_machine_with_no_native_gate(onboard, monkeypatch):
    """Chromium reads no longer raise any dialog, so on the common machine there is none to
    describe. The round trip still happens — it is where the user learns an agent is about to
    read their browser at all — but the copy must not promise a prompt that never comes."""
    from pipeline.ingestion import browser_cookies as bc
    reads = []
    _keys_green(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_x_profiles",
                        lambda: reads.append(1) or ([], None))
    monkeypatch.setattr(onboard_tools, "_warned", lambda: False)
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("chrome")])
    out = onboard()
    assert reads == []
    assert out["status"] == "warned"
    msg = out["message"].lower()
    assert "keychain" not in msg and "full disk access" not in msg
    assert "browser" in msg


def test_prewarn_is_backend_derived_not_hardcoded(onboard, monkeypatch):
    """Firefox reads its own cookie store — no native prompt exists to warn about. The old
    hardcoded copy promised a Chrome Keychain dialog regardless of what was installed; the
    warn must now derive from the backends actually present (guard: hardcoded-consent-prewarn)."""
    from pipeline.ingestion import browser_cookies as bc
    _keys_green(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_x_profiles", lambda: ([], None))
    monkeypatch.setattr(onboard_tools, "_warned", lambda: False)
    monkeypatch.setattr(bc, "installed_backends", lambda: [bc.backend_for("firefox")])
    out = onboard()
    msg = out["message"].lower()
    assert "keychain" not in msg and "chrome" not in msg


def test_the_warning_call_records_that_it_warned(onboard, monkeypatch, tmp_path):
    """Otherwise the second call warns again and never reads — an infinite pre-warn."""
    _at_browser(monkeypatch)
    onboard()
    assert onboard_tools._warned() is True


def test_second_call_actually_reads(onboard, monkeypatch):
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch, profiles=[{"profile": "Default", "label": "@you"}])
    monkeypatch.setattr(onboard_tools, "_save_profile", lambda p: None)
    assert onboard()["phase"] != "browser"


def test_several_profiles_asks_as_a_chat_argument(onboard, monkeypatch):
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch, profiles=[{"profile": "Default", "label": "@a"},
                                       {"profile": "Profile 1", "label": "@b"}])
    out = onboard()
    assert out["status"] == "needs_choice"
    assert "browser_profile" in out["message"]
    assert {c["profile"] for c in out["choices"]} == {"Default", "Profile 1"}


def test_a_chosen_profile_is_persisted_to_settings(onboard, monkeypatch):
    saved = {}
    monkeypatch.setattr(onboard_tools, "_save_profile", lambda p: saved.update(p=p))
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch, profiles=[{"profile": "Profile 1", "label": "@b"}])
    onboard(browser_profile="Profile 1")
    assert saved["p"] == "Profile 1"


def test_an_unknown_profile_is_refused_not_silently_ignored(onboard, monkeypatch):
    saved = {}
    monkeypatch.setattr(onboard_tools, "_save_profile", lambda p: saved.update(p=p))
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch, profiles=[{"profile": "Default", "label": "@a"},
                                       {"profile": "Profile 1", "label": "@b"}])
    out = onboard(browser_profile="Profile 9")
    assert out["status"] == "needs_choice"
    assert not saved                       # persisting a bad pick would break every later read


def test_blocked_read_surfaces_remediation_verbatim(onboard, monkeypatch):
    """browser_cookies.remediation already separates keychain_denied from fda_needed from
    'not logged in', with the right fix for each. Do not rewrite this copy."""
    from pipeline.ingestion import browser_cookies as bc
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    want = bc.remediation("keychain_denied", bc.backend_for("arc"), "X")
    _at_browser(monkeypatch, blocked={"kind": "keychain_denied", "browser": "arc"})
    assert want in onboard()["message"]


def test_no_session_and_nothing_blocked_offers_a_guided_login(onboard, monkeypatch):
    """A user with no X session anywhere is not sent away — Opyt offers to open a browser
    they can log into. The offer is an ARGUMENT, like every other decision this tool takes,
    so nothing opens until they say so."""
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch)
    out = onboard()
    assert out["status"] == "needs_login"
    assert "x.com" in out["message"].lower()
    assert "guided_login" in out["message"]


def test_accepting_the_offer_opens_a_browser_and_returns(onboard, monkeypatch):
    """The window has to outlive the call: a tool call that blocks on a human times out."""
    from pipeline.ingestion import browser_cookies as bc, guided_login
    started = {}
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch)
    monkeypatch.setattr(guided_login, "start", lambda url, browser=None: (
        started.update(url=url) or bc.backend_for("chrome")))
    out = onboard(guided_login=True)
    assert started["url"] == "https://x.com/login"
    assert out["status"] == "awaiting_login"
    assert "Chrome" in out["message"]


def test_a_started_login_that_is_not_finished_says_so(onboard, monkeypatch):
    """"No session" means something different once Opyt has opened a window: re-offering the
    login would read as if the first one never happened."""
    from pipeline.ingestion import browser_cookies as bc
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch)
    monkeypatch.setattr(bc, "opyt_session_backends", lambda: [bc.backend_for("chrome")])
    out = onboard()
    assert out["status"] == "awaiting_login"
    assert "Finish logging in" in out["message"]


def test_no_launchable_browser_falls_back_to_plain_instructions(onboard, monkeypatch):
    """guided_login raises the same typed error as every other 'no session' path, so the
    branch stays one except — and the user still gets something to do."""
    from pipeline.ingestion import guided_login
    from pipeline.ingestion.utils import SyncAuthError
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    _at_browser(monkeypatch)

    def boom(url, browser=None):
        raise SyncAuthError("Opyt could not find a browser it can open for you to log in")

    monkeypatch.setattr(guided_login, "start", boom)
    out = onboard(guided_login=True)
    assert out["status"] == "needs_login"
    assert "could not find a browser" in out["message"]


def test_save_profile_never_writes_the_packaged_default(monkeypatch, tmp_path):
    """⚠️ `config_path()` falls back to the REPO/packaged settings.yaml when no user config
    exists. That file is read-only in a pip install and is the author's template besides — a
    mutable value must resolve through a WRITE path, the pattern the deleted `taxonomy_write_path`
    documented."""
    import yaml
    from opyt_core import config
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv("OPYT_CONFIG", raising=False)
    packaged = config.config_path()                     # no user config yet → the default
    before = packaged.read_text()

    onboard_tools._save_profile("Profile 7")

    assert packaged.read_text() == before               # untouched
    written = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    assert written["cookies"]["profile"] == "Profile 7"


# ── phase 2: consent ────────────────────────────────────────────────────────

def _at_consent(monkeypatch):
    """Keys green, browser settled, question not yet put."""
    _keys_green(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_x_profiles",
                        lambda: ([{"profile": "Default", "label": "@you"}], None))
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_ok", lambda: False)
    monkeypatch.setattr(onboard_tools, "_warned", lambda: True)
    monkeypatch.setattr(onboard_tools, "_save_profile", lambda p: None)
    monkeypatch.setattr(onboard_tools, "_spawn_bookmarks", lambda: True)
    monkeypatch.setattr(onboard_tools, "_spawn_refresh", lambda: True)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 0)
    # ⚠️ STUB THE CURATION RUN. Answering consent falls THROUGH into the curation phase in the
    # same call, so without this every consent test fires a real collector pass — ~9s each of
    # live cookie reads. `_at_curation` restores the real one.
    monkeypatch.setattr(onboard_tools, "_run_curation", lambda: {"status": "stubbed", "ran": {}})


def test_no_consent_argument_asks_and_writes_nothing(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    out = onboard()
    assert out["status"] == "needs_consent"
    # Read the figure off the constant, not a copy of it — a re-measure should not need a test edit.
    assert onboard_tools._MONTHLY_AT_50 in out["message"] and "month" in out["message"]
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


def test_backlog_consent_SPAWNS_because_the_work_exists_now(onboard, monkeypatch):
    fired = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_spawn_bookmarks", lambda: fired.append("bm") or True)
    onboard(consent="backlog")
    assert fired == ["bm"]


def test_refresh_consent_does_NOT_spawn_on_an_empty_roster(onboard, monkeypatch):
    fired = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_spawn_refresh", lambda: fired.append("rf") or True)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 0)
    onboard(consent="refresh")
    assert fired == []          # a child that finds zero pairs is a no-op dressed as an action


def test_refresh_consent_DOES_spawn_when_oracles_already_exist(onboard, monkeypatch):
    fired = []
    _at_consent(monkeypatch)
    monkeypatch.setattr(onboard_tools, "_spawn_refresh", lambda: fired.append("rf") or True)
    monkeypatch.setattr(onboard_tools, "_confirmed_oracles", lambda: 8)
    onboard(consent="refresh")
    assert fired == ["rf"]      # re-entry on a populated store: the work IS waiting


def test_saying_no_on_re_entry_revokes(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    (tmp_path / "oracle_refresh_consent").touch()
    onboard(consent="backlog")
    assert not (tmp_path / "oracle_refresh_consent").exists()


def test_an_unknown_consent_word_is_refused_not_guessed(onboard, monkeypatch, tmp_path):
    _at_consent(monkeypatch)
    out = onboard(consent="yes")
    assert out["status"] == "error"
    assert not (tmp_path / "onboard_consent_asked").exists()


def test_the_prompt_states_both_cost_shapes(onboard, monkeypatch):
    """One-time-and-bounded vs recurring-forever are different commitments. Quoting only a
    per-item rate is useless — a user cannot apply one to an item count they do not know."""
    _at_consent(monkeypatch)
    msg = onboard()["message"].lower()
    assert "one-time" in msg or "one time" in msg
    assert "recurring" in msg or "every" in msg or "keeps" in msg


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
                        lambda: called.append("catchup") or {"status": "ok", "ran": {}})
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

def _all_done(monkeypatch, candidates=0):
    _at_curation(monkeypatch)
    monkeypatch.setattr(onboard_tools.onboard_state, "_curation_ok", lambda: True)
    monkeypatch.setattr(onboard_tools, "_candidate_count", lambda: candidates)


def test_done_names_oracle_as_the_next_call(onboard, monkeypatch):
    _all_done(monkeypatch)
    out = onboard()
    assert out["phase"] == "done" and out["next_tool"] == "oracle"


def test_handoff_bounds_the_first_ingest(onboard, monkeypatch):
    """⚠️ `oracle(action='ingest')` is a synchronous foreground loop over EVERY confirmed Oracle
    with no pick cap and no time budget. Fixing that is out of scope, but `onboard` is what sends
    people into it — so the copy bounds the first pass."""
    _all_done(monkeypatch)
    msg = onboard()["message"]
    assert "3" in msg or "three" in msg.lower()


def test_done_reports_what_actually_landed(onboard, monkeypatch):
    _all_done(monkeypatch, candidates=42)
    assert onboard()["candidates"] == 42


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
