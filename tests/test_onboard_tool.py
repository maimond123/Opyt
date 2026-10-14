"""
tests/test_onboard_tool.py

`onboard` — the thin orchestrator. The properties pinned here are the ones that make it safe
rather than the ones that make it work:

  • NO ARGUMENT MAY EVER CARRY A CREDENTIAL. Secrets travel over loopback; decisions travel
    over chat. A test asserts the signature, so adding an `api_key=` parameter fails the suite.
  • An UNFUNDED OpenRouter account BLOCKS rather than passing through — a store
    that builds and can never be queried is not worth building.
  • Semantic Scholar is never mentioned: AI2 no longer approves third-party key
    requests, so a step telling a user to go get one cannot succeed.
"""

from pipeline.platforms import connections
from pipeline.sharing import account
from pipeline.worker.home import LOCAL_HOME_ID
from pipeline.worker import job_queue, liveness
from pipeline.worker import home as worker_home

import asyncio
import inspect
import re
import time

import pytest

from mcp_server import onboard_tools
from pipeline.curation import collectors
from pipeline.onboarding import handoff, kickoff, phase, roots


class _MCP:
    def __init__(self):
        self.tools = {}

    def tool(self, **kw):
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


def test_a_fresh_home_is_asked_for_a_source_first(onboard, monkeypatch):
    """No keys question before sources (it rides the first Oracle
    confirmation), and nothing said about models: Opyt calls none. `derive` runs for real here;
    only the session probes and the worker are stood in for."""
    _worker_reports(monkeypatch, {"ok": True, "status": "running"})
    for env in ("GITHUB_TOKEN", "OPENALEX_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(connections, "_connected_substack", lambda: False)
    out = onboard()
    assert (out["status"], out["phase"]) == ("needs_source", "sources")
    assert "free" not in out


def test_an_unrecognized_start_word_opens_nothing(onboard, monkeypatch):
    """Refuse, never guess. Guessing here opens the very tab the split exists to announce.
    `openrouter` is a retired word: connecting your own key is gone."""
    calls = _worker_reports(monkeypatch, {"ok": True, "status": "running"})
    monkeypatch.setattr(phase, "derive",
                        lambda **kw: {"phase": "sources", "sources": {}, "curation": {}})
    out = onboard(start="openrouter")
    assert out["status"] == "error"
    assert "Nothing was opened" in out["message"]
    assert calls == []


def test_the_semantic_scholar_KEY_is_never_advertised(onboard, monkeypatch):
    """AI2 no longer approves third-party key requests, so telling a user to get one sends them
    to a form that will reject them.

    The ban is on the CREDENTIAL, not on the company: the author page is a public URL that
    `oracles._scholar_root` accepts and `_unsupported_root` already recommends by name, and a
    27% share of researchers carry no ORCID (measured over 256 authorships), so
    banning the page as well would leave that share of the research root unreachable."""
    monkeypatch.setattr(phase, "derive",
                        lambda **kw: {"phase": "done",
                                      "sources": {"x": False, "substack": False},
                                      "curation": {"applicable": False, "pending": []}})
    blob = (str(onboard()) + (onboard.__doc__ or "")).lower()
    assert "s2_api_key" not in blob
    for phrase in ("semantic scholar key", "semantic scholar api", "semanticscholar.org/api"):
        assert phrase not in blob


# ── phase 1: a source ───────────────────────────────────────────────────────
#
# These helpers patch derive()'s INPUTS, never derive() itself, so the real phase logic runs.

def _at_sources(monkeypatch, *, connected: bool = False, substack: bool = False, **store):
    """Nothing connected and an empty store — so the sources phase is outstanding.

    ⚠️ BOTH session probes are stubbed, always. They scan `opyt_session_backends()`, and a test
    that fakes that list to exercise one probe otherwise sends the other one into the machine's
    REAL browser cookie store — where the developer running the suite is quite possibly logged
    into Substack, and the phase then advances for a reason the test never set up."""

    monkeypatch.setattr(connections, "_connected_x", lambda: connected)
    monkeypatch.setattr(connections, "_connected_substack", lambda: substack)
    # The "done"-too-early gate verifies a connected Substack session is LIVE, not just present,
    # before it lets the curation walk build on it. Default it to ready so a test that
    # sets up a connected session gets the phase it asked for; the gate's own tests set it False.
    monkeypatch.setattr(connections, "managed_substack_session_ready", lambda **kw: True)
    monkeypatch.setattr(phase, "_store_sources",
                        lambda: {**phase._NO_STORE_SOURCES, **store})
    monkeypatch.setattr(collectors, "curation_pending",
                        lambda live: sorted(live))   # nothing read yet


def test_nothing_connected_offers_every_root_not_just_x(onboard, monkeypatch):
    """The earlier version of this phase asked one question — is X connected —
    with no bypass, so a user who reads blogs or watches research topics never got past it.

    The contract is that all four roots are reachable from this one response, and reachable as
    ANSWERS — every root is a `source=` word rather than prose pointing
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
    """`roots.SOURCE_WORDS` derives from two dicts plus `skip`, so a word can never be accepted with
    nothing behind it. Asserted rather than guarded: the alternative is a runtime check on a
    keyspace this module owns outright."""
    assert set(roots.SOURCE_WORDS) == (
        set(roots.LOGIN_URLS) | set(roots.NAMED_ROOTS) | {"skip"})
    assert not set(roots.LOGIN_URLS) & set(roots.NAMED_ROOTS)


@pytest.mark.parametrize("source,must_name", [("research", "orcid.org"),
                                              ("blog", "add_handles")])
def test_a_named_root_returns_the_call_to_make(onboard, monkeypatch, source, must_name):
    """A root with no account returns the QUESTION to put and the CALL to make with the answer.

    `oracle` owns those writes; this phase points at it and never proxies it,
    so what a named root owes the caller is the exact invocation."""
    _at_sources(monkeypatch)

    out = onboard(source=source)

    assert out["status"] == "needs_name"
    assert out["source"] == source
    assert "oracle(action='confirm'" in out["message"]
    assert must_name in out["message"]


def test_no_onboarding_surface_creates_a_standing_watch(onboard, monkeypatch):
    """Onboarding gets content, not commitments. The research root's subject path once kept
    its watch by exception, on the argument that the USER chose a subject. A live transcript
    showed what that choice really was: the host framed a binary,
    the user picked the word "subject" while thinking "here is what I am into", and nine
    interests became three commitments over an empty store. No exceptions survive: no
    onboarding surface names the call that creates a watch, any root, any branch."""
    _at_sources(monkeypatch)
    assert "watch(action=" not in onboard(source="research")["message"]
    assert "watch(action=" not in onboard(source="blog")["message"]
    for candidates in (0, 40):
        _all_done(monkeypatch, candidates=candidates)
        assert "watch(action=" not in onboard()["message"]


def test_a_named_root_assigns_the_lookup_to_the_host_not_the_user(onboard, monkeypatch):
    """This copy once read "Have the user search '<name> ORCID' … and
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
    """The dispatch splits on `roots.LOGIN_URLS` membership. A fallthrough would launch a sign-in
    window at a site the user never named — the same harm `roots.SOURCE_WORDS` refuses an unknown
    word for."""
    from pipeline.platforms.browser import login

    _at_sources(monkeypatch)
    monkeypatch.setattr(login, "start",
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
    from pipeline.platforms.browser import cookies as bc, login

    started = []
    _at_sources(monkeypatch)
    monkeypatch.setattr(login, "start",
                        lambda url: started.append(url) or bc.backend_for("chrome"))

    out = onboard(source="x")

    assert started == ["https://x.com/login"]
    assert out["status"] == "awaiting_login"


def test_an_unknown_source_word_is_refused_not_guessed(onboard, monkeypatch):
    """Guessing here would open a browser window at a site the user never named."""
    from pipeline.platforms.browser import login

    _at_sources(monkeypatch)
    monkeypatch.setattr(login, "start",
                        lambda url: pytest.fail("nothing may open for an unknown word"))

    out = onboard(source="twitter")

    assert out["status"] == "error"


def test_skip_records_the_question_was_put_and_moves_on(onboard, monkeypatch, tmp_path):
    """A marker, because every OTHER way out of this phase is a disk fact, and
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
    from pipeline.platforms.browser import cookies as bc

    _at_sources(monkeypatch)
    monkeypatch.setattr(bc, "opyt_session_backends", lambda: [bc.backend_for("chrome")])

    msg = onboard()["message"]

    assert "`onboard`" in msg and "onboard(source='x')" in msg


def test_source_x_reports_a_missing_launchable_browser(onboard, monkeypatch):
    from pipeline.platforms.browser import login
    from pipeline.ingestion.utils import SyncAuthError

    _at_sources(monkeypatch)
    monkeypatch.setattr(login, "start",
                        lambda url: (_ for _ in ()).throw(SyncAuthError("no launchable browser")))

    out = onboard(source="x")

    assert out["status"] == "needs_login"
    assert "launchable browser" in out["message"]


def test_source_substack_opens_the_substack_sign_in_page(onboard, monkeypatch):
    from pipeline.platforms.browser import cookies as bc, login

    started = []
    _at_sources(monkeypatch)
    monkeypatch.setattr(login, "start",
                        lambda url: started.append(url) or bc.backend_for("chrome"))

    out = onboard(source="substack")

    assert started == ["https://substack.com/sign-in"]
    assert out["status"] == "awaiting_login"


def test_a_substack_session_alone_gets_the_user_past_the_source_phase(onboard, monkeypatch):
    _at_sources(monkeypatch, substack=True)

    assert onboard()["phase"] != "sources"


def test_a_managed_x_session_advances_past_sources(onboard, monkeypatch):
    _at_sources(monkeypatch, connected=True)

    assert onboard()["phase"] != "sources"


def test_signature_is_two_decisions_and_nothing_else(onboard):
    """`connect_x` is one value of `source` rather than a second way to say the same thing.

    `start` passes both bars: the flow reads it, and it opens no door that another argument
    already opens — each word begins a sign-in nothing else can begin. Nothing is asked, so
    there is no consent argument."""
    assert set(inspect.signature(onboard).parameters) == {"source", "start"}


# ── connecting a platform is what starts its work ──────────
#
# There is no consent question. The first `onboard` call that sees a platform connected queues the
# walk of who they follow and the import of what they saved there, and nothing else is needed.

def _connected(monkeypatch):
    """X connected, nothing read yet."""
    _at_sources(monkeypatch, connected=True)


def _jobs(home) -> dict:
    """Every durable rail job the worker would find in this home, keyed by rail."""
    from pipeline.worker.job_queue import RailJobStore
    return {j.rail: j for j in RailJobStore(home / "rail_jobs.db").list_jobs()
            if j.home_id == LOCAL_HOME_ID}


def test_leaving_the_source_phase_WITHOUT_a_session_queues_no_walk(onboard, monkeypatch, tmp_path):
    """A user whose root is a named blog — or who answered `skip` — has connected no session, so
    there is nothing to walk and nothing to import."""
    _at_sources(monkeypatch, oracles=True)

    onboard()

    assert "scrapers" not in _jobs(tmp_path)


def test_a_present_but_not_live_substack_session_holds_at_done(onboard, monkeypatch, tmp_path):
    """The "done"-too-early guard. `derive` marks Substack connected on cookie PRESENCE, but a
    just-signed-in session can be present without answering an authenticated request yet. Onboarding
    must HOLD — not build the curation pull on it and report "nobody".
    Real failure: collectors skipped, 36 subscriptions sitting in the account."""
    _at_sources(monkeypatch, substack=True)
    monkeypatch.setattr(connections, "managed_substack_session_ready", lambda **kw: False)

    out = onboard()

    assert out["status"] == "awaiting_login"
    assert "signed in yet" in out["message"]
    assert "scrapers" not in _jobs(tmp_path)        # no walk and no import woken on a dead session


def test_a_verifier_error_does_not_strand_a_user_who_did_sign_in(onboard, monkeypatch, tmp_path):
    """Fail-safe direction: if the readiness check itself throws, the flow must PROCEED, not trap a
    signed-in user behind a broken probe. The collector records a real skip if the session is
    actually dead — it never invents a connection."""
    _at_sources(monkeypatch, substack=True)

    def _boom(**kw):
        raise RuntimeError("probe blew up")
    monkeypatch.setattr(connections, "managed_substack_session_ready", _boom)

    out = onboard()

    assert out["status"] != "awaiting_login"
    assert "scrapers" in _jobs(tmp_path)


def test_connecting_queues_the_walk_and_the_import_on_one_call(onboard, monkeypatch, tmp_path):
    """A permanent gap, measured on a clean install where `curation_catchup` had run and had no
    rail row and no log. The walk is woken on the call that first SEES the session; a queue
    that waited for a later call could find `pending` empty and never fire. The row IS the
    walk — the worker runs it — so this is also the test that the walk happens at all.

    And nothing is asked first: the same call queues the import, and answers with the curation
    phase rather than a question."""
    _connected(monkeypatch)

    out = onboard()

    jobs = _jobs(tmp_path)
    assert out["phase"] == "curation" and out["status"] != "needs_consent"
    assert out["curation"]["status"] == "queued"
    assert out["backlog_queued"] == ["bookmarks_import"]
    assert jobs["scrapers"].due_at <= time.time() and jobs["scrapers"].started_at is None


def test_setup_ends_owing_the_user_every_rail_a_connection_promises(onboard, monkeypatch, tmp_path):
    """The whole end state. A clean install once finished setup with two rows in
    `rail_jobs.db`, and nothing asserted the SET, so nothing failed.

    The people collectors and both saved-list imports are walks in the
    `scrapers` process, so connecting a platform owes the user exactly that one job."""
    from pipeline.worker.registry import RAILS

    _at_sources(monkeypatch, connected=True, oracles=True)

    onboard()

    assert set(_jobs(tmp_path)) == {"scrapers"}
    assert set(_jobs(tmp_path)) <= set(RAILS), "a name the worker cannot launch is never run"


def test_a_hosted_onboarding_queues_the_same_rails_into_the_shared_database(
        onboard, monkeypatch, tmp_path):
    """Nothing in this module branches on hosted vs local for
    WHETHER to queue, and this asserts it.

    Hosted means two environment values and nothing else: an already-validated subject as the
    home id, and the operator's ONE shared control database, which a child inherits precisely so a product action can queue durable work. The
    rows must land THERE, under that subject — a row in the child's own home is one the single
    worker never opens.
    """
    from pipeline.worker.job_queue import RailJobStore

    subject = "104729183746152938471"
    shared = tmp_path / "var" / "rail_jobs.db"
    monkeypatch.setenv(worker_home.WORKER_HOME_ID_ENV, subject)
    monkeypatch.setenv(worker_home.WORKER_DB_ENV, str(shared))
    _connected(monkeypatch)

    onboard()

    assert {(j.home_id, j.rail) for j in RailJobStore(shared).list_jobs()} == {
        (subject, "scrapers")}
    assert not (tmp_path / "rail_jobs.db").exists(), (
        "a hosted row in the child's own home is one the single worker never opens — the silent "
        "total failure `worker_db_path()` raises to prevent"
    )


def test_only_a_connected_platform_gets_its_import_queued(onboard, monkeypatch, tmp_path):
    """A rail that finds no session is a no-op dressed as an action, whose recorded failure looks
    like a real one. X's import waits until X is connected."""
    _at_sources(monkeypatch, substack=True)

    out = onboard()

    assert out["backlog_queued"] == ["substack_saved_import"]


def test_a_failed_queue_says_it_is_not_queued(onboard, monkeypatch, tmp_path):
    """An operator's misconfigured worker database must not be reported as work under way."""

    _connected(monkeypatch)
    monkeypatch.setattr(job_queue, "request_now", lambda rail, **kw: False)

    out = onboard()

    assert out["curation"]["status"] == "not_queued"
    assert "backlog_queued" not in out


def test_setup_writes_no_consent_marker(onboard, monkeypatch, tmp_path):
    """The five files the consent question used to leave behind are gone with it.
    Nothing reads them, so writing one would be a fact nobody can act on."""
    _connected(monkeypatch)

    onboard()

    for name in ("onboard_consent_asked", "bookmark_catchup_consent",
                 "substack_saved_catchup_consent", "oracle_refresh_consent",
                 "curation_catchup_consent"):
        assert not (tmp_path / name).exists(), name


def _worker_reports(monkeypatch, report: dict) -> list:
    """Stub the repair-or-report helper; returns the list each call appends to."""
    from pipeline.worker import liveness
    calls = []
    monkeypatch.setattr(liveness, "repair_or_report",
                        lambda **kw: calls.append(kw) or report)
    return calls


def _installs(monkeypatch) -> list:
    """Drive the REAL helper against the conftest's tmp plist, recording each install."""
    from opyt_core import install_worker
    from pipeline.worker import liveness
    installs = []
    real_install = install_worker.install
    monkeypatch.setattr(install_worker, "install",
                        lambda **kw: installs.append(True) or real_install(**kw))
    monkeypatch.setattr(liveness, "_wait_for_beat", lambda patience: True)
    return installs


def test_the_sources_step_installs_the_worker_and_says_nothing(onboard, monkeypatch, tmp_path):
    """Every call that takes a step installs the worker. Nothing ingests
    without it, so it has to exist before the first work is queued. And silently: macOS
    only notifies, with nothing to approve, so there is nothing for the user to do."""
    installs = _installs(monkeypatch)
    _at_sources(monkeypatch)

    out = onboard()

    assert out["phase"] == "sources"
    assert installs, "the worker must be in place before anything is queued for it"
    assert "worker" not in out and "worker_problem" not in out


def test_a_refused_word_installs_nothing(onboard, monkeypatch):
    """"Nothing was opened" / "nothing was recorded" must stay true of the whole call."""
    calls = _worker_reports(monkeypatch, {"ok": True, "status": "running"})
    _at_sources(monkeypatch)

    out = onboard(source="myspace")

    assert out["status"] == "error"
    assert calls == []


def test_a_running_worker_says_nothing(onboard, monkeypatch):
    """Nothing to warn about and nothing new to explain — so no key at all."""
    _connected(monkeypatch)
    _worker_reports(monkeypatch, {"ok": True, "status": "running"})

    out = onboard()

    assert "worker" not in out and "worker_problem" not in out


def test_the_promises_wait_for_the_worker_when_it_is_down(onboard, monkeypatch, tmp_path):
    """One seam per rule. "Call again in a few minutes" is a promise the worker
    keeps, so it reads the same worker check the reply reports — never one key promising while
    another warns."""
    _connected(monkeypatch)
    _worker_reports(monkeypatch, {"ok": False, "status": "refused",
                                  "cause": "macOS stopped it.", "step": "Turn it on."})

    out = onboard()

    assert out["phase"] == "curation"
    assert "in a few minutes" not in out["message"]
    assert "they will be read as soon as" in out["message"]
    assert "Turn it on." in out["message"]
    assert out["worker_problem"]["cause"] == "macOS stopped it."


def test_a_broken_worker_is_reported_with_its_cause_and_step(onboard, monkeypatch, tmp_path):
    """FIX, NEVER FALL BACK (ruling 4). The work is queued, but a host that is not told will go
    on saying "it's importing", which is the false claim `liveness.status` was written to
    prevent."""
    _connected(monkeypatch)
    _worker_reports(monkeypatch, {"ok": False, "status": "refused",
                                  "cause": "macOS stopped OPYT's background helper from running.",
                                  "step": "Open System Settings …", "detail": {"x": 1}})

    out = onboard()

    assert "scrapers" in _jobs(tmp_path)                       # the work waits, it is not lost
    problem = out["worker_problem"]
    assert problem["status"] == "refused"
    assert problem["cause"].startswith("macOS stopped")
    assert problem["step"].startswith("Open System Settings")
    assert "Do NOT say anything is importing" in problem["message"]
    assert "worker" not in out


def test_a_hosted_home_is_not_told_its_worker_is_dead(onboard, monkeypatch, tmp_path):
    """A hosted child is a Linux process with nothing to install: the box's resident worker
    claims its rows. This store carries a live beat, the healthy hosted shape, and the real
    helper must read THAT rather than any LaunchAgent."""
    from pipeline.worker.job_queue import SHARED_BEAT_SCOPE, RailJobStore

    _connected(monkeypatch)
    monkeypatch.setenv("OPYT_WORKER_HOME_ID", "1078")
    monkeypatch.setenv("OPYT_WORKER_DB", str(tmp_path / "rail_jobs.db"))
    RailJobStore(tmp_path / "rail_jobs.db").beat(
        SHARED_BEAT_SCOPE, "box:1:abcd", started_at=time.time())

    out = onboard()

    assert "worker_problem" not in out, "the box's worker IS running — do not warn against it"
    assert "worker" not in out, "nothing was installed here"


# ── what connecting imports is said BEFORE it happens ─────────
#
# With the question gone, the source step is the one place the user hears that connecting a
# platform imports what they saved there.

def test_the_source_step_says_connecting_imports_what_they_saved(onboard, monkeypatch):
    _at_sources(monkeypatch)

    msg = onboard()["message"]

    assert "the posts they bookmarked there are imported" in msg
    assert "the posts they saved there are imported" in msg
    assert "consent" not in msg


def test_the_still_open_list_says_what_connecting_imports(onboard, monkeypatch):
    """A platform offered at the end of setup carries the same clause as at the start."""
    _all_done(monkeypatch, candidates=40)   # X connected, Substack not

    msg = onboard()["message"]

    assert "`onboard(source='substack')`" in msg
    assert "and imports the posts you saved there" in msg


def test_the_import_clause_quotes_no_size_and_no_price(onboard, monkeypatch):
    """Both backlogs have been measured exactly once, on one account each — ~1,080 X bookmarks,
    THREE Substack saved posts. Either number here would sell a guess as a measurement."""
    _at_sources(monkeypatch)
    msg = onboard()["message"]
    assert "$" not in msg
    assert not re.search(r"\b\d{2,}\b", msg)      # no item counts, no dollar figures


# ── curation is PER PLATFORM, because platforms are connected one at a time ─────
#
# The gate was once `collectors and not _curation_ok()` — "has ANY collector anywhere
# ever succeeded" — so the first success on the first platform flipped the phase to `done` for
# good, and anything connected afterwards was never read. Measured: a user connected
# Substack, `substack_follows` succeeded, then connected X, and `x_lists`/`x_following`/
# `x_likes` never ran once. `collector_runs` held only Substack rows while `x_lists` returned 6
# candidates the moment it was called by hand — a valid session, silently orphaned.
#
# It reads as a scoring fault, not a missing feature: `screen` ranks on distinct
# (signal_type, platform) pairs, so a platform nobody read cannot corroborate anyone and nothing
# pre-ticks. The user sees "each of these showed up once. Nothing is pre-selected."

def _record_ok(collector):
    from pipeline.curation import clock
    from pipeline.kb import schema
    conn = schema.connect()
    try:
        clock.record_run(conn, collector, status="ok")
    finally:
        conn.close()


def test_a_platform_connected_after_another_succeeded_is_still_read(onboard, monkeypatch):
    """Substack read, X connected later: the phase must come BACK to curation for X alone."""

    monkeypatch.setattr(phase, "_store_sources", lambda: phase._NO_STORE_SOURCES)
    monkeypatch.setattr(connections, "_connected_substack", lambda: True)
    monkeypatch.setattr(connections, "_connected_x", lambda: False)

    _record_ok("substack_follows")
    assert phase.derive()["phase"] == "done"          # Substack alone is finished

    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    state = phase.derive()

    assert state["phase"] == "curation"                        # …and X reopens it
    assert state["curation"]["pending"] == ["x"]               # for X only

    _record_ok("x_lists")
    assert phase.derive()["phase"] == "done"           # one X collector settles X


def test_onboard_runs_no_collector_in_this_process(onboard, monkeypatch, tmp_path):
    """No threads in Claude's server. A walk on a thread dies with Claude, and
    this one ran beside a queued copy of itself. The worker runs it — so no walk may be reached
    from a tool call, and the job row is what remains."""
    from pipeline.curation import walks

    _connected(monkeypatch)
    monkeypatch.setattr(walks.WalkSource, "fetch",
                        lambda *a, **kw: pytest.fail("a walk ran inside the tool call"))

    onboard()

    assert "scrapers" in _jobs(tmp_path)


# ── each import is woken until its walk begins, for a platform found connected ──
#
# The import is the walk of the saved list in the `scrapers` process — it survives Claude
# closing, which a thread in this process would not.

def test_an_import_is_queued_even_when_the_walk_already_read_the_platform(
        onboard, monkeypatch, tmp_path):
    """Keyed on the walk, not on `pending`. The people collectors can read a platform connected
    after another before `onboard` is next called; `pending` is then empty, and an import keyed
    on it would never be woken — the shape of the defect where Substack connected
    after X got no import at all."""
    _connected(monkeypatch)
    monkeypatch.setattr(collectors, "curation_pending", lambda live: [])

    out = onboard()

    assert out["backlog_queued"] == ["bookmarks_import"]
    assert "scrapers" in _jobs(tmp_path)


def test_a_plain_onboard_call_does_not_re_queue_the_saved_list(onboard, monkeypatch, tmp_path):
    """Once its walk has begun, the walk's own refresh keeps it going; `onboard` reports no import
    for it again."""
    from datetime import datetime, timezone

    from pipeline.curation import bookmark_listing, listings
    from pipeline.kb import schema

    _connected(monkeypatch)
    assert onboard()["backlog_queued"] == ["bookmarks_import"]
    conn = schema.connect()
    try:
        listings.init_schema(conn)
        listings.start(conn, bookmark_listing.COLLECTOR, "me", listings.Branch("Bookmarks"),
                       datetime.now(timezone.utc))
    finally:
        conn.close()

    out = onboard()

    assert "backlog_queued" not in out and "saved_content" not in out


def _at_curation(monkeypatch):
    _connected(monkeypatch)


def test_the_curation_phase_says_the_walk_is_running_not_that_setup_is_unfinished(
        onboard, monkeypatch):
    """⚠️ THE ORDINARY STATE NOW, not a race. The walk took ~8s on a thread; in the worker it
    waits its turn, and with no question to answer it has had no time at all. "Setup is not
    finished, call `onboard` again" would send the host into a polling loop over work nobody can
    speed up."""
    _at_curation(monkeypatch)

    out = onboard()

    assert out["phase"] == "curation"
    assert "not ready to choose from yet" in out["message"]
    assert "do not call `screen` yet" in out["message"]
    assert "call `onboard` again in a few minutes" in out["message"]
    assert "not finished" not in out["message"]


# ── phase 4: the handoff ────────────────────────────────────────────────────

def _all_done(monkeypatch, candidates=0, **store):
    _at_curation(monkeypatch)
    monkeypatch.setattr(collectors, "curation_pending", lambda live: [])
    monkeypatch.setattr(handoff, "_candidate_count", lambda: candidates)


def test_done_names_oracle_as_the_next_call(onboard, monkeypatch):
    _all_done(monkeypatch)
    out = onboard()
    assert out["phase"] == "done" and out["next_tool"] == "oracle"


def test_handoff_says_the_pull_is_in_the_background(onboard, monkeypatch):
    """Confirming returns at once and the worker pulls in the background, so the
    done copy points to `oracle(action='progress')` and suggests a small first batch only so each
    person's history fills in sooner. Only in the branch that HAS a screening list: with nothing
    to screen there is no first batch to suggest."""
    _all_done(monkeypatch, candidates=40)
    msg = onboard()["message"]
    assert "oracle(action='progress')" in msg
    assert "a few people" in msg
    assert "foreground" not in msg.lower()


def test_handoff_offers_claude_ai_until_this_install_reaches_the_service(onboard, monkeypatch):
    """Setup tells the user how to read their library from claude.ai and
    ChatGPT. Once this install holds a service token (it shared, or connected Google), not again."""
    _all_done(monkeypatch, candidates=40)
    monkeypatch.setattr(account, "load", lambda: account.Account("https://svc", None))
    assert "connect Opyt to Google" in onboard()["message"]
    monkeypatch.setattr(account, "load", lambda: account.Account("https://svc", "tok"))
    assert "connect Opyt to Google" not in onboard()["message"]


def test_done_reports_what_actually_landed(onboard, monkeypatch):
    _all_done(monkeypatch, candidates=42)
    assert onboard()["candidates"] == 42


def test_a_user_with_no_collector_is_not_told_a_collector_ran(onboard, monkeypatch, tmp_path):
    """⚠️ "The free collectors surfaced 0 people to screen" describes a pass that never happened.
    A blog or research-topic user has no platform session, so nothing collected — and the honest
    version of that sentence is the list of ways they can add something."""

    _all_done(monkeypatch)
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(phase, "_store_sources",
                        lambda: {**phase._NO_STORE_SOURCES,
                                 "oracles": True})

    out = onboard()

    assert out["phase"] == "done"
    assert "collector" not in out["message"] or "no source" in out["message"]
    assert "add_handles" in out["message"] and "hopper" in out["message"]
    # Watches are not an onboarding move, so the ways-in list offers the content routes only.
    assert "watch(action=" not in out["message"]


def _done_blind(monkeypatch):
    """Keys green, no session, nothing named, nothing watched — the user skipped through."""
    _at_sources(monkeypatch)
    monkeypatch.setattr(phase, "_skipped", lambda: True)
    monkeypatch.setattr(handoff, "_candidate_count", lambda: 0)


def test_a_skipped_through_store_is_asked_a_question_not_handed_a_form(onboard, monkeypatch):
    """A store with NO signal — nobody named, nothing watched, no session — would get
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
    # Onboarding creates no standing watches. A watch is a commitment whose
    # wording the user should shape, and that conversation belongs after they have played with
    # the store — so the handoff fills the store (survey, saves, authors) and defers the watch.
    assert "watch(action=" not in msg


def test_a_user_who_named_something_is_not_asked_what_they_are_into(onboard, monkeypatch):
    """The direct question is for the store with no signal. A user who named a blog has
    already answered it, and asking again ignores what they said."""
    _done_blind(monkeypatch)
    monkeypatch.setattr(phase, "_store_sources",
                        lambda: {**phase._NO_STORE_SOURCES,
                                 "oracles": True})

    msg = onboard()["message"]

    assert "What topics are you into" not in msg
    assert "hopper" in msg and "add_handles" in msg


def test_a_watch_only_store_is_still_asked_what_the_user_is_into(onboard, monkeypatch):
    """A store carrying six watches and zero atoms got the
    generic open-roots list, because watch rows counted as "named something". The flow's
    question is "is there anything to PLAY with" — and watches stage an inbox, not a library.
    The opener stays truthful for this state: it credits the watches instead of claiming
    "nothing watched"."""
    _done_blind(monkeypatch)
    monkeypatch.setattr(phase, "_store_sources",
                        lambda: {**phase._NO_STORE_SOURCES,
                                 "topics": True})

    msg = onboard()["message"]

    assert "What topics are you into" in msg
    assert "nothing watched" not in msg
    assert "watching for new papers and repos" in msg


def test_an_unreadable_atom_count_never_claims_an_empty_store(onboard, monkeypatch):
    """"Your store is empty" is a claim about the user's data; a broken read routes to the
    generic branch, which claims nothing."""
    _done_blind(monkeypatch)
    monkeypatch.setattr(handoff, "_atom_count", lambda: None)

    assert "What topics are you into" not in onboard()["message"]


def test_a_connected_X_user_is_still_told_the_other_roots_exist(onboard, monkeypatch):
    """`roots.menu` promises the four roots are not exclusive —
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
    from mcp_server import build
    m = _MCP()
    from mcp_server.onboard_tools import register_onboard_tools
    register_onboard_tools(m)
    assert "onboard" in m.tools

    assert ("mcp_server.onboard_tools", "register_onboard_tools") in build.TOOL_MODULES, \
        "the server no longer registers `onboard`"
    names = {t.name for t in asyncio.run(build.server().list_tools())}
    assert "onboard" in names, "the built server does not list `onboard`"


def test_the_promises_other_modules_make_about_onboard_are_keepable():
    """Strings in the tree tell users to run `onboard`. They must name a tool that exists, and
    none may send them to the `consent` argument, which no longer exists."""
    from pipeline.curation import imports

    assert "consent=" not in open(imports.__file__).read()

    # The store_empty notice names `onboard` as the next tool a new user should call.
    from pipeline.retrieval import notices
    assert "onboard" in open(notices.__file__).read()


def test_source_x_reconnects_a_session_that_is_already_connected(onboard, monkeypatch):
    """An X session can expire or belong to the wrong account after setup is finished.

    `graphql_get` tells the user to reconnect X when x.com rejects the hosted session, so the
    tool has to be able to carry that instruction out without the profile being deleted by hand.
    """
    from pipeline.platforms.browser import cookies as bc, login

    started = []
    _at_sources(monkeypatch, connected=True)
    monkeypatch.setattr(login, "start",
                        lambda url: started.append(url) or bc.backend_for("chrome"))

    assert onboard()["phase"] != "sources"

    out = onboard(source="x")

    assert started == ["https://x.com/login"]
    assert out["status"] == "awaiting_login"


# ── no call waits for a pull: onboard's half ───────────────────────────────────
# An `onboard` call was once cut off at 61 seconds while waiting on a join over ~1,000
# bookmarks. The promise is "setup is done, your saved posts are importing" — weaker, and true.

def test_connecting_carries_no_report_on_the_import(onboard, monkeypatch, tmp_path):
    """No onboarding report. Nothing is in when the import
    is queued, so there is nothing to say; the bookmarks get one plain notice, with the count, when
    they land."""
    _connected(monkeypatch)

    out = onboard()

    assert "scrapers" in _jobs(tmp_path)
    assert "saved_content" not in out


def test_the_host_is_told_not_to_screen_in_the_same_turn(onboard, monkeypatch):
    """The screen is SCORED on these signals — a candidate list built before the walk lands is
    not a thin list, it is a WRONG one, with no symptom."""
    assert "do not call `screen` in this same turn" in \
        kickoff._queue_curation_walk()["host_note"]


def test_a_walk_that_cannot_be_queued_says_so_rather_than_claiming_it_is(onboard, monkeypatch):
    """Fail-safe, and in the honest direction: a candidate list nothing is building must not be
    described as being built."""
    monkeypatch.setattr(job_queue, "request_now", lambda rail, **kw: False)

    out = kickoff._queue_curation_walk()

    assert out["status"] == "not_queued" and "host_note" not in out


# ── local installs are macOS-only ─────────────────────────

def test_a_local_install_off_macos_is_sent_to_a_mac(onboard, monkeypatch, tmp_path):
    """With no threads, a local install with no worker would finish setup with an empty store and
    nothing saying why. So it is told plainly, on the first call, before anything is set up."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(phase, "derive",
                        lambda **kw: pytest.fail("nothing may be set up off macOS"))

    out = onboard()

    assert out["status"] == "unsupported_platform"
    assert liveness.USE_A_MAC in out["message"]


def test_a_hosted_child_on_linux_is_not_refused(onboard, monkeypatch, tmp_path):
    """Hosted runs on Linux and is not this case — its rows go to the box's resident worker."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("OPYT_WORKER_HOME_ID", "1078")
    monkeypatch.setenv("OPYT_WORKER_DB", str(tmp_path / "rail_jobs.db"))
    _at_sources(monkeypatch)

    assert onboard()["status"] != "unsupported_platform"


def test_a_hand_run_worker_off_macos_is_let_through(onboard, monkeypatch, tmp_path):
    """`opyt-worker` by hand is the documented route off macOS. A beating worker is exactly what
    the refusal protects against not having."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(job_queue, "worker_is_alive", lambda **kw: True)
    _at_sources(monkeypatch)

    assert onboard()["status"] != "unsupported_platform"


# ── a platform connected AFTER another ──────────────────────────────────────────
#
# The defect these exist for, found in a real session: X was live, Substack was
# connected later, the saved posts imported in-process — and `substack_saved_catchup` never got
# a job row, so the recurring sync never ran once. This is the ORDINARY path, not an edge case:
# `roots.menu` asks for one root and lists the rest as "still open".
def test_connecting_substack_after_x_queues_its_import(onboard, monkeypatch, tmp_path):
    _connected(monkeypatch)
    assert onboard()["backlog_queued"] == ["bookmarks_import"]

    _at_sources(monkeypatch, connected=True, substack=True)   # Substack arrives afterwards
    out = onboard()

    assert "substack_saved_import" in out["backlog_queued"], (
        "the saved posts must be woken now — an import nobody wakes is the defect")
    job = _jobs(tmp_path).get("scrapers")
    assert job is not None and job.due_at <= time.time()


def test_substack_first_then_x_queues_every_rail(onboard, monkeypatch, tmp_path):
    """THE MIRROR ORDER. Every other test here walks X-first, because that is the order the
    sources prompt lists and the order every measured session happened to take — which is exactly
    why the reverse is worth asserting."""
    _at_sources(monkeypatch, connected=False, substack=True)
    assert onboard()["backlog_queued"] == ["substack_saved_import"], \
        "X is not connected — it has no backlog yet"

    _at_sources(monkeypatch, connected=True, substack=True)   # X arrives afterwards
    out = onboard()

    assert "bookmarks_import" in out["backlog_queued"]
    jobs = _jobs(tmp_path)
    assert set(jobs) == {"scrapers"} and jobs["scrapers"].due_at <= time.time() + 1
