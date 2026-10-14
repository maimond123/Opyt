"""The `line` job (`mcp_server/line.py`): this install holds a line to the sharing service, and
runs the tool calls the web and agents ask for. 
Four things are held here:
  • the allow-list. Until this job the service could not make an install do anything, so what it
    may ask for is this install's own guard, and anything else is refused without running;
  • a call runs the server's own registered tool, on the real store, without the rider flow;
  • a call that changed the library owes the shared copy a push, and one that did not owes none;
  • how long the job stays: not at all with no token, one ask with no web writer, and otherwise
    until the worker stops it, when it asks to come straight back.

The service is a stand-in at `client.line_next` and `client.line_answer`, except in the last
section, where the real job runs under a real worker against a server on this machine.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mcp_server import build, cli, line
from opyt_core import config
from opyt_core.paths import opyt_path
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.sharing import account, client, push_gate
from pipeline.topics import create
from pipeline.worker import job_queue, registry
from pipeline.worker.home import LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV
from pipeline.worker.job_queue import RailJobStore
from pipeline.worker.registry import HOURLY, LINE, SCRAPERS, RailSpec
from pipeline.worker.supervisor import RailWorker
from tests.kb.test_export import _add

ATOM = "x:1"


# ── the allow-list ───────────────────────────────────────────────────────────────
# §14.5, as written. A change to what the service may make an install do is a change here.
PLAN = {
    "hopper": {"reference", "kind_hint", "confirm"},
    "forget": {"atom_id", "atom_ids", "oracle", "confirm"},
    "watch": {"action", "name", "watches", "words", "people", "topic_id", "drop_watches"},
    "add_oracle": {"reference", "confirm", "x_lookback", "web_lookback", "scholar_lookback",
                   "scholar_topics"},
}
# What each of those tools takes from this computer's host and not from the service.
WITHHELD = {
    "hopper": set(),
    "forget": {"source", "delete_atoms"},
    "watch": {"keep", "passed"},
    "add_oracle": {"force"},
}


def test_the_allow_list_is_the_plans_table():
    assert line.ALLOWED == PLAN
    assert line.WATCH_ACTIONS == ("preview", "create", "edit", "drop", "list")


def test_every_parameter_of_an_allowed_tool_is_either_allowed_or_withheld_by_name():
    """The list names real parameters, and a parameter a tool gains later is nobody's by
    default: this fails until it is allowed or withheld on purpose."""
    server = build.server()
    for tool in PLAN:
        takes = set(asyncio.run(server.get_tool(tool)).parameters["properties"])
        assert takes == PLAN[tool] | WITHHELD[tool], tool


@pytest.mark.parametrize("tool, args", [
    ("hopper", {"reference": "https://arxiv.org/abs/2401.00001"}),
    ("hopper", {"reference": "https://example.com/p/x", "kind_hint": "substack", "confirm": True}),
    ("forget", {"atom_id": ATOM, "confirm": True}),
    ("forget", {"atom_ids": [ATOM, "x:2"]}),
    ("forget", {"oracle": "@karpathy", "confirm": True}),
    *[("watch", {"action": action}) for action in ("preview", "create", "edit", "drop", "list")],
    ("watch", {"action": "edit", "topic_id": "t1", "name": "n", "watches": [], "words": [],
               "people": [], "drop_watches": ["x402"]}),
    ("add_oracle", {"reference": "@karpathy", "confirm": True, "x_lookback": "1yr",
                    "web_lookback": "all", "scholar_lookback": "5yr", "scholar_topics": []}),
])
def test_what_the_service_may_ask_for(tool, args):
    assert line.refusal(tool, args) is None


@pytest.mark.parametrize("tool, args", [
    ("oracle", {"action": "confirm", "add_handles": ["@x"]}),     # a tool that is not listed,
    ("onboard", {}),
    ("share", {"confirm": True}),
    ("unshare", {"confirm": True}),
    ("search", {"query": "agents"}),                              # a read included
    ("forget", {"atom_id": ATOM, "delete_atoms": True}),          # an argument that is not,
    ("forget", {"source": "https://blog.example", "confirm": True}),
    ("forget", {"oracle": "@a", "source": "https://blog.example"}),
    ("add_oracle", {"reference": "@karpathy", "force": True}),
    ("hopper", {"reference": "https://example.com", "entry_mode": "oracle-footprint"}),
    ("watch", {"action": "checked"}),                             # an action that is not,
    ("watch", {"action": "checked", "keep": ["c1"]}),
    ("watch", {"name": "no action"}),
    ("watch", {"action": ["list"]}),
    (None, {}),                                                   # and anything misshapen
    (["hopper"], {"reference": "u"}),
    ("hopper", None),
    ("hopper", ["reference"]),
])
def test_anything_else_is_refused_and_not_run(tool, args, monkeypatch):
    def built():
        raise AssertionError("a refused call reached the server")
    monkeypatch.setattr(line, "_server", built)

    ok, answer = line.outcome(tool, args)

    assert ok is False and answer["status"] == "refused" and answer["message"]


# ── the other decisions ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("tool, args, answer, changed", [
    ("hopper", {"confirm": True}, {"status": "saved", "atom_id": "paper:1"}, True),
    ("hopper", {"confirm": True}, {"status": "already_present"}, False),
    ("hopper", {"confirm": True}, {"status": "blocked"}, False),
    ("hopper", {"confirm": True}, {"status": "failed"}, False),
    ("hopper", {"confirm": True}, {"status": "unroutable"}, False),
    ("hopper", {}, {"status": "preview", "already_present": False}, False),
    ("forget", {"atom_id": ATOM, "confirm": True}, {"status": "forgotten"}, True),
    ("forget", {"atom_id": ATOM}, {"status": "preview"}, False),
    ("forget", {"atom_id": ATOM, "confirm": True}, {"status": "not_found"}, False),
    ("forget", {"oracle": "Ann", "confirm": True}, {"status": "ambiguous"}, False),
    ("forget", {"atom_id": "", "confirm": True}, {"status": "error"}, False),
    ("watch", {"action": "create"}, {"status": "ok", "topic_id": "t1"}, True),
    ("watch", {"action": "edit"}, {"status": "ok", "topic_id": "t1"}, True),
    ("watch", {"action": "drop"}, {"status": "ok", "dropped": "n"}, True),
    ("watch", {"action": "create"}, {"status": "refused", "reason": "no watches"}, False),
    ("watch", {"action": "list"}, {"status": "ok", "topics": []}, False),
    ("watch", {"action": "preview"}, {"status": "preview"}, False),
    ("add_oracle", {"confirm": True}, {"added": {"canonical_id": "x:user:1"}}, True),
    ("add_oracle", {"confirm": True}, {"error": "could not resolve", "unresolved": ["@x"]}, False),
    ("add_oracle", {}, {"status": "preview", "name": "Carol"}, False),
])
def test_which_answers_changed_the_library(tool, args, answer, changed):
    assert line.changed_library(tool, args, answer) is changed


@pytest.mark.parametrize("got, step", [
    ({"call": {"id": "c1", "tool": "hopper", "args": {}}}, "run"),
    ({"call": None}, "ask"),
    ({"call": None, "retry_after": 3600}, "leave"),
])
def test_what_an_answer_to_an_ask_calls_for(got, step):
    assert line.next_step(got) == step


@pytest.mark.parametrize("status, leaves", [
    (None, False),                 # no answer at all, or a proxy's page: the service is away
    (500, False), (502, False), (503, False),
    (401, True),                   # the library is gone
    (404, True),                   # a service that does not know the line
    (403, True),
])
def test_which_failures_asking_again_will_not_fix(status, leaves):
    assert line.gives_up(status) is leaves


def test_the_wait_before_asking_again_is_short_then_grows_to_a_cap():
    waits = [line.pause_s(n) for n in range(1, 7)]
    assert waits == sorted(waits) and waits[0] <= 5 and waits[-1] == waits[-2] == 60


# ── a call runs the real tool, without the riders ────────────────────────────────
@pytest.fixture()
def home(kb_home, monkeypatch):
    """A local install's home: its own store and its own control database."""
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    return kb_home


def _topics() -> list[str]:
    with closing(schema.connect()) as conn:
        return [t["name"] for t in create.listing(conn)["topics"]]


def test_a_call_runs_the_tools_own_body_on_this_store(home):
    ok, answer = line.outcome("watch", {"action": "create", "name": "Agent payments",
                                        "watches": [{"name": "x402", "sources": ["github"]}]})

    assert ok is True and answer["status"] == "ok"
    assert [w["name"] for w in answer["watches"]] == ["x402"] and _topics() == ["Agent payments"]
    assert json.loads(json.dumps(answer)) == answer          # the answer goes on the wire


@pytest.mark.parametrize("tool, args, says", [
    ("hopper", {"reference": "not a link", "confirm": True}, ("status", "unroutable")),
    ("forget", {"atom_id": "x:never-stored", "confirm": True}, ("status", "not_found")),
    ("watch", {"action": "list"}, ("status", "ok")),
    ("add_oracle", {"reference": " ", "confirm": True}, ("error", None)),
])
def test_each_allowed_tool_is_reached_and_answers_for_itself(home, tool, args, says):
    """A tool's own "no" is still its answer, so the call ran: `ok` is about the call."""
    ok, answer = line.outcome(tool, args)

    key, value = says
    assert ok is True and key in answer and value in (None, answer[key])


def test_a_web_call_spends_nothing_meant_for_this_computers_host(home, capsys):
    """Every answer a host receives passes the rider flow, which stamps what the call used and
    hands over notices and offers once (`rider_hook.py`). A web call would spend them on a host
    that is not this computer's: the line runs the same tool and leaves them alone."""
    history = opyt_path("offers.json")

    ok, answer = line.outcome("watch", {"action": "list"})
    assert ok is True and answer == {"status": "ok", "topics": []}
    assert not history.exists()

    assert cli.main(["call", "watch", json.dumps({"action": "list"})]) == 0
    assert json.loads(capsys.readouterr().out) == answer     # the same tool, the same answer
    assert "topic" in json.loads(history.read_text())["used"]


def test_a_call_the_tool_would_refuse_from_a_host_is_refused_the_same_way(home):
    """Validation is the registered tool's own: one code path."""
    ok, answer = line.outcome("watch", {"action": "list", "topic_id": 7})

    assert ok is False and answer["status"] == "error"
    assert answer["message"].startswith("ValidationError: ") and "topic_id" in answer["message"]


def test_a_tool_that_raises_is_answered_not_ok_and_not_run_again(home, monkeypatch):
    ran = []

    def boom(conn):
        ran.append(1)
        raise RuntimeError("the store is on fire")
    monkeypatch.setattr(create, "listing", boom)

    assert line.outcome("watch", {"action": "list"}) == (
        False, {"status": "error", "message": "RuntimeError: the store is on fire"})
    assert ran == [1]


def test_a_tool_that_answers_with_no_status_is_still_its_own_answer(home, monkeypatch):
    """`add_oracle` says `added`, not a status. The person is on the roster when it returns."""
    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    with closing(schema.connect()) as conn:
        schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})

    ok, answer = line.outcome("add_oracle", {"reference": "x:user:1", "confirm": True})

    assert ok is True and answer["added"]["name"] == "Carol"
    assert line.changed_library("add_oracle", {"confirm": True}, answer)
    assert json.loads(json.dumps(answer)) == answer


# ── holding the line ─────────────────────────────────────────────────────────────
class Service:
    """The line's two routes. `asks` is what `next` answers, in order: an answer, a failure to
    raise, or something to do first (then the line has nothing handed over)."""

    def __init__(self, *asks):
        self.asks = list(asks)
        self.asked = 0
        self.answers: list[tuple[str, bool, dict]] = []
        self.failing_answers: list[client.Failed] = []

    def line_next(self, acct, wait):
        self.asked += 1
        assert acct.token and 0 < wait <= 50
        got = self.asks.pop(0)
        if isinstance(got, Exception):
            raise got
        if callable(got):
            got()
            return {"call": None}
        return got

    def line_answer(self, acct, call_id, ok, answer):
        if self.failing_answers:
            raise self.failing_answers.pop(0)
        self.answers.append((call_id, ok, answer))
        return {"recorded": True}


NO_WRITER = {"call": None, "retry_after": 3600}
AWAY = client.Failed("unreachable", "could not reach https://svc.test/v1/line/next")


def _call(tool: str, args: dict, call_id: str = "c1") -> dict:
    return {"call": {"id": call_id, "tool": tool, "args": args}}


@pytest.fixture()
def service(home, monkeypatch):
    """An install that shares, and the service it asks."""
    account.save_token(account.load(), "tok-owner")

    def make(*asks):
        svc = Service(*asks)
        monkeypatch.setattr(client, "line_next", svc.line_next)
        monkeypatch.setattr(client, "line_answer", svc.line_answer)
        return svc
    return make


@pytest.fixture()
def woken(monkeypatch):
    """Every job a call site asked for, still queued for real."""
    calls, real = [], job_queue.request_now

    def recording(rail, **kw):
        calls.append(rail)
        return real(rail, **kw)
    monkeypatch.setattr(job_queue, "request_now", recording)
    return calls


def _never_sleeps(seconds):
    raise AssertionError(f"the line waited {seconds}s with nothing to wait for")


def test_an_install_that_never_shared_asks_nothing(home, monkeypatch):
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda service: None)
    monkeypatch.setattr(client, "line_next", lambda *a: pytest.fail("asked with no token"))

    assert line.run() == {"status": "not_shared"}


def test_a_library_with_no_web_writer_asks_once_and_leaves_it_to_the_cadence(service):
    svc = service(NO_WRITER)

    got = line.hold(_never_sleeps)

    assert got["status"] == "nothing_owed" and svc.asked == 1
    assert registry.RAILS[LINE].cadence == HOURLY


def test_nothing_handed_over_asks_again_at_once(service):
    svc = service({"call": None}, {"call": None}, _call("watch", {"action": "list"}), NO_WRITER)

    line.hold(_never_sleeps)

    assert svc.asked == 4
    assert svc.answers == [("c1", True, {"status": "ok", "topics": []})]


def test_an_unreachable_service_is_waited_for_longer_each_time_and_asked_again(service):
    svc = service(AWAY, AWAY, AWAY, AWAY, {"call": None}, AWAY, NO_WRITER)
    slept = []

    assert line.hold(slept.append)["status"] == "nothing_owed"

    assert slept == [5, 15, 60, 60, 5]      # growing, capped, and short again once it answered
    assert svc.asked == 7


def test_a_service_that_is_restarting_is_waited_for_like_one_that_is_unreachable(service):
    svc = service(client.Failed("refused", "the service answered 502: bad gateway", 502),
                  client.Failed("refused", "a proxy's page, not the service's JSON"), NO_WRITER)
    slept = []

    line.hold(slept.append)

    assert slept == [5, 15] and svc.asked == 3


def test_a_revoked_token_ends_the_job(service):
    svc = service(client.Failed("revoked", "Opyt's server no longer holds this library", 401))

    got = line.hold(_never_sleeps)

    assert got["status"] == "revoked" and svc.asked == 1


def test_a_service_that_does_not_know_the_line_is_left_to_the_cadence(service):
    """An install can be newer than the service it shares with. Asking a 404 every minute for
    good would be the cost; the hourly cadence finds the route when it arrives."""
    svc = service(client.Failed("refused", "the service answered 404: Not Found", 404))

    got = line.hold(_never_sleeps)

    assert got["status"] == "refused" and svc.asked == 1


def test_a_token_that_goes_while_the_line_is_held_ends_the_job(service):
    """The account is read before every ask, so the line never asks without a token."""
    svc = service(account.drop_token)

    assert line.hold(_never_sleeps) == {"status": "not_shared"}
    assert svc.asked == 1


def test_a_held_line_the_worker_stops_asks_to_come_straight_back(service, woken, monkeypatch):
    """`child` raises the worker's SIGTERM as a `SystemExit` in the blocked ask. Recorded at its
    hourly cadence, a restart every six hours would leave the web an hour without a line."""
    def stopped(acct, wait):
        raise SystemExit(143)
    monkeypatch.setattr(client, "line_next", stopped)

    assert line.run() == {"status": "stopped"}
    assert woken == [LINE]
    assert RailJobStore().get(LOCAL_HOME_ID, LINE).due_at <= time.time()


# ── what a call leaves behind ────────────────────────────────────────────────────
@pytest.fixture()
def atom(home):
    with closing(schema.connect()) as conn:
        _add(conn, ATOM, "x", "opinion", "x:user:1", [],
             "autonomous agents", entry_mode="user-saved")
    return ATOM


def test_a_call_that_changed_the_library_owes_a_push_and_wakes_the_scrapers(service, woken, atom):
    """The shared copy is what the web reads, so it has to follow: the push is owed, and the
    process whose publishing pass sends it is asked for now."""
    svc = service(_call("forget", {"atom_id": atom, "confirm": True}), NO_WRITER)

    line.hold(_never_sleeps)

    (call_id, ok, answer), = svc.answers
    assert (call_id, ok, answer["status"]) == ("c1", True, "forgotten")
    assert push_gate.owed_at() is not None and woken == [SCRAPERS]
    assert RailJobStore().get(LOCAL_HOME_ID, SCRAPERS).due_at <= time.time()


@pytest.mark.parametrize("args", [
    {"atom_id": ATOM},                                   # a preview
    {"atom_id": "x:never-stored", "confirm": True},      # a repeat: `not_found`
])
def test_a_call_that_changed_nothing_owes_no_push(service, woken, atom, args):
    svc = service(_call("forget", args), NO_WRITER)

    line.hold(_never_sleeps)

    (_, ok, answer), = svc.answers
    assert ok is True and answer["status"] in ("preview", "not_found")
    assert push_gate.owed_at() is None and woken == []


def test_a_refused_call_is_answered_not_ok_and_leaves_nothing(service, woken, atom):
    svc = service(_call("forget", {"atom_id": atom, "confirm": True, "delete_atoms": True}),
                  NO_WRITER)

    line.hold(_never_sleeps)

    (call_id, ok, answer), = svc.answers
    assert (call_id, ok, answer["status"]) == ("c1", False, "refused")
    assert "delete_atoms" in answer["message"]
    with closing(schema.connect()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (atom,)).fetchone()[0]
    assert push_gate.owed_at() is None and woken == []


def test_a_call_that_raised_is_answered_not_ok_and_owes_nothing(service, woken, monkeypatch):
    monkeypatch.setattr(create, "listing",
                        lambda conn: (_ for _ in ()).throw(RuntimeError("the store is on fire")))
    svc = service(_call("watch", {"action": "list"}), NO_WRITER)

    line.hold(_never_sleeps)

    assert svc.answers == [
        ("c1", False, {"status": "error", "message": "RuntimeError: the store is on fire"})]
    assert push_gate.owed_at() is None and woken == []


def test_an_answer_the_service_never_got_still_owes_the_push_and_the_line_goes_on(
        service, woken, atom):
    """The library changed whether or not the report arrived. The service offers the call
    again later (§14.3), and the repeat answers `not_found`."""
    svc = service(_call("forget", {"atom_id": atom, "confirm": True}),
                  _call("forget", {"atom_id": atom, "confirm": True}), NO_WRITER)
    svc.failing_answers = [AWAY]
    slept = []

    line.hold(slept.append)

    assert slept == [5] and svc.asked == 3
    assert [(ok, answer["status"]) for _, ok, answer in svc.answers] == [(True, "not_found")]
    assert push_gate.owed_at() is not None and woken == [SCRAPERS]


# ── under the worker ─────────────────────────────────────────────────────────────
class _Line(BaseHTTPRequestHandler):
    """The service's two routes: the first ask hands over one call, and every later ask is held
    open until the test lets go."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        srv = self.server
        srv.asked.append((self.path, self.headers.get("Authorization")))
        if len(srv.asked) == 1:
            return self._answer(_call("watch", {"action": "list"}))
        srv.holding.set()
        srv.let_go.wait(60)
        self._answer({"call": None})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.answered.append((self.path, body))
        self._answer({"recorded": True})

    def _answer(self, body: dict) -> None:
        data = json.dumps(body).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass                       # the line hung up: it was stopped mid-ask


@pytest.fixture()
def on_this_machine():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Line)
    srv.asked, srv.answered = [], []
    srv.holding, srv.let_go = threading.Event(), threading.Event()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.let_go.set()
    srv.shutdown()
    srv.server_close()


def _shares_with(home, url: str) -> None:
    """What a sharing install's home holds: where the service is, and its owner token."""
    template = (config.REPO_ROOT / "config" / "settings.example.yaml").read_text()
    (home / "settings.yaml").write_text(f"{template}\nservice_url: {url}\n")
    (home / ".env").write_text("OPYT_SERVICE_TOKEN=tok-owner\n")


def _stop_and_reap(worker: RailWorker) -> float:
    """The restart's SIGTERM, and how long the children took to exit."""
    began = time.monotonic()
    worker.ask_children_to_stop()
    for child in worker.active:
        child.process.wait(timeout=30)
    took = time.monotonic() - began
    worker.reap_finished()
    return took


@pytest.mark.loopback
def test_the_registered_job_holds_the_line_and_a_restart_brings_it_straight_back(
        home, on_this_machine):
    """The whole path, as the worker runs it: the registry's own entry through `child`, the real
    requests, a call run and answered, and then the restart. A line that is being held is a read
    blocked for most of a minute, and the stop has to cut it: the job exits at once, healthy,
    with its own row due now, so the next worker lifetime launches it again."""
    srv = on_this_machine
    _shares_with(home, f"http://127.0.0.1:{srv.server_port}")
    store = RailJobStore(home / "rail_jobs.db")
    store.activate(LOCAL_HOME_ID, LINE, due_at=0)
    worker = RailWorker(store)
    worker.launch_available()
    try:
        assert srv.holding.wait(60), (home / "line.log").read_text()
        took = _stop_and_reap(worker)
    finally:
        worker.stop_children()

    assert srv.asked[0] == ("/v1/line/next?wait=50", "Bearer tok-owner")
    assert srv.answered == [("/v1/line/c1/answer",
                             {"ok": True, "answer": {"status": "ok", "topics": []}})]
    assert took < 10, f"the stop waited {took:.0f}s for a held line"
    row = store.get(LOCAL_HOME_ID, LINE)
    assert row.exit_code == 0, (home / "line.log").read_text()
    assert row.due_at <= row.finished_at
    assert store.claim_next(home_id=LOCAL_HOME_ID, rails=(LINE,)) is not None


def test_a_stop_in_the_middle_of_a_call_lets_the_tool_finish_and_answers_nothing(
        home, monkeypatch):
    """A tool body is never cut part-way. The stop ends the job, the body runs to its end on its
    own thread before the process exits, and the answer is not posted: the service offers the
    call again (§14.3), and a repeat is harmless."""
    (home / ".env").write_text("OPYT_SERVICE_TOKEN=tok-owner\n")
    monkeypatch.setattr(registry, "RAILS", {LINE: RailSpec(
        "tests.kb._rail_children:holds_the_line_through_a_slow_call", "line.log", HOURLY,
        beside=True)})
    store = RailJobStore(home / "rail_jobs.db")
    store.activate(LOCAL_HOME_ID, LINE, due_at=0)
    worker = RailWorker(store)
    worker.launch_available()
    try:
        deadline = time.monotonic() + 60
        while not (home / "call-began").exists():
            assert time.monotonic() < deadline, (home / "line.log").read_text()
            time.sleep(0.05)
        _stop_and_reap(worker)
    finally:
        worker.stop_children()

    assert (home / "call-ended").exists() and not (home / "call-answered").exists()
    row = store.get(LOCAL_HOME_ID, LINE)
    assert row.exit_code == 0 and row.due_at <= row.finished_at
