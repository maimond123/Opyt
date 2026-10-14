"""
tests/onboarding/test_optional_keys.py — the optional keys (GitHub, OpenAlex):
the step, the key page on the loopback, GitHub's device sign-in, and that a stored key reaches
the calls that use it.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from types import SimpleNamespace

import pytest

from mcp_server import onboard_tools
from opyt_core import github_device, key_page, keys
from pipeline import credentials
from pipeline.kb import schema
from pipeline.onboarding import optional_keys
from pipeline.platforms.github import api as github_api
from pipeline.platforms.papers import openalex
from pipeline.platforms.papers import registry as paper_registry


@pytest.fixture()
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    for env in ("GITHUB_TOKEN", "OPENALEX_API_KEY", "S2_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    return tmp_path


# ── the question: asked once, on the first confirmation whose picks link code or papers ──
#
# Onboarding is sources → curation → confirm Oracles, and the keys
# question rides the answer to the first confirmation, worded from who was picked. What a pick
# links is read from the store: a `github` or `paper` atom they authored, or one a post of theirs
# linked (`payload.referenced_by`), as the link router classified it when it was stored.

@pytest.fixture()
def conn(home, monkeypatch):
    monkeypatch.setattr(github_device, "CLIENT_ID", "Iv1.test")
    c = schema.connect()
    yield c
    c.close()


def _oracle(conn, uid: str, *members: str) -> str:
    """An Oracle `x:user:<uid>` and the other accounts resolved into its cluster."""
    cid = f"x:user:{uid}"
    schema.upsert_entity(conn, cid, name=f"Person {uid}")
    for m in members:
        schema.upsert_entity(conn, m)
        conn.execute("UPDATE entities SET canonical_id=? WHERE entity_id=?", (cid, m))
    conn.execute("UPDATE entities SET canonical_id=? WHERE entity_id=?", (cid, cid))
    return cid


def _artifact(conn, atom_id: str, kind: str, *, who: str = "someone", linked_by: str = "") -> None:
    refs = [{"by": linked_by, "post": "x:1"}] if linked_by else []
    conn.execute("INSERT INTO atoms(atom_id, source_type, who_id, entry_mode, payload) "
                 "VALUES (?,?,?,?,?)", (atom_id, kind, who, "author_referenced",
                                        json.dumps({"referenced_by": refs})))


def test_the_first_confirmation_with_code_and_papers_asks_for_both(conn):
    a, b, c = _oracle(conn, "1"), _oracle(conn, "2"), _oracle(conn, "3")
    _artifact(conn, "github:o/r1", "github", linked_by=a)
    _artifact(conn, "github:o/r2", "github", linked_by=b)
    _artifact(conn, "paper:p1", "paper", linked_by=b)
    asked = optional_keys.question(conn, [a, b, c])
    assert asked["answers"] == ["github", "openalex"]
    assert ('"2 of the people you just picked share code on GitHub, and 1 cites research '
            'papers. Connect GitHub and add a free OpenAlex key so those come in faster; '
            'without them everything still arrives, just more slowly. Want either?"'
            in asked["message"])
    assert "onboard(start='github')" in asked["message"]
    assert "onboard(start='openalex')" in asked["message"]


def test_it_is_asked_once(conn):
    a = _oracle(conn, "1")
    _artifact(conn, "github:o/r1", "github", linked_by=a)
    assert optional_keys.question(conn, [a]) is not None
    assert optional_keys.question(conn, [a]) is None


def test_picks_that_link_neither_ask_nothing_and_leave_it_for_a_later_confirmation(conn, home):
    a, b = _oracle(conn, "1"), _oracle(conn, "2")
    _artifact(conn, "paper:p1", "paper", linked_by=b)
    assert optional_keys.question(conn, [a]) is None
    assert not (home / optional_keys.KEYS_OFFERED_MARKER).exists()
    assert optional_keys.question(conn, [b])["answers"] == ["openalex"]


def test_a_repo_they_own_counts_as_sharing_code(conn):
    """Their own repos, once discovery joined their GitHub account to them."""
    a = _oracle(conn, "1", "github:alice")
    _artifact(conn, "github:alice/tool", "github", who="github:alice")
    assert "The person you just picked shares code on GitHub." in (
        optional_keys.question(conn, [a])["message"])


def test_only_a_missing_key_is_asked_for(conn):
    a = _oracle(conn, "1")
    _artifact(conn, "github:o/r1", "github", linked_by=a)
    _artifact(conn, "paper:p1", "paper", linked_by=a)
    keys.set_key("OPENALEX_API_KEY", "oa_x")
    asked = optional_keys.question(conn, [a])
    assert asked["answers"] == ["github"]
    assert "OpenAlex" not in asked["message"]


def test_nothing_is_asked_when_both_keys_are_held(conn, home):
    a = _oracle(conn, "1")
    _artifact(conn, "github:o/r1", "github", linked_by=a)
    keys.set_key("GITHUB_TOKEN", "gho_x")
    keys.set_key("OPENALEX_API_KEY", "oa_x")
    assert optional_keys.question(conn, [a]) is None


def test_github_is_not_asked_for_before_its_sign_in_exists(conn, monkeypatch):
    monkeypatch.setattr(github_device, "CLIENT_ID", "")
    a = _oracle(conn, "1")
    _artifact(conn, "github:o/r1", "github", linked_by=a)
    _artifact(conn, "paper:p1", "paper", linked_by=a)
    asked = optional_keys.question(conn, [a])
    assert asked["answers"] == ["openalex"]
    assert "github" not in asked["message"].lower()


@pytest.mark.parametrize("people, counts, said", [
    (3, {"github": 2, "openalex": 1},
     "2 of the people you just picked share code on GitHub, and 1 cites research papers. "
     "Connect GitHub and add a free OpenAlex key so those come in faster; without them "
     "everything still arrives, just more slowly. Want either?"),
    (6, {"github": 1},
     "1 of the people you just picked shares code on GitHub. Connect GitHub so their code "
     "comes in faster; without it everything still arrives, just more slowly. Want to connect "
     "it?"),
    (6, {"openalex": 3},
     "3 of the people you just picked cite research papers. Add a free OpenAlex key so those "
     "papers come in faster; without it everything still arrives, just more slowly. Want to "
     "add it?"),
    (1, {"github": 1, "openalex": 1},
     "The person you just picked shares code on GitHub and cites research papers. Connect "
     "GitHub and add a free OpenAlex key so those come in faster; without them everything "
     "still arrives, just more slowly. Want either?"),
])
def test_the_line_counts_who_was_picked(people, counts, said):
    assert optional_keys.line(people, counts) == said


# ── a key from the page: checked before it is stored ─────────────────────────

class _Resp:
    def __init__(self, status):
        self.status_code = status


@pytest.mark.parametrize("service, status, stored", [
    ("openalex", 200, True),
    ("openalex", 401, False),            # OpenAlex does not know the key: every call would fail
])
def test_a_key_is_stored_only_when_the_service_knows_it(home, monkeypatch, service, status,
                                                        stored):
    monkeypatch.setattr(key_page, "take", lambda: "the-key")
    seen = {}
    monkeypatch.setattr(optional_keys.requests, "get",
                        lambda url, headers, timeout: seen.update(headers) or _Resp(status))
    answer = optional_keys.run(service, "sources")
    env = credentials.SERVICES[service]
    assert (keys._read().get(env) == "the-key") is stored
    assert answer["status"] == ("ok" if stored else "error")
    assert "the-key" in "".join(seen.values())          # the probe sent the candidate key
    assert "the-key" not in str(answer)                 # and the answer never carries it


def test_an_unreachable_service_stores_nothing(home, monkeypatch):
    monkeypatch.setattr(key_page, "take", lambda: "the-key")

    def down(*_a, **_kw):
        raise optional_keys.requests.ConnectionError("down")
    monkeypatch.setattr(optional_keys.requests, "get", down)
    assert optional_keys.run("openalex", "sources")["status"] == "error"
    assert "OPENALEX_API_KEY" not in keys._read()


def test_openalex_from_the_onboard_tool_stores_a_checked_key(home, monkeypatch):
    """From the real entry: model access's turn passes `openalex` on to this step rather than
    refusing it, and the stored key rides the next OpenAlex read."""
    monkeypatch.setattr(key_page, "take", lambda: "oa_good")
    monkeypatch.setattr(optional_keys.requests, "get",
                        lambda url, headers, timeout: _Resp(200))
    m = SimpleNamespace(tools={})
    m.tool = lambda **kw: (lambda fn: m.tools.setdefault(fn.__name__, fn))
    onboard_tools.register_onboard_tools(m)
    answer = m.tools["onboard"](start="openalex")
    assert answer["status"] == "ok" and answer["step"] == "openalex"
    assert credentials.openalex_headers() == {"Authorization": "Bearer oa_good"}


@pytest.mark.loopback
def test_openalex_hands_back_a_link_and_the_next_call_collects_the_key(home, monkeypatch):
    """The page is a link the host gives the user: nothing opens a browser, so it never appears
    unannounced (a host once called the word blind and the page popped up mid-chat). The
    same word again reports that no key has arrived, then collects and checks it once one has."""
    monkeypatch.setattr(key_page.local_auth, "open_browser",
                        lambda url: pytest.fail("the key page must not open a browser"))
    monkeypatch.setattr(optional_keys.requests, "get",
                        lambda url, headers, timeout: _Resp(200))
    try:
        first = optional_keys.run("openalex", "sources")
        url = first["open_this_url"]
        assert first["status"] == "waiting" and url.startswith("http://localhost:")
        assert "**Add your OpenAlex key to Opyt**" in first["message"]

        again = optional_keys.run("openalex", "sources")
        assert (again["status"], again["open_this_url"]) == ("waiting", url)
        assert "has not received a key yet" in again["message"]
        assert keys._read() == {}

        data = urllib.parse.urlencode({"key": "oa_pasted"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5).read()
        done = optional_keys.run("openalex", "sources")
        assert done["status"] == "ok"
        assert keys._read().get("OPENALEX_API_KEY") == "oa_pasted"
        assert key_page.waiting() is None                   # the page closed with its key
    finally:
        key_page.close()


# ── the key page itself, over a real loopback ────────────────────────────────

@pytest.mark.loopback
def test_the_key_page_serves_a_form_and_takes_one_key(monkeypatch):
    monkeypatch.setattr(key_page.local_auth, "open_browser", lambda url: True)
    got = {}
    with key_page.KeyCapture(name="OpenAlex", signup_url="https://openalex.org/settings/api",
                             timeout=10) as cap:
        def submit():
            form = urllib.request.urlopen(cap.url, timeout=5).read().decode()
            got["form"] = form
            data = urllib.parse.urlencode({"key": "  oa_secret  "}).encode()
            urllib.request.urlopen(urllib.request.Request(cap.url, data=data), timeout=5).read()
        t = threading.Thread(target=submit)
        t.start()
        payload = cap.wait()
        t.join()
    assert payload == {"key": "oa_secret"}
    assert 'type=password' in got["form"] and "openalex.org/settings/api" in got["form"]


@pytest.mark.loopback
def test_the_key_page_answers_nothing_off_its_nonce():
    with key_page.KeyCapture(name="OpenAlex", signup_url="https://x", timeout=1) as cap:
        wrong = f"http://localhost:{cap.port}/key/not-the-nonce"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(wrong, timeout=5)
        assert e.value.code == 404
        assert cap.wait() is None


# ── GitHub's device sign-in ──────────────────────────────────────────────────

class _JSON:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


@pytest.fixture()
def device(home, monkeypatch):
    monkeypatch.setattr(github_device, "CLIENT_ID", "Iv1.test")
    monkeypatch.setattr(github_device, "_pending", None)
    monkeypatch.setattr(github_device.local_auth, "open_browser", lambda url: True)
    replies = []
    monkeypatch.setattr(github_device.requests, "post",
                        lambda url, data, headers, timeout: _JSON(replies.pop(0)))
    return replies


def test_github_shows_a_code_then_stores_the_token_once_approved(device):
    device += [{"device_code": "dc", "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device", "expires_in": 900},
               {"error": "authorization_pending"},
               {"access_token": "gho_token"}]
    first = optional_keys.run("github", "sources")
    assert first["status"] == "waiting" and "ABCD-1234" in first["message"]
    assert optional_keys.run("github", "sources")["status"] == "waiting"   # not approved yet
    assert optional_keys.run("github", "sources")["status"] == "ok"
    assert keys._read()["GITHUB_TOKEN"] == "gho_token"
    assert github_device.pending() is None


def test_a_denied_github_sign_in_stores_nothing_and_starts_over(device):
    device += [{"device_code": "dc", "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device", "expires_in": 900},
               {"error": "access_denied"}]
    optional_keys.run("github", "sources")
    assert optional_keys.run("github", "sources")["status"] == "error"
    assert "GITHUB_TOKEN" not in keys._read()
    assert github_device.pending() is None


def test_an_expired_code_is_not_polled(device, monkeypatch):
    device += [{"device_code": "dc", "user_code": "C", "verification_uri": "u",
                "expires_in": 900}]
    github_device.begin()
    monkeypatch.setitem(github_device._pending, "expires_at", time.monotonic() - 1)
    assert github_device.pending() is None


def test_github_without_a_registered_app_says_so(home, monkeypatch):
    monkeypatch.setattr(github_device, "CLIENT_ID", "")
    monkeypatch.setattr(github_device, "_pending", None)
    assert optional_keys.run("github", "sources")["status"] == "error"


# ── a stored key reaches its calls, even in a process started before it ──────

def test_a_key_stored_after_start_beats_the_stale_copy_in_the_environment(home, monkeypatch):
    """The worker once held the spent starter key from its environment for hours after
    the user's own key was stored. The home file is the live record."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-stale")
    keys.set_key("OPENROUTER_API_KEY", "sk-fresh")
    assert credentials.get_credential("openrouter") == "sk-fresh"


def test_the_environment_still_answers_when_the_home_file_has_no_key(home, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    assert credentials.get_credential("github") == "from-env"


def test_a_github_token_stored_mid_session_reaches_the_rest_reads(home):
    keys.set_key("GITHUB_TOKEN", "gho_mid")
    assert github_api.gh_headers()["Authorization"] == "Bearer gho_mid"


def test_the_openalex_key_rides_every_openalex_read(home):
    assert credentials.openalex_headers() == {}
    keys.set_key("OPENALEX_API_KEY", "oa_k")
    assert credentials.openalex_headers() == {"Authorization": "Bearer oa_k"}


class _Breaker:
    def call(self, fn):
        return fn()


def test_openalex_reads_send_the_stored_key(home, monkeypatch):
    keys.set_key("OPENALEX_API_KEY", "oa_k")
    sent = []
    monkeypatch.setattr(paper_registry, "get",
                        lambda url, headers=None: sent.append(headers) or b'{"results": []}')
    openalex.OpenAlexClient(breaker=_Breaker())._call("/works", {})
    openalex.search_works("agent memory", since=None, limit=1, get_breaker=_Breaker)
    assert sent == [{"Authorization": "Bearer oa_k"}] * 2
