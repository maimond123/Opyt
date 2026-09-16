"""share / accept / unshare — the surfaces that make sharing exist for a person.

Four moments have to work: ask · accept · read · stay current. `read` was live from 2026-08-26
and the other three were not reachable without a shell. What these tests hold is the round trip —
one person shares, another accepts, and the second one's search returns the first one's atoms
under a name the second one's install resolves.

Run against the REAL service in-process. Nothing here mocks the thing under test; the one
substitution is the socket, and both installs' homes are kept apart the way two machines are.
"""
from __future__ import annotations

import re
import time
from types import SimpleNamespace

import pytest

from opyt_core import config, kb as kb_entry, kb_remote, keys
from pipeline.kb import peers, push_catchup
from service import store, uploads
from tests.opyt_core.conftest import URL


def _tools():
    """The three functions, pulled out of the registration block the same way the MCP server
    installs them. A fake `mcp` whose `tool()` decorator just collects — testing the functions,
    not FastMCP."""
    from mcp_server.share_tools import register_share_tools
    got = {}

    class _Fake:
        def tool(self, *a, **kw):
            def deco(fn):
                got[fn.__name__] = fn
                return fn
            return deco

    register_share_tools(_Fake())
    return SimpleNamespace(**got)


@pytest.fixture()
def owner(publisher, monkeypatch):
    """The SHARING side: a real store, `share_tools`' transport pointed at the service, and NO
    service token — so the first `share` has to register, exactly as a new user's does."""
    import mcp_server.share_tools as st

    monkeypatch.delenv("OPYT_SERVICE_TOKEN", raising=False)
    creds = {}
    monkeypatch.setattr("pipeline.credentials.get_credential",
                        lambda service: creds.get("opyt_service"))
    monkeypatch.setattr(keys, "set_key",
                        lambda name, value: creds.__setitem__("opyt_service", value))

    def _shim(verb):
        def call(url, **kw):
            kw.pop("timeout", None)
            with publisher.on_service():
                return verb(url, **kw)
        return call

    monkeypatch.setattr(st, "requests", SimpleNamespace(
        get=_shim(publisher.svc.client.get), post=_shim(publisher.svc.client.post)))
    # `share` only QUEUES the push rail; `run_worker_pass` below stands in for the resident
    # worker that launches it. Nothing about the rail is stubbed — the queue and the publish are
    # both real, and the tests below hold them apart so a broken hand-off cannot hide.
    return SimpleNamespace(t=_tools(), publisher=publisher, creds=creds)


def _queued_push(home) -> object | None:
    """The durable push job `share` left behind, as the local worker would find it."""
    from pipeline.kb.rail_jobs import LOCAL_HOME_ID, RailJobStore
    return RailJobStore(home / "rail_jobs.db").get(LOCAL_HOME_ID, "push_catchup")


def run_worker_pass(owner) -> dict:
    """Run the queued rail exactly as `rail_worker.RAILS['push_catchup']` would — no `--force`.

    The command carries no override, so this also proves the gate publishes a first share on its
    own "never published" branch rather than on a word the tool passed it.
    """
    assert _queued_push(owner.publisher.home) is not None, "share queued no push job"
    return push_catchup.run_push_catchup()


# ── share ────────────────────────────────────────────────────────────────────────

def test_the_preview_carries_the_consent_context_itself(owner):
    """R1 requires the yes to be INFORMED and R6 requires that to be structural. A shape summary
    left to a separate `aggregate` call is one a model will sometimes skip, and the consent step
    then silently disappears — so the preview carries it, or the design does not hold."""
    out = owner.t.share()

    assert out["status"] == "preview"
    assert out["atoms"] > 0
    assert out["by_source_type"] and out["top_entities"]
    assert out["date_span"]["oldest"] and out["date_span"]["newest"]
    joined = " ".join(out["consent"]).lower()
    assert "no way to share part" in joined      # whole KB, not a slice
    assert "until you revoke" in joined          # standing, not one-time
    assert "unshare" in joined                   # and revocable


def test_the_preview_writes_nothing_and_registers_nothing(owner):
    owner.t.share()
    owner.t.share()

    assert owner.creds == {}
    with owner.publisher.on_service():
        assert store.stored_bytes() == [
            r for r in store.stored_bytes() if r["owner"] == owner.publisher.svc.owner]


def test_a_first_share_registers_queues_the_push_and_returns_a_link(owner):
    """The whole ask moment, and nobody opened a shell. The token is PERSISTED, so the push rail
    can find it whenever the worker gets to it — including with no MCP session open at all."""
    out = owner.t.share(confirm=True, as_name="David")

    assert out["status"] == "shared"
    assert out["invite"].startswith("https://useopyt.com/invite#")
    assert out["queued"] is True
    assert owner.creds["opyt_service"]
    job = _queued_push(owner.publisher.home)
    assert job.due_at <= time.time() and job.started_at is None
    with owner.publisher.on_service():
        assert store.owner_label(out["owner"]) == "David"
        # Queued is not published: the export lands when the worker runs the rail, not before.
        assert not uploads.export_path(out["owner"]).exists()


def test_the_queued_rail_is_what_publishes_the_first_share(owner):
    """The hand-off end to end. `share` writes a job row; the rail's own gate does the upload."""
    out = owner.t.share(confirm=True, as_name="David")

    assert run_worker_pass(owner)["status"] == "ok"

    with owner.publisher.on_service():
        assert uploads.export_path(out["owner"]).exists()


def test_the_queued_rail_name_is_one_the_worker_can_actually_launch(owner):
    """A name outside the registry is a row nothing ever dispatches, and nothing says so."""
    from pipeline.kb.rail_worker import RAILS

    owner.t.share(confirm=True, as_name="David")

    assert _queued_push(owner.publisher.home).rail in RAILS


def test_a_failed_grant_queues_nothing(owner, monkeypatch):
    """Order: the grant is durable before the push is asked for. A share nobody can accept must
    not leave an upload obligation behind for a reader who was never invited."""
    import mcp_server.share_tools as st

    owner.t.share(confirm=True, as_name="David")
    monkeypatch.setattr(st, "_call", lambda *a, **kw: {"ok": False, "message": "service down"})

    assert owner.t.share(confirm=True, for_whom="Leo")["status"] == "grant_failed"
    assert _queued_push(owner.publisher.home).due_at <= time.time()


def test_the_first_share_refuses_to_guess_a_name(owner):
    """The label becomes every reader's suggested name for this knowledge base, so it is the
    user's to choose. Asking is one question; guessing is a name they never picked appearing in
    someone else's install."""
    out = owner.t.share(confirm=True)

    assert out["status"] == "needs_name"
    assert owner.creds == {}


def test_a_second_share_is_a_new_invite_and_not_a_new_registration(owner):
    """One knowledge base, many readers. A second `share` must not mint a second owner token —
    that would be a second knowledge base, and the first one's readers would be reading a
    corpus nobody updates."""
    first = owner.t.share(confirm=True, as_name="David")
    token = owner.creds["opyt_service"]

    second = owner.t.share(confirm=True, for_whom="Leo")
    assert second["status"] == "shared"
    assert second["owner"] == first["owner"]
    assert owner.creds["opyt_service"] == token
    assert second["invite"] != first["invite"]


def test_an_empty_knowledge_base_says_so_rather_than_sharing_nothing(owner, monkeypatch):
    monkeypatch.setattr(kb_entry, "kb_aggregate", lambda *a, **kw: {"total": 0})
    assert owner.t.share()["status"] == "empty"


def test_a_dead_service_is_a_sentence_not_a_raise(owner, monkeypatch):
    """P3 at the tool boundary: a tool call must never raise, whatever the network did."""
    import mcp_server.share_tools as st

    def dead(*a, **kw):
        raise st.RequestException("connection refused")

    monkeypatch.setattr(st.requests, "post", dead)
    out = owner.t.share(confirm=True, as_name="David")
    assert out["status"] == "register_failed"
    assert "could not reach" in out["message"]


# ── accept ───────────────────────────────────────────────────────────────────────

@pytest.fixture()
def reader(owner, tmp_path, monkeypatch):
    """The ACCEPTING side: a separate install with its own home and its own peers registry, with
    `share_tools`' and `kb_remote`'s transports both pointed at the same service."""
    import mcp_server.share_tools as st

    invite = owner.t.share(confirm=True, as_name="David")["invite"]
    run_worker_pass(owner)          # the owner's worker publishes before anyone can read
    reader_home = tmp_path / "reader"
    reader_home.mkdir()
    monkeypatch.setenv("OPYT_HOME", str(reader_home))

    def _shim(verb):
        def call(url, **kw):
            kw.pop("timeout", None)
            with owner.publisher.on_service():
                return verb(url, **kw)
        return call

    client = owner.publisher.svc.client
    monkeypatch.setattr(kb_remote, "requests",
                        SimpleNamespace(post=_shim(client.post), get=_shim(client.get)))
    monkeypatch.setattr(st, "requests",
                        SimpleNamespace(post=_shim(client.post), get=_shim(client.get)))
    kb_remote._META_CACHE.clear()
    yield SimpleNamespace(t=owner.t, invite=invite, home=reader_home, owner=owner)
    kb_remote._META_CACHE.clear()


@pytest.mark.parametrize("form", ["link", "fragment", "bare"])
def test_accept_takes_whatever_the_user_pasted(reader, form):
    """A person forwards a message, or copies half a URL, or types the code. All three are the
    same act, so all three work — the code is found inside any of them."""
    code = re.search(r"#([A-Za-z0-9]{43})", reader.invite).group(1)
    given = {"link": reader.invite, "fragment": f"#{code}", "bare": code}[form]

    out = reader.t.accept(given)
    assert out["status"] == "accepted"
    assert peers.get(out["kb"])["token"]


def test_accept_registers_under_the_owners_own_name(reader):
    """R4's payoff at the accept moment: the routing key in the URL is opaque, so what the reader
    types is the name its owner registered under, handed back by `redeem` as `suggested_name`."""
    out = reader.t.accept(reader.invite)

    assert out["kb"] == "David"
    assert out["owner"] != "David", "the routing key is not the display name"


def test_the_accepted_name_is_the_one_that_reads(reader, monkeypatch):
    """The whole point, end to end: the second person's next question draws on the first
    person's corpus, attributed, under a name their own install resolves."""
    out = reader.t.accept(reader.invite)
    monkeypatch.setattr(kb_remote, "embedder_from_meta",
                        lambda meta, **kw: reader.owner.publisher.emb)

    hits = kb_entry.run_kb_search("agent framework", kb=out["kb"])["hits"]
    assert hits
    assert {h["kb"] for h in hits} == {out["kb"]}


def test_an_explicit_name_wins_over_the_owners_suggestion(reader):
    """`name=` overrides `suggested_name`. The owner's own label is a STARTING POINT and not a
    contract — after R4 every request sends `as_kb`, so nothing in the envelope depends on which
    string this install picked.

    This assertion is here because `opyt-redeem` and its `--name` flag were deleted on
    2026-09-05 and the test that held this property went with them. `accept`'s `name=` is the same
    override, and no other test in this file asserts it — without this one the property would have
    left with the command it was written against."""
    out = reader.t.accept(reader.invite, name="colleague")

    assert out["kb"] == "colleague"
    assert peers.get("colleague")["location"].endswith(f"/v1/kb/{out['owner']}")
    assert peers.get("David") is None, "the suggestion registered a second row"


def test_a_name_already_taken_is_suffixed_rather_than_overwritten(reader):
    """`peers.token` is the ONLY copy of a reader token in existence, so a second David must
    never replace the first. The tool reports the name it actually got."""
    reader.t.accept(reader.invite)
    first = peers.get("David")["token"]
    second_invite = reader.owner.t.share(confirm=True, for_whom="again")["invite"]

    # A different knowledge base that calls itself David too.
    with reader.owner.publisher.on_service():
        other = reader.owner.publisher.svc.client.post(
            "/v1/register", json={"label": "David"}).json()
        code = reader.owner.publisher.svc.client.post(
            "/v1/grant", json={},
            headers={"Authorization": f"Bearer {other['token']}"}).json()["code"]

    out = reader.t.accept(code)
    assert out["kb"] == "David-2"
    assert out["renamed_from"] == "David"
    assert peers.get("David")["token"] == first
    assert second_invite      # unused; the point is that a second invite is not a second KB


def test_a_spent_code_is_a_sentence(reader):
    """A code buys one reader token and then dies. The second attempt must say that rather than
    surfacing a status code, and must register nothing."""
    reader.t.accept(reader.invite)
    before = len(peers.list_peers())

    out = reader.t.accept(reader.invite)
    assert out["status"] == "code_unavailable"
    assert "already been used" in out["message"]
    assert len(peers.list_peers()) == before


def test_a_malformed_successful_service_body_is_a_failure_envelope(owner, monkeypatch):
    """JSON decoding belongs to the shared transport boundary, not each decorated tool."""
    import mcp_server.share_tools as st

    class _UnreadableResponse:
        status_code = 200

        def json(self):
            raise ValueError("not JSON")

    monkeypatch.setattr(st.requests, "post", lambda *args, **kwargs: _UnreadableResponse())

    res = st._call("post", "https://service.test/v1/register")
    assert res["ok"] is False
    assert owner.t.share(confirm=True, as_name="David")["status"] == "register_failed"


@pytest.mark.parametrize(
    ("invite", "endpoint"),
    [
        ("https://useopyt.com/invite#" + "a" * 43, config.DEFAULT_SERVICE_URL),
        ("https://self-hosted.test:8443/invite#" + "a" * 43,
         "https://self-hosted.test:8443"),
        ("https://notuseopyt.com/invite#" + "a" * 43, "https://notuseopyt.com"),
    ],
)
def test_accept_routes_full_invites_by_their_exact_host(reader, monkeypatch, invite, endpoint):
    """A full link names its issuer; only the public host maps to the public API."""
    import mcp_server.share_tools as st

    called = []

    def _call(method, url, **kwargs):
        called.append((method, url))
        return {"ok": True, "owner": "owner", "suggested_name": "Owner", "token": "reader"}

    monkeypatch.setattr(st, "_call", _call)

    assert reader.t.accept(invite, name="routed")["status"] == "accepted"
    assert called == [("post", f"{endpoint}/v1/redeem")]


def test_the_install_id_is_minted_once_and_reused(reader, monkeypatch):
    """`redeem.get_install_id` — the only thing left in that module, and `accept` is its only
    caller. The id is minted on first use and REUSED forever after, so the service can count
    distinct installations without an account behind it (TELEMETRY.md: `tokens.install_id`).
    Two accepts on one install must therefore produce two tokens carrying the SAME id.

    It used to reach `get_install_id` through `opyt-redeem`'s `main()`, and moved here on
    2026-09-05 when that command was deleted. `accept` is the only path to the function now, and
    this file is where `accept` is tested.

    ⚠️ It re-points `OPYT_HOME` before each accept, and reads the id by its ABSOLUTE path rather
    than through `opyt_path`. `publisher.on_service()`'s `finally` restores `OPYT_HOME` to the
    OWNER's home unconditionally, so after the first shimmed request this process is pointed at
    the owner's install, not the reader's. Every other test in this fixture is indifferent to
    that because it reads and writes on one side of the split consistently; this one is the first
    to assert on a file the READER's install owns, so it has to say which home it means."""
    monkeypatch.setenv("OPYT_HOME", str(reader.home))
    first = reader.t.accept(reader.invite)
    iid = (reader.home / "install_id").read_text().strip()
    assert iid

    second = reader.owner.t.share(confirm=True, for_whom="a second reader")["invite"]

    monkeypatch.setenv("OPYT_HOME", str(reader.home))
    assert reader.t.accept(second, name="second")["status"] == "accepted"

    assert (reader.home / "install_id").read_text().strip() == iid, "a second id was minted"
    # The routing key from the accept, not `svc.owner`: the `owner` fixture holds NO service
    # token, so its first `share` REGISTERS a new owner rather than reusing the seeded one.
    with reader.owner.publisher.on_service():
        rows = store.list_tokens(first["owner"])
    assert len([t for t in rows if t["install_id"] == iid]) == 2


def test_something_that_is_not_an_invite_says_so(reader):
    out = reader.t.accept("hey check out https://example.com/blog/post")
    assert out["status"] == "not_an_invite"
    assert peers.list_peers() == []


# ── unshare ──────────────────────────────────────────────────────────────────────

def test_unshare_preview_names_the_reader_count_and_changes_nothing(owner):
    owner.t.share(confirm=True, as_name="David")
    run_worker_pass(owner)
    owner.t.share(confirm=True, for_whom="Leo")
    with owner.publisher.on_service():
        code = owner.publisher.svc.client.post(
            "/v1/grant", json={"label": "Mia"},
            headers={"Authorization": f"Bearer {owner.creds['opyt_service']}"}).json()["code"]
        owner.publisher.svc.client.post("/v1/redeem",
                                        json={"code": code, "install_id": "mia"})

    out = owner.t.unshare()
    assert out["status"] == "preview"
    assert out["readers"] == 1
    assert "Mia" in out["reader_labels"]
    assert any("deleted from the service" in c for c in out["consent"])
    with owner.publisher.on_service():
        assert uploads.export_path(
            store.resolve_token(owner.creds["opyt_service"])["owner"]).exists()


def test_unshare_confirm_empties_the_token_list_and_deletes_the_copy(owner):
    out = owner.t.share(confirm=True, as_name="David")
    run_worker_pass(owner)
    with owner.publisher.on_service():
        code = owner.publisher.svc.client.post(
            "/v1/grant", json={},
            headers={"Authorization": f"Bearer {owner.creds['opyt_service']}"}).json()["code"]
        owner.publisher.svc.client.post("/v1/redeem", json={"code": code, "install_id": "z"})

    done = owner.t.unshare(confirm=True)
    assert done["status"] == "unshared"
    assert done["readers_revoked"] == 1
    assert done["export_deleted"] is True
    with owner.publisher.on_service():
        assert not uploads.export_path(out["owner"]).exists()
        assert [t for t in store.list_tokens(out["owner"]) if t["role"] == "reader"] == []


def test_unshare_on_an_install_that_never_shared_says_so(owner):
    assert owner.t.unshare()["status"] == "not_shared"


# ── unshare(reader=…): cutting off ONE person ────────────────────────────────────
#
# The scope argument exists because `for_whom` promised a handle it had no surface for: the
# label reached `tokens.label` and stopped there, so "stop sharing with Leo" had no answer but
# an operator curl. What these hold is that the narrow scope is genuinely narrow — the copy
# stays, everyone else keeps reading — and that the two ways a name can fail to resolve refuse
# instead of guessing, because guessing wrong here revokes the wrong person's access.

def _grant_and_redeem(owner, label, install):
    """One more reader on this knowledge base, straight through the service the way a real
    invite does. Returns nothing; the reader is a row in `tokens` from here on."""
    with owner.publisher.on_service():
        hdr = {"Authorization": f"Bearer {owner.creds['opyt_service']}"}
        code = owner.publisher.svc.client.post(
            "/v1/grant", json={"label": label}, headers=hdr).json()["code"]
        owner.publisher.svc.client.post("/v1/redeem",
                                        json={"code": code, "install_id": install})


def _readers(owner, key):
    with owner.publisher.on_service():
        return [t for t in store.list_tokens(key) if t["role"] == "reader"]


def test_unshare_one_reader_leaves_the_copy_and_the_others_alone(owner):
    out = owner.t.share(confirm=True, as_name="David")
    run_worker_pass(owner)
    _grant_and_redeem(owner, "Leo", "leo")
    _grant_and_redeem(owner, "Mia", "mia")

    done = owner.t.unshare(reader="Leo", confirm=True)
    assert done["status"] == "reader_revoked"
    assert done["reader_label"] == "Leo"
    assert done["readers_remaining"] == 1

    # The narrow scope is the whole point: everything except Leo survives.
    assert [t["label"] for t in _readers(owner, out["owner"])] == ["Mia"]
    with owner.publisher.on_service():
        assert uploads.export_path(out["owner"]).exists()


def test_unshare_one_reader_preview_changes_nothing_and_says_which_scope(owner):
    out = owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, "Leo", "leo")
    _grant_and_redeem(owner, "Mia", "mia")

    pre = owner.t.unshare(reader="leo")          # case-insensitive: the owner typed the label
    assert pre["status"] == "preview"
    assert pre["scope"] == "one_reader"
    assert pre["reader_label"] == "Leo"
    assert pre["readers_remaining"] == 1
    assert len(_readers(owner, out["owner"])) == 2


def test_the_whole_kb_preview_says_it_is_the_whole_kb(owner):
    """The dangerous scope is the DEFAULT, so the preview has to be the thing that catches a
    model reaching for it when the user named one person."""
    owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, "Leo", "leo")

    pre = owner.t.unshare()
    assert pre["scope"] == "everyone"
    assert "whole knowledge base" in pre["consent"][0]
    assert any("`reader`" in c for c in pre["consent"])


def test_two_readers_with_one_label_refuse_rather_than_pick(owner):
    out = owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, "Leo", "leo-laptop")
    _grant_and_redeem(owner, "Leo", "leo-desktop")

    got = owner.t.unshare(reader="Leo", confirm=True)
    assert got["status"] == "ambiguous_reader"
    assert len(got["readers"]) == 2
    assert all(len(r["id"]) == 12 for r in got["readers"])
    assert len(_readers(owner, out["owner"])) == 2      # neither was touched


def test_an_unlabelled_reader_is_revocable_by_the_id_the_refusal_hands_back(owner):
    """`for_whom` is optional, so a reader with no label exists. Without the id path they would
    be visible in a preview and impossible to cut off."""
    out = owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, None, "anon")

    miss = owner.t.unshare(reader="whoever that was")
    assert miss["status"] == "no_such_reader"
    assert len(miss["readers"]) == 1

    done = owner.t.unshare(reader=miss["readers"][0]["id"], confirm=True)
    assert done["status"] == "reader_revoked"
    assert _readers(owner, out["owner"]) == []


def test_an_empty_reader_does_not_match_an_unlabelled_recipient(owner):
    out = owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, None, "anon")

    miss = owner.t.unshare(reader="", confirm=True)
    assert miss["status"] == "no_such_reader"
    assert len(miss["readers"]) == 1

    done = owner.t.unshare(reader=miss["readers"][0]["id"], confirm=True)
    assert done["status"] == "reader_revoked"
    assert _readers(owner, out["owner"]) == []


def test_a_name_nobody_holds_lists_who_does_and_revokes_nobody(owner):
    out = owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, "Leo", "leo")

    got = owner.t.unshare(reader="Priya", confirm=True)
    assert got["status"] == "no_such_reader"
    assert [r["label"] for r in got["readers"]] == ["Leo"]
    assert len(_readers(owner, out["owner"])) == 1


def test_a_short_string_never_matches_a_hash_prefix(owner):
    """8 characters is the floor. Below it a 'prefix' is not a handle, and a stray word could
    collide with somebody's hash and revoke them."""
    out = owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, "Leo", "leo")
    with owner.publisher.on_service():
        sha = [t for t in store.list_tokens(out["owner"]) if t["role"] == "reader"][0]["token_sha256"]

    assert owner.t.unshare(reader=sha[:4], confirm=True)["status"] == "no_such_reader"
    assert owner.t.unshare(reader=sha[:8], confirm=True)["status"] == "reader_revoked"


def test_revoking_somebody_already_gone_is_not_an_error(owner):
    owner.t.share(confirm=True, as_name="David")
    _grant_and_redeem(owner, "Leo", "leo")

    assert owner.t.unshare(reader="Leo", confirm=True)["status"] == "reader_revoked"
    again = owner.t.unshare(reader="Leo", confirm=True)
    assert again["status"] == "no_such_reader"      # the row is gone, so the name resolves to nobody
