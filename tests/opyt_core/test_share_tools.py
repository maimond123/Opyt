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
from types import SimpleNamespace

import pytest

from opyt_core import kb as kb_entry, kb_remote, keys
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
    # The rail runs in-process here rather than as a detached child: `spawn_rail` would fork a
    # real subprocess against a real network, and what is under test is that `share` triggers a
    # publish, not that `Popen` works.
    monkeypatch.setattr("pipeline.kb.push_catchup.spawn_push_catchup",
                        lambda force=False: push_catchup.run_push_catchup(force=True)["status"]
                        == "ok")
    return SimpleNamespace(t=_tools(), publisher=publisher, creds=creds)


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


def test_a_first_share_registers_publishes_and_returns_a_link(owner):
    """The whole ask moment, and nobody opened a shell. The token is PERSISTED, so the next
    session's push rail can find it."""
    out = owner.t.share(confirm=True, as_name="David")

    assert out["status"] == "shared"
    assert out["invite"].startswith("https://useopyt.com/invite#")
    assert owner.creds["opyt_service"]
    with owner.publisher.on_service():
        assert store.owner_label(out["owner"]) == "David"
        assert uploads.export_path(out["owner"]).exists()


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


def test_something_that_is_not_an_invite_says_so(reader):
    out = reader.t.accept("hey check out https://example.com/blog/post")
    assert out["status"] == "not_an_invite"
    assert peers.list_peers() == []


# ── unshare ──────────────────────────────────────────────────────────────────────

def test_unshare_preview_names_the_reader_count_and_changes_nothing(owner):
    owner.t.share(confirm=True, as_name="David")
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
