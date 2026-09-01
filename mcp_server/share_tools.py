"""
mcp_server/share_tools.py — the three tools that make knowledge-base sharing exist for a person.

Registered with `register_share_tools(mcp)` like every other tool group. Four moments have to
work — ask · accept · read · stay current — and until these tools existed only *read* did. The
transport, the service, the export builder and the push rail were all built and live; what was
missing was every surface a person touches. Nothing minted an invite, nothing accepted one, and
publishing meant an SSH session.

The constraint everything here answers to is FRICTIONLESS UX, and it is what forces the shapes
below: no shell on either side, no credential a human transcribes, and ONE yes per person,
covering everything after it.

  • `share`   — two-phase. The preview IS the consent context (R1/R6): the yes must be informed,
                and a shape summary that is a separate tool call is one a model will sometimes
                skip, which makes the consent step silently disappear. Building it into the
                preview makes it structural rather than hoped-for.
  • `accept`  — SINGLE-phase, deliberately. A preview cannot validate a one-time code without
                burning it, and pasting the invite is itself the yes.
  • `unshare` — two-phase, and TWO SCOPES under one verb. `reader="Leo"` cuts off one person and
                leaves the copy serving; omitting it cuts everyone AND deletes the copy. Same
                verb with a narrower object, the way `search(kb=)` is, rather than a second tool
                sitting beside this one under a name a model would have to choose between.
                ⚠️The larger act is the DEFAULT, which is the wrong direction for a default and
                is why the preview leads with which scope it is about. `for_whom` at share time
                is what makes a reader nameable here.

Every failure is a sentence in the return value, never a raise (P3).

⚠️THE FIRST PUBLISH IS DETACHED. `share`'s confirm spawns the push rail and returns the invite
immediately, because an inline publish is minutes of residential upload inside one MCP call,
which a host timeout can kill half-done — leaving the person registered, unpublished, and holding
an error instead of a link. The cost is a one-to-few-minute window where the link resolves and
the export does not; `service/app.py`'s `_served` answers that with a sentence saying so.
"""
from __future__ import annotations

import re
import sqlite3

import requests
from requests import RequestException

# A grant code is 43 characters of letters and digits (`store._CODE_LEN`/`_CODE_ALPHABET`). The
# pattern extracts one from a bare code OR from anywhere in a link, so a person can paste the
# whole invite, the URL fragment, or just the code, and all three work. Bounded on both sides so
# it cannot swallow a longer run of characters and hand the service a code that is nearly right.
_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9]{43})(?![A-Za-z0-9])")

_INVITE_BASE = "https://useopyt.com/invite"

_TIMEOUT = 30


def _extract_code(invite: str) -> str | None:
    """The 43-character code inside whatever the user pasted. None if there isn't one."""
    m = _CODE.search(invite or "")
    return m.group(1) if m else None


def _corpus_span() -> tuple[str | None, str | None]:
    """`(oldest, newest)` publication date in the local store — the one thing the consent context
    needs that `kb_aggregate` does not return. Fail-safe: an unreadable store is `(None, None)`,
    and the preview says the span is unknown rather than refusing to preview."""
    from opyt_core.paths import opyt_db
    try:
        conn = sqlite3.connect(f"file:{opyt_db()}?mode=ro", uri=True)
    except Exception:
        return None, None
    try:
        # `IS NULL OR = ''` is how the rest of the codebase spells "undated" — `retrieve.py`'s
        # `_count_undated` uses the same pair, because both forms are in the corpus.
        row = conn.execute("SELECT MIN(when_ts), MAX(when_ts) FROM atoms "
                           "WHERE when_ts IS NOT NULL AND when_ts != ''").fetchone()
        return (row[0], row[1]) if row else (None, None)
    except sqlite3.OperationalError:
        return None, None
    finally:
        conn.close()


def _call(method: str, url: str, *, token: str | None = None, json: dict | None = None) -> dict:
    """One request to the service. `{ok: True, **body}` or `{ok: False, message}` — never raises,
    because every caller here turns a failure into a sentence in a tool result."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = (requests.get(url, headers=headers, timeout=_TIMEOUT) if method == "get"
             else requests.post(url, json=json or {}, headers=headers, timeout=_TIMEOUT))
    except RequestException as e:
        return {"ok": False, "message": f"could not reach {url}: {e}"}
    if not 200 <= r.status_code < 300:
        from opyt_core.kb_remote import error_detail
        return {"ok": False, "message": f"the service answered {r.status_code}: {error_detail(r)}"}
    return {"ok": True, **r.json()}


def _reader_roster(tokens: list[dict]) -> list[dict]:
    """Every reader, as the pair a person can actually name one by: the label their owner gave
    them at `share` time, and a short prefix of the hash. The prefix exists because `for_whom` is
    optional — an unlabelled reader has no other handle, and without one it could be seen in a
    preview and never revoked."""
    return [{"label": t["label"], "id": t["token_sha256"][:12], "since": t["created_at"]}
            for t in tokens if t["role"] == "reader"]


def _unshare_one(url: str, token: str, reader: str, confirm: bool) -> dict:
    """Cut off ONE reader. The served copy stays and everybody else keeps reading.

    Resolution is deliberately strict in one direction and forgiving in the other. A label match
    is case-insensitive, because "leo" and "Leo" are the same person and the owner typed the
    label themselves. But two readers sharing a label REFUSES rather than picking the first: the
    two acts are indistinguishable from here, the wrong one is somebody's access, and the caller
    can re-run with the id that the refusal hands back. Guessing would be silent and wrong half
    the time.

    A hash prefix is accepted at 8 characters or more. Shorter is not a handle, it is a collision
    waiting to happen against a 64-character hex string."""
    state = _call("get", f"{url}/v1/tokens", token=token)
    if not state.get("ok"):
        return {"status": "unreachable", "message": state["message"]}

    readers = [t for t in state["tokens"] if t["role"] == "reader"]
    want = reader.strip()
    hits = [t for t in readers
            if (t["label"] or "").strip().lower() == want.lower()
            or (len(want) >= 8 and t["token_sha256"].startswith(want.lower()))]

    if not hits:
        return {"status": "no_such_reader",
                "readers": _reader_roster(readers),
                "message": (f"nobody holding access to this knowledge base is called "
                            f"{reader!r}. The current readers are listed in `readers` — pass a "
                            f"`label` or an `id` from that list. An unlabelled reader is one "
                            f"who was invited without a `for_whom`.")
                if readers else
                "nobody holds access to this knowledge base, so there is no one to cut off."}

    if len(hits) > 1:
        return {"status": "ambiguous_reader",
                "readers": _reader_roster(hits),
                "message": (f"{len(hits)} readers are called {reader!r}, so this will not guess "
                            f"which one to cut off. Ask the user which, and pass the `id` of "
                            f"that one from `readers`.")}

    hit = hits[0]
    label = hit["label"] or hit["token_sha256"][:12]
    if not confirm:
        return {"status": "preview",
                "scope": "one_reader",
                "reader_label": label,
                "reader_id": hit["token_sha256"][:12],
                "readers_remaining": len(readers) - 1,
                "service": url,
                "consent": [
                    f"This cuts off {label} and nobody else. "
                    f"{len(readers) - 1} other reader(s) keep their access.",
                    "The served copy of this knowledge base stays up.",
                    f"The invite {label} redeemed was already spent, so nothing they hold keeps "
                    f"working. Letting them back in means a new `share` link.",
                ]}

    res = _call("post", f"{url}/v1/revoke", token=token,
                json={"token_sha256": hit["token_sha256"]})
    if not res.get("ok"):
        return {"status": "failed", "message": res["message"]}
    if not res.get("revoked"):
        # The row went between the read above and this write: another session revoked them, or
        # the owner unpublished. The end state is the one asked for, so this is not a failure.
        return {"status": "already_gone", "reader_label": label,
                "message": f"{label} no longer had access, so nothing changed."}
    return {"status": "reader_revoked",
            "reader_label": label,
            "readers_remaining": len(readers) - 1,
            "message": (f"{label} loses access on their very next request. The knowledge base is "
                        f"still shared with everyone else.")}


def register_share_tools(mcp) -> None:

    @mcp.tool()
    def share(confirm: bool = False, as_name: str | None = None,
              for_whom: str | None = None) -> dict:
        """SHARE this knowledge base with someone — returns a link to send them.

        Reach for this when the user says share my KB / send my knowledge base to X / let X search
        what I've read. It hands back one link. The person who opens it can install Opyt if they
        need to, and after that their assistant can search this corpus, with every result
        attributed here.

        TWO-PHASE, and the preview is the consent step, so run it first:
          • confirm=False (the default) → a PREVIEW. It publishes nothing, mints nothing and
            hands nobody anything; on an install that has already shared it asks the service
            whether an export has landed, and that is the only thing it sends. It
            reports WHAT would be shared — how many atoms, from which sources and authors, over
            what date span — and what sharing means. READ THAT BACK to the user before confirming.
            Sharing is the WHOLE knowledge base, standing, not a slice and not a snapshot: there
            is no way to share part of it, anyone given a link keeps access until it is revoked,
            and the copy refreshes itself when people read it.
          • confirm=True → registers this install with the service if it is not registered yet,
            starts the first upload, mints a one-time invite, and returns the link.

        Skip straight to confirm=True only when the user has already said yes to sharing THIS
        knowledge base, having been told what is in it.

        The first share uploads the whole corpus, which runs in the background and usually takes a
        minute or two. The link works immediately: a reader who accepts inside that window is
        registered normally, and it is their first SEARCH that comes back saying the copy is
        still arriving. The next one works. Later shares are instant.

        WHAT THIS IS NOT. It does not publish anything on the open web — the service serves only
        people holding a link, one reader token per link, revocable. It does not send anything to
        the person for you: hand the user the link and let them send it.

        Args:
            confirm: False (default) = preview only, nothing written or sent; True = register,
                publish and mint the invite.
            as_name: what to call this knowledge base's owner. This is what every recipient's
                install suggests as the name they search under, so use the user's own name or
                handle. Required on the FIRST share; ignored afterwards, because the name is
                already registered.
            for_whom: an optional label for who this particular invite is for ("Leo"). It is the
                handle that makes that person nameable afterwards: `unshare(reader="Leo")` cuts
                off exactly them. Without it they can only be named by a token id.

        Returns {status, ...}. A preview carries `atoms`, `by_source_type`, `top_entities`,
        `date_span`, `consent` and `already_shared`. A confirm carries `invite` — the link — plus
        `owner` and `publishing`.
        """
        from opyt_core import config, kb as kb_entry, keys
        from pipeline.credentials import get_credential

        url = config.service_url().rstrip("/")
        token = get_credential("opyt_service")

        if not confirm:
            agg = kb_entry.kb_aggregate()
            if not agg.get("total"):
                return {"status": "empty",
                        "message": "this knowledge base has no atoms yet, so there is nothing to "
                                   "share. Run `onboard` or save something first."}
            oldest, newest = _corpus_span()
            published = False
            if token:
                state = _call("get", f"{url}/v1/tokens", token=token)
                published = bool(state.get("ok") and state.get("last_upload_at"))
            return {
                "status": "preview",
                "atoms": agg["total"],
                "by_source_type": agg["by_source_type"],
                "by_what_kind": agg["by_what_kind"],
                "top_entities": agg["top_entities"],
                "top_topics": agg["top_topics"],
                "trusted_atoms": agg["trusted_atoms"],
                "date_span": {"oldest": oldest, "newest": newest},
                "already_shared": bool(token),
                "already_published": published,
                "service": url,
                "consent": [
                    f"All {agg['total']} atoms would be uploaded to {url} and served from there. "
                    f"There is no way to share part of a knowledge base.",
                    "Anyone you send a link to keeps access until you revoke it — it is standing "
                    "access, not a one-time copy.",
                    "The served copy refreshes itself when somebody reads it, so it stays current "
                    "with what you save from now on.",
                    "`unshare` cuts every reader and deletes the copy from the service.",
                ],
            }

        if not token:
            if not as_name:
                return {"status": "needs_name",
                        "message": "ask the user what to call their knowledge base — their name "
                                   "or handle — and pass it as `as_name`. Everyone they share "
                                   "with sees it as the name to search under."}
            reg = _call("post", f"{url}/v1/register", json={"label": as_name})
            if not reg.get("ok"):
                return {"status": "register_failed", "message": reg["message"]}
            token = reg["token"]
            keys.set_key("OPYT_SERVICE_TOKEN", token)

        grant = _call("post", f"{url}/v1/grant", token=token, json={"label": for_whom})
        if not grant.get("ok"):
            return {"status": "grant_failed", "message": grant["message"]}

        # DETACHED, and after the grant so a failed mint does not leave an upload running for an
        # invite that was never handed over. See the module docstring for why not inline.
        from pipeline.kb.push_catchup import spawn_push_catchup
        publishing = spawn_push_catchup(force=True)

        return {"status": "shared",
                "owner": grant["owner"],
                "invite": f"{_INVITE_BASE}#{grant['code']}",
                "publishing": publishing,
                "message": ("Send them this link. It works immediately; if this is the first "
                            "share the upload finishes in the background, usually within a "
                            "minute or two.")}

    @mcp.tool()
    def accept(invite: str, name: str | None = None) -> dict:
        """ACCEPT an invitation to search somebody else's knowledge base.

        Reach for this the moment the user pastes an Opyt invite link or a grant code, or says
        someone shared their knowledge base with them. One call and it is done: from then on the
        search / open / aggregate tools take `kb='<name>'` and read that person's corpus, with
        every result attributed to them.

        SINGLE-PHASE on purpose — there is no preview. A grant code buys exactly one reader token
        and then dies, so checking it would spend it, and pasting an invite is already the yes.

        Pass whatever the user gave you: the whole link, the fragment, or the bare code. The code
        is found inside any of them.

        Args:
            invite: the invite link or grant code.
            name: OPTIONAL — what `kb=` should call this knowledge base locally. Leave it out and
                the owner's own name is used. Pass one only if the user asks for a specific name.

        Returns {status, kb, owner, message}. `kb` is the name to pass as `kb=` — use THAT string,
        not the owner's, because a name already taken on this install gets a suffix.
        """
        from opyt_core.redeem import get_install_id
        from pipeline.kb import peers

        code = _extract_code(invite)
        if not code:
            return {"status": "not_an_invite",
                    "message": "that does not contain an Opyt invite code. An invite looks like "
                               "https://useopyt.com/invite#<code>, where the code is 43 letters "
                               "and digits. Ask them to send the link again."}

        # The service is the one the LINK points at when it names one, so an invite to somebody's
        # self-hosted service works without this install configuring anything.
        m = re.match(r"(https?://[^\s/]+)", invite.strip())
        from opyt_core import config
        url = (config.service_url() if not m or "useopyt.com" in m.group(1)
               else m.group(1)).rstrip("/")

        res = _call("post", f"{url}/v1/redeem",
                    json={"code": code, "install_id": get_install_id()})
        if not res.get("ok"):
            return {"status": "code_unavailable",
                    "message": ("that invite code has already been used or is not one this "
                                "service issued — a code works exactly once. Ask them to send a "
                                "new one.")}

        asked = name or res.get("suggested_name") or res["owner"]
        kb = peers.add(asked, f"{url}/v1/kb/{res['owner']}", res.get("suggested_name"),
                       token=res["token"])
        out = {"status": "accepted", "kb": kb, "owner": res["owner"],
               "message": f"Search it with kb='{kb}'."}
        if kb != asked:
            out["renamed_from"] = asked
            out["message"] += (f" '{asked}' was already another knowledge base on this install, "
                               f"so this one is '{kb}' — tell the user that.")
        if res.get("notice"):
            out["service_notice"] = res["notice"]
        return out

    @mcp.tool()
    def unshare(confirm: bool = False, reader: str | None = None) -> dict:
        """STOP sharing — either with ONE person, or with everybody. `reader` picks which.

        READ THIS BEFORE CALLING. The two scopes are not the same act and only one of them is
        expensive to undo:
          • `reader="Leo"` → Leo loses access. Everyone else keeps reading, the served copy
            stays, and you can invite Leo again with `share` whenever you like.
          • `reader` omitted → EVERY reader is cut off AND the served copy is deleted. Every
            invite ever sent stops working, so sharing again means re-inviting everyone by hand.

        So when the user names a person — "stop sharing with Leo", "cut Leo off", "revoke Leo" —
        `reader` is REQUIRED. Omitting it there does something much larger than what was asked,
        and the preview is where you catch that: it always says which of the two scopes it is
        about, in its first sentence.

        Reach for the whole-knowledge-base form when the user says stop sharing / unshare / take
        my knowledge base down / revoke everyone. That form is one act and not two, because a
        person who says "stop sharing my KB" means both halves and will not say it twice.

        TWO-PHASE either way:
          • confirm=False (the default) → a PREVIEW of exactly what that scope would do. Nothing
            is changed.
          • confirm=True → does it. Access ends on that reader's very next request; there is no
            refresh cycle and no window.

        Args:
            confirm: False (default) = preview only; True = do it.
            reader: WHO to cut off, matched against the label you gave when you shared
                (`share`'s `for_whom`), case-insensitively. A token hash from a previous preview
                also works, which is how you name a reader you never labelled. Omit ONLY when the
                user means everybody.

        Returns {status, ...}. A `reader` call carries `reader_label`; a whole-knowledge-base
        confirm carries `readers_revoked` and `export_deleted`. `scope` is on every preview and
        says which of the two you are looking at.
        """
        from opyt_core import config
        from pipeline.credentials import get_credential

        token = get_credential("opyt_service")
        if not token:
            return {"status": "not_shared",
                    "message": "this knowledge base has never been shared, so there is nothing "
                               "to stop."}
        url = config.service_url().rstrip("/")

        if reader is not None:
            return _unshare_one(url, token, reader, confirm)

        if not confirm:
            state = _call("get", f"{url}/v1/tokens", token=token)
            if not state.get("ok"):
                return {"status": "unreachable", "message": state["message"]}
            readers = [t for t in state["tokens"] if t["role"] == "reader"]
            return {"status": "preview",
                    "scope": "everyone",
                    "readers": len(readers),
                    "reader_labels": [t["label"] for t in readers if t["label"]],
                    "service": url,
                    "consent": [
                        f"This takes the whole knowledge base down, not one reader. All "
                        f"{len(readers)} of them lose access on their next request.",
                        "The copy of this knowledge base is deleted from the service.",
                        "Every invite already sent stops working. Sharing again means new "
                        "invites for everyone.",
                        "To cut off one person instead, pass their name as `reader`.",
                    ]}

        res = _call("post", f"{url}/v1/unpublish", token=token)
        if not res.get("ok"):
            return {"status": "failed", "message": res["message"]}
        return {"status": "unshared",
                "readers_revoked": res["readers_revoked"],
                "export_deleted": res["export_deleted"],
                "message": (f"{res['readers_revoked']} reader(s) cut off and the served copy "
                            f"deleted. Nothing of this knowledge base is on the service now.")}
