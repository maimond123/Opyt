"""
pipeline/sharing/share.py — the share flow: a consent preview, or an invite with its first push
owed; and `refresh`, which owes a push now instead of at the next pass.

Two-phase: the preview is the consent context, so the yes is informed. Who it is for (`to`) picks
what a yes does:

  person      an invite for someone else, who reads.
  agent       an invite for the user's own agent, which acts as them; worded in `agent_invite.py`,
              which the connector's `share` says too.
  connector   the user's own claude.ai or ChatGPT, which acts as them: no invite, a Google sign-in
              (`link.py`).

The first publish is queued, not inline: the confirm owes a push (`push_gate.owe`), wakes the
scrapers process and returns the invite at once, since an inline upload of minutes can be killed
by a host timeout. Until the pass lands, the link resolves and the export does not. The publishing
gate decides what to push, so a share to a second reader of an unchanged store pushes nothing.

Every failure is a sentence in the return value, never a raise.
"""
from __future__ import annotations

import sqlite3

from pipeline.retrieval import reads
from pipeline.kb import schema
from pipeline.retrieval import filters
from pipeline.worker import job_queue
from pipeline.worker.registry import LINE, SCRAPERS
from pipeline.sharing import account, agent_invite, client, invite, link, push_gate

PERSON, AGENT, CONNECTOR = "person", "agent", "connector"

# What the preview hands the host to ask, so nobody gets a reader's link for an agent.
_WHO = ("ASK who this is for before confirming, unless the user already said: someone else, "
        "who reads (this preview); an agent the user runs, which acts as them (`to='agent'`); "
        "or the user's own claude.ai or ChatGPT (`to='connector'`).")

def run(confirm: bool, as_name: str | None, for_whom: str | None, to: str) -> dict:
    """Preview what sharing means, or do it: for a person or an agent, register if this install
    never has, mint the invite and owe the first push; for the connector, one step of the Google
    sign-in."""
    if to not in (PERSON, AGENT, CONNECTOR):
        return {"status": "error",
                "message": f"`to={to!r}` is not one of `person`, `agent` or `connector`. "
                           "Nothing was shared."}
    acct = account.load()
    if not confirm:
        return _preview(acct, to)
    if to == CONNECTOR:
        return link.run()
    agent = to == AGENT
    if acct.token is None:
        if not as_name:
            return {"status": "needs_name",
                    "message": "ask the user what to call their knowledge base — their name "
                               "or handle — and pass it as `as_name`. Everyone they share "
                               "with sees it as the name to search under."}
        try:
            reg = client.register(acct.url, as_name)
        except client.Failed as e:
            return {"status": "register_failed", "message": str(e)}
        acct = account.save_token(acct, reg["token"])
    try:
        grant = client.grant(acct, for_whom, agent)
    except client.Failed as e:
        return {"status": "grant_failed", "message": str(e)}
    # After the grant, so a failed mint leaves no upload running for an invite never handed over.
    push_gate.owe()
    push_gate.start_fast_hour()
    queued = job_queue.request_now(SCRAPERS)
    if agent:
        # An agent acts through the line, which a library with no writer does not hold: ask now,
        # not at the line's next hourly look.
        job_queue.request_now(LINE)
    # Who the link goes to; the rest is true of either kind of invite.
    lead = agent_invite.handover(acct.url) if agent else "Send them this link."
    return {"status": "shared",
            "owner": grant["owner"],
            "invite": invite.link(grant["code"]),
            "queued": queued,
            "message": (f"{lead} It works immediately; if this is the first "
                        "share the background worker publishes the export a moment later, "
                        "usually within a minute or two. Tell the user, in one line, that "
                        "their shared copy updates itself every hour while this Mac is "
                        "awake, and that saying \"update my shared library\" updates it at "
                        "once.")}


def refresh() -> dict:
    """Push the library to the served copy now, not at the next pass: owe a push and wake the
    scrapers process. The gate still decides, so an unchanged library uploads nothing."""
    if account.load().token is None:
        return {"status": "not_shared",
                "message": "nothing from this library is shared, so there is no copy to update."}
    skip = push_gate.decide(push_gate.store_position(), push_gate.read_watermark())
    if skip is not None:
        return skip
    push_gate.owe()
    return {"status": "refreshing",
            "queued": job_queue.request_now(SCRAPERS),
            "message": "Uploading the library now. Readers see it in a minute or two, longer "
                       "for a large library."}


def _preview(acct: account.Account, to: str) -> dict:
    """What would be shared, and what sharing means: with a person, with an agent that acts as
    the user, or with the user's own claude.ai or ChatGPT. Publishes nothing and mints nothing; on
    an install that has already shared, it asks the service whether an export has landed."""
    agg = reads.kb_aggregate()
    # The user's own claude.ai can be connected before the library holds anything: it fills.
    if not agg.get("total") and to != CONNECTOR:
        return {"status": "empty",
                "message": "this knowledge base has no atoms yet, so there is nothing to "
                           "share. Run `onboard` or save something first."}
    oldest, newest = _corpus_span()
    return {
        "status": "preview",
        "atoms": agg["total"],
        "by_source_type": agg["by_source_type"],
        "by_what_kind": agg["by_what_kind"],
        "top_entities": agg["top_entities"],
        "trusted_atoms": agg["trusted_atoms"],
        "date_span": {"oldest": oldest, "newest": newest},
        "already_shared": bool(acct.token),
        "already_published": _published(acct),
        "service": acct.url,
        **({"who_is_it_for": _WHO} if to == PERSON else {}),
        "consent": [
            f"All {agg['total']} atoms would be uploaded to {acct.url} and served from there. "
            f"There is no way to share part of a knowledge base.",
            *({AGENT: agent_invite.CONSENT, CONNECTOR: link.CONSENT}.get(to) or [
                "Anyone you send a link to keeps access until you revoke it — it is standing "
                "access, not a one-time copy."]),
            "Your shared copy updates itself every hour while this Mac is awake. To "
            "update it right away, say \"update my shared library\".",
            "`unshare` cuts every reader and deletes everything the service holds for this "
            "knowledge base.",
        ],
    }


def _published(acct: account.Account) -> bool:
    """Whether the service has ever received an export from this install. False when it never
    shared, and when the service cannot be asked: a preview must not fail on a dead service."""
    if not acct.token:
        return False
    try:
        return bool(client.tokens(acct).get("last_upload_at"))
    except client.Failed:
        return False


def _corpus_span() -> tuple[str | None, str | None]:
    """`(oldest, newest)` publication date in the local store — the one thing the consent context
    needs that `kb_aggregate` does not return. Fail-safe: an unreadable store is `(None, None)`,
    and the preview says the span is unknown rather than refusing to preview."""
    try:
        conn = schema.connect(read_only=True)
    except Exception:
        return None, None
    try:
        return filters.date_span(conn)
    except sqlite3.OperationalError:
        return None, None
    finally:
        conn.close()
