"""
pipeline/sharing/link.py — connect this library to the user's Google account, so the Opyt
connector (Muse, Claude, ChatGPT) can read it.

It is one of the three things `share` is for (`share.py`, `to="connector"`). The service runs
Google's device sign-in (`service/accounts.py`), and this flow relays it. It takes two calls, like
GitHub's sign-in (`opyt_core/github_device.py`): the first shows a code to enter on Google's page,
and the same call after the user approves finishes it. The code lives on the
service, so a fresh process still finishes (Muse's helper starts one per tool call).

Connecting publishes the whole library to the service and keeps it published. On `linked` a push
is owed and the worker woken. From then on the publishing pass keeps the copy current
(`publishing.py`). Nothing is published before `linked`: an install that never shared
registers with the service to get a token, but that token stays aside (`account.hold_token`) and
the publishing pass cannot see it until Google approves.

Connecting also lets the connector change the library, through the line (`mcp_server/line.py`),
so `linked` wakes that job too: it starts holding the line at once instead of at its next hourly
ask.

`disconnect` is the other direction: it cuts the Google account off and leaves the served copy
and every reader as they were.

Every failure is a sentence in the return value, never a raise.
"""
from __future__ import annotations

from pipeline.sharing import account, client, desktop_only, push_gate
from pipeline.worker import job_queue
from pipeline.worker.registry import LINE, SCRAPERS

# Read to the user before the sign-in starts: what their yes covers.
CONSENT = (
    "The user's own claude.ai or ChatGPT, signed in with the Google account they approve, then "
    "reads this library and acts as them: it can search it, save links, remove items, follow "
    "topics, add people, and invite or cut off readers.",
    "It is standing access: it lasts until `unshare(reader='google')` disconnects it.",
    desktop_only.SENTENCE,
)

# The names `unshare` takes for the Google connection.
NAMES = ("google", "google account", "google-account")


def run() -> dict:
    """One step: `waiting` with a code for the user, `linked`, or `failed`. Registers this
    install with the service first if it never has."""
    acct = account.load()
    try:
        # No label: the connector reads the library as its owner, and no reader sees a name.
        token = client.held_token(acct)
        got = client.link(acct._replace(token=token))
    except client.Failed as e:
        return _failed(str(e))
    if got["status"] == "linked" and acct.token is None:
        account.confirm_token(acct, token)
    if got["status"] == "waiting":
        return {"status": "waiting", "open_this_url": got["verification_url"],
                "message": (
                    f"SAY, in one sentence: on the Google page, enter the code "
                    f"**{got['user_code']}** and approve Opyt. Give the page as a link whose text "
                    "reads **Connect Opyt to Google**, not a raw URL. Then WAIT until they say "
                    "they approved, and call `share(to='connector', confirm=True)` again to finish. Nothing "
                    "is connected until then.")}
    if got["status"] == "linked":
        push_gate.owe()
        push_gate.start_fast_hour()
        job_queue.request_now(SCRAPERS)
        job_queue.request_now(LINE)
        return {"status": "ok", "message": _LINKED}
    return _failed(got.get("message") or "Google did not confirm the account")


def _failed(why: str) -> dict:
    return {"status": "error",
            "message": (f"Connecting to Google did not complete ({why}). Nothing was connected. "
                        "Say so in one sentence; `share(to='connector', confirm=True)` starts again.")}


# The click paths are the website's (`website/index.html`'s claude.ai and chatgpt.com panes,
# `website/llms.txt`); change them together.
_LINKED = (
    "Connected. SAY this as written:\n\n"
    "\"Your library is connected to your Google account. A copy reaches Opyt's server within a "
    "few minutes and stays current while Opyt runs here. To read it:\n"
    "• **claude.ai**: your name at the bottom left → Settings → Connectors → + → Add custom "
    "connector. Paste https://api.useopyt.com/mcp, click Add, then Connect. On the Opyt page "
    "click Allow Access, then sign in with this same Google account. In a new chat, turn Opyt on "
    "from the + in the message box → Connectors.\n"
    "• **chatgpt.com** (paid plans): your profile picture → Settings → Apps & Connectors → "
    "Advanced, and switch Developer mode on. Click Create, name it Opyt, paste "
    "https://api.useopyt.com/mcp as the MCP server URL and choose OAuth. On the Opyt page click "
    "Allow Access, then sign in with this same Google account. In a new chat, turn Opyt on from "
    "the + menu.\n\n"
    f"{desktop_only.SENTENCE}\"")


def connected(listed: dict) -> str | None:
    """When this library was connected to a Google account, in a `GET /v1/tokens` answer; None
    when it is not."""
    return listed.get("google_linked_at")


def names(wanted: str) -> bool:
    """Whether `wanted` is how `unshare` is told to cut the Google connection."""
    return wanted.strip().lower() in NAMES


def roster_row(since: str) -> dict:
    """The Google connection as one row of who can reach the library, beside the readers."""
    return {"label": "the user's Google account (claude.ai, ChatGPT)", "id": NAMES[0],
            "since": since, "acts_as_user": True}


def disconnect(acct: account.Account, confirm: bool) -> dict:
    """Preview cutting the Google connection, or cut it. The served copy and every reader stay."""
    if not confirm:
        return {"status": "preview", "scope": "google_account", "service": acct.url,
                "consent": [
                    "This disconnects the user's Google account and nothing else: their "
                    "claude.ai and ChatGPT can no longer read this library or change it.",
                    "Everyone they shared it with keeps their access, and the copy on Opyt's "
                    "server stays up.",
                    "Connecting again is `share(to='connector')`, with a new sign-in on "
                    "Google's page."]}
    try:
        res = client.unlink(acct)
    except client.Failed as e:
        return {"status": "failed", "message": str(e)}
    if not res.get("unlinked"):
        return {"status": "already_gone",
                "message": "No Google account was connected, so nothing changed."}
    return {"status": "disconnected",
            "message": "The Google account is disconnected: claude.ai and ChatGPT can no longer "
                       "read this library or change it, from their very next request. Everyone "
                       "it was shared with keeps their access."}
