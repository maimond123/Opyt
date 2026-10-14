"""
pipeline/onboarding/optional_keys.py — the optional keys: GitHub and OpenAlex.

None of them is needed for anything to work; each lifts a per-IP anonymous limit (GitHub 60
requests an hour, OpenAlex a daily budget) that pulling the code and papers people link spends.
The question is put once, on the answer to the user's first Oracle confirmation whose picks the
store already shows sharing code or citing papers, worded from them (it rides that answer through the server's rider flow, `mcp_server/riders.py`). Each
`start` word works at any time. GitHub is asked for only once its sign-in exists
(`github_device.CLIENT_ID`). Semantic Scholar is not part of this: AI2 does not approve
third-party key requests, so it is never mentioned; a
user who has one sets it with `opyt-keys`.

  start='github'    GitHub's device sign-in: a code to enter on github.com, then the same word
                    again to collect the token (`opyt_core.github_device`).
  start='openalex'  a one-time key page on this machine (`opyt_core.key_page`), handed back as
                    a link; the same word again collects the key and checks it against OpenAlex
                    before it is stored, because a key it does not know fails every later call.
"""
from __future__ import annotations

import sqlite3

import requests

from opyt_core import github_device, key_page, keys
from opyt_core.credentials_registry import by_service
from opyt_core.paths import opyt_path
from pipeline.credentials import get_credential
from pipeline.kb import schema

STARTS = ("github", "openalex")
# The keys the question offers, in the order it names them.
OFFERED = ("github", "openalex")

# Records that the question was put; `question` is its only writer.
KEYS_OFFERED_MARKER = "onboard_keys_offered"

# What the user is told each service is called.
_NAMES = {"github": "GitHub", "openalex": "OpenAlex"}

# The kind of linked work each key brings in faster: an atom's `source_type`, as the link router
# classified the url when it was stored (`kb.link_router.classify_link`).
_KIND = {"github": "github", "openalex": "paper"}

# What a pick does, said of one person and of several.
_DOES = {"github": ("shares code on GitHub", "share code on GitHub"),
         "openalex": ("cites research papers", "cite research papers")}

# The offer, by which keys are asked for.
_ASK = {
    ("github", "openalex"): ("Connect GitHub and add a free OpenAlex key so those come in faster; "
                             "without them everything still arrives, just more slowly. Want "
                             "either?"),
    ("github",): ("Connect GitHub so their code comes in faster; without it everything still "
                  "arrives, just more slowly. Want to connect it?"),
    ("openalex",): ("Add a free OpenAlex key so those papers come in faster; without it "
                    "everything still arrives, just more slowly. Want to add it?"),
}

# The one read that tells a working key from a wrong one: (url, header name, header value format,
# the statuses that mean the service knows the key). OpenAlex answers 401 to a key it does not
# know.
_PROBES = {
    "openalex": ("https://api.openalex.org/rate-limit", "Authorization", "Bearer {}", {200}),
}
_PROBE_TIMEOUT = 15.0


def wanted() -> tuple[str, ...]:
    """The offered keys worth asking for here, in `OFFERED` order: not stored yet, and GitHub only
    once its sign-in exists."""
    return tuple(s for s in OFFERED
                 if not get_credential(s) and (s != "github" or github_device.CLIENT_ID))


def question(conn: sqlite3.Connection, picked: list[str]) -> dict | None:
    """The question, put once, for the keys still missing that `picked` (the Oracles just
    confirmed) measure above zero for. None when it was put already, nothing is missing, or no
    pick links the work a missing key is for; then nothing is written, so a later confirmation
    can still ask."""
    if opyt_path(KEYS_OFFERED_MARKER).exists() or not (want := wanted()):
        return None
    heads = list(dict.fromkeys(schema.current_canonical(conn, c) for c in picked))
    counts = {key: n for key in want if (n := _people_linking(conn, heads, _KIND[key]))}
    if not counts:
        return None
    _put_marker()
    routes = " ".join(f"{_NAMES[k]} → `onboard(start='{k}')`." for k in counts)
    return {
        "answers": list(counts),
        "message": (
            "After what the rest of this answer says, SAY, word for word, and explain nothing "
            f"more about it: \"{line(len(heads), counts)}\"\n\n"
            f"Then WAIT for the answer. {routes} "
            f"{'One at a time: each sends' if len(counts) > 1 else 'It sends'} them to a browser "
            "page. Not now, or anything else → carry on; this is asked once, and the word works "
            "any time later."),
    }


def line(people: int, counts: dict[str, int]) -> str:
    """What the user hears: how many of the `people` just picked link the work each key in
    `counts` is for, then the offer."""
    clauses = []
    for i, (key, n) in enumerate(counts.items()):
        does = _DOES[key][0 if n == 1 else 1]
        if people == 1:
            clauses.append(f"The person you just picked {does}" if i == 0 else does)
        else:
            clauses.append(f"{n} of the people you just picked {does}" if i == 0
                           else f"{n} {does}")
    said = (" and " if people == 1 else ", and ").join(clauses)
    return f"{said}. {_ASK[tuple(counts)]}"


def _people_linking(conn: sqlite3.Connection, heads: list[str], kind: str) -> int:
    """How many of these Oracles the store shows with work of `kind`: an atom they authored, or
    one a post of theirs linked (`payload.referenced_by`). Read from what is stored, so it holds
    at confirm time, before anything of theirs is pulled: the bookmark import has already stored
    the repos and papers the user's saved posts link."""
    found = 0
    for head in heads:
        members = [r["entity_id"] for r in schema.entities_for_canonical(conn, head)] or [head]
        marks = ",".join("?" * len(members))
        found += conn.execute(
            f"SELECT 1 FROM atoms a WHERE a.source_type = ? AND (a.who_id IN ({marks}) OR "
            f"EXISTS (SELECT 1 FROM json_each(a.payload, '$.referenced_by') r "
            f"WHERE json_extract(r.value, '$.by') IN ({marks}))) LIMIT 1",
            (kind, *members, *members)).fetchone() is not None
    return found


def _put_marker() -> None:
    """Record that the question was put. Best-effort: a home that cannot be written asks again on
    a later confirmation, which is the harmless direction."""
    try:
        p = opyt_path(KEYS_OFFERED_MARKER)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def run(start: str, phase: str) -> dict:
    """One `start` word's step."""
    if start == "github":
        return _github(phase)
    return _key_page(start, phase)


def _github(phase: str) -> dict:
    got = github_device.finish() if github_device.pending() else github_device.begin()
    status = got["status"]
    if status == "stored":
        return {"status": "ok", "phase": phase, "step": "github",
                "message": "GitHub is connected. Say so in one short sentence."}
    if status == "unavailable":
        return {"status": "error", "phase": phase, "step": "github",
                "message": ("GitHub sign-in is not available in this version of Opyt yet. Say "
                            "so in one sentence. Nothing was changed.")}
    if status == "failed":
        return {"status": "error", "phase": phase, "step": "github",
                "message": (f"The GitHub sign-in did not complete ({got.get('message')}). "
                            "Nothing was stored. `onboard(start='github')` starts a fresh one.")}
    return {"status": "waiting", "phase": phase, "step": "github",
            "open_this_url": got["open_this_url"],
            "message": (
                f"SAY, in one sentence: on the GitHub page, enter the code **{got['user_code']}** "
                "and approve Opyt. Give the page as a link whose text reads **Connect GitHub to "
                "Opyt**, not a raw URL. Then WAIT until they say they approved, and call "
                "`onboard(start='github')` again to finish. Nothing is stored until then.")}


def _key_page(service: str, phase: str) -> dict:
    cred, name = by_service(service), _NAMES[service]
    key = key_page.take()
    if key is None:
        again = key_page.waiting()
        url = again or key_page.begin(name=name, signup_url=cred.signup_url)
        said = ("The page has not received a key yet. SAY so in one sentence, and give the same "
                if again else
                f"SAY, in one sentence: on the page, paste their {name} key and press Save (the "
                "page links to where to get a free one). Give ")
        return {"status": "waiting", "phase": phase, "step": service, "open_this_url": url,
                "message": (f"{said}the page as a link whose text reads **Add your {name} key "
                            f"to Opyt**, not a raw URL. Then WAIT until they say they saved it, "
                            f"and call `onboard(start='{service}')` again to finish. Nothing is "
                            "stored until then.")}
    verdict = _probe(service, key)
    if verdict != "ok":
        why = ("did not recognise that key" if verdict == "rejected"
               else "could not be reached to check the key")
        return {"status": "error", "phase": phase, "step": service,
                "message": (f"{name} {why}, so it was NOT stored. Say so in one sentence; "
                            f"`onboard(start='{service}')` gives a fresh page.")}
    keys.set_key(cred.env, key)
    return {"status": "ok", "phase": phase, "step": service,
            "message": f"The {name} key is stored on this machine. Say so in one short sentence."}


def _probe(service: str, key: str) -> str:
    """`ok`, `rejected` (the service does not know this key) or `unreachable`."""
    url, header, fmt, known = _PROBES[service]
    try:
        r = requests.get(url, headers={header: fmt.format(key)}, timeout=_PROBE_TIMEOUT)
    except requests.RequestException:
        return "unreachable"
    if r.status_code in known:
        return "ok"
    return "rejected" if r.status_code in (401, 403) else "unreachable"
