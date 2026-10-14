"""
pipeline/onboarding/handoff.py — the answer at the end of setup: which branch, and its copy.
"""
from __future__ import annotations

from enum import Enum

from pipeline.kb import schema
from pipeline.onboarding import roots, worker_check
from pipeline.oracles import ranking
from pipeline.sharing import account


# The last line of a finished setup, while this install has never reached the service: reading
# the library from claude.ai, ChatGPT or a phone goes through `share`, the one
# way in for all of them.
WEB_OFFER = ("\n\nLast, SAY this line as written: \"Want Opyt in claude.ai, ChatGPT or on "
             "your phone too? Say 'connect Opyt to Google'.\" If they say it, call "
             "`share(to='connector')`.")


class Branch(Enum):
    STILL_SCORING = "still_scoring"
    CHOOSE = "choose"
    FOUND_NOBODY = "found_nobody"
    NO_DISCOVERY = "no_discovery"
    BLIND_STORE = "blind_store"


def branch(*, phase: str, candidates: int, collectors: bool, has_oracles: bool,
           atoms: int | None) -> Branch:
    """Which handoff this home gets. An unreadable atom count (None) routes to the generic
    branch: a broken read must never be what claims the user's store is empty."""
    if phase == "curation":
        return Branch.STILL_SCORING
    if candidates:
        return Branch.CHOOSE
    if collectors:
        return Branch.FOUND_NOBODY
    if has_oracles or atoms != 0:
        return Branch.NO_DISCOVERY
    return Branch.BLIND_STORE


def answer(state: dict, extra: dict) -> dict:
    """What landed, and what to call next, branched on which root the user took and whether the
    store holds any material. `extra` is what `kickoff.start_work` woke, carried on the answer.
    """
    n = _candidate_count()
    done = state["phase"] == "done"
    b = branch(phase=state["phase"], candidates=n, collectors=state["curation"]["applicable"],
               has_oracles=bool(state["sources"].get("oracles")), atoms=_atom_count())

    if b is Branch.STILL_SCORING:
        # The ordinary state right after a platform connects; "a few minutes" is promised only
        # when the worker is running.
        worker = worker_check.cached_report()
        wait = ("call `onboard` again in a few minutes and it says who is ready."
                if worker.get("ok")
                else "they will be read " + worker_check.until_running(worker))
        message = (
            "Every setup question is answered. The people this user already follows are not "
            f"ready to choose from yet: do not call `screen` yet; {wait} Meanwhile these all "
            "work now:\n" + _still_open(state))
    elif b is Branch.CHOOSE:
        message = (
            # Plain phrasing: "screened" reads to the user as a process they are behind on.
            f"Setup is complete. {n} people are ready for the user to choose from.\n\n"
            f"Next: call `oracle` to pick who to trust. Confirming someone returns at once: "
            f"their posts are pulled in the background, and `oracle(action='progress')` shows "
            f"what has landed so far. Starting with a few people means each one's history fills "
            f"in sooner; more can be added any time."
            f"\n\nStill open, and they stay open:\n" + _still_open(state))
    elif b is Branch.FOUND_NOBODY:
        reconnect = " or ".join(f"`onboard(source='{s}')`"
                                for s in roots.connected(state["sources"]))
        message = (
            f"Setup is complete, but the collectors found nobody to screen. That usually means "
            f"the connected session went stale — reconnect with {reconnect} — or that this "
            f"account follows and subscribes to nobody yet.\n\nEither way, these all work "
            f"now:\n" + _still_open(state))
    elif b is Branch.NO_DISCOVERY:
        message = (
            "Setup is complete. Nothing is collected automatically, because no source with "
            "automatic discovery is connected — that is the expected state for a blog, "
            "newsletter or research user, not a failure.\n\n"
            "Every way to give OPYT something to read stays available:\n" + _still_open(state))
    else:
        # The blind store: no signal at all, so ask the user directly (topics first), and the
        # finding is the host's job.
        logins = _login_roots_open(state)
        opener = (
            "Setup is complete, and the topics the user follows are watching for new papers "
            "and repos — but the store itself holds no material yet: nothing to search, nothing "
            "to read."
            if state["sources"].get("topics") else
            "Setup is complete, but OPYT has nothing to read and no signal to guess from — "
            "no session, nobody named, nothing watched.")
        message = (
            opener + "\n\n"
            "So ask the user directly. ONE question, two ways to answer it:\n"
            "• **What topics are you into?** — lead with this; it is the easier question to "
            "answer. Then YOU fill the store. " + roots.FILL_FROM_INTERESTS + " The user "
            "should have material to explore before anything else gets set up.\n"
            "• **Or: who do you already read?** — names, in plain words. YOU web-search for "
            "each person's site, X handle, ORCID or OpenAlex page and pass the URL you "
            "find. Never ask the user for a link or an identifier — the finding is your "
            "job, not theirs. A user with no names is not stuck: the topics route above "
            "does the finding, and its survey's recurring authors are the people to bring "
            "back as candidates to confirm."
            + ("\n\nThe login roots stay open too:\n" + "\n".join(logins) if logins else ""))
    # Not on the blind store, whose answer is already a question for the user.
    if b is not Branch.BLIND_STORE and account.load().token is None:
        message += WEB_OFFER

    return {
        "status": "ok" if done else "in_progress",
        "phase": state["phase"],
        "candidates": n,
        "next_tool": "oracle",
        "message": message,
        **extra,
    }


def _still_open(state: dict) -> str:
    """Every way into OPYT this user has not connected, as a bulleted block.

    Re-offers the other roots because `derive()` never raises them again once one collector is
    connected. Only login roots can be reported as taken; the naming paths are always listed.
    """
    lines = _login_roots_open(state)
    lines += [
        "• **Topics they are into** — the user names interests and YOU fill the store. "
        + roots.FILL_FROM_INTERESTS,
        "• **Anyone, by name** — the user gives a name in plain words; YOU web-search for "
        "their site, X handle, ORCID or OpenAlex page and pass what you find to "
        "`oracle(action='confirm', add_handles=['https://…'])`. Never ask the user for a "
        "URL or an id.",
    ]
    return "\n".join(lines)


def _login_roots_open(state: dict) -> list[str]:
    """The login roots this home has not connected, as bullets. The blind-store handoff uses it
    alone, because there the naming paths are the question being put."""
    on = roots.connected(state["sources"])
    return [f"• **{s.title()}** — `onboard(source='{s}')`, which finds {roots.ROOT_DISCOVERS[s]}, "
            f"and imports {roots.ROOT_IMPORTS[s]}."
            for s in roots.LOGIN_URLS if s not in on]


def _candidate_count() -> int:
    """How many people the free pull surfaced for screening. Fail-safe: unreadable reads as 0,
    which understates progress rather than inventing it."""
    try:
        conn = schema.connect()
        try:
            return ranking.waiting_count(conn)
        finally:
            conn.close()
    except Exception:
        return 0


def _atom_count() -> int | None:
    """How many atoms the store holds, or None when it cannot be read (see `branch`)."""
    try:
        conn = schema.connect()
        try:
            return schema.count_atoms(conn)
        finally:
            conn.close()
    except Exception:
        return None
