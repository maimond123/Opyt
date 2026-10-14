"""
mcp_server/rider_rules.py — which riders a tool call is eligible for, where each lands in that
tool's answer, and the sentence the coverage rider says. Decisions only: plain values in, plain
values out, no reads.

A rider is a notice that rides on an answer it did not come from: another unit's (completion,
the provider notice, what arrived, the share-back offer, reader view's standing line,
onboarding's optional-keys question) or
the server's own (a thin store, and the one offer an answer may carry, `offers.py`).
Who produces each one is `riders.py`'s table; this module decides, per call, which to ask for and
how to place what comes back. Every refusal is decided here, before any producer is asked, because
some producers consume what they hand over.

The server keeps no conversation memory, so eligibility is a function of this call and its
answer alone; a notice that should be said once per conversation tells the host so.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Call:
    """One tool call: its name and its arguments, bound to the tool's own declared defaults, so
    `oracle()` reads as `action="screen"` without this module restating that default."""

    tool: str
    args: dict


# The arrivals notice rides every own-store search until the user acts on it,
# because the server cannot tell a conversation's first search from its tenth. Repeated on every
# call, a notice trains the reader to skip it, so the host is told to say it once and recognize
# the repeats.
ONCE_PER_CONVERSATION = (
    "This rides every search on the user's own store until it is acted on, because the server "
    "keeps no memory of the conversation. Tell the user about it once in this conversation, in "
    "one line, on the first answer that carries it; after that, leave it unless they ask.")

# The `search` arguments that never narrow it. Every other argument is a filter, so a filter
# added to `search` is one here without being listed twice. A narrowed search is thin on purpose,
# so it is never called thin coverage.
NOT_FILTERS = frozenset({"query", "k", "mode", "kb"})


# Notices and keys that already ask the user something. An answer carrying one carries no offer:
# one ask per answer. The new-material notice counts because its tour carries the offer.
# `latest` does not: it is a count to mention once, and it rides nearly every own-store search
# (anything an Oracle posted since the last look), so counting it would silence every offer.
ASKING_CODES = frozenset({"thin_coverage", "import_incomplete", "pull_finished"})
ASKING_KEYS = frozenset({"pull_finished", "new_material", "optional_keys"})


def eligible(call: Call, answer: dict) -> list[str]:
    """The riders this call's answer is eligible for, in the order they are placed. `offer` is
    always last, so it sees whether an earlier rider already asks something (`asks`).

    An offer rides only an answer that worked and holds something and never on a call that
    is mid-setup, mid-decision or irreversible: confirm, ingest, review, add_oracle, forget,
    share, unshare and accept carry none. Onboard's handoff carries one only inside the
    new-material tour. A confirmation of Oracles (`oracle` confirm, `add_oracle` confirm) is
    where onboarding's optional-keys question rides, once."""
    match call.tool:
        case "search":
            return _search(call, answer)
        case "aggregate":
            # The offer only on the reader's own store: it names atoms the reader must hold.
            # Completion rides any aggregate.
            return (["pull_finished", "offer"] if _own(call) and answer.get("total")
                    else ["pull_finished"])
        case "oracle":
            return _oracle(call, answer)
        case "onboard":
            # Only the handoff (the answer naming what to call next) carries what landed.
            return (["pull_finished", "new_material"]
                    if answer.get("next_tool") == "oracle" else [])
        case "open":
            return ["readable", "offer"] if _own(call) and "error" not in answer else []
        case "hopper":
            saved = answer.get("status") in ("saved", "already_present")
            return ["offer"] if call.args.get("confirm") and saved else []
        case "add_oracle":
            return ["optional_keys"] if call.args.get("confirm") and "added" in answer else []
        case _:
            return []


def asks(answer: dict) -> bool:
    """The answer already asks the user something, so it carries no offer."""
    return bool(ASKING_KEYS & answer.keys()) or any(
        n.get("code") in ASKING_CODES for n in answer.get("notices", []))


def _own(call: Call) -> bool:
    """The call reads the reader's own store, not a peer's."""
    return call.args.get("kb") in (None, "me")


def _search(call: Call, answer: dict) -> list[str]:
    own = _own(call)
    riders = []
    # Oracle staleness is about the reader's own Oracles, so it rides only their own store's
    # answer — and not an empty store's, whose one notice is that there is nothing to search.
    if own and not any(n.get("code") == "store_empty" for n in answer.get("notices", [])):
        riders.append("oracles_stale")
    riders.append("pull_finished")
    # What arrived rides the reader's own answers; the share-back offer a peer's that found
    # something.
    if own:
        riders.append("latest")
    elif not own and answer.get("hits"):
        riders.append("reciprocal_offer")
    ranked = answer.get("trace", {}).get("ranked")
    narrowed = any(value for name, value in call.args.items() if name not in NOT_FILTERS)
    if own and not narrowed and isinstance(ranked, int) and ranked < call.args["k"]:
        riders.append("coverage")
    if own and answer.get("hits"):
        riders += ["readable", "offer"]
    return riders


def _oracle(call: Call, answer: dict) -> list[str]:
    """`screen` is the deliberate "what is my people situation" call, so it carries every rider
    that rides unconditionally. A candidates answer that lists people carries an offer."""
    match call.args.get("action"):
        case "screen":
            return ["model_routing", "still_scoring", "pull_finished", "new_material", "offer"]
        case "candidates":
            return ["offer"] if answer.get("candidates") else []
        case "confirm":
            # A venue's entry already asks which subjects to follow: one ask per answer.
            confirmed = answer.get("confirmed") or []
            asks_subjects = any(e.get("needs_topics") for e in confirmed)
            return ["optional_keys"] if confirmed and not asks_subjects else []
    return []


def picked(answer: dict) -> list[str]:
    """The Oracles a confirmation answer just confirmed: `oracle` confirm's `confirmed`, or
    `add_oracle`'s one `added`."""
    if "added" in answer:
        return [answer["added"]["canonical_id"]]
    return [e["canonical_id"] for e in answer.get("confirmed") or []]


def place(answer: dict, rider: str, notice, tool: str) -> dict:
    """`answer` with `notice` placed where `rider` lands on this tool's answer. A new dict: the
    answer handed in is never changed."""
    match rider, tool:
        case "oracles_stale", "search":
            # First among the notices.
            return {**answer, "notices": [notice, *answer.get("notices", [])]}
        case "pull_finished", ("search" | "aggregate"):
            return _with_notice(answer, {"code": "pull_finished", **notice})
        case ("reciprocal_offer" | "coverage"), "search":
            return _with_notice(answer, notice)
        case "latest", "search":
            return {**answer, rider: {**notice, "host_note": ONCE_PER_CONVERSATION}}
    if (rider, tool) in TOP_LEVEL:
        return {**answer, rider: notice}
    raise ValueError(f"no placement for rider {rider!r} on {tool!r}")


# Riders that land as a key of their own, named for the rider, on these tools' answers.
TOP_LEVEL = frozenset({
    *(("offer", tool) for tool in ("search", "aggregate", "oracle", "open", "hopper")),
    ("readable", "search"), ("readable", "open"),
    ("model_routing", "oracle"), ("still_scoring", "oracle"),
    ("pull_finished", "oracle"), ("new_material", "oracle"),
    ("pull_finished", "onboard"), ("new_material", "onboard"),
    ("optional_keys", "oracle"), ("optional_keys", "add_oracle"),
})


def _with_notice(answer: dict, notice: dict) -> dict:
    return {**answer, "notices": [*answer.get("notices", []), notice]}


# ── the server's own riders ──────────────────────────────────────────────────────

def coverage_notice(ranked: int, asked: int, waiting: list[str]) -> dict:
    """What a thin, unfiltered search on the reader's own store means: an enrichment offer, or,
    while an import is still owed (`waiting`), a warning that the store is not finished."""
    if waiting:
        return {
            "code": "import_incomplete", "ranked": ranked, "asked": asked, "waiting": waiting,
            "message": (
                f"This unfiltered search matched {ranked} atoms in the whole store — "
                f"fewer than the {asked} you asked for. Do NOT read that as the user being "
                f"thin on this subject: an import they were promised is still outstanding "
                f"({', '.join(waiting)}), so material that would answer this may simply "
                f"not be in the store yet. Say that plainly, answer from what is here, and "
                f"offer nothing else — an enrichment offer here asks them to go find "
                f"what they already gave OPYT.")}
    return {
        "code": "thin_coverage", "ranked": ranked, "asked": asked,
        "message": (
            f"This unfiltered search matched {ranked} atoms in the whole store — fewer "
            f"than the {asked} you asked for. The user has just named a subject their store "
            f"is thin on, which is the best enrichment signal there is. Offer, in one "
            f"line: to web-search the current landscape on it for them (when it is one named "
            f"thing, such as a model, drug or dataset, `look` searches X, Substack and the "
            f"paper and code sources for it now), and to follow it "
            f"as a topic, so they are told when something new on it appears. If they want "
            f"the topic, write its lists as `watch` describes, show them "
            f"watch(action='preview') and create it once they agree. Make this offer once "
            f"per conversation: if you already made it here, do not repeat it, and if they "
            f"declined, drop it.")}
