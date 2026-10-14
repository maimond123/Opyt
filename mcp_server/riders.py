"""
mcp_server/riders.py — the one attachment flow: after a tool answers, decide its riders, take each
from the unit that owns it, and place each on the answer.

OPYT cannot push: an MCP server speaks when it is called and never otherwise, so what one unit
needs to say (an Oracle pull landed, a model went dead, something new arrived)
rides back on whatever the user calls next. Each owner keeps the job that produces its notice;
this module only asks, at the one seam every answer passes through (`rider_hook.py`).

Every rider is a courtesy on somebody else's answer and must never be able to break it: a rider
that fails contributes nothing, and that rule lives once, in `attach`.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import closing

from mcp_server import offers, rider_rules
from mcp_server.rider_rules import Call
from pipeline import model_routing
from pipeline.arrivals import notice as arrivals
from pipeline.curation import imports, still_scoring
from pipeline.kb import schema
from pipeline.onboarding import optional_keys
from pipeline.retrieval import new_material
from pipeline.sharing import page_words, share_back
from pipeline.oracles import freshness
from pipeline.oracles import milestones
from pipeline.oracles import told
from pipeline.timeparse import utc_now


def _pull_finished(_call: Call, _answer: dict) -> dict | None:
    """oracles: milestones reached while nobody was looking. Stamps as it hands over, so it is
    taken once per call and only for an eligible answer."""
    with closing(schema.connect()) as conn:
        return told.completion_notice(conn)


def _new_material(_call: Call, _answer: dict) -> dict | None:
    """retrieval: material in the store the user has never been told about. Stamps as it hands
    over, like completion."""
    with closing(schema.connect()) as conn:
        return new_material.new_material_notice(
            conn, tour=lambda: offers.tour(conn), importing=imports.in_progress(conn))


def _offer(call: Call, answer: dict) -> dict | None:
    """the server: one thing the reader's own store can do for them, unless the answer already
    asks something. Stamps what it hands over."""
    if rider_rules.asks(answer):
        return None
    with closing(schema.connect()) as conn:
        return offers.offer(conn, call, answer)


def _oracles_stale(_call: Call, _answer: dict) -> dict | None:
    """oracles: the reader's Oracles have fallen behind, so these results may miss what is new."""
    with closing(schema.connect()) as conn:
        return freshness.stale_notice(conn)


def _latest(_call: Call, _answer: dict) -> dict | None:
    """arrivals: how many atoms arrived since the user last looked. Marks nothing seen."""
    with closing(schema.connect()) as conn:
        return arrivals.notice(conn, utc_now())


def _optional_keys(_call: Call, answer: dict) -> dict | None:
    """onboarding: the optional-keys question, on the first confirmation whose picks link code or
    papers. Records that it was put as it hands it over, so it is said once."""
    with closing(schema.connect()) as conn:
        return optional_keys.question(conn, rider_rules.picked(answer))


def _coverage(call: Call, answer: dict) -> dict:
    """What a thin search means, given what this home was promised and has not received: the
    milestones' whole list — each pick's Oracle pulls and the saved-post imports."""
    with closing(schema.connect()) as conn:
        owed = milestones.outstanding(conn)
    return rider_rules.coverage_notice(answer["trace"]["ranked"], call.args["k"], owed)


# Every rider and the job that produces it. A producer is looked up through its owner's module at
# call time, never bound at import, so the owner's module stays the one place it is defined.
PRODUCERS: dict[str, Callable[[Call, dict], object]] = {
    # oracles: Oracles that have fallen behind, on the reader's own search.
    "oracles_stale": _oracles_stale,
    # oracles: an Oracle pull that reached a milestone while nobody was looking.
    "pull_finished": _pull_finished,
    # the server: one offer, measured against the reader's own store. Placed last.
    "offer": _offer,
    # retrieval: material in the store nobody has told the user about.
    "new_material": _new_material,
    # model-access: a registered model that is dead or one withdrawal from it.
    "model_routing": lambda _call, _answer: model_routing.routing_notice(),
    # curation: the roster is still being scored while collectors read the user.
    "still_scoring": lambda _call, _answer: still_scoring.still_scoring_notice(),
    # arrivals: atoms that arrived since the reader last looked. `None` when quiet.
    "latest": _latest,
    # sharing: how to make something readable of an answer, on every own-store search and open.
    "readable": lambda _call, _answer: page_words.READABLE,
    # sharing: the offer to share back, once per peer.
    "reciprocal_offer": lambda _call, answer: share_back.offer(answer["trace"]["kb"]),
    # oracles + curation: a thin search, unless a pull or an import is still owed.
    "coverage": _coverage,
    # onboarding: GitHub and OpenAlex, asked once, on the first confirmation they would help.
    "optional_keys": _optional_keys,
}


def attach(call: Call, answer: dict) -> dict:
    """`answer` with every rider it is eligible for, in order.

    All eligibility is decided before the first producer is asked, so an ineligible answer never
    consumes a notice. A rider that fails, producing or placing, contributes nothing. Every call
    first stamps the feature it used, eligible or not, so an offer never pitches what the user
    already does without a strong fit."""
    offers.note_use(call)
    for rider in rider_rules.eligible(call, answer):
        try:
            notice = PRODUCERS[rider](call, answer)
            if notice is None:
                continue
            answer = rider_rules.place(answer, rider, notice, call.tool)
        except Exception:
            continue
    return answer
