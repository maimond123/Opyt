"""
pipeline/onboarding/roots.py — the four ways into OPYT: each root's word, its sign-in URL, and
what `onboard` says about it before anything is opened.
"""
from __future__ import annotations

from pipeline.platforms.browser import cookies as bc

# Only X and Substack have an account to connect. Research and a personal site are named by the
# user, and `onboard` points at the tool that owns each list (`oracle`) rather than proxying it,
# so that tool stays the only writer. GitHub is not a root: stars do not concentrate on people.


# The roots with a login, and where the user signs in. One dict, so the accepted words and the
# URLs cannot disagree. Substack has no Google SSO, which is what lets it sign in on a
# CDP-launched profile.
LOGIN_URLS = {"x": "https://x.com/login",
              "substack": "https://substack.com/sign-in"}


# What each login root discovers, in one clause of copy. `LOGIN_URLS` stays the one table that
# says a root exists.
ROOT_DISCOVERS = {"x": "people from your Lists, following and likes",
                  "substack": "the writers you follow and the newsletters you subscribe to"}


# What connecting each login root imports, in one clause of copy; `menu` says it before connecting.
ROOT_IMPORTS = {"x": "your bookmarks",
                "substack": "the posts you saved there"}


# How the host fills the store from what a user says they are into. Written once and read
# wherever that question is put — the research root's subject side and both ends of the handoff —
# so the three cannot drift apart. Three separate copies of "web-search the landscape" let a
# researcher's onboarding run four web searches and never ask OpenAlex.
FILL_FROM_INTERESTS = (
    "Pull the one to three NAMES out of what they said — the compound, scaffold, target, model, "
    "dataset, benchmark or program they are really asking about — and call `look` with them: "
    "`look(names=[…], sources=['openalex', 'arxiv'], days=1825)`, adding `'github'` to the "
    "sources when a name is a tool, model or dataset. That searches the papers themselves, with "
    "their abstracts, rather than the web; past 90 days OpenAlex answers with the most-cited "
    "work, which is what a landscape needs. Do NOT ask the user how far back to look: five years "
    "is the window, and you tell them afterwards in one line that you can go further back. SHOW "
    "them the strong finds, each with its `cite` link, and save the ones they pick (`hopper`) — "
    "nothing enters the store without their say-so. Recurring authors are people to offer: "
    "confirm the accepted ones with `oracle(action='confirm', add_handles=['https://…'])`. "
    "Web-search the landscape yourself only for a subject you can pull no name out of. "
    "Following a topic is NOT an onboarding move: it comes later, once there is material the "
    "user has actually read.")


# The roots the user names: no login, so answering one returns the question to put and the call
# to make. Kept apart from `LOGIN_URLS` so that table stays total on its own keys; `SOURCE_WORDS`
# derives from both.
NAMED_ROOTS = {
    "research": (
        "Ask which — a person, or a subject. They produce different things:\n"
        "• **A researcher** — `oracle(action='confirm', add_handles=['https://orcid.org/0000-…'])`. "
        "An OpenAlex author page (https://openalex.org/A…) works too, and is the one to reach for "
        "when they have no ORCID. OPYT resolves either to their publication "
        "record and pulls the papers as atoms; abstracts are free, so the whole back catalogue "
        "is the default window. An academic URL OPYT cannot pull work by (Semantic Scholar, "
        "Google Scholar, DBLP, ResearchGate) is refused by name, with the shape that would have "
        "worked.\n"
        "  ⚠️ A bare NAME does not resolve and must not be passed — OpenAlex returns 16 people "
        "for \"Frances Arnold\". But finding the URL is YOUR job, not the user's: "
        "take the name they give, web-search \"<name> ORCID\" or \"<name> openalex\" yourself, "
        "and pass the page you find. If the name is ambiguous, ask the user to pick between the "
        "real candidates (\"the Caltech chemist?\") — never send them off to fetch an "
        "identifier.\n"
        "• **A subject** — interests, not commitments (NO onboarding "
        "surface sets up a topic to follow, this root included — a user typing what they are "
        "into is not asking for one, whatever word they clicked to get here). Fill the store "
        "instead. " + FILL_FROM_INTERESTS + " You may mention ONCE that OPYT can also follow a "
        "subject for them later, after they have read some of what is now in the store — and "
        "leave it at that."),
    "blog": (
        "Ask which site — a blog, a newsletter, or a personal site — then "
        "`oracle(action='confirm', add_handles=['https://…'])` with the URL. Any http home "
        "works with no per-host setup: the adapter tries sitemap, then RSS/Atom, then a hub "
        "crawl.\n"
        "⚠️ A PLATFORM profile is not a personal site. A GitHub, LinkedIn or Medium-author URL "
        "is refused, and the refusal names the shape that would work instead — usually the "
        "person's X @handle, which `add_handles` also takes bare.\n"
        "A NAME works here too: if the user names a person rather than a site, web-search for "
        "that person's blog or newsletter yourself and confirm the URL you find — do not send "
        "them off to fetch it."),
}


# ...plus `skip`, which opens nothing and records that the question was put.
SOURCE_WORDS = tuple(LOGIN_URLS) + tuple(NAMED_ROOTS) + ("skip",)


def connected(sources: dict) -> list[str]:
    """The login roots with a session connected to OPYT, in `LOGIN_URLS` order. `sources` maps a
    root to whether its session is there."""
    return [s for s in LOGIN_URLS if sources.get(s)]


def menu() -> dict:
    """Nothing in this home will produce atoms or candidates yet. Offer every way to change that.

    Asks for one root: two of them send the user out of the conversation to sign in, so a
    multi-pick is a queue the host would have to carry across that round trip.
    """
    resume = ""
    if bc.opyt_session_backends():
        # A profile on disk does not prove a window is open, so this says both things that might
        # be true.
        resume = ("\n\n⚠️ OPYT's login profile already exists but holds no session it can use. "
                  "If the user has just logged in, call `onboard` again — it will pick the "
                  "session up. Otherwise pick a source below.")

    return {
        "status": "needs_source", "phase": "sources",
        "answers": list(SOURCE_WORDS),
        "message": (
            "OPYT has nothing to read yet. Ask the user which ONE of these to start with, and "
            "pass that single word as `source=`.\n\n"
            "⚠️ ASK FOR ONE, not several. Two of the roots send the user out of the "
            "conversation to sign in, so a multi-pick becomes a queue you have to carry across "
            "that — and a root you forget to come back to is one the user was promised. Nothing "
            "is lost by taking one: no root is ever closed, and when setup finishes OPYT names "
            "every root this home has not connected.\n\n"
            "The first two discover people for the user; the last two are for naming what they "
            "already know.\n\n"
            "1. **X** — `onboard(source='x')`. OPYT opens its own browser profile at x.com; the "
            "user logs in there, and their own windows and logins are untouched. Their Lists, "
            "following and likes become candidates to screen without anyone typing a name, and "
            "the posts they bookmarked there are imported.\n"
            # The desktop flow reconnects on a phone as X's does; no steer-to-computer copy here.
            "2. **Substack** — `onboard(source='substack')`. Same shape: OPYT opens its own "
            "browser profile at Substack's sign-in; the user logs in there (Substack emails "
            "them a code), and the writers they follow and the newsletters they subscribe to "
            "become candidates, and the posts they saved there are imported.\n"
            "3. **Research** — `onboard(source='research')`. A researcher's whole publication "
            "record, or a subject to build the store around. OPYT asks which and hands you "
            "the call.\n"
            "4. **A blog, newsletter or personal site** — `onboard(source='blog')`. Any http "
            "home, no per-host setup.\n"
            "5. **Nothing yet** — `onboard(source='skip')`. Setup continues and this question is "
            "not asked again; any of the four above still works later." + resume),
    }


def name_question(source: str) -> dict:
    """A root the user names → the question to put and the call to make with the answer.

    Opens, writes and records nothing. Status is `needs_name`, not `needs_source`: the sources
    question is answered, and only the identifier is outstanding.
    """
    return {"status": "needs_name", "phase": "sources", "source": source,
            "message": NAMED_ROOTS[source]}


def refuse(source: str) -> dict:
    """An unknown `source` word. Refuse, never guess: guessing here would launch a browser window
    at a site the user did not name."""
    return {"status": "error", "phase": "sources",
            "message": (f"`source={source!r}` is not one of "
                        f"{' | '.join(SOURCE_WORDS)}. Nothing was opened.")}
