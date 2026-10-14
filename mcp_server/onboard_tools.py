"""
mcp_server/onboard_tools.py — `onboard`: ready the machine, then hand off to `oracle`.

The entry point runs onboarding's flow (`pipeline/onboarding/flow.py`).

No parameter of this tool may carry a credential: it would land in the transcript. Secrets travel
over GitHub's device sign-in or a one-time key page on the user's own machine
(`pipeline/onboarding/optional_keys.py`). An S2 author page is not a pullable
root; the academic roots are ORCID and OpenAlex.
"""
from __future__ import annotations

from pipeline.onboarding import flow


def register_onboard_tools(mcp) -> None:

    @mcp.tool(annotations={"title": "Set up Opyt", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True})
    def onboard(source: str | None = None, start: str | None = None) -> dict:
        """**Set up OPYT on this machine.** Call this first on a fresh install, and any time
        setup looks incomplete. Idempotent and re-entrant — it recomputes where you are from
        disk on every call, so calling it twice never repeats a finished step.

        It runs in a few calls, because every step that waits on a human ends one. The step is
        **a source** — where the user already reads. FOUR roots, and X is one answer rather than
        the question: Substack is a peer, `research` takes a researcher or a subject, `blog`
        takes any personal site. They are not exclusive — take one, then call again for another
        — or `skip` and look around first.

        There is nothing else to ask. Connecting X or Substack also imports the posts the user
        saved there, and OPYT keeps everything current in the background.

        Arguments are decisions, never credentials:
          • `source` — one of `x` | `substack` | `research` | `blog` | `skip`, and the roots
            are not exclusive: call again to take another.
              - `x` / `substack` CONNECT that platform. Locally OPYT opens its own persistent
                browser profile at the site's sign-in page; off macOS, `x` returns a short-lived
                link to a private sign-in desktop on Opyt's service instead. This is the ONLY way
                OPYT reads X — no API key, no third party. Works at any point, so an expired or
                wrong-account session can be replaced without redoing setup.
              - `research` / `blog` have no account to connect. They return the question to put
                to the user and the exact `oracle` call to make with the answer.
              - `skip` records the question was put and moves on.
          • `start` — each word sends the user out of the conversation to a browser page, so
            never call it blind: say what the page is for, ask if they are ready, then call it.
            Every word works at any time.
              - `openalex` / `github` — a word the `optional_keys` question named (it rides
                the answer to the user's first Oracle confirmation), or one the user asks for.
                `github` shows a code to enter on github.com; call `start='github'` again once
                they approve, to finish. `openalex` returns a one-time page on the user's own
                machine as a link, for them to paste their key into; call `start='openalex'`
                again once they say they saved it, to check and keep it. The key never enters
                the chat.

        IF THIS CALL RETURNS AN ERROR INSTEAD OF A RESULT — a timeout, a dropped connection —
        the decision in it was already written before the reply was lost, and setup is re-entrant
        by design. So: call `onboard()` with NO arguments to re-read where you are from disk, and
        continue from what it says. Never re-send the same `source`/`start` argument
        blind, and never tell the user their "request timed out" or "failed" — they made a choice,
        not a request, and the choice stands. If you must say anything, say OPYT is still setting
        up in the background.

        Then call `oracle` to choose who to trust.
        """
        return flow.onboard(source, start)
