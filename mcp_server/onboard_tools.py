"""
mcp_server/onboard_tools.py — `onboard`: ready the machine, then hand off to `oracle`.

A thin orchestrator: it sequences existing screen/confirm/ingest code rather than
re-implementing it.

The channel rule: secrets travel over the local loopback callback or a hosted child callback;
decisions and short-lived approval links travel over chat as arguments/results. No parameter of
this tool may ever carry a credential — it would land in the transcript and every later turn's
re-send.

Semantic Scholar is not offered to the user, for two separate reasons that now point the same way.
The KEY is never advertised because AI2 no longer approves third-party key requests, so its row in
`opyt_core/credentials_registry.py` stays present but unadvertised, referred to by SERVICE name
only (this module does not load env) — sending a user to a form that will reject them is the harm.
And since 2026-09-09 an S2 author PAGE is not a root either: `oracles._SCHOLAR_ROOTS` marks it
unpullable, so naming one here would send the user to a URL `add_oracle` refuses. The academic
roots are an ORCID and an OpenAlex author page, and this module names only those.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from opyt_core import openrouter_oauth, readiness, trial
from pipeline.kb import onboard_state


def register_onboard_tools(mcp) -> None:

    @mcp.tool()
    def onboard(consent: str | None = None, source: str | None = None,
                start: str | None = None) -> dict:
        """**Set up OPYT on this machine.** Call this first on a fresh install, and any time
        setup looks incomplete. Idempotent and re-entrant — it recomputes where you are from
        disk on every call, so calling it twice never repeats a finished step.

        It runs in up to four calls, because every step that waits on a human ends one:
          1. **A model provider** — often ZERO calls. OPYT covers the model costs to start
             with and sets this up itself; on hosted OPYT that happens silently inside this
             call and the step never appears. Do not call it a trial, an allowance or free —
             see `_trial_prompt` for why that framing was removed from what users are told. Where it cannot (a local install has to ask
             who they are), it is TWO calls with a human turn between them: the first explains
             and does nothing, then `onboard(start='trial')` opens the sign-in. The same two-call
             shape covers `onboard(start='openrouter')`, which is what runs when there is no
             allowance to claim and again later when one is spent. Nothing gets pasted, ever.
          2. **A source** — where the user already reads. FOUR roots, and X is one answer
             rather than the question: Substack is a peer, `research` takes a researcher or a
             subject, `blog` takes any personal site. They are not exclusive — take one, then
             call again for another — or `skip` and look around first.
          3. **Consent** — one question, two commitments, and you may answer them separately.

        Arguments are decisions, never credentials:
          • `source` — one of `x` | `substack` | `research` | `blog` | `skip`, and the roots
            are not exclusive: call again to take another.
              - `x` / `substack` CONNECT that platform. Locally OPYT opens its own persistent
                browser profile at the site's sign-in page; hosted OPYT returns a short-lived
                link to a private sign-in desktop on the server instead. This is the ONLY way
                OPYT reads X — no API key, no third party. Works at any point, so an expired or
                wrong-account session can be replaced without redoing setup.
              - `research` / `blog` have no account to connect. They return the question to put
                to the user and the exact `oracle` call to make with the answer.
              - `skip` records the question was put and moves on.
          • `consent` — one of `both` | `backlog` | `refresh` | `none`.
          • `start` — `trial` or `openrouter`. Each BEGINS a step that sends the user out of
            the conversation, so neither may be called blind: call `onboard()` with no
            arguments first, say what it returns, ask if they are ready, and only then pass the
            word that reply named. Never choose between the two yourself during setup — the
            no-argument call already picked one, and picking the other overrides a decision
            made from disk.
            `start='openrouter'` ALSO works at any later time, and that is the one case you may
            reach for unprompted: a user who says they would rather use their own OpenRouter
            account. Say what the page is for, ask if they are ready, then call it. It replaces
            whatever key is in place, including a live starter allowance.

        IF THIS CALL RETURNS AN ERROR INSTEAD OF A RESULT — a timeout, a dropped connection —
        the decision in it was already written before the reply was lost, and setup is re-entrant
        by design. So: call `onboard()` with NO arguments to re-read where you are from disk, and
        continue from what it says. Never re-send the same `consent`/`source`/`start` argument
        blind, and never tell the user their "request timed out" or "failed" — they made a choice,
        not a request, and the choice stands. If you must say anything, say OPYT is still setting
        up in the background.

        Then call `oracle` to choose who to trust.
        """
        # THE HOSTED HOME SKIPS THE KEYS PHASE ENTIRELY. A hosted child is already talking to a
        # gateway that authenticated this user before the child existed, so there is nobody left
        # to ask and nothing for them to approve: the allowance is claimed here and the call
        # carries on into the sources question in the same turn. Local cannot do this — it needs
        # a browser and therefore a human turn — so it goes through `_phase_keys` like always.
        minted = _mint_silently(start)
        result = _onboard(consent, source, start)
        if minted:
            # Said ONCE, on the call that minted, and carried out alongside whatever the user is
            # already being asked. A second call would cost the turn this whole design removes.
            result["trial"] = _TRIAL_MINTED_NOTE
        return result


def _mint_silently(start: str | None) -> bool:
    """Claim the allowance with no question and no turn, where that is possible. True if minted.

    Fail-safe in both directions: a gateway that will not mint leaves the phase at `keys`, and
    the caller offers the OpenRouter approval exactly as it would have without a trial.
    """
    state = onboard_state.derive()
    if (state["phase"] == "keys" and start is None
            and state["keys"]["openrouter"]["state"] == "missing"
            and trial.hosted_enabled() and trial.available()):
        return trial.acquire().get("status") == "stored"
    return False


def _onboard(consent: str | None, source: str | None, start: str | None) -> dict:
        state = onboard_state.derive()

        # AN EXPLICIT SWITCH TO THEIR OWN ACCOUNT IS HONORED AT ANY PHASE — the same rule
        # `source` follows below, and for the same reason. A user on a LIVE starter allowance
        # has no phase asking them for a key, so without this the only route to their own
        # OpenRouter account is to spend the allowance first. Somebody who already has an
        # account and says so should not have to burn somebody else's money to be believed.
        if start == "openrouter" and state["phase"] != "keys":
            return _run_openrouter()
        if start == "trial" and state["phase"] != "keys":
            # Refuse, never guess: this home already has a working key, and minting a second
            # allowance for it would spend the operator's money to replace something live.
            return {"status": "error", "phase": state["phase"],
                    "message": ("This install already has a working model provider, so there is "
                                "no allowance to claim. Nothing was changed.")}

        if state["phase"] == "keys":
            return _phase_keys(state, start)

        if source is not None:
            if source not in _SOURCE_WORDS:
                # Refuse, never guess — the same rule the consent words follow. Guessing here
                # would launch a browser window at a site the user did not name.
                return {"status": "error", "phase": "sources",
                        "message": (f"`source={source!r}` is not one of "
                                    f"{' | '.join(_SOURCE_WORDS)}. Nothing was opened.")}
            if source == "skip":
                onboard_state.mark_sources_skipped()
                state = onboard_state.derive()
            elif consent is None:
                # An explicit connect is honored at ANY phase, not only while the sources step
                # is the one outstanding. A connected session still expires, and can belong to
                # the wrong account; `HostedXSession.graphql_get` already tells the user to
                # "reconnect X" when x.com rejects it, and gating this on the phase left that
                # instruction with no way to carry it out short of deleting the profile by hand.
                return (_connect_step(source) if source in _LOGIN_URLS
                        else _named_root_step(source))
            # A source request arriving WITH a consent answer is DEFERRED, never dropped: the
            # answer is the half that cannot be re-offered, and the source is one more call.

        if state["phase"] == "sources":
            return _phase_sources()

        # "Done" too early is the failure this gate catches: `derive` marks a source connected on
        # cookie PRESENCE, but a just-signed-in session can be present without yet answering an
        # authenticated request (Substack's anonymous `sid`; Chrome's lazy cookie flush). Building
        # the consent question and the curation pull on such a session skips every collector and
        # reports "you follow nobody". Verified with a retry, and only while setup is still open —
        # an established home is in phase `done` and never reaches here.
        if state["phase"] in {"consent", "curation"}:
            not_ready = _connected_but_not_ready(state)
            if not_ready:
                return _login_not_ready(not_ready)

        # Connecting a session is the consent boundary for the free curation reads — those
        # collectors scrape a logged-in session, and on 2026-08-20 a cold-start pass read a
        # user's cookie jar before they had typed anything about OPYT. Gated on a session
        # actually being connected, NOT on merely leaving the sources phase: a user who reached
        # `oracle` by naming blog URLs, or answered `skip`, has authorized no session read.
        if _any_session(state):
            from pipeline.kb import curation_catchup
            curation_catchup.grant_consent()

        extra: dict = {}
        # ⚠️ ARM A RUNS BEFORE THE CONSENT QUESTION, and that is the ordering change of
        # 2026-09-13 — not new plumbing. There is no "connect moment" to hook locally:
        # `guided_login.start` launches Chrome and returns immediately, and detection is ONLY
        # `has_managed_x_session()` polled inside `onboard_state.derive()`. So "fire at connect"
        # means "fire on the next `onboard` call once `pending` is non-empty", which is exactly
        # what this is.
        #
        # It runs pre-consent because it NEEDS no answer: the block above already grants curation
        # consent on any live session, since connecting the session IS that consent. Arm B below
        # is the opposite — on a fresh store `bookmark_catchup.consented()` has no marker and no
        # atoms to infer one from, and the saved BODIES are precisely what the question asks
        # about. So: signals before the question, bodies after it.
        #
        # The gain is that the candidate list is already scored while the user is reading the
        # consent prompt, instead of after they answer it.
        arm_a = set(state["curation"]["pending"]) if _any_session(state) else set()
        if arm_a:
            # STARTED, NOT AWAITED (2026-09-14). It is ~30s and ~25 requests, and it was being
            # spent with the user watching a blank screen BEFORE the consent prompt they are
            # waiting for. Nothing between here and `screen` reads a candidate — `screen` is the
            # first surface that does, and it is several turns and one human decision away, by
            # which time this is long finished.
            extra["curation"] = _start_arm_a(arm_a)
            # ⚠️ THE RECURRING HALF IS QUEUED AT THE START SITE (2026-09-16), for the same reason
            # the Substack backlog rail is queued at its start site below: the event that STARTS a
            # walk is the event that schedules its repeat, so the two cannot drift apart.
            #
            # It used to sit after the consent branch, and on a real fresh install it never ran at
            # all. `onboard` is a multi-call flow and the call that starts Arm A is the call that
            # RETURNS the consent prompt — so that line was unreachable on it. By the next call
            # `arm_a` is recomputed from a state the walk has already changed: `curation.pending`
            # is "connected platforms whose own collectors have never succeeded", and one
            # succeeding collector marks the whole platform read. Measured on a clean install
            # 2026-09-16: `x_lists` went ok at 05:33:32, consent was applied at 05:33:38, so
            # `pending` was empty and `arm_a` was empty six seconds before the queue was reached.
            # The condition was SELF-DEFEATING — the rail was queued only while the walk had not
            # succeeded yet — which is why it is not a flaky race: the walk takes ~8s and a human
            # reading a consent prompt takes longer, so it failed for every real user while a
            # one-call test (spawn and consent in the same call) kept passing.
            #
            # ORDER IS STILL SATISFIED, and that is why this is safe here rather than merely
            # earlier: the marker is written by the `grant_consent()` above — connecting the
            # session IS this rail's consent — so it is on disk before the row exists, and the
            # worker cannot claim a job whose consent has not been written. The refresh consent
            # `_follow_consent_with_a_worker` protects is a DIFFERENT marker for a different rail.
            from pipeline.kb.rail_jobs import request_now
            extra["curation_queued"] = request_now("curation_catchup")

        if state["phase"] == "consent" or consent:
            if consent is None:
                # Arm A's report rides along: it already ran, and a prompt that hid it would be
                # asking the user to wait for work that is done.
                return {**_consent_prompt(state), **extra}
            if consent not in _CONSENT_WORDS:
                # Refuse, never guess. "yes" is ambiguous across two commitments with
                # different cost shapes, and one of them cannot be revoked.
                return {"status": "error", "phase": "consent",
                        "message": (f"`consent={consent!r}` is not one of "
                                    f"{' | '.join(_CONSENT_WORDS)}. Nothing was recorded.")}
            applied = _apply_consent(consent, connected=state["sources"])
            state = onboard_state.derive()
            state["consent_applied"] = applied

        if "consent_applied" in state:
            extra["consent_applied"] = state["consent_applied"]
            # ⚠️ ONLY WHEN THE PROMISE IS NOT BEING KEPT. A worker that installed needs no
            # sentence — "it updates on a schedule" is simply true and the user asked for it. A
            # worker that COULD NOT install turns every later "it fills in on its own" into a
            # false claim, and the host has to know before it makes one. Same rule as
            # `install_worker.status`: under-promise when the probe cannot answer.
            worker = (state["consent_applied"].get("worker") or {})
            if worker.get("status") in _WORKER_RUNNING:
                # ⚠️ NAME NO FILE AND NO PROCESS. This dict carried `path` (a `.plist` under
                # LaunchAgents) and `log` (a dotfile under the data home) — the only description
                # of the worker the host could see — so the host described THAT. Measured on a
                # fresh onboarding 2026-09-15: "Opyt installed a background process on your Mac
                # that starts on its own. Its log is in a hidden folder in your home folder."
                # That is how a person describes malware, and it was the reply to a user who had
                # just consented and done nothing wrong.
                #
                # The paths are an operator's surface, not a user's: `install_worker.status()`
                # still returns them to anyone who asks, `_follow_consent_with_a_worker` still
                # builds them, and nothing downstream of here reads them. Only the copy the host
                # composes from is narrowed — which is the same fix shape as the failure branch
                # below, whose `message` has kept that branch honest since it was written.
                # The note rides ON the worker dict, NOT on `scheduled_updates`. That key is the
                # WARNING — "nothing is keeping this promise" — and two tests assert it is absent
                # exactly here, which is the contract: a reader must be able to check one key to
                # know whether the schedule is in danger. Putting reassurance under the same name
                # would make its presence meaningless.
                extra["consent_applied"] = {
                    **state["consent_applied"],
                    "worker": {"status": worker["status"],
                               "host_note": ("Their library keeps itself current from now on, "
                                             "including when this conversation is closed. Say "
                                             "that in ONE plain sentence, and only if it is "
                                             "worth saying at all. Name no file, no folder, no "
                                             "process and no schedule — none of it is theirs to "
                                             "manage, and a person told about a background "
                                             "process that starts on its own hears a description "
                                             "of malware, not a feature they asked for.")},
                }
            elif worker.get("status") in ("unsupported", "not_installed", "ERROR_LOAD"):
                extra["scheduled_updates"] = {
                    "running": False,
                    "detail": worker,
                    "message": ("Their choice to keep sources current IS recorded and stands. "
                                "But nothing on this machine will act on it on its own yet, so "
                                "do NOT tell them things update automatically or fill in in the "
                                "background — say their library updates when they ask for it."),
                }

        # ── ARM B: the saved posts themselves ──────────────────────────────────
        # SPAWNED here, JOINED below (R7). It is not a decision input — nothing between here and
        # the join reads a bookmark atom — so it runs behind the rail queueing and the state
        # re-derive rather than in front of them. But it IS joined before this call returns: the
        # completeness guarantee is that when `onboard` says setup is done, the saved posts are in
        # the store. Only their metered upgrades are still owed, and Enrichment owns those.
        #
        # WHICH PLATFORMS: the ones whose Arm A just ran (a newly connected session), plus every
        # live platform when THIS call is the one that granted backlog consent. Not simply "every
        # live platform every time" — that would re-walk the whole saved list on every `onboard`
        # call for nothing.
        arm_b = set(arm_a)
        if "backlog" in (state.get("consent_applied") or {}).get("granted", []):
            arm_b |= _live_platforms(state)
        # ⚠️ THE RECURRING HALF, FOR A PLATFORM CONNECTED AFTER THE ANSWER (2026-09-14). Arm B
        # runs the import in-process; the rail is what keeps it running later. `_apply_consent`
        # queues that rail only for platforms live AT THE CONSENT MOMENT, so connecting Substack
        # afterwards — the flow `_phase_sources` itself recommends, since the prompt names the
        # unconnected roots as "still open" — imported once and never again. Measured on a real
        # session: consent at 22:47 with only X live, Substack connected at 23:03, the saved
        # posts imported, and `substack_saved_catchup` had no row in `rail_jobs.db` at all.
        # Queueing here makes the event that STARTS an import the event that schedules its
        # repeat, so the two cannot drift apart again.
        #
        # Skips what this call already queued: `request_now` on a live row is harmless, but the
        # consent path's own ordering test reads the queue as a sequence.
        already = set((state.get("consent_applied") or {}).get("queued", []))
        for platform in sorted(arm_b):
            rail = _BACKLOG_RAILS[platform][1]
            if rail in already or not _backlog_consented(platform):
                continue
            if _queue_backlog(platform):
                extra.setdefault("backlog_queued", []).append(rail)
        started = _start_saved_content(arm_b)

        if arm_a:
            # A CHANCE, NOT A GUARANTEE, and that is the whole of what is left here. Arm A is a
            # thread now, so `curation.pending` empties whenever it happens to finish — this
            # re-read simply lets `_handoff` report `done` instead of `curation` when the walk
            # beat this line. The rail row that used to be queued here moved UP to the start site;
            # see the Arm A block for why it could never fire from this side.
            state = onboard_state.derive()

        if saved := _report_saved_content(started):
            extra["saved_content"] = saved

        # Outside the `saved` block: this engine's corpus is the Oracle roster, not the import
        # that just ran, so a store with owed windows and nothing newly saved still wants it.
        if footprint := _start_footprint_enrichment():
            extra["footprint_enrichment"] = footprint

        # An Oracle pull that finished while nobody was looking. Setup is the likeliest place a
        # returning user lands first, and OPYT has no way to have told them sooner — it speaks
        # only when it is called. Once, then never again: `completion_notice` stamps as it hands
        # over. Fail-safe and silent; a courtesy must never be able to break setup.
        try:
            from pipeline.kb import pull_runs
            conn = pull_runs.connect()
            try:
                if (notice := pull_runs.completion_notice(conn)) is not None:
                    extra["pull_finished"] = notice
                # Material the user has never been told about. Setup is where a store first goes
                # from empty to full, so this is the likeliest place it fires — and the place the
                # silence was measured (2026-09-15: 1,004 atoms in, "anything else?" out).
                from pipeline.kb import new_material
                if (landed := new_material.new_material_notice(conn)) is not None:
                    extra["new_material"] = landed
            finally:
                conn.close()
        except Exception:
            pass

        return _handoff(state, **extra)


# Carried back on the call that minted, so the alternative is discoverable without costing a turn
# or a question. A CLAUSE, not a paragraph: the user is mid-setup and did not ask about model
# providers, so anything longer buries the step they are actually on.
_TRIAL_MINTED_NOTE = (
    "Setup is handled — they needed no account and paid nothing. If, and ONLY if, it fits as a "
    "short clause alongside what you are already saying, note once that they can use their own "
    "OpenRouter account instead if they prefer (that runs `onboard(start='openrouter')`, which "
    "works at any time). Never its own paragraph, never a question, never explained."
)


def _phase_keys(state: dict, start: str | None) -> dict:
    """The keys phase is OpenRouter and nothing else, and `onboard_state.derive` agrees: it sets
    `phase == "keys"` from `orx["state"] == "ok"` alone, so this function is unreachable once the
    one required key is live.

    That is why there is no optional-key step here and must not be one. A `skip_github` parameter
    sat in `onboard`'s signature until 2026-09-07, unread by anything for its whole life; any step
    it had gated would have been unreachable for every install whose OpenRouter key already
    worked, and making it reachable means putting an OPTIONAL credential into the phase gate.

    The optional rows in `opyt_core/credentials_registry.py` are set with `opyt-keys`, and none of
    them is needed. Named by SERVICE here, never by env var, for the reason the module docstring
    gives: github is paced for its anonymous search limit in `frontier_sources.GitHubAdapter`, so
    a token there is throughput and not capability; semanticscholar stays unadvertised because
    AI2 no longer approves third-party key requests; opyt_service is written by `share`.
    """
    orx = state["keys"]["openrouter"]

    # THE TRIAL IS OVER — first, because every branch below it would give this user advice that
    # is exactly backwards. `unfunded` sends them to a credits page for an account they never
    # made; `dead` tells them to re-approve, which would mint a second key against a balance
    # that is already spent. The only move left is the one they skipped at setup.
    if orx["state"] == "trial_over":
        if start is None:
            return _trial_over_prompt()
        if start != "openrouter":
            return {"status": "error", "phase": "keys",
                    "message": (f"`start={start!r}` is not `openrouter`. The starter allowance "
                                f"is spent, so a key of their own is the only step left here. "
                                f"Nothing was opened.")}
        return _run_openrouter()

    # ⚠️ A HUMAN ASKING FOR THE REMEDY IS NOT THE PASSIVE PATH, AND BLOCKING IT WAS A DEAD END.
    # `readiness` answers a rejected key with "Call `onboard` again to approve a fresh one", and
    # `allowance_notice` puts `onboard(start='openrouter')` in `next_call` — then this branch
    # refused that exact call and handed the same sentence back. Measured 2026-09-15 on a revoked
    # key: the notice rendered correctly, the user said yes, and setup answered `blocked` with the
    # instruction it had just given. There was no way out of it — `dead` never reached
    # `_run_openrouter`, so a revoked key could not be replaced through the documented path at all.
    #
    # `start='openrouter'` already bypasses this whole function at the top of `onboard`, but only
    # when `phase != "keys"` — and a dead key IS phase keys, so the bypass could never fire for
    # the one state that needs it most.
    #
    # `unknown` joins `dead` because the two are indistinguishable from here BY CONSTRUCTION: the
    # probe runs through the same breaker the failures opened, so "could not verify" is the normal
    # reading of a key that is failing, not a separate condition. Approving a key they turn out
    # not to have needed costs one browser tab; the dead end cost them the product.
    #
    # `unfunded` is DELIBERATELY NOT HERE. Its remedy is money, not another approval — minting a
    # second key against a balance that is already spent is the loop `_NEXT_CALL` omits it to
    # avoid — and `readiness` already names the top-up page for it.
    if orx["state"] in ("dead", "unknown") and start == "openrouter":
        return _run_openrouter()

    # OpenRouter is the ONLY key. An UNFUNDED account blocks exactly like a missing one: pulling
    # into a store that cannot be indexed is the fail-safe violation.
    if orx["state"] in ("unfunded", "dead", "unknown"):
        return {"status": "blocked", "phase": "keys", "openrouter": orx["state"],
                "message": (f"{orx['message']} Nothing was pulled. "
                            f"⚠️ The four curation collectors this would have run use browser sessions, "
                            f"not the OpenRouter key. They are still blocked, because "
                            f"anything they collect cannot be embedded or searched without a "
                            f"working OpenRouter key, and a store you cannot query is not worth "
                            f"building.")}
    if orx["state"] == "missing":
        if start is not None and start not in ("openrouter", "trial"):
            # Refuse, never guess — the same rule `source` and `consent` follow, and here
            # guessing means opening the tab this split exists to announce.
            return {"status": "error", "phase": "keys",
                    "message": (f"`start={start!r}` is not `trial` or `openrouter`, the two "
                                f"steps this phase can begin. Nothing was opened.")}
        if start is None:
            # The SHORTER road is offered when there is one. `trial.available` is false once
            # this home has had its allowance, so a user who spent it and then deleted their
            # key is sent to the approval rather than round-tripping for a refusal.
            return _trial_prompt() if trial.available() else _openrouter_prompt()
        if start == "trial":
            got = trial.acquire()
            if got["status"] == "stored":
                return {"status": "ok", "phase": "keys", "step": "trial",
                        "message": ("The starter allowance is set up — no account, nothing to "
                                    "pay. Call `onboard` again to carry on with setup.")}
            if got["status"] == "unavailable":
                # NOT an error, and not a dead end: the trial is a shortcut, and a shortcut
                # that is closed leaves the user exactly where they would have been anyway.
                return _openrouter_prompt(preface=_unavailable_preface(got.get("reason", "")))
            result = {"status": got["status"], "phase": "keys", "step": "trial",
                      "message": got["message"]}
            if "open_this_url" in got:
                result["open_this_url"] = got["open_this_url"]
            return result
        return _run_openrouter()

    return {"status": "ok", "phase": "keys", "message": "The one required key is live."}


def _unavailable_preface(reason: str) -> str:
    """What to say when the shortcut turned out to be closed for THIS person.

    ⚠️ THE FAILURE THIS FIXES, live on 2026-09-11. A returning user was told "sign in with
    Google, nothing to create, nothing to pay", said yes, and got back: *"the free starter
    credit isn't available on your install, so what I said before about not paying anything no
    longer applies."* An assistant retracting its own promise in public reads as an unreliable
    product, which is a worse outcome than the inconvenience it is describing.

    Three things caused it and all three are fixed here.

    FIRST, the old preface said "this is the ordinary setup path instead. Do not dwell on why",
    which INVITES the reconciliation it forbids: a model handed a contradiction will explain it,
    and telling it not to only makes the explanation apologetic. The instruction now forbids
    referring back at all, and gives a plain sentence to say instead.

    SECOND, "not available" was vague where the truth is specific and harmless — this person has
    already had their allowance. `trial.acquire` has carried the gateway's `reason` all along
    and this branch threw it away, rendering one generic sentence for four different causes.
    Naming the real one is both kinder and shorter.

    THIRD, nothing here says "free" or "credit". The promotional framing was stripped from the
    ask (`_trial_prompt`) and leaked back in through the fallback, which is the usual way such
    wording returns: by a path nobody re-read.

    WHY THE ASK STILL PROMISES. `_trial_prompt` says "nothing to pay" before the gateway has
    been asked whether THIS person may have an allowance — and on a local install it cannot
    know, because the subject is only established by the Google sign-in that has not happened
    yet. Softening the ask to cover this case would tax every first-time user, who is the
    common case and for whom the promise is true, to spare a rare returning one. So the ask
    stays confident and this function carries the cost.
    """
    if reason == "already_claimed":
        said = ("They have already used the starter credits that come with Opyt — one per "
                "person — so setup carries on with their own OpenRouter account.")
    else:
        said = ("The starter credits are not available right now, so setup carries on with "
                "their own OpenRouter account.")
    return (f"SAY THIS FIRST, in one sentence: {said}\n"
            "Do NOT refer back to anything you said earlier, do NOT apologise, and do NOT "
            "explain what changed or why. State it once as a fact and move straight on.\n\n")


def _run_openrouter() -> dict:
    """Begin the user's OWN OpenRouter approval. Reached from two states — a fresh install with
    no trial to offer, and a spent trial — and identical in both, which is the point: the trial
    defers this step and never replaces it."""
    got = openrouter_oauth.acquire()
    result = {"status": got["status"], "phase": "keys", "step": "openrouter",
              "message": got["message"]}
    if "open_this_url" in got:
        # An OpenRouter approval URL has a public PKCE challenge and a short-lived callback
        # capability, never a verifier or API key. The hosted user must receive it because
        # the browser runs on their device, not in the child.
        result["open_this_url"] = got["open_this_url"]
    return result


def _openrouter_prompt(preface: str = "") -> dict:
    """The OpenRouter step, in two sentences. `start='openrouter'` runs it.

    WHY IT IS TWO CALLS. `acquire()` calls `webbrowser.open` and then blocks in the loopback
    capture for up to five minutes, so on 2026-09-09 the approval page appeared in the browser
    before Claude Desktop had rendered a single word — and it could not have rendered one,
    because the tool had not returned. The user met an unfamiliar third-party consent screen
    with no idea what it was or that authorizing was the thing to do. Chat text was not a LATE
    lever there, it was an UNAVAILABLE one. Splitting the call is what makes it available.

    Do not collapse this back into one call. Do not replace `start` with a disk marker either:
    a marker makes "the user was told" a once-per-home fact, and the person who most needs
    telling is the one who abandoned setup a week ago and came back to an unannounced tab.

    BOTH HOMES take this split, though only local had the ordering bug — hosted returns
    `open_this_url` and opens nothing. Hosted pays one round trip it does not need for ordering
    and gets the explanation in exchange, because the page is equally unfamiliar when a hosted
    user clicks through to it.

    IT ALSO ASKS THE HOST TO WAIT. Splitting the call fixed the ORDER and not the PACE: in the
    2026-09-09 terminal run the explanation rendered and `start='openrouter'` fired eight
    seconds later, so the tab still took the screen before the text could be read. The host must
    put the question and wait for an answer. REJECTED — sleeping before `webbrowser.open`: any
    delay is a guess at reading speed baked into code, it wastes a fast reader's time on every
    install, still interrupts a slow one, and blocks the call doing nothing, so the host shows a
    spinner instead of the text.

    LENGTH IS PART OF THE CONTRACT HERE TOO. This carried four bullets until 2026-09-11 and a
    model turned them into a paragraph about model providers, passwords and pricing, aimed at
    somebody who had asked for none of it. `_trial_prompt`'s docstring has the full reasoning;
    the short version is that this string is an instruction to a model that expands it, so the
    cap has to be written into the text.

    `preface` is for ONE caller: a trial that turned out to be unavailable. It is a parameter
    rather than a second prompt function because everything after it — the ask, the ordering
    rule, the request to wait — must not fork.
    """
    return {
        "status": "needs_openrouter", "phase": "keys", "step": "openrouter",
        "message": (
            preface +
            "Nothing has happened yet. Say the line below, ask if they are ready, and WAIT for "
            "an answer — `onboard(start='openrouter')` takes over their screen with a browser "
            "tab, so it must not go in the same turn as this.\n\n"
            "SAY, in at most TWO sentences and no bullets: Opyt uses OpenRouter as its model "
            "provider, so the next step sends them to openrouter.ai — a different company's "
            "site — to sign in or make an account and authorize Opyt. Making the account is "
            f"free. {readiness.COST_NOTE}\n\n"
            "Then ask if they are ready. Do NOT explain what a model provider does, do NOT "
            "mention passwords or what Opyt stores unless they ask, and do NOT describe how "
            "that page looks or where its button is — the page is OpenRouter's and it changes."),
    }


def _trial_prompt() -> dict:
    """The sign-in step, in ONE sentence. `start='trial'` runs it.

    Same two-call split as `_openrouter_prompt`, whose docstring carries the reason: `acquire`
    opens a tab and then blocks, so one call would put a sign-in page on screen before the host
    had said why.

    WHY THIS COPY IS SO SHORT, AND MUST STAY SHORT. Nobody reads this string — it is an
    INSTRUCTION TO A MODEL THAT EXPANDS IT, so what a user reads is a multiple of what is
    written here. The first version listed four bullets and a closing line; live on 2026-09-11
    Claude rendered it as nine sentences about model providers, passwords, and what happens when
    credits run out, at somebody who had just asked to be onboarded. None of it was wrong and
    all of it was too much.
    So: state the ASK, cap the OUTPUT, hold the rest back until asked. The cap is explicit
    because a terse model was already landing in the right place unaided and a verbose one needs
    a ceiling — the copy must not depend on which model is driving.

    NOTHING HERE IS CALLED A TRIAL, AN ALLOWANCE, OR FREE. "Free starter allowance" reads as a
    promotion, and a promotion attached to a sign-in button is the shape of a scam. The honest
    version is duller and shorter. Money is not mentioned at all, because at this moment nothing
    about money is being asked of them; it becomes true later, and `_trial_over_prompt` says so
    then.
    """
    return {
        "status": "needs_trial", "phase": "keys", "step": "trial",
        "message": (
            "Nothing has happened yet. Say the line below, ask if they are ready, and WAIT for "
            "an answer — `onboard(start='trial')` takes over their screen with a browser tab, "
            "so it must not go in the same turn as this.\n\n"
            "SAY, in ONE sentence, no bullets and no preamble: Opyt needs a single Google "
            "sign-in to finish setting up — nothing to create, nothing to pay.\n\n"
            "Then ask if they are ready. Do NOT explain what a model provider is, do NOT "
            "mention credits or costs, and do NOT describe Google's page. Only if they ask why "
            "or what it costs: Opyt covers the model costs to start with, and when that runs "
            "out it says so and helps them connect their own."),
    }


def _trial_over_prompt() -> dict:
    """The credits are spent. The ONE moment OPYT asks a user to pay for anything.

    It is also the moment the design exists to reach: this person has a working knowledge base
    in front of them, so it is a top-up prompt and not a pitch. TWO sentences, for the reason
    `_trial_prompt` sets out — a wall of reassurance here reads as a company that knows it is
    asking for something awkward.

    THE COST SENTENCE IS `readiness.COST_NOTE`, rendered rather than restated, so correcting the
    number is one edit and cannot leave a stale copy behind in this string.
    """
    return {
        "status": "needs_openrouter", "phase": "keys", "step": "openrouter",
        "message": (
            "Nothing has happened yet. Say the lines below, ask if they are ready, and WAIT — "
            "`onboard(start='openrouter')` takes over their screen with a browser tab.\n\n"
            "SAY, in at most TWO sentences and no bullets: the model credits Opyt covered for "
            "them are used up, and everything already collected is still there. To keep going "
            "they connect their own OpenRouter account — free to make, and the credit goes on "
            f"it and stays theirs. {readiness.COST_NOTE}\n\n"
            "Then ask if they are ready. Do NOT describe how that page looks or where its "
            "button is — the page is OpenRouter's and it changes."),
    }


# ── phase 1: a source ───────────────────────────────────────────────────────────────────────
#
# FOUR roots, and only two of them have an account to connect. X has auto-discovery (Lists,
# following, likes) and a login; Substack has subscriptions and a login. Research and a personal
# site have neither — there is no account to read, so the user names them, and the tool that
# OWNS each of those lists is where they are named. This phase points at those tools; it does not
# proxy them.
#
# WHY POINT AND NOT PROXY. `oracle(action='confirm')` carries the preview/confirm split, the three
# lookback selectors, and `_unsupported_root`'s refusal messages. An `onboard(source='research',
# handle=…)` parameter would be a second door onto the same write, and it would either duplicate
# all of that or silently drop it. So a named root returns the QUESTION to put and the CALL to
# make, and the owning tool stays the only writer.
#
# GITHUB IS NOT A ROOT — RULED 2026-09-08 (David), and do not re-propose it. Measured across three
# accounts (maimond123, karpathy, simonw): ~50% of starred-repo owners are organizations, and only
# 6-8% of owners are starred more than once, with the top count almost always the user's own
# repos or their own org. `x_likes` works as "liked AUTHORS + count" because likes CONCENTRATE on
# a person; stars do not, so GitHub candidates arrive as an undifferentiated block at count=1 that
# `screen.Candidate.sort_key` has nothing to order by. GitHub stays a SECONDARY source, where an
# attestation already exists: `oracle_refresh_state.github_owners_from_links` widens a confirmed
# Oracle's cluster onto a ("github", owner) pair.

# The roots with a login, and where the user signs in. One dict, so the accepted words and the
# URLs cannot disagree.
#
# Substack's sign-in page offers exactly two methods — an email box, and "Sign in with password".
# It has NO Google/Apple/Twitter SSO, which is what makes it work in a CDP-launched profile at
# all: `--remote-debugging-port` sets `navigator.webdriver`, and Google refuses sign-in when it
# sees that (see docs/plans/2026-09-07-hosted-x-google-sso-block-context.md). Substack has no
# Google path to refuse.
_LOGIN_URLS = {"x": "https://x.com/login",
               "substack": "https://substack.com/sign-in"}

# What each login root DISCOVERS, in one clause. Copy, not behavior — `_LOGIN_URLS` stays the
# one table that says a root exists and where its sign-in is. This exists because the same
# phrase was written out twice in `_handoff` and would have been written a third time by
# `_still_open`.
_ROOT_DISCOVERS = {"x": "people from your Lists, following and likes",
                   "substack": "the writers you follow and the newsletters you subscribe to"}

# The roots the USER names. No account, no login, nothing to open — OPYT needs an identifier
# before anything happens, so answering one of these returns the question to put and the call to
# make with the answer.
#
# A SECOND dict rather than a `kind` field on one registry, because `_LOGIN_URLS[source]` is a
# total function on its own keyspace and `_any_session` iterates it directly. Folding both into
# one table would make every login-side reader filter by kind first. `_SOURCE_WORDS` derives from
# both, so a word can never exist without a handler.
_NAMED_ROOTS = {
    "research": (
        "Ask which — a person, or a subject. They produce different things:\n"
        "• **A researcher** — `oracle(action='confirm', add_handles=['https://orcid.org/0000-…'])`. "
        "An OpenAlex author page (https://openalex.org/A…) works too, and is the one to reach for "
        "when they have no ORCID. OPYT resolves either to their publication "
        "record and pulls the papers as atoms; abstracts are free, so the whole back catalogue "
        "is the default window. An academic URL OPYT cannot pull work by (Semantic Scholar, "
        "Google Scholar, DBLP, ResearchGate) is refused by name, with the shape that would have "
        "worked.\n"
        "  ⚠️ A bare NAME does not resolve and must not be passed — OpenAlex returned 16 people "
        "for \"Frances Arnold\" on 2026-09-08. But finding the URL is YOUR job, not the user's: "
        "take the name they give, web-search \"<name> ORCID\" or \"<name> openalex\" yourself, "
        "and pass the page you find. If the name is ambiguous, ask the user to pick between the "
        "real candidates (\"the Caltech chemist?\") — never send them off to fetch an "
        "identifier.\n"
        "• **A subject** — interests, not commitments (RULED 2026-09-12: NO onboarding "
        "surface creates a standing watch, this root included — a user typing what they are "
        "into is not asking for one, whatever word they clicked to get here). Fill the store "
        "instead: web-search the landscape on what they name, SHOW them the strong finds — "
        "papers especially, this is the research root — and save the ones they pick; nothing "
        "from a web search enters the store without their say-so. Recurring authors are the "
        "researcher path above: offer them, and confirm the accepted ones. You may mention "
        "ONCE that OPYT can also keep a subject on standing watch later, after they have "
        "read some of what is now in the store — and leave it at that."),
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
_SOURCE_WORDS = tuple(_LOGIN_URLS) + tuple(_NAMED_ROOTS) + ("skip",)


def _any_session(state: dict) -> bool:
    """Is any platform session connected to OPYT right now? The consent boundary for a cookie
    scrape — see the call site."""
    return any(state["sources"][s] for s in _LOGIN_URLS)


def _connected_but_not_ready(state: dict) -> list[str]:
    """Connected login sources whose session is PRESENT but does not answer an authenticated
    request yet — the "done"-too-early case, returned so the caller can hold instead of building
    the whole consent+curation flow on a session that will 401.

    Why this gate exists: `onboard_state.derive` marks a source connected on cookie PRESENCE, and
    presence is not validity. Substack hands anonymous visitors a `substack.sid`, and Chrome
    flushes cookies lazily, so the instant a user clicks "done" the session can be present but not
    usable — which is exactly how a real run on 2026-09-13 skipped every collector and reported
    "you follow nobody" with 36 subscriptions sitting in the account. `managed_substack_session_ready`
    verifies with a short retry that rides out the flush.

    Substack only, deliberately. X's `auth_token` has no anonymous form — presence there is a far
    better proxy for validity — and adding a live X read would risk translating a transient refusal
    into a false "reconnect X". If X ever needs the same guard, it gets its own ready-check.
    """
    if not state["sources"].get("substack"):
        return []
    from pipeline.ingestion.sources.substack import managed_substack_session_ready
    try:
        return [] if managed_substack_session_ready() else ["substack"]
    except Exception:
        # Fail-safe: an unexpected verifier error must not strand a user who did sign in. Let the
        # flow proceed; the collector records a real skip rather than a lie if the session is dead.
        return []


def _login_not_ready(sources: list[str]) -> dict:
    """The window is open and OPYT can see it, but the sign-in has not registered yet. Ask the
    user to finish in that window and come back — nothing was pulled, so nothing is half-done."""
    names = _phrase([s.title() for s in sources])
    return {
        "status": "awaiting_login", "phase": "sources", "source": sources[0],
        "message": (
            f"The {names} window isn't signed in yet — I checked, and the session doesn't answer "
            f"an authenticated request. Finish signing in inside Opyt's window; it can take a few "
            f"seconds to register after you log in. Then tell me you're done and I'll check again. "
            f"Nothing was pulled, so nothing is half-done."),
    }


def _phase_sources() -> dict:
    """Nothing in this home will produce atoms or candidates yet. Offer every way to change that.

    Until 2026-09-07 this phase asked one question — "is X connected" — with no bypass, so a user
    who reads Substack, or follows named blogs, or watches research topics never reached consent
    or curation at all. The list below is what replaced it.

    IT ASKS FOR ONE ROOT, AND THAT IS DELIBERATE. This message read "they are NOT exclusive —
    take one, then call again with another" until 2026-09-09, and a host reasonably took that as
    an invitation to batch: in the terminal run it rendered a multi-select and told the user
    "pick as many as you want and I'll run them one at a time". Two of the roots send the user
    out of the conversation to sign in, so a multi-pick is a queue the host has to carry across
    a human round trip. Dropping the second one silently breaks a promise the user was given,
    and `_still_open` only catches it at the handoff, which can be several turns later.

    The roots really are not exclusive — that part is true and stays. What changed is what it is
    offered AS: a reason nothing is lost by picking one, rather than a reason to pick several.
    """
    from pipeline.ingestion import browser_cookies as bc

    resume = ""
    if bc.opyt_session_backends():
        # A profile on disk does not prove a window is open, and OPYT cannot see one — so this
        # says BOTH things that might be true rather than picking one.
        resume = ("\n\n⚠️ OPYT's login profile already exists but holds no session it can use. "
                  "If the user has just logged in, call `onboard` again — it will pick the "
                  "session up. Otherwise pick a source below.")

    return {
        "status": "needs_source", "phase": "sources",
        "answers": list(_SOURCE_WORDS),
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
            "following and likes become candidates to screen without anyone typing a name.\n"
            # "Same shape" became TRUE again on 2026-09-15, when the paste flow was deleted
            # and Substack adopted X's desktop (docs/plans/
            # 2026-09-15-substack-adopts-the-x-desktop-flow.md). The steer-to-computer copy
            # that briefly lived here served the paste flow's phone dead end; do not bring it
            # back for the desktop flow, which reconnects on a phone exactly as X's does.
            "2. **Substack** — `onboard(source='substack')`. Same shape: OPYT opens its own "
            "browser profile at Substack's sign-in; the user logs in there (Substack emails "
            "them a code), and the writers they follow and the newsletters they subscribe to "
            "become candidates. Their saved posts import separately, at the consent step.\n"
            "3. **Research** — `onboard(source='research')`. A researcher's whole publication "
            "record, or a subject to build the store around. OPYT asks which and hands you "
            "the call.\n"
            "4. **A blog, newsletter or personal site** — `onboard(source='blog')`. Any http "
            "home, no per-host setup.\n"
            "5. **Nothing yet** — `onboard(source='skip')`. Setup continues and this question is "
            "not asked again; any of the four above still works later." + resume),
    }


def _named_root_step(source: str) -> dict:
    """A root the user NAMES → the question to put and the call to make with the answer.

    Nothing is opened, nothing is written, and no marker is recorded — because nothing has
    happened yet. The user has picked a KIND of source, not a source. If they never name one,
    `derive()` still finds no `oracles` and no `watchlist` row and asks again, which is correct:
    the question really is unanswered. `skip` is how they say "not now".

    Status is `needs_name`, not `needs_source`: the sources question IS answered here, and what
    is outstanding is the identifier. A caller that cannot tell those apart would re-print the
    five-root menu at a user who already chose.
    """
    return {"status": "needs_name", "phase": "sources", "source": source,
            "message": _NAMED_ROOTS[source]}


# The two homes need different notes for ONE reason: the desktop is not where the user reads their
# mail, so an emailed sign-in has to finish somewhere else. It does, by itself — see the hosted
# note below. Locally mail and browser are already together, so the code works untouched here and
# leads.
#
# One email carries both a code and a sign-in link.
#
# The paste-the-link clause is here too, because the silence measured on the box follows the
# BROWSER, not the hosting. An OPYT-launched Chrome on a laptop, on the user's own residential IP,
# requested a sign-in email 8 times on 2026-09-08 and received 0, while Safari on the same machine
# got one in 64 seconds. Those windows carried `--remote-debugging-port`; since 2026-09-13
# `cdp.launch` opens the sign-in window with no debugger (the Google-SSO boundary rule).
#
# RE-MEASURED HOSTED, 2026-09-15, portless: 2 requests from the box's exact sign-in launch shape
# (same argv builder, fresh profile, datacenter IP, driven by xdotool so nothing attached), 2
# emails delivered — each stamped the same second as its submit — and the second's CODE typed
# into the requesting browser landed a full session. The old 0-for-8 was almost certainly the
# debugger flag. n=2, not yet a reliability trial, so the third clause below stays as the
# unstuck path; what the result reopens is the DESIGN question (an X-style desktop flow for
# Substack), which is David's to take, not this comment's.
#
# Measurements: docs/plans/2026-09-08-hosted-substack-signin.md, and the delivery re-measurement
# in docs/plans/2026-09-15-substack-sign-in-must-survive-a-phone-handoff.md.
_SIGN_IN_NOTES = {
    "substack": (
        "\n\nSubstack offers three ways in:\n"
        "• **Sign in with password** if they have one — the clean path.\n"
        "• The email box. One email carries a **code** and a sign-in link; use either in the "
        "browser OPYT opened.\n"
        "• If no email arrives within a few minutes, have them request one **in their own "
        "browser** and paste that email's link into OPYT's window. The token is not tied to a "
        "device. ⚠️ In THIS clause only, the code is a trap: the email's code and its link spend "
        "the same single sign-in, so typing the code into their own browser signs them in there "
        "— the one session OPYT cannot read — and the link they were going to paste is dead. "
        "Measured hosted 2026-09-15; the mechanism is Substack's, not the hosting's.\n"
        "⚠️ Finish the sign-in there. A session created in their normal browser is "
        "one OPYT cannot read."),
}

# Substack runs the same desktop as X since 2026-09-15, when the box was measured actually
# receiving Substack's sign-in email under the portless launch (2 requests, 2 same-second
# deliveries, code signed in — docs/plans/2026-09-15-substack-adopts-the-x-desktop-flow.md).
# The guided-paste flow that lived here before, and every clause about copying links out of
# emails, served the delivery silence that measurement retired.
#
# ⚠️ There was a "do not tap the link in that email" warning here, and it is deleted (2026-09-16,
# David, alongside the same sentence on the page). It claimed the link and the code spend one
# sign-in, so tapping the link killed the code. That direction was never measured: what WAS
# measured is the converse — typing the code into Substack's own page killed the link — and the
# only reading pointing the other way says links survive being clicked (four redemptions of three
# links, 2026-09-09). Do not restore it from memory; measure it first, and remember the page
# itself shows Substack's code boxes, which already say what to do.
_HOSTED_SIGN_IN_NOTES = {
    "x": (
        " It shows a private browser desktop running on Opyt's server, which is where the "
        "profile lives. They sign in there as they would in any browser, including Google "
        "sign-in and 2FA."),
    "substack": (
        " It shows the same private browser desktop as X, running on Opyt's server, which is "
        "where the profile lives. They type their email address there; Substack emails them a "
        "code, and they type the code into that desktop. Leaving for the mail app and coming "
        "back is fine — the desktop reconnects. Anyone with a Substack password can use it "
        "instead, from the sign-in page the desktop shows."),
}


def _connect_step(source: str) -> dict:
    """Open OPYT's own login profile for `source`, or mint the hosted sign-in link for it.

    Never scans the user's browsers. The profile OPYT opens here is one it created, which is
    what makes reading it back a read of OPYT's own session rather than of the user's. Both
    homes take the same two branches for every source: hosted mints a link, local opens a
    window, and `_LOGIN_URLS` is the one place either learns where to point.
    """
    from pipeline.ingestion import hosted_browser
    from pipeline.ingestion.utils import SyncAuthError

    if hosted_browser.enabled():
        try:
            login_url = hosted_browser.begin_login(source, _LOGIN_URLS[source])
        except hosted_browser.HostedBrowserError as e:
            return {"status": "needs_login", "phase": "sources", "source": source,
                    "message": str(e)}
        return {
            "status": "awaiting_login",
            "phase": "sources",
            "source": source,
            "login_url": login_url,
            "message": (
                # HOW to give it, not just to give it. `login_url` stays a bare URL because a
                # client that cannot render markdown has to have something printable; the
                # instruction is what a client that CAN render obeys. Saying only "give them
                # the link" left the choice to the host, and it came out a raw URL as often as
                # words. Same sentence in `opyt_core/openrouter_oauth.py`.
                f"A sign-in link is ready in `login_url`. Present it to the user as a link "
                f"whose text reads **Connect {source.title()} to Opyt** — do not print the raw "
                f"URL. Only the user can open it. It works once and expires in ten minutes. "
                f"When their page says the connection is made, call `onboard` again."
                + _HOSTED_SIGN_IN_NOTES.get(source, "")),
        }

    from pipeline.ingestion import guided_login

    try:
        backend = guided_login.start(_LOGIN_URLS[source])
    except SyncAuthError as e:
        return {"status": "needs_login", "phase": "sources", "source": source,
                "message": str(e)}
    return {"status": "awaiting_login", "phase": "sources", "source": source,
            "message": (f"{backend.label} just opened at {_LOGIN_URLS[source]}, on a profile "
                        f"OPYT owns — the user's own windows and logins are untouched. Tell the "
                        f"user to log in there, then call `onboard` again."
                        + _SIGN_IN_NOTES.get(source, ""))}


# ── phase 2: consent ────────────────────────────────────────────────────────────────────────
#
# ONE question, TWO commitments, a split answer allowed, and BOTH cost shapes stated — or it is
# not consent to the recurring half.

_CONSENT_WORDS = {"both": (True, True), "backlog": (True, False),
                  "refresh": (False, True), "none": (False, False)}

# Every `_follow_consent_with_a_worker` status that means SOMETHING WILL CLAIM THE SCHEDULED WORK
# — a fresh install, a rewritten one, one already loaded, and the hosted box's shared resident.
# Kept beside `_CONSENT_WORDS` because it answers the same question from the other side: what the
# user agreed to, and whether anything is actually keeping it.
_WORKER_RUNNING = ("INSTALLED", "UPDATED", "ALREADY_RUNNING", "RESIDENT")


# What each platform's backlog rail imports, in the words the prompt uses. `_BACKLOG_RAILS` is
# the one place the two are paired, so a third saved-content platform is an entry here plus a
# rail, not an edit to the copy in three places.
_BACKLOG_RAILS = {"x": ("your X bookmarks", "bookmark_catchup"),
                  "substack": ("your Substack saved posts", "substack_saved_catchup")}


def _phrase(parts: list[str]) -> str:
    return " and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _consent_prompt(state: dict) -> dict:
    """The one question, and how many of its two commitments apply to this user.

    Commitment 1 is per-platform now that both halves have a rail: X bookmarks via
    `bookmark_catchup`, Substack saved posts via `substack_saved_catchup`. Both are the same act
    — content the user personally saved — and both are one-time and irreversible, so they are one
    commitment with a list, not two questions. It is stated whenever at least one of them is
    connected; a user with neither is asked ONE question, not two.

    ⚠️ The prompt states NO SIZE and NO PRICE for the import, and that is deliberate. The X
    backlog was measured once at ~1,080 items and $0.105 of VLM; a Substack saved list was
    measured once, on one account, at THREE posts. Neither number describes anybody else, and a
    prompt that quoted either would be selling a guess as a measurement. What is honest, and what
    it says, is the shape: the size of the import is how much you have saved.

    All four words stay accepted regardless of what the prompt offered. `backlog` writes BOTH
    rails' markers — the prompt says so in the sentence about connecting the other platform later
    — so a platform connected after this question runs the import the user already agreed to
    rather than needing an answer the phase machine can no longer ask for.
    """
    connected = [p for p in _BACKLOG_RAILS if state["sources"][p]]
    others = [p for p in _BACKLOG_RAILS if p not in connected]

    if connected:
        saved = _phrase([_BACKLOG_RAILS[p][0] for p in connected])
        later = ("" if not others else
                 " Saying yes also covers "
                 + _phrase([_BACKLOG_RAILS[p][0] for p in others])
                 + " if you connect that later, without asking again.")
        commitments = (
            f"1. **Import the posts you saved yourself, now** — ONE-TIME. That means {saved}. "
            f"It runs as soon as you say yes and walks each list end to end, so the size of the "
            f"import is how much you have saved. Reading the lists is free; what costs is "
            f"classifying and embedding what they land.{later}\n"
            "2. **Keep your Oracles current** — RECURRING, forever. It re-pulls each Oracle's "
            "sources on a schedule, so what it costs follows how many people you track and how "
            "much they publish.\n\n"
            "⚠️ These are not symmetrical, and you should know which is which before you "
            "answer. The recurring half can be switched off later (`consent=backlog` or "
            "`consent=none` revokes it). The one-time import CANNOT be turned off once it has "
            "run — there is no revoke for it, and once you have content in the store it counts "
            "as consented from then on. It is one-time and bounded, so the blast radius is "
            "small, but the choice is not reversible the way the other one is.")
        head = ("One question, two separate commitments — answer them together or separately by "
                "calling `onboard` again with `consent=both | backlog | refresh | none`.\n\n")
    else:
        commitments = (
            "**Keep your Oracles current** — RECURRING, forever. It re-pulls each Oracle's "
            "sources on a schedule, so what it costs follows how many people you track and how "
            "much they publish. It can be switched off later — `consent=none` revokes it.\n\n"
            "(The other commitment OPYT asks about is importing a backlog of posts you saved "
            "yourself — X bookmarks or Substack saved posts. Neither is connected, so nothing "
            "would run. `consent=backlog` or `consent=both` still records a yes, and the import "
            "runs when you connect one.)")
        head = ("One question. Answer it by calling `onboard` again with "
                "`consent=refresh` to accept or `consent=none` to decline.\n\n")

    return {
        "status": "needs_consent", "phase": "consent",
        "answers": list(_CONSENT_WORDS),
        "message": head + commitments,
    }


def _queue_backlog(platform: str) -> bool:
    from pipeline.kb.rail_jobs import request_now
    return request_now(_BACKLOG_RAILS[platform][1])


def _backlog_consented(platform: str) -> bool:
    """Has this platform's saved-content import been consented to? Read through `_body_arms` so
    there is ONE reader per platform — the arms resolve their consent callable at call time
    precisely so a test's marker is honored, and a second copy here would be the drift that
    `bookmark_catchup.consented` warns about. Unknown platform reads as no."""
    return any(arm.consent() for arm in _body_arms() if arm.platform == platform)


def _queue_refresh() -> bool:
    from pipeline.kb.rail_jobs import request_now
    return request_now("oracle_refresh")


def _confirmed_oracles() -> int:
    """How many Oracles are confirmed right now. Fail-safe: unreadable store reads as zero,
    which SUPPRESSES the request — the safe direction, since the rail it queues costs money."""
    try:
        from pipeline.kb import oracles, schema
        conn = schema.connect()
        try:
            return len(oracles.confirmed_oracles(conn))
        finally:
            conn.close()
    except Exception:
        return 0


def _apply_consent(word: str, *, connected: dict) -> dict:
    """Write the markers, then queue a rail IFF that rail has work right now — each rail's own
    cadence alone would leave "import now" meaning nothing for up to six hours.

    ORDER IS LOAD-BEARING: `grant_consent()` writes the marker to disk BEFORE the job row exists.
    The rail child reads that marker for itself, and the resident worker can claim a due-now job
    within a second, so queueing first is a real race in which the child reads a marker its own
    trigger has not written yet and exits reporting no consent.

    Queued, not spawned: a durable row outlives this MCP session, and the worker's recorded exit
    code — not the fact that a `Popen` returned — is what says the pass happened.

    One `backlog` answer, one marker PER RAIL. The markers stay separate because the rails do:
    each reads its own before it spends, and `bookmark_catchup.consented` says why sharing one is
    wrong — opting into one loop must never silently opt you into another with a different
    request pattern. What the single answer buys is that the user is asked once about one act.

    Every queue is gated on the rail having work RIGHT NOW: a child that finds nothing to do is a
    no-op dressed as an action, and its recorded failure looks like a real one. A backlog rail
    walks one platform's saved list, so with that platform unconnected it has none — the marker
    is still written, which is what makes connecting it later run the import instead of needing a
    question the phase machine can no longer put."""
    from pipeline.kb import bookmark_catchup, oracle_refresh, substack_saved_catchup
    want_backlog, want_refresh = _CONSENT_WORDS[word]
    queued = []

    if want_backlog:
        for platform, granter in (("x", bookmark_catchup), ("substack", substack_saved_catchup)):
            granter.grant_consent()
            if connected.get(platform) and _queue_backlog(platform):
                queued.append(_BACKLOG_RAILS[platform][1])

    if want_refresh:
        oracle_refresh.grant_consent()
        if _confirmed_oracles() > 0 and _queue_refresh():
            queued.append("oracle_refresh")
    else:
        # A toggle should toggle both ways — this is the only way to opt back out from chat.
        oracle_refresh.revoke_consent()

    onboard_state.mark_asked()
    return {"granted": [k for k, v in (("backlog", want_backlog),
                                       ("refresh", want_refresh)) if v],
            "queued": queued,
            "worker": _follow_consent_with_a_worker(want_refresh)}


def _follow_consent_with_a_worker(want_refresh: bool) -> dict:
    """Install the resident worker when the user consents to recurring refresh — and remove it
    when they take that consent back.

    ⚠️ THE CONSENT AND THE WORKER ARE THE SAME DECISION, AND SPLITTING THEM MADE THE CONSENT A
    LIE. The user is asked to agree to "**Keep your Oracles current** — RECURRING, forever", they
    say yes, a marker is written, a job row is queued — and then nothing on the machine ever
    claims that row, because `rail_worker` is the only thing that does and it was left as a
    command the user had to find and run. `_start_saved_content` says this in as many words:
    "on a from-source install there is no worker, so the row sits with `started_at` NULL forever
    and the import the user consented to never happens at all." Measured twice: 2026-09-13 found
    two jobs unclaimed since 13:50, and a fresh onboarding on 2026-09-14 left `bookmark_catchup`
    and `substack_saved_catchup` queued at 15:58 and still unstarted an hour later.
    `install_worker.status` already carried the warning — "ASK THIS BEFORE TELLING A USER THAT
    WORK CONTINUES ON ITS OWN" — and the one place that could act on it did not.

    So there is nothing extra to ask. A person who agreed to work happening forever, on a
    schedule, without them present, has agreed to the process that does it; a second prompt would
    be asking twice for one decision, which is the same objection `footprint_enrichment` records
    against re-triggering a pull the user already authorised. The ONE-TIME import half
    (`consent=backlog`) installs nothing — it is bounded, it runs in-process, and it does not
    outlive the session.

    REVOKE UNINSTALLS. A resident process the user has withdrawn consent for is worse than one
    that was never installed, and `revoke_consent()` above is already the toggle's other half.
    Only removed when this call is the one that revoked; a `backlog`-only answer from someone who
    never consented to refresh finds nothing to remove and says so.

    macOS only, because `install_worker` is macOS only by ruling F1 — "an unverified resident
    service is worse than an honest absence". Elsewhere this reports `unsupported` and the caller
    tells the truth about it rather than promising a schedule nothing keeps.

    Fail-safe, and the direction is deliberate: ANY failure here is reported, never raised, and
    never rolls the consent back. The markers are the user's decision and they stand; what a
    failure costs is the automation, which the report then has to be honest about. `uvx_command`
    raising (no `uv` on the machine) is the expected form of that.
    """
    from opyt_core import install_worker
    from pipeline.kb import rail_jobs

    # NOT THIS PROCESS'S WORKER TO INSTALL. A hosted child queues into the box's shared control
    # database, and `opyt-worker.service` — a resident process that outlives every child — is what
    # claims those rows. `install_worker` cannot see it and never could: it is macOS-only by
    # ruling F1, so on that Ubuntu box `supported` is False and the caller warned every remote
    # user that nothing would act on the consent they had just given, on the one home where the
    # schedule is kept by something that was already running. Ask what claims the row.
    #
    # Nothing is installed or removed on this branch in either direction, and REVOKE STILL
    # REVOKES: `_apply_consent` writes or clears the refresh marker before it calls here, and the
    # rail reads that marker for itself on every pass.
    if rail_jobs.queue_is_shared():
        return {"status": "RESIDENT",
                "note": ("A resident worker on this host claims scheduled work, so there is no "
                         "per-user service to install or remove. Their choice stands in the "
                         "refresh marker the rail reads.")}

    try:
        state = install_worker.status()
        if not state["supported"]:
            return {"status": "unsupported", "ran": state["ran"],
                    "note": ("This platform has no approved resident worker, so scheduled "
                             "refreshes only run while something runs `opyt-worker` by hand.")}
        if not want_refresh:
            return install_worker.uninstall()
        if state["installed"] and state["loaded"]:
            return {"status": "ALREADY_RUNNING", "log": str(install_worker.PLIST_PATH)}
        return install_worker.install()
    except Exception as e:
        return {"status": "not_installed", "error": f"{type(e).__name__}: {e}",
                "note": ("Consent is recorded, but nothing on this machine will act on it yet — "
                         "say so rather than promising a schedule.")}


# ── phase 3: the free curation pull ─────────────────────────────────────────────────────────

def _spawn(target):
    """Start `target` on a daemon thread and RETURN the thread, so the caller can join it.

    An indirection with exactly one job: tests replace it with a synchronous call, because a real
    thread outliving a test's monkeypatching would hit the actual network after the stub is gone.
    Same shape, and the same reason, as `sitting_tools._spawn`. A stub returns None, which every
    caller here tolerates: since 2026-09-14 nobody joins these threads, so the returned handle is
    kept only so a test can see that something was started."""
    import threading
    t = threading.Thread(target=target, name="opyt-saved-content", daemon=True)
    t.start()
    return t


def _pull_x_bookmarks() -> None:
    """The X body arm, on the thread — the FREE pass, `enrich=False`.

    ⚠️ THIS USED TO BE THE FULL `sync_bookmarks`, thread fetches and VLM included, spawned as one
    background unit and never joined. That was neither arm: it put the whole metered backlog
    behind a thread nobody waited on, and `_ConvoFetcher` disables for the run on its first
    refusal — so on David's own store it wrote 150 atoms of 1,001 and the other 851 were not late,
    they were never written at all.

    Split in two on 2026-09-13 (R2). What runs HERE is what the free Bookmarks payload can answer
    on its own — 11 requests of a 500/15-min bucket, every atom written and searchable — and it is
    JOINED before `onboard` returns, so a finished setup means finished coverage. The metered
    upgrades are `pipeline/kb/enrichment.py`, the only genuinely background part of this, because
    it is the only part blocked by a rate meter.

    ITS OWN CONNECTION: SQLite connections are not shared across threads and the caller's is
    mid-request. Swallows everything — see `_start_saved_content`."""
    try:
        from pipeline.kb import schema
        from pipeline.kb.embed import get_kb_embedder
        from pipeline.kb.ingest_x import sync_bookmarks
        conn = schema.connect()
        try:
            sync_bookmarks(conn, get_kb_embedder(), enrich=False)
        finally:
            conn.close()
    except Exception:
        return
    # ⚠️ STARTED HERE, ON THIS THREAD, BECAUSE THIS IS WHERE THE ORDERING IS KNOWN. Enrichment
    # walks the SAME corpus this arm just wrote, and starting it against a half-written one spends
    # metered requests on bookmarks whose atoms do not exist yet. `onboard` used to guarantee that
    # by JOINING this thread — which is exactly what the client kept killing. The constraint was
    # never "the caller must wait"; it was "enrichment must start after the bodies land", and that
    # is a fact this thread knows and the caller does not.
    #
    # Only on the success path: an import that raised did not finish writing the corpus.
    _start_enrichment()


def _pull_substack_saved() -> None:
    """The Substack body arm, on the thread — the same split, one platform over: the saved LIST
    was already read in the blocking pass (`substack_saved_signals`), and this is the posts behind
    it, at full body, cleaned, chunked, embedded, images read by a VLM.

    Same connection rule and the same swallow as `_pull_x_bookmarks`."""
    try:
        from pipeline.kb import schema
        from pipeline.kb.embed import get_kb_embedder
        from pipeline.kb.ingest_curation import sync_substack_saved
        conn = schema.connect()
        try:
            sync_substack_saved(conn, get_kb_embedder())
        finally:
            conn.close()
    except Exception:
        pass


@dataclass(frozen=True)
class _BodyArm:
    """One platform's saved-content import: who consents to it, what it runs, what to say."""
    platform: str
    source: str                       # what the caller reports this arm as
    consent: Callable[[], bool]       # resolved at call time, so a test's marker is honored
    pull: Callable[[], None]
    message: str


def _body_arms() -> tuple[_BodyArm, ...]:
    """The arms, built at CALL time. The consent readers live on rail modules that tests
    monkeypatch, and a tuple built at import would bind the originals."""
    from pipeline.kb import bookmark_catchup, substack_saved_catchup

    return (
        _BodyArm("x", "x-bookmarks", bookmark_catchup.consented, _pull_x_bookmarks,
                 ("The bookmarks are being written now — ALL of them, not a window's worth — "
                  "and each one is searchable the moment it lands. ⚠️ Say they are ARRIVING, not "
                  "that they are in: this call did not wait for the import and nothing here "
                  "knows when it finished. Their thread context and the descriptions of any "
                  "images they carry fill in behind that, as Enrichment; that part is metered by "
                  "x.com and takes as many 15-minute windows as the backlog needs. Say nothing "
                  "about when Enrichment finishes, and nothing about the posts being unavailable "
                  "until it does.")),
        _BodyArm("substack", "substack-saved", substack_saved_catchup.consented,
                 _pull_substack_saved,
                 ("The saved Substack posts are being fetched now. The publications you saved "
                  "from already count toward the candidate list — this is the posts. Substack "
                  "rate-limits the reader endpoint and each post is fetched at full body, so "
                  "this is the slow one, and it may still be going long after setup reads as "
                  "done. That is normal and is not worth flagging as a problem. There is no "
                  "Enrichment pass behind it, because the bodies are not in the list payload the "
                  "way X's are.")),
    )


def _start_saved_content(platforms: set[str]) -> list[tuple[_BodyArm, object]]:
    """Start each consented platform's saved-content import on its own thread. `[]` if none did.

    SPAWNED HERE AND NOT JOINED (R7 overturned, 2026-09-14 — see `_report_saved_content`).
    Nothing between here and `screen` reads a saved-post atom, and the ordering that DID need the
    join — Enrichment starting against a finished corpus — moved onto this arm's own thread,
    which is the only place that fact is actually known.

    IN-PROCESS, which is the whole point. The consent step queues `bookmark_catchup` and
    `substack_saved_catchup` through `request_now`, and that only writes a row — the resident
    worker is the sole thing that ever claims it. On a from-source install there is no worker, so
    the row sits with `started_at` NULL forever and the import the user consented to never happens
    at all. This is the path that does not depend on a process the user may not have. The rails
    stay queued and remain the RECURRING answer; if a worker does exist, single-flight keeps them
    from overlapping.

    PER PLATFORM, and each arm gated on its OWN consent marker. Substack's is deliberately
    separate from X's — `substack_saved_catchup` explains why at length, and the short version is
    that a user who answered `backlog` to a prompt naming only X consented to X.

    Fail-safe in layers: a platform not in `platforms` or without consent is skipped, each thread
    body swallows anything its adapter raises, and a spawn failure drops that arm while the others
    and the curation pass still stand.
    """
    started: list[tuple[_BodyArm, object]] = []
    for arm in _body_arms():
        if arm.platform not in platforms or not arm.consent():
            continue
        try:
            started.append((arm, _spawn(arm.pull)))
        except Exception:
            continue
    return started


def _report_saved_content(started: list[tuple[_BodyArm, object]]) -> dict | None:
    """Say what was STARTED. None if nothing was.

    ⚠️ THIS USED TO JOIN, AND THE JOIN IS WHAT GOT THIS CALL KILLED. `_join_saved_content` waited
    for every spawned import with no timeout — correct under R1, and fatal under a client that
    cuts the call at 60 seconds. `onboard(consent='both')` was truncated at 17:33:32 on
    2026-09-14 with ~1,000 bookmarks still being written, and the user was told their setup
    request had timed out. A completeness guarantee that is real and unreachable is worth nothing.

    Nothing downstream needs the wait: nothing between here and `screen` reads a bookmark atom,
    and the store is resumable regardless — every atom written is durable and the next call picks
    up where this one reached. What DID need the wait was starting Enrichment against a finished
    corpus, and that moved onto the arm's own thread (`_BodyArm.after`), which is where the
    ordering actually belongs.

    The copy says "importing", not "imported", because that is what is true when this returns."""
    if not started:
        return None
    arms = [a for a, _t in started]
    return {"status": "importing", "sources": [a.source for a in arms],
            "message": ("Setup is done, and the saved posts are importing now — nothing is "
                        "waiting on them and nobody needs to sit here for them. "
                        + " ".join(a.message for a in arms)),
            "host_note": ("Say the import is UNDERWAY, never that it has finished — you do not "
                          "know that yet and nothing here will tell you. Do not offer to check, "
                          "do not wait for it, and do not describe it as slow or as a problem.")}


def _start_enrichment() -> dict | None:
    """Hand the metered upgrades to `enrichment`, once the bodies really are in the store.

    Called from the X bookmark arm's own thread, at the point that is true — see
    `_pull_x_bookmarks`. It used to be called by `onboard` after joining that thread, and the join
    is what the client kept killing.

    X only. §5 of the ruling has the three structural reasons Substack has no analogue: no rate
    meter exists there at all (so under R2 nothing there is ever background), the bodies are not
    in the list payload, and an Oracle's Substack IS their archive, so there is no cheap breadth
    arm to defer anything from.

    Fail-safe: a start that fails is dropped, and the `bookmark_catchup` rail remains the
    recurring answer regardless."""
    try:
        from pipeline.kb import enrichment
        return enrichment.start_background()
    except Exception:
        return None


def _start_footprint_enrichment() -> dict | None:
    """The OTHER Enrichment engine — the one that deepens Oracle timelines. Same name to the user
    (R6), different meter and different corpus; see `pipeline/kb/footprint_enrichment.py`.

    Started here as well as after a foreground ingest because an install that already has Oracles
    can reach this point with windows still owed from a previous session — a server that was
    killed mid-pass, most often. It returns `nothing_owed` without starting a thread when there is
    no such backlog, so the common first-run case costs one query.

    Fail-safe, the same as its sibling: a start that fails is reported as not started."""
    try:
        from pipeline.kb import footprint_enrichment
        out = footprint_enrichment.start_background()
        return out if out.get("status") != "nothing_owed" else None
    except Exception:
        return None


def _live_platforms(state: dict) -> set[str]:
    """The auto-discovery platforms with a session connected right now — the ones a saved-content
    import could run for at all."""
    return {p for p in _BACKLOG_RAILS if state["sources"].get(p)}


def _run_curation(platforms: set[str] | None = None) -> dict:
    """The four FREE people-only collectors: X Lists, following, likes, Substack subscriptions.

    This is `curation_catchup`, not `curation_pull` — the content-bearing arms of
    `curation_pull` cost money and time, so `onboard` opens the bookmark backlog's existing
    budgeted rail rather than carrying the pull itself.

    NEVER `curation_pull(tiered=True)` here: the ladder's gate reads the whole store's
    signalled-entity count, so on any established store it clears `sufficient_at` after Tier 1
    and permanently skips following and likes — the exact two collectors this needs.

    `force=True` bypasses the 6 h per-collector floor (meant for a background loop, not a fresh
    setup call) but not single-flight. It leaves a known Substack-saved-posts gap.

    SCOPED to `platforms` — the ones `onboard_state.derive` reports as `curation.pending`, i.e.
    connected and never read. The pass used to cover all four collectors every time, which was
    harmless only because it ran exactly once ever; now that a later connection can re-enter this
    phase, an unscoped `force=True` would re-walk the platform that already succeeded and spend
    its requests again for nothing.
    """
    from pipeline.kb.curation_catchup import run_curation_catchup
    return run_curation_catchup(force=True, platforms=platforms)


def _start_arm_a(platforms: set[str]) -> dict:
    """ARM A — the free signal walk, STARTED and not awaited.

    ⚠️ IT USED TO BLOCK, under R2's original reading: background ⟺ blocked by a rate meter, and
    this is not — it is 11 requests of a 500/15-min bucket across four op buckets (measured
    2026-09-13), so it cannot rate-limit. R2 gained a second clause on 2026-09-14 (*or unbounded
    in duration*) and this walk is the borderline case: ~30 seconds on one measured account, and
    linear in how many people the user follows.

    What settles it is not the duration but WHERE the duration was being spent. This ran on the
    call that RETURNS THE CONSENT PROMPT, so the user sat in front of a blank screen for half a
    minute and then was asked a question. Started instead, the same work happens behind the time
    they spend reading that question and deciding — which is the gain the 2026-09-13 ordering
    change was reaching for in the first place.

    THE SECOND REASON IT BLOCKED IS STILL SATISFIED, and it is the one that matters: the screen is
    SCORED on these signals — a `save` is the second distinct (signal_type, platform) pair that
    lifts a person over `screen.CORROBORATION_MIN`, so a candidate list built before the walk
    lands is not a thin list, it is a WRONG one with no symptom. But `screen` is several turns and
    one human decision away, and nothing between here and there reads a candidate. The ordering
    holds with room to spare; it simply is not this call's job to enforce it by waiting.

    Consent is granted by the caller BEFORE this starts, so the recurring rail can be queued
    without waiting for a status this no longer returns."""
    try:
        _spawn(lambda: _run_curation(platforms or None))
    except Exception as e:
        return {"status": "not_started", "error": f"{type(e).__name__}: {e}"}
    return {"status": "scoring",
            "host_note": ("The people this user already curates are being scored right now. It "
                          "is quick and it is free, and it will be finished long before "
                          "`oracle(action='screen')` needs it. Do not wait for it, do not "
                          "mention it as a step, and do not call `screen` in this same turn.")}


# ── phase 4: derived progress, and the handoff ──────────────────────────────────────────────

def _atom_count() -> int | None:
    """How many atoms the store holds, or None when it cannot be read. None routes the handoff
    to the GENERIC done-branch: "your store is empty" is a claim about the user's data, and a
    broken read must never be the thing that makes it."""
    try:
        from pipeline.kb import schema
        conn = schema.connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
        finally:
            conn.close()
    except Exception:
        return None


def _candidate_count() -> int:
    """How many people the free pull surfaced for screening. Fail-safe: unreadable reads as 0,
    which understates progress rather than inventing it."""
    try:
        from pipeline.kb import schema, screen
        conn = schema.connect()
        try:
            return len(screen.rank_candidates(conn))
        finally:
            conn.close()
    except Exception:
        return 0


_CONSENT_KEY = {"x": "bookmark", "substack": "substack_saved"}


def _unoffered_backlog(state: dict) -> str:
    """One line offering an import the consent question never actually put, or "".

    The consent phase is asked ONCE — `onboard_consent_asked` closes it and `derive` never
    reopens it — so a platform that gains a rail AFTER a user has answered has no way back to the
    question. That is not hypothetical: every user who answered before `substack_saved_catchup`
    existed answered a prompt that named X bookmarks and nothing else, and using that answer for
    Substack would be consent obtained for a different question. They keep the answer they gave;
    this tells them the other import exists and how to ask for it.

    Silent once the marker is written, so it never nags a user who already said yes or no in a
    prompt that did name the platform.

    Keyed on the MANAGED-session probe, and it under-offers: a user logged into Substack in their
    normal Chrome reads as not-connected here, while `sources.substack.readable` says their
    collector would work fine. That is the accepted cost of not reading the cookie jar before
    consent (`has_managed_substack_session` records the same trade from the other side). Keying
    on `readable()` instead is not the fix — it is True on EVERY local home, so the offer would
    reach users who have never touched Substack. `onboard(consent='backlog')` is always accepted,
    so the under-offered user can still ask; the over-offered one could only be told to ignore it.
    """
    pending = [p for p in _BACKLOG_RAILS
               if state["sources"][p] and not state["consent"][_CONSENT_KEY[p]]]
    if not pending:
        return ""
    what = _phrase([_BACKLOG_RAILS[p][0] for p in pending])
    return (f"\n\nOPYT has not imported {what}. That is a one-time, irreversible import of "
            f"content you saved yourself, and the consent question you answered did not offer "
            f"it. `onboard(consent='backlog')` runs it; ignoring this leaves it undone.")


def _still_open(state: dict) -> str:
    """Every way into OPYT this user has not connected, as a bulleted block.

    ⚠️ THE PROMISE THIS KEEPS. `_phase_sources` tells the user the four roots are not exclusive
    — "answering one never closes the others" — and then `derive()` closes the question: ONE
    connected collector sets `sources.ok`, so the sources phase never returns and `onboard`
    never raises the other roots again. Until 2026-09-09 only the no-collector branch of
    `_handoff` re-offered them, so a user who connected X and got a screening list was never
    told Substack existed. That contradiction, not the host's single-select rendering, is what
    made the second root easy to never come back to.

    Only the LOGIN roots can be reported as taken or not taken. `research` and `blog` leave an
    `oracles` or `watchlist` row, and a screened X candidate leaves the same row, so there is no
    way to tell a named root from a screened one. Both naming paths stay open forever anyway,
    so they are always listed.
    """
    lines = _login_roots_open(state)
    lines += [
        "• **Topics they are into** — the user names interests and YOU fill the store: "
        "web-search the landscape, show them the strong finds, and save the ones they pick "
        "(`hopper`) — nothing from a web search is added without their say-so. Standing "
        "watches are NOT an onboarding move; they come later, once there is material the "
        "user has actually read.",
        "• **Anyone, by name** — the user gives a name in plain words; YOU web-search for "
        "their site, X handle, ORCID or OpenAlex page and pass what you find to "
        "`oracle(action='confirm', add_handles=['https://…'])`. Never ask the user for a "
        "URL or an id.",
    ]
    return "\n".join(lines)


def _login_roots_open(state: dict) -> list[str]:
    """The login roots this home has not connected, as bullets. Split out of `_still_open` so
    the blind-store handoff can re-offer the logins WITHOUT the naming bullets — there, the
    naming paths ARE the question being put, and listing them twice reads as a form."""
    return [f"• **{s.title()}** — `onboard(source='{s}')`, which finds {_ROOT_DISCOVERS[s]}."
            for s in _LOGIN_URLS if not state["sources"][s]]


def _what_opyt_can_do() -> str:
    """One sentence saying OPYT reads, not only collects. Everything specific waits for data.

    ⚠️ THE GAP THIS CLOSES. Every branch of `_handoff` returns `next_tool: "oracle"`, and its
    only list — `_still_open` — enumerates more ways to FEED OPYT. So the end of setup talked
    exclusively about input, and a user who had just finished it had been told about one of the
    nine things this surface does.

    SHRUNK TO A SENTENCE ON 2026-09-12, which is what the first version of this said should
    happen: it shipped five bulleted capabilities as a placeholder and its own docstring promised
    to shrink "when the grounded one lands, rather than sit beside it". The grounded one landed
    (`opyt_core/suggest`, reached from `oracle(action='ingest')`), and for one turn the two sat
    beside each other — a general list of tools here, a measured choice of directions one call
    later. That is the duplication the brevity rule exists to prevent, arrived at by two sessions
    that could not see each other.

    WHY ANYTHING SURVIVES HERE AT ALL. At this moment there is genuinely nothing to measure:
    `_candidate_count` counts people waiting to be SCREENED, and no atom exists until the first
    ingest finishes. So the honest content is exactly one fact — that the collecting is in
    service of reading — and the user reaches the specifics one call later, against a store that
    can be measured. Naming a tool here would be guessing which one fits a corpus that does not
    exist yet, which is the mistake the fixed menu made at the other end.
    """
    return ("\n\nSay ONCE, in one sentence, that collecting is the setup and not the point — "
            "OPYT is for reading a whole subject end to end, and the first ingest is what makes "
            "that possible. Name no tool: nothing is in the store yet, so there is nothing to "
            "recommend against.")


def _handoff(state: dict, **extra) -> dict:
    """What landed, and what to call next. No new state — everything here comes from `derive()`
    plus a candidate count.

    The done copy branches on WHICH ROOT the user took, because one sentence cannot describe
    both. A user who connected X has a screening list and should be sent to `oracle` with the
    first batch bounded — `oracle(action='ingest')` is a synchronous foreground loop with no
    pick cap or time budget. A user who named blogs, watched topics, or skipped has NO
    collector, so "the free collectors surfaced 0 people" would report a pass that never ran.

    The last branch splits again — on MATERIAL, not on whether anything was named (RULED
    2026-09-12, after a store carrying six watches and zero atoms was handed the generic
    open-roots list). The flow's question is "is there anything to play with": oracles will
    ingest into reading material, so a user who named one gets the open-roots list — but
    watchlist rows alone stage an inbox, not a library, and a user whose store holds no atoms
    and no oracles gets the direct question instead — topics first, names second — with the
    finding assigned to the host in both cases. An unreadable atom count routes to the GENERIC
    branch: "your store is empty" is a claim, and a broken read must not make it.
    """
    n = _candidate_count()
    done = state["phase"] == "done"
    collectors = state["curation"]["applicable"]

    if not done:
        message = (f"Setup is not finished yet — currently at the `{state['phase']}` step. "
                   f"Call `onboard` again to continue.")
    elif n:
        message = (
            # "Screened" is OPYT's word, not the user's, and it reached them unchanged on
            # 2026-09-15 — "829 people are waiting to be screened" — where it reads as a process
            # they are behind on rather than a list they get to choose from. The host says what
            # it is handed, so the plain phrasing belongs here, not in a note asking for one.
            f"Setup is complete. {n} people are ready for the user to choose from.\n\n"
            f"Next: call `oracle` to pick who to trust. Confirm THREE TO FIVE people first and "
            f"ingest those before adding more — the first ingest runs in the foreground and "
            f"walks every person you confirm, so a large first batch means a long wait with no "
            f"partial result.\n\nStill open, and they stay open:\n" + _still_open(state))
    elif collectors:
        connected = [s for s in _LOGIN_URLS if state["sources"][s]]
        reconnect = " or ".join(f"`onboard(source='{s}')`" for s in connected)
        message = (
            f"Setup is complete, but the collectors found nobody to screen. That usually means "
            f"the connected session went stale — reconnect with {reconnect} — or that this "
            f"account follows and subscribes to nobody yet.\n\nEither way, these all work "
            f"now:\n" + _still_open(state))
    elif state["sources"].get("oracles") or _atom_count() != 0:
        message = (
            "Setup is complete. Nothing is collected automatically, because no source with "
            "automatic discovery is connected — that is the expected state for a blog, "
            "newsletter or research user, not a failure.\n\n"
            "Every way to give OPYT something to read stays available:\n" + _still_open(state))
    else:
        # The BLIND store: setup was skipped through, so nobody is named, nothing is watched,
        # and no session exists — there is no signal to measure and nothing for the tour to
        # say. The one honest move left is to ask the user directly, and the asking has rules:
        # topics lead because "what are you into" is easier to answer than "name five writers",
        # and whichever way they answer, the FINDING is the host's job — the user's vocabulary
        # here is topics and names, never URLs or identifiers.
        logins = _login_roots_open(state)
        opener = (
            "Setup is complete, and standing watches from earlier are staging candidates — "
            "but the store itself holds no material yet: nothing to search, nothing to read."
            if state["sources"].get("watchlist") else
            "Setup is complete, but OPYT has nothing to read and no signal to guess from — "
            "no session, nobody named, nothing watched.")
        message = (
            opener + "\n\n"
            "So ask the user directly. ONE question, two ways to answer it:\n"
            "• **What topics are you into?** — lead with this; it is the easier question to "
            "answer. Then YOU fill the store: web-search the landscape on what they name, "
            "SHOW them the strong finds, and save the ones they pick — nothing from a web "
            "search enters the store without their say-so. Recurring authors the same way: "
            "offer them, and confirm the accepted ones with "
            "`oracle(action='confirm', add_handles=['https://…'])`. The user should have "
            "material to explore before anything else gets set up. Do NOT create standing "
            "watches here: a watch is a commitment whose wording the user should shape, and "
            "that conversation belongs later, after they have played with what is in the "
            "store.\n"
            "• **Or: who do you already read?** — names, in plain words. YOU web-search for "
            "each person's site, X handle, ORCID or OpenAlex page and pass the URL you "
            "find. Never ask the user for a link or an identifier — the finding is your "
            "job, not theirs. A user with no names is not stuck: the topics route above "
            "does the finding, and its survey's recurring authors are the people to bring "
            "back as candidates to confirm."
            + ("\n\nThe login roots stay open too:\n" + "\n".join(logins) if logins else ""))

    return {
        "status": "ok" if done else "in_progress",
        "phase": state["phase"],
        "candidates": n,
        "next_tool": "oracle",
        "message": (message + _unoffered_backlog(state) + _what_opyt_can_do()
                    if done else message),
        **extra,
    }
