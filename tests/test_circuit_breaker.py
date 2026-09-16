import time

from pipeline.circuit_breaker import CircuitBreaker


def _tripped(tmp_path, *, cooldown: float, service="test"):
    b = CircuitBreaker(service, threshold=1, cooldown=cooldown, db_path=tmp_path / "opyt.db")
    b.record_failure("outage")
    return b


def test_only_one_caller_claims_the_half_open_trial(tmp_path):
    """A recovered breaker admits one probe, not every concurrent caller.

    A REAL cooldown, because the cooldown is also the abandonment window (see
    `test_an_abandoned_trial_is_re_granted`): a breaker configured to back off for zero seconds
    is asking to be retried immediately, and has no window in which to hold a claim. Every
    production caller uses 900 s or longer."""
    breaker = _tripped(tmp_path, cooldown=0.0)
    breaker.cooldown = 900.0                     # tripped instantly, then a real back-off window

    assert breaker.allow() is False              # still inside the cooldown
    breaker.cooldown = 0.0
    assert breaker.allow() is True               # the one trial
    breaker.cooldown = 900.0
    assert breaker.allow() is False              # a concurrent caller gets nothing


def test_peek_never_claims_the_trial(tmp_path):
    """The bug this method exists for. `allow` CLAIMS the half-open trial, so a pre-check that
    asked with it consumed the very trial the real call was about to make — the call then found
    HALF_OPEN, refused, and recorded no outcome. Measured on the live store 2026-09-08:
    `api.openalex.org` stranded 257 hours and `export.arxiv.org` 30, against a 15-minute
    cooldown, with both paper adapters silently dead the whole time."""
    breaker = _tripped(tmp_path, cooldown=0.0)

    assert breaker.peek() is True                # a trial is available…
    assert breaker.peek() is True                # …and asking twice does not use it up
    assert breaker.allow() is True               # …so the real call still gets it


def test_peek_is_false_while_the_breaker_is_backing_off(tmp_path):
    assert _tripped(tmp_path, cooldown=900.0).peek() is False


def test_an_abandoned_trial_is_re_granted(tmp_path):
    """Without this, HALF_OPEN is ABSORBING: `call` refuses it, so no outcome is ever recorded,
    so it never leaves. A process that dies between claiming the trial and recording its outcome
    would kill the service permanently."""
    breaker = _tripped(tmp_path, cooldown=0.0)
    assert breaker.allow() is True               # claimed, now HALF_OPEN
    breaker.cooldown = 0.05
    assert breaker.allow() is False              # still within the window

    time.sleep(0.06)                             # the claimant never came back

    assert breaker.allow() is True               # re-granted rather than stranded


def test_retry_after_reports_a_pending_trial_as_unavailable(tmp_path):
    """A claimed trial whose outcome is still pending is exactly as unavailable as an open
    breaker. Reporting 0 for it told every caller to retry immediately into a refusal."""
    breaker = _tripped(tmp_path, cooldown=0.0)
    breaker.allow()                              # → HALF_OPEN
    breaker.cooldown = 900.0

    assert breaker.retry_after() > 0


def test_a_success_closes_the_breaker(tmp_path):
    breaker = _tripped(tmp_path, cooldown=0.0)

    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.peek() is True
    assert breaker.allow() is True               # CLOSED — no trial to claim


# ── the breaker must not take opyt.db's write lock ──────────────────────────────

def test_breaker_state_never_lands_in_the_knowledge_base(tmp_path, monkeypatch):
    """MEASURED on the box 2026-09-16: 12 `database is locked` failures during one connect, 6 of
    them OCR reads that were then discarded, because every `allow()`/`record_*` takes
    `BEGIN IMMEDIATE` — an EXCLUSIVE write lock — and it was taking it on `opyt.db`, the file a
    concurrent ingest holds in far longer transactions than this module's 5s `busy_timeout`.

    Asserting on the PATH rather than on a timing race, because the race is what the separation
    removes: if breaker state ever lands back in `opyt.db`, the contention returns silently and
    only a two-process box would show it."""
    import sqlite3

    from opyt_core.paths import opyt_db
    from pipeline import circuit_breaker as cb

    home = tmp_path / "home"
    monkeypatch.setenv("OPYT_HOME", str(home))

    breaker = cb.CircuitBreaker("openrouter", threshold=1, cooldown=900.0)
    breaker.record_failure("outage")

    assert cb.breaker_db_path() == home / "circuit_breaker.db"
    assert cb.breaker_db_path().exists()
    # The knowledge base must be untouched — not merely free of the table, but never created.
    assert not opyt_db().exists()

    # The state really is in the new file, so this is separation and not a silent no-op.
    with sqlite3.connect(cb.breaker_db_path()) as conn:
        assert conn.execute(
            "SELECT state FROM circuit_breaker WHERE service='openrouter'").fetchone()[0] == "open"


def test_the_breaker_creates_a_home_that_does_not_exist_yet(tmp_path, monkeypatch):
    """`sqlite3.connect` raises on a missing PARENT, not a missing file. Riding on `opyt.db` hid
    that — something else always made the home first. Its own file has no such guarantor, and a
    first call on a fresh home would otherwise raise where it used to work. Same trap `0e8b5438`
    fixed in the profile lease."""
    from pipeline import circuit_breaker as cb

    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "never-created"))
    assert cb.CircuitBreaker("openrouter").allow() is True


def test_the_status_snapshot_reads_the_file_the_breakers_write(tmp_path, monkeypatch):
    """`status()` resolves its own default path, so it is a SECOND place that can disagree about
    where state lives — and moving the breaker off `opyt.db` broke exactly this, invisibly:
    `status()` kept the old resolver, `readiness.provider_blocked` wraps it in
    `except Exception: return None`, and a hard NameError came back as the cheerful answer "no
    provider is blocked". Five tests failed two layers away with nothing pointing here.

    A fail-safe that silences a bug in the thing it guards is why this needs its own assertion
    rather than trusting the other tests to notice."""
    from pipeline import circuit_breaker as cb

    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    breaker = cb.CircuitBreaker("openrouter", threshold=1, cooldown=900.0)
    breaker.record_failure("HTTP 403: Key limit exceeded")

    seen = {row["service"]: row["state"] for row in cb.status()}
    assert seen.get("openrouter") == "open", "status() is reading a different database"
