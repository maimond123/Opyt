"""
pipeline/ingestion/browser_cookies.py

Shared local-session cookie reader for the browser-scrapes (X bookmarks, Claude chats,
Substack): one place owns browser/profile enumeration and cookie reads instead of each
source duplicating `browser_cookie3` glue.

Resolves two dimensions: browser (Chrome/Brave/Edge/Vivaldi/Opera read by OPYT itself,
plus Arc/Firefox/Safari delegated to browser_cookie3) and profile (each Chromium profile
is its own cookie sandbox, so only the Chromium family gets a profile picker). Scans
installed browsers priority-first and stops at the first one holding a session;
`$OPYT_BROWSER` / explicit args override auto-pick, and a blocked read raises an
actionable `SyncAuthError` rather than reading as "not logged in." This layer never
prompts or guesses silently — the interactive picker lives in the CLI/onboarding layer.

**How a Chromium session is read, and why no Keychain dialog appears.** Asking macOS for a
browser's `<Browser> Safe Storage` key from Python is what raises the native dialog, so this
module never asks. Detection is a row-PRESENCE check against a copy of the profile's cookie
SQLite — the value stays encrypted. The one chosen profile is then decrypted by the browser
itself: its cookie DB is copied into a throwaway user-data-dir and that same browser is
launched headless against it over CDP (`cdp.py`), which decrypts as itself. Measured basis:
docs/plans/2026-08-30-cdp-cookie-transplant-and-guided-login.md.

Arc, Firefox and Safari keep the `browser_cookie3` path: none of them speaks CDP. Firefox
reads its own store with no OS prompt; Safari's Full Disk Access grant and Arc's Keychain
dialog are gates this technique cannot remove, so `consent` still names them.

macOS-focused: the Chromium profile roots and `.app` paths below are macOS paths.
Windows/Linux ride browser_cookie3's own cross-platform fallback (base=None → delegate),
not first-class.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from pipeline.ingestion import cdp
from pipeline.ingestion.utils import log, SyncAuthError

_APP_SUPPORT = Path.home() / "Library" / "Application Support"

# Where a browser's executable lives on macOS. Both are real install locations, and both are
# resolved at call time — the distributability invariant forbids baking in a user's path.
_APP_DIRS = (Path("/Applications"), Path.home() / "Applications")

# Suffix that marks a profile root OPYT itself created via guided login, keeping its key
# distinct from the same browser's real profiles (`chrome` vs `chrome@opyt`).
OPYT_KEY_SUFFIX = "@opyt"


# ── Backend registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BrowserBackend:
    """One browser OPYT knows how to read a local session from.

    `base` is the macOS Chromium profile root OPYT enumerates for the profile picker and
    reads by transplant; None means "delegate to browser_cookie3" (Firefox's profiles.ini,
    Safari's Cookies.binarycookies, Arc's non-standard layout — bc3 handles multi-profile
    merge + paths itself).

    `app` is the macOS .app bundle name, set exactly on the backends OPYT launches itself,
    and `reader_name` is the browser_cookie3 attribute, set exactly on the delegated ones.
    The two are mutually exclusive on purpose: they are the two ways a store gets read, and
    `base is not None` picks between them.

    `consent` names the native gate a read may trip: Safari's Full Disk Access, Arc's
    Keychain dialog (bc3 asks macOS for its Safe Storage key), or none — a transplant read
    never touches the Keychain, and Firefox reads its own store."""
    key: str                  # "chrome","brave",…,"firefox","safari" (+ OPYT_KEY_SUFFIX)
    label: str                # user-facing name
    base: Path | None         # Chromium profile root → transplant; None → delegate
    app: str | None           # macOS .app bundle name, for the backends OPYT launches
    reader_name: str | None   # attribute on browser_cookie3, for the delegated backends
    consent: str              # "keychain" | "fda" | "none"

    def reader(self) -> Callable:
        return getattr(_import_bc3(), self.reader_name)

    def app_path(self) -> Path | None:
        """This browser's executable, or None if it is not installed where OPYT can launch
        it. Resolved per call rather than stored, so the registry stays a pure literal."""
        if not self.app:
            return None
        for d in _APP_DIRS:
            exe = d / f"{self.app}.app" / "Contents" / "MacOS" / self.app
            if exe.exists():
                return exe
        return None


# Priority order — auto-detect tries these top-to-bottom and a lone logged-in
# candidate wins. Chrome first (most common); Safari last (best-effort, needs FDA).
_BACKENDS: list[BrowserBackend] = [
    BrowserBackend("chrome",  "Chrome",  _APP_SUPPORT / "Google/Chrome",               "Google Chrome",  None, "none"),
    BrowserBackend("brave",   "Brave",   _APP_SUPPORT / "BraveSoftware/Brave-Browser", "Brave Browser",  None, "none"),
    BrowserBackend("edge",    "Edge",    _APP_SUPPORT / "Microsoft Edge",              "Microsoft Edge", None, "none"),
    BrowserBackend("vivaldi", "Vivaldi", _APP_SUPPORT / "Vivaldi",                     "Vivaldi",        None, "none"),
    BrowserBackend("opera",   "Opera",   _APP_SUPPORT / "com.operasoftware.Opera",     "Opera",          None, "none"),
    # Arc's on-disk layout (Arc/User Data) is less standard — delegate to bc3 rather
    # than risk enumerating the wrong root and silently missing a logged-in session.
    # That keeps Arc on the Keychain path the Chromium backends above left.
    BrowserBackend("arc",     "Arc",     None, None, "arc",     "keychain"),
    BrowserBackend("firefox", "Firefox", None, None, "firefox", "none"),
    BrowserBackend("safari",  "Safari",  None, None, "safari",  "fda"),
]

_BACKEND_BY_KEY = {b.key: b for b in _BACKENDS}


def _import_bc3():
    try:
        import browser_cookie3
        return browser_cookie3
    except ImportError as e:
        raise RuntimeError(
            "browser-cookie3 not installed (pip install browser-cookie3)"
        ) from e


# ── OPYT's own login profiles (created by guided_login, read like any other) ─────

def opyt_session_root() -> Path:
    """Where a guided login parks the browser profile it created: one directory per browser
    key, each a real Chromium user-data-dir. Keeping that shape is what lets those sessions
    be read by the SAME transplant as every other profile, instead of a second read path."""
    from opyt_core.paths import opyt_path
    return opyt_path("browser-sessions")


def opyt_session_backends() -> list[BrowserBackend]:
    """The registry entries for profile roots OPYT created itself, if any exist.

    Only transplant-capable browsers qualify: a guided login is readable precisely because
    OPYT can relaunch the browser that wrote it."""
    root = opyt_session_root()
    if not root.exists():
        return []
    out: list[BrowserBackend] = []
    for d in sorted(root.iterdir()):
        b = _BACKEND_BY_KEY.get(d.name)
        if b is None or b.app is None or not d.is_dir():
            continue
        out.append(replace(b, key=b.key + OPYT_KEY_SUFFIX,
                           label=f"Opyt login — {b.label}", base=d))
    return out


def installed_backends() -> list[BrowserBackend]:
    """The registry filtered to what's actually present on this machine, in priority
    order. Chromium: the profile root AND the browser executable exist (a read launches
    that executable, so a missing app is a backend that cannot work). Firefox/Safari/Arc:
    their default cookie store exists (best-effort — a missing path just drops the backend,
    never crashes).

    OPYT's own login profiles come LAST, so a session in one of the user's real browsers
    always wins and a guided login is never silently preferred over it."""
    out: list[BrowserBackend] = []
    for b in _BACKENDS:
        if b.base is not None:
            if b.base.exists() and b.app_path() is not None:
                out.append(b)
        elif _delegated_store_exists(b):
            out.append(b)
    return out + opyt_session_backends()


def _delegated_store_exists(backend: BrowserBackend) -> bool:
    """Best-effort presence check for a bc3-delegated backend (Firefox/Safari/Arc).
    Cheap heuristics on the default macOS store; on any doubt, include it (so we don't
    silently skip a browser the user is actually logged into)."""
    home = Path.home()
    if backend.key == "safari":
        return (home / "Library/Containers/com.apple.Safari/Data/Library/Cookies/"
                "Cookies.binarycookies").exists() or (
                home / "Library/Cookies/Cookies.binarycookies").exists()
    if backend.key == "firefox":
        return (home / "Library/Application Support/Firefox/Profiles").exists()
    if backend.key == "arc":
        return (home / "Library/Application Support/Arc").exists()
    return True


# ── Chromium profile enumeration (OPYT's, parameterized by base) ─────────────────

def _chromium_cookie_files(base: Path) -> list[tuple[str, Path]]:
    """Every Chromium profile's cookie DB under `base`, as (profile_dir, path). Newer
    stores under Network/Cookies, older under Cookies — take whichever exists."""
    out: list[tuple[str, Path]] = []
    if not base.exists():
        return out
    for pdir in [base / "Default"] + sorted(base.glob("Profile *")):
        for cf in (pdir / "Network" / "Cookies", pdir / "Cookies"):
            if cf.exists():
                out.append((pdir.name, cf))
                break
    return out


def _chromium_profile_labels(base: Path) -> dict:
    """Map profile dir → human label (name + email), from the browser's 'Local State'
    info cache, so the picker shows 'Profile 5 (Work — a@b.com)'."""
    try:
        cache = (json.loads((base / "Local State").read_text())
                 .get("profile", {}).get("info_cache", {}))
    except Exception:
        return {}
    labels = {}
    for dirname, info in cache.items():
        name = info.get("name") or dirname
        email = info.get("user_name")
        labels[dirname] = f"{name} — {email}" if email else name
    return labels


# ── The exact-domain rule, in the two forms the read needs ───────────────────────
#
# Both must stay in step. A bare suffix match (`host_key LIKE '%x.com'`) also matches
# adgrx.com — that mistake transplanted 22 unrelated ad-tracker rows on the first spike run.
# A host belongs to `domain` only when it IS the domain or is a dot-separated subdomain.

def _domain_sql(domains: list[str]) -> tuple[str, list]:
    """(SQL fragment, params) matching a Chromium `host_key` against any of `domains`."""
    parts, params = [], []
    for d in domains:
        parts.append("(host_key = ? OR host_key = ? OR host_key LIKE ?)")
        params += [d, "." + d, "%." + d]
    return "(" + " OR ".join(parts) + ")", params


def _domain_matches(host: str, domains: list[str]) -> bool:
    """The Python twin of `_domain_sql`, for cookies that arrive over CDP rather than SQL."""
    host = host.lstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


# ── Chromium: copy, detect by row presence, decrypt via the browser itself ───────

# A cookie written seconds ago can live only in the WAL, so the sidecars travel with the DB;
# copying `Cookies` alone can read as "not logged in" immediately after a login.
_COOKIE_SIDECARS = ("-wal", "-shm", "-journal")


def _copy_cookie_db(src: Path, dst: Path) -> None:
    """Copy a (possibly live) Chromium cookie DB and its sidecars to `dst`.

    Everything downstream works on the copy: a copy cannot be locked by a running browser,
    and pruning it cannot touch the user's real store."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    for suffix in _COOKIE_SIDECARS:
        side = src.with_name(src.name + suffix)
        if side.exists():
            shutil.copy2(side, dst.with_name(dst.name + suffix))


def _chromium_has_cookie(cookie_file: Path, domains: list[str], name: str) -> bool:
    """Does this profile hold a non-empty `name` cookie for any of `domains`?

    Row PRESENCE only — the value stays encrypted, so this never asks macOS for the
    browser's Safe Storage key and never raises a Keychain dialog. That is what makes the
    scan cheap enough to run across every profile: the expensive read (`_read_chromium`)
    happens once, for the profile that was actually chosen."""
    where, params = _domain_sql(domains)
    with tempfile.TemporaryDirectory(prefix="opyt-cookie-scan-") as td:
        db = Path(td) / "Cookies"
        _copy_cookie_db(cookie_file, db)
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                f"SELECT 1 FROM cookies WHERE name = ? AND {where} "
                f"AND (length(encrypted_value) > 0 OR length(value) > 0) LIMIT 1",
                [name] + params).fetchone()
        finally:
            conn.close()
    return row is not None


def _prune_to_domains(db: Path, domains: list[str]) -> None:
    """Delete every row outside `domains` from a COPIED cookie DB, so the browser OPYT
    launches only ever holds the session that was asked for. Leaves one clean file: the
    journal-mode switch folds any WAL in before the delete."""
    where, params = _domain_sql(domains)
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(f"DELETE FROM cookies WHERE NOT {where}", params)
        conn.commit()
    finally:
        conn.close()
    for suffix in _COOKIE_SIDECARS:
        db.with_name(db.name + suffix).unlink(missing_ok=True)


def _read_chromium(backend: BrowserBackend, domains: list[str], cookie_file: Path) -> dict:
    """Decrypt one Chromium profile's cookies by making the browser do it.

    Copy that profile's cookie DB into a throwaway user-data-dir, prune it to `domains`,
    and launch THAT browser headless against it. Chrome fetches its own Safe Storage key as
    itself — it is on that keychain item's ACL — so the values come back in plaintext over
    CDP with no dialog. `Storage.getCookies` is browser-scoped, so nothing is navigated to
    and the read never touches the network.

    Two invariants from the spike: launch the browser the DB came FROM (Safe Storage keys
    are per application — Brave cannot decrypt Chrome's rows), and delete the temp profile
    after every read, so the user's real browser stays the session's only home."""
    exe = backend.app_path()
    if exe is None:
        raise cdp.CDPError(
            f"{backend.label} is not installed where Opyt can launch it")

    # Mirror the source layout: a build reads its cookie DB back from wherever it wrote it
    # (Default/Cookies on older profiles, Default/Network/Cookies on newer ones).
    rel = ("Network", "Cookies") if cookie_file.parent.name == "Network" else ("Cookies",)
    data_dir = Path(tempfile.mkdtemp(prefix="opyt-transplant-"))
    try:
        _copy_cookie_db(cookie_file, data_dir.joinpath("Default", *rel))
        _prune_to_domains(data_dir.joinpath("Default", *rel), domains)
        with cdp.controlled_browser(exe, user_data_dir=data_dir, headless=True) as session:
            raw = cdp.get_cookies(session)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return {c["name"]: c["value"] for c in raw
            if _domain_matches(c.get("domain", ""), domains)}


# ── Reads + failure classification ───────────────────────────────────────────────

def _read_one(backend: BrowserBackend, domains: list[str], *, cookie_file: Path | None = None):
    """Read `domains` from one browser/profile, merged into a single name→value dict.
    Returns (cookies, error): a permission/decrypt/launch failure surfaces as error (with an
    empty dict); "logged out" surfaces as ({}, None); a hit surfaces as (cookies, None).
    Cookies are kept in-memory and never logged.

    Chromium reads go through the browser itself and REQUIRE `cookie_file` (which profile).
    Arc/Firefox/Safari go through browser_cookie3, which finds their stores itself."""
    if backend.base is not None:
        try:
            return _read_chromium(backend, domains, cookie_file), None
        except Exception as e:                   # browser missing, launch failed, DB unreadable
            return {}, e

    reader = backend.reader()
    merged: dict = {}
    last_err: Exception | None = None
    for domain in domains:
        try:
            jar = reader(domain_name=domain)
            for c in jar:
                merged[c.name] = c.value
        except Exception as e:  # keychain denied, FDA needed, locked DB, decrypt error
            last_err = e
    if merged:
        return merged, None
    return {}, last_err


def _classify_failure(backend: BrowserBackend, exc: Exception) -> str:
    """Turn a raw read exception into a kind ∈ {browser_launch, keychain_denied, fda_needed,
    other}, driven mainly by the backend's consent gate (the strongest signal) with a message
    fallback. This is what lets a blocked read produce an ACTIONABLE error instead of
    collapsing into a generic 'not logged in'."""
    if isinstance(exc, cdp.CDPError):
        # A transplant read: the session is there, the browser would not start or answer.
        return "browser_launch"
    msg = str(exc).lower()
    permissionish = isinstance(exc, PermissionError) or "operation not permitted" in msg \
        or "permission denied" in msg
    if backend.consent == "fda":
        # Safari's store lives under a protected Containers path; a blocked read is
        # almost always Full Disk Access.
        if permissionish or "safari" in msg or "cookies.binarycookies" in msg:
            return "fda_needed"
    if backend.consent == "keychain":
        if any(t in msg for t in ("keychain", "security", "safe storage", "denied",
                                  "-128", "user canceled", "could not decrypt")):
            return "keychain_denied"
        if permissionish:
            return "keychain_denied"
    return "other"


def list_logged_in(domains, auth_cookie: str, *, browsers=None):
    """Scan installed backends (priority order) for sessions carrying `auth_cookie`.

    Priority-first / stop-on-match: iterate backends top-to-bottom and RETURN as soon
    as one browser yields a session. Within that winning browser, ALL profiles are still
    enumerated, so a multi-profile ambiguity is surfaced (the classic Chrome picker); across
    browsers we prefer higher priority and let $OPYT_BROWSER override. A browser whose read
    was BLOCKED is NOT a match — we record the failure and keep going, so an all-blocked
    scan can still produce an actionable remediation.

    Candidates carry NO cookie values. Detecting a Chromium session is a row-presence check
    against a copy of its cookie DB — no decryption, no Keychain, no browser launch — so
    there is nothing decrypted to hand back, and `read_cookies` does the one real read for
    the one profile that gets chosen. (Arc/Firefox/Safari do decrypt to detect, and pay a
    second cheap read at that point; a discovery API that returns live session secrets for
    some backends and not others would be a shape every caller has to branch on.)

    Returns (candidates, failures):
      - candidate = {browser, profile, label, consent, backend, cookie_file}
      - failure   = {browser, profile, kind, detail}, kind ∈
                    {browser_launch, keychain_denied, fda_needed, other}
    `browsers` (key or iterable of keys) restricts the scan to those browsers. Pure
    data: no prompts, no silent choice."""
    if isinstance(domains, str):
        domains = [domains]
    want = None
    if browsers is not None:
        want = {browsers} if isinstance(browsers, str) else set(browsers)

    candidates: list[dict] = []
    failures: list[dict] = []
    for backend in installed_backends():
        if want is not None and backend.key not in want:
            continue
        before = len(candidates)
        if backend.base is not None:
            # An OPYT-owned root has exactly one profile and it is OPYT's, so its own label
            # ("Opyt login — Chrome") is the honest one. Reading Local State there returns the
            # browser's default profile name — "Your Chrome" — which is indistinguishable from
            # the user's real Default in a picker.
            labels = ({} if backend.key.endswith(OPYT_KEY_SUFFIX)
                      else _chromium_profile_labels(backend.base))
            for name, cf in _chromium_cookie_files(backend.base):
                try:
                    present = _chromium_has_cookie(cf, domains, auth_cookie)
                except Exception as e:
                    failures.append({"browser": backend.key, "profile": name,
                                     "kind": _classify_failure(backend, e),
                                     "detail": str(e)})
                    continue
                if present:
                    candidates.append({"browser": backend.key, "profile": name,
                                       "label": labels.get(name, backend.label),
                                       "consent": backend.consent,
                                       "backend": backend, "cookie_file": cf})
        else:
            cookies, err = _read_one(backend, domains)
            if err is not None:
                failures.append({"browser": backend.key, "profile": None,
                                 "kind": _classify_failure(backend, err),
                                 "detail": str(err)})
            elif cookies.get(auth_cookie):
                candidates.append({"browser": backend.key, "profile": None,
                                   "label": backend.label, "consent": backend.consent,
                                   "backend": backend, "cookie_file": None})
        if len(candidates) > before:
            break  # first browser with a session wins; don't touch (or launch) the rest
    return candidates, failures


def _fmt_candidate(c: dict) -> str:
    """'chrome/Profile 3 (Work — a@b.com)' or 'safari (Safari)' — one option line."""
    who = f"{c['browser']}/{c['profile']}" if c.get("profile") else c["browser"]
    return f"{who} ({c['label']})"


def _resolved_browser_override(explicit: str | None) -> str | None:
    """The browser to pin, if any: explicit arg → $OPYT_BROWSER → settings.yaml
    cookies.browser (the last two resolved by opyt_core.config.cookie_browser). 'auto'/''
    mean no pin (auto-detect). Never raises (config is best-effort)."""
    if explicit and explicit.strip().lower() not in ("", "auto"):
        return explicit.strip().lower()
    try:
        from opyt_core.config import cookie_browser
        return cookie_browser()
    except Exception:
        return None


def _resolved_profile_override(env_var: str | None) -> str | None:
    """The profile to pin: $env_var (X_CHROME_PROFILE) → settings.yaml cookies.profile.
    Mirrors _resolved_browser_override so both halves of the choice resolve the same way.
    Never raises (config is best-effort)."""
    if env_var and (val := os.getenv(env_var)):
        return val.strip() or None
    try:
        from opyt_core.config import cookie_profile
        return cookie_profile()
    except Exception:
        return None


def pick(candidates: list[dict], failures: list[dict], *, browser: str | None = None,
         profile: str | None = None, env_var: str | None = None, source: str = "site") -> dict:
    """Choose one candidate. Resolution order: explicit browser/profile → $OPYT_BROWSER
    / $env_var → auto (lone candidate wins). Ambiguity or absence raises a SyncAuthError:
    multiple candidates → one that LISTS the options; zero candidates with a blocked read
    → an actionable remediation; zero candidates and nothing blocked → "not logged in"."""
    browser = _resolved_browser_override(browser)
    profile = profile or _resolved_profile_override(env_var)

    pool = candidates
    if browser:
        pool = [c for c in pool if c["browser"] == browser]
        if not pool:
            blocked = [f for f in failures if f["browser"] == browser]
            if blocked:
                raise SyncAuthError(
                    remediation(blocked[0]["kind"], backend_for(browser), source))
            raise SyncAuthError(
                f"No {source} login found in {browser}. Log into {source} in {browser}, "
                f"or unset $OPYT_BROWSER to try every browser.")

    if profile:
        chosen = next((c for c in pool if c["profile"] == profile), None)
        if not chosen:
            avail = ", ".join(_fmt_candidate(c) for c in pool) or "none"
            raise SyncAuthError(
                f"Profile {profile!r} is not logged into {source}. Logged-in: {avail}")
        return chosen

    if len(pool) == 1:
        return pool[0]

    if not pool:
        if failures:
            f = _worst_failure(failures)
            raise SyncAuthError(remediation(f["kind"], backend_for(f["browser"]), source))
        raise SyncAuthError(
            f"No {source} login found in any supported browser — log into {source} in "
            f"Chrome/Brave/Edge/Firefox/Safari, then re-run.")

    opts = "; ".join(_fmt_candidate(c) for c in pool)
    hint = "$OPYT_BROWSER" + (f" / ${env_var}" if env_var else "")
    raise SyncAuthError(
        f"Multiple browsers/profiles are logged into {source}: {opts}. Choose with {hint}.")


def _worst_failure(failures: list[dict]) -> dict:
    """When every read was blocked, surface the most actionable kind first so the
    remediation the user sees is the one most likely to unblock them."""
    order = {"browser_launch": 0, "keychain_denied": 1, "fda_needed": 2, "other": 3}
    return sorted(failures, key=lambda f: order.get(f["kind"], 4))[0]


def read_cookies(domains, auth_cookie: str, *, browser: str | None = None,
                 profile: str | None = None, env_var: str | None = None,
                 source: str = "site") -> dict:
    """Scan → pick → read the cookie dict for the chosen `source` session. The generic
    entry every cookie-scrape ingester calls; source-specific wrappers just fix the args.

    The scan finds sessions without decrypting any; this is where the ONE chosen session is
    actually read, so a machine with eight Chromium profiles launches a browser once.
    Raises SyncAuthError (not logged in / ambiguous / blocked) — never crashes."""
    if isinstance(domains, str):
        domains = [domains]
    scan = _resolved_browser_override(browser)
    candidates, failures = list_logged_in(domains, auth_cookie,
                                          browsers=[scan] if scan else None)
    chosen = pick(candidates, failures, browser=browser, profile=profile,
                  env_var=env_var, source=source)
    who = f"{chosen['browser']}/{chosen['profile']}" if chosen.get("profile") else chosen["browser"]
    backend = chosen["backend"]
    cookies, err = _read_one(backend, domains, cookie_file=chosen["cookie_file"])
    if err is not None:
        raise SyncAuthError(remediation(_classify_failure(backend, err), backend, source))
    if not cookies:
        # The row was there and the read came back empty: reporting "not logged in" would
        # be wrong, and returning {} would send an empty Cookie header and 401 far from here.
        raise SyncAuthError(
            f"Found a {source} session in {who}, but reading it back returned nothing. Open "
            f"{backend.label}, confirm you are still signed in to {source}, then re-run.")
    log(f"[cookies] using {source} session from {who} ({chosen['label']})")
    return cookies


# ── Consent copy (shown by the CLI / relayed by MCP results) ─────────────────────

def consent_prewarn(backend: BrowserBackend | None) -> str | None:
    """Copy to show BEFORE the first read of `backend`, when a native consent prompt is
    coming. Returned (not printed) so the caller decides the channel. None = no prompt,
    which is now the Chromium answer: a transplant read decrypts inside the browser and
    never asks macOS for anything."""
    if backend is None:
        return None
    if backend.consent == "keychain":
        return (f"macOS will ask for Keychain access so Opyt can read your {backend.label} "
                f"session locally — this stays on your machine. Click Allow (or Always Allow "
                f"to skip this next time).")
    if backend.consent == "fda":
        # The local-only reassurance rides BOTH branches: it is true of every backend, and this
        # copy runs exactly where a user is deciding whether to let an agent read their browser.
        return (f"{backend.label}'s cookies need Full Disk Access. If the read fails, grant it "
                f"in System Settings → Privacy & Security → Full Disk Access for your terminal "
                f"(or the Opyt app), then retry. Nothing leaves your machine.")
    return None


def prewarn_installed() -> str | None:
    """Pre-read consent copy for onboarding, where a native prompt CAN be pre-warned but
    the resolved backend isn't known until the scan. Returns the message for the
    highest-priority installed backend that trips a gate — Arc's Keychain or Safari's Full
    Disk Access. None when nothing installed needs consent, which is the common case now
    that Chrome/Brave/Edge/Vivaldi/Opera read without a dialog."""
    for b in installed_backends():
        msg = consent_prewarn(b)
        if msg:
            return msg
    return None


def remediation(kind: str, backend: BrowserBackend | None, source: str) -> str:
    """Actionable failure copy per kind — what the user should DO to unblock `source`."""
    label = backend.label if backend else "browser"
    if kind == "browser_launch":
        return (f"Opyt found your {source} session in {label} but could not start {label} to "
                f"read it. Opyt launches the browser itself (that is what keeps macOS from "
                f"asking for your Keychain), so it needs {label} installed in /Applications. "
                f"Check that, then re-run.")
    if kind == "fda_needed":
        return (f"Reading {label}'s cookies for {source} needs Full Disk Access. Grant it in "
                f"System Settings → Privacy & Security → Full Disk Access for your terminal "
                f"(or the Opyt app), then re-run. If it's already granted, quit and reopen "
                f"{label} so it flushes the current session.")
    if kind == "keychain_denied":
        return (f"macOS Keychain access was denied, so Opyt couldn't read your {label} session "
                f"for {source}. Re-run and click Allow (or Always Allow) when macOS asks to use "
                f"'{label} Safe Storage'. Nothing leaves your machine.")
    return (f"Couldn't read your {label} session for {source}. Log into {source} in a supported "
            f"browser (Chrome/Brave/Edge/Firefox/Safari), then re-run.")


def backend_for(key: str) -> BrowserBackend | None:
    """The registry entry for a candidate/failure's `browser` key, including the dynamic
    `<browser>@opyt` entries for profiles a guided login created."""
    if key.endswith(OPYT_KEY_SUFFIX):
        return next((b for b in opyt_session_backends() if b.key == key), None)
    return _BACKEND_BY_KEY.get(key)


def build_cookie_header(cookies: dict) -> str:
    """The `Cookie:` request header value from a name→value cookie dict."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())
