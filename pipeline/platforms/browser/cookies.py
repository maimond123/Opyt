"""The managed local-session cookie reader. A source that needs a logged-in session reads exactly
one OPYT-managed profile here (`read_opyt_cookies`), created by `login.py`'s guided login; the user's own
browser profiles are never scanned for a session. When no managed session exists, or more than
one, the read refuses with an actionable `SyncAuthError` rather than guessing.

No Keychain dialog: this module never asks macOS for a browser's Safe Storage key. Detection is
a row-presence check against a copy of the profile's cookie SQLite; the one chosen profile is then
decrypted by the browser itself, launched headless over CDP (`cdp.py`) against a copied DB.

The backend registry also lists the user's installed Chromium browsers, because guided login
opens one of them on OPYT's own profile.

macOS-focused: the Chromium profile roots and `.app` paths below are macOS paths.

One exception comes first: a session handed in through the environment (`env_session`), for a
machine with no OPYT profile to read, such as an agent's own computer given a signed-in session.
When it is set, no profile is scanned.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.platforms.browser import cdp
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
    """One Chromium browser OPYT can launch: to read a managed session by transplant, or to
    open guided login on OPYT's own profile.

    `base` is the macOS Chromium profile root; `app` is the macOS .app bundle name OPYT
    launches."""
    key: str                  # "chrome","brave",… (+ OPYT_KEY_SUFFIX)
    label: str                # user-facing name
    base: Path                # Chromium profile root
    app: str                  # macOS .app bundle name

    def app_path(self) -> Path | None:
        """This browser's executable, or None if it is not installed where OPYT can launch
        it. Resolved per call rather than stored, so the registry stays a pure literal."""
        for d in _APP_DIRS:
            exe = d / f"{self.app}.app" / "Contents" / "MacOS" / self.app
            if exe.exists():
                return exe
        return None


# Priority order — guided login opens the first one installed. Chrome first (most common).
_BACKENDS: list[BrowserBackend] = [
    BrowserBackend("chrome",  "Chrome",  _APP_SUPPORT / "Google/Chrome",               "Google Chrome"),
    BrowserBackend("brave",   "Brave",   _APP_SUPPORT / "BraveSoftware/Brave-Browser", "Brave Browser"),
    BrowserBackend("edge",    "Edge",    _APP_SUPPORT / "Microsoft Edge",              "Microsoft Edge"),
    BrowserBackend("vivaldi", "Vivaldi", _APP_SUPPORT / "Vivaldi",                     "Vivaldi"),
    BrowserBackend("opera",   "Opera",   _APP_SUPPORT / "com.operasoftware.Opera",     "Opera"),
]

_BACKEND_BY_KEY = {b.key: b for b in _BACKENDS}


# ── OPYT's own login profiles (created by guided login, read like any other) ─────

def opyt_session_root() -> Path:
    """Where a guided login parks the browser profile it created: one directory per browser
    key, each a real Chromium user-data-dir, so the same transplant reads them."""
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
        if b is None or not d.is_dir():
            continue
        out.append(replace(b, key=b.key + OPYT_KEY_SUFFIX,
                           label=f"Opyt login — {b.label}", base=d))
    return out


def installed_backends() -> list[BrowserBackend]:
    """The registry filtered to what's actually present on this machine, in priority
    order: the profile root and the browser executable exist (a read launches that
    executable). OPYT's own login profiles come last."""
    out = [b for b in _BACKENDS if b.base.exists() and b.app_path() is not None]
    return out + opyt_session_backends()


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


# ── The exact-domain rule, in the two forms the read needs ───────────────────────
#
# Both must stay in step. A host belongs to `domain` only when it is the domain or a
# dot-separated subdomain; a bare suffix match (`LIKE '%x.com'`) would also match adgrx.com.

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

    Row presence only — the value stays encrypted, so no Keychain dialog; the expensive read
    (`_read_chromium`) happens once, for the chosen profile."""
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
    """Delete every row outside `domains` from a copied cookie DB, so the browser OPYT
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

    Copy that profile's cookie DB into a throwaway user-data-dir, prune it to `domains`, and
    launch that browser headless against it; it decrypts as itself, with no dialog and no
    network. Launch the browser the DB came from (Safe Storage keys are per application), and
    delete the temp profile after every read."""
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
    """Read `domains` from one browser profile, merged into a single name→value dict.
    Returns (cookies, error): a decrypt/launch failure surfaces as error (with an empty
    dict); "logged out" surfaces as ({}, None); a hit surfaces as (cookies, None).
    Cookies are kept in-memory and never logged.

    The read goes through the browser itself and requires `cookie_file` (which profile)."""
    try:
        return _read_chromium(backend, domains, cookie_file), None
    except Exception as e:                   # browser missing, launch failed, DB unreadable
        return {}, e


def _classify_failure(exc: Exception) -> str:
    """Turn a raw read exception into a kind ∈ {browser_launch, other}, so a blocked read gets
    an actionable error instead of a generic 'not logged in'."""
    if isinstance(exc, cdp.CDPError):
        # A transplant read: the session is there, the browser would not start or answer.
        return "browser_launch"
    return "other"


def list_logged_in(domains, auth_cookie: str, *, browsers=None):
    """Scan installed backends (priority order) for sessions carrying `auth_cookie`.

    Priority-first / stop-on-match: return as soon as one browser yields a session, after
    enumerating all its profiles so a multi-profile ambiguity is surfaced. A blocked probe is
    recorded as a failure, not a match.

    Candidates carry no cookie values (detection is a row-presence check);
    `_read_selected_candidate` does the one real read.

    Returns (candidates, failures):
      - candidate = {browser, profile, label, backend, cookie_file}
      - failure   = {browser, profile, backend, kind, detail}, kind ∈ {browser_launch, other}
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
        for name, cf in _chromium_cookie_files(backend.base):
            try:
                present = _chromium_has_cookie(cf, domains, auth_cookie)
            except Exception as e:
                failures.append({"browser": backend.key, "profile": name, "backend": backend,
                                 "kind": _classify_failure(e), "detail": str(e)})
                continue
            if present:
                candidates.append({"browser": backend.key, "profile": name,
                                   "label": backend.label,
                                   "backend": backend, "cookie_file": cf})
        if len(candidates) > before:
            break  # first browser with a session wins; don't touch (or launch) the rest
    return candidates, failures


def env_session(domains: list[str]) -> dict | None:
    """The session in `OPYT_COOKIES_<SITE>`, or None when it is unset.

    `<SITE>` is the first domain's name: `x.com` reads `OPYT_COOKIES_X`, `substack.com` reads
    `OPYT_COOKIES_SUBSTACK`. The value is a `Cookie:` header, `name=value; name=value`.
    """
    raw = os.environ.get("OPYT_COOKIES_" + domains[0].split(".")[0].upper(), "").strip()
    if not raw:
        return None
    pairs = (part.strip().split("=", 1) for part in raw.split(";") if "=" in part)
    return {name.strip(): value.strip() for name, value in pairs}


def list_opyt_logged_in(domains, auth_cookie: str):
    """Find sessions only in OPYT-created browser profiles, or the one in the environment.

    `list_logged_in` stops after the first browser with a session. A managed read instead
    needs to detect every OPYT profile so it can reject ambiguity rather than choose one, so
    each scan is pinned to one key produced by `opyt_session_backends`; normal browser keys
    never enter it.
    """
    env = env_session(domains)
    if env is not None:
        return ([{"browser": "env", "profile": None, "label": "environment"}]
                if env.get(auth_cookie) else []), []
    candidates: list[dict] = []
    failures: list[dict] = []
    for backend in opyt_session_backends():
        found, blocked = list_logged_in(domains, auth_cookie, browsers=[backend.key])
        candidates.extend(found)
        failures.extend(blocked)
    return candidates, failures


def _worst_failure(failures: list[dict]) -> dict:
    """When every read was blocked, surface the most actionable kind first so the
    remediation the user sees is the one most likely to unblock them."""
    order = {"browser_launch": 0, "other": 1}
    return sorted(failures, key=lambda f: order.get(f["kind"], 2))[0]


def _read_selected_candidate(domains, chosen: dict, source: str) -> dict:
    """Read one detected session and turn a failed read into the typed user-facing error."""
    who = (f"{chosen['browser']}/{chosen['profile']}"
           if chosen.get("profile") else chosen["browser"])
    backend = chosen["backend"]
    cookies, err = _read_one(backend, domains, cookie_file=chosen["cookie_file"])
    if err is not None:
        raise SyncAuthError(remediation(_classify_failure(err), backend, source))
    if not cookies:
        # The row was there and the read came back empty: reporting "not logged in" would
        # be wrong, and returning {} would send an empty Cookie header and 401 far from here.
        raise SyncAuthError(
            f"Found a {source} session in {who}, but reading it back returned nothing. Open "
            f"{backend.label}, confirm you are still signed in to {source}, then re-run.")
    log(f"[cookies] using {source} session from {who} ({chosen['label']})")
    return cookies


def choose_managed_candidate(candidates: list[dict], failures: list[dict], source: str) -> dict:
    """The one managed session to read, or a SyncAuthError saying why there is none.

    One candidate is used. Several are refused as ambiguous rather than chosen between. None
    with failures surfaces the most actionable failure; none without failures means `source`
    was never connected."""
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        if failures:
            failure = _worst_failure(failures)
            raise SyncAuthError(remediation(failure["kind"], failure["backend"], source))
        raise SyncAuthError(
            f"No OPYT-managed {source} session found. Call onboard(source='{source.lower()}') "
            f"to connect {source}, then re-run.")
    raise SyncAuthError(
        f"Multiple OPYT-managed {source} sessions are active. Keep one session connected "
        f"and remove the others, then re-run.")


def read_opyt_cookies(domains, auth_cookie: str, *, source: str) -> dict:
    """Read one managed `source` session from exactly one OPYT-created browser profile, or the
    one in the environment."""
    if isinstance(domains, str):
        domains = [domains]
    env = env_session(domains)
    if env is not None:
        if not env.get(auth_cookie):
            # Refused here, so a half-pasted session never becomes a 401 far from this read.
            raise SyncAuthError(f"The {source} session in the environment has no {auth_cookie} "
                                f"cookie. Set the full Cookie header, then re-run.")
        log(f"[cookies] using {source} session from the environment")
        return env
    candidates, failures = list_opyt_logged_in(domains, auth_cookie)
    chosen = choose_managed_candidate(candidates, failures, source)
    return _read_selected_candidate(domains, chosen, source)


def remediation(kind: str, backend: BrowserBackend | None, source: str) -> str:
    """Actionable failure copy per kind — what the user should do to unblock `source`."""
    label = backend.label if backend else "browser"
    if kind == "browser_launch":
        return (f"Opyt found your {source} session in {label} but could not start {label} to "
                f"read it. Opyt launches the browser itself (that is what keeps macOS from "
                f"asking for your Keychain), so it needs {label} installed in /Applications. "
                f"Check that, then re-run.")
    return (f"Couldn't read your {label} session for {source}. Call "
            f"onboard(source='{source.lower()}') to connect {source} again, then re-run.")


def backend_for(key: str) -> BrowserBackend | None:
    """The registry entry for a candidate/failure's `browser` key, including the dynamic
    `<browser>@opyt` entries for profiles a guided login created."""
    if key.endswith(OPYT_KEY_SUFFIX):
        return next((b for b in opyt_session_backends() if b.key == key), None)
    return _BACKEND_BY_KEY.get(key)


def build_cookie_header(cookies: dict) -> str:
    """The `Cookie:` request header value from a name→value cookie dict."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())
