"""
tests/test_install_worker.py — the local half of the worker's deployment configuration.

The LaunchAgent is the one place a local install can weld itself to a folder that later moves,
so these tests hold the distributability invariant: the launcher is the published entry point
through an absolute `uvx`, never this checkout. `launchctl` itself is stubbed — a test must not
load a real agent into the developer's login session.
"""
from __future__ import annotations

import plistlib
import subprocess
import sys

import pytest

from opyt_core import install_client, install_worker


@pytest.fixture
def launchctl(monkeypatch) -> list[list[str]]:
    """Record every `launchctl` invocation instead of running one.

    Patched at `_launchctl`, not at `subprocess.run`. Two reasons, and the second one is why it
    moved: `install_worker.subprocess` IS the shared module, so the old seam replaced
    `subprocess.run` for everything in the process; and the repo-wide `_no_resident_service` guard
    now stubs `_launchctl` by default, so a fake one layer below it would never be reached — an
    autouse fixture silently shadowing a test's own patch is the inversion of the
    whoever-fakes-it-wins rule that both guards are built on. Same seam, the test's patch wins.
    """
    calls: list[list[str]] = []

    def fake(*argv):
        calls.append(["launchctl", *argv])
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    monkeypatch.setattr(install_worker, "_launchctl", fake)
    return calls


@pytest.fixture
def uvx(monkeypatch, tmp_path):
    """A resolvable `uvx`, so the plist can be built without uv on the test machine."""
    fake = tmp_path / "bin" / "uvx"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.touch()
    monkeypatch.setattr("opyt_core.install_client.shutil.which", lambda _: str(fake))
    return fake


def test_the_agent_launches_the_published_entry_point_not_this_checkout(
        uvx, monkeypatch, tmp_path):
    """`uvx` supplies the interpreter and the package, so no path in a PUBLISHED install's plist
    can go stale. That is still the shipped rule; what changed on 2026-09-14 is that it is now
    scoped to a published install rather than applied to every one, because a CHECKOUT pointed at
    `uvx --from opyt==<same version>` silently runs different code — see
    `test_a_checkout_installs_a_worker_that_runs_that_checkout`.

    The mock builds its launcher through `uvx_command` rather than writing an argv out by hand,
    because `agent_plist` carries `running_distribution`'s launcher VERBATIM: a hand-written pin
    here would assert a shape that function can no longer produce, and would have hidden the
    2026-09-14 merge that put `@latest` in `uvx_command` while this test still spelled a pin.
    The `@latest` rule itself is pinned one test down, on `uvx_command`, where it lives."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "running_distribution",
                        lambda: {"kind": "published", "root": "/site-packages",
                                 "launcher": install_client.uvx_command("opyt-worker")})

    plist = install_worker.agent_plist()

    assert plist["ProgramArguments"] == [str(uvx), "--from", "opyt@latest", "opyt-worker"]
    assert plist["Label"] == "com.useopyt.worker"
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True
    assert str(tmp_path / "home" / "worker.log") == plist["StandardErrorPath"]


def test_the_launcher_resolves_the_newest_build_on_every_start(uvx):
    """A pin — and a BARE requirement — both freeze a user on the build they installed.

    Measured 2026-09-14 on uv 0.12.13 with a warm cache: `--from opyt==<v>` and `--from opyt`
    both launch in 36 ms and never query the index, while `--from opyt@latest` takes 226 ms and
    does. So the fix for "my users never get my fixes" is not deleting the `==`; it is `@latest`
    specifically, and this test exists because deleting the `==` LOOKS like it works.
    """
    argv = install_client.uvx_command("opyt-mcp")

    assert argv[-2:] == ["opyt@latest", "opyt-mcp"]
    assert not any("==" in a for a in argv), "a pin freezes every installed user"
    assert "opyt" not in argv, "a bare requirement freezes them just as hard, but invisibly"


def test_the_agent_exposes_no_port(uvx, monkeypatch, tmp_path):
    """A resident scheduler that listened would be a second way into a user's corpus."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))

    assert "Sockets" not in install_worker.agent_plist()


def test_an_explicitly_chosen_data_home_is_pinned_into_the_agent(uvx, monkeypatch, tmp_path):
    """launchd hands an agent almost no environment, so a home this shell chose must travel."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "elsewhere"))

    assert install_worker.agent_plist()["EnvironmentVariables"] == {
        "OPYT_HOME": str(tmp_path / "elsewhere")}


def test_the_default_data_home_stays_derived_at_runtime(uvx, monkeypatch):
    monkeypatch.delenv("OPYT_HOME", raising=False)

    assert "EnvironmentVariables" not in install_worker.agent_plist()


def test_install_replaces_a_loaded_agent_rather_than_stacking_one(uvx, launchctl, monkeypatch,
                                                                  tmp_path):
    """Bootstrapping over a loaded label fails, so the unload has to come first every time."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")

    first = install_worker.install()
    second = install_worker.install()

    assert (first["status"], second["status"]) == ("INSTALLED", "UPDATED")
    assert [call[1] for call in launchctl] == ["bootout", "bootstrap"] * 2
    written = plistlib.loads((tmp_path / "agents" / "worker.plist").read_bytes())
    assert written == install_worker.agent_plist()


def test_uninstall_stops_the_agent_and_leaves_the_jobs_database(uvx, launchctl, monkeypatch,
                                                                tmp_path):
    """A reinstall resumes the same schedule; removing the queue would drop consented work."""
    home = tmp_path / "home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")
    install_worker.install()
    (home / "rail_jobs.db").write_text("durable")

    result = install_worker.uninstall()

    assert result["status"] == "REMOVED"
    assert launchctl[-1][1] == "bootout"
    assert not (tmp_path / "agents" / "worker.plist").exists()
    assert (home / "rail_jobs.db").read_text() == "durable"


def test_uninstalling_what_was_never_installed_is_not_an_error(launchctl, monkeypatch, tmp_path):
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "absent.plist")

    assert install_worker.uninstall()["status"] == "NOT_PRESENT"
    assert launchctl == []


def test_an_unapproved_platform_is_refused_rather_than_half_supported(
        launchctl, monkeypatch, tmp_path):
    """Ruling F1: no Linux user unit and no Windows task until one is verified.

    Refused means nothing happens — not a plist written somewhere launchd will never read.
    """
    monkeypatch.setattr(install_worker.sys, "platform", "linux")
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")

    assert install_worker.main([]) == 1
    assert not (tmp_path / "agents").exists()
    assert launchctl == []


# ── consent installs it; there is nothing extra to ask ─────────────────────────

def _follow(want_refresh, monkeypatch, *, status, install=None, uninstall=None):
    from mcp_server import onboard_tools
    monkeypatch.setattr(install_worker, "status", lambda: status)
    monkeypatch.setattr(install_worker, "install",
                        install or (lambda **kw: {"status": "INSTALLED"}))
    monkeypatch.setattr(install_worker, "uninstall",
                        uninstall or (lambda **kw: {"status": "REMOVED"}))
    return onboard_tools._follow_consent_with_a_worker(want_refresh)


_ABSENT = {"supported": True, "installed": False, "loaded": False, "ran": False}

# What `status()` reports on the hosted box: Linux, so no approved resident
# service of its OWN — which says nothing about the one systemd is already running.
_LINUX = {"supported": False, "installed": False, "loaded": False, "ran": False}


def test_consenting_to_recurring_refresh_installs_the_worker(monkeypatch):
    """⚠️ THE CONSENT AND THE WORKER ARE ONE DECISION. Splitting them made the consent a lie: the
    user agreed to "keep your Oracles current — RECURRING, forever", a marker was written, a job
    row was queued, and nothing on the machine ever claimed it. Measured 2026-09-13 (two jobs
    unclaimed since 13:50) and again on a fresh onboarding 2026-09-14 (two queued at 15:58, still
    unstarted an hour later). A person who agreed to work happening forever without them present
    has agreed to the process that does it; asking again asks twice for one decision."""
    assert _follow(True, monkeypatch, status=_ABSENT)["status"] == "INSTALLED"


def test_revoking_it_removes_the_worker(monkeypatch):
    """A toggle toggles both ways. A resident process the user has withdrawn consent for is worse
    than one that was never installed."""
    assert _follow(False, monkeypatch, status=_ABSENT)["status"] == "REMOVED"


def test_an_already_running_worker_is_left_alone(monkeypatch):
    """Re-onboarding must not bootout and re-bootstrap a healthy agent mid-pass."""
    live = {"supported": True, "installed": True, "loaded": True, "ran": True}

    def _boom(**kw):
        raise AssertionError("a loaded agent must not be reinstalled")

    assert _follow(True, monkeypatch, status=live, install=_boom)["status"] == "ALREADY_RUNNING"


def test_an_installed_but_unloaded_agent_is_repaired(monkeypatch):
    """A plist on disk that launchd is not running is the shape a half-failed install leaves, and
    it is indistinguishable from "scheduled" to every other reader. Re-install rather than
    report success."""
    stale = {"supported": True, "installed": True, "loaded": False, "ran": False}
    assert _follow(True, monkeypatch, status=stale)["status"] == "INSTALLED"


def test_a_hosted_home_reports_the_worker_it_has_instead_of_denying_it(monkeypatch):
    """⚠️ THE ONE HOME WHERE THE SCHEDULE IS DEFINITELY KEPT WAS THE ONE DENYING IT.

    `install_worker` is macOS-only by ruling F1, so on the hosted box `supported` is False — and
    the caller turns that into "nothing on this machine will act on it on its own" for the host to
    relay. On that box `gateway/deploy/opyt-worker.service` is a resident worker whose entire job
    is claiming the rows consent just queued. The probe was answering "is there a LaunchAgent" for
    a question that is "will anything claim this row".
    """
    monkeypatch.setenv("OPYT_WORKER_HOME_ID", "1078")
    monkeypatch.setenv("OPYT_WORKER_DB", "/var/lib/opyt-worker/rail_jobs.db")

    def _boom(**kw):
        raise AssertionError("a hosted home has no per-user service to install")

    out = _follow(True, monkeypatch, status=_LINUX, install=_boom)

    assert out["status"] == "RESIDENT"


def test_revoking_on_a_hosted_home_removes_nothing_and_still_revokes(monkeypatch):
    """There is no per-user service to take away, and taking the consent away is not this
    function's job anyway: `_apply_consent` clears the refresh marker before it calls here, and
    the rail reads that marker on every pass."""
    monkeypatch.setenv("OPYT_WORKER_HOME_ID", "1078")
    monkeypatch.setenv("OPYT_WORKER_DB", "/var/lib/opyt-worker/rail_jobs.db")

    def _boom(**kw):
        raise AssertionError("a hosted home has no per-user service to remove")

    assert _follow(False, monkeypatch, status=_LINUX, uninstall=_boom)["status"] == "RESIDENT"


def test_a_local_home_that_relocated_its_database_still_gets_its_agent(monkeypatch):
    """The marker is the HOME ID, not the database path, and this is why. A local user may point
    `OPYT_WORKER_DB` wherever they like; only the gateway sets `OPYT_WORKER_HOME_ID`. Reading the
    database path here would have taken the LaunchAgent away from that user and left them with
    nothing claiming their rows — the exact failure this whole branch exists to prevent."""
    monkeypatch.delenv("OPYT_WORKER_HOME_ID", raising=False)
    monkeypatch.setenv("OPYT_WORKER_DB", "/Users/someone/elsewhere/rail_jobs.db")

    assert _follow(True, monkeypatch, status=_ABSENT)["status"] == "INSTALLED"


def test_a_platform_with_no_approved_service_says_so_instead_of_pretending(monkeypatch):
    """Ruling F1: macOS only, because "an unverified resident service is worse than an honest
    absence". The honest absence still has to reach the host, or it promises a schedule."""
    out = _follow(True, monkeypatch,
                  status={"supported": False, "installed": False, "loaded": False, "ran": False})
    assert out["status"] == "unsupported" and "by hand" in out["note"]


def test_a_failed_install_never_rolls_the_consent_back(monkeypatch):
    """The markers are the user's decision and they stand. What a failure costs is the automation,
    and the report is what has to be honest about it — `uvx_command` raising (no `uv` on the
    machine) is the expected form."""
    def _raise(**kw):
        raise FileNotFoundError("uvx not found")

    out = _follow(True, monkeypatch, status=_ABSENT, install=_raise)
    assert out["status"] == "not_installed" and "uvx not found" in out["error"]


# ── the worker must run the same code as the server that installed it ──────────

def test_a_checkout_installs_a_worker_that_runs_that_checkout(monkeypatch, tmp_path):
    """⚠️ VERSION IS NOT IDENTITY, and that is what made this invisible. A checkout and the
    published wheel both report `0.1.0a5`, so a plist reading `uvx --from opyt==0.1.0a5` looks
    like the code in front of you. On 2026-09-14 a worker installed from a worktree launched the
    PyPI build against the same store — two codebases, one database, identical version strings,
    and no surface anywhere saying so. It presents as "the fix didn't work"."""
    root = tmp_path / "checkout"
    (root / "opyt_core").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'opyt'\n")
    (root / "opyt_core" / "__init__.py").write_text("")
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "opyt-worker").write_text("#!/bin/sh\n")
    (bin_dir / "python").write_text("#!/bin/sh\n")

    monkeypatch.setattr(install_worker.sys, "executable", str(bin_dir / "python"))
    monkeypatch.setitem(sys.modules, "opyt_core",
                        type(sys)("opyt_core"))
    sys.modules["opyt_core"].__file__ = str(root / "opyt_core" / "__init__.py")

    dist = install_worker.running_distribution()

    assert dist["kind"] == "source"
    assert dist["launcher"] == [str(bin_dir / "opyt-worker")]


def test_a_published_install_still_goes_through_uvx(uvx, monkeypatch, tmp_path):
    """The shipped path is unchanged. A published install has no checkout to point at, and
    welding one to a folder that can move is what `uvx_command` was written to prevent."""
    site = tmp_path / "site-packages" / "opyt_core"
    site.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "opyt_core", type(sys)("opyt_core"))
    sys.modules["opyt_core"].__file__ = str(site / "__init__.py")

    dist = install_worker.running_distribution()

    assert dist["kind"] == "published"
    assert "uvx" in dist["launcher"][0] and "--from" in dist["launcher"]


def test_status_reports_which_build_an_installed_agent_actually_launches(launchctl, uvx,
                                                                        monkeypatch, tmp_path):
    """A worker running different code from the server that installed it is invisible without
    this. `status()` is what every "work continues on its own" sentence rests on, so it is also
    where the mismatch has to become visible."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")
    install_worker.install()

    st = install_worker.status()

    assert st["running_from"] in ("source", "published")
    assert st["agent_launches"] == install_worker.running_distribution()["launcher"]
