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

from opyt_core import install_client, install_worker, launcher
from opyt_core.paths import opyt_home

from pipeline.worker import liveness


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
    can go stale. That rule is scoped to a published install rather than applied to every one, because a CHECKOUT pointed at
    `uvx --from opyt==<same version>` silently runs different code — see
    `test_a_checkout_installs_a_worker_that_runs_that_checkout`.

    The mock builds its launcher through `uvx_command` rather than writing an argv out by hand,
    because `agent_plist` carries `running_distribution`'s launcher VERBATIM: a hand-written pin
    here would assert a shape that function can no longer produce, and would have hidden a
    merge that put `@latest` in `uvx_command` while this test still spelled a pin.
    The `@latest` rule itself is pinned one test down, on `uvx_command`, where it lives."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "running_distribution",
                        lambda: {"kind": "published", "root": "/site-packages",
                                 "launcher": launcher.uvx_command("opyt-worker")})

    plist = install_worker.install(dry_run=True)["would_write"]

    assert plist["ProgramArguments"] == [str(uvx), "--from", "opyt@latest", "opyt-worker"]
    assert plist["Label"] == "com.useopyt.worker"
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True
    assert str(tmp_path / "home" / "worker.log") == plist["StandardErrorPath"]


def test_the_launcher_resolves_the_newest_build_on_every_start(uvx):
    """A pin — and a BARE requirement — both freeze a user on the build they installed.

    Measured on uv 0.12.13 with a warm cache: `--from opyt==<v>` and `--from opyt`
    both launch in 36 ms and never query the index, while `--from opyt@latest` takes 226 ms and
    does. So the fix for "my users never get my fixes" is not deleting the `==`; it is `@latest`
    specifically, and this test exists because deleting the `==` LOOKS like it works.
    """
    argv = launcher.uvx_command("opyt-mcp")

    assert argv[-2:] == ["opyt@latest", "opyt-mcp"]
    assert not any("==" in a for a in argv), "a pin freezes every installed user"
    assert "opyt" not in argv, "a bare requirement freezes them just as hard, but invisibly"


def test_the_agent_exposes_no_port(uvx, monkeypatch, tmp_path):
    """A resident scheduler that listened would be a second way into a user's corpus."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))

    assert "Sockets" not in install_worker.install(dry_run=True)["would_write"]


def test_a_shell_s_data_home_never_travels_into_the_agent(uvx, monkeypatch, tmp_path):
    """A custom home is NOT a local-install option. Copying the installing shell's `$OPYT_HOME`
    into the agent made it half of one: the MCP client's config never carried it, so the worker
    and the server would have used two stores and nothing would have said so."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "elsewhere"))

    assert "EnvironmentVariables" not in install_worker.install(dry_run=True)["would_write"]


def test_the_default_data_home_stays_derived_at_runtime(uvx, monkeypatch):
    monkeypatch.delenv("OPYT_HOME", raising=False)

    assert "EnvironmentVariables" not in install_worker.install(dry_run=True)["would_write"]


def test_install_replaces_a_loaded_agent_rather_than_stacking_one(uvx, launchctl, monkeypatch,
                                                                  tmp_path):
    """Bootstrapping over a loaded label fails, so the unload has to come first every time."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")

    first = install_worker.install()
    second = install_worker.install()

    assert (first["status"], second["status"]) == ("INSTALLED", "UPDATED")
    # `enable` sits between them: bootout does NOT clear a sticky disable.
    assert [call[1] for call in launchctl] == ["bootout", "enable", "bootstrap"] * 2
    written = plistlib.loads((tmp_path / "agents" / "worker.plist").read_bytes())
    assert written == install_worker.install(dry_run=True)["would_write"]


def test_uninstall_stops_the_agent_and_leaves_the_jobs_database(uvx, launchctl, monkeypatch,
                                                                tmp_path):
    """A reinstall resumes the same schedule; removing the queue would drop queued work."""
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


# ── repair or report: fix the worker, never fall back ───
#
# Every local install gets the worker (always), so no setup answer installs or removes it —
# `repair_or_report` is what every caller asks instead. A broken worker loses
# nothing, because a job is a durable row; it must never wait SILENTLY, so each case below ends
# in a beating worker or in a cause and the user's one step.

_ABSENT = {"supported": True, "installed": False, "loaded": False,
           "beating": False, "disabled": None}
_LOADED = {**_ABSENT, "installed": True, "loaded": True, "disabled": False}


def _never(name):
    def boom(**kw):
        raise AssertionError(f"{name} must not run here")
    return boom


def _report(monkeypatch, tmp_path, *, status, install=None, repair=None, beats=(True,),
            platform="darwin", patience=0.0):
    """Drive `repair_or_report` with every side door stubbed. `beats` is what successive
    heartbeat reads return; the last one repeats."""
    from pipeline.worker import job_queue
    reads = list(beats)
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker.sys, "platform", platform)
    monkeypatch.setattr(liveness, "status", lambda: status)
    monkeypatch.setattr(install_worker, "install",
                        install or (lambda **kw: {"status": "INSTALLED"}))
    monkeypatch.setattr(install_worker, "repair",
                        repair or (lambda **kw: {"status": "REPAIRED"}))
    monkeypatch.setattr(install_worker, "uninstall", _never("uninstall"))
    monkeypatch.setattr(job_queue, "worker_is_alive",
                        lambda **kw: reads.pop(0) if len(reads) > 1 else reads[0])
    monkeypatch.setattr("time.sleep", lambda s: None)
    return liveness.repair_or_report(patience=patience)


def test_a_beating_worker_is_left_alone(monkeypatch, tmp_path):
    """Re-onboarding must not bootout and re-bootstrap a healthy agent mid-pass."""
    out = _report(monkeypatch, tmp_path, status={**_LOADED, "beating": True},
                  install=_never("install"), repair=_never("repair"))
    assert out["ok"] and out["status"] == "running"


def test_a_missing_agent_is_installed(monkeypatch, tmp_path):
    """Never installed, or the plist was deleted: put it back. Removing it is no longer a way to
    opt out of anything — declining refresh is, and it leaves the worker."""
    out = _report(monkeypatch, tmp_path, status=_ABSENT, repair=_never("repair"))
    assert out["ok"] and out["status"] == "installed"


def test_an_installed_but_unloaded_agent_is_reloaded(monkeypatch, tmp_path):
    """A plist on disk that launchd is not running is the shape a half-failed install leaves, and
    it is indistinguishable from "scheduled" to every other reader."""
    stale = {**_LOADED, "loaded": False}
    out = _report(monkeypatch, tmp_path, status=stale, repair=_never("repair"),
                  install=lambda **kw: {"status": "UPDATED"})
    assert out["ok"] and out["status"] == "reloaded"


def test_a_sticky_disable_is_repaired_not_reinstalled(monkeypatch, tmp_path):
    """The one not-loaded reason that never clears on its own. `repair()` is the
    narrow fix for exactly it."""
    disabled = {**_LOADED, "loaded": False, "disabled": True}
    out = _report(monkeypatch, tmp_path, status=disabled, install=_never("install"))
    assert out["ok"] and out["status"] == "repaired"


def test_a_just_woken_mac_is_not_judged_on_one_read(monkeypatch, tmp_path):
    """The worker beats every `BEAT_SECONDS` and reads dead after `BEAT_STALE_AFTER`, so a Mac
    just woken reads dead until its loop's next pass. Wait for it; touch nothing."""
    out = _report(monkeypatch, tmp_path, status=_LOADED, beats=(False, False, True),
                  install=_never("install"), repair=_never("repair"), patience=5.0)
    assert out["ok"] and out["status"] == "running"


def test_a_worker_that_never_beats_is_reported_with_its_log(monkeypatch, tmp_path):
    """Loaded and silent through the wait: it starts and dies, most likely a release that crashes
    on start. The log is the evidence, and it goes in `detail` — never in the person's words."""
    home = tmp_path / "home"
    home.mkdir()
    (home / install_worker.WORKER_LOG).write_text("booting\nTraceback: boom\n")

    out = _report(monkeypatch, tmp_path, status=_LOADED, beats=(False,),
                  install=_never("install"), repair=_never("repair"))

    assert not out["ok"] and out["status"] == "not_beating"
    assert out["detail"]["log_tail"][-1] == "Traceback: boom"
    assert "Traceback" not in out["cause"] and "log" not in out["cause"]
    assert out["step"]


def test_an_unreadable_heartbeat_is_reported_after_rereading(monkeypatch, tmp_path):
    out = _report(monkeypatch, tmp_path, status={**_LOADED, "beating": None}, beats=(None,),
                  install=_never("install"), repair=_never("repair"), patience=2.0)
    assert not out["ok"] and out["status"] == "unreadable"


def test_launchd_refusing_the_agent_names_the_switch(monkeypatch, tmp_path):
    """The errno 5 is relayed as what a person can act on: the Login Items toggle."""
    out = _report(monkeypatch, tmp_path, status=_ABSENT,
                  install=lambda **kw: {"status": "ERROR_LOAD", "cause": "launchd_disabled"})
    assert not out["ok"] and out["status"] == "refused"
    assert "Allow in the Background" in out["step"]
    assert out["detail"]["cause"] == "launchd_disabled"


def test_no_uv_is_reported_with_the_one_step(monkeypatch, tmp_path):
    """`uvx_command` raising is the expected form of a launcher that does not exist."""
    def _raise(**kw):
        raise FileNotFoundError("uvx not found")

    out = _report(monkeypatch, tmp_path, status=_ABSENT, install=_raise)
    assert not out["ok"] and out["status"] == "not_installed"
    assert "uv" in out["step"]


def test_off_macos_the_step_is_hosted(monkeypatch, tmp_path):
    """Ruling 5: local installs are macOS-only, and Windows and Linux users go to hosted."""
    out = _report(monkeypatch, tmp_path, status={**_ABSENT, "supported": False},
                  platform="linux", install=_never("install"))
    assert not out["ok"] and out["status"] == "unsupported"
    assert out["step"] == liveness.USE_A_MAC


def test_off_macos_a_beating_worker_is_running(monkeypatch, tmp_path):
    """B-W1: a worker run by hand (`opyt-worker`, the documented route off macOS) beats and
    claims, yet the report said `unsupported` before it read the beat, so every promise of
    background work told that home it had none."""
    out = _report(monkeypatch, tmp_path, status={**_ABSENT, "supported": False, "beating": True},
                  platform="linux", install=_never("install"), repair=_never("repair"))
    assert out["ok"] is True and out["status"] == "running"


@pytest.mark.parametrize("alive, ok, status", [(True, True, "resident"),
                                               (None, True, "resident"),
                                               (False, False, "hosted_down")])
def test_a_hosted_home_reads_the_box_resident_and_installs_nothing(monkeypatch, tmp_path,
                                                                   alive, ok, status):
    """⚠️ THE ONE HOME WHERE THE SCHEDULE IS DEFINITELY KEPT WAS ONCE THE ONE DENYING IT. A hosted
    child is Linux, so `status()["supported"]` is False — which says nothing about the systemd
    unit claiming every user's rows. An unreadable beat keeps the promise (an unknown must not
    re-create the certain under-promise); a stale one is down, and ours to fix."""
    monkeypatch.setenv("OPYT_WORKER_HOME_ID", "1078")
    monkeypatch.setenv("OPYT_WORKER_DB", "/var/lib/opyt-worker/rail_jobs.db")

    out = _report(monkeypatch, tmp_path, status=_ABSENT, platform="linux", beats=(alive,),
                  install=_never("install"), repair=_never("repair"))

    assert out["ok"] is ok and out["status"] == status
    assert out["step"] is None


def test_a_local_home_that_relocated_its_database_still_gets_its_agent(monkeypatch, tmp_path):
    """The marker is the HOME ID, not the database path. A local user may point `OPYT_WORKER_DB`
    wherever they like; only a hosted home sets `OPYT_WORKER_HOME_ID`."""
    monkeypatch.delenv("OPYT_WORKER_HOME_ID", raising=False)
    monkeypatch.setenv("OPYT_WORKER_DB", "/Users/someone/elsewhere/rail_jobs.db")

    out = _report(monkeypatch, tmp_path, status=_ABSENT)
    assert out["ok"] and out["status"] == "installed"


def test_a_probe_that_raises_is_reported_never_raised(monkeypatch, tmp_path):
    def _raise():
        raise OSError("launchctl vanished")

    monkeypatch.setattr(install_worker.sys, "platform", "darwin")
    monkeypatch.setattr(liveness, "status", _raise)

    out = liveness.repair_or_report(patience=0)
    assert not out["ok"] and out["status"] == "unknown"
    assert "launchctl vanished" in out["detail"]["error"]


# ── the worker must run the same code as the server that installed it ──────────

def test_a_checkout_installs_a_worker_that_runs_that_checkout(monkeypatch, tmp_path):
    """⚠️ VERSION IS NOT IDENTITY, and that is what made this invisible. A checkout and the
    published wheel both report `0.1.0a5`, so a plist reading `uvx --from opyt==0.1.0a5` looks
    like the code in front of you. A worker installed from a worktree once launched the
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
    monkeypatch.setattr(launcher, "__file__", str(root / "opyt_core" / "launcher.py"))

    dist = launcher.running_distribution()

    assert dist["kind"] == "source"
    assert dist["launcher"] == [str(bin_dir / "opyt-worker")]


def test_a_published_install_still_goes_through_uvx(uvx, monkeypatch, tmp_path):
    """The shipped path is unchanged. A published install has no checkout to point at, and
    welding one to a folder that can move is what `uvx_command` was written to prevent."""
    site = tmp_path / "site-packages" / "opyt_core"
    site.mkdir(parents=True)
    monkeypatch.setattr(launcher, "__file__", str(site / "launcher.py"))

    dist = launcher.running_distribution()

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

    st = liveness.status()

    assert st["running_from"] in ("source", "published")
    assert st["agent_launches"] == launcher.running_distribution()["launcher"]


def test_status_only_reads(monkeypatch, tmp_path):
    """A4: the probe never writes. On an empty home it creates no `rail_jobs.db`."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "agent_state", lambda: {"installed": False})

    st = liveness.status()

    assert not (opyt_home() / "rail_jobs.db").exists()
    assert st == {"installed": False, "beating": False}


# ── the sticky launchd disable ──────────

class _Launchd:
    """A `launchctl` fake WITH a persistent disable flag — the stub that can say no.

    The repo-wide conftest guard returns 0 for every subcommand, so `bootstrap` could never
    fail in a test and the whole class of "launchctl refused us" bugs was unreachable by
    construction. This fake models the piece that mattered: a disabled label refuses
    `bootstrap` with errno 5, `bootout` does NOT clear the flag, and only an `enable` in
    TARGET form (`gui/<uid>/<label>`) does.
    """

    def __init__(self, *, disabled=False, loaded=False, bootstrap_fails_anyway=False,
                 print_disabled_rc=0):
        self.disabled = disabled
        self.loaded = loaded
        self.bootstrap_fails_anyway = bootstrap_fails_anyway
        self.print_disabled_rc = print_disabled_rc
        self.calls: list[list[str]] = []

    def __call__(self, *argv):
        self.calls.append(list(argv))
        cmd = argv[0]
        rc, out, err = 0, "", ""
        if cmd == "enable":
            if argv[1].count("/") == 2 and argv[1].endswith(install_worker.LABEL):
                self.disabled = False
            else:
                rc, err = 113, "Unrecognized target specifier."  # domain form: launchd refuses
        elif cmd == "bootstrap":
            if self.disabled or self.bootstrap_fails_anyway:
                rc, err = 5, "Bootstrap failed: 5: Input/output error\n"
            else:
                self.loaded = True
        elif cmd == "bootout":
            self.loaded = False           # boots out; NEVER clears the disable — the trap
        elif cmd == "print":
            rc = 0 if self.loaded else 113
        elif cmd == "print-disabled":
            rc = self.print_disabled_rc
            verdict = "disabled" if self.disabled else "enabled"
            out = f'\t"{install_worker.LABEL}" => {verdict}\n'
        return subprocess.CompletedProcess(list(argv), rc, stdout=out, stderr=err)


@pytest.fixture
def _installed_plist(uvx, monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    plist = tmp_path / "agents" / "worker.plist"
    monkeypatch.setattr(install_worker, "PLIST_PATH", plist)
    plist.parent.mkdir(parents=True)
    plist.write_bytes(b"<plist/>")
    return plist


def test_a_disabled_machine_recovers_through_install(uvx, monkeypatch, tmp_path):
    """`install()` used to write the plist, bootout and bootstrap — and `bootout` does not
    clear a disable, so every install on a disabled machine failed identically forever with
    an error naming nothing real. `enable` (TARGET form) before `bootstrap` is the fix."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")
    fake = _Launchd(disabled=True)
    monkeypatch.setattr(install_worker, "_launchctl", fake)

    out = install_worker.install()

    assert out["status"] == "INSTALLED"
    enables = [c for c in fake.calls if c[0] == "enable"]
    assert enables == [["enable", f"gui/{install_worker.os.getuid()}/{install_worker.LABEL}"]]


def test_an_unknown_bootstrap_failure_keeps_the_raw_detail_and_no_cause(
        uvx, monkeypatch, tmp_path):
    """enable ran, the label reads enabled, bootstrap still failed: genuinely unknown
    territory. Inventing a cause here would send the user chasing the wrong fault."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")
    fake = _Launchd(bootstrap_fails_anyway=True)
    monkeypatch.setattr(install_worker, "_launchctl", fake)

    out = install_worker.install()

    assert out["status"] == "ERROR_LOAD"
    assert "cause" not in out and "remedy" not in out
    assert "Input/output error" in out["detail"]


def test_a_diagnosed_disable_names_its_cause_and_remedy(uvx, monkeypatch, tmp_path):
    """When bootstrap fails AND the label measurably reads disabled, `cause` is what lets a
    host say something actionable instead of relaying errno 5."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker, "PLIST_PATH", tmp_path / "agents" / "worker.plist")
    fake = _Launchd(disabled=True, bootstrap_fails_anyway=True)  # enable clears; bootstrap still refuses
    monkeypatch.setattr(install_worker, "_launchctl", fake)
    fake.disabled = True

    # Freeze the disable back on after enable, modeling a re-assertion mid-install.
    real_call = fake.__call__

    def sticky(*argv):
        res = real_call(*argv)
        fake.disabled = True
        return res

    monkeypatch.setattr(install_worker, "_launchctl", sticky)

    out = install_worker.install()

    assert out["status"] == "ERROR_LOAD"
    assert out["cause"] == "launchd_disabled"
    assert out["remedy"].startswith("launchctl enable gui/")


def test_disabled_probe_parses_the_verdict_not_the_label(monkeypatch):
    """`print-disabled` lists the label when it is ENABLED too, so a presence grep would
    report every healthy machine as broken."""
    monkeypatch.setattr(install_worker, "_launchctl", _Launchd(disabled=False))
    assert install_worker._disabled() is False

    monkeypatch.setattr(install_worker, "_launchctl", _Launchd(disabled=True))
    assert install_worker._disabled() is True


def test_an_unreadable_disable_list_answers_none(monkeypatch):
    monkeypatch.setattr(install_worker, "_launchctl", _Launchd(print_disabled_rc=113))
    assert install_worker._disabled() is None


def test_repair_fixes_exactly_the_measured_fault(_installed_plist, monkeypatch):
    fake = _Launchd(disabled=True)
    monkeypatch.setattr(install_worker, "_launchctl", fake)

    out = install_worker.repair()

    assert out["status"] == "REPAIRED"
    assert ["enable", f"gui/{install_worker.os.getuid()}/{install_worker.LABEL}"] in fake.calls
    assert fake.loaded is True


def test_repair_is_inert_everywhere_else(_installed_plist, monkeypatch):
    """Healthy, already loaded, unreadable probe, or not installed: repair must never be able
    to resurrect an agent a user deliberately removed, so each case makes NO bootstrap call."""
    for fake in (_Launchd(disabled=False),                       # healthy, just not loaded
                 _Launchd(disabled=False, loaded=True),          # already running
                 _Launchd(disabled=True, print_disabled_rc=1)):  # probe answers None
        monkeypatch.setattr(install_worker, "_launchctl", fake)
        assert install_worker.repair()["status"] == "NO_ACTION"
        assert not any(c[0] == "bootstrap" for c in fake.calls)

    _installed_plist.unlink()                                    # --uninstall removed the plist
    fake = _Launchd(disabled=True)
    monkeypatch.setattr(install_worker, "_launchctl", fake)
    assert install_worker.repair()["status"] == "NO_ACTION"
    assert not any(c[0] == "bootstrap" for c in fake.calls)


# ── the cached form: the one read behind every background promise ─────────────

def test_the_cached_report_reuses_a_fresh_answer(monkeypatch, tmp_path):
    """Hot paths (a `search` suggestion) ask on every call; within the window they get the last
    answer, without another `launchctl` round."""
    reads = []
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(liveness, "_repair_or_report",
                        lambda patience: reads.append(patience) or {"ok": True, "status": "running"})

    first = liveness.repair_or_report_cached()
    second = liveness.repair_or_report_cached()

    assert first == second and reads == [0], "one check, and it never waits"


def test_the_cached_report_reads_again_once_stale(monkeypatch, tmp_path):
    reads = []
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(liveness, "_repair_or_report",
                        lambda patience: reads.append(patience) or {"ok": True, "status": "running"})

    liveness.repair_or_report_cached()
    liveness.repair_or_report_cached(max_age=0)

    assert len(reads) == 2


def test_a_full_check_feeds_the_cache(monkeypatch, tmp_path):
    """`onboard` runs the patient check once per call; the sentences built later in that call
    read its answer back rather than checking again."""
    reads = []
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(liveness, "_repair_or_report",
                        lambda patience: reads.append(patience) or {"ok": False, "status": "x"})

    liveness.repair_or_report(patience=5)

    assert liveness.repair_or_report_cached() == {"ok": False, "status": "x"}
    assert reads == [5]


def test_the_cache_never_answers_for_another_home(monkeypatch, tmp_path):
    reads = []
    monkeypatch.setattr(liveness, "_repair_or_report",
                        lambda patience: reads.append(1) or {"ok": True, "status": "running"})

    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "a"))
    liveness.repair_or_report_cached()
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "b"))
    liveness.repair_or_report_cached()

    assert len(reads) == 2


def test_a_worker_put_back_seconds_ago_is_starting_not_failing(monkeypatch, tmp_path):
    """With no wait, a just-loaded worker has not beaten yet. Calling that "keeps stopping"
    would be a guess, and the cache would hold it for thirty seconds."""
    out = _report(monkeypatch, tmp_path, status=_ABSENT, beats=(False,))
    assert out["ok"] and out["status"] == "starting"


def test_a_repair_that_did_nothing_is_not_called_starting(monkeypatch, tmp_path):
    """`repair()` re-reads and can decline; nothing was put back, so a silent worker is still
    reported as not beating."""
    disabled = {**_LOADED, "disabled": True}
    out = _report(monkeypatch, tmp_path, status=disabled, beats=(False,),
                  repair=lambda **kw: {"status": "NO_ACTION"}, install=_never("install"))
    assert not out["ok"] and out["status"] == "not_beating"


# ── the report's decisions, one row each ────────────────────────────────────────
@pytest.mark.parametrize("alive, status, ok", [
    (False, "hosted_down", False),                       # the box's worker is down
    (True, "resident", True),
    (None, "resident", True),                            # unreadable is not down
])
def test_the_hosted_report(alive, status, ok):
    report = liveness.hosted_report(alive)
    assert (report["status"], report["ok"], report["step"]) == (status, ok, None)


@pytest.mark.parametrize("beating, supported, status, ok", [
    (True, True, "running", True),                       # a live beat on a Mac
    (True, False, "running", True),                      # off macOS too: run by hand (B-W1)
    (False, False, "unsupported", False),                # a silent home off macOS
    (None, False, "unsupported", False),
    (False, True, None, None),                           # a silent Mac: act on it
])
def test_the_report_that_needs_no_action(beating, supported, status, ok):
    report = liveness.report_before_acting({**_ABSENT, "beating": beating,
                                            "supported": supported})
    if status is None:
        assert report is None
        return
    assert (report["status"], report["ok"]) == (status, ok)
    if status == "unsupported":
        assert report["step"] == liveness.USE_A_MAC


def test_a_missing_uv_is_not_installed_with_its_one_step():
    report = liveness.uv_missing(FileNotFoundError("uvx not on PATH"))
    assert (report["status"], report["ok"]) == ("not_installed", False)
    assert "uv" in report["step"] and report["detail"] == {"error": "uvx not on PATH"}


@pytest.mark.parametrize("st, expected", [
    ({"installed": True, "loaded": False, "disabled": True}, "repair"),     # sticky disable
    ({"installed": False, "loaded": False, "disabled": None}, "install"),   # no agent
    ({"installed": True, "loaded": False, "disabled": False}, "install"),   # not loaded
    ({"installed": True, "loaded": True, "disabled": False}, None),         # loaded and silent
    ({"installed": True, "loaded": True, "disabled": None}, None),
])
def test_what_fixes_a_silent_local_worker(st, expected):
    assert liveness.action_for(st) == expected


@pytest.mark.parametrize("acted, beat, status, ok", [
    (None, True, "running", True),
    ({"status": "REPAIRED"}, True, "repaired", True),
    ({"status": "INSTALLED"}, True, "installed", True),
    ({"status": "UPDATED"}, True, "reloaded", True),
    ({"status": "INSTALLED"}, False, "starting", True),  # just put back: not judged yet
    ({"status": "NO_ACTION"}, None, "unreadable", False),
    (None, None, "unreadable", False),
    (None, False, "not_beating", False),                 # loaded and silent through the wait
    ({"status": "ERROR_LOAD"}, None, "refused", False),    # launchd refused: no wait
])
def test_the_report_once_the_wait_is_over(acted, beat, status, ok):
    report = liveness.report_after(acted, beat, ["a log line"])
    assert (report["status"], report["ok"]) == (status, ok)
    if status == "not_beating":
        assert report["detail"] == {"acted": None, "log_tail": ["a log line"]}
