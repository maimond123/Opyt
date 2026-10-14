"""
tests/test_install_entry_points.py — what install's three console scripts print and return.

Pinned so
moving code cannot change a command's output by accident. Everything runs in tmp homes: the
conftest fixture already redirects the LaunchAgent and fakes `launchctl`.
"""
from __future__ import annotations

import json
import plistlib
import stat

import pytest

from opyt_core import install_client, install_worker, keys


@pytest.fixture
def uvx(monkeypatch, tmp_path):
    fake = tmp_path / "bin" / "uvx"
    fake.parent.mkdir()
    fake.write_text("")
    monkeypatch.setattr(install_client.shutil, "which", lambda _name: str(fake))
    return fake


def test_install_worker_prints_the_installed_agent_and_its_log(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_worker.sys, "platform", "darwin")

    assert install_worker.main([]) == 0

    out = capsys.readouterr().out
    assert out.startswith(f"[INSTALLED] {install_worker.LABEL} -> {install_worker.PLIST_PATH}")
    assert f"logs: {tmp_path / 'home' / 'worker.log'}" in out
    assert install_worker.main([]) == 0
    assert capsys.readouterr().out.startswith("[UPDATED]")


def test_install_worker_dry_run_prints_the_plist_and_writes_nothing(monkeypatch,
                                                                    capsys):
    monkeypatch.setattr(install_worker.sys, "platform", "darwin")

    assert install_worker.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("[DRY_RUN]")
    assert "<key>ProgramArguments</key>" in out
    assert not install_worker.PLIST_PATH.exists()


def test_install_worker_uninstall_prints_removed(monkeypatch, capsys):
    monkeypatch.setattr(install_worker.sys, "platform", "darwin")
    install_worker.main([])
    capsys.readouterr()

    assert install_worker.main(["--uninstall"]) == 0

    assert capsys.readouterr().out.startswith("[REMOVED]")
    assert not install_worker.PLIST_PATH.exists()


@pytest.fixture
def clients(monkeypatch, tmp_path):
    paths = {name: tmp_path / name / "mcp.json" for name in install_client.CLIENTS}
    monkeypatch.setattr(install_client, "CLIENTS", paths)
    return paths


def test_install_client_merges_beside_other_servers_and_backs_up(clients, uvx, capsys):
    path = clients["cursor"]
    path.parent.mkdir()
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}, "opyt": {}}}))

    assert install_client.main(["--cursor"]) == 0

    out = capsys.readouterr().out
    assert out == (f"[INSTALLED] cursor -> {path}  (backup: {path}.opyt-bak)\n")
    cfg = json.loads(path.read_text())
    assert cfg["mcpServers"] == {
        "other": {"command": "x"},
        "Opyt": {"command": str(uvx), "args": ["--from", "opyt@latest", "opyt-mcp"]}}
    assert install_client.main(["--cursor"]) == 0
    assert capsys.readouterr().out == f"[ALREADY_CURRENT] cursor -> {path}\n"


def test_install_client_refuses_a_malformed_config_unless_forced(clients, uvx, capsys):
    path = clients["windsurf"]
    path.parent.mkdir()
    path.write_text("{ not json")

    assert install_client.main(["--windsurf"]) == 1
    assert capsys.readouterr().out.startswith(f"[ERROR_MALFORMED] windsurf -> {path}")
    assert path.read_text() == "{ not json"

    assert install_client.main(["--windsurf", "--force"]) == 0
    assert list(json.loads(path.read_text())["mcpServers"]) == ["Opyt"]


def test_install_client_dry_run_writes_nothing(clients, uvx, capsys):
    assert install_client.main(["--all", "--dry-run"]) == 0

    out = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in out] == ["[DRY_RUN]"] * 3
    assert not any(p.exists() for p in clients.values())


def test_install_client_needs_a_target(clients):
    with pytest.raises(SystemExit) as e:
        install_client.main([])
    assert e.value.code == 2


def test_opyt_keys_set_writes_owner_only_and_never_prints_the_value(monkeypatch, tmp_path,
                                                                    capsys):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("# mine\nA=1\n")

    assert keys.main(["--set", "B= secret-value "]) == 0

    out = capsys.readouterr().out
    assert f"[set] B -> {tmp_path / '.env'}" in out
    assert "secret-value" not in out
    assert (tmp_path / ".env").read_text() == "A=1\nB=secret-value\n"
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_opyt_keys_rejects_a_set_without_a_name(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        keys.main(["--set", "=x"])
    assert e.value.code == 2


def test_the_written_plist_is_the_one_install_reports(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    res = install_worker.install()
    written = plistlib.loads(install_worker.PLIST_PATH.read_bytes())

    assert res["status"] == "INSTALLED"
    assert written["Label"] == install_worker.LABEL
    assert written["KeepAlive"] is True and written["ThrottleInterval"] == 60
    assert written["StandardOutPath"] == str(tmp_path / "home" / "worker.log")


@pytest.mark.parametrize("installed, loaded, disabled, expected", [
    (False, False, True, False),     # removed on purpose: never resurrected
    (True, True, None, False),       # already running
    (True, False, None, False),      # the disable list could not be read
    (True, False, False, False),     # not loaded for some other reason
    (True, False, True, True),       # the sticky disable: the one fault repair fixes
])
def test_needs_repair(installed, loaded, disabled, expected):
    state = {"installed": installed, "loaded": loaded, "disabled": disabled}
    assert install_worker.needs_repair(state) is expected


SPEC = {"command": "/bin/uvx", "args": ["--from", "opyt@latest", "opyt-mcp"]}


@pytest.mark.parametrize("cfg, servers, migrated, existed", [
    ({}, {"Opyt": SPEC}, [], False),
    ({"mcpServers": []}, {"Opyt": SPEC}, [], False),
    ({"mcpServers": {"x": {}}}, {"x": {}, "Opyt": SPEC}, [], False),
    ({"mcpServers": {"Opyt": {"old": 1}}}, {"Opyt": SPEC}, [], True),
    ({"mcpServers": {"opyt": {}, "Opyt": {}}}, {"Opyt": SPEC}, ["opyt"], True),
])
def test_merge_server(cfg, servers, migrated, existed):
    out, dropped, was_there = install_client.merge_server(cfg, SPEC)
    assert out["mcpServers"] == servers
    assert (dropped, was_there) == (migrated, existed)


def test_install_worker_off_macos_says_macos_only_and_gives_no_linux_advice(monkeypatch, capsys):
    """Local installs are macOS-only, so the command offers Linux nothing."""
    monkeypatch.setattr(install_worker.sys, "platform", "linux")

    assert install_worker.main([]) == 1

    err = capsys.readouterr().err
    assert "runs on macOS only" in err
    assert "systemd" not in err and "Linux" not in err


def test_set_key_reads_env_the_way_every_other_reader_does(monkeypatch, tmp_path):
    """One parser (python-dotenv) for `.env`, so a quoted value means
    the same thing to the writer as to `load_dotenv` and `get_credential`."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    (tmp_path / ".env").write_text('A="x"\nexport C=z\n')

    keys.set_key("B", "y")

    assert (tmp_path / ".env").read_text() == "A=x\nC=z\nB=y\n"


def test_agent_plist_is_built_from_plain_values(tmp_path):
    plist = install_worker.agent_plist(["/bin/opyt-worker"], tmp_path / "worker.log")

    assert plist["ProgramArguments"] == ["/bin/opyt-worker"]
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == str(tmp_path / "worker.log")
