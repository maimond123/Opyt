from pathlib import Path

from pipeline.ingestion import cdp


def test_launch_removes_previous_devtools_endpoint_before_starting(monkeypatch, tmp_path):
    marker = tmp_path / cdp._PORT_FILE
    marker.write_text("41998\n/devtools/browser/old\n")

    class _Process:
        pass

    monkeypatch.setattr(cdp.subprocess, "Popen", lambda *args, **kwargs: _Process())

    cdp._launch(Path("/usr/bin/chromium"), tmp_path, headless=True, url=None)

    assert not marker.exists()


def test_launch_attaches_no_debugger_to_the_sign_in_window(monkeypatch, tmp_path):
    """Google refuses sign-in to a debugged browser, so the human-driven window must carry no
    --remote-debugging-port — the same boundary the hosted login desktop already obeys."""
    captured = {}

    def _popen(argv, **kwargs):
        captured["argv"] = argv
        return object()

    monkeypatch.setattr(cdp.subprocess, "Popen", _popen)

    cdp.launch(Path("/usr/bin/chromium"), tmp_path / "profile", url="https://x.com/login")

    assert not any(a.startswith("--remote-debugging-port") for a in captured["argv"])
    assert not any(a.startswith("--headless") for a in captured["argv"])
    assert captured["argv"][-1] == "https://x.com/login"


def test_controlled_browser_reaps_after_forced_kill(monkeypatch, tmp_path):
    """An owned Chrome that ignores terminate must still be collected after kill."""
    class _Process:
        def __init__(self):
            self.terminated = self.killed = False
            self.waits = []

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise cdp.subprocess.TimeoutExpired("chrome", timeout)

    class _Socket:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.closed = False

        def close(self):
            self.closed = True

    proc = _Process()
    monkeypatch.setattr(cdp, "_launch", lambda *args, **kwargs: proc)
    monkeypatch.setattr(cdp, "_wait_for_endpoint", lambda *args, **kwargs: "ws://chrome")
    monkeypatch.setattr(cdp, "_Socket", _Socket)

    with cdp.controlled_browser(Path("/usr/bin/chromium"), user_data_dir=tmp_path) as sock:
        assert sock.endpoint == "ws://chrome"

    assert sock.closed
    assert proc.terminated and proc.killed
    assert proc.waits == [10, None]
