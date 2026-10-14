"""`render.is_js_shell` on real pages, `render.filled`'s fallbacks, and one real render."""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

from pipeline.platforms.browser import render

FIXTURES = Path(__file__).parent / "fixtures" / "render"


@pytest.mark.parametrize("html, shell", [
    ((FIXTURES / "aleximas_shell.html").read_text(), True),   # 1,130 bytes, 2 scripts, 0 links
    ((FIXTURES / "danintheory.html").read_text(), False),     # a plain site: ~65 links
    ("<html><body><p>Hello</p></body></html>", False),       # no scripts: nothing to fill in
    ("<div id=root></div><script src=a.js></script>", True),
    ("<script src=a.js></script><a href=/a>a</a><a href=/b>b</a><a href=/c>c</a>", False),
    ("<script src=a.js></script><p>" + "word " * 200 + "</p>", False),   # real text ships
    ("<script>var x = '" + "y" * 5000 + "'</script>", True),            # script text is not text
    ("", False),
])
def test_is_js_shell(html, shell):
    assert render.is_js_shell(html) is shell


def test_filled_keeps_a_finished_page_without_rendering(monkeypatch):
    monkeypatch.setattr(render, "rendered_html", lambda url: pytest.fail("rendered a real page"))
    page = (FIXTURES / "danintheory.html").read_text()
    assert render.filled(page, "https://danintheory.com") is page


def test_filled_falls_back_to_the_plain_html_when_nothing_renders(monkeypatch):
    shell = (FIXTURES / "aleximas_shell.html").read_text()
    monkeypatch.setattr(render, "rendered_html", lambda url: None)
    assert render.filled(shell, "https://aleximas.com") == shell
    assert render.filled(None, "https://aleximas.com") is None


def test_rendered_html_runs_the_page_scripts():
    """A real headless Chrome on a local page whose links exist only after its script runs."""
    if render._browser_binary() is None:
        pytest.skip("no Chromium-family browser installed")
    page = (b"<div id=app></div><script>document.getElementById('app').innerHTML="
            b"'<a href=\"https://example.org/built\">built</a>'</script>")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        html = render.rendered_html(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
    assert html and "https://example.org/built" in html
