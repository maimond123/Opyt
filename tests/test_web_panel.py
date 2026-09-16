"""
tests/test_web_panel.py

The one Opyt-styled browser page. Two properties are pinned, and neither is about how the panel
looks — the markup is free to change:

  • Every slot is ESCAPED. These pages carry a filesystem path and a server-supplied message,
    and one of them renders into a browser on the user's own machine.
  • BOTH homes render through this module. A second copy of the markup is the defect the
    `one-opyt-panel` guard bans; this asserts the local half of it behaviourally.
"""

from opyt_core import keys, local_auth, web_panel


def test_every_slot_is_escaped():
    html = web_panel.render("<t>", ok=True, headline="<script>x</script>", detail="a & b")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "a &amp; b" in html


def test_ok_and_not_ok_render_different_marks():
    """`ok` is the flow's outcome, not a page kind: a failure still renders a full panel."""
    assert web_panel.render("t", ok=True, headline="h", detail="d") != \
           web_panel.render("t", ok=False, headline="h", detail="d")


def test_the_local_callback_page_is_a_rendered_panel():
    """⚠️ THE 2026-09-09 FINDING. This page is the FIRST Opyt-served page a local user ever
    sees — the OpenRouter approval is phase 1 of onboarding — and it returned a bare `<h2>` in
    the browser's default serif while the hosted home had a styled panel.

    Asserted by COMPARISON, never on the copy or on a markup literal: everything before the
    headline is the panel's chrome, and it must be byte-identical to what `render` emits. A
    test that pinned the sentence would break on every reword; a test that pinned the markup
    would trip the `one-opyt-panel` guard, which bans that literal outside `web_panel`."""
    assert local_auth._OK_PAGE.split("<h1>")[0] == web_panel.render(
        "Opyt", ok=True, headline="any", detail="any").split("<h1>")[0]


def test_the_callback_page_names_no_filesystem_path():
    """It named the keys file until 2026-09-09 — first as a hardcoded `~/.opyt/.env`, wrong
    under `$OPYT_HOME`, then as the resolved path, right and still not wanted. Somebody who has
    just authorised a third party needs the reassurance and the next step, not the location of
    a dotfile holding a secret.

    `env_path()` is COMPUTED rather than spelled here, so this holds on any machine and under
    any `$OPYT_HOME`."""
    page = local_auth._OK_PAGE

    assert str(keys.env_path()) not in page
    assert ".env" not in page
    assert "~" not in page
