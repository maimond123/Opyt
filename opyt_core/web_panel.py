"""
opyt_core/web_panel.py
The one Opyt-styled page served into a browser.

Four surfaces finish a browser-facing flow, and until 2026-09-09 only the hosted ones looked
like Opyt. The local loopback returned a bare `<h2>` and a `<p>` in the browser's default serif
— and that page is the FIRST thing a new local user ever sees, because the OpenRouter approval
is phase 1 of onboarding. Unstyled Times New Roman on white was the product's first impression.

Lives in `opyt_core` because BOTH homes render it and the dependency only runs one way: the
client wheel ships `opyt_core*` and deliberately excludes `gateway*` (see `pyproject.toml`'s
packages-find list), while the gateway is deployed from a full checkout with
`pip install -e ".[server]"`, so `opyt_core` is present on that box by construction. A client
that imported `gateway` would violate the distributable invariant; a gateway that imports
`opyt_core` is the sanctioned direction.

No template engine, no CSS file, no static assets. One function returning one string, because
the two servers that call it have no way to serve a second request for a stylesheet: the
loopback listener is single-request by design, and adding an asset route to it is the thing
`local_auth`'s docstring forbids.
"""
from __future__ import annotations

from html import escape as html_escape


def render(title: str, *, ok: bool, headline: str, detail: str) -> str:
    """One Opyt panel. `title` is the browser tab; `headline` and `detail` are the card.

    `ok` picks the glyph and its colour, and it means "did the flow the user just finished
    succeed", not "is this page an error page" — a failure still renders a full panel with the
    instruction that matters, which is how to get back to where they were.
    """
    mark = "&#10003;" if ok else "!"
    tone = "var(--accent)" if ok else "var(--faint)"
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
        f"<title>{html_escape(title)}</title><style>"
        ":root { color-scheme: light; --bg:#fff; --surface:#f4f4f4; --text:#141414;"
        "  --muted:#5c5c5c; --faint:#8f8f8f; --accent:#dd3418; --line:#e3e3e3; }"
        "body { min-height:100vh; margin:0; display:grid; place-items:center;"
        "  background:var(--bg); color:var(--text); font:14px/1.6 ui-monospace, 'SF Mono',"
        "  Menlo, Consolas, monospace; padding:1rem; }"
        ".card { width:min(30rem,100%); border:1px solid var(--line); border-radius:10px;"
        "  background:var(--surface); padding:2.5rem 1.5rem; text-align:center; }"
        ".brand { display:flex; align-items:center; justify-content:center; gap:.55rem;"
        "  margin-bottom:1.6rem; font-weight:600; }"
        ".tile { width:22px; height:22px; border-radius:4px; }"
        f".mark {{ display:grid; width:3.4rem; height:3.4rem; margin:0 auto 1.1rem;"
        f"  place-items:center; border-radius:50%; background:{tone}; color:#fff;"
        "  font-size:1.7rem; line-height:1; }"
        "h1 { margin:0 0 .5rem; font-size:1.15rem; letter-spacing:-.02em; }"
        "p { margin:0; color:var(--muted); }"
        "</style></head><body><main class=card>"
        "<div class=brand><svg class=tile viewBox=\"0 0 64 64\" aria-hidden=true>"
        "<rect width=64 height=64 fill=#141414 />"
        "<path d=\"M18 16 L36 32 L18 48\" stroke=#ffffff stroke-width=7 fill=none />"
        "<rect x=40 y=41 width=11 height=7 fill=#dd3418 /></svg>Opyt</div>"
        f"<p class=mark aria-hidden=true>{mark}</p>"
        f"<h1>{html_escape(headline)}</h1><p>{html_escape(detail)}</p>"
        "</main></body></html>")
