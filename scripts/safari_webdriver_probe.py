"""Drive REAL Safari through safaridriver and measure what a window actually received.

    safaridriver -p 4444 &                       # once; needs `safaridriver --enable` first
    python scripts/safari_popup_probe.py &       # the page the `open` mode clicks
    python scripts/safari_webdriver_probe.py direct https://substack.com/sign-in 2
    python scripts/safari_webdriver_probe.py open  row3-popup-plain-nudge 3

Three fixes for Safari's "blank popup" shipped on 2026-09-11/12 and all three were REASONED
rather than measured. All three were wrong, and this is the instrument that showed why: the
window is not unpainted, it is EMPTY. On the first navigation to substack.com in a fresh Safari
session the document commits with `transferSize: 0` and an `outerHTML` of exactly
`<html><head></head><body></body></html>` -- 39 characters -- while `readyState` reads
"complete" and `location.href` reads the right URL. It never fills in; a reload pulls the whole
45 KB page. Nothing about `window.open` is involved: `direct` reproduces it with no popup at all.

Modes:
    direct <url> [n]    navigate the session's own window. The control that acquits window.open.
    open <variant> [n]  click a variant button on the probe page, then measure the window it
                        opened. Variant names come from that page's /variants.
    reload <url>        navigate, measure, reload, measure. The before/after pair.

⚠️ Read `transferSize` before anything else. Zero means the bytes never arrived and no amount of
window sizing, chrome suppression or resize nudging can matter. `htmlLen` under 200 is the same
fact stated another way.

⚠️ safaridriver sessions get an EPHEMERAL profile -- no cookies, no history for the host. That
may be precisely the cold state that triggers this, so a finding here is worth confirming in a
normal Safari before it justifies shipping anything.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

DRIVER = "http://127.0.0.1:4444"
PROBE = "http://127.0.0.1:8782"
EL = "element-6066-11e4-a52e-4f735466cecf"

# `transferSize` and `encodedBodySize` are the whole point: they separate "Safari received the
# document and failed to show it" from "Safari received nothing", and only the first of those is
# a rendering bug.
MEASURE = """
const n = performance.getEntriesByType('navigation')[0] || {};
const r = document.body ? document.body.getBoundingClientRect() : {width: 0, height: 0};
return {
  href: location.href, title: document.title, ready: document.readyState,
  kids: document.body ? document.body.childElementCount : -1,
  htmlLen: document.documentElement.outerHTML.length,
  transferSize: n.transferSize, encodedSize: n.encodedBodySize,
  resources: performance.getEntriesByType('resource').length,
  outer: [outerWidth, outerHeight], inner: [innerWidth, innerHeight],
  body: [Math.round(r.width), Math.round(r.height)],
  laidOut: !!document.elementFromPoint(10, 10),
};
"""


def call(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{DRIVER}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["value"]
    except urllib.error.HTTPError as e:
        return {"__error__": json.loads(e.read() or b"{}")}


def show(label: str, g: dict) -> None:
    verdict = "BLANK" if g.get("htmlLen", 0) < 200 else "painted"
    print(f"  {label:<16} {verdict:<8} transfer={g.get('transferSize')} "
          f"kids={g.get('kids')} htmlLen={g.get('htmlLen')} res={g.get('resources')} "
          f"outer={g.get('outer')} laidOut={g.get('laidOut')} "
          f"ready={g.get('ready')} title={g.get('title')!r}", flush=True)


def measure(sid: str, label: str, settle: float = 3.5) -> dict:
    time.sleep(settle)
    g = call("POST", f"/session/{sid}/execute/sync", {"script": MEASURE, "args": []})
    show(label, g)
    return g


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "direct"
    arg = sys.argv[2] if len(sys.argv) > 2 else "https://substack.com/sign-in"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    sid = call("POST", "/session",
               {"capabilities": {"alwaysMatch": {"browserName": "Safari"}}})
    if "sessionId" not in sid:
        # The commonest cause by far, and the message WebDriver gives for it is opaque.
        sys.exit(f"no session: {sid}\n(is an older session still paired? restart safaridriver)")
    sid = sid["sessionId"]
    print(f"session {sid}  mode={mode} arg={arg}\n", flush=True)
    try:
        if mode == "reload":
            call("POST", f"/session/{sid}/url", {"url": arg})
            measure(sid, "first load")
            call("POST", f"/session/{sid}/refresh")
            measure(sid, "after reload")
            return
        if mode == "direct":
            for i in range(n):
                call("POST", f"/session/{sid}/url", {"url": arg})
                measure(sid, f"load #{i}")
            return
        # `open` mode. The click has to be a WebDriver click: it carries user activation, and a
        # script-driven .click() does not -- which would make this measure a different code path
        # than the one a user takes. When Safari is not frontmost the click silently does
        # nothing, which shows up here as "NO NEW WINDOW"; retry rather than reach for a script
        # click.
        call("POST", f"/session/{sid}/url", {"url": PROBE + "/?names=unique"})
        opener = call("GET", f"/session/{sid}/window")
        seen = set(call("GET", f"/session/{sid}/window/handles"))
        for i in range(n):
            call("POST", f"/session/{sid}/window", {"handle": opener})
            el = call("POST", f"/session/{sid}/element",
                      {"using": "css selector", "value": f"#real-{arg}"})
            if "__error__" in el:
                sys.exit(f"no button #real-{arg} — is the probe page serving on {PROBE}?")
            call("POST", f"/session/{sid}/element/{el[EL]}/click")
            fresh: list[str] = []
            deadline = time.time() + 20
            while time.time() < deadline:
                fresh = [h for h in call("GET", f"/session/{sid}/window/handles")
                         if h not in seen]
                if fresh:
                    break
                time.sleep(0.2)
            if not fresh:
                print(f"  open #{i}: NO NEW WINDOW (Safari not frontmost? retry)", flush=True)
                continue
            seen.add(fresh[0])
            call("POST", f"/session/{sid}/window", {"handle": fresh[0]})
            measure(sid, f"open #{i}")
    finally:
        call("DELETE", f"/session/{sid}")


if __name__ == "__main__":
    main()
