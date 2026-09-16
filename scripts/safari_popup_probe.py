"""Measure what Safari actually does to a `window.open`ed window, one features string at a time.

    python scripts/safari_popup_probe.py     # then open http://127.0.0.1:8782/ in SAFARI

Three fixes for Safari's blank popup shipped between 2026-09-11 and 2026-09-12 and all three
were wrong, because all three were REASONED. The symptom rules out the obvious cause: resizing
the window by hand makes the content appear, so the page loaded and the window did not lay out.
Nothing short of Safari itself can say whether that window is zero-sized or sized-and-unpainted,
and those two answers have different fixes. This asks it.

Each row below opens a window with one candidate features string. The opened page is served from
here, so it can POST what it sees back — `innerWidth`, `innerHeight`, `outerWidth`,
`outerHeight`, `document.readyState`, the body's laid-out rect, and a requestAnimationFrame
count, sampled at load, at 150 ms (when the shipped nudge fires), at 600 ms, and on every resize.
The resize sample is the decisive one: it is the exact gesture that fixes the window by hand, so
its before/after numbers say what the gesture changed.

⚠️ The probe target is same-origin and trivial. Substack is neither, so each row also has a
button that opens the REAL sign-in URL with the same features string — unmeasurable, but David
can see whether it comes up blank. A variant that paints the probe and not Substack means the
failure is in the loaded document, not in the window.

Read the results two ways: they render on the opener page as they arrive, and every one is
printed here with its variant name.
"""
from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

PORT = 8782

# The candidates, in the order they are worth trying. Rows 1-3 are the three that shipped and
# failed, kept so this run says what each of them actually did rather than what it was believed
# to do. Rows 4-6 are the untried ones, and row 4 is the oldest known WebKit workaround.
#
# `blank_first` is not a features string but a sequencing choice: open about:blank, THEN assign
# `location.href`. `nudge` replays the shipped `resizeTo` pair at 150 ms.
_SIZE = "width=480,height=720"
VARIANTS = [
    ("row1-sized-only", f"{_SIZE},left=__LEFT__,top=__TOP__", False, False,
     "sizing only, no popup=1 — what shipped first; gave a TAB"),
    ("row2-popup-chromeless", f"popup=1,{_SIZE},left=__LEFT__,top=__TOP__,"
     "menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=yes", False, False,
     "popup=1 with suppressed chrome — gave a BLANK window"),
    ("row3-popup-plain-nudge", f"popup=1,{_SIZE},left=__LEFT__,top=__TOP__", False, True,
     "popup=1 plain + resizeTo nudge — LIVE ON PRODUCTION, still blank"),
    ("row4-blank-first", f"popup=1,{_SIZE},left=__LEFT__,top=__TOP__", True, False,
     "open about:blank first, then set location.href — NEVER TRIED"),
    ("row5-blank-first-nudge", f"popup=1,{_SIZE},left=__LEFT__,top=__TOP__", True, True,
     "blank first AND the nudge — never tried"),
    ("row6-sized-only-nudge", f"{_SIZE},left=__LEFT__,top=__TOP__", False, True,
     "row 1's features with row 3's nudge — never tried; row 1 at least painted"),
    # THE CANDIDATE FIX. The blank window transferred zero bytes and a reload always renders, so
    # this re-assigns `location.href` once at 600 ms. An opener may write a cross-origin
    # window's location -- it may not READ it -- so this is the one repair available without
    # being able to tell a blank window from a good one.
    ("row7-renavigate", f"popup=1,{_SIZE},left=__LEFT__,top=__TOP__", False, False,
     "production's features + one re-assignment of location.href at 600 ms", True),
    # Chrome suppression was stripped from Safari on the theory that it caused the blank window.
    # That theory is dead, so the flags are worth having back: they are what makes this a
    # chromeless popup instead of a full browser window. This checks they do not bring the blank
    # back when the re-navigation is also in play.
    ("row8-chromeless-renavigate",
     f"popup=1,{_SIZE},left=__LEFT__,top=__TOP__,"
     "menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=yes", False, False,
     "suppressed chrome AND the re-navigation — what Chrome already gets", True),
]

_OPENER = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safari popup probe</title>
<style>
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 32px; max-width: 900px; }
  h1 { font-size: 19px; }
  p.lede { color: #444; }
  table { border-collapse: collapse; width: 100%; margin-top: 20px; }
  td, th { border-bottom: 1px solid #ddd; padding: 10px 8px; vertical-align: top;
           text-align: left; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  button { font: inherit; padding: 6px 12px; margin: 0 4px 4px 0; cursor: pointer; }
  .note { font-size: 13px; color: #666; }
  #out { white-space: pre-wrap; font: 12px/1.45 ui-monospace, Menlo, monospace;
         background: #111; color: #d8d8d8; padding: 14px; border-radius: 6px;
         margin-top: 24px; min-height: 120px; }
  .ua { font-size: 12px; color: #666; margin-top: 8px; }
</style>
<h1>Safari popup probe</h1>
<p class="lede">Click <b>probe</b> on a row: it opens a window with that features string, and
that window reports its own size and layout back here. Then click <b>Substack</b> on the same
row and just look at it — blank or not. Close each window before the next row.</p>
<p class="note">⚠️ Run this in <b>Safari</b>. In Chrome every row will pass and nothing is
learned.</p>
<table id="rows"></table>
<div class="ua"></div>
<div id="out">waiting for the first report…</div>
<script>
const VARIANTS = __VARIANTS__;
const SUBSTACK = 'https://substack.com/sign-in';
const out = document.getElementById('out');
let lines = [];

function log(text) {
  lines.push(text);
  out.textContent = lines.join('\\n');
}

// `left`/`top` are resolved HERE rather than baked into the string, because they are relative to
// this window and the shipped code computes them the same way.
function features(raw) {
  const left = Math.max(window.screenX, window.screenX + window.outerWidth - 480 - 24);
  return raw.replace('__LEFT__', left).replace('__TOP__', window.screenY + 48);
}

function nudge(popup) {
  setTimeout(() => {
    try { popup.resizeTo(480, 719); popup.resizeTo(480, 720); log('  … nudge fired'); }
    catch (e) { log('  … nudge THREW: ' + e); }
  }, 150);
}

// ⚠️ The window NAME is a variable, not a constant. The shipped code passes a fixed
// 'opyt-substack' every time, and Safari's choice of tab-vs-window was measured on 2026-09-12 to
// ALTERNATE across repeated identical calls -- so whether a window by that name was opened
// before is a candidate cause, and has to be switchable to be ruled in or out.
const UNIQUE_NAMES = new URLSearchParams(location.search).get('names') === 'unique';
let opens = 0;

function run(v, target, measured) {
  const feat = features(v.features);
  const winName = UNIQUE_NAMES ? 'probe-' + v.name + '-' + (++opens) : 'probe-' + v.name;
  log('▶ ' + v.name + (measured ? ' [probe]' : ' [substack]') + '\\n  features: ' + feat +
      (v.blankFirst ? '\\n  sequencing: about:blank first, then location.href' : ''));
  let popup;
  if (v.blankFirst) {
    popup = window.open('', winName, feat);
    if (popup) popup.location.href = target;
  } else {
    popup = window.open(target, winName, feat);
  }
  if (!popup) { log('  ✗ window.open returned NULL — blocked'); return; }
  log('  opened; opener sees closed=' + popup.closed);
  if (v.nudge) nudge(popup);
  // Writing another window's location is allowed across origins; reading it is not. So this
  // cannot check whether the window came up blank -- it re-navigates unconditionally.
  if (v.renav) {
    setTimeout(() => {
      try { popup.location.href = target; log('  … re-navigated'); }
      catch (e) { log('  … re-navigate THREW: ' + e); }
    }, 600);
  }
  // Read the window back from OUT here too: if the opener's view of its size disagrees with the
  // window's own view, that gap is itself the finding.
  setTimeout(() => {
    try { log('  opener sees outer ' + popup.outerWidth + '×' + popup.outerHeight +
               ' at ' + popup.screenX + ',' + popup.screenY); }
    catch (e) { log('  opener cannot read it back (cross-origin, expected for substack)'); }
  }, 700);
}

const rows = document.getElementById('rows');
rows.innerHTML = '<tr><th>variant</th><th>what it is</th><th></th></tr>';
for (const v of VARIANTS) {
  const tr = document.createElement('tr');
  const name = document.createElement('td');
  name.innerHTML = '<code>' + v.name + '</code>';
  const what = document.createElement('td');
  what.textContent = v.note;
  const act = document.createElement('td');
  const probe = document.createElement('button');
  probe.textContent = 'probe';
  // Ids so WebDriver can click these. A WebDriver click carries user activation; calling
  // .click() from an injected script does not, and window.open without activation is blocked --
  // which would read as "Safari refused the popup" and be an artifact of the instrument.
  probe.id = 'probe-' + v.name;
  probe.onclick = () => run(v, '/target?v=' + encodeURIComponent(v.name), true);
  const real = document.createElement('button');
  real.textContent = 'Substack';
  real.id = 'real-' + v.name;
  real.onclick = () => run(v, SUBSTACK, false);
  act.append(probe, real);
  tr.append(name, what, act);
  rows.append(tr);
}
document.querySelector('.ua').textContent = navigator.userAgent;

// The probe target reports by postMessage (instant, and works even if it never paints) and by
// POST (so the server log keeps it after this page is gone).
window.addEventListener('message', (e) => {
  if (e.origin !== location.origin || !e.data || e.data.kind !== 'probe') return;
  const d = e.data;
  log('  ◀ ' + d.variant + ' @' + d.at +
      ': inner ' + d.innerWidth + '×' + d.innerHeight +
      ', outer ' + d.outerWidth + '×' + d.outerHeight +
      ', body ' + d.bodyWidth + '×' + d.bodyHeight +
      ', readyState=' + d.readyState + ', frames=' + d.frames +
      ', visibility=' + d.visibility);
});
</script>
"""

_TARGET = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>probe target</title>
<style>
  html, body { height: 100%; margin: 0; }
  body { background: #1d4ed8; color: #fff; font: 16px/1.5 -apple-system, system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; justify-content: center;
         text-align: center; padding: 20px; box-sizing: border-box; }
  h1 { font-size: 22px; margin: 0 0 12px; }
  pre { font: 12px/1.4 ui-monospace, Menlo, monospace; background: rgba(0,0,0,.28);
        padding: 12px; border-radius: 6px; text-align: left; max-width: 100%;
        overflow: auto; }
</style>
<h1>If you can read this, the window painted.</h1>
<p>Resize this window by hand — that is the gesture being measured.</p>
<pre id="log">…</pre>
<script>
// A blue page with big text: "did it paint" is answerable by eye, and the numbers below say why.
const variant = new URLSearchParams(location.search).get('v') || 'unknown';
let frames = 0;
(function count() { frames++; requestAnimationFrame(count); })();

function sample(at) {
  const rect = document.body.getBoundingClientRect();
  const d = {
    kind: 'probe', variant, at,
    innerWidth: innerWidth, innerHeight: innerHeight,
    outerWidth: outerWidth, outerHeight: outerHeight,
    screenX: screenX, screenY: screenY,
    bodyWidth: Math.round(rect.width), bodyHeight: Math.round(rect.height),
    readyState: document.readyState, visibility: document.visibilityState,
    devicePixelRatio: devicePixelRatio, frames: frames,
    // The OPENER's state, because a Safari window in macOS full screen cannot put a sized
    // popup beside itself -- it has no beside. If the opener's outer size equals the whole
    // screen and its origin is 0,0, that is what happened, and no features string can fix it.
    screenAvail: [screen.availWidth, screen.availHeight],
    screenTotal: [screen.width, screen.height],
    openerOuter: opener ? [opener.outerWidth, opener.outerHeight] : null,
    openerAt: opener ? [opener.screenX, opener.screenY] : null,
  };
  document.getElementById('log').textContent = JSON.stringify(d, null, 1);
  try { if (opener) opener.postMessage(d, location.origin); } catch (e) {}
  // keepalive so a sample taken as the window closes still lands.
  try { fetch('/report', {method: 'POST', keepalive: true,
                          headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify(d)}); } catch (e) {}
}

sample('script-run');
addEventListener('load', () => sample('load'));
// 150 ms is when the shipped nudge fires, so this brackets it: the pair says whether the nudge
// changed anything at all.
setTimeout(() => sample('t=140ms-pre-nudge'), 140);
setTimeout(() => sample('t=300ms-post-nudge'), 300);
setTimeout(() => sample('t=1200ms'), 1200);
let resizes = 0;
addEventListener('resize', () => sample('resize#' + (++resizes)));
</script>
"""


def _as_dicts() -> list[dict]:
    """One shape for both the page and the WebDriver runner, so they can never disagree."""
    return [{"name": v[0], "features": v[1], "blankFirst": v[2], "nudge": v[3], "note": v[4],
             "renav": v[5] if len(v) > 5 else False} for v in VARIANTS]


async def opener(request):
    payload = json.dumps(_as_dicts())
    return HTMLResponse(_OPENER.replace("__VARIANTS__", payload),
                        headers={"Cache-Control": "no-store"})


async def target(request):
    return HTMLResponse(_TARGET, headers={"Cache-Control": "no-store"})


async def report(request):
    d = await request.json()
    print(f"  {d.get('variant'):<24} {str(d.get('at')):<20} "
          f"inner {d.get('innerWidth')}×{d.get('innerHeight')}  "
          f"outer {d.get('outerWidth')}×{d.get('outerHeight')}  "
          f"body {d.get('bodyWidth')}×{d.get('bodyHeight')}  "
          f"ready={d.get('readyState')} frames={d.get('frames')} "
          f"vis={d.get('visibility')} dpr={d.get('devicePixelRatio')}\n"
          f"  {'':<24} {'':<20} opener {d.get('openerOuter')} at {d.get('openerAt')}  "
          f"screen avail {d.get('screenAvail')} total {d.get('screenTotal')}", flush=True)
    return JSONResponse({"ok": True})


async def variants(request):
    """The table as JSON, so a WebDriver runner iterates the SAME list the page renders."""
    return JSONResponse(_as_dicts())


app = Starlette(routes=[
    Route("/", opener),
    Route("/variants", variants),
    Route("/target", target),
    Route("/report", report, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    print(f"  popup probe on http://127.0.0.1:{PORT}/   — open it in SAFARI, not Chrome")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
