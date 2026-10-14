"""
service/x_signin_page.py — what a person sees when they open an X sign-in link from their
assistant: the streamed sign-in window, and the panel shown instead when a link cannot open.

Its phone keyboard, reconnect and "Start over" come from `/static/desktop.js`. There is no "Done
signing in" button: the desktop closes itself on X's home page, and this page polls for that. The
remote screen is laptop-sized (X refuses a phone-shaped one) with Chrome's window narrow in its
top-left corner, so this page shows only that window, which holds X's whole sign-in form and fits
a phone's width.
"""
from __future__ import annotations

import json

from starlette.responses import HTMLResponse

from opyt_core import web_panel


def page_html(nonce: str) -> str:
    """The page for one link. It carries only the link's own claim path: the tab's token comes
    from claiming, so a fetch that runs no script (a link scanner) gets a page and nothing else."""
    return (_PAGE.replace("__CSS__", _CSS)
            .replace("__NONCE__", json.dumps(nonce))
            .replace("__CLAIM__", json.dumps(f"/x-signin/{nonce}/claim")))


def refusal(headline: str, status_code: int) -> HTMLResponse:
    """A link that cannot open: says what to do next, never why."""
    return HTMLResponse(
        web_panel.render("Connect X to Opyt", ok=False, headline=headline,
                         detail="Go back to your conversation and ask to connect X again."),
        status_code=status_code, headers={"Cache-Control": "no-store"})


_CSS = """:root { color-scheme: light; --bg:#fff; --surface:#f4f4f4; --text:#141414;
  --muted:#5c5c5c; --faint:#8f8f8f; --accent:#dd3418; --line:#e3e3e3; }
* { box-sizing: border-box; }
body { min-height:100vh; margin:0; background:var(--bg); color:var(--text); font:14px/1.6
  ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
.shell { display:flex; flex-direction:column; gap:.5rem; height:100vh;
  height:var(--visual-height, 100dvh); width:min(1600px, 100%); margin:0 auto;
  padding:1.75rem 1rem 1.25rem; }
.top { display:flex; align-items:center; gap:.9rem; }
.brand { display:flex; align-items:center; gap:.45rem; font-size:.95rem; font-weight:600;
  color:var(--muted); }
.tile { width:20px; height:20px; border-radius:5px; }
h1 { margin:0; padding-left:.9rem; border-left:1px solid var(--line); font-size:.95rem;
  font-weight:600; }
#status { margin:0; max-width:76ch; color:var(--muted); font-size:.82rem; }
.stage { position:relative; flex:1; min-height:0; display:grid; place-items:start center;
  overflow:hidden; }
.stage.typing { align-items:start; overflow:hidden; }
#keys { position:absolute; top:0; left:0; width:1px; height:1px; padding:0; border:0;
  opacity:0; pointer-events:none; }
/* Chrome's window: a clip over the top-left corner of the whole remote screen, drawn inside it.
   The script sizes both once the desktop reports its shape; this is the empty box until then. */
#window { position:relative; width:min(100%, 500px); aspect-ratio:5/9; max-height:100%;
  overflow:hidden; border:1px solid var(--line); border-radius:10px;
  background:var(--surface); }
#screen { width:var(--screen-w, 100%); height:var(--screen-h, 100%); }
/* Over the window until X has drawn its page: the browser on the server takes seconds to load it,
   and a blank white box reads as broken. */
#loading { position:absolute; inset:0; display:grid; place-items:center; padding:1.5rem;
  background:var(--surface); color:var(--muted); font-size:.85rem; text-align:center; }
#loading.gone { display:none; }
#loading .spin { display:inline-block; }
.live { border-color:var(--accent); box-shadow:0 0 0 3px rgba(221,52,24,.13); }
.actions { display:flex; justify-content:flex-end; gap:.6rem; }
#keyboard, #restart { min-height:2.6rem; border:1px solid var(--line); border-radius:7px;
  padding:0 1.1rem; background:var(--bg); color:var(--text); font:600 .82rem inherit;
  white-space:nowrap; cursor:pointer; }
#restart:disabled { color:var(--faint); cursor:default; }
#keyboard { display:none; }
#keyboard[aria-pressed=true] { border-color:var(--accent); color:var(--accent); }
@media (pointer: coarse) { #keyboard { display:block; } }
.spin { display:none; width:.8em; height:.8em; margin-right:.5em; vertical-align:-.08em;
  border:2px solid currentColor; border-top-color:transparent; border-radius:50%;
  animation:spin .7s linear infinite; }
#status.busy .spin { display:inline-block; }
@keyframes spin { to { transform:rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spin { animation-duration:2.4s; } }
#done { display:none; place-items:center; width:100%; height:100%; border:1px solid var(--line);
  border-radius:10px; background:var(--surface); text-align:center; padding:2rem 1rem; }
#done.show { display:grid; }
#done .mark { display:grid; width:3.4rem; height:3.4rem; margin:0 auto 1.1rem; place-items:center;
  border-radius:50%; background:var(--accent); color:#fff; font-size:1.7rem; line-height:1; }
#done.failed .mark { background:var(--faint); }
#done h2 { margin:0 0 .5rem; font-size:1.15rem; letter-spacing:-.02em; }
#done p { margin:0; color:var(--muted); }
@media (max-width:560px) { .shell { padding:.9rem .7rem 1rem; gap:.45rem; }
  .top { gap:.6rem; } h1 { padding-left:.6rem; } #status { font-size:.78rem; }
  #keyboard, #restart { flex:1 1 0; min-height:2.75rem; padding:0 .5rem; } }
"""


# noVNC and `desktop.js` are vendored in `service/static/` and served at `/static/`.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name=viewport content="width=device-width,initial-scale=1">
<meta name=theme-color content="#ffffff"><title>Connect X · Opyt</title>
<link rel="icon" href="/favicon.ico" sizes="48x48">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.png"><style>
__CSS__</style></head><body><main class=shell>
<div class=top><div class=brand><svg class=tile viewBox="0 0 64 64" aria-hidden="true"><rect
width="64" height="64" fill="#141414"/><path d="M18 16 L36 32 L18 48" stroke="#ffffff"
stroke-width="7" fill="none"/><rect x="40" y="41" width="11" height="7" fill="#dd3418"/></svg>Opyt
</div><h1>Connect your X account</h1></div>
<p id=status><span class=spin aria-hidden="true"></span><span id=status-text>Opening a private
sign-in window…</span></p>
<noscript><p id=status>This page needs JavaScript to show the sign-in window.</p></noscript>
<div class=stage>
<textarea id=keys autocomplete=off autocorrect=off autocapitalize=off spellcheck=false
  aria-label="Type into the X sign-in window" tabindex=-1></textarea>
<div id=window><div id=screen aria-label="X sign-in window"></div>
<div id=loading role=status><p><span class=spin aria-hidden="true"></span>Opening X\u2019s sign-in
page\u2026</p></div></div>
<div id=done role=status aria-live=polite><div><p class=mark aria-hidden="true">&#10003;</p>
<h2 id=done-title></h2><p id=done-detail></p></div></div>
</div>
<div class=actions id=actions><button id=keyboard type=button aria-pressed=false
  data-show="Keyboard" data-hide="Hide keyboard">Keyboard</button>
<button id=restart type=button>Start over</button></div>
</main><script type=module>
import {attachDesktop} from '/static/desktop.js';
const status = document.getElementById('status');
const statusText = document.getElementById('status-text');
const stage = document.querySelector('.stage');
const screen = document.getElementById('screen');
const crop = document.getElementById('window');
const loading = document.getElementById('loading');
const OPENING = 'Opening X\u2019s sign-in page\u2026';
const done = document.getElementById('done');
const AGAIN = 'Go back to your conversation and ask to connect X again.';
let finished = false;
let desktop = null;

function setStatus(text, busy) {
  statusText.textContent = text;
  status.classList.toggle('busy', Boolean(busy));
}
setStatus(OPENING, true);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function finish(ok, title, detail) {
  if (finished) { return; }
  finished = true;
  if (desktop) { desktop.stop(); }
  crop.style.display = 'none';
  document.getElementById('actions').style.display = 'none';
  setStatus('');
  done.classList.toggle('failed', !ok);
  done.querySelector('.mark').textContent = ok ? '\\u2713' : '!';
  document.getElementById('done-title').textContent = title;
  document.getElementById('done-detail').textContent = detail;
  done.classList.add('show');
}

function ended(state) {
  if (state === 'connected') {
    finish(true, 'X is connected', 'Go back to your conversation. Opyt takes it from here.');
  } else if (state === 'failed') {
    finish(false, 'X was not connected', AGAIN);
  } else {
    finish(false, 'This sign-in expired', AGAIN);
  }
}

// Claim the sign-in for this tab. Loading the page claimed nothing, because link scanners fetch
// links before people do; the tab's token is kept for this tab only, so a reload reconnects to the
// same sign-in and any other tab is told it is open elsewhere.
const KEY = 'opyt-x-signin:' + __NONCE__;
const kept = (() => { try { return sessionStorage.getItem(KEY); } catch (_) { return null; } })();
let claimed;
try {
  const response = await fetch(__CLAIM__, {
    method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session: kept}),
  });
  if (!response.ok) { throw new Error(String(response.status)); }
  claimed = await response.json();
} catch (_) {
  finish(false, 'The sign-in window could not open', AGAIN);
  throw _;
}
if (claimed.status === 'taken') {
  finish(false, 'This sign-in is open in another window',
         'Finish it there, or go back to your conversation and ask to connect X again.');
  throw new Error('taken');
}
if (claimed.status !== 'claimed') { ended(claimed.status); throw new Error(claimed.status); }
try { sessionStorage.setItem(KEY, claimed.session); } catch (_) { /* no reconnect on reload */ }
const BASE = '/x-signin/s/' + claimed.session;

// One desktop runs at a time, so a page can arrive while someone else is signing in. It waits
// its turn here: the link stays good until it expires.
let started;
for (;;) {
  try {
    const response = await fetch(BASE + '/start', {method: 'POST', credentials: 'same-origin'});
    if (!response.ok) { throw new Error(String(response.status)); }
    started = await response.json();
  } catch (_) {
    finish(false, 'The sign-in window could not open', AGAIN);
    throw _;
  }
  if (started.status !== 'busy') { break; }
  setStatus('Someone else is signing in right now. Yours opens as soon as they finish…', true);
  await sleep(4000);
}
if (started.status !== 'running') { ended(started.status); throw new Error(started.status); }

// The whole remote screen is drawn at one scale inside #window, which clips it to Chrome's window
// at its top-left corner, so nothing ever scrolls sideways (sideways scrolling did not hold on a
// phone). A phone fits the window to its width: X's form is in the window's top half,
// and with the keyboard up the page scrolls to the field tapped (`/static/desktop.js`). A laptop
// fits the whole window. CSS variables, not inline sizes, carry the screen's size, because
// desktop.js clears the inline height when the keyboard goes away.
const touch = window.matchMedia('(pointer: coarse)').matches;
const area = stage.getBoundingClientRect();
const win = started.window;
const scale = touch || area.width < 700 ? area.width / win.width
  : Math.min(area.width / win.width, area.height / win.height);
crop.style.width = Math.round(win.width * scale) + 'px';
crop.style.height = Math.round(win.height * scale) + 'px';
crop.style.aspectRatio = 'auto';
crop.style.maxHeight = 'none';
screen.style.setProperty('--screen-w', Math.round(started.width * scale) + 'px');
screen.style.setProperty('--screen-h', Math.round(started.height * scale) + 'px');

const STEPS = window.matchMedia('(pointer: coarse)').matches
  ? 'Sign in to X: tap where you want to type. This page finishes by itself.'
  : 'Sign in to X. This page finishes by itself once you are in.';
// Whether X has drawn its page in the window yet; the server says so through `status`.
let drawn = false;
function showDrawn(isDrawn) {
  drawn = isDrawn;
  loading.classList.toggle('gone', isDrawn);
  setStatus(isDrawn ? STEPS : OPENING, !isDrawn);
}
desktop = attachDesktop({
  screen, stage, keys: document.getElementById('keys'),
  button: document.getElementById('keyboard'),
  streamPath: BASE + '/stream', expiresIn: started.expires_in,
  onStatus(state) {
    if (finished) { return; }
    crop.classList.toggle('live', state === 'live');
    if (state === 'live') { setStatus(drawn ? STEPS : OPENING, !drawn); }
    else if (state === 'lost') { setStatus('Reconnecting to the sign-in window…', true); }
  },
});

// The desktop closes itself on X's home page; this is how the page learns it did. In between,
// it holds the browser open until Chrome has written the session down, up to half a minute.
// While X's page is still loading this asks often, so the window shows the moment it is drawn.
(async () => {
  while (!finished) {
    await sleep(drawn ? 2000 : 500);
    try {
      const response = await fetch(BASE + '/status', {credentials: 'same-origin'});
      const state = (await response.json()).status;
      if (state === 'loading' || state === 'running') {
        if (drawn !== (state === 'running')) { showDrawn(state === 'running'); }
      } else if (state === 'finishing') {
        desktop.stop();
        crop.style.display = 'none';
        document.getElementById('actions').style.display = 'none';
        setStatus('You are signed in. Opyt is finishing up; this can take half a minute\u2026',
                  true);
      } else {
        ended(state);
      }
    } catch (_) { /* the next poll tries again */ }
  }
})();

// X's "Continue with Google" opens a window with no way to close it (the screen has no window
// manager), so the way out lives on this page.
const restart = document.getElementById('restart');
restart.onclick = async () => {
  restart.disabled = true;
  setStatus('Reopening X\\u2019s sign-in page\\u2026', true);
  try {
    const response = await fetch(BASE + '/restart', {method: 'POST', credentials: 'same-origin'});
    if (!response.ok) { throw new Error(String(response.status)); }
  } catch (_) {
    setStatus('The sign-in page could not be reopened. ' + AGAIN);
    return;
  }
  showDrawn(false);
  restart.disabled = false;
};
</script></body></html>"""
