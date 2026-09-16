/* One hosted sign-in desktop, attached to a page.
 *
 * Three things live here that plain noVNC does not give a phone, and both sign-in pages import
 * them rather than carrying two copies of a diff algorithm and a retry loop:
 *
 *   1. A keyboard. noVNC attaches its `Keyboard` to the <canvas>, and iOS and Android open the
 *      on-screen keyboard only for a focused EDITABLE element, so on a phone the canvas alone
 *      can be tapped but never typed into.
 *   2. A reconnect. Reading a 2FA code means leaving the browser, and a backgrounded tab loses
 *      its WebSocket while the desktop on the box keeps running.
 *   3. Room for the keyboard, which covers the bottom of the screen without changing any CSS
 *      length the layout can see, and a way to still SEE the field being typed into once it
 *      has taken half the screen.
 *
 * Design record: docs/plans/2026-09-09-hosted-signin-from-a-phone.md.
 */
import RFB from '/static/novnc/core/rfb.js';
import Keyboard from '/static/novnc/core/input/keyboard.js';
import keysyms from '/static/novnc/core/input/keysymdef.js';
import KeyTable from '/static/novnc/core/input/keysym.js';

// The hidden field always holds this much text. Backspace is a key event we can only infer from
// text going missing, so there has to be text to lose before the user has typed anything.
const BUFFER_LENGTH = 100;
const BUFFER = '_'.repeat(BUFFER_LENGTH);
// Retry spacing. The first drop is usually the tab coming back from the background and connects
// at once; the ceiling is low because the desktop's whole life is minutes.
const RETRY_STEP_MS = 1000;
const RETRY_CEILING_MS = 5000;
// How long out of sight makes a live-looking connection worth replacing on return.
const STALE_AWAY_MS = 5000;

/* Attach to one desktop.
 *
 *   screen      the element noVNC draws into
 *   stage       the element `screen` sits in; the window onto it while the keyboard is up
 *   keys        a hidden <textarea>: what the phone's keyboard actually types into
 *   button      the control that raises and dismisses that keyboard
 *   streamPath  the stream capability, same-origin, ws: or wss: chosen from the page
 *   expiresIn   seconds the capability has left, from the server that minted it, so the page
 *               never carries its own copy of the sign-in TTL
 *   onStatus    'live' | 'lost' | 'gone'
 */
export function attachDesktop({screen, stage, keys, button, streamPath, expiresIn,
                               onStatus}) {
  const endpoint = new URL(streamPath, window.location.href);
  endpoint.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const deadline = Date.now() + expiresIn * 1000;

  let rfb = null;
  let stopped = false;
  let replacing = false;
  let attempts = 0;
  let timer = null;
  let hiddenAt = 0;

  function connect() {
    timer = null;
    rfb = new RFB(screen, endpoint.href);
    rfb.scaleViewport = true;
    rfb.clipViewport = true;
    rfb.showDotCursor = true;
    // A tap must not move focus to the canvas while the user is typing: that closes the
    // phone's keyboard mid-password. noVNC's own gesture handler already calls
    // preventDefault() on canvas touches, so nothing else is needed to hold the focus.
    rfb.focusOnClick = document.activeElement !== keys;
    rfb.addEventListener('connect', () => {
      attempts = 0;
      onStatus('live');
    });
    rfb.addEventListener('disconnect', () => {
      rfb = null;
      if (replacing) { replacing = false; connect(); return; }
      if (stopped) { return; }
      if (Date.now() >= deadline) { onStatus('gone'); return; }
      onStatus('lost');
      retry();
    });
  }

  function retry() {
    if (stopped || timer !== null) { return; }
    attempts += 1;
    timer = setTimeout(connect, Math.min(attempts * RETRY_STEP_MS, RETRY_CEILING_MS));
  }

  // Coming back from another app is both the moment a drop gets noticed and the moment a
  // connection can no longer be trusted. A phone that was suspended, or that changed network
  // on the way, leaves a socket that is dead without being closed: no frame ever arrives and
  // nothing fires, so the desktop simply stops moving. Replacing it costs one full frame and
  // about 150 ms, and it is the only way to be sure.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') { hiddenAt = Date.now(); return; }
    if (stopped || Date.now() >= deadline) { return; }
    // Long enough that pulling down a notification is not a reconnect, short enough that any
    // trip to a mail or authenticator app is.
    const away = Date.now() - hiddenAt;
    if (rfb !== null && away < STALE_AWAY_MS) { return; }
    clearTimeout(timer);
    timer = null;
    attempts = 0;
    if (rfb === null) { connect(); return; }
    replacing = true;
    rfb.disconnect();
  });

  bindKeyboard({keys, button, screen, stage,
                send: (...args) => rfb && rfb.sendKey(...args),
                focus: (wanted) => { if (rfb) { rfb.focusOnClick = wanted; } }});
  trackSoftKeyboard();
  connect();

  return {
    stop() {
      stopped = true;
      clearTimeout(timer);
      timer = null;
      if (rfb) { rfb.disconnect(); rfb = null; }
    },
  };
}

/* The keyboard, in two layers, because no single one covers every phone.
 *
 * noVNC's `Keyboard` on the hidden field handles every key event the browser identifies: a
 * hardware keyboard, and Enter and Backspace from iOS. It calls preventDefault() on those, so
 * the character never reaches the field and no second event follows.
 *
 * What it cannot identify falls through to the field's value: Android reports keyCode 229 for
 * on-screen keys, and autocorrect rewrites whole words with no key event at all. Diffing the
 * value against the last one turns both into key events. This is noVNC's own algorithm from
 * `app/ui.js`, which is where the underscore buffer and the reset rules come from too.
 */
function bindKeyboard({keys, button, screen, stage, send, focus}) {
  const keyboard = new Keyboard(keys);
  keyboard.onkeyevent = (keysym, code, down) => send(keysym, code, down);
  keyboard.grab();

  let last = BUFFER;

  function reset() {
    keys.value = BUFFER;
    last = BUFFER;
    keys.setSelectionRange(BUFFER_LENGTH, BUFFER_LENGTH);
  }

  keys.addEventListener('input', () => {
    const now = keys.value;
    // Trailing whitespace does not always reach `value.length`, so prefer the caret.
    const newLength = Math.max(keys.selectionStart, now.length);
    let inserted = newLength - last.length;
    let deleted = inserted < 0 ? -inserted : 0;
    // A correction rewrites the middle of the buffer without changing its length, so compare
    // rather than trusting the lengths: the first character that differs is where to start.
    for (let i = 0; i < Math.min(last.length, newLength); i++) {
      if (now.charAt(i) !== last.charAt(i)) {
        inserted = newLength - i;
        deleted = last.length - i;
        break;
      }
    }
    for (let i = 0; i < deleted; i++) {
      send(KeyTable.XK_BackSpace, 'Backspace');
    }
    for (let i = newLength - inserted; i < newLength; i++) {
      send(keysyms.lookup(now.charCodeAt(i)));
    }
    if (newLength > 2 * BUFFER_LENGTH || newLength < 1) {
      reset();
    } else {
      last = now;
    }
  });

  // The button says which of the two things it will do next. Both words come from the page,
  // which owns every sentence a user reads; this only says which one is true.
  function label(open) {
    button.setAttribute('aria-pressed', open ? 'true' : 'false');
    const said = open ? button.dataset.hide : button.dataset.show;
    if (said) { button.textContent = said; }
  }

  /* The desktop while the keyboard is up.
   *
   * A phone's keyboard takes about half the screen, and contain-fitting a 500x959 desktop into
   * the strip left over renders it at about a tenth of its size: measured 2026-09-09 in a
   * 390x844 viewport, the stage falls to 368x216 and the canvas with it, to 112 CSS pixels
   * wide. Nothing on it is readable, and least of all the field being typed into, which is the
   * one thing the user needs to see at that exact moment.
   *
   * So the desktop KEEPS the size it had, and the stage becomes a window onto it, scrolled to
   * wherever the user last tapped. Hiding the keyboard puts the whole desktop back. Freezing
   * also makes this deterministic across browsers: measured the same day, Chromium re-fits the
   * canvas when the stage shrinks and iOS Safari does not, so one of the two was always wrong.
   */
  let tappedAt = null;
  let held = false;

  function reposition() {
    if (!held || stage === undefined || stage === null) { return; }
    const height = screen.getBoundingClientRect().height;
    const visible = stage.clientHeight;
    if (height <= visible) { stage.scrollTop = 0; return; }
    const wanted = tappedAt === null ? 0 : tappedAt - visible / 2;
    stage.scrollTop = Math.max(0, Math.min(wanted, height - visible));
  }

  function hold(wanted) {
    if (stage === undefined || stage === null || wanted === held) { return; }
    held = wanted;
    if (!held) {
      screen.style.height = '';
      screen.style.maxHeight = '';
      stage.classList.remove('typing');
      stage.scrollTop = 0;
      return;
    }
    // Taken BEFORE the keyboard arrives, which is the last moment the desktop is at the size
    // this is preserving. `max-height` has to go with it: it is a percentage of the stage, so
    // it would clamp the height back down the instant the stage shrinks.
    screen.style.height = `${screen.getBoundingClientRect().height}px`;
    screen.style.maxHeight = 'none';
    stage.classList.add('typing');
    reposition();
  }

  // The keyboard arrives over a fraction of a second, so the stage is still full height at the
  // moment focus lands and there is nothing to scroll yet. Watching the stage itself is the
  // direct signal, and it covers a rotation or a second keyboard row as well as the first
  // appearance -- `visualViewport` only reports the last of those.
  if (stage !== undefined && stage !== null) {
    new ResizeObserver(reposition).observe(stage);
  }

  keys.addEventListener('focus', () => {
    focus(false);
    label(true);
    hold(true);
  });
  keys.addEventListener('blur', () => {
    focus(true);
    label(false);
    hold(false);
  });

  // Which way the button goes has to be read one event EARLY. Pressing it blurs the field
  // first on any browser that focuses buttons (Chrome does, Safari does not), so by the time
  // the click arrives `document.activeElement` is the button and the answer is always "open
  // it" -- which is what the button did: it raised the keyboard and could never put it away.
  let wasOpen = false;
  button.addEventListener('pointerdown', () => { wasOpen = document.activeElement === keys; });
  button.addEventListener('click', () => {
    if (wasOpen) { keys.blur(); return; }
    reset();
    // The focus has to happen inside the click, not after an await: a phone only opens its
    // keyboard for a focus a user gesture caused.
    keys.focus();
  });

  // On a touch screen, a tap on the desktop is also how the keyboard gets raised.
  //
  // Nothing in RFB reports what the tap landed on, so this cannot tell a text field from a
  // button and does not try: every tap raises the keyboard, and the button beside the desktop
  // puts it away again. A spurious raise costs one tap and some of the screen; not raising it
  // costs a user who taps a field, sees no keyboard, and has no way to guess that a control
  // outside the desktop is what they needed. That was measured on a real phone, 2026-09-09.
  //
  // The focus is synchronous inside the gesture because a phone opens its keyboard for no
  // other kind of focus. On a pointer-fine device the canvas keeps handling keys exactly as
  // it did before this file existed, and the only tap that reaches here is a re-focus while
  // the field is already open.
  const touch = window.matchMedia('(pointer: coarse)').matches;
  screen.addEventListener('pointerdown', (event) => {
    // Where in the desktop the tap landed, in CSS pixels from its top. With the keyboard up
    // this is the field being typed into, by definition, and it is the only thing on the
    // desktop this page can locate: RFB says nothing about what is under a click.
    tappedAt = event.clientY - screen.getBoundingClientRect().top;
    if (touch || button.getAttribute('aria-pressed') === 'true') { keys.focus(); }
  });

  reset();
}

/* The on-screen keyboard covers the bottom of the screen and changes no CSS length: `dvh` is
 * about browser chrome, not the keyboard. `visualViewport` is the only thing that reports it,
 * so the pages size their column from `--visual-height` and the desktop stays whole while the
 * user types into it, at a smaller scale, instead of half of it sitting behind the keys.
 */
function trackSoftKeyboard() {
  const viewport = window.visualViewport;
  if (!viewport) { return; }
  const apply = () => {
    document.documentElement.style.setProperty('--visual-height', `${viewport.height}px`);
  };
  viewport.addEventListener('resize', apply);
  apply();
}
