// service/pages/comments.js — comments on a page of sources, for anyone holding its link.
//
// The service keeps the comments and answers JSON (`service/pages/comments.py`). Everything a
// reader sees is drawn here. A comment is on a passage of one source or of the answer (item 0),
// on a whole source, or a reply to either. A passage is recorded as its text with the text around
// it, and found again in the source's text as this browser holds it, so a highlight never depends
// on how the page was drawn. On a wide screen a passage's comments sit in the right margin, level
// with it; on a narrow one they open from the bottom.
//
// Nobody signs in. The name a reader types is kept in this browser (`opyt-name`), and so is the
// key that lets them delete a comment they posted (`opyt-comment-keys`). If the comments cannot
// be read, nothing here is drawn and the page reads as it would without them.
(function () {
  "use strict";
  var main = document.querySelector("main"), root = document.documentElement;
  if (!main || !window.fetch) return;
  var api = location.pathname.replace(/\/+$/, "") + "/comments";
  var WIDE = matchMedia("(min-width: 1180px)"), COARSE = matchMedia("(pointer: coarse)");
  var CONTEXT = 64;   // characters kept either side of a passage (`comments.MAX_CONTEXT`)
  var BUBBLE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 3h11v8H8.3L5 13.8V11H2.5Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  var LINE = ' fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var PREV = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5"' + LINE;
  var NEXT = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3 5 5-5 5"' + LINE;
  var CLOSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 3.5 9 9m0-9-9 9"' + LINE;

  var threads = [];      // in reading order: {id, item, quote, prefix, suffix, all: [comment], marks: [mark], at}
  var limits = { body: 2000, name: 40, quote: 2000 };
  var active = null;     // the open thread's id; "new" while a comment is written; "none" for the empty sheet
  var draft = null;      // what the comment being written is on: {item, start, end, quote, prefix, suffix, marks}
  var typed = "";        // the words in the open form, kept across a redraw
  var renaming = false;
  var wanted = null;     // the margin as the bar's button left it: true, false, or null for "when there are comments"
  var picked = null;     // the selection, as a draft, while it can be commented on
  var held = null;       // the same, kept from the press on the button to its click
  var name = stored("opyt-name") || "";
  var margin, sheet, scrim, pill, fab, toggle, started = false;

  // ── this browser's memory ──────────────────────────────────────────────────────
  function stored(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function keep(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function keys() { try { return JSON.parse(stored("opyt-comment-keys") || "{}") || {}; } catch (e) { return {}; } }
  function keepKey(id, key) {
    var k = keys();
    if (key) k[id] = key; else delete k[id];
    keep("opyt-comment-keys", JSON.stringify(k));
  }

  // ── small things ───────────────────────────────────────────────────────────────
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function button(cls, label, onclick) {
    var b = el("button", cls, label);
    b.type = "button";
    b.addEventListener("click", onclick);
    return b;
  }
  function iconButton(svg, label, onclick) {
    var b = button("cm-ico", null, onclick);
    b.innerHTML = svg;
    b.setAttribute("aria-label", label);
    return b;
  }
  function each(list, fn) { Array.prototype.forEach.call(list, fn); }
  function byId(id) { return threads.filter(function (t) { return t.id === id; })[0]; }
  function total() { return threads.reduce(function (n, t) { return n + t.all.length; }, 0); }
  function plural(n) { return n + (n === 1 ? " comment" : " comments"); }
  function initials(who) {
    var w = who.match(/[^\s.,;:!?()@_-]+/g) || [];
    return ((w[0] || "·").charAt(0) + (w[1] ? w[1].charAt(0) : "")).toUpperCase();
  }
  function when(ts) {
    var d = new Date(ts * 1000);
    return d.toDateString() === new Date().toDateString()
      ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      : d.toLocaleDateString([], { month: "short", day: "numeric" });
  }
  function send(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (got) {
          if (!r.ok) throw new Error(typeof got.detail === "string" ? got.detail : "That did not go through. Try again.");
          return got;
        });
      });
  }

  // ── a source's text, and a passage in it ───────────────────────────────────────
  function bodyOf(n) { var a = document.getElementById("s" + n); return a && a.querySelector(".body"); }
  function textNodes(b) {
    var walk = document.createTreeWalker(b, NodeFilter.SHOW_TEXT), out = [], t;
    while ((t = walk.nextNode())) out.push(t);
    return out;
  }
  // Wrap characters [start, end) of the source's text in highlights, one per text node touched.
  function mark(b, start, end, id, cls) {
    var pos = 0, made = [];
    textNodes(b).forEach(function (t) {
      var len = t.data.length, a = Math.max(start - pos, 0), z = Math.min(end - pos, len);
      pos += len;
      if (a >= z) return;
      // the line breaks between two blocks are in the text, and are nothing to highlight
      if (!t.data.slice(a, z).trim() && /^(DIV|UL|OL|BLOCKQUOTE|TABLE|THEAD|TBODY|TR)$/.test(t.parentNode.nodeName)) return;
      var mid = a ? t.splitText(a) : t;
      if (z - a < mid.data.length) mid.splitText(z - a);
      var m = el("mark", "cm-hl" + (cls ? " " + cls : ""));
      m.dataset.thread = id;
      m.tabIndex = 0;
      mid.parentNode.insertBefore(m, mid);
      m.appendChild(mid);
      made.push(m);
    });
    // a passage across a link or a bold word is several highlights: only its two ends are rounded
    if (made.length) {
      made[0].classList.add("cm-a");
      made[made.length - 1].classList.add("cm-z");
    }
    return made;
  }
  function unmark() {
    each(main.querySelectorAll("mark.cm-hl"), function (m) {
      var p = m.parentNode;
      while (m.firstChild) p.insertBefore(m.firstChild, m);
      p.removeChild(m);
      p.normalize();
    });
  }
  // Where a thread's passage is in its source's text now: [start, end], or null when it is a
  // comment on the whole source or the passage is no longer there.
  function find(th) {
    var b = bodyOf(th.item);
    if (!b || !th.quote) return null;
    var text = b.textContent, before = th.prefix || "";
    var i = text.indexOf(before + th.quote + (th.suffix || ""));
    i = i < 0 ? text.indexOf(th.quote) : i + before.length;
    return i < 0 ? null : [i, i + th.quote.length];
  }
  // The selection as something to comment on, or null: it has to be text inside one source, or
  // inside the answer (item 0).
  function selected() {
    var sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    var range = sel.getRangeAt(0), top = range.commonAncestorContainer;
    var b = (top.nodeType === 1 ? top : top.parentNode).closest(".item .body, .answer .body");
    if (!b) return null;
    var lead = document.createRange();
    lead.selectNodeContents(b);
    lead.setEnd(range.startContainer, range.startOffset);
    var text = b.textContent, start = lead.toString().length, end = start + range.toString().length;
    while (start < end && /\s/.test(text.charAt(start))) start++;
    while (end > start && /\s/.test(text.charAt(end - 1))) end--;
    if (start === end || end - start > limits.quote) return null;
    return {
      item: +b.closest(".item, .answer").id.slice(1), start: start, end: end, quote: text.slice(start, end),
      prefix: text.slice(Math.max(0, start - CONTEXT), start), suffix: text.slice(end, end + CONTEXT), marks: []
    };
  }

  // ── reading the comments ───────────────────────────────────────────────────────
  function load() {
    return fetch(api, { headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("no comments"); return r.json(); })
      .then(function (got) {
        var by = {};
        limits = got.limits || limits;
        threads = [];
        got.comments.forEach(function (c) {
          if (c.thread === c.id) {
            by[c.id] = { id: c.id, item: c.item, quote: c.quote, prefix: c.prefix, suffix: c.suffix, all: [c], marks: [], at: -1 };
            threads.push(by[c.id]);
          } else if (by[c.thread]) by[c.thread].all.push(c);
        });
        if (active && active !== "new" && active !== "none" && !by[active]) active = null;
        if (!started) start();
        paint();
        draw();
      });
  }

  // Put every highlight where its passage is, and count each source's own comments.
  function paint() {
    unmark();
    threads.forEach(function (th) {
      var at = find(th);
      th.at = at ? at[0] : -1;
      th.marks = at ? mark(bodyOf(th.item), at[0], at[1], th.id) : [];
    });
    threads.sort(function (a, b) { return a.item - b.item || a.at - b.at; });
    if (draft) draft.marks = draft.quote ? mark(bodyOf(draft.item), draft.start, draft.end, "new", "cm-pending") : [];
    each(main.querySelectorAll(".cm-count"), function (c) {
      var n = loose(+c.dataset.item).reduce(function (k, t) { return k + t.all.length; }, 0);
      c.hidden = !n;
      c.lastChild.textContent = plural(n) + " on the whole source";
    });
  }
  // A source's threads that no highlight leads to: those on the whole of it.
  function loose(item) { return threads.filter(function (t) { return t.item === item && !t.marks.length; }); }

  // ── drawing ────────────────────────────────────────────────────────────────────
  function draw() {
    var wide = WIDE.matches, n = total(), cbtn = document.getElementById("contents-btn");
    var shown = wide && (active === "new" || (wanted === null ? threads.length > 0 : wanted));
    root.classList.toggle("cm-on", shown);
    toggle.lastChild.textContent = n ? plural(n) : "Comments";
    toggle.setAttribute("aria-pressed", shown ? "true" : "false");
    fab.lastChild.textContent = n || "";
    fab.style.right = (cbtn && cbtn.offsetWidth ? cbtn.offsetWidth + 26 : 16) + "px";
    margin.textContent = "";
    if (sheet) { sheet.remove(); scrim.remove(); sheet = scrim = null; }
    if (shown) {
      threads.forEach(function (th) { margin.appendChild(threadCard(th)); });
      if (active === "new") margin.appendChild(newCard());
      else if (!threads.length) margin.appendChild(hintCard());
    } else if (!wide && active) openSheet();
    each(main.querySelectorAll("mark.cm-hl"), function (m) { m.classList.toggle("on", m.dataset.thread === active); });
    place();
  }

  function entry(th, c) {
    var e = el("div", "cm-entry"), who = el("div", "cm-who"), at = el("span", "cm-when", when(c.made));
    var av = el("span", "cm-av", initials(c.name));
    av.setAttribute("aria-hidden", "true");
    at.title = new Date(c.made * 1000).toLocaleString();
    who.appendChild(av);
    who.appendChild(el("span", "cm-name", c.name));
    who.appendChild(at);
    e.appendChild(who);
    e.appendChild(el("p", "cm-text", c.body));
    if (keys()[c.id]) {
      (WIDE.matches ? e : who).appendChild(button("cm-link cm-del", "Delete", function (ev) {
        ev.stopPropagation();
        remove(th, c);
      }));
    }
    return e;
  }

  function quoted(text) {
    var p = el("p", "cm-quote");
    p.appendChild(el("span", "", text));
    return p;
  }

  // The form a comment or a reply is written in. `th` is the thread replied to, or null.
  function form(th) {
    var f = el("div", "cm-form"), who = null, err = el("p", "cm-error"), acts = el("div", "cm-acts");
    if (!name || renaming) {
      who = el("input", "cm-field");
      who.type = "text";
      who.placeholder = "Your name";
      who.maxLength = limits.name;
      who.value = name;
      who.autocomplete = "name";
      who.setAttribute("aria-label", "Your name");
      f.appendChild(who);
      if (!name) f.appendChild(el("p", "cm-note", "Asked once. This browser remembers it on every reader view."));
    } else if (!th) {
      var as = el("div", "cm-as"), av = el("span", "cm-av", initials(name));
      av.setAttribute("aria-hidden", "true");
      as.appendChild(av);
      as.appendChild(el("span", "cm-name", name));
      as.appendChild(button("cm-link", "Change name", function () { renaming = true; draw(); focus(); }));
      f.appendChild(as);
    }
    var words = el("textarea", "cm-field");
    words.rows = th ? 1 : 3;
    words.placeholder = th ? "Reply" : "Add a comment";
    words.maxLength = limits.body;
    words.value = typed;
    words.setAttribute("aria-label", th ? "Reply" : "Comment");
    words.addEventListener("input", function () {
      typed = words.value;
      words.style.height = "auto";
      words.style.height = Math.min(words.scrollHeight + 2, 220) + "px";
      place();
    });
    words.addEventListener("keydown", function (e) { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) go(); });
    f.appendChild(words);
    if (!th) f.appendChild(el("p", "cm-note", "Anyone with this link will see your name and comment."));
    err.setAttribute("role", "alert");
    f.appendChild(err);
    if (!th && WIDE.matches) acts.appendChild(button("cm-quiet", "Cancel", shut));
    var post = button("cm-go", th ? "Reply" : "Comment", go);
    acts.appendChild(post);
    f.appendChild(acts);

    function go() {
      var as = who ? who.value.replace(/\s+/g, " ").trim() : name, body = words.value.trim();
      if (!as) { who.focus(); return; }
      if (!body) { words.focus(); return; }
      post.disabled = true;
      send(api, th ? { name: as, body: body, thread: th.id }
                   : { name: as, body: body, item: draft.item, quote: draft.quote || null,
                       prefix: draft.prefix || null, suffix: draft.suffix || null })
        .then(function (got) {
          name = as;
          keep("opyt-name", as);
          keepKey(got.comment.id, got.key);
          typed = "";
          draft = null;
          renaming = false;
          active = got.comment.thread;
          return load();
        })
        .catch(function (e) { post.disabled = false; err.textContent = e.message; place(); });
    }
    return f;
  }
  function focus() {
    var at = (sheet || margin).querySelector(".cm-form input, .cm-form textarea");
    if (at && !at.value) { at.focus({ preventScroll: WIDE.matches }); return; }
    at = (sheet || margin).querySelector(".cm-form textarea");
    if (at) at.focus({ preventScroll: WIDE.matches });
  }

  // ── wide: cards in the margin ──────────────────────────────────────────────────
  function threadCard(th) {
    var c = el("aside", "cm-card" + (active === th.id ? " on" : ""));
    c.dataset.thread = th.id;
    if (!th.marks.length) {
      c.appendChild(el("p", "cm-label", "On the whole source"));
      if (th.quote) c.appendChild(quoted(th.quote));
    }
    th.all.forEach(function (one) { c.appendChild(entry(th, one)); });
    if (active === th.id) c.appendChild(form(th));
    c.addEventListener("click", function (e) { if (active !== th.id && !e.target.closest("button")) show(th.id); });
    return c;
  }
  function newCard() {
    var c = el("aside", "cm-card cm-new on");
    c.dataset.thread = "new";
    if (!draft.quote) c.appendChild(el("p", "cm-label", "On the whole source"));
    c.appendChild(form(null));
    return c;
  }
  function hintCard() {
    var c = el("aside", "cm-card");
    c.dataset.thread = "none";
    c.appendChild(el("p", "cm-hint", "No comments yet. Select any text to comment on it, or use “Comment on this source” under a source."));
    return c;
  }
  // How far down the page a card wants to be: level with its passage, or with its source's head.
  function level(id) {
    var th = id === "new" ? draft : byId(id), m = th && th.marks[0];
    if (m) return m.getBoundingClientRect().top - 4;
    var a = th ? document.getElementById("s" + th.item) : main.querySelector(".item");
    return a ? a.getBoundingClientRect().top + (th ? 36 : 0) : main.getBoundingClientRect().top;
  }
  // Each card level with what it is about, the open one exactly so; the rest give way to it.
  function place() {
    if (!root.classList.contains("cm-on")) { main.style.minHeight = ""; return; }
    var base = main.getBoundingClientRect().top, gap = 10, on = -1, y, i, bottom = 0;
    var rows = Array.prototype.map.call(margin.children, function (c) {
      return { card: c, want: level(c.dataset.thread) - base, h: c.offsetHeight };
    }).sort(function (a, b) { return a.want - b.want; });
    rows.forEach(function (r, k) { if (r.card.classList.contains("on")) on = k; });
    if (on < 0) {
      y = -Infinity;
      rows.forEach(function (r) { r.top = Math.max(r.want, y); y = r.top + r.h + gap; });
    } else {
      rows[on].top = rows[on].want;
      y = rows[on].top + rows[on].h + gap;
      for (i = on + 1; i < rows.length; i++) { rows[i].top = Math.max(rows[i].want, y); y = rows[i].top + rows[i].h + gap; }
      y = rows[on].top - gap;
      for (i = on - 1; i >= 0; i--) { rows[i].top = Math.min(rows[i].want, y - rows[i].h); y = rows[i].top - gap; }
    }
    rows.forEach(function (r) {
      r.card.style.top = Math.round(r.top) + "px";
      bottom = Math.max(bottom, r.top + r.h);
    });
    main.style.minHeight = Math.ceil(bottom + 24) + "px";
  }

  // ── narrow: one thread at a time, from the bottom ──────────────────────────────
  function openSheet() {
    var th = byId(active), i = threads.indexOf(th), bar = el("div", "cm-bar"), grab = el("div", "cm-grab");
    var about = active === "new" ? draft : th;
    scrim = el("div", "cm-scrim");
    scrim.addEventListener("click", function () { if (active !== "new" || !typed.trim()) shut(); });
    sheet = el("section", "cm-sheet");
    sheet.setAttribute("role", "dialog");
    sheet.setAttribute("aria-label", active === "new" ? "New comment" : "Comments");
    grab.setAttribute("aria-hidden", "true");
    sheet.appendChild(grab);
    bar.appendChild(el("span", "cm-label", th ? "Comment " + (i + 1) + " of " + threads.length
                                             : active === "new" ? "New comment" : "Comments"));
    if (th) {
      var prev = iconButton(PREV, "Previous comment", function () { show(threads[i - 1].id); });
      var next = iconButton(NEXT, "Next comment", function () { show(threads[i + 1].id); });
      prev.disabled = i === 0;
      next.disabled = i === threads.length - 1;
      bar.appendChild(prev);
      bar.appendChild(next);
    }
    bar.appendChild(iconButton(CLOSE, "Close", shut));
    sheet.appendChild(bar);
    if (about) sheet.appendChild(about.quote ? quoted(about.quote) : el("p", "cm-label", "On the whole source"));
    if (th) th.all.forEach(function (one) { sheet.appendChild(entry(th, one)); });
    if (about) sheet.appendChild(form(th || null));
    else sheet.appendChild(el("p", "cm-hint", "No comments yet. Select any text to comment on it, or use “Comment on this source” under a source."));
    document.body.appendChild(scrim);
    document.body.appendChild(sheet);
    lift();
    var at = about && (about.marks[0] || document.getElementById("s" + about.item)), mine = sheet;
    if (!at) return;
    reveal(at, "smooth");
    // a picture that loads as the sheet opens moves the text and stops that scroll: look once more
    setTimeout(function () { if (sheet === mine) reveal(at, "auto"); }, 600);
  }
  // Bring what the comments are about into the part of the screen the sheet leaves.
  function reveal(at, how) {
    var top = at.getBoundingClientRect().top;
    if (top < 56 || top > innerHeight * 0.24) scrollBy({ top: top - innerHeight * 0.14, behavior: how });
  }
  // A phone's keyboard covers the bottom of the screen: keep the sheet above it.
  function lift() {
    var vv = window.visualViewport;
    if (sheet && vv) sheet.style.bottom = Math.max(0, innerHeight - vv.height - vv.offsetTop) + "px";
  }

  // ── what a reader does ─────────────────────────────────────────────────────────
  function show(id) {
    active = id;
    typed = "";
    renaming = false;
    if (wanted === false) wanted = null;
    draw();
  }
  function shut() {
    var had = draft;
    active = draft = null;
    typed = "";
    renaming = false;
    if (had) paint();
    draw();
  }
  function compose(on) {
    var sel = window.getSelection();
    draft = on;
    active = "new";
    typed = "";
    renaming = false;
    picked = held = null;
    pill.hidden = true;
    if (sel) sel.removeAllRanges();
    paint();
    draw();
    focus();
  }
  function remove(th, c) {
    var whole = c.id === th.id;
    if (whole && th.all.length > 1 && !window.confirm("Delete this comment and its replies?")) return;
    send(api + "/" + encodeURIComponent(c.id) + "/delete", { key: keys()[c.id] }).then(function () {
      keepKey(c.id, null);
      if (whole) active = null;
      return load();
    }).catch(function () {});
  }
  // Offer to comment on the selection: beside it for a mouse, at the bottom of the screen for a
  // finger, where the phone's own Copy menu is not.
  function offer() {
    if (held) return;
    picked = active === "new" ? null : selected();
    pill.hidden = !picked;
    if (!picked) return;
    var dock = !WIDE.matches || COARSE.matches;
    pill.classList.toggle("docked", dock);
    pill.lastChild.textContent = dock ? "Comment on selection" : "Comment";
    pill.style.left = pill.style.top = "";
    if (dock) return;
    var rects = window.getSelection().getRangeAt(0).getClientRects(), end = rects[rects.length - 1];
    if (!end) return;
    var most = scrollX + root.clientWidth - pill.offsetWidth - 12;
    pill.style.left = Math.max(12, Math.min(end.right + scrollX - 28, most)) + "px";
    pill.style.top = (end.bottom + scrollY + 8) + "px";
  }

  // ── once, when the comments have been read for the first time ──────────────────
  function start() {
    var end = document.querySelector(".bar .end"), share = document.getElementById("share-btn"), timer;
    started = true;
    margin = el("div", "cm-margin");
    main.appendChild(margin);

    toggle = button("theme-btn cm-toggle", null, function () {
      wanted = !root.classList.contains("cm-on");
      if (!wanted && active) { shut(); return; }
      draw();
    });
    toggle.innerHTML = BUBBLE;
    toggle.appendChild(el("span", "lbl"));
    if (end) end.insertBefore(toggle, share);

    fab = button("cm-fab", null, function () { show(threads.length ? threads[0].id : "none"); });
    fab.innerHTML = BUBBLE;
    fab.setAttribute("aria-label", "Comments");
    fab.appendChild(el("span"));
    document.body.appendChild(fab);

    pill = button("cm-pill", null, function () {
      var on = held || picked;
      held = null;
      if (on) compose(on);
    });
    pill.hidden = true;
    pill.innerHTML = BUBBLE;
    pill.appendChild(el("span"));
    // pressing it must not take the selection it is about
    pill.addEventListener("pointerdown", function (e) { e.preventDefault(); held = picked; });
    document.body.appendChild(pill);

    each(main.querySelectorAll(".item"), function (a) {
      var n = +a.id.slice(1), foot = a.querySelector(".foot"), body = a.querySelector(".body");
      var whole = button("cm-src", null, function () { compose({ item: n, marks: [] }); });
      var count = button("cm-count", null, function () { var th = loose(n)[0]; if (th) show(th.id); });
      whole.innerHTML = BUBBLE;
      whole.appendChild(el("span", "", "Comment on this source"));
      if (foot) foot.insertBefore(whole, foot.firstChild);
      count.innerHTML = BUBBLE;
      count.hidden = true;
      count.dataset.item = n;
      count.appendChild(el("span"));
      if (body) a.insertBefore(count, body);
    });

    main.addEventListener("click", function (e) {
      var m = e.target.closest("mark.cm-hl");
      if (m && m.dataset.thread !== "new" && window.getSelection().isCollapsed) show(m.dataset.thread);
    });
    main.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && e.target.matches("mark.cm-hl") && e.target.dataset.thread !== "new") show(e.target.dataset.thread);
    });
    // a click anywhere else closes the open thread; a comment being written stays until cancelled
    document.addEventListener("click", function (e) {
      if (!WIDE.matches || !active || active === "new") return;
      if (!e.target.closest(".cm-card, mark.cm-hl, .cm-pill, .cm-toggle")) shut();
    });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && active) shut(); });
    document.addEventListener("selectionchange", function () {
      clearTimeout(timer);
      timer = setTimeout(offer, 180);
    });
    WIDE.addEventListener("change", draw);
    addEventListener("resize", place);
    if (window.ResizeObserver) new ResizeObserver(place).observe(main);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(place);
    if (window.visualViewport) {
      visualViewport.addEventListener("resize", lift);
      visualViewport.addEventListener("scroll", lift);
    }
    // what others wrote while this tab was away, unless something is being written here
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && !typed && active !== "new") load().catch(function () {});
    });
  }

  load().catch(function () {});
})();
