
(function () {
  var root = document.documentElement, btn = document.getElementById("theme-btn"),
      name = document.getElementById("theme-name");
  function cur() { return root.getAttribute("data-theme") === "dark" ? "dark" : "light"; }
  function label() { name.textContent = cur() === "dark" ? "light" : "dark"; }
  btn.addEventListener("click", function () {
    var next = cur() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("opyt-theme", next); } catch (e) {}
    label();
  });
  label();
  // The font menu. A face's stylesheet loads the first time the menu opens, so each name shows in
  // its own face, and a pick is remembered for this reader.
  var fbtn = document.getElementById("font-btn"), menu = document.getElementById("font-menu"),
      faces = menu.querySelectorAll("[data-font]"), DEFAULT = faces[0].dataset.font, loaded = false;
  function face() { return root.getAttribute("data-font") || DEFAULT; }
  function addFont(href) {
    if (!href || document.querySelector('link[href="' + href + '"]')) return;
    var l = document.createElement("link"); l.rel = "stylesheet"; l.href = href;
    document.head.appendChild(l);
  }
  function mark() {
    faces.forEach(function (b) { b.setAttribute("aria-checked", b.dataset.font === face() ? "true" : "false"); });
  }
  function show(open) {
    menu.hidden = !open;
    fbtn.setAttribute("aria-expanded", open ? "true" : "false");
    if (open && !loaded) { loaded = true; faces.forEach(function (b) { addFont(b.dataset.href); }); }
  }
  fbtn.addEventListener("click", function (e) { e.stopPropagation(); show(menu.hidden); });
  menu.addEventListener("click", function (e) {
    var b = e.target.closest("[data-font]");
    if (!b) return;
    if (b.dataset.font === DEFAULT) root.removeAttribute("data-font");
    else { addFont(b.dataset.href); root.setAttribute("data-font", b.dataset.font); }
    try { localStorage.setItem("opyt-font", b.dataset.font); } catch (e) {}
    mark(); show(false);
  });
  document.addEventListener("click", function (e) { if (!menu.hidden && !menu.contains(e.target)) show(false); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") show(false); });
  mark();

  // Share: the phone's share sheet on a touch screen, otherwise the link is copied.
  var share = document.getElementById("share-btn"), toast = document.getElementById("toast"), timer;
  function say(text) {
    toast.textContent = text; toast.classList.add("on");
    clearTimeout(timer); timer = setTimeout(function () { toast.classList.remove("on"); }, 3600);
  }
  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (ok, no) {
      var a = document.createElement("textarea");
      a.value = text; a.setAttribute("readonly", ""); a.style.position = "fixed"; a.style.opacity = "0";
      document.body.appendChild(a); a.select();
      try { document.execCommand("copy") ? ok() : no(); } catch (e) { no(); }
      a.remove();
    });
  }
  share.addEventListener("click", function () {
    var url = location.href.split("#")[0];
    if (navigator.share && matchMedia("(pointer: coarse)").matches) {
      navigator.share({ title: document.title, url: url }).catch(function () {});
      return;
    }
    copy(url).then(function () {
      say("Link copied. Anyone with it can read this page. It is deleted " + share.dataset.days + " days after it was last opened.");
    }, function () { say(url); });
  });

  // Contents: marks where the reader is, opens from the rail (desktop) or the button (narrow).
  var nav = document.getElementById("contents"), rail = document.getElementById("rail"),
      cbtn = document.getElementById("contents-btn"),
      links = nav.querySelectorAll(".panel a"), ticks = nav.querySelectorAll(".tick"),
      targets = Array.prototype.map.call(links, function (a) { return document.getElementById(a.dataset.for); });
  function spy() {
    var line = innerHeight * 0.3, cur = 0;
    targets.forEach(function (t, i) {
      if (t && t.getBoundingClientRect().top <= line) cur = i;
    });
    var id = links[cur].dataset.for, src = links[cur].dataset.in || id;
    [links, ticks].forEach(function (set) {
      set.forEach(function (el) {
        el.classList.toggle("on", el.dataset.for === id);
        el.classList.toggle("in", el.dataset.for === src);
      });
    });
  }
  var queued = false;
  addEventListener("scroll", function () {
    if (!queued) { queued = true; requestAnimationFrame(function () { queued = false; spy(); }); }
  }, { passive: true });
  spy();
  function opened(open) {
    nav.classList.toggle("open", open);
    cbtn.setAttribute("aria-expanded", open ? "true" : "false");
  }
  rail.addEventListener("click", function () {
    var pinned = nav.classList.toggle("pinned");
    nav.classList.toggle("shut", !pinned);   // folding closes it now, not when the pointer leaves
    rail.setAttribute("aria-expanded", pinned ? "true" : "false");
  });
  nav.addEventListener("mouseleave", function () { nav.classList.remove("shut"); });
  cbtn.addEventListener("click", function (e) { e.stopPropagation(); opened(!nav.classList.contains("open")); });
  nav.addEventListener("click", function (e) {
    if (e.target.closest(".panel a")) opened(false);
  });
  document.addEventListener("click", function (e) {
    if (nav.classList.contains("open") && !nav.querySelector(".panel").contains(e.target)) opened(false);
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") opened(false); });
})();
