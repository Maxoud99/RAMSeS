/* A sticky contents rail, shared by the documentation and report pages — and
 * built from the same parts as the explanation's own "On this page" rail
 * (result.html), so the three views of one result carry one component: one
 * width, one title, one way of marking where the reader is, a strip of chips
 * on a narrow screen. Items are `{ id, num, label, children }`; `num` is
 * optional and `children` renders as a second level. */

import { el } from "./dom.js";
import { trackSections, revealInRail } from "./spy.js";

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

/* The browser resolves the fragment before the page has fetched anything, so by
 * the time the section exists the jump has been missed; doing it here is what
 * makes an inbound link land on its own section. Animated, so the movement shows
 * how far down the page the section is. */
export function jumpToHash(root) {
  const id = decodeURIComponent((location.hash || "").replace(/^#/, ""));
  if (!id) return;
  const target = document.getElementById(id);
  if (!target) return;
  root.querySelectorAll(".is-target").forEach((n) => n.classList.remove("is-target"));
  target.scrollIntoView({
    behavior: REDUCED_MOTION.matches ? "auto" : "smooth", block: "start" });
  target.classList.add("is-target");
}

/* Hiding the contents, remembered under `storageKey` — one per page, since the
 * lists are worth different amounts. Only the wide layout honours it (the
 * stylesheet keys off `is-collapsed` from the two-column breakpoint up): on a
 * narrow screen the contents are a one-line strip, and collapsing that left an
 * empty bordered box around a lone » button. Shared with the explanation's
 * rail, so collapsing works, and looks, the same on all three views. */
export function wireCollapse(layout, toggle, storageKey) {
  const apply = (collapsed) => {
    layout.classList.toggle("is-collapsed", collapsed);
    toggle.textContent = collapsed ? "»" : "«";
    toggle.title = collapsed ? "Show the contents" : "Hide the contents";
    toggle.setAttribute("aria-label", toggle.title);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    try { localStorage.setItem(storageKey, collapsed ? "1" : "0"); } catch (e) {}
  };
  let collapsed = false;
  try { collapsed = localStorage.getItem(storageKey) === "1"; } catch (e) {}
  toggle.addEventListener("click", () => { collapsed = !collapsed; apply(collapsed); });
  apply(collapsed);
}

/* Taking the click keeps the fragment in the URL (so Back works and the link is
 * copyable) but leaves the movement to us — a native jump is instant, and the
 * smooth scroll would then run from a position already at the destination. */
function followLink(root, event, id, follow) {
  event.preventDefault();
  if (location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
  follow(id);
  jumpToHash(root);
}

function link(root, item, follow) {
  return el("a", { href: `#${item.id}`, "data-target": item.id,
                   onclick: (event) => followLink(root, event, item.id, follow) },
    item.num ? el("span", { class: "rail-num", text: item.num }) : null,
    el("span", { class: "rail-label", text: item.label }));
}

function contentsList(root, items, follow) {
  return el("ol", { class: "rail-list sidenav-list" },
    items.map((item) => el("li", {},
      link(root, item, follow),
      (item.children || []).length
        ? el("ol", { class: "rail-sublist sidenav-sublist" },
            item.children.map((child) => el("li", {}, link(root, child, follow))))
        : null)));
}

/* Contents on the left (a strip across the top on a narrow screen), `body`
 * beside it. The reader's position is marked on the links from the scroll —
 * the link of the section they are in, and its parent section's — so the
 * contents say "you are here" as well as "go there". */
export function sideNavLayout(root, { title, items, storageKey, body }) {
  const order = [];
  items.forEach((item) => {
    order.push({ id: item.id, parent: null });
    (item.children || []).forEach((child) => order.push({ id: child.id, parent: item.id }));
  });
  const links = new Map();
  let list = null;
  let nav = null;

  const follow = trackSections(order.map((entry) => entry.id), {
    mark: (id) => {
      const entry = order.find((e) => e.id === id);
      const parent = entry && entry.parent;
      links.forEach((a, key) => {
        const on = !!entry && key === id;
        a.classList.toggle("is-current", on);
        a.classList.toggle("is-current-parent", !!parent && key === parent);
        if (on) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
      });
      // A subsection's link is not drawn in the narrow strip; its section's is.
      const shown = entry && links.get(id);
      const target = shown && shown.getClientRects().length ? shown : links.get(parent);
      revealInRail(target, list, nav);
    },
  });

  list = contentsList(root, items, follow);
  list.querySelectorAll("a[data-target]").forEach((a) => links.set(a.dataset.target, a));

  const titleId = `${storageKey}-title`;
  const toggle = el("button", { type: "button", class: "rail-collapse no-print" });
  nav = el("nav", { class: "page-rail sidenav no-print", "aria-labelledby": titleId },
    el("div", { class: "rail-head" },
      el("h2", { class: "rail-title", id: titleId, text: title }), toggle),
    list);
  const layout = el("div", { class: "rail-layout sidenav-layout" }, nav, body);
  wireCollapse(layout, toggle, storageKey);

  window.addEventListener("hashchange", () => {
    follow(decodeURIComponent((location.hash || "").replace(/^#/, "")));
    jumpToHash(root);
  });
  return layout;
}
