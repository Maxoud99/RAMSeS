/* A sticky contents sidebar, shared by the documentation and report pages.
 * Items are `{ id, label, children }`; `children` renders as a second level. */

import { el } from "./dom.js";

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

/* Taking the click keeps the fragment in the URL (so Back works and the link is
 * copyable) but leaves the movement to us — a native jump is instant, and the
 * smooth scroll would then run from a position already at the destination. */
function followLink(root, event, id) {
  event.preventDefault();
  if (location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
  jumpToHash(root);
}

function link(root, item) {
  return el("a", { href: `#${item.id}`, text: item.label,
                   onclick: (event) => followLink(root, event, item.id) });
}

function contentsList(root, items) {
  return el("ol", { class: "sidenav-list" },
    items.map((item) => el("li", {},
      link(root, item),
      (item.children || []).length
        ? el("ol", { class: "sidenav-sublist" },
            item.children.map((child) => el("li", {}, link(root, child))))
        : null)));
}

/* Contents on the left, `body` on the right. Collapsing is remembered under
 * `storageKey` — one per page, since the two lists are worth different amounts. */
export function sideNavLayout(root, { title, items, storageKey, body }) {
  const list = contentsList(root, items);
  const toggle = el("button", { type: "button", class: "sidenav-toggle no-print" });
  const layout = el("div", { class: "sidenav-layout" });

  const apply = (collapsed) => {
    layout.classList.toggle("is-collapsed", collapsed);
    toggle.textContent = collapsed ? "»" : "«";
    toggle.title = collapsed ? "Show the contents" : "Hide the contents";
    toggle.setAttribute("aria-label", toggle.title);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    list.hidden = collapsed;
    try { localStorage.setItem(storageKey, collapsed ? "1" : "0"); } catch (e) {}
  };

  let collapsed = false;
  try { collapsed = localStorage.getItem(storageKey) === "1"; } catch (e) {}
  toggle.addEventListener("click", () => { collapsed = !collapsed; apply(collapsed); });

  layout.append(
    el("aside", { class: "sidenav card card-tight" },
      el("div", { class: "row-between sidenav-head" },
        el("h2", { class: "sidenav-title", text: title }), toggle),
      list),
    body);
  apply(collapsed);
  window.addEventListener("hashchange", () => jumpToHash(root));
  return layout;
}
