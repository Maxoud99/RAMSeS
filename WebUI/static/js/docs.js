/* The stage documentation page.
 *
 * One section per stage of the pipeline, not per explanation card: the genetic
 * algorithm contributes two cards and LinTS two more, and both of each pair land
 * in the same section. Sections are declared once, server-side, in
 * artifacts.DOC_SECTIONS.
 *
 * A section describes how that stage of RAMSeS works, ahead of any explanation
 * of the explainability layer on top of it. The prose is static — it describes
 * the framework, not a run — and is authored in artifacts.DOC_SECTIONS, so this
 * module only lays it out.
 */

import { $, el, getJSON } from "./dom.js";

const root = $("#docs-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;

const NAV_KEY = "ramses-docs-nav-collapsed";
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

/* Scroll to the fragment the reader arrived with.
 *
 * The browser resolves `#ga` before this page has fetched anything, so by the
 * time the section exists the jump has already been missed. Doing it here is
 * what makes the stage cards' links land on their own section rather than at
 * the top of the page.
 *
 * Animated rather than instant, so the movement shows how far down the page the
 * section is — an instant jump between two similar-looking cards gives no sense
 * of having moved at all. Honoured only when the reader has not asked for
 * reduced motion; that preference is a system setting and a smooth scroll is
 * exactly the kind of movement it is about. */
function jumpToHash() {
  const id = decodeURIComponent((location.hash || "").replace(/^#/, ""));
  if (!id) return;
  const target = document.getElementById(id);
  if (!target) return;
  // Cleared first: following a second link would otherwise leave two sections
  // claiming to be the one the reader was sent to.
  root.querySelectorAll(".is-target").forEach((n) => n.classList.remove("is-target"));
  target.scrollIntoView({
    behavior: REDUCED_MOTION.matches ? "auto" : "smooth", block: "start" });
  target.classList.add("is-target");
}

/* Follow a contents link without letting the browser jump first.
 *
 * A plain <a href="#ga"> is navigated natively — instantly — and only then does
 * hashchange fire, so the smooth scroll would run from a position already at the
 * destination and animate nothing. Taking the click means the fragment is still
 * put in the URL (pushState, so Back works and the link is copyable) but the
 * movement is ours. */
function followLink(event, id) {
  event.preventDefault();
  if (location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
  jumpToHash();
}

/* One block of a section's prose.
 *
 * Four shapes, all authored server-side in artifacts.DOC_SECTIONS: a plain
 * paragraph, a paragraph opening with a bold label, a list, and a block of
 * notation. Built as nodes rather than injected as markup so the text stays
 * text — nothing here needs to carry formatting, and the page never has to
 * trust a string. */
function blockNode(block) {
  if (block.formula) {
    // <pre>, so the alignment the notation was written with survives; scrolls
    // in its own box rather than forcing the page sideways on a narrow screen.
    return el("pre", { class: "docs-formula" }, el("code", { text: block.formula }));
  }
  if (block.list) {
    return el(block.ordered ? "ol" : "ul", { class: "docs-list" },
      block.list.map((item) => el("li", { text: item })));
  }
  if (block.lead) {
    return el("p", {}, el("strong", { text: block.lead }), ` ${block.text}`);
  }
  return el("p", { text: block.text });
}

function proseBlocks(blocks) {
  return el("div", { class: "prose docs-prose" }, (blocks || []).map(blockNode));
}

/* A section, with its subsections as N.M anchors of their own.
 *
 * The number is computed here rather than stored, so inserting a section
 * renumbers everything after it without an edit to the text. A section's own
 * blocks are its opening and sit above the first subsection. */
function sectionNode(section, number) {
  const node = el("section", { class: "card", id: section.id },
    el("h2", {}, el("span", { class: "docs-num", text: `${number}.` }), ` ${section.title}`),
    proseBlocks(section.blocks));
  (section.subsections || []).forEach((sub, i) => {
    node.append(
      el("h3", { class: "docs-subhead", id: sub.id },
        el("span", { class: "docs-num", text: `${number}.${i + 1}` }), ` ${sub.title}`),
      proseBlocks(sub.blocks));
  });
  return node;
}

/* The contents, as a collapsible sidebar.
 *
 * Sticky rather than scrolling away: the sections run long, and the whole point
 * of a contents list is to be reachable from the middle of one. Collapsing is
 * remembered, because a reader who wants the prose at full width wants it on
 * every section, not just the one they collapsed it on. */
function sidebarNode(sections, layout) {
  // Two levels: the sections, and inside each the N.M anchors. A section long
  // enough to need navigating inside is exactly the one whose entry in a
  // flat contents list would send a reader to the top of several screens of
  // prose. `gan` has no subsections and simply gets no nested list.
  const list = el("ol", { class: "docs-contents" },
    sections.map((s, i) => el("li", {},
      el("a", { href: `#${s.id}`, text: `${i + 1}. ${s.title}`,
                onclick: (event) => followLink(event, s.id) }),
      (s.subsections || []).length
        ? el("ol", { class: "docs-subcontents" },
            s.subsections.map((sub, j) => el("li", {},
              el("a", { href: `#${sub.id}`, text: `${i + 1}.${j + 1} ${sub.title}`,
                        onclick: (event) => followLink(event, sub.id) }))))
        : null)));

  const toggle = el("button", { type: "button", class: "docs-nav-toggle no-print" });
  const apply = (collapsed) => {
    layout.classList.toggle("is-collapsed", collapsed);
    toggle.textContent = collapsed ? "»" : "«";
    toggle.title = collapsed ? "Show the contents" : "Hide the contents";
    toggle.setAttribute("aria-label", toggle.title);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    list.hidden = collapsed;
    try { localStorage.setItem(NAV_KEY, collapsed ? "1" : "0"); } catch (e) {}
  };

  let collapsed = false;
  try { collapsed = localStorage.getItem(NAV_KEY) === "1"; } catch (e) {}
  toggle.addEventListener("click", () => { collapsed = !collapsed; apply(collapsed); });

  const nav = el("aside", { class: "docs-nav card card-tight" },
    el("div", { class: "row-between docs-nav-head" },
      el("h2", { class: "docs-nav-title", text: "Stages" }), toggle),
    list);
  apply(collapsed);
  return nav;
}

async function main() {
  let payload;
  try {
    payload = await getJSON(`/api/docs/${dataset}/${entity}`);
  } catch (e) {
    root.replaceChildren(el("p", { class: "notice", text:
      "No documentation for this entity yet — it has no explanations on disk." }));
    return;
  }
  const sections = payload.sections || [];
  if (!sections.length) {
    root.replaceChildren(el("p", { class: "notice", text:
      "No stage documentation for this entity yet." }));
    return;
  }
  const body = el("div", { class: "stack docs-body" },
    ...sections.map((s, i) => sectionNode(s, i + 1)));
  const layout = el("div", { class: "docs-layout" });
  layout.append(sidebarNode(sections, layout), body);
  root.replaceChildren(layout);
  jumpToHash();
}

window.addEventListener("hashchange", jumpToHash);
main();
