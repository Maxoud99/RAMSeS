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
import { jumpToHash, sideNavLayout } from "./sidenav.js";

const root = $("#docs-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;

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

/* The N.M anchors are what the sidebar's second level points at. */
function contentsItems(sections) {
  return sections.map((s, i) => ({
    id: s.id,
    label: `${i + 1}. ${s.title}`,
    children: (s.subsections || []).map((sub, j) => ({
      id: sub.id, label: `${i + 1}.${j + 1} ${sub.title}` })),
  }));
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
  const body = el("div", { class: "stack sidenav-body" },
    ...sections.map((s, i) => sectionNode(s, i + 1)));
  root.replaceChildren(sideNavLayout(root, {
    title: "Stages",
    items: contentsItems(sections),
    storageKey: "ramses-docs-nav-collapsed",
    body,
  }));
  jumpToHash(root);
}

main();
