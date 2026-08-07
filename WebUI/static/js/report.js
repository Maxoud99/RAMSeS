/* The comprehensive report page.
 *
 * This is the pipeline's own record — module timings, memory, per-stage
 * rankings, the final decision — and it is deliberately kept apart from the
 * narrated explanation. The text is shown verbatim: it is the binding source
 * for these numbers, so the page splits it into sections for navigation and
 * changes nothing else.
 */

import { $, el, getJSON, timeAgo } from "./dom.js";

const root = $("#report-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;

const RULE = /^={10,}\s*$/;

/* The writer emits `rule / title / rule`, then the body, for every section.
 * Delimiters therefore come in pairs; an odd count means the format changed,
 * and the caller falls back to showing the file as one block rather than
 * guessing. */
function parseSections(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const rules = [];
  lines.forEach((line, i) => { if (RULE.test(line)) rules.push(i); });
  if (rules.length < 2 || rules.length % 2 !== 0) return null;

  const sections = [];
  for (let p = 0; p < rules.length; p += 2) {
    const open = rules[p], close = rules[p + 1];
    if (close <= open) return null;
    const head = lines.slice(open + 1, close).filter((l) => l.trim());
    const bodyEnd = p + 2 < rules.length ? rules[p + 2] : lines.length;
    const body = lines.slice(close + 1, bodyEnd);
    // Trim the blank lines the separators leave behind.
    while (body.length && !body[0].trim()) body.shift();
    while (body.length && !body[body.length - 1].trim()) body.pop();
    sections.push({
      title: head[0] || "Report",
      subtitle: head.slice(1),
      body: body.join("\n"),
    });
  }
  return sections;
}

function slug(title) {
  return "section-" + title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function sectionCard(section) {
  const id = slug(section.title);
  return el("section", { class: "card stack" },
    el("h2", { id, text: section.title }),
    section.subtitle.length
      ? el("p", { class: "small muted mono", text: section.subtitle.join(" · ") })
      : null,
    section.body ? el("pre", { class: "report-body", text: section.body }) : null);
}

function header(report, sections) {
  const meta = [
    report.iteration !== null && report.iteration !== undefined
      ? `iteration ${report.iteration}` : null,
    report.generated_at ? timeAgo(report.generated_at) : null,
    report.name,
  ].filter(Boolean).join(" · ");

  const nav = (sections || [])
    // The first block is the title page and the last is "END OF REPORT";
    // neither is somewhere a reader wants to jump to.
    .filter((s) => s.body)
    .map((s) => el("a", { class: "button small", href: `#${slug(s.title)}`, text: s.title }));

  // The title page carries the run's own stamp (dataset, iteration, the exact
  // time it was written). It has no body, so it is not rendered as a section —
  // keep its lines here rather than dropping them.
  const titlePage = (sections || []).find((s) => !s.body && s.subtitle.length);

  return el("section", { class: "stack" },
    el("div", { class: "row-between" },
      el("h1", { text: `${dataset} · entity ${entity}` }),
      el("a", { class: "button no-print", href: report.download_url, text: "Download .txt" })),
    el("p", { class: "muted", style: "max-width: 68ch;" },
      "The pipeline's comprehensive results — measured timings, memory, the ranking each " +
      "stage produced and the final decision. These numbers are the binding record; the " +
      "explanation describes them but does not restate them."),
    titlePage
      ? el("p", { class: "small muted mono", text: titlePage.subtitle.join(" · ") }) : null,
    el("p", { class: "small muted mono", text: meta }),
    nav.length ? el("div", { class: "row no-print" }, nav) : null);
}

async function render() {
  let report;
  try {
    report = await getJSON(`/api/comprehensive/${dataset}/${entity}`);
  } catch (error) {
    root.replaceChildren(el("section", { class: "card stack" },
      el("h1", { text: "No comprehensive report" }),
      el("p", { class: "muted", text: (error.detail && error.detail.hint)
        || `Nothing has been written for ${dataset} / ${entity}.` }),
      el("p", {}, el("a", { class: "button", href: "/", text: "Configure a run" }))));
    return;
  }

  const sections = parseSections(report.text);
  const children = [header(report, sections)];
  if (sections) sections.filter((s) => s.body).forEach((s) => children.push(sectionCard(s)));
  else children.push(el("section", { class: "card" },
    el("pre", { class: "report-body", text: report.text })));

  root.replaceChildren(el("div", { class: "stack-lg" }, children.filter(Boolean)));

  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

render();
