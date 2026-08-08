/* The explanation page.
 *
 * Two disclosure mechanisms, both native <details> so they work without JS and
 * are keyboard/screen-reader correct with zero ARIA:
 *   - summary -> full narrative
 *   - the INFO glossary, closed by default and styled as reference material
 *
 * When the API says summary_is_full (which is the case today, since no
 * summariser has been chosen yet), the narrative disclosure renders open and
 * labelled "Full text" — the same DOM, so landing a real summariser changes
 * only the payload.
 */

import { $, $$, el, getJSON, pct, proseNode, familyClass, timeAgo, postJSON } from "./dom.js";
import { openLightbox, attachLightbox } from "./gallery.js";

const root = $("#result-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;

function figureNode(fig) {
  if (fig.variants && fig.variants.length) return variantFigure(fig);
  const img = el("img", { src: fig.src, alt: fig.title, loading: "lazy" });
  const caption = el("figcaption", {}, el("strong", { text: fig.title }),
    fig.caption ? ` ${fig.caption}` : "",
    fig.n_older ? el("span", { class: "muted", text: ` · ${fig.n_older} older version(s) hidden` }) : null);
  const figure = el("figure", {}, img, caption);
  img.addEventListener("click", () => openLightbox([fig], 0));
  return figure;
}

/* A toggle over alternative renderings of one plot (top-3 vs all detectors,
 * F1 vs PR-AUC, plain vs annotated) rather than N near-duplicate thumbnails. */
function variantFigure(spec) {
  let index = spec.default || 0;
  const img = el("img", { src: spec.variants[index].src, alt: spec.title, loading: "lazy" });
  const tabs = el("div", { class: "variant-tabs" });
  const caption = el("figcaption", {}, el("strong", { text: spec.title }),
    spec.caption ? ` ${spec.caption}` : "");

  spec.variants.forEach((variant, i) => {
    const button = el("button", {
      type: "button", "aria-pressed": String(i === index), text: variant.title,
      onclick: () => {
        index = i;
        img.src = variant.src;
        $$("button", tabs).forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
      },
    });
    tabs.append(button);
  });
  img.addEventListener("click", () => openLightbox(spec.variants, index));
  return el("figure", {}, tabs, img, caption);
}

function galleryButton(stage, payload) {
  const group = (payload.plots || {})[stage.plot_group] || {};
  const extra = [];
  const gallery = group.gallery || [];
  if (gallery.length) {
    extra.push(el("button", {
      type: "button", class: "no-print",
      text: `Browse ${gallery.length} more plot${gallery.length === 1 ? "" : "s"}`,
      onclick: () => openLightbox(gallery, 0),
    }));
  }
  for (const descriptor of (payload.plots || {})._galleries || []) {
    if (!descriptor.id.startsWith(stage.plot_group)) continue;
    extra.push(el("button", {
      type: "button", class: "no-print",
      text: `${descriptor.title} (${descriptor.count})`,
      onclick: () => openGallery(descriptor),
    }));
  }
  return extra.length ? el("div", { class: "row" }, extra) : null;
}

async function openGallery(descriptor) {
  const page = await getJSON(
    `/api/plots/${dataset}/${entity}/gallery/${descriptor.id}?offset=0&limit=60`);
  openLightbox(page.items, 0, {
    total: page.total,
    loadMore: async (offset) => {
      const next = await getJSON(
        `/api/plots/${dataset}/${entity}/gallery/${descriptor.id}?offset=${offset}&limit=60`);
      return next.items;
    },
  });
}

function infoDisclosure(info) {
  if (!info) return null;
  return el("details", { class: "info" },
    el("summary", { text: "ⓘ What these terms mean" }),
    el("div", { class: "info-body" }, proseNode(info, "")));
}

/* What the extended view adds, per stage — the reader should know what a click
 * buys before spending it. */
const EXTENDED_LABEL = {
  ga_selection: "Read the full explanation, including the detectors left out",
  monte_carlo: "Read the full explanation, including the winning noise ranges",
  off_by_threshold: "Read the full explanation, including property importances",
};

function narrativeDisclosure(stage) {
  // When the summary IS the whole narrative the same markup renders open and
  // relabelled, so a stage moving in or out of summarisation changes only the
  // payload.
  const isFull = stage.summary_is_full;
  const details = el("details", { class: "narrative" });
  if (isFull) details.setAttribute("open", "");
  details.append(
    el("summary", { text: isFull ? "Full text"
      : (EXTENDED_LABEL[stage.key] || "Read the full explanation") }),
    proseNode(stage.full));
  return details;
}

/* A deterministic ranking rendered from the IR, for stages whose answer is an
 * order rather than a story. Detector and source names get the mono face so a
 * column of them lines up. */
function summaryTable(spec) {
  const head = el("tr", {}, spec.columns.map((c, i) =>
    el("th", { class: spec.align[i] === "num" ? "num" : "" }, c)));
  const body = spec.rows.map((row) => el("tr", {}, row.map((cell, i) => {
    const kind = spec.align[i];
    return el("td", {
      class: kind === "num" ? "num" : kind === "name" ? "detector" : "",
      text: cell === null || cell === undefined ? "—" : String(cell),
    });
  })));
  return el("div", { class: "table-scroll" },
    el("table", { class: "summary-table" },
      el("thead", {}, head), el("tbody", {}, body)));
}

function regimeSection(stage) {
  if (!stage.regimes || !stage.regimes.length) return null;
  const rows = stage.regimes.map((regime) => el("div", { class: "regime" },
    // The narrated sentence when the model wrote one; the IR's own text is the
    // fallback so a regime is never blank.
    el("div", {}, el("p", { class: "prose", text: regime.narrated || regime.text })),
    regime.plot
      ? figureNode({ src: regime.plot, title: `Regime ${regime.index}`,
                     caption: `Windows ${regime.start}–${regime.end}, led by ${regime.leader}.` })
      : el("p", { class: "muted small", text: "No plot for this regime." })));
  // Printable: this disclosure is the only place the per-regime prose appears
  // now that the stage has no full-text disclosure, and the print stylesheet
  // forces every <details> open.
  return el("details", {},
    el("summary", { text: `Each regime with its channel attribution (${stage.regimes.length})` }),
    el("div", { class: "stack" }, rows));
}

function stageCard(stage, payload) {
  const faith = stage.faithfulness || {};
  const header = el("header", {},
    el("div", { class: "stage-title" },
      el("h3", { id: `stage-${stage.key}`, text: stage.title }),
      stage.top_pick ? el("span", { class: familyClass(stage.top_pick) },
                            `first: ${stage.top_pick}`) : null,
      stage.words ? el("span", { class: "muted small", text: `${stage.words} words` }) : null),
    stage.question ? el("p", { class: "muted small", text: stage.question }) : null);

  const body = el("div", { class: "stack" });
  body.append(infoDisclosure(stage.info));
  body.append(proseNode(stage.summary));
  if (stage.summary_table) body.append(summaryTable(stage.summary_table));
  // `extended_in` means another section of this card already shows what the
  // summary held back — Thompson's regime walk, beside its per-regime plots —
  // so a full-text disclosure here would just repeat it without them.
  if (!stage.extended_in) {
    if (!stage.summary_is_full) body.append(narrativeDisclosure(stage));
    else if (stage.full && stage.full !== stage.summary) body.append(narrativeDisclosure(stage));
  }

  const group = (payload.plots || {})[stage.plot_group] || {};
  if (group.headline && group.headline.length) {
    body.append(el("div", { class: "figures" }, group.headline.map(figureNode)));
  }
  const regimes = regimeSection(stage);
  if (regimes) body.append(regimes);

  const gallery = galleryButton(stage, payload);
  if (gallery) body.append(gallery);

  if (stage.caveats && stage.caveats.length) {
    body.append(el("details", {},
      el("summary", { text: `Caveats (${stage.caveats.length})` }),
      el("ul", { class: "small muted" }, stage.caveats.map((c) => el("li", { text: c })))));
  }

  const footer = el("div", { class: "row small muted no-print" },
    faith.n_claims !== undefined && faith.n_claims !== null
      ? el("span", { text: `${pct(faith.hallucination_rate)} unsupported over ${faith.n_claims} claims · ${pct(faith.omission_rate)} omitted` })
      : null,
    faith.repaired ? el("span", { class: "badge badge-warn", text: "repaired" }) : null,
    el("a", { href: `/api/explanations/${dataset}/${entity}/download?stage=${stage.key}`,
              text: "Download .txt" }));

  return el("section", { class: "card stage-card stack" }, header, body, footer);
}

function decisionHero(payload) {
  const decision = payload.decision || {};
  const chosen = Array.isArray(decision.chosen) ? decision.chosen
    : decision.chosen ? [decision.chosen] : [];
  const faith = payload.faithfulness || {};

  const byline = [];
  if (payload.model) byline.push(`generated by ${payload.model}`);
  if (faith.n_claims) {
    byline.push(`${pct(faith.hallucination_rate)} unsupported over ${faith.n_claims} claims`);
    byline.push(`${pct(faith.omission_rate)} of required facts omitted`);
  }

  return el("section", { class: "decision-hero stack" },
    el("div", { class: "row-between" },
      el("h1", { text: `${payload.dataset} · entity ${payload.entity}` }),
      payload.generated_at
        ? el("span", { class: "muted small", text: timeAgo(payload.generated_at) }) : null),
    // Verbatim from the grounded decision atom — the page invents nothing.
    payload.decision_text
      ? el("p", { class: "verdict", text: payload.decision_text })
      : el("p", { class: "muted", text: "No decision recorded for this run." }),
    chosen.length
      ? el("div", { class: "row" }, chosen.map((d) => el("span", { class: familyClass(d), text: d })))
      : null,
    decision.reason ? el("p", { class: "prose small muted", text: decision.reason }) : null,
    byline.length ? el("p", { class: "byline", text: byline.join(" · ") }) : null);
}

function consensusStrip(payload) {
  if (!payload.agreement || !payload.agreement.length) return null;
  const chips = payload.agreement.map((a) => {
    const glyph = a.agrees === true ? "✓" : a.agrees === false ? "≠" : "–";
    const cls = a.agrees === true ? "badge badge-ok"
      : a.agrees === false ? "badge badge-warn" : "badge badge-muted";
    return el("span", { class: cls, title: `${a.source}: ${a.top_pick}` },
      `${glyph} ${a.source.replace(/_/g, " ")}: ${a.top_pick || "—"}`);
  });
  return el("section", { class: "card card-tight stack" },
    el("h2", { class: "small muted", text: "Where the stages agreed" }),
    el("div", { class: "row" }, chips));
}

function missingSection(payload) {
  if (!payload.missing_stages || !payload.missing_stages.length) return null;
  return el("section", { class: "card card-tight stack" },
    el("h2", { class: "small muted", text: "Stages without an explanation" }),
    el("ul", { class: "small" }, payload.missing_stages.map((m) =>
      el("li", {}, el("strong", { text: m.title }), " — ",
         m.note || m.status.replace(/_/g, " ")))));
}

/* The pipeline's own report, linked rather than inlined: it belongs beside the
 * explanation, not inside it, and its numbers are the binding ones. */
function comprehensiveCard(payload) {
  const report = payload.comprehensive;
  if (!report) return null;
  const meta = [
    report.iteration !== null && report.iteration !== undefined
      ? `iteration ${report.iteration}` : null,
    report.generated_at ? timeAgo(report.generated_at) : null,
    report.name,
  ].filter(Boolean).join(" · ");

  return el("section", { class: "card stack" },
    el("div", { class: "row-between" },
      el("h2", { id: "comprehensive", text: "Comprehensive results" }),
      el("div", { class: "row no-print" },
        el("a", { class: "button primary", href: report.url, text: "Open report →" }),
        el("a", { class: "button", href: report.download_url, text: "Download .txt" }))),
    el("p", { class: "prose small muted" },
      "Measured module timings, memory, the ranking each stage produced and the final " +
      "decision, exactly as the pipeline wrote them. Separate from the explanation above, " +
      "and the binding record for any number that appears in both."),
    el("p", { class: "small muted mono", text: meta }));
}

function appendix(payload) {
  const rows = payload.stages.map((s) => {
    const f = s.faithfulness || {};
    return el("tr", {},
      el("td", { text: s.title }),
      el("td", { class: "num", text: s.words || "—" }),
      el("td", { class: "num", text: pct(f.hallucination_rate) }),
      el("td", { class: "num", text: pct(f.omission_rate) }),
      el("td", { class: "num", text: f.n_claims ?? "—" }));
  });
  return el("details", { class: "card" },
    el("summary", { text: "Appendix: faithfulness and provenance" }),
    el("div", { class: "stack" },
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", { text: "Stage" }), el("th", { class: "num", text: "Words" }),
          el("th", { class: "num", text: "Unsupported" }), el("th", { class: "num", text: "Omitted" }),
          el("th", { class: "num", text: "Claims" }))),
        el("tbody", {}, rows)),
      el("p", { class: "small muted", text:
        `Explanations read from myresults/explanations_nl/${payload.dataset}/${payload.entity}/` +
        (payload.iteration !== null && payload.iteration !== undefined
          ? ` (iteration ${payload.iteration})` : "") }),
      el("p", {}, el("a", { href: `/api/explanations/${dataset}/${entity}/download?stage=global`,
                            text: "Download the full report (.txt)" }))));
}

function bulkToggles() {
  const infoButton = $("#toggle-all-info");
  const fullButton = $("#toggle-all-full");
  let infoOpen = false;
  try { infoOpen = localStorage.getItem("ramses-info-open") === "1"; } catch (e) {}

  const applyInfo = () => {
    $$("details.info").forEach((d) => { d.open = infoOpen; });
    infoButton.setAttribute("aria-pressed", String(infoOpen));
    try { localStorage.setItem("ramses-info-open", infoOpen ? "1" : "0"); } catch (e) {}
  };
  applyInfo();   // a reader who wants definitions wants them everywhere

  infoButton.addEventListener("click", () => { infoOpen = !infoOpen; applyInfo(); });

  let allOpen = false;
  fullButton.addEventListener("click", () => {
    allOpen = !allOpen;
    $$("details.narrative").forEach((d) => { d.open = allOpen; });
    fullButton.textContent = allOpen ? "Collapse all" : "Expand all";
  });
}

async function render() {
  let payload;
  try {
    payload = await getJSON(`/api/explanations/${dataset}/${entity}`);
  } catch (error) {
    root.replaceChildren(el("section", { class: "card stack" },
      el("h1", { text: "No explanation yet" }),
      el("p", { class: "muted", text: error.detail && error.detail.hint
        ? error.detail.hint
        : `Nothing has been generated for ${dataset} / ${entity}.` }),
      el("p", {}, el("a", { class: "button", href: "/", text: "Configure a run" }))));
    return;
  }

  const children = [decisionHero(payload), consensusStrip(payload)];
  if (payload.degraded) {
    children.push(el("p", { class: "banner banner-warn", text:
      "This result predates the structured global report, so only the per-stage " +
      "explanations are shown." }));
  }
  payload.stages.forEach((stage) => children.push(stageCard(stage, payload)));
  children.push(missingSection(payload));
  children.push(comprehensiveCard(payload));
  children.push(appendix(payload));

  root.replaceChildren(el("div", { class: "stack-lg" }, children.filter(Boolean)));

  // Reachable from the top of the page too, not only after scrolling past the
  // stages — but only when the report actually exists on disk.
  if (payload.comprehensive) {
    const link = $("#open-report");
    link.href = payload.comprehensive.url;
    link.hidden = false;
  }

  bulkToggles();
  attachLightbox();

  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

render();
