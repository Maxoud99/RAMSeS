/* The explanation page.
 *
 * The summary -> full narrative disclosure is a native <details>, so it works
 * without JS and is keyboard/screen-reader correct with zero ARIA.
 *
 * When the API says summary_is_full — a stage whose summariser found nothing to
 * hold back — it renders open and labelled "Full text", the same DOM either way.
 */

import { $, $$, el, getJSON, pct, proseNode, familyClass, timeAgo, postJSON,
         makeSortable } from "./dom.js";
import { openLightbox, attachLightbox } from "./gallery.js";

const root = $("#result-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;

/* A caption's title and its sentence are now set in the same colour, so the
 * colon is what separates them. Added only when there is a sentence to
 * separate: a bare title ending in ":" would promise one. */
function captionTitle(title, caption) {
  return caption ? `${title}:` : title;
}

function figureNode(fig) {
  if (fig.pair_picker) return pairPickerFigure(fig);
  if (fig.variants && fig.variants.length) return variantFigure(fig);
  const img = el("img", { src: fig.src, alt: fig.title, loading: "lazy" });
  const caption = el("figcaption", {}, el("strong", { text: captionTitle(fig.title, fig.caption) }),
    fig.caption ? ` ${fig.caption}` : "",
    fig.n_older ? el("span", { text: ` · ${fig.n_older} older version(s) hidden` }) : null);
  const figure = el("figure", {}, img, caption);
  img.addEventListener("click", () => openLightbox([fig], 0));
  return figure;
}

/* Two detector pickers and a figure drawn per request.
 *
 * Not a variant toggle: with 11 detectors there are 55 unordered pairs, so the
 * images cannot be enumerated up front. The server renders one from the IR's
 * per-detector context feature shares. The initial pair is the ranking's first two —
 * the winner and the runner-up — which reproduces the static figure the
 * pipeline writes, so the default view is unchanged by this control existing.
 */
function pairPickerFigure(spec) {
  const names = spec.pair_picker.detectors;
  const url = (a, b) =>
    `${spec.pair_picker.endpoint}?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
  let a = names[0];
  let b = names[1];

  const img = el("img", { src: url(a, b), alt: spec.title, loading: "lazy" });
  const status = el("span", { class: "small muted" });
  const picker = (initial, onPick) => {
    const select = el("select", { onchange: (e) => onPick(e.target.value) });
    names.forEach((n) =>
      select.append(el("option", { value: n, selected: n === initial }, n)));
    return select;
  };
  const refresh = () => {
    // A detector compared with itself has an all-zero gap and the server
    // refuses it, so the pickers never hold the same name: the other side
    // steps aside rather than the request failing.
    if (a === b) {
      const other = names.find((n) => n !== a);
      if (!other) return;
      b = other;
      $$("select", tabs)[1].value = b;
    }
    status.textContent = "";
    img.src = url(a, b);
  };
  img.addEventListener("error", () => {
    status.textContent = "No decomposition available for this pair.";
  });

  const tabs = el("div", { class: "variant-tabs" },
    el("label", { class: "small muted", text: "Ahead" }),
    picker(a, (v) => { a = v; refresh(); }),
    el("label", { class: "small muted", text: "vs" }),
    picker(b, (v) => { b = v; refresh(); }),
    status);
  const caption = el("figcaption", {},
    el("strong", { text: captionTitle(spec.title, spec.caption) }),
    spec.caption ? ` ${spec.caption}` : "");
  img.addEventListener("click", () =>
    openLightbox([{ src: img.src, caption: spec.caption,
                    title: `${a} vs ${b}` }], 0));
  return el("figure", {}, tabs, img, caption);
}

/* A toggle over alternative renderings of one plot (top-3 vs all detectors,
 * F1 vs PR-AUC, plain vs annotated) rather than N near-duplicate thumbnails. */
function variantFigure(spec) {
  let index = spec.default || 0;
  const img = el("img", { src: spec.variants[index].src, alt: spec.title, loading: "lazy" });
  const tabs = el("div", { class: "variant-tabs" });
  // The caption follows the active variant unless the spec overrides it for
  // the whole group. Variants are not always the same quantity in different
  // slices — Thompson's regime tabs switch between a share of the expected
  // reward and a deviation from a typical window — so a fixed caption would
  // describe whichever one happened to be first.
  const captionText = el("span", {});
  const captionHead = el("strong", {});
  const setCaption = () => {
    const text = spec.caption || spec.variants[index].caption || "";
    captionHead.textContent = captionTitle(spec.title, text);
    captionText.textContent = text ? ` ${text}` : "";
  };
  const caption = el("figcaption", {}, captionHead, captionText);
  setCaption();

  // Past four options a tab strip wraps onto a second row and stops reading as
  // a control, so the same choice becomes a <select>. `select_label` opts a
  // group in explicitly for a named axis of choice (off-by's competitor).
  const asSelect = spec.select_label || spec.variants.length > 4;
  if (asSelect) {
    const select = el("select", {
      onchange: (e) => {
        index = Number(e.target.value);
        img.src = spec.variants[index].src;
        setCaption();
      },
    });
    spec.variants.forEach((variant, i) =>
      select.append(el("option", { value: String(i), selected: i === index },
                       variant.title)));
    tabs.append(
      el("label", { class: "small muted", text: spec.select_label || "Show" }),
      select);
  } else {
    spec.variants.forEach((variant, i) => {
      const button = el("button", {
        type: "button", "aria-pressed": String(i === index), text: variant.title,
        onclick: () => {
          index = i;
          img.src = variant.src;
          setCaption();
          $$("button", tabs).forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
        },
      });
      tabs.append(button);
    });
  }
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

/* One line per term, in the space the glossary used to fill.
 *
 * A definition list rather than prose: these are lookups, not something to be
 * read through, and the two-column grid lets a reader find the term they came
 * for without reading the ones they did not. */
function termsNode(stage) {
  const terms = stage.terms || [];
  if (!terms.length) return null;
  // flatMap, not map: el() flattens its children exactly one level, so a list
  // of [dt, dd] pairs would reach it as arrays and be stringified.
  return el("dl", { class: "terms" },
    terms.flatMap(([term, definition]) => [
      el("dt", { text: `${term}:` }),
      el("dd", { text: definition }),
    ]));
}

/* The link to this stage's section of the documentation.
 *
 * The glossaries used to open every card, ahead of its finding, and they run
 * long enough to be the first thing a reader had to get past. They now live on
 * their own page; this is the pointer to the section holding this stage's.
 *
 * A new tab, because a reader looking a term up is in the middle of reading the
 * card and wants to come back to it. `rel=noopener` because target=_blank hands
 * the opened page a reference to this one otherwise. */
function docsLink(stage) {
  if (!stage.doc_section) return null;
  return el("a", { class: "button no-print stage-docs-link",
                   href: `/docs/${dataset}/${entity}#${stage.doc_section}`,
                   target: "_blank", rel: "noopener",
                   text: "Stage description for the interested ↗" });
}

/* What the extended view adds, per stage — the reader should know what a click
 * buys before spending it. */
const EXTENDED_LABEL = {
  ga_selection: "Read the full explanation, including the detectors left out",
  monte_carlo: "Read the full explanation, including the winning noise ranges",
  off_by_threshold: "Read the full explanation, including property importances",
  gan: "Read the full explanation, including property importances",
  thompson_ranking: "Read the full explanation, including how leadership changed hands",
};

/* Stages whose headline figure is a surrogate decision tree.
 *
 * The tree is the evidence behind the prose, not the finding itself, and at
 * full width it is the tallest thing on the card — so it opens on demand, the
 * way the full narrative does. Same <details> idiom, so it prints open. */
const TREE_LABEL = {
  off_by_threshold: "Show the decision trees behind these rules",
  gan: "Show the decision trees behind these rules",
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
  // `collapse_after` hides the tail rather than dropping it: a decomposition
  // whose parts sum to the whole is only honest if every part is reachable, so
  // the rows are all in the DOM (and so all printed) with the overflow folded.
  const fold = spec.collapse_after || 0;
  const body = spec.rows.map((row, r) => el("tr",
    { class: fold && r >= fold ? "extra" : "" }, row.map((cell, i) => {
      const kind = spec.align[i];
      return el("td", {
        class: kind === "num" ? "num" : kind === "name" ? "detector" : "",
        text: cell === null || cell === undefined ? "—" : String(cell),
      });
    })));
  const table = el("table", { class: "summary-table" },
    el("thead", {}, head), el("tbody", {}, body));
  // The fold is declared on the element so sorting can re-apply it: hiding a
  // fixed set of rows would survive a re-order and hide the wrong ones.
  if (fold) table.dataset.collapseAfter = String(fold);
  makeSortable(table);
  const wrap = el("div", { class: "table-scroll" }, table);
  if (!fold || spec.rows.length <= fold) return wrap;

  const hidden = spec.rows.length - fold;
  const button = el("button", { class: "link-button",
    text: `Show all ${spec.rows.length} rows (${hidden} more)` });
  button.addEventListener("click", () => {
    const shown = table.classList.toggle("expanded");
    button.textContent = shown
      ? `Show top ${fold} only`
      : `Show all ${spec.rows.length} rows (${hidden} more)`;
  });
  return el("div", {}, wrap, button);
}

/* One regime's figure. Several sets for the same regime become a variant
 * toggle; the server orders them, so the default is whatever it put first. */
function regimeFigure(regime) {
  const figures = regime.plots || (regime.plot
    ? [{ src: regime.plot, title: `Regime ${regime.index}`,
         caption: regime.plot_caption }]
    : []);
  if (!figures.length) {
    return el("p", { class: "muted small", text: "No plot for this regime." });
  }
  const fallback = `Windows ${regime.start}–${regime.end}, led by ${regime.leader}.`;
  const prepared = figures.map((f) => ({
    ...f,
    title: f.title || `Regime ${regime.index}`,
    caption: f.caption || fallback,
  }));
  if (prepared.length === 1) return figureNode(prepared[0]);
  return figureNode({ title: `Regime ${regime.index}`,
                      variants: prepared, default: 0 });
}

function regimeSection(stage) {
  if (!stage.regimes || !stage.regimes.length) return null;
  const rows = stage.regimes.map((regime) => el("div", { class: "regime" },
    // The narrated sentence when the model wrote one; the IR's own text is the
    // fallback so a regime is never blank.
    el("div", {}, el("p", { class: "prose", text: regime.narrated || regime.text })),
    // Several figures per regime become a variant toggle, first one default —
    // the expected-reward contribution, with SHAP's deviation view a click
    // away. Titles and captions come from the server: these sets show different
    // quantities over the same window range, so a generic "windows X–Y" here
    // would under-describe all of them.
    regimeFigure(regime)));
  // Printable: this disclosure is the only place the per-regime prose appears
  // now that the stage has no full-text disclosure, and the print stylesheet
  // forces every <details> open.
  return el("details", {},
    el("summary", { text: `Each regime with its context feature attribution (${stage.regimes.length})` }),
    el("div", { class: "stack" }, rows));
}

function stageCard(stage, payload) {
  const faith = stage.faithfulness || {};
  const header = el("header", {},
    el("div", { class: "stage-title" },
      el("h3", { id: `stage-${stage.key}`, text: stage.title }),
      stage.top_pick ? el("span", { class: familyClass(stage.top_pick) },
                            `first: ${stage.top_pick}`) : null,
      stage.words ? el("span", { class: "muted small", text: `${stage.words} words` }) : null,
      // Beside the word count and pushed to the right edge: it is a way out of
      // the card, not part of its finding, so it sits with the metadata rather
      // than in the reading order of the prose.
      docsLink(stage)),
    stage.question ? el("p", { class: "muted small", text: stage.question }) : null,
    // Names what the definition list under it is for. Only when there is one:
    // the sentence otherwise promises terms that never arrive. "this stage"
    // rather than the title, which is in the heading directly above it.
    (stage.terms || []).length
      ? el("p", { class: "muted small", text:
          "The following terms and metrics are used to explain this stage:" })
      : null);

  const body = el("div", { class: "stack" });
  // The prose is older than the facts it describes, so it may be narrating a
  // previous run. Said plainly and at the top, because everything below it —
  // summary, table, regime sentences — inherits the doubt.
  if (stage.stale) {
    body.append(el("p", { class: "notice", text:
      "⚠ This explanation is older than the results it describes: the stage was "
      + "re-run without re-generating its narrative. Re-run the narrator to refresh it." }));
  }
  body.append(termsNode(stage));
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
    const figures = el("div", { class: "figures" }, group.headline.map(figureNode));
    const treeLabel = TREE_LABEL[stage.key];
    body.append(treeLabel
      ? el("details", {}, el("summary", { text: treeLabel }), figures)
      : figures);
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

  // Provenance only, and dim: these are measurements of the prose rather than
  // part of it. Nothing else is left in the row, so it is dropped entirely when
  // the run recorded no faithfulness numbers.
  const provenance = [
    faith.n_claims !== undefined && faith.n_claims !== null
      ? el("span", { text: `${pct(faith.hallucination_rate)} unsupported over ${faith.n_claims} claims · ${pct(faith.omission_rate)} omitted` })
      : null,
    faith.repaired ? el("span", { class: "badge badge-warn", text: "repaired" }) : null,
  ].filter(Boolean);
  const footer = provenance.length
    ? el("div", { class: "row small dim no-print" }, provenance) : null;

  return el("section", { class: "card stage-card stack" }, header, body, footer);
}

/* Two of the algorithms answer two questions each and so occupy two cards. Said
 * once, immediately above the first of the pair, so the second card is read as
 * its other half rather than as a separate stage. Keyed on the first card's
 * stage: a run that explained only the second gets no preface, because there is
 * then no pair on the page to introduce. */
const PAIR_PREFACE = {
  ga_selection: "GA Ensemble is explained in two sections: selection explanation "
              + "and combination explanation.",
  thompson_ranking: "Thompson Sampling is explained in two sections: ranking "
                  + "explanation and selection explanation.",
};

function pairPreface(stage) {
  const text = PAIR_PREFACE[stage.key];
  if (!text) return null;
  // `.prose`, so it is set in the same serif face and size as the narratives it
  // introduces — it is read as part of them, not as a UI label. The 68ch
  // measure is dropped: it exists to keep a paragraph readable over many lines,
  // and here it was breaking a single sentence that fits the card.
  return el("section", { class: "card card-tight" },
    el("p", { class: "prose", style: "margin: 0; max-width: none;", text }));
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
      el("h1", { text: `${payload.dataset_label || payload.dataset} · entity ${payload.entity}` }),
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

/* Display names for the agreement sources.
 *
 * The keys are the global IR's own — `borderline` for the off-by test,
 * `robust_consensus` for the robustness aggregation — and they stay as they
 * are: they are written into every result tree on disk and into the atom ids
 * the verifier joins on. This maps them to the names the stage cards use, so
 * one method is not called two things on one page. An unmapped source falls
 * back to its key with the underscores opened out, as before. */
const SOURCE_LABEL = {
  thompson: "Thompson",
  monte_carlo: "Monte Carlo",
  borderline: "Off-by-threshold",
  gan: "GAN",
  robust_consensus: "Robustness Aggregated",
};

const METRIC_LABEL = { f1: "F1", pr_auc: "PR-AUC", vus: "VUS" };

const sourceLabel = (source) => SOURCE_LABEL[source] || source.replace(/_/g, " ");

/* "GAN_F1" for a per-metric ranking, plain "Thompson" for the two that have
 * no metric of their own. `stage` and `metric` are absent on result trees
 * written before the strip was split per metric, hence the fallback. */
function agreementLabel(a) {
  const stage = sourceLabel(a.stage || a.source);
  const metric = METRIC_LABEL[a.metric];
  return metric ? `${stage}_${metric}` : stage;
}

function consensusStrip(payload) {
  if (!payload.agreement || !payload.agreement.length) return null;
  const panel = el("div", { class: "agreement-rankings", hidden: true });
  // Each chip toggles its own ranking, so several can stand open side by side
  // and be read against one another.
  const open = new Set();

  // Each stage owns a column, in order of first appearance, so every metric row
  // lines up under the first one and a ranking never moves when another opens.
  const stageColumn = new Map();
  payload.agreement.forEach((a) => {
    const stage = a.stage || a.source;
    if (!stageColumn.has(stage)) stageColumn.set(stage, stageColumn.size + 1);
  });
  const columnOf = (a) => stageColumn.get(a.stage || a.source);

  const renderPanel = () => {
    const shown = payload.agreement.filter((a) => open.has(a.source));
    panel.replaceChildren(...shown.map((a) => el("div",
      { style: `grid-column: ${columnOf(a)};` },
      el("h3", { class: "small", text: agreementLabel(a) }),
      el("ol", { class: "small mono ranking-list" },
        ...a.ranking.map((name) => el("li", { text: name }))))));
    panel.hidden = !shown.length;
  };

  const chips = payload.agreement.map((a) => {
    const glyph = a.agrees === true ? "✓" : a.agrees === false ? "≠" : "–";
    const cls = a.agrees === true ? "badge badge-ok"
      : a.agrees === false ? "badge badge-warn" : "badge badge-muted";
    const label = `${glyph} ${agreementLabel(a)}: ${a.top_pick || "—"}`;
    const column = `grid-column: ${columnOf(a)};`;
    if (!(a.ranking || []).length) {
      return el("span", { class: cls, style: column,
                          title: `${agreementLabel(a)}: ${a.top_pick}` }, label);
    }
    const chip = el("button", {
      type: "button", class: `${cls} badge-button`, style: column,
      "aria-expanded": "false",
      title: `Show or hide the full ${agreementLabel(a)} ranking`,
    }, label);
    chip.addEventListener("click", () => {
      if (open.has(a.source)) open.delete(a.source); else open.add(a.source);
      chip.setAttribute("aria-expanded", open.has(a.source) ? "true" : "false");
      renderPanel();
    });
    return chip;
  });

  return el("section", { class: "card card-tight stack" },
    el("h2", { class: "small muted", text: "Where the stages agreed" }),
    el("div", { class: "agreement-grid" }, chips),
    panel);
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
      el("a", { class: "button primary no-print", href: report.url,
                text: "Open report →" })),
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
  const table = makeSortable(el("table", {},
    el("thead", {}, el("tr", {},
      el("th", { text: "Stage" }), el("th", { class: "num", text: "Words" }),
      el("th", { class: "num", text: "Unsupported" }), el("th", { class: "num", text: "Omitted" }),
      el("th", { class: "num", text: "Claims" }))),
    el("tbody", {}, rows)));
  return el("details", { class: "card" },
    el("summary", { text: "Appendix: faithfulness and provenance" }),
    el("div", { class: "stack" },
      table,
      el("p", { class: "small muted", text:
        `Explanations read from myresults/explanations_nl/${payload.dataset}/${payload.entity}/` +
        (payload.iteration !== null && payload.iteration !== undefined
          ? ` (iteration ${payload.iteration})` : "") }),
      el("p", {}, el("a", { href: `/api/explanations/${dataset}/${entity}/download?stage=global`,
                            text: "Download the full report (.txt)" }))));
}

/* The "ⓘ Definitions" bulk toggle is gone with the glossaries it opened: they
 * are a page of their own now, so there is nothing on this one left to expand. */
function bulkToggles() {
  const fullButton = $("#toggle-all-full");
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
  payload.stages.forEach((stage) => {
    children.push(pairPreface(stage));
    children.push(stageCard(stage, payload));
  });
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
