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
import { trackSections, revealInRail } from "./spy.js";
import { wireCollapse } from "./sidenav.js";

const root = $("#result-root");
const dataset = root.dataset.dataset;
const entity = root.dataset.entity;
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

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
    const label = `Browse ${gallery.length} more plot${gallery.length === 1 ? "" : "s"}`;
    extra.push(el("button", {
      type: "button", class: "no-print", "aria-label": label,
      onclick: () => openLightbox(gallery, 0),
    }, el("span", { text: "Browse more plots" }),
       el("span", { class: "count", "aria-hidden": "true", text: String(gallery.length) })));
  }
  for (const descriptor of (payload.plots || {})._galleries || []) {
    if (!descriptor.id.startsWith(stage.plot_group)) continue;
    // The frame count is set apart from the title so a row of these scans as
    // names with numbers, not as one run of text per button.
    extra.push(el("button", {
      type: "button", class: "no-print",
      "aria-label": `${descriptor.title} (${descriptor.count})`,
      onclick: () => openGallery(descriptor),
    }, el("span", { text: descriptor.title }),
       el("span", { class: "count", "aria-hidden": "true", text: String(descriptor.count) })));
  }
  return extra.length
    ? el("div", { class: "gallery-links no-print" },
        el("span", { class: "gallery-label", text: "More figures" }), extra)
    : null;
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
  //
  // Wrapped with the sentence that names the list, so the two travel together:
  // on a wide screen they are one marginal note beside the narrative. The
  // sentence appears only when there is a list, as before — it otherwise
  // promises terms that never arrive.
  return el("aside", { class: "stage-terms" },
    el("p", { class: "terms-label", text:
      "The following terms and metrics are used to explain this stage:" }),
    el("dl", { class: "terms" },
      terms.flatMap(([term, definition]) => [
        el("dt", { text: `${term}:` }),
        el("dd", { text: definition }),
      ])));
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
  return el("a", { class: "button no-print stage-docs-link icon-launch",
                   href: `/docs/${dataset}/${entity}#${stage.doc_section}`,
                   target: "_blank", rel: "noopener",
                   text: "Stage description for the interested" });
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
  // aria-expanded both for assistive tech and for the stylesheet, which turns
  // the fold's chevron the way a <summary>'s turns.
  const button = el("button", { type: "button", class: "link-button", "aria-expanded": "false",
    text: `Show all ${spec.rows.length} rows (${hidden} more)` });
  button.addEventListener("click", () => {
    const shown = table.classList.toggle("expanded");
    button.setAttribute("aria-expanded", String(shown));
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

/* The section's number, as the documentation numbers its sections: the
 * figures are numbered after it ("Figure 3.2"). */
const sectionNumber = (n) => String(n);

function stageCard(stage, payload, number) {
  const faith = stage.faithfulness || {};
  // The stage's metadata as a small fact list. On a wide screen it is the
  // margin note beside the title; on a narrow one, a row under it.
  const facts = [
    stage.top_pick ? el("div", {},
      el("dt", { text: "First pick" }),
      el("dd", {}, el("span", { class: familyClass(stage.top_pick), text: stage.top_pick })))
      : null,
    stage.words ? el("div", {},
      el("dt", { text: "Narrative" }),
      el("dd", { class: "muted", text: `${stage.words} words` }))
      : null,
  ].filter(Boolean);
  const header = el("header", { class: "stage-head" },
    el("span", { class: "stage-num", "aria-hidden": "true", text: sectionNumber(number) }),
    el("div", { class: "stage-heading" },
      el("h3", { id: `stage-${stage.key}`, text: stage.title }),
      stage.question ? el("p", { class: "stage-question", text: stage.question }) : null),
    el("div", { class: "stage-meta" },
      facts.length ? el("dl", { class: "stage-facts" }, facts) : null,
      // With the metadata rather than in the prose: it is a way out of the
      // section, not part of its finding.
      docsLink(stage)));

  const body = el("div", { class: "stage-body stack" });
  // The prose is older than the facts it describes, so it may be narrating a
  // previous run. Said plainly and at the top, because everything below it —
  // summary, table, regime sentences — inherits the doubt.
  if (stage.stale) {
    body.append(el("p", { class: "notice", text:
      "This explanation is older than the results it describes: the stage was "
      + "re-run without re-generating its narrative. Re-run the narrator to refresh it." }));
  }
  // Guarded like `regimes` and `gallery` below: termsNode returns null for a
  // stage with no definitions, and Node.append — unlike el() — does not skip
  // null; it stringifies it, so the card would open with the word "null".
  const terms = termsNode(stage);
  if (terms) body.append(terms);
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
    body.append(el("details", { class: "caveats" },
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
    ? el("div", { class: "stage-foot row small dim no-print" }, provenance) : null;

  return el("section", { class: "stage-card" }, header, body, footer);
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
  // `.prose`, so it is set in the same serif face as the narratives it
  // introduces — it is read as part of them, not as a UI label. Set as a
  // part-opener (.pair-preface in the stylesheet, which also drops the
  // measure: one sentence that fits the column should not be broken by it).
  return el("section", { class: "pair-preface" },
    el("p", { class: "prose", text }));
}

function decisionHero(payload) {
  const decision = payload.decision || {};
  const chosen = Array.isArray(decision.chosen) ? decision.chosen
    : decision.chosen ? [decision.chosen] : [];
  const faith = payload.faithfulness || {};

  // What used to be the one-line byline, as a ruled fact list in the margin:
  // the same fields under the same conditions, each findable on its own.
  const fact = (term, ...value) => el("div", {}, el("dt", { text: term }), el("dd", {}, ...value));
  const facts = [
    decision.framework_choice
      ? fact("Decision", el("span", { class: "is-decision",
                                      text: decision.framework_choice.replace(/_/g, " ") }))
      : null,
    chosen.length
      ? fact("Chosen", chosen.map((d) => el("span", { class: familyClass(d), text: d })))
      : null,
    decision.decision_metric_label ? fact("Fitness", decision.decision_metric_label) : null,
    payload.model ? fact("Narrated by", el("span", { class: "mono", text: payload.model })) : null,
    // Claims checked, unsupported and omitted moved to the last tile, under
    // the same condition, so they are said once.
    payload.generated_at ? fact("Generated", timeAgo(payload.generated_at)) : null,
  ].filter(Boolean);

  return el("section", { class: "decision-hero", id: "decision" },
    el("div", { class: "hero-main" },
      el("h1", { text: `${payload.dataset_label || payload.dataset} · entity ${payload.entity}` }),
      // Verbatim from the grounded decision atom — the page invents nothing.
      payload.decision_text
        ? el("p", { class: "verdict", text: payload.decision_text })
        : el("p", { class: "verdict muted", text: "No decision recorded for this run." }),
      decisionTiles(decision, faith),
      decision.reason ? el("p", { class: "prose small muted hero-reason", text: decision.reason }) : null),
    facts.length
      ? el("aside", { class: "hero-facts", "aria-label": "Decision and provenance" },
          el("dl", { class: "facts" }, facts))
      : null);
}

/* The decision's numbers as a ruled row of figures: the two candidates it
 * weighed, the margin between them, and how much of the prose the verifier
 * could not support. Every value is read from the record as written — rounded
 * nowhere, derived from nothing — and a tile whose number the record does not
 * hold (an older result tree) is simply not drawn. The margin carries the
 * record's own sign, set with a true minus so "−0.1463" is as wide as "+0.1463";
 * which side is ahead is left to that sign rather than worded by the page. */
function decisionTiles(decision, faith) {
  const num = (v) => typeof v === "number" && Number.isFinite(v);
  const metric = decision.decision_metric_label || decision.decision_metric_formula || "fitness";
  const names = (v) => (Array.isArray(v) ? v : v ? [v] : []);
  const chips = (list) => list.map((d) => el("span", { class: familyClass(d), text: d }));
  // The flag sits with the detector it names rather than in the label or
  // beside the number, so neither of those changes height on the chosen tile.
  const tile = (label, value, sub, chosen) => el("div",
    { class: chosen ? "kpi is-chosen" : "kpi" },
    el("dt", { class: "kpi-label", text: label }),
    el("dd", { class: "kpi-value", text: value }),
    el("dd", { class: "kpi-sub" }, sub && sub.length ? sub : null,
      chosen ? el("span", { class: "kpi-flag", text: "✓ chosen" }) : null));

  const tiles = [];
  if (num(decision.single_model_score)) {
    tiles.push(tile(`Single model · ${metric}`, String(decision.single_model_score),
      chips(names(decision.single_model)), decision.framework_choice === "single_model"));
  }
  if (num(decision.ensemble_score)) {
    tiles.push(tile(`GA ensemble · ${metric}`, String(decision.ensemble_score),
      chips(names(decision.ensemble)), decision.framework_choice === "ensemble"));
  }
  if (num(decision.score_margin_ensemble_minus_single)) {
    const margin = decision.score_margin_ensemble_minus_single;
    // No-break spaces round the minus, so a wrapped label never starts its
    // second line with "− single" and reads like a list bullet.
    tiles.push(tile("Margin · ensemble\u00a0−\u00a0single",
      `${margin > 0 ? "+" : ""}${String(margin).replace("-", "−")}`,
      [el("span", { text: `in ${metric}` })]));
  }
  // "Unsupported claims", not "faithfulness": the number is the share the
  // verifier could not ground, so 0.0% is the good end — a label promising
  // faithfulness above a big 0.0% reads as the opposite.
  if (faith && num(faith.n_claims) && faith.n_claims > 0) {
    tiles.push(tile("Unsupported claims", pct(faith.hallucination_rate),
      [el("span", { text: `of ${faith.n_claims} checked`
        + (num(faith.omission_rate) ? ` · ${pct(faith.omission_rate)} of facts omitted` : "") })]));
  }
  return tiles.length
    ? el("dl", { class: "kpis", "aria-label": "The decision in numbers" }, tiles) : null;
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

  return el("section", { class: "agreement-panel stack", id: "agreement" },
    el("div", {},
      el("h2", { class: "small muted", text: "Where the stages agreed" }),
      // The key to the glyphs: `agrees` is whether the source's first pick is
      // the final single model (Explainability/ir.py, agrees_with_final_single).
      el("p", { class: "agreement-legend" },
        el("span", { class: "glyph", text: "✓" }), " same first pick as the final single model · ",
        el("span", { class: "glyph", text: "≠" }), " a different first pick · ",
        el("span", { class: "glyph", text: "–" }), " not compared. Open one to see its whole ranking.")),
    el("div", { class: "agreement-grid" }, chips),
    panel);
}

function missingSection(payload) {
  if (!payload.missing_stages || !payload.missing_stages.length) return null;
  return el("section", { class: "card card-tight stack" },
    el("h2", { class: "small muted", text: "Stages without an explanation" }),
    el("ul", { class: "small missing-list" }, payload.missing_stages.map((m) =>
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

  return el("section", { class: "card end-card" },
    el("div", { class: "row-between" },
      el("h2", { id: "comprehensive", text: "Comprehensive results" }),
      el("a", { class: "button primary icon-arrow no-print", href: report.url,
                text: "Open report" })),
    el("p", { class: "prose small muted" },
      "Measured module timings, memory, the ranking each stage produced and the final " +
      "decision, exactly as the pipeline wrote them. Separate from the explanation above, " +
      "and the binding record for any number that appears in both."),
    el("p", { class: "small muted mono end-meta", text: meta }));
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
  return el("details", { class: "card appendix", id: "appendix" },
    el("summary", { text: "Appendix: faithfulness and provenance" }),
    el("div", { class: "stack" },
      // Scrolls in its own box on a phone, like every other wide table.
      el("div", { class: "table-scroll" }, table),
      el("p", { class: "small muted", text:
        `Explanations read from myresults/explanations_nl/${payload.dataset}/${payload.entity}/` +
        (payload.iteration !== null && payload.iteration !== undefined
          ? ` (iteration ${payload.iteration})` : "") }),
      el("p", {}, el("a", { href: `/api/explanations/${dataset}/${entity}/download?stage=global`,
                            text: "Download the full report (.txt)" }))));
}

/* Figures are numbered per section — "Figure 3.2" is the second figure of
 * section 03 — and the number is written once, here, rather than counted in
 * CSS: a CSS counter skips whatever sits in a closed disclosure, so every
 * number after it would shift when the reader opened one. */
function numberFigures() {
  $$(".stage-card", root).forEach((card, s) => {
    $$("figure > figcaption", card).forEach((caption, f) => {
      caption.dataset.fig = `${s + 1}.${f + 1}`;
    });
  });
}

/* "On this page": every section of the explanation, numbered as the
 * sections are, in the rail beside them (a strip of chips on a narrow screen).
 * Built from the same payload the sections are, so the two cannot disagree.
 * Each stage carries the detector it put first — what the contents grid under
 * the hero used to add, before the rail made that grid a second copy of
 * itself. Plain fragment links — the browser does the jump, so Back and
 * copy-link behave as they always have. */
function pageRail(payload) {
  const nav = $("#page-rail");
  const list = $("#page-rail-list");
  if (!nav || !list) return;
  const item = (id, num, label, pick) => el("li", {},
    el("a", { href: `#${id}`, "data-target": id },
      el("span", { class: "rail-num", "aria-hidden": num ? null : "true", text: num }),
      el("span", { class: "rail-label", text: label }),
      pick ? el("span", { class: `rail-pick ${familyClass(pick)}`, text: pick }) : null));
  const items = [item("decision", "", "Decision")];
  if (payload.agreement && payload.agreement.length) items.push(item("agreement", "", "Agreement"));
  payload.stages.forEach((stage, i) =>
    items.push(item(`stage-${stage.key}`, sectionNumber(i + 1), stage.title, stage.top_pick)));
  if (payload.comprehensive) items.push(item("comprehensive", "", "Comprehensive results"));
  items.push(item("appendix", "", "Appendix"));
  list.replaceChildren(...items);
  nav.hidden = false;
  const layout = nav.closest(".rail-layout");
  const collapse = $("#rail-collapse");
  if (layout && collapse) wireCollapse(layout, collapse, "ramses-result-nav-collapsed");
  spyOnSections(nav, list);
}

/* Marks the rail link of the section being read, with the rule the report and
 * the documentation use (spy.js). Presentation only: a class and
 * aria-current on the links. Each target counts from the top of the section
 * around it — the stage card, not its heading — so a reader in the middle of
 * a long stage is still in that stage. */
function spyOnSections(nav, list) {
  const links = $$("a[data-target]", list);
  if (!links.length) return;
  const byId = new Map(links.map((a) => [a.dataset.target, a]));
  let current = null;
  const follow = trackSections(links.map((a) => a.dataset.target), {
    sectionOf: (node) =>
      node.closest(".stage-card, .end-card, .decision-hero, .agreement-panel, details.card") || node,
    mark: (id) => {
      if (current) { current.classList.remove("is-current"); current.removeAttribute("aria-current"); }
      current = byId.get(id) || null;
      if (!current) return;
      current.classList.add("is-current");
      current.setAttribute("aria-current", "location");
      revealInRail(current, list, nav);
    },
  });
  // The link the reader followed is the answer, at once and for as long as
  // the jump takes — not whichever section the scroll happens to pass.
  links.forEach((a) => a.addEventListener("click", () => follow(a.dataset.target)));
  window.addEventListener("hashchange", () =>
    follow(decodeURIComponent((location.hash || "").replace(/^#/, ""))));
}

/* Land an inbound #fragment on its own heading.
 *
 * The browser resolves the fragment before the payload has arrived, so its
 * jump is missed; and one scroll once the sections exist is not enough either,
 * because the figures above the target load as it comes into range and push it
 * down — a cold /result/…#stage-gan used to stop over a thousand pixels short.
 * So the target's closed disclosures are opened, the jump is made, and it is
 * made again whenever the page changes height, until the page has been still
 * for a moment or the reader takes over (wheel, touch, key, click). Instant:
 * on a cold load there is no "from" for an animation to show. */
function openFromHash() {
  const id = decodeURIComponent((location.hash || "").replace(/^#/, ""));
  if (!id) return false;
  const target = document.getElementById(id);
  if (!target || !root.contains(target)) return false;
  for (let d = target.closest("details:not([open])"); d; d = d.closest("details:not([open])")) {
    d.open = true;
  }
  const pin = () => target.scrollIntoView({ block: "start", behavior: "instant" });
  pin();
  const QUIET_MS = 1200, MAX_MS = 8000, INPUTS = ["wheel", "touchstart", "keydown", "pointerdown"];
  let quiet = 0;
  const observer = new ResizeObserver(() => {
    pin();
    clearTimeout(quiet);
    quiet = setTimeout(stop, QUIET_MS);
  });
  const cap = setTimeout(stop, MAX_MS);
  function stop() {
    observer.disconnect();
    clearTimeout(quiet);
    clearTimeout(cap);
    INPUTS.forEach((type) => window.removeEventListener(type, stop, true));
  }
  INPUTS.forEach((type) => window.addEventListener(type, stop, { capture: true, passive: true }));
  observer.observe(root);
  quiet = setTimeout(stop, QUIET_MS);
  return true;
}

/* A margin glossary longer than the prose beside it.
 *
 * The terms float into the margin column and a figure band spans that column,
 * so a band has to clear the float: where six terms sat beside a short summary
 * (Thompson Sampling: Selection), the reading column stood empty for a quarter
 * of a screen before the figures. When that would happen, the first band
 * starts where the prose ends and stays out of the margin — it gives up the
 * margin's width, not its place — and the glossary carries on beside it.
 * Measured rather than guessed, because it depends on the fonts, the width and
 * which disclosures are open, and re-measured when any of those change. */
const BANDS = ":scope > .figures, :scope > details:not(.narrative):not(.caveats), :scope > .gallery-links";
const OVERRUN_PX = 48;

function fitMarginNotes() {
  $$(".stage-body", root).forEach((body) => {
    $$(".beside-terms", body).forEach((node) => node.classList.remove("beside-terms"));
    const terms = $(":scope > .stage-terms", body);
    const band = $(BANDS, body);
    if (!terms || !band || getComputedStyle(terms).float === "none") return;
    let prose = band.previousElementSibling;
    if (prose === terms) prose = null;
    if (!prose) return;
    const overrun = terms.getBoundingClientRect().bottom - prose.getBoundingClientRect().bottom;
    if (overrun > OVERRUN_PX) band.classList.add("beside-terms");
  });
}

function watchMarginNotes() {
  let frame = 0;
  const later = () => {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; fitMarginNotes(); });
  };
  fitMarginNotes();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(later);
  window.addEventListener("resize", later);
  // `toggle` does not bubble; captured here, it covers every disclosure.
  root.addEventListener("toggle", later, true);
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
    root.replaceChildren(el("section", { class: "card stack empty-card" },
      el("h1", { text: "No explanation yet" }),
      el("p", { class: "muted", text: error.detail && error.detail.hint
        ? error.detail.hint
        : `Nothing has been generated for ${dataset} / ${entity}.` }),
      // The page's one way forward, so it is drawn as its primary action.
      el("p", {}, el("a", { class: "button primary icon-arrow", href: "/", text: "Configure a run" }))));
    return;
  }

  // No second table of contents under the hero: the rail beside the page (a
  // strip of chips above it on a narrow screen) lists the same sections with
  // the same numbers, and now carries each stage's first pick too, so a grid
  // repeating it here made two navigations for one page.
  const children = [decisionHero(payload), consensusStrip(payload)];
  if (payload.degraded) {
    children.push(el("p", { class: "banner banner-warn", text:
      "This result predates the structured global report, so only the per-stage " +
      "explanations are shown." }));
  }
  payload.stages.forEach((stage, i) => {
    children.push(pairPreface(stage));
    children.push(stageCard(stage, payload, i + 1));
  });
  children.push(missingSection(payload));
  children.push(comprehensiveCard(payload));
  children.push(appendix(payload));

  root.replaceChildren(el("div", { class: "stack-lg" }, children.filter(Boolean)));
  numberFigures();


  bulkToggles();
  attachLightbox();
  pageRail(payload);
  watchMarginNotes();

  openFromHash();
  // A fragment followed later (the rail, the contents) is a native jump; this
  // only adds what the browser does not do: opening the disclosure it is in.
  window.addEventListener("hashchange", openFromHash);
}

render();
