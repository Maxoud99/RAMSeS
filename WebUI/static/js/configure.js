/* The run form.
 *
 * The command preview comes from the server's dry-run endpoint rather than
 * being rebuilt here, so what the user copies is exactly what gets spawned.
 */

import { $, $$, el, getJSON, postJSON, pct, timeAgo, familyClass } from "./dom.js";

const STAGE_LABELS = {
  ga: "Genetic algorithm", thompson: "Thompson Sampling", gan: "GAN perturbations",
  offby: "Off-by-threshold", montecarlo: "Monte Carlo",
};
const ALL_STAGES = Object.keys(STAGE_LABELS);

let catalog = null;
let selectedDetectors = new Set();
// name -> {available, family}, so the training banner can tell a picked
// detector that exists on disk from one that has to be fitted first.
let detectorInfo = new Map();

function datasetOf() { return $("#dataset").value; }
function entityOf() { return $("#entity").value; }

function selectedStages() {
  return $$("#stages input:checked").map((i) => i.value);
}

function currentBody(extra = {}) {
  return {
    dataset: datasetOf(),
    entity: entityOf(),
    // Every chosen type, in the order chosen; the run injects each one.
    anomaly_types: selectedTypes.slice(),
    anomaly_rate: Number($("#anomaly_rate").value) || null,
    decision_metrics: metricWeights(),
    stages: selectedStages(),
    detectors: Array.from(selectedDetectors),
    explain: $("#explain").checked,
    // `parallel` is no longer offered: runs go sequential. build_argv still
    // honours it, so a caller that sets it programmatically keeps working.
    overwrite: $("#overwrite").checked,
    iteration: Number($("#iteration").value) || 5,
    timeout: Number($("#timeout").value) || 14400,
    llm_model: $("#llm_model").value.trim() || null,
    llm_base_url: $("#llm_base_url").value.trim() || null,
    ...onlineBody(),
    ...extra,
  };
}

/* ── Rendering ─────────────────────────────────────────────────────────────── */

function renderWarnings() {
  const box = $("#warnings");
  box.replaceChildren(...(catalog.warnings || []).map((w) =>
    el("p", { class: "banner banner-warn", text: w.text })));
}

function renderDatasets() {
  const select = $("#dataset");
  // `value` stays the canonical key the CLI expects; only the label is pretty.
  // A dataset whose key says little (TSB-AD-U, UCR) carries its note after
  // the name, so the picker reads as what each one is.
  select.replaceChildren(...catalog.datasets.map((d) =>
    el("option", { value: d.name, disabled: !d.runnable },
      (d.label || d.name) + (d.note ? ` — ${d.note}` : "")
      + (d.runnable ? "" : " (not supported)"))));
  select.addEventListener("change", () => {
    labelChoice = null;          // each dataset starts from its own default
    renderEntities(); refreshDetectors();
  });
  renderEntities();
}

function entitiesOf() {
  const dataset = catalog.datasets.find((d) => d.name === datasetOf());
  return dataset ? dataset.entities : [];
}

/* The browser filters the datalist as you type, so the field is both the filter
 * and the choice. It also accepts free text, which a <select> could not — hence
 * entityIsValid below, which is what now guards the run. */
function renderEntities() {
  const dataset = catalog.datasets.find((d) => d.name === datasetOf());
  const trained = new Set((dataset && dataset.trained) || []);
  const entities = entitiesOf().slice()
    // Entities with checkpoints first: those run immediately, the rest have to
    // train from scratch. Junk files in the data directory sink to the bottom.
    .sort((a, b) => (trained.has(b) - trained.has(a)) || a.localeCompare(b, undefined, { numeric: true }));

  $("#entity-options").replaceChildren(...entities.map((e) =>
    el("option", { value: e }, trained.has(e) ? "" : "needs training")));
  const input = $("#entity");
  // TSB-AD-U has 870 series: the field is the filter, so it says how many it filters.
  input.placeholder = entities.length > 1 ? `Type to filter ${entities.length} entities…`
                                          : "Type to filter…";
  if (input.value && !entities.includes(input.value)) input.value = "";
  syncEntityValidity();
}

function entityIsValid() {
  const value = $("#entity").value.trim();
  return !!value && entitiesOf().includes(value);
}

function syncEntityValidity() {
  const input = $("#entity");
  const empty = !input.value.trim();
  const bad = !empty && !entityIsValid();
  input.setAttribute("aria-invalid", bad ? "true" : "false");
  $("#entity-error").hidden = !bad;
  if (bad) $("#entity-error").textContent = `No entity named "${input.value.trim()}".`;
}

/* ── Anomaly types: Carbon's MultiSelect ─────────────────────────────────────
 *
 * More than one type can be injected; each adds its own copies of the series
 * (catalog `copies`) and every stage evaluates all of them, so the field says
 * what the choice costs. A select-only combobox: focus stays on the field
 * while its list is open and aria-activedescendant names the active option;
 * Up and Down move, Space or Enter toggles, Home and End jump, Escape and Tab
 * close. Spikes is the default; at least one is required (readiness()). */

let selectedTypes = ["spikes"];
let activeType = 0;

function anomalyTypes() {
  return (catalog && catalog.anomaly_types) || [];
}

function typeCopies(token) {
  const t = anomalyTypes().find((x) => x.token === token);
  return t && Number.isFinite(t.copies) ? t.copies : 1;
}

function shortTypeLabel(token) {
  const t = anomalyTypes().find((x) => x.token === token);
  return t ? (t.label || token).replace(/ \(.*\)$/, "") : token;
}

function renderAnomalyTypes() {
  const types = anomalyTypes();
  if (!types.some((t) => t.token === "spikes")) selectedTypes = types.length ? [types[0].token] : [];
  const list = $("#anomaly_types-list");
  list.replaceChildren(...types.map((t, i) => el("li", {
    role: "option", id: `anomaly-type-${t.token}`, class: "multiselect-option",
    "data-token": t.token, "data-index": String(i), "aria-selected": "false",
    onclick: () => { activeType = i; toggleType(t.token); },
    onmousemove: () => { if (activeType !== i) { activeType = i; syncTypeList(); } },
  },
    el("span", { class: "multiselect-check", "aria-hidden": "true" }),
    el("span", { class: "multiselect-label", text: t.label || t.token }),
    el("span", { class: "multiselect-copies", text: `${t.copies || 1} ${t.copies === 1 || !t.copies ? "copy" : "copies"}` }))));

  const button = $("#anomaly_types-button");
  button.addEventListener("click", () => setTypeList(list.hidden));
  button.addEventListener("keydown", (event) => {
    const open = !list.hidden;
    const last = types.length - 1;
    const moves = { ArrowDown: activeType + 1, ArrowUp: activeType - 1, Home: 0, End: last };
    if (event.key in moves) {
      event.preventDefault();
      if (!open) { setTypeList(true); return; }
      activeType = Math.max(0, Math.min(last, moves[event.key]));
      syncTypeList();
    } else if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      if (!open) setTypeList(true);
      else toggleType(types[activeType].token);
    } else if (event.key === "Escape" && open) {
      event.preventDefault();
      setTypeList(false);
    } else if (event.key === "Tab" && open) {
      setTypeList(false);
    }
  });
  $("#anomaly_types-clear").addEventListener("click", () => {
    selectedTypes = [];
    syncTypes();
    onChange();
    button.focus();
  });
  // A press anywhere outside the field closes the list.
  document.addEventListener("pointerdown", (event) => {
    if (!list.hidden && !$("#anomaly_types").contains(event.target)) setTypeList(false);
  });
  syncTypes();
}

function setTypeList(open) {
  const list = $("#anomaly_types-list");
  const button = $("#anomaly_types-button");
  list.hidden = !open;
  button.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) {
    const first = anomalyTypes().findIndex((t) => selectedTypes.includes(t.token));
    activeType = first >= 0 ? first : 0;
  }
  syncTypeList();
}

function toggleType(token) {
  if (selectedTypes.includes(token)) selectedTypes = selectedTypes.filter((t) => t !== token);
  else selectedTypes.push(token);
  syncTypes();
  onChange();
}

/* The list's state: which options are chosen, which one is active. */
function syncTypeList() {
  const list = $("#anomaly_types-list");
  const button = $("#anomaly_types-button");
  $$("#anomaly_types-list [role=option]").forEach((option, i) => {
    option.setAttribute("aria-selected", selectedTypes.includes(option.dataset.token) ? "true" : "false");
    option.classList.toggle("is-active", !list.hidden && i === activeType);
  });
  const active = $$("#anomaly_types-list [role=option]")[activeType];
  if (!list.hidden && active) {
    button.setAttribute("aria-activedescendant", active.id);
    // Kept in view inside the list, without moving the page.
    if (active.offsetTop < list.scrollTop) list.scrollTop = active.offsetTop;
    else if (active.offsetTop + active.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = active.offsetTop + active.offsetHeight - list.clientHeight;
    }
  } else {
    button.removeAttribute("aria-activedescendant");
  }
}

/* The field, its count tag, the cost line and the notes, from the choice. */
function syncTypes() {
  syncTypeList();
  const n = selectedTypes.length;
  const labels = selectedTypes.map(shortTypeLabel);
  $("#anomaly_types-value").textContent = n ? labels.join(", ") : "Choose at least one";
  $("#anomaly_types-value").classList.toggle("is-placeholder", !n);
  $("#anomaly_types-tag").hidden = n < 2;
  $("#anomaly_types-count").textContent = String(n);
  const copies = selectedTypes.reduce((sum, t) => sum + typeCopies(t), 0);
  $("#anomaly-copies").textContent = n
    ? `${n} ${n === 1 ? "type" : "types"} · ${copies} ${copies === 1 ? "copy" : "copies"} of the series`
      + (copies > 1 ? "; every stage evaluates each copy." : ".")
    : "";
  const notes = anomalyTypes().filter((t) => selectedTypes.includes(t.token) && t.note);
  $("#anomaly-note").replaceChildren(...notes.map((t) => el("li", {},
    el("span", { class: "anomaly-note-type", text: `${shortTypeLabel(t.token)}: ` }), t.note)));
}

function renderDecisionMetrics() {
  $("#decision_metrics").replaceChildren(...(catalog.decision_metrics || []).map((m) =>
    el("label", { class: "toggle" },
      el("input", { type: "checkbox", value: m.token,
                    checked: m.token === "f1" || m.token === "pr_auc",
                    onchange: onChange }),
      el("span", { text: m.label || m.token }),
      el("input", { type: "number", class: "metric-weight", value: "1",
                    min: "0", step: "0.05", "data-metric": m.token,
                    title: "Weight in the fitness function", oninput: onChange }))));
  syncMetricNotes();
}

function selectedMetrics() {
  return $$("#decision_metrics input[type=checkbox]:checked").map((i) => i.value);
}

/* Sent as metric -> weight. The server normalises, so these need not sum to 1;
 * an unchecked metric is simply absent. */
function metricWeights() {
  const out = {};
  for (const token of selectedMetrics()) {
    const box = $(`#decision_metrics input[data-metric="${token}"]`);
    const weight = box ? Number(box.value) : 1;
    out[token] = Number.isFinite(weight) && weight >= 0 ? weight : 1;
  }
  return out;
}

function fitnessFormula() {
  const symbols = new Map((catalog.decision_metrics || [])
    .map((m) => [m.token, m.symbol || m.token]));
  const weights = metricWeights();
  const tokens = Object.keys(weights);
  const total = tokens.reduce((sum, t) => sum + weights[t], 0);
  if (!tokens.length || !(total > 0)) return "";
  if (tokens.length === 1) return symbols.get(tokens[0]);
  return tokens
    .map((t) => `${+(weights[t] / total).toFixed(3)} * ${symbols.get(t)}`)
    .join(" + ");
}

/* VUS judges a detector over a range of buffer widths, which a short entity's
 * Thompson windows cannot fill. Warn rather than refuse. */
function syncMetricNotes() {
  const chosen = selectedMetrics();
  const picked = new Set(chosen);
  $$("#decision_metrics input.metric-weight").forEach((box) => {
    box.disabled = !picked.has(box.dataset.metric);
    box.hidden = box.disabled;
  });
  const formula = fitnessFormula();
  const formulaLine = $("#metric-formula");
  if (formulaLine) formulaLine.textContent = formula ? `Fitness = ${formula}` : "";
  // The field's own error is the readiness check's sentence, so it and the
  // summary's checklist say the same thing about the same rule.
  const fitness = readinessOf("fitness");
  $("#metric-error").hidden = fitness.ok;
  $("#metric-error").textContent = `${fitness.todo}.`;
  const entity = entityOf();
  const short = (datasetOf() || "").toLowerCase() === "skab";
  $("#metric-note").textContent =
    (chosen.includes("vus") && short && entity)
      ? "VUS performs poorly in Thompson Sampling in short entities."
      : "";
}

/* Paper Table I group -> css suffix. Unknown or missing becomes "none", which
 * has no colour rule, so a family added without a group degrades to the plain
 * chip rather than to a broken class name. */
function groupClass(group) {
  return String(group || "none").toLowerCase();
}

/* Group -> the detector names in it, rebuilt whenever the entity changes.
 * `syncGroupButtons` reads it on every selection change to decide which buttons
 * are lit, so it has to outlive the render that created the buttons. */
let groupMembers = new Map();

/* One select-all button per group, built from what the API reports rather than
 * from a list here, so a group added to DETECTOR_GROUPS appears with no change
 * to this file. A group with no detectors in the pool (FM today) is rendered
 * disabled rather than hidden: the taxonomy is the paper's, and showing where
 * a future foundation model would land is more use than silently omitting it.
 *
 * The buttons are toggles, not one-way selects: a second click clears the group
 * again. Nothing in the detector pool is a checkbox any more — group, family
 * and detector all say "chosen" the same way, with aria-pressed and a filled
 * button — so one glance answers what is selected at every level. */
function renderGroupButtons(detectors) {
  const box = $("#detector-groups");
  if (!box) return;
  const order = [];
  groupMembers = new Map();
  detectors.forEach((d) => {
    const group = d.group;
    if (!group) return;
    if (!groupMembers.has(group)) { groupMembers.set(group, []); order.push(group); }
    groupMembers.get(group).push(d.name);
  });
  for (const group of (catalog.detector_groups || [])) {
    if (!groupMembers.has(group)) { groupMembers.set(group, []); order.push(group); }
  }
  const labels = (catalog && catalog.group_labels) || {};
  box.replaceChildren(...order.map((group) => {
    const names = groupMembers.get(group);
    const label = labels[group] || group;
    return el("button", {
      type: "button",
      class: `small grp-${groupClass(group)}`,
      "data-group": group,
      "aria-pressed": "false",
      disabled: names.length === 0,
      title: names.length
        ? `Select or clear all ${names.length} ${label} detectors`
        : `No ${label} detectors in the pool yet`,
      onclick: () => {
        // Same rule as the family button: if any member is unselected, select
        // the whole group; if they are all selected, clear it.
        const turnOn = names.some((n) => !selectedDetectors.has(n));
        names.forEach((n) => {
          if (turnOn) selectedDetectors.add(n);
          else selectedDetectors.delete(n);
        });
        onChange();
      },
    }, label);
  }));
  syncGroupButtons();
}

/* A group reads as chosen only when every one of its detectors is. Anything
 * less is "partial", which is not the same claim and gets its own state — a
 * button that lit up on one chip out of fifteen would be telling the reader
 * the group is selected when it is not. */
function syncGroupButtons() {
  $$("#detector-groups button").forEach((button) => {
    const names = groupMembers.get(button.dataset.group) || [];
    const chosen = names.filter((n) => selectedDetectors.has(n)).length;
    const all = names.length > 0 && chosen === names.length;
    // "mixed" is the toggle button's third state: assistive tech announces
    // "partially pressed", which is exactly the claim `is-partial` draws.
    button.setAttribute("aria-pressed", all ? "true" : chosen > 0 ? "mixed" : "false");
    button.classList.toggle("is-on", all);
    button.classList.toggle("is-partial", chosen > 0 && !all);
  });
}

/* The same treatment one level down, for the chips and the family buttons.
 *
 * Chips used to be a <label> wrapping a checkbox, which meant the pool had
 * three different ways of saying "chosen" on one screen: a tick for a
 * detector, a filled button for a group, and nothing at all for a family. They
 * are all buttons now and all say it the same way, so `selectedDetectors` —
 * which has always been the only thing `build_argv` reads — is the single
 * source of truth and the DOM is purely its reflection.
 *
 * A family gets `is-partial` for the same reason a group does: some of LOF
 * picked is not LOF picked. */
function syncDetectorButtons() {
  $$("#detectors button[data-detector]").forEach((button) => {
    const on = selectedDetectors.has(button.dataset.detector);
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.classList.toggle("is-on", on);
  });
  $$("#detectors button[data-family]").forEach((button) => {
    const names = $$(`#detectors button[data-detector^="${button.dataset.family}_"]`)
      .map((chip) => chip.dataset.detector)
      .filter((name) => name.slice(0, name.lastIndexOf("_")) === button.dataset.family);
    const chosen = names.filter((n) => selectedDetectors.has(n)).length;
    const all = names.length > 0 && chosen === names.length;
    button.setAttribute("aria-pressed", all ? "true" : chosen > 0 ? "mixed" : "false");
    button.classList.toggle("is-on", all);
    button.classList.toggle("is-partial", chosen > 0 && !all);
  });
}

/* One info circle per family, listing every instance's hyperparameters.
 *
 * These used to sit on the chips themselves, which spent a line of the row on
 * four near-identical strings ("contamination 0.1", "0.15", "0.2", "0.25") to
 * make a distinction the reader almost never acts on. Per family they are one
 * hover away and, grouped together, they answer the question the reader
 * actually has — what separates LOF_1 from LOF_4 — which four scattered chips
 * never did.
 *
 * The circle sits beside the family button rather than inside a chip's <label>,
 * so hovering or clicking it cannot toggle a checkbox.
 */
function familyParams(family, members) {
  const lines = members.map((d) => {
    // An untrained instance still has hyperparameters — the grid defines them
    // whether or not a checkpoint exists — so it is named like the rest and
    // only marked as not yet on disk.
    const suffix = d.available ? "" : "  (untrained)";
    if (!d.params) return `${d.name}: no parameters recorded${suffix}`;
    const parts = [];
    if (d.params.label) parts.push(d.params.label);
    if (d.params.window_size !== null && d.params.window_size !== undefined) {
      parts.push(`window ${d.params.window_size}`);
    }
    if (d.params.window_step !== null && d.params.window_step !== undefined) {
      parts.push(`step ${d.params.window_step}`);
    }
    return `${d.name}: ${parts.length ? parts.join(", ") : "no varying parameter"}${suffix}`;
  });
  return el("span", {
    class: "paraminfo",
    tabindex: "0",                     // reachable without a mouse
    role: "note",
    "aria-label": `Hyperparameters of the ${family} instances: ${lines.join("; ")}`,
    "data-tip": lines.join("\n"),
  }, "i");
}

async function refreshDetectors() {
  const dataset = datasetOf(), entity = entityOf();
  const box = $("#detectors");
  if (!dataset || !entity) { box.replaceChildren(); return; }

  // Availability is per entity (it depends on which .pth files exist), so it
  // needs its own request rather than coming from the cached catalog.
  let detectors;
  try {
    detectors = (await getJSON(
      `/api/detectors/${encodeURIComponent(dataset)}/${encodeURIComponent(entity)}`)).detectors;
  } catch (e) {
    detectors = (catalog.all_detectors || []).map(
      (name) => ({ name, available: true, params: null }));
  }

  // Nothing is selected by default. What is on disk used to be pre-ticked,
  // which quietly decided the pool for anyone who did not read it — and with 69
  // detectors that is a large decision to make on someone's behalf. An empty
  // start means the run is whatever was deliberately picked.
  selectedDetectors = new Set();
  detectorInfo = new Map(detectors.map(
    (d) => [d.name, { available: !!d.available,
                      family: d.family || d.name.split("_")[0] }]));

  renderGroupButtons(detectors);

  const byFamily = {};
  detectors.forEach((d) => {
    const family = d.family || d.name.split("_")[0];
    (byFamily[family] = byFamily[family] || []).push(d);
  });

  // The API lists families group by group (Table I order), so each run of one
  // group becomes a block under its own heading: thirty-odd rows read as four
  // labelled blocks. Presentation only — the rows and their buttons are built
  // exactly as before, and selection still reads `selectedDetectors` alone.
  const groupLabels = (catalog && catalog.group_labels) || {};
  const blocks = [];
  Object.entries(byFamily).forEach(([family, members]) => {
    const group = members[0].group || "";
    let block = blocks[blocks.length - 1];
    if (!block || block.group !== group) {
      block = { group, rows: [] };
      blocks.push(block);
    }
    block.rows.push(familyRow(family, members));
  });
  box.replaceChildren(...blocks.map(({ group, rows }) => el("div",
    { class: `pool-group grp-${groupClass(group)}` },
    group
      ? el("h3", { class: "pool-group-title" }, groupLabels[group] || group,
          el("span", { class: "pool-group-count",
                       text: `${rows.length} ${rows.length === 1 ? "family" : "families"}` }))
      : null,
    rows)));
  onChange();
}

/* One family's row: the select-all button, the hyperparameter circle, then a
 * chip per detector. */
function familyRow(family, members) {
  // Clicking the family name toggles the whole family: if any member is
  // unselected it selects them all, otherwise it clears them.
  const familyLabel = family;
  const familyButton = el("button", {
    type: "button", class: `small family-button grp-${groupClass(members[0].group)}`,
    "data-family": family,
    "aria-pressed": "false",
    // Its minimum width lives in the stylesheet (.family-button), with the
    // reason it is a minimum rather than a fixed width.
    title: `Select or clear every ${familyLabel} detector`,
    onclick: () => {
      const turnOn = members.some((d) => !selectedDetectors.has(d.name));
      members.forEach((d) => {
        if (turnOn) selectedDetectors.add(d.name);
        else selectedDetectors.delete(d.name);
      });
      onChange();
    },
  }, familyLabel);

  const chips = members.map((d) => {
    const group = groupClass(d.group);
    const chip = el("button", {
      type: "button",
      class: `toggle grp-${group}${d.available ? "" : " untrained"}`,
      "data-detector": d.name,
      "aria-pressed": "false",
      title: d.available ? ""
        : `${d.name} has no trained model for this entity — selecting `
          + `it trains the ${d.family || d.name.split("_")[0]} family first.`,
      onclick: () => {
        if (selectedDetectors.has(d.name)) selectedDetectors.delete(d.name);
        else selectedDetectors.add(d.name);
        onChange();
      },
    }, el("span", { class: "detector", text: d.name }));
    return chip;
  });

  // The chips share one box, so a long row wraps inside the chip column of
  // the pool's grid rather than back under the family button.
  return el("div", { class: "pool-row" },
            familyButton, familyParams(family, members),
            el("div", { class: "pool-chips" }, chips));
}

/* What a run would have to train before it could start.
 *
 * Training is per FAMILY, not per detector (Utils/pipeline_spec.families_for),
 * so ticking one untrained detector trains its whole hyperparameter grid. That
 * is the difference between a two-minute run and a long one, so it is stated
 * before the click rather than discovered in the log. */
function renderTrainingBanner() {
  const banner = $("#training-banner");
  if (!banner) return;
  const families = familiesToTrain();
  banner.hidden = !families.length;
  if (!families.length) return;

  // Every member of a trained family, not just the ones ticked: the trainer
  // fits the whole grid either way.
  const willTrain = [...detectorInfo.entries()]
    .filter(([, info]) => families.includes(info.family))
    .map(([name]) => name);
  // Prose, so the families read the way the chips above them do.
  const familyNames = families.slice();
  const plural = families.length === 1 ? "family" : "families";
  banner.replaceChildren(
    el("strong", { text: "Training first." }),
    el("span", { text: ` This run trains the ${familyNames.join(" and ")} ${plural} `
      + `before it starts the selection, because you picked at least one detector `
      + `that wasn't trained for this entity yet. Training is per family, so all of `
      + `${willTrain.join(", ")} are fitted.` }));
}

/* The families a run would have to fit first: those of every picked detector
 * with nothing on disk. Read by the banner (which spells it out) and by the
 * summary (which only flags it). */
function familiesToTrain() {
  const families = [];
  for (const name of selectedDetectors) {
    const info = detectorInfo.get(name);
    if (info && !info.available && !families.includes(info.family)) families.push(info.family);
  }
  return families;
}

/* One line under each stage's name saying what it contributes, so "Off-by-
 * threshold" is a choice the reader can make without the paper open. Display
 * text only; the token is still the checkbox's value. */
const STAGE_HINTS = {
  ga: "Picks which detectors to stack into an ensemble",
  thompson: "Ranks single detectors window by window",
  gan: "Robustness · realistic drifts made by a GAN",
  offby: "Robustness · points on the decision threshold",
  montecarlo: "Robustness · Gaussian noise on the signal",
};

function renderStages() {
  $("#stages").replaceChildren(...ALL_STAGES.map((token) =>
    el("label", { class: "toggle stage-option" },
      el("input", { type: "checkbox", value: token, checked: true, onchange: onChange }),
      el("span", { class: "stage-option-text" },
        el("span", { class: "stage-option-title", text: STAGE_LABELS[token] }),
        STAGE_HINTS[token]
          ? el("span", { class: "stage-option-hint", text: STAGE_HINTS[token] }) : null))));
}

/* What has to be true before a run can start, as one list.
 *
 * The Run button, the summary's checklist, the line docked beside Run on a
 * phone and the fitness field's own error all read this, so the rule exists
 * once: a check added here gates the button and appears in the checklist
 * together, and the two cannot drift. Each entry says what it is when met and
 * what to do when it is not — the second is the sentence that explains a
 * disabled Run button. A further part of the form (the Online phase) adds its
 * own entry here and nowhere else. */
function readiness() {
  const metrics = selectedMetrics();
  return [
    { key: "entity", ok: entityIsValid(),
      done: "Entity chosen", todo: "Choose an entity" },
    { key: "anomalies", ok: selectedTypes.length > 0,
      done: "At least one anomaly type", todo: "Select at least one anomaly type" },
    { key: "detectors", ok: selectedDetectors.size >= 2,
      done: "At least two detectors", todo: "Select at least two detectors" },
    { key: "stages", ok: selectedStages().length > 0,
      done: "At least one stage", todo: "Select at least one stage" },
    { key: "fitness", ok: metrics.length > 0 && !!fitnessFormula(),
      done: "A weighted fitness metric",
      todo: metrics.length ? "Give at least one metric a weight above zero"
                           : "Choose at least one metric" },
    ...onlineReadiness(),
  ];
}

function readinessOf(key, checks = readiness()) {
  return checks.find((c) => c.key === key);
}

function poolSize() {
  return detectorInfo.size || (catalog && catalog.all_detectors || []).length;
}

const STAGE_SHORT = {
  ga: "GA", thompson: "Thompson", gan: "GAN", offby: "Off-by-threshold",
  montecarlo: "Monte Carlo",
};

/* The summary beside the form. A read-only reflection of state the form
 * already holds — nothing here is read back by currentBody() — so it can only
 * ever describe the run, never change it. */
function renderSummary(checks) {
  const dataset = catalog && catalog.datasets.find((d) => d.name === datasetOf());
  const datasetLabel = dataset ? (dataset.label || dataset.name) : (datasetOf() || "—");
  $("#sum-target").textContent = entityIsValid()
    ? `${datasetLabel} · entity ${entityOf().trim()}` : `${datasetLabel} · no entity yet`;

  const total = poolSize();
  const count = selectedDetectors.size;
  const shown = poolShown();
  $("#sum-detectors").textContent = shown ? `${count} of ${total}` : "—";
  $("#sum-meter").style.width = shown && total ? `${Math.min(100, (count / total) * 100)}%` : "0%";
  // The banner under the pool states the detail; this only flags that it
  // exists — one name, or a count, never the list: with All picked it was
  // twenty-five names over nine lines, and it pushed the Run button out of
  // the sticky summary on a laptop screen.
  const training = familiesToTrain();
  $("#sum-training").hidden = !training.length;
  $("#sum-training").textContent = training.length === 1 ? `${training[0]} trains first`
    : training.length ? `${training.length} families train first` : "";

  const stages = selectedStages();
  const partial = stages.length > 0 && stages.length < ALL_STAGES.length;
  // Filtered: replaceChildren(), unlike el(), would write a null as "null".
  $("#sum-stages").replaceChildren(...[
    ...(stages.length
      ? stages.map((t) => el("span", { class: "sum-tag", text: STAGE_SHORT[t] || t }))
      : [el("span", { class: "dim", text: "none" })]),
    // The partial-run banner under the stages says what is skipped; this only
    // flags that the run is one.
    partial ? el("span", { class: "sum-warn", text: "Partial run: no final decision" }) : null,
  ].filter(Boolean));
  $("#sum-fitness").textContent = fitnessFormula() || "—";
  const copies = selectedTypes.reduce((sum, t) => sum + typeCopies(t), 0);
  $("#sum-anomalies").replaceChildren(...(selectedTypes.length
    ? [...selectedTypes.map((t) => el("span", { class: "sum-tag", text: shortTypeLabel(t) })),
       el("span", { class: "sum-sub", text: `${copies} ${copies === 1 ? "copy" : "copies"} of the series` })]
    : [el("span", { class: "dim", text: "none" })]));
  $("#sum-explain").textContent = $("#explain").checked ? "On — every stage narrated" : "Off";
  $("#sum-online-row").hidden = !onlineOn();
  $("#sum-online").textContent = onlineOn() ? onlineSummary() : "";

  // The glyph comes from data-ok in the stylesheet; the hidden prefix says the
  // same thing to a screen reader, which would not announce a ::before tick.
  $("#sum-checks").replaceChildren(...checks.map((c) =>
    el("li", { "data-ok": c.ok ? "true" : "false" },
      el("span", { class: "visually-hidden", text: c.ok ? "Done: " : "To do: " }),
      c.ok ? c.done : c.todo)));
  const open = checks.filter((c) => !c.ok);
  const state = $("#sum-state");
  state.dataset.ok = open.length ? "false" : "true";
  state.textContent = open.length ? `${open.length} to go` : "Ready";
  const next = $("#sum-next");
  next.dataset.ok = state.dataset.ok;
  // A live region: rewritten only when what it says changes, or every chip
  // click would announce the same sentence again.
  const nextText = open.length ? open[0].todo : `Ready to run ${$("#sum-target").textContent}`;
  if (next.textContent !== nextText) next.textContent = nextText;
}

/* The banner encodes what a partial run actually does, up front, rather than
 * letting the user discover it four minutes later. */
function renderPartialBanner() {
  const stages = selectedStages();
  const banner = $("#partial-banner");
  const partial = stages.length > 0 && stages.length < ALL_STAGES.length;
  banner.hidden = !partial;
  if (!partial) return;
  const children = [
    el("strong", { text: "Partial run." }),
    el("span", { text: " Rank aggregation, the final ensemble-vs-single decision and the " +
      "comprehensive report are skipped." }),
  ];
  if ($("#explain").checked) {
    children.push(el("p", { class: "banner-detail", text:
      "The pipeline only writes explanations on a full run, so they will be generated " +
      "in a second pass once the stages finish." }));
  }
  banner.replaceChildren(...children);
}

function renderResults() {
  const box = $("#results");
  const results = catalog.results || [];
  const explained = new Set(results.map((r) => `${r.dataset}/${r.entity}`));
  // Entities with online results and no explanation (a run without
  // --explain): listed after the explanations, linking to their online page.
  const onlineOnly = (catalog.online_results || [])
    .filter((o) => !explained.has(`${o.dataset}/${o.entity}`));
  if (!results.length && !onlineOnly.length) {
    box.replaceChildren(el("div", { class: "empty-state" },
      el("p", { class: "muted" },
        "No explanations yet — run the pipeline with explanations enabled.")));
    return;
  }
  // An index row per result: title, what was decided, provenance, and the way
  // in. Same fields as before; the classes are what lay them out. A result
  // with online data carries its own Online link beside the row — a second
  // link, not one nested in the first.
  const rows = results.map((r) => {
    const row = el("a", {
      class: "result-row", href: `/result/${r.dataset}/${r.entity}`,
    },
      el("span", {},
        el("span", { class: "result-row-title", text: `${r.dataset_label || r.dataset} · entity ${r.entity}` }),
        el("span", { class: "result-row-meta small muted", text:
          `${r.framework_choice ? r.framework_choice.replace(/_/g, " ") : "no decision"} · ${r.n_stages} stages explained` })),
      el("span", { class: "result-row-stats", text:
        [r.hallucination_rate !== null && r.hallucination_rate !== undefined
          ? `${pct(r.hallucination_rate)} unsupported` : null,
         timeAgo(r.generated_at)].filter(Boolean).join(" · ") }),
      el("span", { class: "result-row-go", "aria-hidden": "true", text: "Read" }));
    if (!r.has_online || !r.online) return row;
    return el("div", { class: "result-entry" }, row,
      el("a", { class: "result-online", href: r.online.page_url || `/online/${r.dataset}/${r.entity}`,
                "aria-label": `Online phase of ${r.dataset_label || r.dataset} · entity ${r.entity}` },
        "Online"));
  });
  const onlineRows = onlineOnly.map((o) => el("a", {
    class: "result-row", href: o.page_url || `/online/${o.dataset}/${o.entity}`,
  },
    el("span", {},
      el("span", { class: "result-row-title", text: `${o.dataset_label || o.dataset} · entity ${o.entity}` }),
      el("span", { class: "result-row-meta small muted", text: [
        "online phase only",
        (o.scenarios || []).length === 1 ? "1 scenario" : `${(o.scenarios || []).length} scenarios`,
        Number.isFinite(o.n_windows) ? `${o.n_windows} windows` : null,
        o.status,
      ].filter(Boolean).join(" · ") })),
    el("span", { class: "result-row-stats", text: timeAgo(o.updated_at || o.generated_at) }),
    el("span", { class: "result-row-go", "aria-hidden": "true", text: "Open" })));
  box.replaceChildren(...rows,
    ...(onlineRows.length
      ? [el("h3", { class: "archive-sub", text: "Online results without an explanation" }), ...onlineRows]
      : []));
}

/* The fix for a model the server does not have, naming what it does have so
 * the reader can pick one without leaving the page. */
function missingModelText(llm) {
  const installed = llm.available_models || [];
  const more = (llm.n_available_models || 0) - installed.length;
  const alternatives = installed.length
    ? `, or set "LLM model" under Advanced to one that is: ${installed.join(", ")}`
      + (more > 0 ? ` and ${more} more` : "") + "."
    : ". No other model is installed on this server.";
  return `${llm.model} is not installed on this LLM server, so the run will finish `
    + `but write no narratives. Install it with \`ollama pull ${llm.model}\``
    + alternatives;
}

/* Answers arrive out of order when the LLM fields are typed into quickly; only
 * the newest request may write the line. */
let healthTicket = 0;
let healthTimer = null;

/* The LLM line under Options.
 *
 * "Reachable" was never the whole answer: Ollama answers /models whether or
 * not the model a run asks for is pulled, and a missing model costs the same
 * as a missing server — the run finishes with no narratives. `model_available`
 * is null when the server could not say; that reads as plain "reachable"
 * rather than as a warning nobody can act on. `checked` is false when "LLM
 * base URL" names another server: the health route never probes one, so
 * neither verdict about the default server would be about this run.
 *
 * Every child passed to replaceChildren is a node. Unlike el(), the DOM does
 * not skip null there — it stringifies it, which is how this line used to end
 * in a literal "null" whenever the server was up. Each badge leads with a
 * glyph, so its verdict survives without the colour. */
async function renderHealth() {
  // Whatever the run would ask for: the Advanced overrides when typed, sent
  // exactly as currentBody() sends them to the run.
  const query = new URLSearchParams();
  const model = $("#llm_model").value.trim();
  const baseUrl = $("#llm_base_url").value.trim();
  if (model) query.set("model", model);
  if (baseUrl) query.set("base_url", baseUrl);
  const ticket = ++healthTicket;
  let health;
  try {
    health = await getJSON(`/api/health${query.toString() ? `?${query}` : ""}`);
  } catch (e) { return; /* health is advisory */ }
  if (ticket !== healthTicket) return;

  const llm = health.llm || {};
  const where = el("span", {}, el("span", { class: "mono", text: llm.model }), " at ",
                   el("span", { class: "mono", text: llm.base_url }));
  let parts;
  if (llm.checked === false) {
    parts = [
      el("span", { class: "badge badge-muted" }, "– LLM server not checked"), where,
      el("span", { class: "muted", text:
        " — a custom server is not probed from this page. Make sure it is running"
        + " and serves this model, or the run will write no narratives." })];
  } else if (!llm.reachable) {
    parts = [
      el("span", { class: "badge badge-warn" }, "! LLM server unreachable"), where,
      el("span", { class: "muted", text:
        " — the run will finish but write no narratives. Start it with `ollama serve`." })];
  } else if (llm.model_available === false) {
    parts = [
      el("span", { class: "badge badge-warn" }, "! LLM model not installed"), where,
      el("p", { class: "banner banner-warn llm-missing", text: missingModelText(llm) })];
  } else {
    parts = [
      el("span", { class: "badge badge-ok" },
         llm.model_available ? "✓ LLM model installed" : "✓ LLM server reachable"),
      where];
  }
  $("#llm-health").replaceChildren(...parts);
}

/* ── Online phase ─────────────────────────────────────────────────────────── */

/* The scenarios, as the form offers them. "Compare all four" replays every
 * scenario on the same windows (`--strategy all`) and is the default: the
 * page this produces is a comparison. Keys are the CLI's
 * (Utils/pipeline_spec.ONLINE_SCENARIOS). */
const ONLINE_SCENARIOS = [
  { key: "all", title: "Compare all four",
    hint: "Replays every scenario below on the same windows, one after another, for a side-by-side comparison." },
  { key: "fixed", title: "Fixed",
    hint: "Never re-optimizes: the offline selection scores every window." },
  { key: "periodic", title: "Periodic",
    hint: "Re-optimizes every M windows." },
  { key: "threshold", title: "Threshold",
    hint: "Re-optimizes when the ensemble's fitness or the single model's score falls more than X% below its baseline." },
  { key: "adaptive_threshold", title: "Adaptive threshold",
    hint: "Re-optimizes below mean − k·std of the last N scores; the X% rule until N scores exist." },
];
const REOPTIMIZING = ["periodic", "threshold", "adaptive_threshold"];

/* Each parameter, the scenarios that read it, and the rule the server checks
 * (Utils/pipeline_spec parse_*), so a value the run would refuse is caught
 * here, with the same bounds. */
const ONLINE_FIELDS = [
  { key: "update_interval", name: "M", label: "Re-optimize every (M windows)", value: "5", int: true, min: "1",
    uses: ["periodic"], note: "A re-optimization starts at windows M, 2M, 3M …",
    check: (v) => (v >= 1 ? "" : "M must be at least 1") },
  { key: "drop_pct", name: "X", label: "Drop below the baseline (X %)", value: "20", int: false, min: "0", max: "100",
    uses: ["threshold", "adaptive_threshold"],
    note: "The baseline is the first window scored with the selection in use.",
    check: (v) => (v > 0 && v < 100 ? "" : "X must be greater than 0 and less than 100") },
  { key: "adaptive_window", name: "N", label: "Rolling window (N scores)", value: "10", int: true, min: "2",
    uses: ["adaptive_threshold"], note: "Scores since the last switch-in.",
    check: (v) => (v >= 2 ? "" : "N must be at least 2") },
  { key: "adaptive_k", name: "k", label: "Standard deviations (k)", value: "2", int: false, min: "0",
    uses: ["adaptive_threshold"], note: "Threshold = mean − k·std of those N scores.",
    check: (v) => (v >= 0 ? "" : "k must not be negative") },
  { key: "reopt_delay", name: "L", label: "Switch-in delay (L windows)", value: "2", int: true, min: "0",
    uses: REOPTIMIZING,
    note: "Triggered at window t, a re-optimization runs in the background and is switched in at t + L; the stream waits for it there. 0 re-scores window t.",
    check: (v) => (v >= 0 ? "" : "L must be 0 or more") },
  { key: "max_online_windows", name: "Windows to process", label: "Windows to process", value: "", int: true, optional: true, min: "1",
    uses: ["fixed", ...REOPTIMIZING], note: "Leave empty for every online window.",
    check: (v) => (v >= 1 ? "" : "Windows to process must be at least 1") },
];

/* What the windows are scored on (--online_labels). Real labels are the
 * default wherever the dataset has them (catalog `default_online_labels`). */
const LABEL_SOURCES = [
  { key: "real", title: "Real labels — no injection online",
    hint: "Windows are scored on the series' own labels; nothing is injected into the online stream, and a re-optimization selects on the labels seen so far." },
  { key: "injected", title: "Injected anomalies",
    hint: "The synthetic anomalies are injected into the online stream and the windows are scored on them, as before." },
];

let wantOnline = false;
let labelChoice = null;     // null: the dataset's default; else what the reader picked
// none | loading | ok | bad (too short) | nolabels (nothing to score on real
// labels) | unknown | bad-iteration
let onlineFeasibility = { state: "none" };
let feasibilityTicket = 0;
let feasibilityTimer = null;
const feasibilityCache = new Map();

function onlineScenario() {
  const picked = $("#online-scenarios input:checked");
  return picked ? picked.value : "all";
}

function onlineOn() {
  return wantOnline && onlineFeasibility.state !== "bad";
}

function currentDataset() {
  return catalog && catalog.datasets.find((d) => d.name === datasetOf());
}

function labelsAvailable() {
  const d = currentDataset();
  return !!(d && d.labels_available);
}

/* The label source the run will use: the reader's pick where the data allows
 * it, else the dataset's default. */
function labelSource() {
  if (labelChoice === "real" && labelsAvailable()) return "real";
  if (labelChoice === "injected") return "injected";
  return labelsAvailable() ? "real" : "injected";
}

function fieldUsed(field, scenario = onlineScenario()) {
  return scenario === "all" || field.uses.includes(scenario);
}

/* A field's value as the server would parse it: a number, null when an
 * optional field is empty, or an error sentence. */
function readField(field) {
  const box = $(`#online-${field.key}`);
  const raw = (box ? box.value : "").trim().replace(/%$/, "");
  const what = field.int ? "a whole number" : "a number";
  if (!raw) return field.optional ? { value: null } : { error: `${field.name} needs ${what}` };
  const value = Number(raw);
  if (!Number.isFinite(value) || (field.int && !Number.isInteger(value))) {
    return { error: `${field.name} must be ${what}` };
  }
  const problem = field.check(value);
  return problem ? { error: problem } : { value };
}

function onlineErrors() {
  return ONLINE_FIELDS.filter((f) => fieldUsed(f))
    .map((f) => readField(f).error).filter(Boolean);
}

/* What the request carries: only what the chosen scenario reads, so the
 * command shows no flag that would change nothing. Off, nothing but the
 * switch itself. A value this form finds wrong is still sent as typed, so the
 * command preview shows the server's own verdict on it rather than a command
 * that silently drops the flag. */
function onlineBody() {
  if (!onlineOn()) return { enable_online: false };
  const scenario = onlineScenario();
  // Always explicit, so the command records which labels were scored on.
  const body = { enable_online: true, online_strategy: scenario,
                 online_labels: labelSource(),
                 inject_online_regime: labelSource() === "injected"
                   && $("#inject_online_regime").checked };
  for (const field of ONLINE_FIELDS) {
    if (!fieldUsed(field, scenario)) continue;
    const read = readField(field);
    if (read.error) body[field.key] = $(`#online-${field.key}`).value.trim();
    else if (read.value !== null && read.value !== undefined) body[field.key] = read.value;
  }
  return body;
}

function onlineReadiness() {
  if (!onlineOn()) return [];
  const errors = onlineErrors();
  if (onlineFeasibility.state === "nolabels") {
    errors.unshift("the online part has no labelled anomaly to score; choose injected anomalies");
  }
  return [{ key: "online", ok: !errors.length, done: "Online settings valid",
            todo: errors.length ? `Fix the online settings: ${errors[0]}` : "" }];
}

function onlineSummary() {
  const scenario = onlineScenario();
  const value = (key) => {
    const field = ONLINE_FIELDS.find((f) => f.key === key);
    const read = readField(field);
    return read.error ? "?" : read.value;
  };
  const parts = [];
  if (scenario === "all") parts.push("All four scenarios");
  else if (scenario === "fixed") parts.push("Fixed");
  else if (scenario === "periodic") parts.push(`Periodic, every ${value("update_interval")} windows`);
  else if (scenario === "threshold") parts.push(`Threshold, X = ${value("drop_pct")}%`);
  else parts.push(`Adaptive, N = ${value("adaptive_window")}, k = ${value("adaptive_k")}`);
  if (scenario !== "fixed") parts.push(`L = ${value("reopt_delay")}`);
  parts.push(labelSource() === "real" ? "real labels" : "injected anomalies");
  const f = onlineFeasibility.state === "ok" ? onlineFeasibility.data : null;
  const regime = labelSource() === "injected" && $("#inject_online_regime").checked;
  const cap = value("max_online_windows");
  const available = f ? (regime ? f.regime_n_windows : f.n_windows) : null;
  if (Number.isFinite(cap) && cap > 0) parts.push(`${available ? Math.min(cap, available) : cap} windows`);
  else if (available) parts.push(`${available} windows`);
  if (regime) parts.push("regime shift");
  return parts.join(" · ");
}

function targetText() {
  const dataset = catalog && catalog.datasets.find((d) => d.name === datasetOf());
  return `${dataset ? (dataset.label || dataset.name) : datasetOf()} · entity ${entityOf().trim()}`;
}

/* The scenario tiles and the parameter fields, built once. */
function buildOnline() {
  $("#online-labels").append(...LABEL_SOURCES.map((s) => el("label", { class: "radio-tile" },
    el("input", { type: "radio", name: "online_labels", value: s.key,
                  checked: s.key === "injected",
                  onchange: () => { labelChoice = s.key; refreshFeasibility(); } }),
    el("span", { class: "radio-tile-text" },
      el("span", { class: "radio-tile-title", text: s.title }),
      el("span", { class: "radio-tile-hint", text: s.hint })))));
  const scenarios = $("#online-scenarios");
  scenarios.append(...ONLINE_SCENARIOS.map((s) => el("label", {
    class: `radio-tile${s.key === "all" ? " is-wide" : ""}`,
  },
    el("input", { type: "radio", name: "online_strategy", value: s.key, checked: s.key === "all",
                  onchange: () => { renderOnline(); onChange(); } }),
    el("span", { class: "radio-tile-text" },
      el("span", { class: "radio-tile-title", text: s.title }),
      el("span", { class: "radio-tile-hint", text: s.hint })))));
  $("#online-params").replaceChildren(...ONLINE_FIELDS.map((f) => el("div", {
    class: "field", "data-field": f.key,
  },
    el("label", { for: `online-${f.key}`, text: f.label }),
    el("input", { id: `online-${f.key}`, type: "number", value: f.value,
                  min: f.min, max: f.max || null, step: f.int ? "1" : "any",
                  inputmode: f.int ? "numeric" : "decimal",
                  placeholder: f.optional ? "All" : null,
                  "aria-describedby": `online-${f.key}-note online-${f.key}-error`,
                  oninput: () => { renderOnline(); onChange(); } }),
    el("p", { class: "field-note", id: `online-${f.key}-note`, text: f.note }),
    el("p", { class: "field-error", id: `online-${f.key}-error`, role: "alert", hidden: true }))));

  $("#enable_online").addEventListener("change", (event) => {
    wantOnline = event.target.checked;
    renderOnline();
  });
  $("#inject_online_regime").addEventListener("change", renderOnline);
  $("#iteration").addEventListener("input", () => {
    clearTimeout(feasibilityTimer);
    feasibilityTimer = setTimeout(refreshFeasibility, 400);
  });
}

/* Draws the part from its state: the switch (disabled, with the server's
 * reason, for an entity too short), the tiles, the fields the chosen
 * scenario reads, their errors, and the regime-shift note. */
/* "N windows of W samples (step S): …", plus, on real labels, how many of
 * them hold a labelled anomaly — the only windows that can be scored. */
function windowsLine(data) {
  const n = data.n_windows;
  let text = `${n} ${n === 1 ? "window" : "windows"} of ${data.window_size} samples `
    + `(step ${data.step_size}): the last 20% of the ${data.total_samples}-sample test series.`;
  if (data.label_source === "real" && data.n_windows_with_anomaly !== null
      && data.n_windows_with_anomaly !== undefined) {
    text += ` ${data.n_windows_with_anomaly} of ${n} windows contain a labelled anomaly`;
    if (data.n_windows_scorable !== data.n_windows_with_anomaly) {
      text += ` (${data.n_windows_scorable} scorable)`;
    }
    text += data.n_windows_scorable > 0 && data.n_windows_scorable < n
      ? "; the others have no score and show as gaps." : ".";
  }
  return text;
}

function renderLabelTiles() {
  const available = labelsAvailable();
  const source = labelSource();
  $$("#online-labels input").forEach((input) => {
    input.checked = input.value === source;
    input.disabled = input.value === "real" && !available;
  });
  const d = currentDataset();
  const name = d ? (d.label || d.name) : "This dataset";
  $("#online-labels-note").textContent = available
    ? (source === "real"
      ? `${name} has labelled test anomalies, so they are the default; the offline phase still selects on injected anomalies.`
      : `${name} has labelled test anomalies; injected anomalies are chosen instead, for comparison.`)
    : `${name} provides no labelled test anomalies, so the windows are scored on injected ones.`;
  // Real-label mode injects nothing online, the regime shift does.
  const regime = $("#inject_online_regime");
  regime.disabled = source === "real";
  if (source === "real") regime.checked = false;
}

function renderOnline() {
  const f = onlineFeasibility;
  const toggle = $("#enable_online");
  const blocked = f.state === "bad";
  toggle.disabled = blocked;
  toggle.checked = wantOnline && !blocked;
  const note = $("#online-feasibility");
  const target = targetText();
  renderLabelTiles();
  if (f.state === "bad") {
    note.replaceChildren(el("p", { class: "banner banner-warn" },
      el("strong", { text: `${target} is too short for the online phase.` }),
      ` ${f.data.reason || ""}`));
  } else if (f.state === "nolabels") {
    note.replaceChildren(el("p", { class: "banner banner-warn" },
      el("strong", { text: `${target}: no labelled anomaly in the online part — nothing to score.` }),
      ` ${windowsLine(f.data)} Choose injected anomalies below to run the online phase on this entity anyway.`));
  } else if (f.state === "ok") {
    note.replaceChildren(el("p", { class: "field-note" }, windowsLine(f.data)));
  } else if (f.state === "loading") {
    note.replaceChildren(el("p", { class: "field-note", text: `Checking that ${target} is long enough…` }));
  } else if (f.state === "unknown") {
    note.replaceChildren(el("p", { class: "field-note", text:
      `The length of ${target} could not be checked here; the run reports it if it is too short.` }));
  } else if (f.state === "bad-iteration") {
    note.replaceChildren(el("p", { class: "field-error", text:
      "Iteration (under Advanced) must be a whole number: it sizes the online windows." }));
  } else {
    note.replaceChildren(el("p", { class: "field-note", text:
      "Choose an entity to check that it is long enough: online windows are at least 100 samples." }));
  }

  $("#online-options").hidden = !onlineOn();
  const scenario = onlineScenario();
  ONLINE_FIELDS.forEach((field) => {
    const box = $(`#online-params [data-field="${field.key}"]`);
    const used = fieldUsed(field, scenario);
    box.hidden = !used;
    const read = readField(field);
    const input = $(`#online-${field.key}`);
    const error = $(`#online-${field.key}-error`);
    const bad = used && !!read.error;
    input.setAttribute("aria-invalid", bad ? "true" : "false");
    error.hidden = !bad;
    error.textContent = bad ? `${read.error}.` : "";
    if (field.key === "max_online_windows" && f.state === "ok") {
      input.placeholder = `All ${$("#inject_online_regime").checked ? f.data.regime_n_windows : f.data.n_windows}`;
    }
  });
  $("#online-regime-note").textContent = labelSource() === "real"
    ? "Needs injected anomalies: with real labels nothing is injected into the online stream."
    : f.state === "ok"
    ? `Runs the online part through scale and wander shifts and replays them as one longer stream, `
      + `so the selection meets data it was not chosen for: ${f.data.regime_n_windows} windows instead of ${f.data.n_windows}.`
    : "Runs the online part through scale and wander shifts and replays them as one longer stream, "
      + "so the selection meets data it was not chosen for.";
  onChange();
}

/* Asks the server whether the entity can run the online phase with this
 * Iteration — the pipeline's own split and window functions — and keeps the
 * answer per entity and iteration, since the form asks on every change. */
async function refreshFeasibility() {
  clearTimeout(feasibilityTimer);
  if (!entityIsValid()) {
    onlineFeasibility = { state: "none" };
    renderOnline();
    return;
  }
  const dataset = datasetOf(), entity = entityOf().trim();
  const iteration = $("#iteration").value.trim() || "5";
  const labels = labelSource();
  const key = `${dataset}|${entity}|${iteration}|${labels}`;
  const ticket = ++feasibilityTicket;
  if (feasibilityCache.has(key)) {
    onlineFeasibility = feasibilityCache.get(key);
    renderOnline();
    return;
  }
  onlineFeasibility = { state: "loading" };
  renderOnline();
  let answer;
  try {
    const data = await getJSON(`/api/online/feasibility/${encodeURIComponent(dataset)}/`
      + `${encodeURIComponent(entity)}?iteration=${encodeURIComponent(iteration)}`
      + `&online_labels=${encodeURIComponent(labels)}`);
    // Not ok with windows to spare: real labels, and none of them can be
    // scored. The reader can still switch to injected anomalies.
    answer = { state: data.ok ? "ok" : (data.n_windows > 0 && data.label_source === "real"
                                         ? "nolabels" : "bad"), data };
    feasibilityCache.set(key, answer);
  } catch (error) {
    answer = { state: error.status === 400 ? "bad-iteration" : "unknown" };
  }
  if (ticket !== feasibilityTicket) return;
  onlineFeasibility = answer;
  renderOnline();
}

/* ── Interaction ──────────────────────────────────────────────────────────── */

let previewTimer = null;

/* Whether the detector pool is on the page. Until an entity names one there is
 * nothing to pick from, so the pool's own count and its error say nothing
 * yet — the summary's checklist already says "Choose an entity" — instead of a
 * red "Select at least two detectors" under a pool that does not exist and a
 * count of the whole catalogue that changes once the real pool arrives. */
function poolShown() {
  return $("#detectors").childElementCount > 0;
}

function onChange() {
  const count = selectedDetectors.size;
  $("#detector-count").textContent = poolShown()
    ? `${count} of ${poolSize()} selected` : "No pool yet";
  const checks = readiness();
  const detectors = readinessOf("detectors", checks);
  $("#detector-error").hidden = detectors.ok || !poolShown();
  if (!detectors.ok) $("#detector-error").textContent = `${detectors.todo}.`;
  syncMetricNotes();
  // One predicate for the button and the checklist that explains it.
  $("#start").disabled = !checks.every((c) => c.ok);

  syncGroupButtons();
  syncDetectorButtons();
  renderTrainingBanner();
  renderPartialBanner();
  renderSummary(checks);
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updatePreview, 150);
}

async function updatePreview() {
  if (!entityIsValid() || !selectedMetrics().length) return;
  try {
    const data = await postJSON("/api/runs", currentBody({ dry_run: true }));
    $("#command").textContent = data.command;
  } catch (error) {
    $("#command").textContent = `# ${error.message}`;
  }
}

async function submit(event) {
  event.preventDefault();
  $("#start-error").textContent = "";
  $("#start").disabled = true;
  try {
    const data = await postJSON("/api/runs", currentBody());
    location.href = data.url;
  } catch (error) {
    $("#start").disabled = false;
    $("#start-error").textContent = error.status === 409
      ? `A run is already in progress. View it at /run/${error.detail.active_job_id}`
      : error.message;
  }
}

async function init() {
  catalog = await getJSON("/api/catalog");
  renderWarnings();
  renderDatasets();
  renderAnomalyTypes();
  renderDecisionMetrics();
  renderStages();
  buildOnline();
  renderResults();
  const health = renderHealth();
  watchDock();
  fitSummary();
  await refreshDetectors();
  // With no entity yet the pool is never built, so nothing above has run the
  // readiness checks: run them once, so the summary and the Run button start
  // from the same verdict instead of from blank rows and an enabled button.
  if (!entityIsValid()) onChange();

  // `input` covers typing and picking from the list; the detector fetch waits
  // for a name that exists, so a half-typed one costs no request.
  $("#entity").addEventListener("input", () => {
    syncEntityValidity();
    refreshFeasibility();
    if (entityIsValid()) refreshDetectors(); else onChange();
  });
  $("#dataset").addEventListener("change", refreshFeasibility);
  renderOnline();
  $("#run-form").addEventListener("submit", submit);
  $$("#run-form input, #run-form select").forEach((node) =>
    node.addEventListener("change", onChange));
  // Re-checked as either override is typed, so picking an installed model
  // from the warning's list turns the line green before the run is started,
  // and naming another server drops a verdict about the default one.
  ["#llm_model", "#llm_base_url"].forEach((field) =>
    $(field).addEventListener("input", () => {
      clearTimeout(healthTimer);
      healthTimer = setTimeout(renderHealth, 400);
    }));

  $$("[data-select]").forEach((button) => button.addEventListener("click", () => {
    const all = button.dataset.select === "all";
    $$("#detectors button[data-detector]").forEach((chip) => {
      const name = chip.dataset.detector;
      if (all) selectedDetectors.add(name); else selectedDetectors.delete(name);
    });
    onChange();
  }));

  $$("[data-stages]").forEach((button) => button.addEventListener("click", () => {
    const wanted = button.dataset.stages === "robustness"
      ? ["gan", "offby", "montecarlo"] : ALL_STAGES;
    $$("#stages input").forEach((input) => { input.checked = wanted.includes(input.value); });
    onChange();
  }));

  $("#copy-command").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#command").textContent);
      $("#copy-command").textContent = "Copied";
      setTimeout(() => { $("#copy-command").textContent = "Copy"; }, 1200);
    } catch (e) { /* clipboard may be blocked; the text is selectable anyway */ }
  });

  landOnHash(health);
}

/* The reading pages' "Results" breadcrumb links to #results-title. The
 * browser resolved that fragment before the catalogue, the stages and the
 * health line had been drawn, and they then pushed the list down: the reader
 * arrived with "Previous results" at the foot of the screen. So the jump is
 * made again once the page is drawn, and once more when the health line lands
 * — unless the reader has moved the page in between. */
function landOnHash(health) {
  const id = decodeURIComponent((location.hash || "").replace(/^#/, ""));
  const target = id && document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ block: "start" });
  const landed = window.scrollY;
  Promise.resolve(health).finally(() => {
    if (window.scrollY === landed) target.scrollIntoView({ block: "start" });
  });
}

/* The docked Run row's height, for the stylesheet, which reserves that much
 * room under the page and keeps keyboard focus clear of it
 * (scroll-padding-bottom). Measured rather than assumed: the row grows when
 * #start-error shows or the line beside the button wraps. Zero when the row is
 * not docked (the wide layout, where it sits in the summary). */
function watchDock() {
  const dock = $(".summary-actions");
  if (!dock || !("ResizeObserver" in window)) return;
  new ResizeObserver(() => {
    const docked = getComputedStyle(dock).position === "fixed";
    document.documentElement.style.setProperty("--dock-h",
      docked ? `${Math.ceil(dock.getBoundingClientRect().height)}px` : "0px");
  }).observe(dock);
}

/* On a wide screen the summary sticks under the header and scrolls inside
 * itself, with its Run row pinned to the foot of that box. Until it has risen
 * to the header, a box sized to the whole window ran on below it, so Run was
 * out of sight at the top of the page on every laptop. The box is held to the
 * room it has on screen right now (--summary-room, read by the stylesheet),
 * re-measured as the page scrolls or anything above it changes height.
 * Presentation only; nothing here is read back by the form. */
function fitSummary() {
  const aside = $("#run-summary");
  if (!aside) return;
  let frame = 0;
  const fit = () => {
    frame = 0;
    const sticky = getComputedStyle(aside).position === "sticky";
    const top = Math.max(aside.getBoundingClientRect().top, 0);
    const room = Math.floor(window.innerHeight - top - 16);
    if (sticky && room > 0) aside.style.setProperty("--summary-room", `${room}px`);
    else aside.style.removeProperty("--summary-room");
  };
  const later = () => { if (!frame) frame = requestAnimationFrame(fit); };
  window.addEventListener("scroll", later, { passive: true });
  window.addEventListener("resize", later);
  if ("ResizeObserver" in window) new ResizeObserver(later).observe(document.body);
  fit();
}

init();
