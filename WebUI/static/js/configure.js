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
    stages: selectedStages(),
    detectors: Array.from(selectedDetectors),
    explain: $("#explain").checked,
    // `parallel` and `enable_online` are no longer offered: runs go sequential
    // and offline. build_argv still honours both, so a caller that sets them
    // programmatically keeps working.
    overwrite: $("#overwrite").checked,
    iteration: Number($("#iteration").value) || 5,
    timeout: Number($("#timeout").value) || 7200,
    llm_model: $("#llm_model").value.trim() || null,
    llm_base_url: $("#llm_base_url").value.trim() || null,
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
  select.replaceChildren(...catalog.datasets.map((d) =>
    el("option", { value: d.name, disabled: !d.runnable },
      `${d.label || d.name} — ${d.n_entities} entities` +
      (d.runnable ? "" : " (not supported)"))));
  select.addEventListener("change", () => { renderEntities(); refreshDetectors(); });
  renderEntities();
}

function renderEntities() {
  const dataset = catalog.datasets.find((d) => d.name === datasetOf());
  const filter = ($("#entity-filter").value || "").toLowerCase();
  const trained = new Set((dataset && dataset.trained) || []);
  const entities = (dataset ? dataset.entities : [])
    .filter((e) => !filter || e.toLowerCase().includes(filter))
    // Entities with checkpoints first: those run immediately, the rest have to
    // train from scratch. Junk files in the data directory sink to the bottom.
    .sort((a, b) => (trained.has(b) - trained.has(a)) || a.localeCompare(b, undefined, { numeric: true }));

  const select = $("#entity");
  const previous = select.value;
  select.replaceChildren(...entities.map((e) => el("option", { value: e },
    trained.has(e) ? e : `${e} — needs training`)));
  if (entities.includes(previous)) select.value = previous;
  // size must stay 1: any larger turns the <select> into an always-open
  // listbox that never collapses after a choice.
  select.size = 1;
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
 * again. They carry aria-pressed rather than a checkbox because they act on a
 * set of chips that each have their own checkbox — a fourth checkbox governing
 * the other twelve reads as a member of the same list rather than a control
 * over it. */
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
        $$("#detectors input").forEach((input) => {
          if (input.disabled || !names.includes(input.value)) return;
          input.checked = turnOn;
          if (turnOn) selectedDetectors.add(input.value);
          else selectedDetectors.delete(input.value);
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
    button.setAttribute("aria-pressed", all ? "true" : "false");
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
    (d) => [d.name, { available: !!d.available, family: d.family || d.name.split("_")[0] }]));

  renderGroupButtons(detectors);

  const byFamily = {};
  detectors.forEach((d) => {
    const family = d.family || d.name.split("_")[0];
    (byFamily[family] = byFamily[family] || []).push(d);
  });

  box.replaceChildren(...Object.entries(byFamily).map(([family, members]) => {
    const inputs = [];

    // Clicking the family name toggles the whole family: if any member is
    // unchecked it selects them all, otherwise it clears them.
    const familyButton = el("button", {
      type: "button", class: `small grp-${groupClass(members[0].group)}`,
      style: "width: 7.5em; text-align: left; font-family: var(--font-detector);",
      // Always the full name here, since the label may be abbreviated.
      title: `Select or clear every ${family} detector`,
      onclick: () => {
        const turnOn = members.some((d) => !selectedDetectors.has(d.name));
        inputs.forEach((input) => {
          input.checked = turnOn;
          if (turnOn) selectedDetectors.add(input.value);
          else selectedDetectors.delete(input.value);
        });
        onChange();
      },
    }, family);

    const chips = members.map((d) => {
      const input = el("input", {
        type: "checkbox", value: d.name, checked: false,
        onchange: () => {
          if (input.checked) selectedDetectors.add(d.name);
          else selectedDetectors.delete(d.name);
          onChange();
        },
      });
      inputs.push(input);
      const group = groupClass(d.group);
      return el("label", {
        class: `toggle grp-${group}${d.available ? "" : " untrained"}`,
        title: d.available ? ""
          : `${d.name} has no trained model for this entity — selecting it trains `
            + `the ${d.family || d.name.split("_")[0]} family first.`,
      }, input, el("span", { class: "detector", text: d.name }));
    });

    return el("div", { class: "row" },
              familyButton, familyParams(family, members), ...chips);
  }));
  onChange();
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
  const families = [];
  for (const name of selectedDetectors) {
    const info = detectorInfo.get(name);
    if (info && !info.available && !families.includes(info.family)) families.push(info.family);
  }
  banner.hidden = !families.length;
  if (!families.length) return;

  // Every member of a trained family, not just the ones ticked: the trainer
  // fits the whole grid either way.
  const willTrain = [...detectorInfo.entries()]
    .filter(([, info]) => families.includes(info.family))
    .map(([name]) => name);
  const plural = families.length === 1 ? "family" : "families";
  banner.replaceChildren(
    el("strong", { text: "Training first." }),
    el("span", { text: ` This run trains the ${families.join(" and ")} ${plural} `
      + `before it starts the selection, because you picked at least one detector `
      + `that wasn't trained for this entity yet. Training is per family, so all of `
      + `${willTrain.join(", ")} are fitted.` }));
}

function renderStages() {
  $("#stages").replaceChildren(...ALL_STAGES.map((token) =>
    el("label", { class: "toggle" },
      el("input", { type: "checkbox", value: token, checked: true, onchange: onChange }),
      el("span", { text: STAGE_LABELS[token] }))));
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
    children.push(el("p", { class: "small", style: "margin: var(--sp-2) 0 0;", text:
      "The pipeline only writes explanations on a full run, so they will be generated " +
      "in a second pass once the stages finish." }));
  }
  banner.replaceChildren(...children);
}

function renderResults() {
  const box = $("#results");
  const results = catalog.results || [];
  if (!results.length) {
    box.replaceChildren(el("p", { class: "muted" },
      "No explanations yet — run the pipeline with explanations enabled."));
    return;
  }
  box.replaceChildren(...results.map((r) => el("a", {
    class: "card card-tight row-between", href: `/result/${r.dataset}/${r.entity}`,
    style: "text-decoration: none; color: inherit;",
  },
    el("div", {},
      el("strong", { text: `${r.dataset} · ${r.entity}` }),
      el("div", { class: "small muted", text:
        `${r.framework_choice ? r.framework_choice.replace(/_/g, " ") : "no decision"} · ${r.n_stages} stages explained` })),
    el("div", { class: "small muted mono", text:
      [r.hallucination_rate !== null && r.hallucination_rate !== undefined
        ? `${pct(r.hallucination_rate)} unsupported` : null,
       timeAgo(r.generated_at)].filter(Boolean).join(" · ") }))));
}

async function renderHealth() {
  try {
    const health = await getJSON("/api/health");
    const llm = health.llm || {};
    $("#llm-health").replaceChildren(
      el("span", { class: llm.reachable ? "badge badge-ok" : "badge badge-warn" },
         llm.reachable ? "✓ LLM server reachable" : "! LLM server unreachable"),
      el("span", { text: ` ${llm.model} at ${llm.base_url}` }),
      llm.reachable ? null : el("span", { class: "muted", text:
        " — the run will finish but write no narratives. Start it with `ollama serve`." }));
  } catch (e) { /* health is advisory */ }
}

/* ── Interaction ──────────────────────────────────────────────────────────── */

let previewTimer = null;

function onChange() {
  const count = selectedDetectors.size;
  $("#detector-count").textContent =
    `${count} of ${detectorInfo.size || (catalog && catalog.all_detectors || []).length} selected`;
  const tooFew = count < 2;
  $("#detector-error").hidden = !tooFew;
  if (tooFew) $("#detector-error").textContent = "Select at least two detectors.";
  $("#start").disabled = tooFew || !selectedStages().length || !entityOf();

  syncGroupButtons();
  renderTrainingBanner();
  renderPartialBanner();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updatePreview, 150);
}

async function updatePreview() {
  if (!entityOf()) return;
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
  renderStages();
  renderResults();
  renderHealth();
  await refreshDetectors();

  $("#entity").addEventListener("change", refreshDetectors);
  $("#entity-filter").addEventListener("input", renderEntities);
  $("#run-form").addEventListener("submit", submit);
  $$("#run-form input, #run-form select").forEach((node) =>
    node.addEventListener("change", onChange));

  $$("[data-select]").forEach((button) => button.addEventListener("click", () => {
    const all = button.dataset.select === "all";
    $$("#detectors input").forEach((input) => {
      if (input.disabled) return;
      input.checked = all;
      if (all) selectedDetectors.add(input.value); else selectedDetectors.delete(input.value);
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
}

init();
